"""
MultiLoRAVLLMRollout — multi-adapter vLLM serving for diverse exploration.

Extends VLLMRollout to serve multiple LoRA adapters simultaneously, enabling:
  - Diverse exploration strategies in a single batch (creative, analytical, careful, aggressive)
  - 12x throughput via Punica SGMV kernels (batched LoRA across adapters)
  - Per-request LoRA assignment (different adapters for different prompts)
  - Memory overhead: ~200MB per rank-16 adapter

Architecture:
  - Base model loaded once in vLLM (stays resident)
  - Multiple LoRA adapters loaded simultaneously (up to max_loras)
  - Each generation request can specify which adapter to use
  - Punica kernels batch LoRA operations across different adapters

Usage:
    rollout = MultiLoRAVLLMRollout(
        model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        max_loras=8,
        lora_adapters={
            "creative": "path/to/lora1",
            "analytical": "path/to/lora2",
        },
    )
    results = rollout.generate_batch_with_loras(
        prompts, lora_assignments=["creative", "analytical", ...], n=8
    )
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

# Must be set before importing vLLM
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import sys as _sys
_venv_cuda = os.path.join(_sys.prefix, "lib", "python3.12",
                          "site-packages", "nvidia", "cu13")
if os.path.isdir(_venv_cuda):
    os.environ.setdefault("CUDA_HOME", _venv_cuda)
    os.environ["PATH"] = _venv_cuda + "/bin:" + os.environ.get("PATH", "")
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_venv_cuda}/lib:{_ld}" if _ld else f"{_venv_cuda}/lib"

import numpy as np

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

from iloptimus.core.rl_factory.training.vllm_rollout import VLLMRollout, RolloutResult


class MultiLoRAVLLMRollout(VLLMRollout):
    """
    Extended VLLMRollout supporting multiple LoRA adapters for diverse exploration.

    Key advantages over single-LoRA VLLMRollout:
      1. Serve 8-16 adapters simultaneously (12x throughput via Punica SGMV)
      2. Per-request LoRA assignment (different strategies per prompt)
      3. Diverse exploration in a single batched call
      4. No adapter reload between requests (all loaded in memory)

    Usage:
        rollout = MultiLoRAVLLMRollout(
            model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            max_loras=8,
            lora_adapters={"creative": path1, "analytical": path2},
        )
        results = rollout.generate_batch_with_loras(prompts, ["creative", "analytical"])
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device: str = "cuda",
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 2048,
        max_loras: int = 8,
        max_lora_rank: int = 16,
        lora_adapters: Optional[dict[str, str]] = None,
        dtype: str = "bfloat16",
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        kv_cache_dtype: str = "auto",
    ):
        """
        Args:
            model_name: Base model name.
            max_loras: Maximum number of LoRA adapters to serve simultaneously.
            lora_adapters: Dict mapping adapter names to paths. Loaded at init.
            enable_prefix_caching: Enable automatic prefix caching for shared prompts.
            enable_chunked_prefill: Enable chunked prefill for better batching.
            max_num_seqs: Maximum concurrent sequences.
            max_num_batched_tokens: Maximum tokens per batch.
            kv_cache_dtype: KV cache dtype ("auto", "fp8").
        """
        if not HAS_VLLM:
            raise ImportError("vllm is required. Install with: pip install vllm")

        self.model_name = model_name
        self.enable_lora = True
        self.lora_adapter_path = None
        self._lora_id_counter = 0

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build vLLM engine with multi-LoRA + optimizations
        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            trust_remote_code=True,
            # Continuous batching optimizations
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        # FP8 KV cache for 2x memory reduction
        if kv_cache_dtype != "auto":
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
            if kv_cache_dtype == "fp8":
                engine_kwargs["calculate_kv_scales"] = True

        self.llm = LLM(**engine_kwargs)

        # Load multiple LoRA adapters
        self.lora_requests: dict[str, LoRARequest] = {}
        self.current_lora_request = None  # For backward compat

        if lora_adapters:
            for name, path in lora_adapters.items():
                self._load_lora_adapter(name, path)

    def _load_lora_adapter(self, name: str, path: str) -> None:
        """Load a single LoRA adapter and register it."""
        self._lora_id_counter += 1
        lora_id = self._lora_id_counter
        self.lora_requests[name] = LoRARequest(
            lora_name=name,
            lora_int_id=lora_id,
            lora_path=path,
        )

    def add_lora_adapter(self, name: str, path: str) -> None:
        """Add a new LoRA adapter at runtime."""
        if name in self.lora_requests:
            raise ValueError(f"LoRA adapter '{name}' already loaded")
        self._load_lora_adapter(name, path)

    def reload_lora(self, adapter_path: str, name: str = "default") -> None:
        """Reload a specific LoRA adapter after training step."""
        if name in self.lora_requests:
            # Update the path for existing adapter
            self._lora_id_counter += 1
            self.lora_requests[name] = LoRARequest(
                lora_name=name,
                lora_int_id=self._lora_id_counter,
                lora_path=adapter_path,
            )
        else:
            self._load_lora_adapter(name, adapter_path)

        # Set as current for backward compat
        self.current_lora_request = self.lora_requests.get(name)

    def generate_batch_with_loras(
        self,
        prompts: list[str],
        lora_assignments: list[str],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> RolloutResult:
        """
        Generate completions with per-prompt LoRA assignment.

        Each prompt is generated using its assigned LoRA adapter, all in a single
        batched vLLM call. This enables diverse exploration strategies simultaneously.

        Args:
            prompts: List of prompt strings.
            lora_assignments: List of LoRA adapter names (same length as prompts).
            n: Number of completions per prompt.
            max_tokens: Max new tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            seed: Random seed.
            system_prompt: Optional system prompt (shared across all, benefits from prefix caching).

        Returns:
            RolloutResult with all completions.
        """
        if len(prompts) != len(lora_assignments):
            raise ValueError(
                f"prompts ({len(prompts)}) and lora_assignments ({len(lora_assignments)}) must have same length"
            )

        # Format prompts with chat template
        formatted = [self.format_prompt(p, system_prompt) for p in prompts]

        # Expand: each prompt repeated n times
        expanded_prompts = formatted * n
        expanded_loras = lora_assignments * n
        prompt_indices = [i for i in range(len(prompts)) for _ in range(n)]

        # Build per-request LoRA request list
        lora_request_list = []
        for lora_name in expanded_loras:
            req = self.lora_requests.get(lora_name)
            if req is None:
                raise ValueError(f"LoRA adapter '{lora_name}' not loaded. Available: {list(self.lora_requests.keys())}")
            lora_request_list.append(req)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

        start_time = time.time()

        # Generate with per-request LoRA
        # vLLM supports passing lora_request as a list for per-request assignment
        outputs = self.llm.generate(
            expanded_prompts,
            sampling_params,
            lora_request=lora_request_list,
        )
        gen_time = time.time() - start_time

        # Extract results
        responses = []
        full_texts = []
        response_token_ids = []
        prompt_token_ids = []

        for output in outputs:
            prompt_ids = list(output.prompt_token_ids)
            completion = output.outputs[0]
            response_text = completion.text
            response_ids = list(completion.token_ids)

            prompt_token_ids.append(prompt_ids)
            response_token_ids.append(response_ids)
            responses.append(response_text)
            full_texts.append(self.tokenizer.decode(prompt_ids) + response_text)

        return RolloutResult(
            prompts=expanded_prompts,
            responses=responses,
            full_texts=full_texts,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            logprobs=[None] * len(responses),
            generation_time_s=gen_time,
            num_samples=len(responses),
        )

    def generate_diverse_batch(
        self,
        prompts: list[str],
        n: int = 8,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> RolloutResult:
        """
        Generate completions using ALL loaded LoRA adapters in round-robin fashion.

        Each prompt gets completions from different adapters, maximizing diversity.
        Useful when you want diverse exploration without manually assigning adapters.

        Args:
            prompts: List of prompts.
            n: Completions per prompt (distributed across adapters).
            max_tokens: Max tokens per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            RolloutResult with diverse completions.
        """
        adapter_names = list(self.lora_requests.keys())
        if not adapter_names:
            # Fall back to no LoRA
            return self.generate_batch(
                prompts, n=n, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                seed=seed, system_prompt=system_prompt,
            )

        # Round-robin assignment
        lora_assignments = []
        for i in range(len(prompts)):
            for j in range(n):
                adapter_idx = (i * n + j) % len(adapter_names)
                lora_assignments.append(adapter_names[adapter_idx])

        # Expand prompts
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * n)

        # Use the per-LoRA generation
        formatted = [self.format_prompt(p, system_prompt) for p in expanded_prompts]

        lora_request_list = [
            self.lora_requests[name] for name in lora_assignments
        ]

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

        start_time = time.time()
        outputs = self.llm.generate(
            formatted,
            sampling_params,
            lora_request=lora_request_list,
        )
        gen_time = time.time() - start_time

        responses = []
        full_texts = []
        response_token_ids = []
        prompt_token_ids = []

        for output in outputs:
            prompt_ids = list(output.prompt_token_ids)
            completion = output.outputs[0]
            responses.append(completion.text)
            response_token_ids.append(list(completion.token_ids))
            prompt_token_ids.append(prompt_ids)
            full_texts.append(self.tokenizer.decode(prompt_ids) + completion.text)

        return RolloutResult(
            prompts=formatted,
            responses=responses,
            full_texts=full_texts,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            logprobs=[None] * len(responses),
            generation_time_s=gen_time,
            num_samples=len(responses),
        )

    @property
    def loaded_adapters(self) -> list[str]:
        """Return names of all loaded LoRA adapters."""
        return list(self.lora_requests.keys())
