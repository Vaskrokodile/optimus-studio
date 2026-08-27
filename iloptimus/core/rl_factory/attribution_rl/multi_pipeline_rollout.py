"""Multi-pipeline vLLM rollout: batch all pipelines' generation into one call.

Each RL pipeline has its own LoRA adapter (targeting 5% of params).
vLLM's multi-LoRA serving batches all requests together — 20 pipelines
at the same speed as 1.

Architecture:
  - Single vLLM instance, base model resident in GPU memory.
  - One LoRA adapter per pipeline (up to ``max_loras`` concurrent adapters).
  - ``generate_batch`` takes a flat list of ``PipelineRequest`` objects coming
    from *all* pipelines, expands each prompt by ``n_completions``, builds a
    per-request ``LoRARequest`` list, and issues a SINGLE ``vllm.generate()``
    call. Results are then grouped back by ``pipeline_id``.
  - In mock mode (no vLLM installed) a user-supplied mock generation function
    is used so the whole thing is testable on CPU / CI without torch or vllm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    LoRARequest = None  # type: ignore

try:
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False
    AutoTokenizer = None  # type: ignore

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineRequest:
    """A single generation request belonging to one RL pipeline.

    Attributes:
        pipeline_id: Which pipeline this request belongs to.
        prompt: The prompt text.
        n_completions: How many completions to generate for this prompt.
        lora_adapter_name: Which LoRA adapter to use (usually == pipeline_id).
        sampling_params: Generation kwargs (temperature, max_tokens, top_p, ...).
    """
    pipeline_id: str
    prompt: str
    n_completions: int = 1
    lora_adapter_name: str = ""
    sampling_params: dict = field(default_factory=dict)


@dataclass
class PipelineBatchResult:
    """Generation results for a single pipeline, grouped from the batched call.

    Attributes:
        pipeline_id: Which pipeline these results belong to.
        prompts: The (expanded) prompts, one per generated completion.
        responses: The generated completions (one per expanded prompt).
        full_texts: prompt + response concatenated.
        generation_time: Wall-clock seconds for the batched generation.
        n_samples: Total number of completions for this pipeline.
    """
    pipeline_id: str
    prompts: list[str]
    responses: list[str]
    full_texts: list[str]
    generation_time: float
    n_samples: int


@dataclass
class MultiPipelineRolloutConfig:
    """Configuration for :class:`MultiPipelineVLLMRollout`."""
    model_name: str = ""
    max_loras: int = 20  # support 20 concurrent pipelines
    max_lora_rank: int = 16
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    kv_cache_dtype: str = "auto"
    dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


class MultiPipelineVLLMRollout:
    """Single vLLM instance serving all 20 RL pipelines' generation requests.

    All pipelines' requests are batched into ONE ``vllm.generate()`` call,
    with per-request LoRA assignment. This is the key throughput win: 20
    pipelines run at (roughly) the speed of 1.

    Works in two modes:
      * **vLLM mode** (``HAS_VLLM`` is True): real vLLM engine with multi-LoRA.
      * **Mock mode** (no vLLM): uses a user-supplied mock generation function
        so the entire multi-pipeline orchestration can be unit-tested on CPU.
    """

    def __init__(
        self,
        config: MultiPipelineRolloutConfig,
        lora_adapters: Optional[dict[str, str]] = None,
    ):
        self.config = config
        self.mock_mode = not HAS_VLLM

        # adapter_name -> adapter_path
        self.lora_adapter_paths: dict[str, str] = {}
        # adapter_name -> LoRARequest (vLLM mode only)
        self.lora_requests: dict[str, Any] = {}
        # pipeline_id -> adapter_name (default: same as pipeline_id)
        self.pipeline_adapters: dict[str, str] = {}
        self._lora_id_counter = 0

        # Mock generation function (set via set_mock_generate_fn)
        self._mock_generate_fn: Optional[Callable[[str, int], list[str]]] = None

        # Stats
        self.total_requests = 0
        self.total_tokens = 0
        self.per_pipeline_stats: dict[str, dict[str, int]] = {}
        self._n_batch_calls = 0

        if not self.mock_mode:
            # ---- Real vLLM initialization ----
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            engine_kwargs = dict(
                model=config.model_name,
                dtype=config.dtype,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                enable_lora=True,
                max_loras=config.max_loras,
                max_lora_rank=config.max_lora_rank,
                trust_remote_code=True,
                max_num_seqs=config.max_num_seqs,
                max_num_batched_tokens=config.max_num_batched_tokens,
                enable_prefix_caching=config.enable_prefix_caching,
                enable_chunked_prefill=config.enable_chunked_prefill,
            )
            if config.kv_cache_dtype != "auto":
                engine_kwargs["kv_cache_dtype"] = config.kv_cache_dtype
                if config.kv_cache_dtype == "fp8":
                    engine_kwargs["calculate_kv_scales"] = True

            self.llm = LLM(**engine_kwargs)
        else:
            # ---- Mock mode ----
            self.llm = None
            self.tokenizer = None

        # Load any adapters provided at init time
        if lora_adapters:
            for name, path in lora_adapters.items():
                self._register_adapter(name, path)

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def _register_adapter(self, name: str, path: str) -> Any:
        """Register a LoRA adapter (or just record its path in mock mode)."""
        self.lora_adapter_paths[name] = path
        if not self.mock_mode:
            self._lora_id_counter += 1
            req = LoRARequest(
                lora_name=name,
                lora_int_id=self._lora_id_counter,
                lora_path=path,
            )
            self.lora_requests[name] = req
            return req
        return None

    def add_pipeline_adapter(self, pipeline_id: str, adapter_path: str) -> None:
        """Add a new LoRA adapter for a pipeline.

        The adapter name defaults to the pipeline_id.
        """
        adapter_name = pipeline_id
        self._register_adapter(adapter_name, adapter_path)
        self.pipeline_adapters[pipeline_id] = adapter_name

    def reload_pipeline_adapter(self, pipeline_id: str, adapter_path: str) -> None:
        """Reload a pipeline's LoRA adapter after a training step.

        In vLLM mode this creates a NEW LoRARequest with an incremented ID
        (vLLM needs a fresh int id to pick up the new weights on disk).
        In mock mode we just update the recorded path.
        """
        adapter_name = self.pipeline_adapters.get(pipeline_id, pipeline_id)
        self.lora_adapter_paths[adapter_name] = adapter_path
        if not self.mock_mode:
            self._lora_id_counter += 1
            self.lora_requests[adapter_name] = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=self._lora_id_counter,
                lora_path=adapter_path,
            )

    # ------------------------------------------------------------------
    # Mock support
    # ------------------------------------------------------------------

    def set_mock_generate_fn(
        self, fn: Callable[[str, int], list[str]]
    ) -> None:
        """Set a mock generation function for testing without vLLM.

        ``fn(prompt: str, n: int) -> list[str]`` of length ``n``.
        """
        self._mock_generate_fn = fn

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _ensure_pipeline_stats(self, pipeline_id: str) -> None:
        if pipeline_id not in self.per_pipeline_stats:
            self.per_pipeline_stats[pipeline_id] = {
                "requests": 0,
                "tokens": 0,
            }

    def _record_stats(self, pipeline_id: str, n_samples: int, n_tokens: int) -> None:
        self._ensure_pipeline_stats(pipeline_id)
        self.total_requests += 1
        self.total_tokens += n_tokens
        self.per_pipeline_stats[pipeline_id]["requests"] += 1
        self.per_pipeline_stats[pipeline_id]["tokens"] += n_tokens

    def get_stats(self) -> dict:
        """Return rollout statistics."""
        avg_batch = (
            self.total_requests / self._n_batch_calls
            if self._n_batch_calls > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "per_pipeline": {
                pid: dict(stats)
                for pid, stats in self.per_pipeline_stats.items()
            },
            "avg_batch_size": float(avg_batch),
        }

    # ------------------------------------------------------------------
    # The key method: batch ALL pipelines' requests into one call
    # ------------------------------------------------------------------

    def generate_batch(
        self, requests: list[PipelineRequest]
    ) -> list[PipelineBatchResult]:
        """Batch ALL pipelines' requests into a single vLLM generate() call.

        Steps:
          1. Expand each prompt by ``n_completions``.
          2. Build a per-expanded-request LoRARequest list.
          3. Issue ONE ``vllm.generate()`` call (or mock fn) for everything.
          4. Group the flat output back by ``pipeline_id``.

        Returns one :class:`PipelineBatchResult` per distinct pipeline_id
        present in ``requests`` (in first-seen order).
        """
        if not requests:
            return []

        # Default sampling params merged from the request-level dict.
        def _merge_sp(req: PipelineRequest) -> dict:
            defaults = dict(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1024,
                seed=None,
            )
            defaults.update(req.sampling_params or {})
            return defaults

        # --- Expand prompts and track which pipeline each belongs to ---
        expanded_prompts: list[str] = []
        expanded_pipeline_ids: list[str] = []
        expanded_adapter_names: list[str] = []
        expanded_sp: list[dict] = []
        # preserve first-seen order of pipelines for the result list
        pipeline_order: list[str] = []
        seen: set[str] = set()

        for req in requests:
            pid = req.pipeline_id
            if pid not in seen:
                seen.add(pid)
                pipeline_order.append(pid)
            adapter_name = req.lora_adapter_name or self.pipeline_adapters.get(
                pid, pid
            )
            sp = _merge_sp(req)
            for _ in range(req.n_completions):
                expanded_prompts.append(req.prompt)
                expanded_pipeline_ids.append(pid)
                expanded_adapter_names.append(adapter_name)
                expanded_sp.append(sp)

        start_time = time.time()

        if not self.mock_mode:
            # ---- Real vLLM path ----
            # Use the sampling params from the first expanded request as the
            # SamplingParams template (vLLM takes a single SamplingParams per
            # generate() call; per-request variation is handled by expansion).
            sp0 = expanded_sp[0]
            sampling_params = SamplingParams(
                n=1,
                temperature=sp0.get("temperature", 0.8),
                top_p=sp0.get("top_p", 0.95),
                max_tokens=sp0.get("max_tokens", 1024),
                seed=sp0.get("seed"),
            )

            lora_request_list = []
            for adapter_name in expanded_adapter_names:
                req = self.lora_requests.get(adapter_name)
                if req is None:
                    raise ValueError(
                        f"LoRA adapter '{adapter_name}' not loaded. "
                        f"Available: {list(self.lora_requests.keys())}"
                    )
                lora_request_list.append(req)

            outputs = self.llm.generate(
                expanded_prompts,
                sampling_params,
                lora_request=lora_request_list,
            )

            flat_responses: list[str] = []
            for output in outputs:
                flat_responses.append(output.outputs[0].text)
        else:
            # ---- Mock path ----
            if self._mock_generate_fn is None:
                raise RuntimeError(
                    "Mock mode active but no mock generate function set. "
                    "Call set_mock_generate_fn() first."
                )
            flat_responses = []
            for prompt, sp in zip(expanded_prompts, expanded_sp):
                # mock fn returns n completions; we only need 1 per expanded
                # slot, but call with n=1 to keep the signature uniform.
                completions = self._mock_generate_fn(prompt, 1)
                if not completions:
                    completions = [""]
                flat_responses.append(completions[0])

        gen_time = time.time() - start_time

        # --- Group results back by pipeline_id ---
        grouped: dict[str, dict] = {
            pid: {"prompts": [], "responses": []}
            for pid in pipeline_order
        }
        for prompt, response, pid in zip(
            expanded_prompts, flat_responses, expanded_pipeline_ids
        ):
            grouped[pid]["prompts"].append(prompt)
            grouped[pid]["responses"].append(response)

        results: list[PipelineBatchResult] = []
        for pid in pipeline_order:
            prompts = grouped[pid]["prompts"]
            responses = grouped[pid]["responses"]
            full_texts = [p + r for p, r in zip(prompts, responses)]
            n_samples = len(responses)
            # crude token estimate: whitespace tokens
            n_tokens = sum(len(r.split()) for r in responses)
            self._record_stats(pid, n_samples, n_tokens)
            results.append(
                PipelineBatchResult(
                    pipeline_id=pid,
                    prompts=prompts,
                    responses=responses,
                    full_texts=full_texts,
                    generation_time=gen_time,
                    n_samples=n_samples,
                )
            )

        self._n_batch_calls += 1
        return results

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release resources."""
        if not self.mock_mode and self.llm is not None:
            del self.llm
            self.llm = None
        self.lora_requests.clear()
