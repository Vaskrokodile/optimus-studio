"""
GRPOTrainer — Group Relative Policy Optimization for art painting RL.

GRPO algorithm (DeepSeek, 2024):
  1. Sample B prompts
  2. For each prompt, generate N completions via vLLM (total B*N samples)
  3. Score each completion (Java execution + CLIP aesthetic scoring)
  4. Compute group-relative advantages:
       A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
     This replaces the value-function baseline from PPO — no critic needed.
  5. Policy gradient loss:
       L_pg = -mean(A_i * log_prob(response_i | prompt_i))
  6. KL penalty (to prevent the policy from drifting too far):
       L_kl = beta * mean(KL(pi_theta || pi_ref))
  7. Total loss = L_pg + L_kl
  8. Backprop + optimizer step (LoRA parameters only)
  9. Save LoRA adapter → reload into vLLM for next rollout

This trainer uses:
  - vLLM for ultra-batched generation (the fast path)
  - HF transformers + PEFT LoRA for gradient computation and updates
  - The base model (without LoRA) as the KL reference
"""

from __future__ import annotations

import os
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_HF = True
except ImportError:
    HAS_HF = False

from iloptimus.core.rl_factory.environments.art_painting_java import (
    extract_java_code,
    compute_art_reward,
)
from iloptimus.core.rl_factory.training.java_executor import JavaExecutorPool, JavaExecutionResult
from iloptimus.core.rl_factory.training.aesthetic_scorer import AestheticScorer


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    base_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    # GRPO
    num_prompts_per_step: int = 8        # B
    num_completions_per_prompt: int = 8  # N (total samples per step = B*N = 64)
    max_response_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    kl_beta: float = 0.04               # KL penalty coefficient
    advantage_eps: float = 1e-4         # for numerical stability
    group_normalize: bool = True         # normalize advantages within each prompt group

    # Training
    learning_rate: float = 1e-5
    num_steps: int = 500
    save_every: int = 50
    eval_every: int = 25
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # Java execution
    java_workers: int = 4
    java_timeout: float = 15.0

    # Aesthetic scoring
    aesthetic_batch_size: int = 64
    aesthetic_device: str = "cuda"

    # Generation
    vllm_gpu_memory_utilization: float = 0.25  # leave room for training model + CLIP
    system_prompt: str = (
        "You are an artist who paints using Java code. You write beautiful, "
        "visually striking Java programs using java.awt.Graphics2D that create "
        "stunning images. Always respond with a complete, compilable Java class "
        "in a ```java code block."
    )

    # Logging
    log_dir: str = "logs/art_rl"
    output_dir: str = "outputs/art_rl"


class GRPOTrainer:
    """
    GRPO training loop for art painting RL.

    Orchestrates: vLLM generation → Java execution → aesthetic scoring →
    advantage computation → policy gradient update → LoRA sync.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load training model (HF transformers + LoRA) ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # explicit GPU 0 to avoid vLLM memory conflicts
        )

        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()

        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # No separate reference model — use PEFT's disable_adapter() to get
        # base model logprobs for KL. This saves ~1GB VRAM on a 12GB GPU.
        self.ref_model = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # --- Java executor pool ---
        self.java_pool = JavaExecutorPool(num_workers=config.java_workers)

        # --- Aesthetic scorer (lazy load — heavy) ---
        self._aesthetic_scorer: Optional[AestheticScorer] = None

        # --- vLLM rollout (lazy load — heavy) ---
        self._vllm_rollout = None
        self._lora_temp_dir = tempfile.mkdtemp(prefix="grpo_lora_")

        # --- Logging ---
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_reward = 0.0

    @property
    def vllm_rollout(self):
        """Lazy-load vLLM to avoid GPU OOM during model setup."""
        if self._vllm_rollout is None:
            from iloptimus.core.rl_factory.training.vllm_rollout import VLLMRollout
            self._vllm_rollout = VLLMRollout(
                model_name=self.config.base_model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=2048,
                enable_lora=True,
                max_loras=1,
                max_lora_rank=self.config.lora_rank,
            )
        return self._vllm_rollout

    @property
    def aesthetic_scorer(self):
        """Lazy-load aesthetic scorer."""
        if self._aesthetic_scorer is None:
            self._aesthetic_scorer = AestheticScorer(
                device=self.config.aesthetic_device,
                batch_size=self.config.aesthetic_batch_size,
            )
        return self._aesthetic_scorer

    def train(self, prompt_fn) -> None:
        """
        Main training loop.

        Args:
            prompt_fn: Callable(seed) -> str that generates art prompts.
        """
        print("=" * 60)
        print("Starting GRPO training for Java art painting")
        print(f"  Model: {self.config.base_model_name}")
        print(f"  LoRA rank: {self.config.lora_rank}")
        print(f"  Batch: {self.config.num_prompts_per_step} prompts × "
              f"{self.config.num_completions_per_prompt} completions = "
              f"{self.config.num_prompts_per_step * self.config.num_completions_per_prompt} samples/step")
        print(f"  Steps: {self.config.num_steps}")
        print("=" * 60)

        # Start Java pool
        print("Starting Java execution pool...")
        self.java_pool.start()
        print(f"  {self.config.java_workers} JVM workers ready")

        # Initial LoRA sync to vLLM
        self._sync_lora_to_vllm()

        for step in range(self.config.num_steps):
            self.step = step
            step_start = time.time()

            # 1. Generate prompts
            prompts = [
                prompt_fn(seed=step * 1000 + i)
                for i in range(self.config.num_prompts_per_step)
            ]

            # 2. Generate completions with vLLM (ultra-batched)
            gen_start = time.time()
            rollout = self.vllm_rollout.generate_batch(
                prompts,
                n=self.config.num_completions_per_prompt,
                max_tokens=self.config.max_response_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                system_prompt=self.config.system_prompt,
            )
            gen_time = time.time() - gen_start

            # 3. Execute Java code + score images
            score_start = time.time()
            rewards, reward_infos = self._score_completions(
                rollout.responses, prompts, rollout.num_samples
            )
            score_time = time.time() - score_start

            # 4. Compute group-relative advantages
            advantages = self._compute_advantages(
                rewards, self.config.num_completions_per_prompt
            )

            # 5. Compute logprobs and GRPO loss
            train_start = time.time()
            # 6. Compute GRPO loss and backprop (gradient accumulation inside)
            self.optimizer.zero_grad()
            loss, loss_info = self._compute_grpo_loss(
                rollout.prompt_token_ids,
                rollout.response_token_ids,
                advantages,
            )
            # Gradients already accumulated by _compute_grpo_loss
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            train_time = time.time() - train_start

            # 7. Sync LoRA to vLLM
            sync_start = time.time()
            self._sync_lora_to_vllm()
            sync_time = time.time() - sync_start

            # 8. Log
            step_time = time.time() - step_start
            mean_reward = float(np.mean(rewards))
            max_reward = float(np.max(rewards))
            compile_rate = float(np.mean([ri["compile_success"] for ri in reward_infos]))
            exec_rate = float(np.mean([ri["execution_success"] for ri in reward_infos]))
            mean_aesthetic = float(np.mean([ri["aesthetic_score"] for ri in reward_infos]))

            print(
                f"Step {step:4d} | reward: {mean_reward:.4f} (max {max_reward:.4f}) | "
                f"aesthetic: {mean_aesthetic:.4f} | "
                f"compile: {compile_rate:.2%} | exec: {exec_rate:.2%} | "
                f"loss: {loss.item():.4f} | "
                f"gen: {gen_time:.1f}s score: {score_time:.1f}s "
                f"train: {train_time:.1f}s sync: {sync_time:.1f}s "
                f"total: {step_time:.1f}s"
            )

            # Save best model
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self._save_model(suffix="best")

            # Periodic save
            if (step + 1) % self.config.save_every == 0:
                self._save_model(suffix=f"step_{step+1}")

        # Final save
        self._save_model(suffix="final")
        self._cleanup()

    def _score_completions(
        self,
        responses: list[str],
        prompts: list[str],
        num_samples: int,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Execute Java code from each response and score the resulting images.

        Pipeline:
          1. Extract Java code from each response
          2. Batch-compile and execute via JavaExecutorPool
          3. Batch-score images via AestheticScorer
          4. Compute final rewards
        """
        # Extract Java code
        codes = []
        code_found = []
        for resp in responses:
            code = extract_java_code(resp)
            codes.append(code)
            code_found.append(code is not None)

        # Execute Java code in parallel
        # For responses without code, skip execution
        png_results: list[Optional[bytes]] = [None] * num_samples
        compile_success = [False] * num_samples
        exec_success = [False] * num_samples
        exec_errors = [""] * num_samples

        # Prepare sources for execution (only those with extracted code)
        sources_to_run = []
        indices_to_run = []
        for i, code in enumerate(codes):
            if code is not None:
                sources_to_run.append(code)
                indices_to_run.append(i)

        if sources_to_run:
            results = self.java_pool.execute_batch(
                sources_to_run, timeout=self.config.java_timeout
            )
            for idx, result in zip(indices_to_run, results):
                compile_success[idx] = result.error_type != "compile"
                exec_success[idx] = result.success
                if result.success:
                    png_results[idx] = result.png_bytes
                else:
                    exec_errors[idx] = result.error or "unknown error"

        # Score images with CLIP aesthetic predictor
        valid_pngs = []
        valid_indices = []
        for i, png in enumerate(png_results):
            if png is not None:
                valid_pngs.append(png)
                valid_indices.append(i)

        aesthetic_scores = np.zeros(num_samples)
        if valid_pngs:
            scores = self.aesthetic_scorer.score_png_bytes_batch(valid_pngs)
            for idx, score in zip(valid_indices, scores.tolist()):
                aesthetic_scores[idx] = score

        # Compute final rewards
        rewards = np.zeros(num_samples)
        reward_infos = []
        for i in range(num_samples):
            reward, info = compute_art_reward(
                response=responses[i],
                aesthetic_score=aesthetic_scores[i],
                compile_success=compile_success[i],
                execution_success=exec_success[i],
                code=codes[i],
                diversity_penalty=0.0,  # TODO: implement diversity check
            )
            rewards[i] = reward
            info["exec_error"] = exec_errors[i]
            reward_infos.append(info)

        return rewards, reward_infos

    def _compute_advantages(
        self, rewards: np.ndarray, group_size: int
    ) -> torch.Tensor:
        """
        Compute GRPO group-relative advantages.

        For each group of N completions from the same prompt:
          A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

        If group_normalize is False, just center: A_i = r_i - mean(r_group)
        """
        advantages = np.zeros_like(rewards)
        num_groups = len(rewards) // group_size

        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group_rewards = rewards[start:end]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std()

            if self.config.group_normalize:
                advantages[start:end] = (group_rewards - mean_r) / (std_r + self.config.advantage_eps)
            else:
                advantages[start:end] = group_rewards - mean_r

        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def _compute_grpo_loss(
        self,
        prompt_token_ids: list[list[int]],
        response_token_ids: list[list[int]],
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss: policy gradient + KL penalty.

        L = -mean(A_i * log_prob(response_i | prompt_i))
            + beta * mean(KL(pi_theta || pi_ref))
        """
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        num_samples = len(prompt_token_ids)
        batch_size = 1  # process 1 sample at a time to save VRAM

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_adv = advantages[batch_start:batch_end]

            batch_logprobs = []
            batch_ref_logprobs = []

            for i in range(batch_start, batch_end):
                p_ids = prompt_token_ids[i]
                r_ids = response_token_ids[i]

                if len(r_ids) == 0:
                    batch_logprobs.append(torch.tensor(0.0, device=self.device))
                    batch_ref_logprobs.append(torch.tensor(0.0, device=self.device))
                    continue

                # Build full sequence: prompt + response
                full_ids = p_ids + r_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)

                # Response token positions in the shifted sequence:
                # logits[t] predicts token[t+1], so response token r_ids[j]
                # is predicted by logits[len(p_ids) - 1 + j]
                resp_start = len(p_ids) - 1
                resp_end = resp_start + len(r_ids)
                resp_labels = torch.tensor([r_ids], dtype=torch.long, device=self.device)

                # --- Policy forward pass (with LoRA, gradients on) ---
                # Don't pass labels — avoids HF's internal float32 loss upcast
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits

                # Slice only the response-predicting positions to save VRAM
                resp_logits = logits[:, resp_start:resp_end, :].contiguous()
                del logits, outputs

                # Compute per-token logprobs using cross_entropy (memory-efficient)
                # cross_entropy computes log_softmax + NLL internally without
                # materializing the full log_softmax tensor
                flat_logits = resp_logits.view(-1, resp_logits.size(-1))
                token_logprobs = -F.cross_entropy(
                    flat_logits, resp_labels.view(-1), reduction="none"
                )
                del resp_logits, flat_logits

                sample_logprob = token_logprobs.mean()
                batch_logprobs.append(sample_logprob)

                # --- Reference forward pass (LoRA disabled, no grad) ---
                with torch.no_grad():
                    with self.model.disable_adapter():
                        ref_outputs = self.model(input_ids=input_ids)
                    ref_logits = ref_outputs.logits[:, resp_start:resp_end, :].contiguous()
                    del ref_outputs

                    flat_ref = ref_logits.view(-1, ref_logits.size(-1))
                    ref_token_logprobs = -F.cross_entropy(
                        flat_ref, resp_labels.view(-1), reduction="none"
                    )
                    del ref_logits, flat_ref

                    ref_sample_logprob = ref_token_logprobs.mean()
                    batch_ref_logprobs.append(ref_sample_logprob)

            batch_logprobs = torch.stack(batch_logprobs)
            batch_ref_logprobs = torch.stack(batch_ref_logprobs)

            # Policy gradient loss: -mean(A * log_prob)
            pg_loss = -(batch_adv * batch_logprobs).mean()

            # KL penalty: mean(log_prob_theta - log_prob_ref)
            kl_loss = (batch_logprobs - batch_ref_logprobs).mean()

            batch_loss = pg_loss + self.config.kl_beta * kl_loss

            if batch_start == 0:
                total_pg_loss = pg_loss.detach()
                total_kl_loss = kl_loss.detach()

            # Backprop each mini-batch separately (gradient accumulation)
            # Gradients accumulate in .grad — caller must zero_grad before calling
            scale = 1.0 / ((num_samples + batch_size - 1) // batch_size)
            (batch_loss * scale).backward()

        loss_info = {
            "pg_loss": float(total_pg_loss),
            "kl_loss": float(total_kl_loss),
        }
        return torch.tensor(0.0), loss_info  # loss already backpropped

    def _sync_lora_to_vllm(self) -> None:
        """Save LoRA adapter and reload into vLLM."""
        adapter_path = os.path.join(self._lora_temp_dir, f"step_{self.step}")
        self.model.save_pretrained(adapter_path)
        self.vllm_rollout.reload_lora(adapter_path)

    def _save_model(self, suffix: str = "") -> None:
        """Save the LoRA adapter."""
        save_path = self.output_dir / f"lora_{suffix}"
        self.model.save_pretrained(str(save_path))
        print(f"  Saved LoRA adapter to {save_path}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.java_pool.stop()
        if self._vllm_rollout is not None:
            self._vllm_rollout.shutdown()
