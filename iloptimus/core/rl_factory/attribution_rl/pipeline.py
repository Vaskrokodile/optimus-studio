"""A single RL pipeline: environment + targeted LoRA + GRPO trainer.

Each pipeline trains only ~5% of params (attribution-guided LoRA).
Multiple pipelines share a single vLLM instance via
:class:`~rl_factory.attribution_rl.multi_pipeline_rollout.MultiPipelineVLLMRollout`.

The pipeline is designed to work WITHOUT torch/vllm: the GRPO loss is recorded
as a mock scalar (no actual backprop) and the environment + verifier are plain
callables, so the whole thing can be unit-tested on CPU with a mock env.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from iloptimus.core.rl_factory.attribution_rl.targeted_lora import TargetedLoRAConfig


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a single RL pipeline.

    Attributes:
        pipeline_id: Unique id for this pipeline (also used as LoRA adapter name).
        target_domain: The capability domain this pipeline targets (e.g. "math").
        env_factory: Callable that constructs the environment when invoked.
        lora_config: Targeted LoRA config (which 5% of params to train).
        group_size: N completions per prompt (GRPO group size).
        num_prompts_per_step: How many distinct prompts to sample per training step.
        learning_rate: Learning rate for the LoRA update.
        kl_beta: KL penalty coefficient.
        max_steps: Maximum number of training steps.
        save_every: Save (and reload into vLLM) the adapter every N steps.
    """
    pipeline_id: str
    target_domain: str
    env_factory: Optional[Callable] = None
    lora_config: Optional[TargetedLoRAConfig] = None
    group_size: int = 8
    num_prompts_per_step: int = 4
    learning_rate: float = 1e-5
    kl_beta: float = 0.04
    max_steps: int = 100
    save_every: int = 10


@dataclass
class PipelineStepResult:
    """Result of a single training step for one pipeline."""
    pipeline_id: str
    step: int
    mean_reward: float
    mean_advantage: float
    loss: float
    kl_divergence: float
    n_samples: int
    generation_time: float
    training_time: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RLPipeline:
    """A single RL training pipeline.

    Each pipeline owns:
      * an environment (created from ``config.env_factory`` or passed directly),
      * a targeted LoRA config,
      * a GRPO-style trainer that computes group-relative advantages and a
        (mock) policy-gradient loss.

    The pipeline does NOT own a vLLM instance — generation is delegated to the
    shared :class:`MultiPipelineVLLMRollout`. The pipeline only produces
    :class:`PipelineRequest` objects and consumes :class:`PipelineBatchResult`
    objects.
    """

    def __init__(self, config: PipelineConfig, env: Any = None):
        self.config = config
        if env is not None:
            self.env = env
        elif config.env_factory is not None:
            self.env = config.env_factory()
        else:
            self.env = None

        self.step_count = 0
        self.history: list[PipelineStepResult] = []

        # Current adapter path (updated by the LoRA save callback)
        self._current_adapter_path: str = ""

        # Default LoRA config if none provided. The adapter NAME used by vLLM
        # is the pipeline_id (TargetedLoRAConfig itself has no adapter_name
        # field — it only describes which layers/modules to train).
        if self.config.lora_config is None:
            self.config.lora_config = TargetedLoRAConfig(
                target_domain=config.target_domain,
            )

    # ------------------------------------------------------------------
    # Properties for PipelineManager compatibility
    # ------------------------------------------------------------------

    @property
    def pipeline_id(self) -> str:
        """Expose pipeline_id for PipelineManager._pipeline_id()."""
        return self.config.pipeline_id

    def get_lora_adapter_path(self) -> str:
        """Return the current LoRA adapter path for vLLM reload."""
        return self._current_adapter_path

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def generate_prompts(self) -> list:
        """Reset the environment and produce PipelineRequest objects.

        Returns one :class:`PipelineRequest` per prompt, each requesting
        ``group_size`` completions.
        """
        # Late import to avoid a hard circular dependency at module load time.
        from iloptimus.core.rl_factory.attribution_rl.multi_pipeline_rollout import PipelineRequest

        prompts = self._get_prompts_from_env()
        # The LoRA adapter name used by vLLM is the pipeline_id.
        adapter_name = self.config.pipeline_id
        requests = []
        for prompt in prompts:
            requests.append(
                PipelineRequest(
                    pipeline_id=self.config.pipeline_id,
                    prompt=prompt,
                    n_completions=self.config.group_size,
                    lora_adapter_name=adapter_name,
                )
            )
        return requests

    def _get_prompts_from_env(self) -> list[str]:
        """Get a batch of prompts from the environment.

        Supports gymnasium-style envs (``reset()`` returns a tuple/dict) and
        simple mock envs (``reset()`` returns a list of strings, or exposes a
        ``prompts`` attribute / ``get_prompts()`` method).
        """
        if self.env is None:
            return [f"[{self.config.pipeline_id}] prompt {i}"
                    for i in range(self.config.num_prompts_per_step)]

        # Reset the env first to ensure problems are generated
        if hasattr(self.env, "reset"):
            self.env.reset()

        # Try a get_prompts() helper first
        if hasattr(self.env, "get_prompts"):
            prompts = self.env.get_prompts()
            if isinstance(prompts, list):
                return prompts[:self.config.num_prompts_per_step]

        # gymnasium-style reset
        if hasattr(self.env, "reset"):
            result = self.env.reset()
            # gymnasium: reset() -> (obs, info)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            # obs may itself be a list of prompt strings
            if isinstance(obs, list) and obs and isinstance(obs[0], str):
                return obs
            if isinstance(info, dict) and "prompts" in info:
                return list(info["prompts"])

        # Fallback: env exposes a .prompts attribute
        if hasattr(self.env, "prompts"):
            return list(self.env.prompts)

        return [f"[{self.config.pipeline_id}] prompt {i}"
                for i in range(self.config.num_prompts_per_step)]

    # ------------------------------------------------------------------
    # Result processing / scoring
    # ------------------------------------------------------------------

    def process_results(self, results) -> dict:
        """Take generation results, score them with the environment's verifier.

        Returns ``{"rewards": np.ndarray, "responses": list[str],
        "per_sample": list[dict]}``.
        """
        responses = list(results.responses)
        prompts = list(results.prompts)
        n = len(responses)

        rewards = np.zeros(n, dtype=np.float64)
        per_sample: list[dict] = []

        for i, resp in enumerate(responses):
            reward = self._score_response(resp, prompts[i] if i < len(prompts) else "")
            rewards[i] = float(reward)
            per_sample.append({
                "index": i,
                "response": resp,
                "reward": float(reward),
            })

        return {
            "rewards": rewards,
            "responses": responses,
            "per_sample": per_sample,
        }

    def _score_response(self, response: str, prompt: str = "") -> float:
        """Score a single response using the environment's verifier.

        Supports envs that expose ``verify(response)`` or ``score(response)``,
        or a gymnasium-style ``step(action)`` returning a reward. Falls back
        to a length-based score so tests can use a trivial mock env.
        """
        if self.env is not None:
            if hasattr(self.env, "verify"):
                try:
                    res = self.env.verify(response)
                    if isinstance(res, (int, float)):
                        return float(res)
                    if hasattr(res, "score"):
                        return float(res.score)
                    if isinstance(res, dict) and "score" in res:
                        return float(res["score"])
                except Exception:
                    pass
            if hasattr(self.env, "score"):
                try:
                    return float(self.env.score(response))
                except Exception:
                    pass
            if hasattr(self.env, "step"):
                try:
                    step_out = self.env.step(response)
                    # gymnasium: (obs, reward, terminated, truncated, info)
                    if isinstance(step_out, tuple) and len(step_out) >= 2:
                        return float(step_out[1])
                except Exception:
                    pass

        # Fallback: length-based reward (normalized)
        return float(len(response.split()))

    # ------------------------------------------------------------------
    # GRPO advantages
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """GRPO group-relative advantages.

        Reshape ``rewards`` into groups of ``group_size`` and compute
        ``(r_i - mean_group) / (std_group + eps)`` per group.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        group_size = self.config.group_size
        eps = 1e-8

        n = len(rewards)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        # If rewards aren't a clean multiple of group_size, pad/truncate to the
        # nearest full groups so the reshape is well-defined.
        n_groups = n // group_size
        usable = n_groups * group_size
        if usable == 0:
            # Single (partial) group: center around its own mean
            mean = rewards.mean()
            std = rewards.std()
            return (rewards - mean) / (std + eps)

        reshaped = rewards[:usable].reshape(n_groups, group_size)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        adv = (reshaped - means) / (stds + eps)
        return adv.reshape(-1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, results) -> PipelineStepResult:
        """Run one (mock) GRPO training step.

        Processing results -> rewards -> advantages -> (mock) loss.
        No actual backprop is performed (torch-free).
        """
        t0 = time.time()
        processed = self.process_results(results)
        rewards = processed["rewards"]
        advantages = self.compute_advantages(rewards)

        # Mock GRPO loss: -mean(advantages) (policy gradient surrogate).
        # In a real trainer this would be the per-token surrogate + KL penalty.
        if len(advantages) > 0:
            surrogate = -float(np.mean(advantages))
        else:
            surrogate = 0.0

        # Mock KL divergence (no reference model in torch-free mode)
        kl = 0.0
        loss = surrogate + self.config.kl_beta * kl

        self.step_count += 1

        # Update adapter path (simulated — in real training this would save
        # the LoRA adapter to disk after the optimizer step)
        self._current_adapter_path = (
            f"/tmp/lora_adapters/{self.config.pipeline_id}/step_{self.step_count}"
        )

        mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        mean_advantage = float(np.mean(advantages)) if len(advantages) > 0 else 0.0

        step_result = PipelineStepResult(
            pipeline_id=self.config.pipeline_id,
            step=self.step_count,
            mean_reward=mean_reward,
            mean_advantage=mean_advantage,
            loss=loss,
            kl_divergence=kl,
            n_samples=len(rewards),
            generation_time=getattr(results, "generation_time", 0.0),
            training_time=time.time() - t0,
        )
        self.history.append(step_result)
        return step_result

    # ------------------------------------------------------------------
    # LoRA update callback
    # ------------------------------------------------------------------

    def get_lora_update_callback(self) -> Callable:
        """Return a function that saves the LoRA adapter and returns its path.

        The returned callback is invoked by the PipelineManager after a
        training step to (a) persist the freshly-trained adapter and (b) hand
        the path back so vLLM can reload it.

        In torch-free mode this is a mock that just synthesises a path string
        based on the pipeline id and current step.
        """
        pipeline_id = self.config.pipeline_id

        def _save_adapter() -> str:
            path = f"/tmp/lora_adapters/{pipeline_id}/step_{self.step_count}"
            self._current_adapter_path = path
            return path

        return _save_adapter

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        mean_reward = (
            float(np.mean([h.mean_reward for h in self.history]))
            if self.history else 0.0
        )
        mean_loss = (
            float(np.mean([h.loss for h in self.history]))
            if self.history else 0.0
        )
        return {
            "step": self.step_count,
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "history": list(self.history),
        }
