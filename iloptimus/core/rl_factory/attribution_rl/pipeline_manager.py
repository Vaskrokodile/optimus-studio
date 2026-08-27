"""Pipeline manager: orchestrate 20 concurrent RL pipelines.

Collects generation requests from all pipelines, batches them into a single
vLLM call via MultiPipelineVLLMRollout, distributes results back, and
coordinates per-pipeline training steps.

The key property: all N pipelines' generation runs in ONE batched vLLM call,
so N pipelines train at ~the same speed as 1 pipeline (batch_efficiency ~1.0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np

from iloptimus.core.rl_factory.attribution_rl.multi_pipeline_rollout import (
    MultiPipelineVLLMRollout,
    MultiPipelineRolloutConfig,
    PipelineRequest,
    PipelineBatchResult,
)


@dataclass
class PipelineManagerConfig:
    """Configuration for the PipelineManager."""
    n_pipelines: int = 20
    steps_per_round: int = 1          # steps each pipeline runs per round
    sync_adapters_every: int = 1      # how often to reload LoRA adapters into vLLM
    parallel_reward_workers: int = 4
    log_every: int = 1


@dataclass
class RoundResult:
    """Result of one round of batched RL training across all pipelines."""
    round: int
    total_requests: int
    total_samples: int
    generation_time: float
    training_time: float
    per_pipeline: dict[str, dict] = field(default_factory=dict)
    batch_efficiency: float = 0.0  # actual_batch_size / n_pipelines (~1.0 ideal)


class PipelineManager:
    """Orchestrates N concurrent RL pipelines with batched generation.

    Each pipeline object is expected to implement:
      - generate_prompts() -> list[PipelineRequest]
      - train_step(result: PipelineBatchResult) -> dict
      - get_lora_adapter_path() -> str  (for adapter reload)
      - get_stats() -> dict
    """

    def __init__(
        self,
        config: PipelineManagerConfig,
        rollout: MultiPipelineVLLMRollout,
        pipelines: list,
    ):
        self.config = config
        self.rollout = rollout
        self.pipelines = list(pipelines)

        if len(self.pipelines) != config.n_pipelines:
            raise ValueError(
                f"Expected {config.n_pipelines} pipelines, got {len(self.pipelines)}"
            )

        self._round = 0
        self._round_results: list[RoundResult] = []
        self._total_samples = 0
        self._total_gen_time = 0.0
        self._total_train_time = 0.0

    # ------------------------------------------------------------------
    # Core round logic
    # ------------------------------------------------------------------
    def run_round(self) -> RoundResult:
        """Run one round of batched RL training across all pipelines.

        1. Collect prompts from ALL pipelines.
        2. Flatten into one list of PipelineRequest.
        3. Single MultiPipelineVLLMRollout.generate_batch(all_requests).
        4. Group results by pipeline_id.
        5. For each pipeline: train_step(results).
        6. Every sync_adapters_every rounds: reload LoRA adapters.
        """
        round_idx = self._round
        self._round += 1

        # 1+2. Collect & flatten all requests.
        all_requests = self.collect_all_requests()
        total_requests = len(all_requests)

        # 3. Single batched generation call for ALL pipelines.
        gen_start = time.time()
        all_results = self.rollout.generate_batch(all_requests)
        generation_time = time.time() - gen_start

        # 4. Group results by pipeline_id.
        by_pipeline = self.distribute_results(all_results)

        # 5. Per-pipeline training step.
        train_start = time.time()
        per_pipeline: dict[str, dict] = {}
        for pipeline in self.pipelines:
            pid = self._pipeline_id(pipeline)
            result = by_pipeline.get(pid)
            if result is None:
                # Pipeline produced no requests this round; skip.
                per_pipeline[pid] = {"skipped": True}
                continue
            for _ in range(self.config.steps_per_round):
                step_result = pipeline.train_step(result)
            per_pipeline[pid] = step_result
        training_time = time.time() - train_start

        # 6. Sync adapters.
        if self.config.sync_adapters_every > 0 and (
            round_idx % self.config.sync_adapters_every == 0
        ):
            self.sync_adapters()

        total_samples = sum(r.n_samples for r in all_results)

        # batch_efficiency: how well we batched. Ideal = 1.0 (all pipelines
        # served in one call). actual_batch_size = number of distinct
        # pipelines that had requests / n_pipelines.
        n_active = len(by_pipeline)
        batch_efficiency = n_active / self.config.n_pipelines if self.config.n_pipelines > 0 else 0.0

        result = RoundResult(
            round=round_idx,
            total_requests=total_requests,
            total_samples=total_samples,
            generation_time=generation_time,
            training_time=training_time,
            per_pipeline=per_pipeline,
            batch_efficiency=batch_efficiency,
        )
        self._round_results.append(result)
        self._total_samples += total_samples
        self._total_gen_time += generation_time
        self._total_train_time += training_time

        if self.config.log_every > 0 and (round_idx % self.config.log_every == 0):
            self._log_round(result)

        return result

    def run(self, n_rounds: int) -> list[RoundResult]:
        """Run n_rounds of run_round(). Returns all round results."""
        results: list[RoundResult] = []
        for _ in range(n_rounds):
            results.append(self.run_round())
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def collect_all_requests(self) -> list[PipelineRequest]:
        """Call generate_prompts() on each pipeline and flatten into one list."""
        all_requests: list[PipelineRequest] = []
        for pipeline in self.pipelines:
            reqs = pipeline.generate_prompts()
            all_requests.extend(reqs)
        return all_requests

    def distribute_results(
        self, all_results: list[PipelineBatchResult]
    ) -> dict[str, PipelineBatchResult]:
        """Group results by pipeline_id."""
        by_pipeline: dict[str, PipelineBatchResult] = {}
        for r in all_results:
            by_pipeline[r.pipeline_id] = r
        return by_pipeline

    def sync_adapters(self) -> None:
        """Reload each pipeline's LoRA adapter into vLLM."""
        for pipeline in self.pipelines:
            pid = self._pipeline_id(pipeline)
            path = self._get_lora_path(pipeline)
            if path is not None:
                self.rollout.reload_pipeline_adapter(pid, path)

    def get_global_stats(self) -> dict:
        """Return global training statistics across all rounds."""
        per_pipeline: dict[str, dict] = {}
        for pipeline in self.pipelines:
            pid = self._pipeline_id(pipeline)
            stats_fn = getattr(pipeline, "get_stats", None)
            if callable(stats_fn):
                per_pipeline[pid] = stats_fn()

        avg_batch_eff = (
            float(np.mean([r.batch_efficiency for r in self._round_results]))
            if self._round_results else 0.0
        )

        return {
            "rounds": self._round,
            "total_samples": self._total_samples,
            "avg_batch_efficiency": avg_batch_eff,
            "per_pipeline": per_pipeline,
            "total_gen_time": self._total_gen_time,
            "total_train_time": self._total_train_time,
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _pipeline_id(pipeline: Any) -> str:
        """Get a pipeline's id, preferring a .pipeline_id attribute."""
        pid = getattr(pipeline, "pipeline_id", None)
        if pid is not None:
            return str(pid)
        return str(id(pipeline))

    @staticmethod
    def _get_lora_path(pipeline: Any) -> Optional[str]:
        """Get a pipeline's LoRA adapter path if available."""
        fn = getattr(pipeline, "get_lora_adapter_path", None)
        if callable(fn):
            return fn()
        return getattr(pipeline, "lora_adapter_path", None)

    def _log_round(self, result: RoundResult) -> None:
        print(
            f"Round {result.round:4d} | requests: {result.total_requests} | "
            f"samples: {result.total_samples} | "
            f"batch_eff: {result.batch_efficiency:.2f} | "
            f"gen: {result.generation_time:.3f}s | "
            f"train: {result.training_time:.3f}s"
        )
