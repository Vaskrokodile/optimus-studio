"""RL batching: param-split architecture for parallel RL training.

This is the novel pipeline: the model's parameters are split into N parts,
each trained on a different RL environment via targeted LoRA. vLLM batches
all pipelines' generation into a single call, so N pipelines train at ~1x speed.

This module wraps the attribution_rl pipeline manager and provides a
simplified API for the RL mode orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rl_factory.attribution_rl.pipeline_manager import (
    PipelineManager,
    PipelineManagerConfig,
    RoundResult,
)
from ..rl_factory.attribution_rl.multi_pipeline_rollout import (
    MultiPipelineVLLMRollout,
    MultiPipelineRolloutConfig,
    PipelineRequest,
    PipelineBatchResult,
)
from ..rl_factory.attribution_rl.targeted_lora import (
    TargetedLoRAConfig,
    TargetedLoRASelector,
)
from ..rl_factory.attribution_rl.attribution import AttributionEngine


@dataclass
class RLBatchingConfig:
    """Configuration for RL batching (param-split across environments)."""

    n_pipelines: int = 20
    steps_per_round: int = 1
    sync_adapters_every: int = 1
    parallel_reward_workers: int = 4
    # Targeted LoRA: what fraction of params each pipeline trains
    lora_param_fraction: float = 0.05  # 5% of params per pipeline
    lora_rank: int = 16
    # vLLM multi-LoRA config
    max_loras: int = 20
    model_name: str = ""
    # Mock mode (for testing without vLLM)
    mock_mode: bool = False
    mock_generate_fn: Any = None  # callable(prompt, n) -> list[str]


@dataclass
class RLBatchingResult:
    """Result of one RL batching round."""

    round: int
    total_samples: int
    generation_time: float
    training_time: float
    batch_efficiency: float
    per_pipeline: dict[str, dict] = field(default_factory=dict)
    total_reward: float = 0.0
    mean_reward: float = 0.0


def create_rl_batching_rollout(
    config: RLBatchingConfig,
    model_name: str = "",
) -> MultiPipelineVLLMRollout:
    """Create a multi-pipeline vLLM rollout for RL batching.

    In mock mode (no vLLM), uses a user-supplied generate function.
    """
    rollout_config = MultiPipelineRolloutConfig(
        model_name=model_name or config.model_name,
        max_loras=config.max_loras,
        max_lora_rank=config.lora_rank,
    )
    rollout = MultiPipelineVLLMRollout(rollout_config)
    # Set mock generation function if provided (used when vLLM is not installed)
    if config.mock_generate_fn is not None:
        rollout.set_mock_generate_fn(config.mock_generate_fn)
    return rollout


def create_targeted_lora_selector(
    config: RLBatchingConfig,
    model_name: str = "",
) -> TargetedLoRASelector:
    """Create a targeted LoRA selector that picks which layers to train."""
    lora_config = TargetedLoRAConfig(
        param_fraction=config.lora_param_fraction,
        rank=config.lora_rank,
    )
    return TargetedLoRASelector(lora_config)


def run_rl_batching_round(
    manager: PipelineManager,
    config: RLBatchingConfig,
) -> RLBatchingResult:
    """Run one round of RL batching across all pipelines.

    This collects generation requests from all N pipelines, batches them
    into a single vLLM call, distributes results, and runs per-pipeline
    training steps.
    """
    round_result = manager.run_round()

    # Aggregate stats
    total_reward = 0.0
    rewards = []
    for pipeline_id, stats in round_result.per_pipeline.items():
        reward = stats.get("mean_reward", 0.0)
        total_reward += reward
        rewards.append(reward)

    mean_reward = total_reward / len(rewards) if rewards else 0.0

    return RLBatchingResult(
        round=round_result.round,
        total_samples=round_result.total_samples,
        generation_time=round_result.generation_time,
        training_time=round_result.training_time,
        batch_efficiency=round_result.batch_efficiency,
        per_pipeline=round_result.per_pipeline,
        total_reward=total_reward,
        mean_reward=mean_reward,
    )


def build_pipeline_requests(
    env_templates: list[dict[str, Any]],
    n_completions: int = 8,
) -> list[PipelineRequest]:
    """Build pipeline requests from a list of environment template specs.

    Each spec should have:
    - pipeline_id: unique identifier for this pipeline
    - prompt: the problem prompt
    - lora_adapter_name: which LoRA adapter to use
    - sampling_params: generation kwargs
    """
    requests: list[PipelineRequest] = []
    for spec in env_templates:
        requests.append(PipelineRequest(
            pipeline_id=spec["pipeline_id"],
            prompt=spec["prompt"],
            n_completions=spec.get("n_completions", n_completions),
            lora_adapter_name=spec.get("lora_adapter_name", spec["pipeline_id"]),
            sampling_params=spec.get("sampling_params", {"temperature": 0.7, "max_tokens": 1024}),
        ))
    return requests
