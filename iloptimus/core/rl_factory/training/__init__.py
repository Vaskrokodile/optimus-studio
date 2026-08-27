"""Training infrastructure for rl-factory (vLLM rollouts + GRPO + async RL).

Heavy dependencies (torch, vLLM, PIL) are imported lazily/guarded so that the
lightweight modules — ``JavaExecutor``, ``CurriculumSelector``, ``AsyncGRPOTrainer``,
``PrefixCacheRollout``, ``SpeculativeRollout``, ``MemoryEfficientTraining``,
``ParallelRewardComputer``, ``R3Routing`` — remain usable in minimal
environments (e.g. CI with only numpy installed).
"""

# Lightweight modules — numpy/stdlib only, always available.
from iloptimus.core.rl_factory.training.java_executor import JavaExecutor, JavaExecutorPool, JavaExecutionResult
from iloptimus.core.rl_factory.training.curriculum_selector import CurriculumSelector, TaskStats
from iloptimus.core.rl_factory.training.async_grpo_trainer import AsyncGRPOConfig, AsyncGRPOTrainer
from iloptimus.core.rl_factory.training.prefix_cache_rollout import PrefixCacheConfig, PrefixCacheRollout
from iloptimus.core.rl_factory.training.speculative_rollout import SpeculativeConfig, SpeculativeRollout
from iloptimus.core.rl_factory.training.memory_efficient import MemoryConfig, TokenSelector, MemoryTracker
from iloptimus.core.rl_factory.training.parallel_reward import (
    ParallelRewardConfig,
    ParallelRewardComputer,
    AdaptiveRolloutAllocator,
)
from iloptimus.core.rl_factory.training.r3_routing import R3Config, RoutingRecorder, RoutingReplayer

__all__ = [
    # Lightweight (always available)
    "JavaExecutor", "JavaExecutorPool", "JavaExecutionResult",
    "CurriculumSelector", "TaskStats",
    "AsyncGRPOConfig", "AsyncGRPOTrainer",
    "PrefixCacheConfig", "PrefixCacheRollout",
    "SpeculativeConfig", "SpeculativeRollout",
    "MemoryConfig", "TokenSelector", "MemoryTracker",
    "ParallelRewardConfig", "ParallelRewardComputer", "AdaptiveRolloutAllocator",
    "R3Config", "RoutingRecorder", "RoutingReplayer",
    # Heavy (optional, guarded below)
    "AestheticScorer", "AestheticPredictor",
    "GRPOTrainer", "GRPOConfig",
    "VLLMRollout", "RolloutResult",
    "MultiLoRAVLLMRollout",
]

# Heavy, optional dependencies — only exported when available.
try:  # aesthetic scorer needs torch + PIL
    from iloptimus.core.rl_factory.training.aesthetic_scorer import AestheticScorer, AestheticPredictor
except ImportError:
    pass

try:  # GRPO trainer needs torch
    from iloptimus.core.rl_factory.training.grpo_trainer import GRPOTrainer, GRPOConfig
except ImportError:
    pass

try:  # vLLM rollout needs torch + vllm
    from iloptimus.core.rl_factory.training.vllm_rollout import VLLMRollout, RolloutResult
except ImportError:
    pass

try:  # multi-LoRA rollout needs torch + vllm
    from iloptimus.core.rl_factory.training.multi_lora_rollout import MultiLoRAVLLMRollout
except ImportError:
    pass
