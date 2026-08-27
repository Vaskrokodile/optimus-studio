"""Attribution-guided batched RL training system.

Uses slm-microscope-style attribution to determine which weights/layers handle
which capabilities, then trains 20 RL pipelines concurrently — each targeting
only 5% of total params via targeted LoRA — with vLLM batching all pipelines'
generation into a single batched call.

Modules:
- attribution: AttributionEngine (causal patching + attention spec + weight sensitivity)
- targeted_lora: TargetedLoRASelector (select which 5% of layers to train)
- multi_pipeline_rollout: MultiPipelineVLLMRollout (batch all pipelines' generation)
- pipeline: RLPipeline (single pipeline: env + LoRA + GRPO)
- attribution_grpo: AttributionGRPOTrainer (GRPO with NAT + R3 routing)
- pipeline_manager: PipelineManager (orchestrate 20 concurrent pipelines)
"""
from iloptimus.core.rl_factory.attribution_rl.attribution import (
    AttributionEngine,
    AttributionResult,
    TraceData,
)
from iloptimus.core.rl_factory.attribution_rl.targeted_lora import (
    TargetedLoRAConfig,
    TargetedLoRASelector,
)
from iloptimus.core.rl_factory.attribution_rl.multi_pipeline_rollout import (
    MultiPipelineVLLMRollout,
    MultiPipelineRolloutConfig,
    PipelineRequest,
    PipelineBatchResult,
)
from iloptimus.core.rl_factory.attribution_rl.pipeline import (
    RLPipeline,
    PipelineConfig,
    PipelineStepResult,
)
from iloptimus.core.rl_factory.attribution_rl.attribution_grpo import (
    AttributionGRPOTrainer,
    AttributionGRPOConfig,
)
from iloptimus.core.rl_factory.attribution_rl.pipeline_manager import (
    PipelineManager,
    PipelineManagerConfig,
    RoundResult,
)

__all__ = [
    "AttributionEngine", "AttributionResult", "TraceData",
    "TargetedLoRAConfig", "TargetedLoRASelector",
    "MultiPipelineVLLMRollout", "MultiPipelineRolloutConfig",
    "PipelineRequest", "PipelineBatchResult",
    "RLPipeline", "PipelineConfig", "PipelineStepResult",
    "AttributionGRPOTrainer", "AttributionGRPOConfig",
    "PipelineManager", "PipelineManagerConfig", "RoundResult",
]
