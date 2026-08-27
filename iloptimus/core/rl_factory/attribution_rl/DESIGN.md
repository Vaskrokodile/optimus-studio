# Attribution-Guided Batched RL Training System

## Vision

Use slm-microscope's attribution approach to identify which weights/layers/experts
handle which capabilities, then train 20 RL pipelines concurrently — each targeting
only 5% of total params — with vLLM batching all pipelines' generation into a single
batched call (same speed as 1 pipeline).

## Core Insight

vLLM's multi-LoRA serving can batch requests from multiple LoRA adapters at near-zero
overhead (Punica SGMV kernels). If each RL pipeline trains only 5% of params via a
targeted LoRA adapter, 20 pipelines can share a single vLLM instance. The attribution
system determines WHICH 5% each pipeline should train.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         PipelineManager              │
                    │  (orchestrates 20 concurrent RL     │
                    │   pipelines, each with its own      │
                    │   targeted LoRA + environment)      │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │    MultiPipelineVLLMRollout          │
                    │  (single vLLM instance, batches all  │
                    │   20 pipelines' generation requests, │
                    │   each tagged with its LoRA adapter) │
                    └──────────┬──────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
   │ Pipeline 1  │    │ Pipeline 2   │    │ Pipeline 20  │
   │ LoRA: L5-8  │    │ LoRA: L12-16 │    │ LoRA: L28-32 │
   │ Env: Math   │    │ Env: Coding  │    │ Env: Logic   │
   │ GRPO update │    │ GRPO update  │    │ GRPO update  │
   └─────────────┘    └──────────────┘    └──────────────┘
```

## Modules

### 1. AttributionEngine (attribution.py)
Adapted from slm-microscope's attribution_v2.py:
- **Causal attribution**: activation patching — replace layer N's output with domain B's, measure logit shift
- **Attention head specialization**: per-head domain distance from attention pattern features
- **Weight matrix sensitivity**: cosine distance of projection outputs across domains
- **Layer importance ranking**: combined score (0.3*MLP + 0.2*attn + 0.5*causal)
- **Param knowledge map**: classify each weight matrix as domain-specific / shared / dead

Output: `AttributionResult` — per-layer, per-weight-matrix, per-domain importance scores.

### 2. TargetedLoRA (targeted_lora.py)
Uses attribution to select WHICH layers get LoRA adapters:
- Input: AttributionResult + target capability + param budget (5%)
- Output: list of layer indices + module names to apply LoRA to
- Only trains the layers that attribution says handle the target capability
- This is the "5% of params" — not random layers, but the RIGHT layers

### 3. MultiPipelineVLLMRollout (multi_pipeline_rollout.py)
Extends MultiLoRAVLLMRollout for multi-pipeline RL:
- Single vLLM instance serving all 20 pipelines' LoRA adapters
- `generate_batch(pipeline_requests)`: batches all pipelines' requests into one vLLM call
- Each request tagged with its pipeline's LoRA adapter
- Tracks per-pipeline generation stats
- Prefix caching across shared system prompts

### 4. AttributionGRPO (attribution_grpo.py)
GRPO training that only updates attributed layers:
- Group-relative advantages (no critic)
- Policy gradient + KL penalty
- Only computes gradients for LoRA layers (the attributed 5%)
- Per-pipeline training step (each pipeline updates its own LoRA independently)
- R3 routing replay for MoE stability

### 5. RLPipeline (pipeline.py)
A single RL training pipeline:
- Has its own TargetedLoRA config (which 5% to train)
- Has its own environment (from rl_factory.environments)
- Has its own GRPO trainer
- Generates rollouts via the shared MultiPipelineVLLMRollout
- Updates its LoRA adapter after each step

### 6. PipelineManager (pipeline_manager.py)
Orchestrates 20 concurrent pipelines:
- Creates pipelines with attribution-guided LoRA targeting
- Batches all pipelines' generation requests
- Distributes results back to each pipeline
- Coordinates LoRA reloads after training steps
- Tracks global training stats

## Data Flow

1. **Attribution phase** (one-time):
   - Run attribution on base model → AttributionResult
   - For each target capability, select 5% of layers → TargetedLoRA configs

2. **Training phase** (per step):
   - Each pipeline generates prompts from its environment
   - PipelineManager collects all prompts from all 20 pipelines
   - MultiPipelineVLLMRollout batches all prompts into one vLLM call
   - Each prompt tagged with its pipeline's LoRA adapter
   - vLLM generates all completions in one batched call
   - Results distributed back to each pipeline
   - Each pipeline scores completions (parallel reward computation)
   - Each pipeline computes GRPO loss and updates its LoRA
   - Each pipeline reloads its LoRA into vLLM

## Key Properties

- **20x pipeline parallelism at 1x speed**: vLLM batches all generation
- **5% params per pipeline**: attribution-guided LoRA targeting
- **No capability interference**: each pipeline trains different layers
- **MoE stable**: R3 routing replay prevents training collapse
- **Memory efficient**: only 5% params have gradients per pipeline
