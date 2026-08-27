#!/usr/bin/env python
"""End-to-end demo: 20 attribution-guided RL pipelines batched via vLLM.

This demo shows the full system working:
1. Run attribution on a mock model → identify which layers handle which capabilities
2. Create 20 pipelines, each targeting 5% of params via attribution-guided LoRA
3. Batch all 20 pipelines' generation into a single (mock) vLLM call
4. Run 3 rounds of GRPO training, all pipelines concurrently
5. Show that batch efficiency is ~1.0 (all pipelines batched together)

No torch/vLLM required — uses mock generation. Ready to swap in real vLLM
on a GPU box by providing a real MultiPipelineVLLMRollout.
"""
from __future__ import annotations

import numpy as np
import time

from iloptimus.core.rl_factory.attribution_rl.attribution import (
    AttributionEngine,
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
)
from iloptimus.core.rl_factory.attribution_rl.pipeline import (
    RLPipeline,
    PipelineConfig,
)
from iloptimus.core.rl_factory.attribution_rl.attribution_grpo import (
    AttributionGRPOTrainer,
    AttributionGRPOConfig,
)
from iloptimus.core.rl_factory.attribution_rl.pipeline_manager import (
    PipelineManager,
    PipelineManagerConfig,
)


# ─── 1. Mock model dimensions (simulating a ~1B param model) ───────────────
N_LAYERS = 32
N_HEADS = 16
D_MODEL = 2048
D_MLP = 8192
DOMAINS = ["math", "coding", "reasoning", "english", "logic"]
N_PIPELINES = 20


def make_mock_traces(n_layers, n_heads, d_model, d_mlp, domains, n_prompts=8, seed=42):
    """Generate synthetic model traces with domain-specific activation patterns."""
    rng = np.random.default_rng(seed)
    traces_by_domain = {}
    for d_idx, domain in enumerate(domains):
        traces = []
        for p in range(n_prompts):
            seq_len = 16
            # Domain-specific bias: each domain activates different layers/heads
            domain_bias = rng.normal(0, 0.1, d_model)
            # Layers 0-10: math-heavy, 11-20: coding-heavy, 21-31: reasoning-heavy
            layer_weights = np.ones(n_layers)
            if domain == "math":
                layer_weights[:11] *= 2.0
            elif domain == "coding":
                layer_weights[11:21] *= 2.0
            elif domain == "reasoning":
                layer_weights[21:] *= 2.0
            elif domain == "english":
                layer_weights[5:15] *= 1.8
            elif domain == "logic":
                layer_weights[15:25] *= 1.8

            residuals = np.zeros((n_layers + 1, seq_len, d_model))
            for l in range(n_layers + 1):
                base = rng.normal(0, 1, (seq_len, d_model))
                if l < n_layers:
                    base += layer_weights[l] * domain_bias
                residuals[l] = base

            # Attention: (n_layers, n_heads, seq_len, seq_len) — each row sums to 1
            attention = np.zeros((n_layers, n_heads, seq_len, seq_len))
            for l in range(n_layers):
                for h in range(n_heads):
                    # Base: uniform-ish distribution
                    base = rng.dirichlet(np.ones(seq_len))
                    attention[l, h] = np.tile(base, (seq_len, 1))
            # Domain-specific attention patterns
            if domain == "math":
                attention[:, :8, :, 0] += 0.3  # first-token attention
            elif domain == "coding":
                attention[:, 8:, :, -1] += 0.3  # last-token attention
            # Renormalize rows
            attention = attention / attention.sum(axis=-1, keepdims=True)

            mlp_acts = np.zeros((n_layers, seq_len, d_mlp))
            for l in range(n_layers):
                base = rng.normal(0, 1, (seq_len, d_mlp))
                # Domain-specific neurons fire
                neuron_start = (d_idx * d_mlp) // len(domains)
                neuron_end = ((d_idx + 1) * d_mlp) // len(domains)
                base[:, neuron_start:neuron_end] += layer_weights[l] * 2.0
                mlp_acts[l] = np.maximum(base, 0)  # ReLU

            logits = rng.normal(0, 1, (seq_len, 32000))

            traces.append(
                TraceData(
                    residuals=residuals,
                    attention=attention,
                    mlp_acts=mlp_acts,
                    logits=logits,
                    domain=domain,
                )
            )
        traces_by_domain[domain] = traces
    return traces_by_domain


def make_mock_env(domain, seed=42):
    """Create a mock environment that generates domain-specific prompts."""
    class MockEnv:
        def __init__(self):
            self.domain = domain
            self.rng = np.random.default_rng(seed)
            self.step_count = 0

        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.step_count = 0
            prompts = [f"Solve this {domain} problem #{i}: ..." for i in range(4)]
            obs = {"prompt": prompts[0]}
            info = {"problem_id": f"{domain}_0", "all_prompts": prompts}
            return obs, info

        def get_prompts(self):
            return [f"Solve this {domain} problem #{i}: ..." for i in range(4)]

        def verify(self, response):
            # Score by length (longer = better, up to a point)
            score = min(len(response) / 200.0, 1.0)
            correct = score > 0.5
            return type("Result", (), {"correct": correct, "score": score})()

        def step(self, action):
            self.step_count += 1
            return {"prompt": "..."}, 0.5, False, False, {}

    return MockEnv()


def mock_generate_fn(prompt: str, n: int) -> list[str]:
    """Mock vLLM generation: return n plausible-looking responses."""
    rng = np.random.default_rng(hash(prompt) % 2**32)
    responses = []
    for i in range(n):
        length = int(rng.integers(50, 200))
        resp = f"ANSWER: {rng.integers(0, 100)} " + "x" * length
        responses.append(resp)
    return responses


def main():
    print("=" * 70)
    print("Attribution-Guided Batched RL Training Demo")
    print("20 pipelines × 5% params each, batched via vLLM")
    print("=" * 70)

    # ─── Step 1: Attribution ──────────────────────────────────────────────
    print("\n[1] Running attribution on mock model...")
    print(f"    Model: {N_LAYERS} layers, {N_HEADS} heads, d_model={D_MODEL}")
    t0 = time.time()
    traces = make_mock_traces(N_LAYERS, N_HEADS, D_MODEL, D_MLP, DOMAINS)
    engine = AttributionEngine(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_mlp=D_MLP,
        domains=DOMAINS,
    )
    attr_result = engine.run(traces)
    print(f"    Attribution done in {time.time()-t0:.2f}s")
    print(f"    Domains: {attr_result.domains}")
    for domain in DOMAINS:
        scores = attr_result.layer_importance.get(domain, [])
        top3 = sorted(enumerate(scores), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"L{l}={s:.3f}" for l, s in top3)
        print(f"    {domain:>10s}: top layers = {top3_str}")

    # ─── Step 2: Targeted LoRA selection ──────────────────────────────────
    print(f"\n[2] Selecting targeted LoRA configs (5% budget per pipeline)...")
    selector = TargetedLoRASelector(attr_result)
    lora_configs = {}
    for domain in DOMAINS:
        cfg = selector.select_layers(
            domain=domain,
            budget_fraction=0.05,
            params_per_layer=12 * D_MODEL * D_MLP // N_LAYERS,  # rough estimate
        )
        lora_configs[domain] = cfg
        report = selector.get_coverage_report(cfg)
        print(f"    {domain:>10s}: {len(cfg.selected_layers)} layers selected "
              f"({report.get('fraction_of_params', 0):.1%} of params)")

    # ─── Step 3: Create 20 pipelines ──────────────────────────────────────
    print(f"\n[3] Creating {N_PIPELINES} RL pipelines...")
    pipelines = []
    for i in range(N_PIPELINES):
        domain = DOMAINS[i % len(DOMAINS)]
        env = make_mock_env(domain, seed=100 + i)
        lora_cfg = lora_configs[domain]
        # Override with pipeline-specific ID
        lora_cfg.target_domain = domain
        pconfig = PipelineConfig(
            pipeline_id=f"pipeline_{i}",
            target_domain=domain,
            lora_config=lora_cfg,
            group_size=8,
            num_prompts_per_step=4,
        )
        pipeline = RLPipeline(config=pconfig, env=env)
        pipelines.append(pipeline)
        if i < 5 or i >= N_PIPELINES - 2:
            print(f"    pipeline_{i}: domain={domain}, LoRA layers={len(lora_cfg.selected_layers)}")
        elif i == 5:
            print(f"    ... ({N_PIPELINES - 7} more pipelines) ...")

    # ─── Step 4: Setup multi-pipeline vLLM rollout ────────────────────────
    print(f"\n[4] Setting up multi-pipeline vLLM rollout (mock mode)...")
    rollout_config = MultiPipelineRolloutConfig(
        model_name="mock-model",
        max_loras=N_PIPELINES,
        enable_prefix_caching=True,
    )
    rollout = MultiPipelineVLLMRollout(rollout_config)
    rollout.set_mock_generate_fn(mock_generate_fn)

    # Register each pipeline's LoRA adapter
    for i, pipeline in enumerate(pipelines):
        rollout.add_pipeline_adapter(f"pipeline_{i}", f"/tmp/mock_lora_{i}")

    # ─── Step 5: Run PipelineManager ──────────────────────────────────────
    print(f"\n[5] Running 3 rounds of batched RL training...")
    pm_config = PipelineManagerConfig(
        n_pipelines=N_PIPELINES,
        sync_adapters_every=1,
        log_every=1,
    )
    manager = PipelineManager(pm_config, rollout, pipelines)

    results = manager.run(n_rounds=3)

    # ─── Step 6: Report results ───────────────────────────────────────────
    print(f"\n[6] Results:")
    print(f"{'Round':>6} {'Requests':>10} {'Samples':>10} {'Gen Time':>10} "
          f"{'Train Time':>10} {'Batch Eff':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.round:>6} {r.total_requests:>10} {r.total_samples:>10} "
              f"{r.generation_time:>9.3f}s {r.training_time:>9.3f}s "
              f"{r.batch_efficiency:>10.2f}")

    # Per-pipeline stats
    print(f"\n    Per-pipeline stats (last round):")
    last_round = results[-1]
    for pid, stats in sorted(last_round.per_pipeline.items())[:5]:
        if hasattr(stats, "mean_reward"):
            print(f"      {pid}: reward={stats.mean_reward:.3f}, "
                  f"loss={stats.loss:.4f}, n={stats.n_samples}")
        else:
            print(f"      {pid}: {stats}")
    print(f"      ... ({N_PIPELINES - 5} more)")

    # Global stats
    global_stats = manager.get_global_stats()
    print(f"\n    Global stats:")
    print(f"      Total rounds: {global_stats['rounds']}")
    print(f"      Total samples: {global_stats['total_samples']}")
    print(f"      Avg batch efficiency: {global_stats['avg_batch_efficiency']:.2f}")
    print(f"      Total gen time: {global_stats['total_gen_time']:.3f}s")
    print(f"      Total train time: {global_stats['total_train_time']:.3f}s")

    # ─── Key insight ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT: 20 pipelines batched into single vLLM call per round.")
    print(f"  Batch efficiency: {global_stats['avg_batch_efficiency']:.2f}")
    print(f"  (1.0 = perfect batching, all 20 pipelines at same speed as 1)")
    print(f"  Each pipeline trains only 5% of params (attribution-guided LoRA)")
    print(f"  → 20x pipeline parallelism at ~1x speed cost")
    print("=" * 70)


if __name__ == "__main__":
    main()
