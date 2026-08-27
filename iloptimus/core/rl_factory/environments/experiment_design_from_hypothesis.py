"""
ExperimentDesignFromHypothesis: Design experiments to test research hypotheses.

Environment concept:
  The model is given a research hypothesis (drawn from real 2026 arxiv papers)
  and must design a complete experiment to test it: baselines, metrics, controls,
  ablations, and predicted outcomes. This trains experimental reasoning — the
  ability to think like a researcher who must design rigorous, falsifiable
  experiments that isolate the variable of interest.

  Why: knowing WHAT to test is different from knowing HOW to test it. A real
  researcher must choose baselines that isolate the contribution, metrics that
  capture the relevant signal, controls that rule out confounds, and ablations
  that reveal which components matter. This environment trains that skill using
  real hypotheses from 2026 AI research.

  All reasoning traces are manually authored — no scripts generate them.

Verification:
  - Baselines identified correctly
  - Metrics match ground truth
  - Controls and ablations identified
  - Experimental reasoning quality

Reward design:
  baseline_match * 0.3 + metric_match * 0.25 + control_match * 0.2 + reasoning_quality * 0.25
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — real 2026 research hypotheses with manually authored experiment designs
# ---------------------------------------------------------------------------

_EXPERIMENT_DESIGN_PROBLEMS = [
    {
        "paper_title": "EP-GRPO: Entropy-Progress Aligned GRPO with Implicit Process Guidance",
        "arxiv_id": "2605.04960",
        "domain": "RL training methods",
        "hypothesis": (
            "GRPO's credit assignment fails because it treats all tokens uniformly, "
            "but high-entropy decision points are more informative than low-entropy "
            "deterministic tokens. If we gate advantage modulation by token entropy "
            "and extract implicit process signals from policy divergence, we can "
            "provide dense token-level feedback without external reward models."
        ),
        "baselines": [
            "Standard GRPO with uniform token-level advantage",
            "GRPO with external process reward model (PRM)",
            "GRPO with learned critic for value estimation",
            "REINFORCE++ with group normalization",
        ],
        "metrics": [
            "Accuracy on mathematical reasoning benchmarks (AIME, MATH, GSM8K)",
            "Training efficiency (tokens to convergence)",
            "Token-level credit assignment quality (correlation with true step importance)",
            "Zero-variance collapse rate during training",
            "Entropy distribution across training",
        ],
        "controls": [
            "Same base model and training data across all methods",
            "Same reward function (verifiable rewards only)",
            "Same group size for fair comparison",
            "Same token budget per problem",
        ],
        "ablations": [
            "Remove entropy gating (uniform modulation) — isolates entropy contribution",
            "Remove implicit process signals (outcome-only) — isolates process guidance",
            "Remove cumulative entropy mapping (standard normalization) — isolates progress alignment",
            "Vary the entropy threshold for gating — sensitivity analysis",
        ],
        "reasoning_trace": (
            "To test whether entropy-gated modulation improves credit assignment, we "
            "need to isolate the entropy signal from other improvements. The key "
            "baseline is standard GRPO — same algorithm, same rewards, same model — "
            "but with uniform token advantages. This isolates the variable: if "
            "entropy-gated modulation helps, it must be because prioritizing "
            "high-entropy tokens improves learning. We also need an external PRM "
            "baseline to show we match learned process supervision without the cost. "
            "The metrics must capture both outcome (accuracy) and process (credit "
            "assignment quality). The zero-variance collapse rate is critical because "
            "it's a known failure mode of GRPO that the method claims to address. "
            "The ablations decompose the three contributions: entropy gating, implicit "
            "process signals, and progress-aligned normalization. Each ablation removes "
            "one component to measure its individual contribution. The controls ensure "
            "that differences come from the algorithm, not from confounds like different "
            "data or model size."
        ),
        "baseline_keywords": ["GRPO", "PRM", "critic", "REINFORCE", "uniform"],
        "metric_keywords": ["accuracy", "efficiency", "credit", "variance", "entropy"],
        "control_keywords": ["same", "model", "data", "reward", "group", "budget"],
    },
    {
        "paper_title": "CLEAR: Continuous Latent Adapter Routing for Safety Alignment",
        "arxiv_id": "2608.21278",
        "domain": "AI safety",
        "hypothesis": (
            "Global safety tuning degrades utility because it modifies the entire model. "
            "If we use a conditional gate that activates a safety LoRA adapter only on "
            "harmful inputs, we can preserve utility on benign prompts while maintaining "
            "safety on harmful ones."
        ),
        "baselines": [
            "Standard SFT safety fine-tuning (global)",
            "Standard LoRA safety adapter (always-on, no gating)",
            "DPO-based safety alignment",
            "RLHF with safety reward model",
            "No safety intervention (base model upper bound for utility)",
        ],
        "metrics": [
            "Attack Success Rate (ASR) on HarmBench",
            "Utility preservation (GSM8K accuracy, MT-Bench score)",
            "False refusal rate on benign prompts",
            "Gate activation distribution (what fraction of inputs trigger safety)",
            "Latency overhead from gating mechanism",
        ],
        "controls": [
            "Same base model (Llama-3-8B-Instruct) across all methods",
            "Same safety training data",
            "Same evaluation suite (HarmBench + GSM8K + MT-Bench)",
            "Same inference hardware for latency comparison",
        ],
        "ablations": [
            "Binary gate instead of continuous (tests if continuous control matters)",
            "Gate on different hidden layers (tests which layer captures safety relevance)",
            "Gate without LoRA (tests if gating alone without adapter helps)",
            "LoRA without gate (tests if always-on adapter is the problem)",
            "Vary gate threshold (sensitivity to activation sensitivity)",
        ],
        "reasoning_trace": (
            "The hypothesis has two parts: (1) conditional intervention preserves "
            "utility, and (2) a continuous gate is better than binary. To test (1), "
            "we compare against always-on LoRA — if the gate helps, it must be because "
            "selective activation preserves the backbone's native capabilities. The "
            "no-intervention baseline establishes the utility ceiling. To test (2), "
            "the binary gate ablation shows whether continuous control provides value "
            "beyond simple on/off. The metrics must capture the safety-utility tradeoff "
            "explicitly: ASR for safety, GSM8K/MT-Bench for utility, and false refusal "
            "for the cost of being too cautious. The gate activation distribution is "
            "a diagnostic metric — if the gate activates on everything, it's no better "
            "than global tuning; if it never activates, safety fails. The layer "
            "ablation tests where safety-relevant information lives in the model. "
            "The LoRA-without-gate ablation is the most important: it directly tests "
            "the hypothesis that gating (not just having an adapter) is what preserves "
            "utility."
        ),
        "baseline_keywords": ["SFT", "LoRA", "DPO", "RLHF", "base", "always-on"],
        "metric_keywords": ["ASR", "utility", "GSM8K", "refusal", "gate", "latency"],
        "control_keywords": ["same", "model", "data", "evaluation", "hardware"],
    },
    {
        "paper_title": "FORGE: Fused On-Register Gradient Elimination",
        "arxiv_id": "2606.22932",
        "domain": "Efficient training",
        "hypothesis": (
            "Gradients are transient artifacts consumed exactly once by the optimizer. "
            "If we fuse gradient computation with optimizer consumption in registers, "
            "we can eliminate the gradient memory pool entirely without approximation."
        ),
        "baselines": [
            "Standard AdamW with materialized gradients",
            "Gradient checkpointing (recomputation)",
            "8-bit optimizer (bitsandbytes)",
            "CPU offloading of optimizer states",
            "Low-rank gradient projection (GaLore)",
        ],
        "metrics": [
            "Peak GPU memory during training",
            "Training throughput (tokens/second)",
            "Training stability (loss curve smoothness)",
            "Final model quality (perplexity/accuracy)",
            "Memory breakdown by component (gradients, optimizer states, activations)",
        ],
        "controls": [
            "Same model architecture and size",
            "Same training data and hyperparameters",
            "Same batch size and sequence length",
            "Same hardware (GPU type, memory)",
            "Same optimizer (AdamW) for fair comparison",
        ],
        "ablations": [
            "FORGE with fp32 optimizer states vs int8 (tests state quantization contribution)",
            "FORGE on different architectures (transformer, SSM, MLP mixer) — tests architecture-agnostic claim",
            "FORGE combined with gradient checkpointing (tests composability)",
            "FORGE with different optimizer types (AdamW, Muon, Lion) — tests optimizer-agnostic claim",
            "Partial FORGE (only some layers) — tests if benefit requires full application",
        ],
        "reasoning_trace": (
            "The hypothesis is that gradients don't need to be stored because they're "
            "consumed immediately. The critical baseline is standard AdamW — this "
            "establishes the memory floor with materialized gradients. Gradient "
            "checkpointing is a different approach to the same problem (memory) but "
            "trades compute for memory, while FORGE claims to save both. The 8-bit "
            "optimizer baseline tests whether quantizing the gradient (a compression "
            "approach) is worse than eliminating it entirely. The exactness proof is "
            "key: we need to verify that FORGE produces identical results to standard "
            "training, not just similar results. The metrics must separate memory "
            "(the primary claim) from speed (secondary benefit) and quality (must not "
            "degrade). The memory breakdown is essential — if gradients were only 10% "
            "of memory, eliminating them wouldn't matter much. The architecture "
            "ablation tests the generality claim: if FORGE only works on transformers, "
            "the insight is less general. The composability ablation with gradient "
            "checkpointing is important because real training pipelines combine "
            "multiple memory-saving techniques — if FORGE conflicts with checkpointing, "
            "it's less practical."
        ),
        "baseline_keywords": ["AdamW", "checkpointing", "8-bit", "offloading", "GaLore"],
        "metric_keywords": ["memory", "throughput", "stability", "quality", "breakdown"],
        "control_keywords": ["same", "architecture", "data", "hyperparameters", "hardware", "optimizer"],
    },
    {
        "paper_title": "SRPO: Self-Reflective Policy Optimization",
        "arxiv_id": "2608.23493",
        "domain": "Agentic AI",
        "hypothesis": (
            "Self-reflection can serve as a dense reward signal. If the model analyzes "
            "its own failed trajectories, synthesizes error patterns into reflection "
            "patches, and uses these to score new rollouts, we can convert sparse "
            "terminal rewards into dense token-level supervision without external critics."
        ),
        "baselines": [
            "Standard GRPO with outcome-only rewards",
            "GRPO with external process reward model",
            "Scaled SFT on high-quality reasoning traces",
            "GRPO with learned critic (PPO-style)",
            "Self-consistency voting (inference-time only)",
        ],
        "metrics": [
            "Accuracy on AIME'24, WebShop, ALFWorld, SWE-Bench-Lite",
            "Training FLOPs to reach target accuracy",
            "Token-level credit assignment quality",
            "Reflection patch quality (does it identify the actual error?)",
            "Generalization across task types (math vs agentic)",
        ],
        "controls": [
            "Same base model (Qwen3-8B)",
            "Same training data distribution",
            "Same reward function for outcome verification",
            "Same compute budget for fair FLOPs comparison",
            "Same evaluation benchmarks",
        ],
        "ablations": [
            "Remove reset-with-memory (inject reflection into context) — tests if clean reset matters",
            "Remove reflection patch (use raw error description) — tests if synthesis matters",
            "Use external critic instead of self-reflection — tests if self-generated signals suffice",
            "Vary reflection patch length — tests information density vs noise tradeoff",
            "Apply only to math (not agentic) — tests domain generality",
        ],
        "reasoning_trace": (
            "The hypothesis claims self-reflection can replace external supervision for "
            "credit assignment. The critical test is against external PRM — if self-"
            "reflection matches PRM quality, the cost savings are enormous. The scaled "
            "SFT baseline establishes the compute-intensive upper bound: if SRPO matches "
            "SFT quality at 0.08x FLOPs, the efficiency claim is validated. The reset-"
            "with-memory ablation is the most important: it tests whether the mechanism "
            "of injecting guidance (clean reset vs context injection) matters. If context "
            "injection works just as well, the reset-with-memory design is unnecessary. "
            "The self-vs-external critic ablation directly tests the core claim: can the "
            "model serve as its own teacher? If the external critic is significantly "
            "better, self-reflection isn't sufficient. The metrics span both reasoning "
            "(AIME) and agentic (WebShop, ALFWorld, SWE-Bench) tasks to test generality. "
            "The FLOPs metric is essential because the paper's key claim is efficiency — "
            "if SRPO uses the same FLOPs as SFT, the contribution is just the credit "
            "assignment, not the efficiency."
        ),
        "baseline_keywords": ["GRPO", "PRM", "SFT", "critic", "PPO", "self-consistency"],
        "metric_keywords": ["accuracy", "FLOPs", "credit", "reflection", "generalization"],
        "control_keywords": ["same", "model", "data", "reward", "compute", "benchmarks"],
    },
    {
        "paper_title": "Mechanistic Tomography: Designed Measurement for Interpretability",
        "arxiv_id": "2608.19338",
        "domain": "AI interpretability",
        "hypothesis": (
            "Activation patching, gradients, and Hessian-vector products are all solving "
            "the same underlying measurement problem. If we formalize this as y = Ax + w "
            "and design measurements optimally, we can recover effect maps with fewer "
            "interventions and theoretical guarantees on sufficiency."
        ),
        "baselines": [
            "Standard activation patching (one component at a time)",
            "Integrated gradients",
            "Sparse autoencoders (SAE) for feature discovery",
            "Attention pattern analysis",
            "Random intervention baseline (lower bound)",
        ],
        "metrics": [
            "Held-out R² of recovered effect map",
            "Number of interventions needed for sufficient recovery",
            "Ability to identify known circuits (IOI, induction)",
            "Recovery of interaction effects (pair-wise)",
            "Computational cost of measurement design",
        ],
        "controls": [
            "Same model (GPT-2-small, Qwen-2.5-7B)",
            "Same tasks (IOI, refusal)",
            "Same access level (forward-only vs gradient access)",
            "Known ground truth on Tracr (compiled model with known circuits)",
        ],
        "ablations": [
            "Random measurements vs designed measurements — tests if design matters",
            "First-order only (no Hessian) — tests if interactions are important",
            "Sparse vs dense aggregate measurements — tests sparsity contribution",
            "Different measurement budgets — tests how recovery scales with interventions",
            "Gradient-based vs forward-only — tests value of gradient access",
        ],
        "reasoning_trace": (
            "The hypothesis unifies multiple interpretability methods under a measurement "
            "framework. The critical test is whether DESIGNED measurements outperform "
            "RANDOM ones — if not, the framework adds no value over standard patching. "
            "The Tracr control is essential: Tracr models have known circuits, so we "
            "can verify the recovered map is correct, not just internally consistent. "
            "The R² metric on held-out data tests generalization: does the recovered "
            "map predict the effect of interventions NOT used in recovery? The "
            "interaction ablation (first-order only) tests whether pair interactions "
            "matter — if they do, standard patching misses important structure. The "
            "intervention count metric is the practical contribution: if designed "
            "measurements need 10x fewer interventions, the method is practical for "
            "large models where exhaustive patching is infeasible. The access-level "
            "control (forward-only vs gradient) tests whether the framework works in "
            "restricted settings where gradients aren't available."
        ),
        "baseline_keywords": ["patching", "gradients", "SAE", "attention", "random"],
        "metric_keywords": ["R²", "interventions", "circuits", "interaction", "cost"],
        "control_keywords": ["same", "model", "task", "access", "Tracr", "ground-truth"],
    },
    {
        "paper_title": "Actor-Curator: Co-adaptive Curriculum Learning",
        "arxiv_id": "2602.20532",
        "domain": "RL training methods",
        "hypothesis": (
            "Curriculum learning should directly optimize for policy improvement, not "
            "proxy objectives like difficulty scores. If we formulate problem selection "
            "as a non-stationary bandit and train a neural curator with regret "
            "guarantees, we can achieve faster convergence than heuristic curricula."
        ),
        "baselines": [
            "Uniform sampling (no curriculum)",
            "Difficulty-based curriculum (easy-to-hard)",
            "Uncertainty-based sampling (focus on uncertain problems)",
            "Self-paced learning (model selects own difficulty)",
            "Random curriculum (control for selection effect)",
        ],
        "metrics": [
            "Accuracy on AIME2024, ARC-1D",
            "Training steps to reach target accuracy (speedup)",
            "Regret of the curator (theoretical and empirical)",
            "Curriculum diversity (are different problems selected over time?)",
            "Policy improvement per training step",
        ],
        "controls": [
            "Same base model across all curriculum methods",
            "Same problem bank (same set of available problems)",
            "Same RL algorithm (GRPO) and hyperparameters",
            "Same total training compute",
            "Same evaluation benchmarks",
        ],
        "ablations": [
            "Remove non-stationarity handling (stationary bandit) — tests if adaptation matters",
            "Remove neural curator (random arm selection within bandit) — tests if learned selection helps",
            "Use proxy objective (difficulty score) instead of direct improvement — tests the core claim",
            "Vary the bandit exploration parameter — sensitivity analysis",
            "Fixed curriculum vs adaptive — tests value of adaptation",
        ],
        "reasoning_trace": (
            "The core claim is that directly optimizing for policy improvement beats "
            "proxy objectives. The critical ablation is replacing direct improvement "
            "with difficulty scores — if this ablation performs as well, the direct "
            "optimization isn't necessary. The uniform sampling baseline establishes "
            "the no-curriculum floor. The difficulty-based and uncertainty-based "
            "baselines are the standard alternatives — if Actor-Curator doesn't beat "
            "these, the contribution is marginal. The non-stationarity ablation is "
            "key: as the policy improves, the optimal curriculum changes. If a "
            "stationary bandit works as well, the non-stationarity handling is "
            "unnecessary. The regret metric connects theory to practice: the paper "
            "proves regret guarantees, and we need to verify they hold empirically. "
            "The curriculum diversity metric prevents a degenerate solution where "
            "the curator selects the same few problems. The speedup metric (28.6% "
            "on AIME, 30.5% on ARC-1D) is the headline result — it must be measured "
            "under identical compute budgets to be meaningful."
        ),
        "baseline_keywords": ["uniform", "difficulty", "uncertainty", "self-paced", "random"],
        "metric_keywords": ["accuracy", "speedup", "regret", "diversity", "improvement"],
        "control_keywords": ["same", "model", "problem", "RL", "compute", "benchmarks"],
    },
    {
        "paper_title": "Learning When to Think: Adaptive Reasoning",
        "arxiv_id": "2608.20256",
        "domain": "LLM reasoning",
        "hypothesis": (
            "A model can learn to allocate reasoning effort adaptively by choosing a "
            "mode (NoThink, Short, Long) as its first output token. Shaped rewards and "
            "hard token caps can prevent mode collapse while enabling 40%+ token reduction."
        ),
        "baselines": [
            "Fixed Long reasoning (always think fully)",
            "Fixed Short reasoning (always brief)",
            "External router model (separate model decides mode)",
            "Random mode selection (control for selection effect)",
            "NoThink only (always answer immediately)",
        ],
        "metrics": [
            "Accuracy on MATH500, GSM8K (in-distribution and transfer)",
            "Mean response length (token efficiency)",
            "Mode distribution (are all three modes used?)",
            "Mode accuracy correlation (do harder problems get more thinking?)",
            "Mode collapse rate (does one mode dominate?)",
        ],
        "controls": [
            "Same base model (1.5B distilled)",
            "Same training data (MATH)",
            "Same RL algorithm (GRPO)",
            "Same total token budget for evaluation",
            "Same evaluation benchmarks",
        ],
        "ablations": [
            "Remove shaped reward (uniform reward across modes) — tests if shaping prevents collapse",
            "Remove hard token caps (soft caps only) — tests if hard boundaries matter",
            "Two modes instead of three (Short + Long only) — tests if NoThink adds value",
            "External router instead of first-token — tests if integrated routing is necessary",
            "Vary temperature for mode selection — sensitivity analysis",
        ],
        "reasoning_trace": (
            "The hypothesis has three components: (1) first-token routing works, (2) "
            "shaped reward prevents collapse, (3) hard caps keep modes distinct. Each "
            "needs its own test. The external router baseline tests (1): if a separate "
            "router works as well, the integrated approach isn't necessary (though it's "
            "simpler). The shaped reward ablation tests (2): without shaping, does the "
            "model collapse to one mode? The hard cap ablation tests (3): do modes "
            "blur without strict boundaries? The mode distribution metric is the key "
            "diagnostic — if one mode is used 95% of the time, the system has collapsed. "
            "The mode-accuracy correlation tests whether the router actually sorts by "
            "difficulty: if easy problems get Long mode, the routing isn't working. "
            "The transfer to GSM8K is critical: if the adaptive reasoning skill is "
            "MATH-specific, it hasn't learned a general 'when to think' heuristic. "
            "The two-mode ablation tests whether NoThink (immediate answer) adds value "
            "or if Short is sufficient for easy problems."
        ),
        "baseline_keywords": ["fixed", "external", "router", "random", "NoThink"],
        "metric_keywords": ["accuracy", "length", "mode", "distribution", "collapse", "transfer"],
        "control_keywords": ["same", "model", "data", "GRPO", "budget", "benchmarks"],
    },
    {
        "paper_title": "Simulator Collapse in Multi-Agent RL",
        "arxiv_id": "2608.12253",
        "domain": "Agentic AI",
        "hypothesis": (
            "Single frozen LLM simulators cause mode collapse in multi-agent RL because "
            "the policy overfits to the simulator's dominant mode. Diverse simulators "
            "(verbalized sampling or co-training) should prevent this and improve "
            "generalization to real users."
        ),
        "baselines": [
            "Single frozen simulator (standard practice)",
            "Multiple frozen simulators (different models)",
            "Human evaluation (gold standard)",
            "No simulator (self-play only)",
            "Rule-based simulator (non-LLM)",
        ],
        "metrics": [
            "Success rate on held-out simulators (generalization)",
            "Success rate with real human users",
            "Policy diversity (strategy distribution breadth)",
            "Simulator mode collapse metrics (entropy of simulator outputs)",
            "Training stability (does policy converge with diverse simulators?)",
        ],
        "controls": [
            "Same policy model and architecture",
            "Same task (Persuasion for Good, τ²-bench, CooperBench)",
            "Same RL algorithm",
            "Same total training steps",
            "Same evaluation protocol for human studies",
        ],
        "ablations": [
            "Verbalized sampling only (inference-time) — tests inference solution",
            "Co-training only (training-time) — tests training solution",
            "Both combined — tests if benefits are additive",
            "Vary number of co-trained simulators — tests scaling with diversity",
            "Vary verbalized sampling temperature — tests diversity vs coherence tradeoff",
        ],
        "reasoning_trace": (
            "The hypothesis is that simulator diversity prevents policy overfitting. "
            "The critical test is generalization to HELD-OUT simulators and real humans "
            "— if the policy only works on the training simulator, it has overfit. The "
            "single frozen simulator baseline is the standard practice and should show "
            "the collapse problem. The multiple frozen simulators baseline tests whether "
            "just using different models helps (a simpler solution than co-training). "
            "The human evaluation is the ultimate test — if gains don't transfer to "
            "real users, the contribution is academic. The policy diversity metric is "
            "key: if the policy uses diverse strategies, it's less likely to have "
            "overfit to one simulator mode. The verbalized-sampling-only vs co-training-"
            "only ablation separates the two solutions: verbalized sampling is cheaper "
            "(inference-time only) but might be less effective than co-training (which "
            "changes the training distribution). The combined test checks if they're "
            "additive or redundant. The scaling with number of simulators tests whether "
            "more diversity always helps or if there are diminishing returns."
        ),
        "baseline_keywords": ["frozen", "multiple", "human", "self-play", "rule-based"],
        "metric_keywords": ["success", "generalization", "diversity", "collapse", "stability"],
        "control_keywords": ["same", "policy", "task", "RL", "steps", "protocol"],
    },
    {
        "paper_title": "Spectral Compact Training via Permanent Truncated SVD",
        "arxiv_id": "2604.00733",
        "domain": "Efficient training",
        "hypothesis": (
            "Weight matrices have low-rank structure during training, not just after. "
            "If we permanently store weights as SVD factors and retract to the Stiefel "
            "manifold after each step, we can train without ever materializing the dense "
            "matrix, achieving 100x+ memory reduction."
        ),
        "baselines": [
            "Standard dense training (AdamW)",
            "LoRA fine-tuning (low-rank adaptation)",
            "Gradient checkpointing",
            "8-bit optimizer",
            "Post-hoc SVD compression (train dense, compress after)",
        ],
        "metrics": [
            "Peak GPU memory during training",
            "Training throughput (tokens/second)",
            "Final perplexity (model quality)",
            "Loss curve across ranks (do all ranks converge?)",
            "Orthogonality drift (how fast U, V drift without retraction)",
        ],
        "controls": [
            "Same model architecture (SmolLM2-1.7B)",
            "Same training data",
            "Same training steps",
            "Same learning rate schedule",
            "Same evaluation benchmarks",
        ],
        "ablations": [
            "Remove Stiefel retraction (let U, V drift) — tests if orthogonality matters",
            "Post-hoc SVD (train dense, compress) — tests if permanent factoring is necessary",
            "Different retraction methods (QR vs Cayley vs projection) — tests retraction choice",
            "Vary rank (32, 64, 128, 256) — tests rank vs quality tradeoff",
            "Different layer types (attention vs MLP) — tests where low-rank helps most",
        ],
        "reasoning_trace": (
            "The hypothesis challenges the standard approach: train dense, compress "
            "after. The critical test is whether PERMANENT factoring (never materialize "
            "dense) matches post-hoc compression in quality. If post-hoc is just as "
            "good, the permanent approach only saves memory during training, not "
            "inference. The Stiefel retraction ablation is the most important: without "
            "it, U and V drift from orthogonality, breaking the spectral properties. "
            "If training still works without retraction, the orthogonality constraint "
            "is unnecessary. The rank sweep tests the core assumption: if all ranks "
            "converge to the same loss, the weight matrices are indeed low-rank during "
            "training. If higher ranks are needed, the low-rank assumption is weaker "
            "than expected. The orthogonality drift metric is diagnostic: it tells us "
            "how fast the factors degrade, which determines how often retraction is "
            "needed. The layer-type ablation tests whether different layers have "
            "different rank requirements — attention matrices might be inherently "
            "lower rank than MLP matrices. The LoRA baseline is important because LoRA "
            "also uses low-rank but only for adaptation, not full training."
        ),
        "baseline_keywords": ["dense", "LoRA", "checkpointing", "8-bit", "post-hoc"],
        "metric_keywords": ["memory", "throughput", "perplexity", "loss", "orthogonality"],
        "control_keywords": ["same", "architecture", "data", "steps", "schedule", "benchmarks"],
    },
    {
        "paper_title": "Subspace-Aware Sparse Autoencoders for Mechanistic Interpretability",
        "arxiv_id": "2606.06333",
        "domain": "AI interpretability",
        "hypothesis": (
            "Standard SAEs' single-direction decoder assumption causes feature splitting "
            "because model features are multi-dimensional. If we use learned decoder "
            "subspaces with block sparsity, we can consolidate fragmented features and "
            "recover coherent multi-dimensional representations."
        ),
        "baselines": [
            "Standard SAE (single-direction decoder)",
            "TopK SAE",
            "JumpReLU SAE",
            "BatchTopK SAE",
            "PCA (no learned dictionary)",
        ],
        "metrics": [
            "Feature coherence (do recovered features correspond to interpretable concepts?)",
            "Feature splitting rate (how many near-collinear latents per concept?)",
            "Reconstruction loss (MSE between original and reconstructed activations)",
            "Sparsity (L0 norm of latent activations)",
            "Downstream task performance (does SASA help interpretability tasks?)",
        ],
        "controls": [
            "Same model (GPT-2 small, Gemma-2-2B, Qwen3-8B-Base)",
            "Same activation layer for probing",
            "Same training data for SAE",
            "Same dictionary size (fair comparison)",
            "Same evaluation protocol",
        ],
        "ablations": [
            "Vary block size r (tests minimum r for consolidation)",
            "Remove nuclear-norm regularizer (tests adaptive rank contribution)",
            "Remove Top-s group gating (tests block sparsity contribution)",
            "Single-direction decoder with same capacity (controls for parameter count)",
            "Different numbers of groups K (tests if more groups help)",
        ],
        "reasoning_trace": (
            "The hypothesis is that multi-dimensional features cause splitting in "
            "standard SAEs. The critical test is whether SASA consolidates features "
            "that standard SAEs split — this requires a metric for 'feature splitting' "
            "(near-collinear latents per concept). The single-direction decoder with "
            "same capacity ablation is the key control: if SASA wins just because it "
            "has more parameters, the contribution is capacity, not structure. The "
            "block size ablation tests the theoretical prediction: once r >= d_i "
            "(intrinsic dimension), a single group should be optimal. If performance "
            "keeps improving beyond r = d_i, the theory is wrong. The nuclear-norm "
            "regularizer ablation tests whether adaptive rank matters or if fixed rank "
            "suffices. The reconstruction loss is a sanity check — SASA shouldn't "
            "sacrifice reconstruction for interpretability. The feature coherence "
            "metric is the hardest to define but the most important: does SASA recover "
            "features that humans find meaningful? This requires manual evaluation or "
            "probe-based metrics."
        ),
        "baseline_keywords": ["SAE", "TopK", "JumpReLU", "BatchTopK", "PCA"],
        "metric_keywords": ["coherence", "splitting", "reconstruction", "sparsity", "downstream"],
        "control_keywords": ["same", "model", "layer", "data", "dictionary", "protocol"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def experiment_design_from_hypothesis_generator(seed: int) -> Problem:
    """Generate an ExperimentDesignFromHypothesis problem."""
    rng = random.Random(seed)
    paper = rng.choice(_EXPERIMENT_DESIGN_PROBLEMS)

    prompt = (
        f"## Experiment Design from Hypothesis\n\n"
        f"### Source Paper: {paper['paper_title']}\n"
        f"**Domain:** {paper['domain']}\n"
        f"**arXiv:** {paper['arxiv_id']}\n\n"
        f"### Research Hypothesis\n{paper['hypothesis']}\n\n"
        f"### Your Task\n"
        f"You are a senior ML researcher designing an experiment to test this "
        f"hypothesis. Design a rigorous, falsifiable experiment with:\n\n"
        f"1. **BASELINES**: What methods should you compare against? Why each?\n"
        f"2. **METRICS**: What should you measure? What signal does each metric capture?\n"
        f"3. **CONTROLS**: What must be held constant across conditions?\n"
        f"4. **ABLATIONS**: What components should be removed individually to isolate contributions?\n"
        f"5. **PREDICTED_OUTCOMES**: What result would confirm vs refute the hypothesis?\n\n"
        f"Format your answer as:\n"
        f"BASELINES:\n- <baseline 1>: <why>\n- <baseline 2>: <why>\n...\n\n"
        f"METRICS:\n- <metric 1>: <signal captured>\n...\n\n"
        f"CONTROLS:\n- <control 1>\n...\n\n"
        f"ABLATIONS:\n- <ablation 1>: <what it isolates>\n...\n\n"
        f"PREDICTED_OUTCOMES:\n- If hypothesis is true: <expected result>\n- If hypothesis is false: <expected result>"
    )

    difficulty = 0.75 + len(paper["baselines"]) * 0.01
    difficulty = min(0.95, difficulty)

    return Problem(
        id=f"experiment_design_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "paper_title": paper["paper_title"],
            "arxiv_id": paper["arxiv_id"],
            "domain": paper["domain"],
            "hypothesis": paper["hypothesis"],
            "baselines": paper["baselines"],
            "metrics": paper["metrics"],
            "controls": paper["controls"],
            "ablations": paper["ablations"],
            "reasoning_trace": paper["reasoning_trace"],
            "baseline_keywords": paper["baseline_keywords"],
            "metric_keywords": paper["metric_keywords"],
            "control_keywords": paper["control_keywords"],
        },
        token_budget=4096,
        source="experiment_design_from_hypothesis_generator",
    )


experiment_design_from_hypothesis_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ExperimentDesignVerifier(Verifier):
    """Verify an experiment design.

    Checks:
    1. Baselines identified (keyword matching + count)
    2. Metrics match ground truth (keyword matching)
    3. Controls identified (keyword matching)
    4. Ablations present (structural check)
    5. Reasoning quality (causal language, depth)

    reward = baseline_match * 0.3 + metric_match * 0.25 + control_match * 0.2 + reasoning_quality * 0.25
    """

    def __init__(
        self,
        baselines: list[str],
        metrics: list[str],
        controls: list[str],
        baseline_keywords: list[str],
        metric_keywords: list[str],
        control_keywords: list[str],
        reasoning_trace: str,
    ):
        super().__init__()
        self._baselines = baselines
        self._metrics = metrics
        self._controls = controls
        self._baseline_keywords = [k.lower() for k in baseline_keywords]
        self._metric_keywords = [k.lower() for k in metric_keywords]
        self._control_keywords = [k.lower() for k in control_keywords]
        self._reasoning_trace = reasoning_trace.lower()

    def verify(self, response: str) -> VerifierResult:
        response_lower = response.lower()

        # --- Baselines ---
        baseline_section = self._extract_section(response, "BASELINES")
        baseline_text = baseline_section.lower() if baseline_section else response_lower
        baseline_items = len(re.findall(r"^[-*]\s", baseline_section, re.MULTILINE))
        baseline_kw_found = sum(1 for kw in self._baseline_keywords if kw in baseline_text)
        baseline_match = (
            (baseline_kw_found / len(self._baseline_keywords) if self._baseline_keywords else 0.0) * 0.6
            + min(1.0, baseline_items / max(1, len(self._baselines))) * 0.4
        )

        # --- Metrics ---
        metric_section = self._extract_section(response, "METRICS")
        metric_text = metric_section.lower() if metric_section else response_lower
        metric_kw_found = sum(1 for kw in self._metric_keywords if kw in metric_text)
        metric_match = metric_kw_found / len(self._metric_keywords) if self._metric_keywords else 0.0

        # --- Controls ---
        control_section = self._extract_section(response, "CONTROLS")
        control_text = control_section.lower() if control_section else response_lower
        control_kw_found = sum(1 for kw in self._control_keywords if kw in control_text)
        control_match = control_kw_found / len(self._control_keywords) if self._control_keywords else 0.0

        # --- Ablations ---
        ablation_section = self._extract_section(response, "ABLATIONS")
        ablation_items = len(re.findall(r"^[-*]\s", ablation_section, re.MULTILINE)) if ablation_section else 0
        ablation_score = min(1.0, ablation_items / 4.0)  # expect at least 4 ablations

        # --- Reasoning quality ---
        predicted_section = self._extract_section(response, "PREDICTED_OUTCOMES")
        all_reasoning = (baseline_section + " " + metric_section + " " +
                         control_section + " " + ablation_section + " " +
                         predicted_section).lower()

        # Causal reasoning indicators
        causal_indicators = [
            "because", "to test", "isolates", "if", "then", "confound",
            "control for", "rule out", "falsif", "confirm", "refute",
            "the key", "critical", "necessary", "sufficient",
        ]
        causal_found = sum(1 for ind in causal_indicators if ind in all_reasoning)
        causal_score = min(1.0, causal_found / 6.0)

        # Falsifiability check
        has_falsification = bool(
            re.search(r"if.*false|refute|falsif|would.*not|fail", all_reasoning)
        )
        has_confirmation = bool(
            re.search(r"if.*true|confirm|expect|predict", all_reasoning)
        )
        falsifiability_score = (1.0 if has_falsification else 0.0) * 0.5 + (1.0 if has_confirmation else 0.0) * 0.5

        # Depth
        total_words = len(all_reasoning.split())
        depth_score = min(1.0, total_words / 150.0)

        reasoning_quality = causal_score * 0.4 + falsifiability_score * 0.3 + depth_score * 0.3

        # --- Final score ---
        score = (
            baseline_match * 0.3
            + metric_match * 0.25
            + control_match * 0.2
            + reasoning_quality * 0.25
        )
        correct = score >= 0.55

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "baseline_match": baseline_match,
                "metric_match": metric_match,
                "control_match": control_match,
                "ablation_score": ablation_score,
                "reasoning_quality": reasoning_quality,
                "baseline_keywords_found": float(baseline_kw_found),
                "metric_keywords_found": float(metric_kw_found),
                "control_keywords_found": float(control_kw_found),
                "causal_indicators_found": float(causal_found),
                "has_falsification": float(has_falsification),
                "has_confirmation": float(has_confirmation),
            },
            diagnostics=(
                f"baseline={baseline_match:.2f} metric={metric_match:.2f} "
                f"control={control_match:.2f} reasoning={reasoning_quality:.2f} "
                f"(ablations={ablation_items} causal={causal_found}/6 "
                f"falsif={has_falsification} confirm={has_confirmation})"
            ),
        )

    def _extract_section(self, response: str, section_name: str) -> str:
        pattern = rf"{section_name}\s*:?\s*\n(.*?)(?=\n[A-Z_]+\s*:?\s*\n|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ExperimentDesignFromHypothesisEnv(BatchEnvBase):
    """ExperimentDesignFromHypothesis: design experiments to test research hypotheses.

    Batch-aware: N parallel attempts; reward = best experiment design.
    Trains experimental reasoning — designing rigorous, falsifiable experiments
    that isolate variables and rule out confounds.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = experiment_design_from_hypothesis_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ExperimentDesignVerifier(
            baselines=problem.metadata["baselines"],
            metrics=problem.metadata["metrics"],
            controls=problem.metadata["controls"],
            baseline_keywords=problem.metadata["baseline_keywords"],
            metric_keywords=problem.metadata["metric_keywords"],
            control_keywords=problem.metadata["control_keywords"],
            reasoning_trace=problem.metadata["reasoning_trace"],
        )

    def _check_format(self, response: str) -> float:
        sections = ["BASELINES", "METRICS", "CONTROLS", "ABLATIONS", "PREDICTED_OUTCOMES"]
        found = sum(1 for s in sections if re.search(rf"{s}\s*:", response, re.IGNORECASE))
        return found / len(sections)

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": best_score,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores)/len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"PREDICTED_OUTCOMES\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
