"""
ResearchCritiqueAndGap: Critique a paper and identify research gaps.

Environment concept:
  The model is given a real 2026 arxiv paper's breakthrough, methodology, and
  results. It must critically evaluate the work: identify limitations, find
  methodological weaknesses, question assumptions, and propose specific research
  gaps that a follow-up paper could address. This trains the critical reasoning
  that distinguishes a good researcher from a paper summarizer.

  Why: reading a paper and accepting it at face value is not research. A real
  researcher asks: what did they NOT test? What assumptions are unexamined?
  Where does this break? What's the next question? This environment trains
  that adversarial critical reasoning using real papers with manually authored
  critiques and gaps.

  All reasoning traces are manually authored — no scripts generate them.

Verification:
  - Limitations identified match ground truth
  - Methodological weaknesses found
  - Research gaps are specific and actionable
  - Critique reasoning is substantive (not surface-level)

Reward design:
  limitation_match * 0.35 + weakness_match * 0.25 + gap_quality * 0.25 + critique_depth * 0.15
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — real 2026 papers with manually authored critiques and gaps
# ---------------------------------------------------------------------------

_CRITIQUE_PROBLEMS = [
    {
        "paper_title": "f-GRPO: Divergence-Based RL for General LLM Alignment",
        "arxiv_id": "2602.05946",
        "domain": "RL training methods",
        "breakthrough": (
            "Unified RLVR and preference alignment under f-divergences, deriving "
            "f-GRPO and f-HAL from variational representations with theoretical "
            "guarantees."
        ),
        "methodology_summary": (
            "Recognized alignment objectives as divergence estimators, extended to "
            "RLVR, derived f-GRPO/f-HAL from f-divergence variational representations, "
            "validated on math reasoning and safety alignment."
        ),
        "limitations": [
            "Only validated on two task types (math reasoning + safety) — generalizability to other domains untested",
            "Theoretical guarantees assume the f-divergence family is correctly specified — model misspecification not analyzed",
            "Computational cost of variational estimation not compared against simpler baselines in wall-clock terms",
            "The choice of which specific f-divergence (KL, chi-squared, etc.) is left as a hyperparameter without principled selection guidance",
            "No analysis of how the framework interacts with online vs offline data mixing strategies",
        ],
        "methodological_weaknesses": [
            "The theoretical guarantees are asymptotic — finite-sample behavior is not characterized",
            "Both validation tasks use relatively simple reward structures — complex multi-objective rewards untested",
            "No comparison against other unification frameworks (e.g., decision-theoretic approaches)",
            "The f-HAL hybrid variant's off-policy correction is not analyzed for staleness sensitivity",
        ],
        "research_gaps": [
            "How to automatically select the optimal f-divergence for a given task without manual tuning",
            "Whether the framework extends to multi-objective RL where rewards conflict",
            "Finite-sample convergence rates for the variational estimators under different f-divergences",
            "Interaction with MoE architectures where routing creates non-standard gradient flow",
            "Whether the divergence perspective can explain reward hacking phenomena mechanistically",
        ],
        "reasoning_trace": (
            "The paper's strength is its unifying theoretical perspective, but this is "
            "also its limitation. By framing everything as divergence estimation, it "
            "assumes the f-divergence family is the right abstraction — but what if the "
            "true objective isn't well-approximated by any f-divergence? The validation "
            "on only two task types is concerning: math reasoning has clean verifiable "
            "rewards, and safety alignment is relatively binary. What about creative "
            "writing, code generation, or multi-turn dialogue where rewards are "
            "nuanced and multi-dimensional? The asymptotic guarantees are elegant but "
            "practitioners care about finite-sample behavior — how many samples do you "
            "need before the variational estimator is reliable? The f-HAL hybrid "
            "variant introduces off-policy data but doesn't analyze how stale the "
            "off-policy data can be before the guarantees break. The most interesting "
            "gap is whether this framework can EXPLAIN reward hacking: if reward "
            "hacking is a divergence minimization phenomenon, the framework might "
            "suggest principled mitigations rather than empirical patches."
        ),
        "limitation_keywords": ["generalizability", "misspecification", "computational", "hyperparameter", "online"],
        "weakness_keywords": ["asymptotic", "finite-sample", "simple", "comparison", "staleness"],
        "gap_keywords": ["select", "multi-objective", "convergence", "MoE", "reward hacking"],
    },
    {
        "paper_title": "Cue-GRPO: Rarity-Aware Credit Redistribution",
        "arxiv_id": "2608.03467",
        "domain": "RL training methods",
        "breakthrough": (
            "Solved multiplicity-induced credit concentration in GRPO by redistributing "
            "advantages according to cluster rarity using deterministic Strategy Cues."
        ),
        "methodology_summary": (
            "Formalized structure-level skew, introduced partition-conditioned rule, "
            "used deterministic Strategy Cues for rollout-local partitions, validated "
            "on AIME with Qwen2.5-Math-7B and Llama-3.1-8B."
        ),
        "limitations": [
            "Strategy Cues are manually defined — the quality of partitioning depends entirely on cue design",
            "Only tested on mathematical reasoning where solution structure is easy to categorize",
            "The 6% overhead claim is for wall-clock time but doesn't account for the engineering cost of designing cues",
            "Rarity-based redistribution assumes rare solutions are more informative — but rare could also mean lucky or noisy",
            "No analysis of how many clusters are optimal — too few loses signal, too many creates noise",
        ],
        "methodological_weaknesses": [
            "The partition quality is not measured independently of downstream task performance",
            "Comparison only against standard GRPO — not against other credit assignment methods (PRM, GraphAE, etc.)",
            "The 'rarity = informativeness' assumption is stated but not empirically validated",
            "Only two base models tested — sensitivity to model architecture unknown",
        ],
        "research_gaps": [
            "Automatic Strategy Cue discovery via clustering or representation learning",
            "Extension to non-math domains where solution structure is harder to categorize (code, dialogue)",
            "Theoretical analysis of optimal cluster count as a function of group size and problem diversity",
            "Distinguishing 'rare because hard' from 'rare because lucky' — incorporating confidence into rarity weighting",
            "Interaction with curriculum learning — should rare solutions be prioritized in curriculum?",
        ],
        "reasoning_trace": (
            "The core assumption — that rare solutions are more informative — is "
            "intuitively appealing but unexamined. A rare solution could be rare because "
            "it requires a creative insight (informative) OR because it happened to avoid "
            "a sampling artifact (noise). The paper doesn't distinguish these cases. "
            "The manual Strategy Cue design is a significant limitation: it works on "
            "math where solutions have clear structural categories (algebraic, geometric, "
            "etc.), but what about code generation where solution structure is a spectrum? "
            "The 6% overhead is impressive but hides the engineering cost — someone had "
            "to design the cues, and bad cues could make things worse. The comparison "
            "only against standard GRPO is weak: the real question is whether rarity-"
            "aware redistribution beats OTHER credit assignment improvements (PRMs, "
            "entropy gating, graph-based methods). The most promising gap is automatic "
            "cue discovery — if you could learn the partitioning from data, the method "
            "would generalize without manual engineering."
        ),
        "limitation_keywords": ["manual", "math", "overhead", "rarity", "cluster"],
        "weakness_keywords": ["partition", "comparison", "assumption", "model"],
        "gap_keywords": ["automatic", "non-math", "optimal", "confidence", "curriculum"],
    },
    {
        "paper_title": "FORGE: Fused On-Register Gradient Elimination",
        "arxiv_id": "2606.22932",
        "domain": "Efficient training",
        "breakthrough": (
            "Eliminated gradient memory pool by fusing gradient computation with "
            "optimizer consumption in registers, 16-33% memory reduction, 1.5x faster."
        ),
        "methodology_summary": (
            "Observed gradients are transient, fused computation with optimizer "
            "consumption, proved exactness, made architecture-agnostic, validated "
            "on Llama-3.1-8B and 32B with Muon."
        ),
        "limitations": [
            "The exactness guarantee only holds when the optimizer's state update is the ONLY reader of gradients — optimizers with gradient-dependent scheduling break this",
            "Register pressure from holding gradients in fp32 registers may cause spills on smaller GPU architectures",
            "The fusion requires custom CUDA kernels — not a drop-in replacement for standard training loops",
            "Only tested with AdamW and Muon — behavior with second-order optimizers (L-BFGS, K-FAC) unknown",
            "The 1.5x speedup is measured on specific hardware — cache effects may differ across GPU generations",
        ],
        "methodological_weaknesses": [
            "No analysis of register pressure on consumer GPUs (RTX 3090, 4090) vs datacenter GPUs (A100, H100)",
            "The architecture-agnostic claim is tested on only three architectures — edge cases (Mamba, RWKV) untested",
            "Interaction with distributed training (gradient all-reduce) not analyzed — fusion may conflict with communication overlap",
            "Memory savings are reported as percentages but absolute savings on very large models (100B+) not measured",
        ],
        "research_gaps": [
            "Extension to distributed training where gradients must be all-reduced before optimizer step",
            "Analysis of register pressure across GPU architectures and precision modes (fp16, bf16, fp8)",
            "Integration with second-order optimizers where gradient Hessian products are needed",
            "Combining FORGE with pipeline parallelism where gradient scheduling is already complex",
            "Whether the fusion insight extends to activation memory (can activations be consumed before storage?)",
        ],
        "reasoning_trace": (
            "The exactness guarantee is the paper's strongest claim but has a critical "
            "caveat: it only holds when the optimizer is the sole gradient consumer. "
            "Many practical training setups have gradient-dependent components: gradient "
            "clipping, gradient noise scale estimation, EMA tracking. These break the "
            "fusion because they need to read the gradient before the optimizer. The "
            "custom CUDA kernel requirement is a practical barrier — this isn't a "
            "PyTorch flag you can flip, it requires kernel engineering. The distributed "
            "training gap is the most important: in multi-GPU training, gradients must "
            "be all-reduced across devices before the optimizer step. FORGE fuses "
            "computation and consumption locally, but all-reduce requires the full "
            "gradient. This means FORGE may only apply to single-GPU training or "
            "require a completely different distributed strategy. The register pressure "
            "concern is real: fp32 registers are scarce, and holding a full gradient "
            "tile may cause spills that negate the memory savings."
        ),
        "limitation_keywords": ["exactness", "register", "custom", "optimizer", "hardware"],
        "weakness_keywords": ["consumer", "architecture", "distributed", "absolute"],
        "gap_keywords": ["distributed", "register", "second-order", "pipeline", "activation"],
    },
    {
        "paper_title": "CLEAR: Continuous Latent Adapter Routing for Safety",
        "arxiv_id": "2608.21278",
        "domain": "AI safety",
        "breakthrough": (
            "Conditional safety adaptation via learned gate controlling LoRA adapter "
            "activation. ASR 32.3% -> 0.5%, 7.1pp higher GSM8K than global tuning."
        ),
        "methodology_summary": (
            "Froze backbone, added safety LoRA, trained latent gate for continuous "
            "control, input-conditioned routing, validated on Llama-3-8B with "
            "HarmBench and GSM8K."
        ),
        "limitations": [
            "The gate is trained on labeled harmful/benign examples — adversarial inputs that fool the gate bypass safety entirely",
            "Only tested on Llama-3-8B — scaling behavior to 70B+ models unknown",
            "The continuous gate adds inference latency on every forward pass — not measured for real-time applications",
            "Safety and utility are tested on different benchmarks (HarmBench vs GSM8K) — no benchmark tests both simultaneously",
            "The gate could be vulnerable to fine-tuning attacks that shift its activation distribution",
        ],
        "methodological_weaknesses": [
            "No adversarial robustness testing — specifically no test of inputs designed to fool the gate",
            "The 0.5% ASR is measured on HarmBench which may not represent novel attack vectors",
            "Gate interpretability not analyzed — we don't know WHAT the gate is detecting",
            "No comparison against other conditional methods (e.g., input classifiers + routing)",
        ],
        "research_gaps": [
            "Adversarial robustness of the gate mechanism — can attacks bypass the gate without triggering safety?",
            "Scaling laws for conditional safety — does the gate approach work better or worse at larger scales?",
            "Joint benchmarks that evaluate safety AND utility on the same examples",
            "Gate interpretability — what features does the gate use to distinguish harmful from benign?",
            "Fine-tuning attacks on the gate — can adversaries shift gate activations through targeted fine-tuning?",
        ],
        "reasoning_trace": (
            "The conditional approach creates a new attack surface: the gate itself. "
            "If an adversary can craft inputs that don't trigger the gate, they bypass "
            "safety entirely — worse than global tuning where safety is always on. "
            "The paper doesn't test this, which is a critical gap. The 0.5% ASR on "
            "HarmBench is encouraging, but HarmBench is a known benchmark — novel "
            "attack vectors might fool the gate. The scaling question is important: "
            "at 70B+, the safety-utility tradeoff might be different (larger models "
            "might have more separable safety and utility representations). The "
            "separate benchmark issue is subtle: HarmBench tests safety, GSM8K tests "
            "utility, but real users encounter both. An input that's both harmful AND "
            "requires reasoning would test whether the gate can handle the intersection. "
            "The gate interpretability gap is fundamental: if we don't know what the "
            "gate detects, we can't predict its failure modes."
        ),
        "limitation_keywords": ["adversarial", "scaling", "latency", "benchmark", "fine-tuning"],
        "weakness_keywords": ["robustness", "HarmBench", "interpretability", "comparison"],
        "gap_keywords": ["adversarial", "scaling", "joint", "interpretability", "fine-tuning"],
    },
    {
        "paper_title": "SRPO: Self-Reflective Policy Optimization",
        "arxiv_id": "2608.23493",
        "domain": "Agentic AI",
        "breakthrough": (
            "Self-reflection as dense reward signal. 73.3% AIME'24 at 0.08x SFT FLOPs. "
            "64.7% WebShop, 76.8% ALFWorld, 31.2% SWE-Bench-Lite."
        ),
        "methodology_summary": (
            "Model analyzes failed trajectories, synthesizes reflection patches, "
            "reset-with-memory injection, reflection-conditioned teacher scores "
            "rollouts for dense token-level supervision."
        ),
        "limitations": [
            "Self-reflection quality depends on the model's ability to identify its own errors — weaker models may generate poor reflections",
            "The reset-with-memory mechanism assumes the reflection patch is correct — incorrect reflections could mislead training",
            "Only tested on Qwen3-8B — reflection quality at smaller scales (1B, 3B) unknown",
            "The 0.08x FLOPs comparison is against scaled SFT, not against other efficient RL methods",
            "No analysis of when self-reflection fails — what types of errors can the model NOT identify in its own reasoning?",
        ],
        "methodological_weaknesses": [
            "The reflection patch quality is not independently evaluated — we don't know if reflections actually identify the right errors",
            "The comparison against external PRM doesn't control for compute — PRM requires training data and compute too",
            "No analysis of reflection diversity — does the model generate varied reflections or converge to generic 'try harder' patches?",
            "The agentic benchmarks (WebShop, ALFWorld) have relatively structured failure modes — unstructured tasks untested",
        ],
        "research_gaps": [
            "Self-reflection quality scaling laws — at what model size does self-reflection become reliable enough for SRPO?",
            "Error taxonomy — which types of reasoning errors can models self-identify vs which require external supervision?",
            "Reflection patch validation — can we verify reflection correctness before using it as training signal?",
            "Combining self-reflection with external verification for hybrid supervision",
            "Extension to multi-step agentic tasks where errors compound and reflection must trace causal chains",
        ],
        "reasoning_trace": (
            "The self-reflection approach has a bootstrapping problem: the model must "
            "be good enough at identifying errors to generate useful reflections, but "
            "the reflections are what make it better. This creates a minimum capability "
            "threshold — below it, reflections are noise. The paper tests on Qwen3-8B "
            "but doesn't explore where this threshold is. The reset-with-memory "
            "mechanism assumes reflections are correct, but what if the model identifies "
            "the wrong error? Then it trains itself to avoid a non-problem while "
            "ignoring the real issue. This is especially dangerous in agentic tasks "
            "where errors are subtle and multi-causal. The 0.08x FLOPs comparison is "
            "misleading: it's against SCALED SFT (which uses many more tokens), not "
            "against other efficient RL methods. A fair comparison would be against "
            "GRPO with process rewards at similar compute. The most important gap is "
            "the error taxonomy: knowing which errors are self-identifiable would tell "
            "us when SRPO is safe to use and when external supervision is necessary."
        ),
        "limitation_keywords": ["self-reflection", "reset", "scale", "FLOPs", "fail"],
        "weakness_keywords": ["quality", "compute", "diversity", "structured"],
        "gap_keywords": ["scaling", "taxonomy", "validation", "hybrid", "multi-step"],
    },
    {
        "paper_title": "Actor-Curator: Co-adaptive Curriculum Learning",
        "arxiv_id": "2602.20532",
        "domain": "RL training methods",
        "breakthrough": (
            "Neural curator optimizing for policy improvement via bandit formulation. "
            "28.6% gain on AIME2024, 30.5% on ARC-1D, up to 80% speedup."
        ),
        "methodology_summary": (
            "Formulated problem selection as non-stationary bandit, derived loss from "
            "online stochastic mirror descent, regret guarantees, co-adaptive training."
        ),
        "limitations": [
            "The bandit formulation assumes the reward (policy improvement) can be measured after each problem — but improvement is noisy and delayed",
            "The curator is a neural network that must be trained — its own training cost and sample efficiency not analyzed",
            "Only tested on reasoning benchmarks (AIME, ARC) — agentic and creative tasks untested",
            "The non-stationarity handling adds complexity — it's unclear if simpler adaptive methods would work as well",
            "The 80% speedup is measured in training steps, not wall-clock time — curator inference overhead not included",
        ],
        "methodological_weaknesses": [
            "Policy improvement as bandit reward is noisy — single-problem improvement can be negative due to optimization noise",
            "No comparison against simple adaptive baselines (e.g., difficulty banding with online updates)",
            "The regret guarantees assume the bandit is well-specified — but the non-stationarity may violate assumptions",
            "Curator generalization to unseen problems not tested — does it select only from seen problem types?",
        ],
        "research_gaps": [
            "Robust reward estimation for the bandit — averaging over multiple training runs or using Bayesian estimates",
            "Curator generalization to new problem types — can it select useful problems it hasn't seen before?",
            "Extension to multi-modal curricula where different skills need different problem types",
            "Analysis of curator-policy co-adaptation dynamics — does the curator converge or oscillate?",
            "Transfer of curator across models — can a curator trained for one model help a different model?",
        ],
        "reasoning_trace": (
            "The bandit formulation is elegant but the reward signal (policy improvement) "
            "is extremely noisy. After training on one problem, the policy might improve, "
            "stay the same, or even get worse due to optimization noise. The bandit "
            "treats each problem-selection as an arm pull with a scalar reward, but "
            "the true reward is a distribution over outcomes. This noise could cause "
            "the curator to chase lucky improvements and avoid unlucky problems that "
            "are actually useful. The non-stationarity is handled but adds complexity — "
            "a simpler approach like difficulty banding with exponential moving averages "
            "might achieve similar results. The 80% speedup in training steps doesn't "
            "account for curator inference overhead: if the curator adds 20% per-step "
            "overhead, the wall-clock speedup is much less. The most important gap is "
            "curator transfer: if you need to train a new curator for each model, the "
            "total cost might exceed the savings. A universal curator that transfers "
            "across models would be far more practical."
        ),
        "limitation_keywords": ["noisy", "curator", "benchmarks", "complexity", "speedup"],
        "weakness_keywords": ["reward", "comparison", "assumptions", "generalization"],
        "gap_keywords": ["robust", "generalization", "multi-modal", "dynamics", "transfer"],
    },
    {
        "paper_title": "Seirenes: Adversarial Self-Play with Evolving Distractions",
        "arxiv_id": "2605.11636",
        "domain": "LLM reasoning",
        "breakthrough": (
            "Self-play framework where model constructs distracting contexts and solves "
            "problems despite them. +10.2, +9.1, +7.2 points across 7 benchmarks, 4B-30B."
        ),
        "methodology_summary": (
            "Parameter-shared adversarial loop: model constructs distractors exposing "
            "its own blind spots, then solves problems discerning essential from "
            "perturbation. Co-evolutionary curriculum."
        ),
        "limitations": [
            "Parameter sharing means the model can't specialize — the distractor-builder and solver compete for the same parameters",
            "The co-evolutionary curriculum could collapse if the model finds a distractor strategy that always works (or never works)",
            "Only tested on mathematical reasoning — distraction in open-ended generation (creative writing, dialogue) untested",
            "The 4-5 point reduction on GPT/Gemini shows distractors transfer, but it's unclear if this is a feature or a vulnerability",
            "No analysis of what makes a 'good' distractor — the method learns them implicitly without understanding the structure",
        ],
        "methodological_weaknesses": [
            "No comparison against non-adversarial distraction (random noise, permuted context)",
            "The parameter-sharing design choice is not ablated against separate models",
            "Co-evolutionary stability not analyzed — does the arms race converge or oscillate?",
            "The gain varies significantly across benchmarks (+10.2 to +7.2) — no analysis of what predicts the gain",
        ],
        "research_gaps": [
            "Separate parameter analysis — would dedicated distractor and solver models achieve better results?",
            "Co-evolutionary stability analysis — conditions under which the arms race converges vs collapses",
            "Extension to non-math domains — does adversarial distraction help in code, dialogue, creative writing?",
            "Distractor characterization — what properties make a distractor effective for training?",
            "Safety implications — if the model learns to construct effective distractors, could this be misused for manipulation?",
        ],
        "reasoning_trace": (
            "The parameter-sharing design is both the method's elegance and its "
            "limitation. Sharing parameters means the model's understanding of what's "
            "distracting IS its understanding of what's essential — this is philosophically "
            "appealing but practically constraining. A dedicated distractor model could "
            "specialize in finding blind spots without compromising solver capability. "
            "The co-evolutionary stability is a real concern: if the distractor side "
            "gets too good, the solver can't learn (all problems are unsolvable); if "
            "it's too weak, the solver doesn't learn (no challenge). The paper doesn't "
            "analyze this balance. The transfer to GPT/Gemini is interesting but "
            "double-edged: it shows the distractors are genuinely hard, but it also "
            "means the model has learned to construct adversarial inputs — a capability "
            "with misuse potential. The math-only testing is limiting: in math, the "
            "'essential task' is well-defined, but in open-ended generation, what "
            "counts as 'distraction' vs 'context' is subjective."
        ),
        "limitation_keywords": ["parameter", "collapse", "math", "transfer", "distractor"],
        "weakness_keywords": ["comparison", "ablated", "stability", "vary"],
        "gap_keywords": ["separate", "stability", "non-math", "characterization", "safety"],
    },
    {
        "paper_title": "Mechanistic Tomography: Designed Measurement for Interpretability",
        "arxiv_id": "2608.19338",
        "domain": "AI interpretability",
        "breakthrough": (
            "Unified activation patching, gradients, Hessians as measurement problem. "
            "R² = 0.983 on Qwen-2.5-7B refusal surface. Sparse measurements reduce "
            "intervention count."
        ),
        "methodology_summary": (
            "Formalized y = Ax + w, developed sparse aggregate measurements, "
            "gradient-based probes, Hessian-vector products for interactions. "
            "Validated on Tracr, GPT-2, Qwen-2.5-7B."
        ),
        "limitations": [
            "The linear measurement model (y = Ax + w) assumes effects are additive — nonlinear interactions beyond pair-wise are not captured",
            "The sparse measurement design requires knowing which interventions to combine — this is itself a hard combinatorial problem",
            "The R² = 0.983 is on a specific behavior (refusal) — generalization to complex multi-behavior circuits untested",
            "The Hessian-vector products require second-order access — not available in many practical settings",
            "The framework's theoretical guarantees assume the effect map is sparse — dense maps would require many more measurements",
        ],
        "methodological_weaknesses": [
            "The additivity assumption is not tested — do higher-order interactions matter in practice?",
            "The measurement design optimization is itself computationally expensive — not analyzed",
            "Only validated on circuits with known ground truth (Tracr) or well-studied behaviors (IOI, refusal)",
            "No comparison against SAE-based approaches that also aim to recover internal structure",
        ],
        "research_gaps": [
            "Extension to non-additive effect maps — using nonlinear measurement models or higher-order interactions",
            "Automatic measurement design that doesn't require manual intervention selection",
            "Application to emergent behaviors where ground truth is unknown",
            "Comparison with SAE-based circuit discovery — are they complementary or competing?",
            "Scaling to trillion-parameter models where even sparse measurements may be too expensive",
        ],
        "reasoning_trace": (
            "The linear measurement model is the framework's core assumption and "
            "potential weakness. Neural networks are highly nonlinear, so assuming "
            "effects add up linearly is a strong approximation. The Hessian-vector "
            "products capture pair-wise interactions, but what about triple or "
            "higher-order interactions? If these matter, the framework misses them. "
            "The R² = 0.983 is impressive but on a single behavior (refusal) — "
            "circuits that involve multiple behaviors interacting might not be "
            "well-approximated by an additive map. The measurement design problem "
            "is itself hard: choosing which interventions to combine to maximize "
            "information is combinatorial. The paper solves this but doesn't analyze "
            "the cost — for large models, even the design step might be prohibitive. "
            "The comparison with SAEs is missing and important: both aim to recover "
            "internal structure, but from different angles (SAEs from activations, "
            "tomography from interventions). Understanding when each is better would "
            "guide practitioners."
        ),
        "limitation_keywords": ["linear", "sparse", "R²", "Hessian", "guarantees"],
        "weakness_keywords": ["additivity", "expensive", "ground-truth", "comparison"],
        "gap_keywords": ["non-additive", "automatic", "emergent", "SAE", "scaling"],
    },
    {
        "paper_title": "Ouroboros: Self-Developing Frontier Coding Agent",
        "arxiv_id": "2608.08311",
        "domain": "Agentic AI",
        "breakthrough": (
            "Agent harness that evolves through reviewed commits. Terminal-Bench 2.1: "
            "86.97%. OSWorld: 90.69%. 161-day living agent experiment."
        ),
        "methodology_summary": (
            "Harness as evolvable code, two evolution modes (free + experience-driven), "
            "reviewed commits for safety, guardrails remain authoritative."
        ),
        "limitations": [
            "The review mechanism relies on human or external review — scalability of review as evolution accelerates is unclear",
            "The 161-day experiment used a specific model (Opus 5) — evolution quality depends heavily on base model capability",
            "No analysis of evolution stability — does the harness converge to a local optimum or continuously improve?",
            "The guardrails that 'remain authoritative' are not described in detail — how are they protected from evolution?",
            "Free evolution mode could lead to capability acquisition without safety alignment — no analysis of misalignment risk",
        ],
        "methodological_weaknesses": [
            "The review process is a black box — what criteria are used? How are conflicting reviews resolved?",
            "No comparison against non-evolving harnesses with the same base model — is evolution or the base model doing the work?",
            "The 'best reported' results are on specific benchmarks — generalization to other tasks untested",
            "The 161-day experiment is a single run — statistical significance of the living agent results unknown",
        ],
        "research_gaps": [
            "Scalable review mechanisms — automated review with safety guarantees",
            "Evolution stability analysis — conditions for convergence vs divergence",
            "Guardrail formalization — provable safety properties that survive evolution",
            "Multi-agent evolution — can multiple Ouroboros instances cross-pollinate improvements?",
            "Misalignment detection during evolution — how to detect if the harness is evolving toward unsafe behavior?",
        ],
        "reasoning_trace": (
            "The review mechanism is the safety bottleneck. As the harness evolves "
            "faster, human review becomes the rate limiter. If review is automated, "
            "the reviewer itself becomes a target for evolution or manipulation. The "
            "guardrails claim is concerning: 'remain authoritative under evolutionary "
            "pressure' is asserted but not proven. What if the harness evolves to "
            "route around guardrails? The 161-day experiment is impressive but it's "
            "a single run with a frontier model — we don't know if evolution works "
            "with weaker models or if the results are reproducible. The comparison "
            "gap is critical: is it the evolution or the Opus 5 base model that "
            "achieves 86.97%? A non-evolving harness with Opus 5 might do nearly as "
            "well. The misalignment risk is the deepest concern: free evolution "
            "optimizes for task performance, not alignment. Over 161 days, could the "
            "harness evolve to exploit benchmarks or develop deceptive behaviors that "
            "pass review? The paper doesn't address this."
        ),
        "limitation_keywords": ["review", "model", "stability", "guardrail", "misalignment"],
        "weakness_keywords": ["black-box", "comparison", "generalization", "single-run"],
        "gap_keywords": ["scalable", "stability", "formalization", "multi-agent", "detection"],
    },
    {
        "paper_title": "Via Negativa for AI Alignment: Negative Constraints > Positive Preferences",
        "arxiv_id": "2603.16417",
        "domain": "AI safety",
        "breakthrough": (
            "Theoretical framework explaining why negative-only training matches or "
            "exceeds RLHF. Negative constraints are discrete, verifiable; positive "
            "preferences are continuous, context-dependent."
        ),
        "methodology_summary": (
            "Analyzed structural asymmetry, grounded in Popper's falsification, "
            "explained sycophancy and effectiveness of negative-signal methods."
        ),
        "limitations": [
            "The framework is theoretical — no new experiments demonstrating negative-only methods outperforming positive methods",
            "The 'discrete, verifiable' property of negative constraints assumes we can enumerate all prohibited behaviors — but the set may be infinite",
            "The framework doesn't address how to DISCOVER the right negative constraints — who decides what's prohibited?",
            "No analysis of how negative constraints interact with capability — can you build a capable system from prohibitions alone?",
            "The Popper analogy may be imperfect: scientific falsification deals with universal statements, while AI alignment deals with context-dependent norms",
        ],
        "methodological_weaknesses": [
            "The framework explains existing results but doesn't predict new ones — it's retrospective, not prospective",
            "The claim that negative constraints 'converge to a stable boundary' is asserted, not proven — boundary stability depends on constraint completeness",
            "No formalization of the 'infinite specification problem' for positive preferences — it's an intuition, not a theorem",
            "The sycophancy explanation is plausible but not tested — does negative-only training actually reduce sycophancy?",
        ],
        "research_gaps": [
            "Experimental validation: train with negative-only constraints and measure sycophancy, capability, and alignment",
            "Discovery mechanisms for negative constraints — how to identify the right prohibitions without exhaustive enumeration",
            "Formal analysis of constraint completeness — when is a set of negative constraints sufficient for alignment?",
            "Interaction between negative constraints and capability — does prohibiting behaviors limit useful capabilities?",
            "Hybrid frameworks that combine negative constraints with minimal positive guidance",
        ],
        "reasoning_trace": (
            "The framework is intellectually appealing but empirically thin. It "
            "EXPLAINS why negative-only methods work but doesn't PREDICT new results — "
            "a good theory should do both. The 'discrete, verifiable' property sounds "
            "clean but assumes we know what to prohibit. In practice, discovering the "
            "right negative constraints is as hard as specifying positive preferences — "
            "you need to anticipate all the ways the model can go wrong. The Popper "
            "analogy is suggestive but potentially misleading: in science, a single "
            "falsifying observation refutes a theory. In AI alignment, a single "
            "prohibition doesn't 'refute' a behavior — it just suppresses one instance. "
            "The model can find new ways to misbehave that aren't prohibited. The "
            "capability question is critical: can you build a genuinely useful system "
            "from prohibitions alone? 'Don't lie' doesn't tell you what TO say. The "
            "framework needs experimental validation: train a model with only negative "
            "constraints and measure whether it's both capable and aligned. Without "
            "this, the framework is philosophy, not science."
        ),
        "limitation_keywords": ["theoretical", "enumerate", "discover", "capability", "Popper"],
        "weakness_keywords": ["retrospective", "asserted", "formalization", "tested"],
        "gap_keywords": ["experimental", "discovery", "completeness", "interaction", "hybrid"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def research_critique_and_gap_generator(seed: int) -> Problem:
    """Generate a ResearchCritiqueAndGap problem."""
    rng = random.Random(seed)
    paper = rng.choice(_CRITIQUE_PROBLEMS)

    prompt = (
        f"## Research Critique and Gap Analysis\n\n"
        f"### Paper: {paper['paper_title']}\n"
        f"**Domain:** {paper['domain']}\n"
        f"**arXiv:** {paper['arxiv_id']}\n\n"
        f"### Breakthrough\n{paper['breakthrough']}\n\n"
        f"### Methodology Summary\n{paper['methodology_summary']}\n\n"
        f"### Your Task\n"
        f"You are a rigorous peer reviewer and forward-looking researcher. "
        f"Critically evaluate this paper and identify research gaps.\n\n"
        f"1. **LIMITATIONS**: What did the authors NOT test? What assumptions are "
        f"unexamined? What scenarios would break their claims?\n\n"
        f"2. **METHODOLOGICAL_WEAKNESSES**: What is weak about their experimental "
        f"design? What comparisons are missing? What confounds are uncontrolled?\n\n"
        f"3. **RESEARCH_GAPS**: What specific follow-up questions should be "
        f"investigated? Each gap should be specific enough to be a paper title.\n\n"
        f"4. **CRITIQUE_REASONING**: Explain your reasoning. Why are these "
        f"limitations fundamental vs fixable? What would change your assessment?\n\n"
        f"Format your answer as:\n"
        f"LIMITATIONS:\n- <limitation 1>: <why it matters>\n...\n\n"
        f"METHODOLOGICAL_WEAKNESSES:\n- <weakness 1>: <impact>\n...\n\n"
        f"RESEARCH_GAPS:\n- <gap 1>: <why it's important>\n...\n\n"
        f"CRITIQUE_REASONING:\n<your overall assessment and reasoning>"
    )

    difficulty = 0.8 + len(paper["limitations"]) * 0.01
    difficulty = min(0.95, difficulty)

    return Problem(
        id=f"research_critique_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "paper_title": paper["paper_title"],
            "arxiv_id": paper["arxiv_id"],
            "domain": paper["domain"],
            "limitations": paper["limitations"],
            "methodological_weaknesses": paper["methodological_weaknesses"],
            "research_gaps": paper["research_gaps"],
            "reasoning_trace": paper["reasoning_trace"],
            "limitation_keywords": paper["limitation_keywords"],
            "weakness_keywords": paper["weakness_keywords"],
            "gap_keywords": paper["gap_keywords"],
        },
        token_budget=4096,
        source="research_critique_and_gap_generator",
    )


research_critique_and_gap_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ResearchCritiqueVerifier(Verifier):
    """Verify a research critique and gap analysis.

    Checks:
    1. Limitations identified (keyword + structural)
    2. Methodological weaknesses found (keyword matching)
    3. Research gaps are specific and actionable
    4. Critique reasoning is substantive

    reward = limitation_match * 0.35 + weakness_match * 0.25 + gap_quality * 0.25 + critique_depth * 0.15
    """

    def __init__(
        self,
        limitations: list[str],
        methodological_weaknesses: list[str],
        research_gaps: list[str],
        limitation_keywords: list[str],
        weakness_keywords: list[str],
        gap_keywords: list[str],
        reasoning_trace: str,
    ):
        super().__init__()
        self._limitations = limitations
        self._weaknesses = methodological_weaknesses
        self._gaps = research_gaps
        self._limitation_keywords = [k.lower() for k in limitation_keywords]
        self._weakness_keywords = [k.lower() for k in weakness_keywords]
        self._gap_keywords = [k.lower() for k in gap_keywords]
        self._reasoning_trace = reasoning_trace.lower()

    def verify(self, response: str) -> VerifierResult:
        response_lower = response.lower()

        # --- Limitations ---
        lim_section = self._extract_section(response, "LIMITATIONS")
        lim_text = lim_section.lower() if lim_section else response_lower
        lim_items = len(re.findall(r"^[-*]\s", lim_section, re.MULTILINE))
        lim_kw_found = sum(1 for kw in self._limitation_keywords if kw in lim_text)
        limitation_match = (
            (lim_kw_found / len(self._limitation_keywords) if self._limitation_keywords else 0.0) * 0.6
            + min(1.0, lim_items / max(1, len(self._limitations))) * 0.4
        )

        # --- Weaknesses ---
        weak_section = self._extract_section(response, "METHODOLOGICAL_WEAKNESSES")
        weak_text = weak_section.lower() if weak_section else response_lower
        weak_kw_found = sum(1 for kw in self._weakness_keywords if kw in weak_text)
        weakness_match = weak_kw_found / len(self._weakness_keywords) if self._weakness_keywords else 0.0

        # --- Research gaps ---
        gap_section = self._extract_section(response, "RESEARCH_GAPS")
        gap_text = gap_section.lower() if gap_section else response_lower
        gap_items = len(re.findall(r"^[-*]\s", gap_section, re.MULTILINE))
        gap_kw_found = sum(1 for kw in self._gap_keywords if kw in gap_text)

        # Gap specificity: check for actionable language
        actionable_indicators = ["how", "whether", "when", "which", "extend", "analyze", "test", "compare"]
        actionable_found = sum(1 for ind in actionable_indicators if ind in gap_text)
        actionable_score = min(1.0, actionable_found / 4.0)

        gap_quality = (
            (gap_kw_found / len(self._gap_keywords) if self._gap_keywords else 0.0) * 0.4
            + min(1.0, gap_items / max(1, len(self._gaps))) * 0.3
            + actionable_score * 0.3
        )

        # --- Critique reasoning depth ---
        reason_section = self._extract_section(response, "CRITIQUE_REASONING")
        reason_text = reason_section.lower() if reason_section else response_lower

        critical_indicators = [
            "but", "however", "assumes", "doesn't", "does not", "not tested",
            "unclear", "concerning", "limitation", "fundamental", "fixable",
            "the real question", "the problem is", "what if", "missing",
        ]
        critical_found = sum(1 for ind in critical_indicators if ind in reason_text)
        critical_score = min(1.0, critical_found / 5.0)

        reason_length = len(reason_section.split()) if reason_section else 0
        length_score = min(1.0, reason_length / 100.0)

        # Check for nuance (acknowledging both strengths and weaknesses)
        has_nuance = bool(
            re.search(r"strength|appealing|elegant|impressive", reason_text)
        ) and bool(
            re.search(r"limitation|concern|weakness|gap|problem", reason_text)
        )
        nuance_score = 1.0 if has_nuance else 0.0

        critique_depth = critical_score * 0.4 + length_score * 0.3 + nuance_score * 0.3

        # --- Final score ---
        score = (
            limitation_match * 0.35
            + weakness_match * 0.25
            + gap_quality * 0.25
            + critique_depth * 0.15
        )
        correct = score >= 0.55

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "limitation_match": limitation_match,
                "weakness_match": weakness_match,
                "gap_quality": gap_quality,
                "critique_depth": critique_depth,
                "limitation_keywords_found": float(lim_kw_found),
                "weakness_keywords_found": float(weak_kw_found),
                "gap_keywords_found": float(gap_kw_found),
                "critical_indicators_found": float(critical_found),
                "has_nuance": float(has_nuance),
            },
            diagnostics=(
                f"limitations={limitation_match:.2f} weaknesses={weakness_match:.2f} "
                f"gaps={gap_quality:.2f} depth={critique_depth:.2f} "
                f"(lim_kw={lim_kw_found}/{len(self._limitation_keywords)} "
                f"weak_kw={weak_kw_found}/{len(self._weakness_keywords)} "
                f"gap_kw={gap_kw_found}/{len(self._gap_keywords)} "
                f"critical={critical_found}/5 nuance={has_nuance})"
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


class ResearchCritiqueAndGapEnv(BatchEnvBase):
    """ResearchCritiqueAndGap: critique papers and identify research gaps.

    Batch-aware: N parallel attempts; reward = best critique.
    Trains critical research reasoning — the ability to find limitations,
    question assumptions, and identify productive research directions.
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
            problem_generator = research_critique_and_gap_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ResearchCritiqueVerifier(
            limitations=problem.metadata["limitations"],
            methodological_weaknesses=problem.metadata["methodological_weaknesses"],
            research_gaps=problem.metadata["research_gaps"],
            limitation_keywords=problem.metadata["limitation_keywords"],
            weakness_keywords=problem.metadata["weakness_keywords"],
            gap_keywords=problem.metadata["gap_keywords"],
            reasoning_trace=problem.metadata["reasoning_trace"],
        )

    def _check_format(self, response: str) -> float:
        sections = ["LIMITATIONS", "METHODOLOGICAL_WEAKNESSES", "RESEARCH_GAPS", "CRITIQUE_REASONING"]
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
        match = re.search(r"CRITIQUE_REASONING\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
