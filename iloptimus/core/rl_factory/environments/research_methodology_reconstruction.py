"""
ResearchMethodologyReconstruction: Reconstruct the methodology chain from a paper's outcome.

Environment concept:
  The model is given a paper's breakthrough result and key metrics. It must
  reconstruct the METHODOLOGY — the chain of experimental decisions, the
  reasoning behind each design choice, and why this approach succeeded where
  prior work failed. This trains deep research reasoning: understanding not
  just WHAT was discovered but HOW researchers got there and WHY each decision
  was made.

  Why: A real ML researcher doesn't just read results — they reverse-engineer
  the thought process. This environment forces the model to think like a
  researcher who must justify every experimental decision with causal reasoning
  about why it was necessary and what would have gone wrong without it.

  The reasoning traces are manually authored from real 2026 arxiv papers —
  no scripts generate them. Each problem contains a hand-written methodology
  chain with decision points, alternatives considered, and failure modes avoided.

Verification:
  - Methodology steps match ground truth (keyword + structural matching)
  - Key decisions identified correctly
  - Reasoning chain is causal (each step follows from the previous)

Reward design:
  step_match * 0.4 + decision_match * 0.3 + reasoning_quality * 0.3
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — real 2026 arxiv papers with manually authored methodology chains
# Each entry contains the breakthrough, the methodology steps, key decisions,
# and the reasoning trace explaining WHY each decision was made.
# ---------------------------------------------------------------------------

_METHODOLOGY_PROBLEMS = [
    {
        "paper_title": "f-GRPO: Divergence-Based Reinforcement Learning for General LLM Alignment",
        "arxiv_id": "2602.05946",
        "domain": "RL training methods",
        "breakthrough": (
            "Unified disparate alignment objectives (RLVR and preference alignment) "
            "under a single mathematical framework using f-divergences, deriving "
            "f-GRPO (on-policy) and f-HAL (hybrid on/off-policy) from variational "
            "representations of f-divergences."
        ),
        "key_results": (
            "Superior performance and flexibility compared to current methods across "
            "both RLVR (Math Reasoning) and PA (Safety Alignment) tasks."
        ),
        "methodology_steps": [
            "Recognized that preference alignment objectives act as divergence estimators between aligned and unaligned response distributions",
            "Extended this insight from preference alignment to RLVR where only environmental rewards are available",
            "Derived f-GRPO and f-HAL from variational representations of f-divergences",
            "Provided theoretical guarantees that these objectives improve average reward after alignment",
            "Validated empirically on both RLVR (Math Reasoning) and PA tasks (Safety Alignment)",
        ],
        "key_decisions": [
            "Use f-divergences as the unifying mathematical framework rather than algorithm-specific mechanics",
            "Treat RLVR as a divergence estimation problem to apply well-understood statistical tools",
            "Avoid ad-hoc baseline constructions that plague existing methods",
            "Derive both on-policy (f-GRPO) and hybrid (f-HAL) variants from the same framework",
        ],
        "reasoning_trace": (
            "The researchers started by observing that preference alignment objectives "
            "are fundamentally estimating a divergence between distributions. The key "
            "insight was that this perspective generalizes: RLVR is also a divergence "
            "estimation problem, just with a different reward structure. By using "
            "f-divergences (a broad family that includes KL, chi-squared, and others), "
            "they could unify both settings under one framework. The variational "
            "representation of f-divergences provides a practical way to compute these "
            "objectives without explicit density estimation. This avoids the ad-hoc "
            "baseline constructions that make existing methods fragile — instead of "
            "guessing what baseline to subtract, the framework tells you. The theoretical "
            "guarantees ensure the objectives actually improve reward, not just look "
            "good on paper. The dual validation (RLVR + PA) demonstrates the framework's "
            "generality rather than overfitting to one setting."
        ),
        "methodology_keywords": ["divergence", "variational", "f-divergence", "alignment", "theoretical", "guarantee"],
        "decision_keywords": ["unify", "framework", "baseline", "on-policy", "hybrid"],
    },
    {
        "paper_title": "Cue-GRPO: Rarity-Aware Credit Redistribution for GRPO",
        "arxiv_id": "2608.03467",
        "domain": "RL training methods",
        "breakthrough": (
            "Identified and solved 'multiplicity-induced structure-level credit "
            "concentration' in GRPO, where recurring correct solution forms accumulate "
            "credit proportional to sampling frequency rather than rarity."
        ),
        "key_results": (
            "Improves AIME repeated-sampling performance with largest gains at high "
            "sampling budgets. Only 6% wall-clock training overhead over GRPO. "
            "Tested on Qwen2.5-Math-7B and Llama-3.1-8B-Instruct."
        ),
        "methodology_steps": [
            "Formalized how GRPO's completion-level uniformity creates structure-level skew",
            "Identified that common correct solutions get over-credited while rare correct solutions are under-credited",
            "Introduced partition-conditioned rule that redistributes positive advantages according to cluster rarity",
            "Used deterministic Strategy Cues to construct rollout-local partitions of verified-correct traces",
            "Avoided auxiliary-model inference by using deterministic cues instead of learned clustering",
        ],
        "key_decisions": [
            "Redistribute credit by cluster rarity rather than uniform per-completion assignment",
            "Use deterministic Strategy Cues instead of learned clustering to avoid auxiliary model overhead",
            "Construct partitions within rollouts rather than globally to keep it rollout-local",
            "Target high sampling budgets where the bias is most pronounced",
        ],
        "reasoning_trace": (
            "The researchers noticed a subtle statistical bias in GRPO: when you sample "
            "multiple correct solutions, some solution forms appear more frequently than "
            "others. GRPO treats all correct solutions equally, so the common forms "
            "accumulate more total credit simply because they're sampled more often. "
            "This is wrong — rare solution forms represent harder-to-discover reasoning "
            "patterns and should receive MORE credit per occurrence, not less. The fix "
            "is to partition correct solutions by structural similarity (using Strategy "
            "Cues) and redistribute credit inversely to cluster frequency. The key "
            "design decision was using deterministic cues rather than a learned clustering "
            "model — this keeps overhead at 6% wall-clock and avoids introducing a new "
            "model that could itself be a source of bias. The rollout-local partitioning "
            "ensures the redistribution happens within each group, preserving GRPO's "
            "group-relative structure."
        ),
        "methodology_keywords": ["multiplicity", "credit", "rarity", "partition", "cluster", "redistribute"],
        "decision_keywords": ["deterministic", "strategy", "cue", "overhead", "rollout-local"],
    },
    {
        "paper_title": "FORGE: Fused On-Register Gradient Elimination for Memory-Efficient LLM Training",
        "arxiv_id": "2606.22932",
        "domain": "Efficient training",
        "breakthrough": (
            "Eliminated the gradient memory pool entirely by applying the optimizer "
            "to each weight-gradient tile in fp32 registers immediately after "
            "computation, reducing peak memory by 16-33% while running 1.5x faster."
        ),
        "key_results": (
            "Llama-3.1-8B: peak memory 62.0 GB -> 48.4 GB, 1.5x faster. "
            "With int8 moments: 35.3 GB. Trains 32B model with Muon on H200 "
            "where standard Muon does not fit."
        ),
        "methodology_steps": [
            "Observed that reverse-mode differentiation computes every weight gradient, writes it to memory, then optimizer reads it back",
            "Recognized that each gradient is used exactly once and then discarded",
            "Identified that the gradient pool is an artifact of scheduling, not a learning requirement",
            "Applied optimizer to each weight-gradient tile in fp32 registers immediately after computation",
            "Proved the fused step is exact whenever only the optimizer's state update reads the gradient",
            "Made the method architecture-agnostic (transformers, SSMs, MLP mixers)",
        ],
        "key_decisions": [
            "Fuse gradient computation and optimizer consumption rather than compressing gradients",
            "Keep gradients in registers rather than writing to memory",
            "Prove exactness conditions rather than accepting approximation",
            "Make it composable with other memory-saving techniques (quantized states, low-rank)",
        ],
        "reasoning_trace": (
            "The breakthrough came from questioning a fundamental assumption that "
            "everyone else took for granted: that gradients must be materialized and "
            "stored in memory. The researchers traced the data flow and realized that "
            "each gradient is computed, written to memory, read back by the optimizer, "
            "and then never used again. This is purely a scheduling artifact — the "
            "backward pass produces gradients in order, and the optimizer consumes them "
            "in the same order. By fusing these two operations, the gradient never "
            "needs to leave the registers where it was computed. The exactness proof "
            "was critical: it shows this isn't an approximation but a mathematically "
            "identical computation, just scheduled differently. The architecture-agnostic "
            "design works because the insight (gradients are transient) is true for "
            "any architecture that uses reverse-mode autodiff. The composability with "
            "other techniques is important because it changes WHEN gradients are consumed, "
            "not WHAT is computed, so it doesn't conflict with quantization or low-rank."
        ),
        "methodology_keywords": ["gradient", "register", "memory", "optimizer", "fuse", "exact"],
        "decision_keywords": ["eliminate", "transient", "schedule", "compose", "architecture-agnostic"],
    },
    {
        "paper_title": "CLEAR: Continuous Latent Adapter Routing for Utility-Preserving LLM Safety Alignment",
        "arxiv_id": "2608.21278",
        "domain": "AI safety",
        "breakthrough": (
            "A conditional safety adaptation framework using a lightweight hidden-state "
            "gate to continuously control safety LoRA adapter activation strength, "
            "reducing safety-utility trade-off dramatically."
        ),
        "key_results": (
            "On Llama-3-8B-Instruct: HarmBench ASR 32.3% -> 0.5%, while achieving "
            "up to 7.1pp higher GSM8K accuracy than globally applied SFT or LoRA."
        ),
        "methodology_steps": [
            "Identified that globally applied safety tuning degrades performance on benign prompts by modifying the entire model",
            "Froze the backbone LLM and introduced a safety-specific LoRA adapter",
            "Trained a learned latent gate that continuously controls adapter activation strength based on input",
            "Performed input-conditioned adapter routing: benign prompts receive minimal intervention, harmful prompts trigger strong safety responses",
            "Validated on Llama-3-8B-Instruct using HarmBench for safety and GSM8K for utility",
        ],
        "key_decisions": [
            "Make safety intervention conditional rather than global",
            "Use a continuous gate rather than binary on/off switching",
            "Freeze backbone to preserve native reasoning capabilities",
            "Separate safety and utility as independent concerns rather than trading one for the other",
        ],
        "reasoning_trace": (
            "The key insight is that safety interventions should be conditional, not "
            "global. Previous approaches assumed safety requires modifying the entire "
            "model, which inevitably degrades benign-task performance. CLEAR recognizes "
            "that safety and utility are separable: a harmful prompt needs safety "
            "intervention, but a math problem doesn't. The continuous gate (rather than "
            "binary) is important because some prompts are ambiguous — a partial "
            "activation provides graceful degradation rather than a cliff edge. Freezing "
            "the backbone is the mechanism that preserves utility: the model's native "
            "reasoning is untouched, and the adapter only kicks in when the gate "
            "activates it. The 7.1pp GSM8K improvement over global LoRA is the proof: "
            "when you don't corrupt the model's math ability with safety training, it "
            "performs better. This works where others failed because it treats safety "
            "as a selective intervention rather than a global constraint."
        ),
        "methodology_keywords": ["conditional", "gate", "adapter", "safety", "utility", "freeze"],
        "decision_keywords": ["continuous", "selective", "separable", "intervention", "backbone"],
    },
    {
        "paper_title": "SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning",
        "arxiv_id": "2608.23493",
        "domain": "Agentic AI",
        "breakthrough": (
            "Internalized self-reflection as a mechanism for dense reward generation, "
            "enabling LLMs to analyze their own trajectories, synthesize error patterns "
            "into reflection patches, and use these as dense token-level training signals "
            "without external critics."
        ),
        "key_results": (
            "73.3% on AIME'24 using only 0.08x the training FLOPs of scaled SFT. "
            "64.7% on WebShop, 76.8% on ALFWorld, 31.2% on SWE-Bench-Lite."
        ),
        "methodology_steps": [
            "Identified that trajectory-level rewards assign uniform advantage to every step, failing to identify which decisions mattered",
            "Designed LLMs to analyze completed trajectories and synthesize concise reflection patches capturing error patterns",
            "Prepended reflection patches to original prompt via reset-with-memory mechanism",
            "Used reflection-conditioned teacher to score student on-policy rollouts, yielding dense token-level supervision",
            "Transformed sparse terminal supervision into dense learning signals while preserving task specification fidelity",
        ],
        "key_decisions": [
            "Use the model as its own teacher rather than requiring external critics",
            "Generate reflection patches that capture error patterns, not just error locations",
            "Use reset-with-memory to inject learned guidance without corrupting task specification",
            "Train with only 8% of SFT FLOPs by leveraging dense self-generated signals",
        ],
        "reasoning_trace": (
            "The core problem is credit assignment: when a trajectory fails, which step "
            "caused it? Standard RL gives every step the same negative advantage, which "
            "is wrong — some steps were fine, others were the critical mistake. SRPO's "
            "insight is that the model itself can identify its errors by reflecting on "
            "completed trajectories. The reflection patch is a compressed representation "
            "of 'what went wrong and how to avoid it.' The reset-with-memory mechanism "
            "is crucial: it regenerates from a clean initial state (preserving the task "
            "specification) while injecting the learned guidance. This prevents the "
            "reflection from corrupting the task understanding. The reflection-conditioned "
            "teacher then scores new rollouts, producing dense token-level supervision "
            "from what was originally sparse terminal feedback. The 0.08x FLOPs "
            "efficiency comes from the density of the signal — each training example "
            "provides much more learning signal than outcome-only RL."
        ),
        "methodology_keywords": ["reflection", "credit", "trajectory", "dense", "patch", "self"],
        "decision_keywords": ["self-teacher", "reset-with-memory", "synthesize", "token-level", "FLOPs"],
    },
    {
        "paper_title": "Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability",
        "arxiv_id": "2608.19338",
        "domain": "AI interpretability",
        "breakthrough": (
            "Unified activation patching, gradients, Hessian-vector products, and subset "
            "interventions as a shared measurement problem, providing principled recovery "
            "of internal mechanisms and intervention effects."
        ),
        "key_results": (
            "On GPT-2-small IOI: reproduced conditional backup and identified Name Mover "
            "interaction. On Qwen-2.5-7B: calibrated additive map achieved held-out "
            "R² = 0.983 on refusal-response surface."
        ),
        "methodology_steps": [
            "Formalized the measurement problem as y_tilde = Ax + w where A records interventions, x is the effect map, w is error",
            "Developed sparse aggregate measurements for forward-only access that recover finite-effect maps with fewer interventions",
            "Created gradient-based finite probes that substantially improve local attribution maps",
            "Introduced lifted measurements and designed Hessian-vector products to recover pair interactions missed by first-order maps",
            "Validated on two-HMM belief-state model, Tracr, GPT-2-small IOI, and Qwen-2.5-7B",
        ],
        "key_decisions": [
            "Treat all interpretability methods as solving the same underlying measurement problem",
            "Use sparse aggregate measurements to reduce intervention count below exhaustive coordinate patching",
            "Design measurements rather than using ad-hoc interventions",
            "Provide theoretical guarantees on measurement sufficiency",
        ],
        "reasoning_trace": (
            "The field of mechanistic interpretability has many tools — activation "
            "patching, gradients, Hessians — but they're used ad-hoc without a unifying "
            "theory. The researchers recognized that all these methods are solving the "
            "same problem: recovering a hidden effect map from partial observations. "
            "By formalizing this as y = Ax + w (a linear measurement model), they could "
            "import decades of compressed sensing and experimental design theory. The "
            "sparse aggregate measurements are the key practical contribution: instead "
            "of patching one component at a time (exponential in the number of "
            "components), you can design measurements that recover the same information "
            "with far fewer interventions. The Hessian-vector products capture "
            "interactions that first-order methods miss — crucial because neural "
            "networks are highly interactive systems. The R² = 0.983 on Qwen-2.5-7B "
            "shows the recovered map generalizes, meaning the measurement design was "
            "sufficient to capture the relevant structure."
        ),
        "methodology_keywords": ["measurement", "tomography", "sparse", "Hessian", "intervention", "recovery"],
        "decision_keywords": ["unify", "design", "sufficiency", "aggregate", "interaction"],
    },
    {
        "paper_title": "Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits",
        "arxiv_id": "2602.20532",
        "domain": "RL training methods",
        "breakthrough": (
            "Trained a neural curator that dynamically selects training problems by "
            "directly optimizing for expected policy performance improvement using a "
            "bandit formulation."
        ),
        "key_results": (
            "28.6% relative gain on AIME2024, 30.5% on ARC-1D over strongest baseline, "
            "with up to 80% speedup."
        ),
        "methodology_steps": [
            "Formulated problem selection as a non-stationary stochastic bandit problem",
            "Derived principled loss function based on online stochastic mirror descent",
            "Established regret guarantees under partial feedback",
            "Trained neural curator to adaptively curate problems from large banks",
            "Directly optimized for expected policy performance improvement rather than proxy objectives",
        ],
        "key_decisions": [
            "Optimize the meta-objective (policy improvement) directly rather than proxy objectives like difficulty scores",
            "Use bandit formulation to provide theoretical regret guarantees",
            "Make curriculum co-adaptive: curator selects, actor trains, improvement feeds curator",
            "Treat problem selection as non-stationary because the policy changes over time",
        ],
        "reasoning_trace": (
            "Existing curriculum learning methods use proxy objectives — difficulty "
            "scores, uncertainty estimates, or hand-crafted heuristics. The problem is "
            "that these proxies don't directly measure what we want: does training on "
            "this problem actually improve the policy? The bandit formulation is the "
            "key insight: each problem is an arm, the reward is the policy improvement "
            "after training on it, and the curator must balance exploration (trying new "
            "problems) with exploitation (focusing on known-useful ones). Online "
            "stochastic mirror descent provides regret guarantees, meaning the curator "
            "provably converges to the optimal selection strategy. The non-stationarity "
            "is critical: as the policy improves, the optimal curriculum changes. "
            "Problems that were useful early become too easy, and new problems become "
            "tractable. The co-adaptive design ensures the curator and policy improve "
            "together — the curator learns what helps the current policy, and the "
            "policy's improvement creates new learning opportunities."
        ),
        "methodology_keywords": ["bandit", "curator", "curriculum", "mirror", "descent", "regret"],
        "decision_keywords": ["meta-objective", "non-stationary", "co-adaptive", "proxy", "improvement"],
    },
    {
        "paper_title": "Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation",
        "arxiv_id": "2608.20256",
        "domain": "LLM reasoning",
        "breakthrough": (
            "Demonstrated that a model can learn to allocate its own reasoning effort "
            "by choosing, as the first token of its response, one of three modes: "
            "NoThink, Short, or Long."
        ),
        "key_results": (
            "Three modes emerge without collapsing. Policy stays close to base accuracy "
            "on MATH500 (0.782 vs 0.796) while cutting mean response length from 4,796 "
            "to 2,811 tokens (41% reduction). Transfers with 76% token reduction on GSM8K."
        ),
        "methodology_steps": [
            "Identified that fixed token budgets lead to over-computation on easy problems and insufficient computation on hard ones",
            "Designed routing token as the first token of the response, choosing among NoThink, Short, or Long modes",
            "Learned the routing token inside GRPO with no separate router through shaped reward",
            "Made each mode worthwhile at a different response length with hard per-mode token caps",
            "Trained a 1.5B distilled model on MATH and evaluated transfer to other benchmarks",
        ],
        "key_decisions": [
            "Use the first token as the routing mechanism rather than a separate router model",
            "Apply shaped reward that makes each mode worthwhile at different lengths",
            "Use hard per-mode token caps to keep modes distinct and prevent mode collapse",
            "Learn routing inside GRPO rather than as a separate training phase",
        ],
        "reasoning_trace": (
            "The fundamental inefficiency in reasoning models is that they apply the "
            "same compute budget to all problems. Easy problems get over-thought (wasting "
            "tokens) and hard problems get under-thought (not enough reasoning). The "
            "solution seems obvious: allocate compute adaptively. But the challenge is "
            "HOW to learn this. A separate router model adds complexity and inference "
            "overhead. The researchers' insight was to use the first generated token "
            "as the routing signal — the model itself decides how much to think before "
            "it starts thinking. The shaped reward is the key training trick: it makes "
            "each mode worthwhile at a different length, so the model has incentive to "
            "use all three modes rather than collapsing to the one that works best on "
            "average. The hard caps prevent mode collapse by ensuring modes stay "
            "distinct — without caps, 'Short' would drift toward 'Long' because longer "
            "reasoning usually helps. The 41% token reduction with minimal accuracy loss "
            "shows the router successfully sorts problems by difficulty. The 76% "
            "reduction on GSM8K transfer shows the skill generalizes — the model learned "
            "a general 'when to think' heuristic, not a MATH-specific one."
        ),
        "methodology_keywords": ["adaptive", "routing", "token", "mode", "budget", "allocate"],
        "decision_keywords": ["first-token", "shaped-reward", "hard-caps", "mode-collapse", "transfer"],
    },
    {
        "paper_title": "Seirenes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning",
        "arxiv_id": "2605.11636",
        "domain": "LLM reasoning",
        "breakthrough": (
            "Self-play RL framework that transforms contextual interference from a "
            "failure mode into an internal training signal by having a single model "
            "both construct distracting contexts and solve problems by discerning "
            "essential task from perturbations."
        ),
        "key_results": (
            "Across seven math reasoning benchmarks and model scales 4B-30B: average "
            "gains of +10.2, +9.1, +7.2 points. Distracting contexts from 4B model "
            "reduce accuracy of GPT and Gemini by ~4-5 points."
        ),
        "methodology_steps": [
            "Identified that contextual interference is a failure mode where models cannot discern essential task from perturbations",
            "Designed parameter-shared adversarial self-play loop where model both constructs and solves",
            "Model trained to construct plausible yet distracting contexts that expose its own reasoning blind spots",
            "Same model trained to solve problems by discerning essential task from these perturbations",
            "Continuous interaction sustains informative co-evolutionary curriculum as model improves",
        ],
        "key_decisions": [
            "Use a single model for both construction and solution (parameter sharing)",
            "Transform the failure mode into a training signal rather than just avoiding it",
            "Create co-evolutionary curriculum where distractor difficulty scales with model improvement",
            "Use adversarial structure to move beyond superficial pattern matching",
        ],
        "reasoning_trace": (
            "Most reasoning training focuses on clean problems. But real-world reasoning "
            "is messy — there are distractions, irrelevant information, and misleading "
            "context. Models that train only on clean problems fail when faced with "
            "noise. Seirenes' insight is to use the model's OWN vulnerability to "
            "distractions as a training signal. The model plays both roles: it constructs "
            "distractors that fool itself, then must solve the problem despite those "
            "distractors. This creates a co-evolutionary arms race: as the model gets "
            "better at ignoring distractors, it must also get better at creating them "
            "(since the same parameters are shared). The parameter sharing is key — "
            "it means the model's understanding of what's distracting IS its "
            "understanding of what's essential. The 4-5 point reduction on GPT/Gemini "
            "shows that the 4B model's distractors are genuinely hard, not just "
            "noise — they expose reasoning blind spots that even frontier models have. "
            "This works because the adversarial structure forces the model to anchor "
            "its capabilities in robust underlying reasoning rather than superficial "
            "pattern matching."
        ),
        "methodology_keywords": ["self-play", "adversarial", "distraction", "co-evolutionary", "perturbation", "discern"],
        "decision_keywords": ["parameter-shared", "failure-mode", "arms-race", "blind-spots", "robust"],
    },
    {
        "paper_title": "Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution",
        "arxiv_id": "2608.08311",
        "domain": "Agentic AI",
        "breakthrough": (
            "A self-developing agent harness whose tools, context assembly, prompts, "
            "and core implementation improve through reviewed commits that become the "
            "runtime for later work."
        ),
        "key_results": (
            "Opus 5 on Terminal-Bench 2.1: 86.97% (best reported). OSWorld-Verified: "
            "90.69%. Five-rollout CL-Bench: 0.2301 (new SOTA). Hope: 161-day living-agent "
            "experiment in free evolution."
        ),
        "methodology_steps": [
            "Designed model-harness system where the harness itself evolves through reviewed commits",
            "Implemented recursive free evolution where improvement is itself a task",
            "Implemented experience-driven core evolution where ordinary work exposes bugs and inefficiencies",
            "Required reviewed commits before changes become part of runtime (safety)",
            "Maintained guardrails that must remain authoritative under evolutionary pressure",
        ],
        "key_decisions": [
            "Treat the harness as evolvable code rather than fixed infrastructure",
            "Use two evolution modes: free (proactive) and experience-driven (reactive)",
            "Require review before commits become runtime (prevents destabilizing changes)",
            "Keep guardrails authoritative even under evolutionary pressure",
        ],
        "reasoning_trace": (
            "Most agent harnesses are designed once and never change. But as the agent "
            "works on diverse tasks, it encounters failure patterns that reveal harness "
            "limitations — bad context assembly, missing tools, inefficient prompts. "
            "Ouroboros treats the harness itself as code that the agent can improve. "
            "The two evolution modes cover both proactive improvement (free evolution: "
            "'make yourself better') and reactive improvement (experience-driven: 'fix "
            "what broke'). The reviewed commit requirement is the safety mechanism — "
            "it prevents the agent from making destabilizing changes while still "
            "allowing learning from experience. The guardrails are designed to remain "
            "authoritative even as the core evolves, creating a stable safety boundary "
            "within which evolution can happen freely. The 161-day Hope experiment "
            "demonstrates sustained evolution without collapse, showing the review "
            "mechanism is sufficient to prevent degeneration. This works because the "
            "agent can observe its own failure patterns and propose targeted "
            "improvements, with review ensuring safety."
        ),
        "methodology_keywords": ["evolution", "harness", "commit", "review", "guardrail", "runtime"],
        "decision_keywords": ["evolvable", "two-mode", "safety", "proactive", "reactive"],
    },
    {
        "paper_title": "Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior",
        "arxiv_id": "2603.16417",
        "domain": "AI safety",
        "breakthrough": (
            "Provided unified theoretical account explaining why negative-only training "
            "methods match or exceed standard RLHF, arguing negative constraints are "
            "structurally superior to positive preferences due to their discrete, "
            "verifiable nature."
        ),
        "key_results": (
            "Framework explains why Negative Sample Reinforcement achieves parity with "
            "PPO on math reasoning, Distributional Dispreference Optimization trains "
            "effectively using only dispreferred samples, and Constitutional AI "
            "outperforms pure RLHF on harmlessness."
        ),
        "methodology_steps": [
            "Analyzed the structural asymmetry between positive preferences and negative constraints",
            "Showed positive preferences encode continuously coupled, context-dependent values that cannot be exhaustively specified",
            "Demonstrated negative constraints encode discrete, finite, independently verifiable prohibitions",
            "Grounded analysis in Popper's falsification logic and epistemology of negative knowledge",
            "Explained both sycophancy failure of preference-based RLHF and effectiveness of negative-signal methods",
        ],
        "key_decisions": [
            "Frame alignment as a falsification problem rather than a verification problem",
            "Use negative constraints (what is wrong) rather than positive preferences (what is better)",
            "Ground the analysis in epistemology (Popper) rather than just empirical observations",
            "Explain multiple empirical phenomena under a single theoretical framework",
        ],
        "reasoning_trace": (
            "The puzzle: why do negative-only training methods (which only tell the "
            "model what NOT to do) work as well as or better than positive preference "
            "methods (which tell the model what TO do)? The answer is epistemological. "
            "Positive preferences suffer from an infinite specification problem: what "
            "are ALL the good responses? This is context-dependent, continuously "
            "coupled, and impossible to exhaustively enumerate. Negative constraints "
            "have a finite verification problem: is this response wrong? This is "
            "discrete, independently verifiable, and convergent. The asymmetry is "
            "fundamental — falsification (showing something is wrong) is always easier "
            "than verification (showing something is right). This explains sycophancy: "
            "positive preferences train the model to agree with the user, which is a "
            "surface correlate of 'good' that doesn't capture the underlying value. "
            "Negative constraints avoid this because 'don't be sycophantic' is a clear, "
            "verifiable prohibition. The framework also explains why Constitutional AI "
            "works: constitutions are primarily lists of things NOT to do, which aligns "
            "with the negative constraint structure."
        ),
        "methodology_keywords": ["negative", "constraint", "falsification", "verifiable", "preference", "epistemology"],
        "decision_keywords": ["via-negativa", "Popper", "discrete", "convergent", "sycophancy"],
    },
    {
        "paper_title": "Spectral Compact Training: Pre-Training LLMs via Permanent Truncated SVD",
        "arxiv_id": "2604.00733",
        "domain": "Efficient training",
        "breakthrough": (
            "Replaced dense weight matrices with permanent truncated SVD factors where "
            "the full dense matrix is never materialized, achieving up to 199x memory "
            "reduction per MLP layer at rank 32."
        ),
        "key_results": (
            "70B-parameter architecture training on Steam Deck: 7.2 GB peak memory vs. "
            "1,245 GB dense FP32. Rank 128: 11.7x MLP compression, 20.0 GB GPU memory. "
            "All ranks converge to same loss floor (~4.2-4.5)."
        ),
        "methodology_steps": [
            "Permanently stored every weight matrix in compact SVD factors: W = U diag(s) V^T",
            "Ensured U and V have orthonormal columns and s contains singular values",
            "Never constructed the dense matrix during training",
            "Flowed gradients through compact spectral factors via standard backpropagation",
            "Retracted U, V to Stiefel manifold via QR decomposition after each optimizer step",
        ],
        "key_decisions": [
            "Use permanent factored form throughout training rather than post-hoc compression",
            "Retract to Stiefel manifold after each step to maintain orthogonality",
            "Use QR decomposition for retraction (ensures factors stay orthonormal)",
            "Exploit low-rank structure of weight matrices as a training-time property, not just inference-time",
        ],
        "reasoning_trace": (
            "Most compression methods work AFTER training: train a dense model, then "
            "compress it. The insight here is that weight matrices in deep networks "
            "have low-rank structure DURING training, not just after. By permanently "
            "working in the factored form (W = U diag(s) V^T), you never need to "
            "materialize the dense matrix, saving memory throughout training. The "
            "Stiefel manifold retraction is the critical technical detail: after each "
            "gradient update, U and V drift away from orthonormality. The QR "
            "decomposition retracts them back, preserving the spectral properties that "
            "make the factored form stable. The convergence to the same loss floor "
            "across ranks (32-256) is surprising and important: it suggests the rank "
            "bottleneck is not as severe as expected for LLM training. The 199x memory "
            "reduction at rank 32 is because memory scales with rank * dimensions rather "
            "than dimensions^2. The Steam Deck result (7.2 GB for 70B) is a dramatic "
            "demonstration that the memory wall can be broken by working in the right "
            "representation space."
        ),
        "methodology_keywords": ["SVD", "spectral", "factored", "Stiefel", "orthonormal", "rank"],
        "decision_keywords": ["permanent", "retraction", "QR", "low-rank", "never-materialize"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def research_methodology_reconstruction_generator(seed: int) -> Problem:
    """Generate a ResearchMethodologyReconstruction problem."""
    rng = random.Random(seed)
    paper = rng.choice(_METHODOLOGY_PROBLEMS)

    methodology_steps = paper["methodology_steps"]
    key_decisions = paper["key_decisions"]
    reasoning_trace = paper["reasoning_trace"]

    prompt = (
        f"## Research Methodology Reconstruction\n\n"
        f"### Paper: {paper['paper_title']}\n"
        f"**Domain:** {paper['domain']}\n"
        f"**arXiv:** {paper['arxiv_id']}\n\n"
        f"### Breakthrough\n{paper['breakthrough']}\n\n"
        f"### Key Results\n{paper['key_results']}\n\n"
        f"### Your Task\n"
        f"You are a senior ML researcher reviewing this paper. Reconstruct the "
        f"methodology chain that led to this breakthrough. For each step:\n"
        f"1. What experimental decision was made?\n"
        f"2. Why was this decision necessary (what would have gone wrong without it)?\n"
        f"3. What alternative was considered and rejected?\n\n"
        f"Then identify the KEY DESIGN DECISIONS that made this work where prior "
        f"approaches failed.\n\n"
        f"Format your answer as:\n"
        f"METHODOLOGY:\n"
        f"1. <step 1>\n"
        f"   - Why: <reasoning>\n"
        f"   - Alternative: <what was rejected>\n"
        f"2. <step 2>\n"
        f"   ...\n\n"
        f"KEY_DECISIONS:\n"
        f"- <decision 1>\n"
        f"- <decision 2>\n"
        f"...\n\n"
        f"REASONING_TRACE:\n"
        f"<explain the causal logic of why this approach works where others failed>"
    )

    difficulty = 0.7 + len(methodology_steps) * 0.02
    difficulty = min(0.95, difficulty)

    return Problem(
        id=f"research_methodology_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "paper_title": paper["paper_title"],
            "arxiv_id": paper["arxiv_id"],
            "domain": paper["domain"],
            "methodology_steps": methodology_steps,
            "key_decisions": key_decisions,
            "reasoning_trace": reasoning_trace,
            "methodology_keywords": paper["methodology_keywords"],
            "decision_keywords": paper["decision_keywords"],
        },
        token_budget=4096,
        source="research_methodology_reconstruction_generator",
    )


research_methodology_reconstruction_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ResearchMethodologyVerifier(Verifier):
    """Verify a methodology reconstruction.

    Checks:
    1. Methodology steps match ground truth (keyword overlap + structural)
    2. Key decisions identified (keyword matching)
    3. Reasoning trace quality (causal language, depth indicators)

    reward = step_match * 0.4 + decision_match * 0.3 + reasoning_quality * 0.3
    """

    def __init__(
        self,
        methodology_steps: list[str],
        key_decisions: list[str],
        methodology_keywords: list[str],
        decision_keywords: list[str],
        reasoning_trace: str,
    ):
        super().__init__()
        self._methodology_steps = methodology_steps
        self._key_decisions = key_decisions
        self._methodology_keywords = [k.lower() for k in methodology_keywords]
        self._decision_keywords = [k.lower() for k in decision_keywords]
        self._reasoning_trace = reasoning_trace.lower()

    def verify(self, response: str) -> VerifierResult:
        response_lower = response.lower()

        # --- Check methodology section ---
        methodology_section = self._extract_section(response, "METHODOLOGY")
        if methodology_section:
            meth_text = methodology_section.lower()
        else:
            meth_text = response_lower

        # Count methodology keywords found
        meth_keywords_found = sum(
            1 for kw in self._methodology_keywords if kw in meth_text
        )
        meth_keyword_fraction = meth_keywords_found / len(self._methodology_keywords) if self._methodology_keywords else 0.0

        # Check for numbered steps (structural)
        numbered_steps = len(re.findall(r"^\s*\d+\.", methodology_section, re.MULTILINE))
        step_structure_score = min(1.0, numbered_steps / max(1, len(self._methodology_steps)))

        step_match = meth_keyword_fraction * 0.6 + step_structure_score * 0.4

        # --- Check key decisions section ---
        decisions_section = self._extract_section(response, "KEY_DECISIONS")
        if decisions_section:
            dec_text = decisions_section.lower()
        else:
            dec_text = response_lower

        dec_keywords_found = sum(
            1 for kw in self._decision_keywords if kw in dec_text
        )
        decision_match = dec_keywords_found / len(self._decision_keywords) if self._decision_keywords else 0.0

        # --- Check reasoning trace quality ---
        reasoning_section = self._extract_section(response, "REASONING_TRACE")
        if reasoning_section:
            reason_text = reasoning_section.lower()
        else:
            reason_text = response_lower

        # Check for causal reasoning indicators
        causal_indicators = [
            "because", "therefore", "thus", "since", "as a result",
            "the key insight", "the reason", "this works because",
            "without this", "instead of", "unlike", "prior work",
            "the problem is", "the challenge", "if we", "would have",
        ]
        causal_found = sum(1 for ind in causal_indicators if ind in reason_text)
        causal_score = min(1.0, causal_found / 5.0)

        # Check for depth (length of reasoning section)
        reason_length = len(reasoning_section.split()) if reasoning_section else 0
        length_score = min(1.0, reason_length / 80.0)

        # Check for reasoning trace keyword overlap
        reason_keywords = [
            "insight", "approach", "decision", "alternative",
            "failure", "work", "fundamental", "structure",
        ]
        reason_kw_found = sum(1 for kw in reason_keywords if kw in reason_text)
        reason_kw_score = min(1.0, reason_kw_found / 4.0)

        reasoning_quality = causal_score * 0.4 + length_score * 0.3 + reason_kw_score * 0.3

        # --- Compute final score ---
        score = step_match * 0.4 + decision_match * 0.3 + reasoning_quality * 0.3
        correct = score >= 0.6

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "step_match": step_match,
                "decision_match": decision_match,
                "reasoning_quality": reasoning_quality,
                "methodology_keywords_found": float(meth_keywords_found),
                "decision_keywords_found": float(dec_keywords_found),
                "causal_indicators_found": float(causal_found),
            },
            diagnostics=(
                f"step_match={step_match:.2f} decision_match={decision_match:.2f} "
                f"reasoning_quality={reasoning_quality:.2f} "
                f"(meth_kw={meth_keywords_found}/{len(self._methodology_keywords)} "
                f"dec_kw={dec_keywords_found}/{len(self._decision_keywords)} "
                f"causal={causal_found}/5)"
            ),
        )

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a named section from the response."""
        # Try to find the section header and extract until the next section or end
        pattern = rf"{section_name}\s*:?\s*\n(.*?)(?=\n[A-Z_]+\s*:?\s*\n|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ResearchMethodologyReconstructionEnv(BatchEnvBase):
    """ResearchMethodologyReconstruction: reconstruct methodology from paper outcome.

    Batch-aware: N parallel attempts; reward = best methodology reconstruction.
    Trains deep research reasoning — understanding HOW researchers reached
    their results, not just WHAT the results are.
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
            problem_generator = research_methodology_reconstruction_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ResearchMethodologyVerifier(
            methodology_steps=problem.metadata["methodology_steps"],
            key_decisions=problem.metadata["key_decisions"],
            methodology_keywords=problem.metadata["methodology_keywords"],
            decision_keywords=problem.metadata["decision_keywords"],
            reasoning_trace=problem.metadata["reasoning_trace"],
        )

    def _check_format(self, response: str) -> float:
        has_meth = bool(re.search(r"METHODOLOGY\s*:", response, re.IGNORECASE))
        has_dec = bool(re.search(r"KEY_DECISIONS\s*:", response, re.IGNORECASE))
        has_reason = bool(re.search(r"REASONING_TRACE\s*:", response, re.IGNORECASE))
        if has_meth and has_dec and has_reason:
            return 1.0
        score = 0.0
        if has_meth:
            score += 0.33
        if has_dec:
            score += 0.33
        if has_reason:
            score += 0.34
        return score

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
        match = re.search(r"REASONING_TRACE\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
