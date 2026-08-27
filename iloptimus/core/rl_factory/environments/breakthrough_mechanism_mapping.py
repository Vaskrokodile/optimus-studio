"""
BreakthroughMechanismMapping: Identify the deep mechanism behind breakthroughs.

Environment concept:
  The model is given 2-3 related 2026 arxiv papers and must identify the DEEP
  MECHANISM — the structural insight that explains WHY each breakthrough works.
  This trains the highest level of research reasoning: not just understanding
  individual papers, but seeing the underlying principle that connects them.

  Why: a great researcher doesn't just read papers — they see patterns across
  papers. "Oh, these three papers all exploit the same insight: that X is
  transient/redundant/conditional." This environment trains that cross-paper
  mechanism abstraction using real 2026 papers with manually authored mechanism
  analyses.

  All reasoning traces are manually authored — no scripts generate them.

Verification:
  - Mechanism keywords matched
  - Cross-paper connections identified
  - Structural insight articulated
  - Mechanism reasoning is deep (not surface-level)

Reward design:
  mechanism_match * 0.35 + connection_match * 0.25 + insight_quality * 0.25 + depth * 0.15
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — groups of related 2026 papers sharing a deep mechanism
# ---------------------------------------------------------------------------

_MECHANISM_PROBLEMS = [
    {
        "mechanism_name": "Transience Exploitation",
        "mechanism_description": (
            "Several breakthroughs exploit the fact that certain computational "
            "artifacts are TRANSIENT — they exist only briefly and are never needed "
            "again. By recognizing this transience, these methods eliminate storage "
            "that everyone else assumed was necessary."
        ),
        "papers": [
            {
                "title": "FORGE: Fused On-Register Gradient Elimination",
                "arxiv_id": "2606.22932",
                "breakthrough": "Eliminated gradient memory pool by consuming gradients in registers immediately after computation.",
                "mechanism_connection": "Gradients are transient — computed once, consumed once by optimizer, never needed again. By fusing computation and consumption, the gradient never needs to be stored.",
            },
            {
                "title": "OASIS: Online Activation Subspace Learning",
                "arxiv_id": "2604.09406",
                "breakthrough": "Reduced activation memory by projecting onto evolving subspace, enabling gradients and optimizer states in the subspace.",
                "mechanism_connection": "Activations are transient in a different sense — they're needed for backward pass but then discarded. By recognizing they live in a low-dimensional subspace, OASIS stores only the projection, not the full activation.",
            },
            {
                "title": "Scroll: Context as an Environment",
                "arxiv_id": "2608.21690",
                "breakthrough": "Kept agent history outside model context as executable Session Environment, only projecting explicitly requested information into working view.",
                "mechanism_connection": "Context tokens are transient in the agent's attention — most history is never referenced again. By keeping it external and projecting on demand, Scroll avoids storing what won't be used.",
            },
        ],
        "structural_insight": (
            "The deep insight is that much of what we store in ML systems is "
            "transient — it exists for a specific consumer at a specific time, and "
            "is never needed again. Gradients are consumed by the optimizer. "
            "Activations are consumed by the backward pass. Context tokens are "
            "consumed by attention. In each case, the artifact is stored because "
            "of a scheduling assumption (store now, consume later), not because "
            "of a learning requirement. By fusing production and consumption, or "
            "by recognizing the low-dimensional subspace where the artifact lives, "
            "we can eliminate storage that was assumed necessary. The general "
            "principle: QUESTION WHAT MUST BE STORED. If something is consumed "
            "exactly once, it doesn't need to be stored — it can be streamed. If "
            "something lives in a low-dimensional manifold, it doesn't need full "
            "representation — it can be projected."
        ),
        "mechanism_keywords": ["transient", "store", "consume", "fuse", "subspace", "project", "stream", "schedule"],
        "connection_keywords": ["gradient", "activation", "context", "memory", "eliminate", "reduce"],
    },
    {
        "mechanism_name": "Conditional Activation",
        "mechanism_description": (
            "Several breakthroughs exploit the insight that interventions should be "
            "CONDITIONAL on the input, not globally applied. By activating mechanisms "
            "only when needed, these methods avoid the collateral damage of global "
            "interventions."
        ),
        "papers": [
            {
                "title": "CLEAR: Continuous Latent Adapter Routing",
                "arxiv_id": "2608.21278",
                "breakthrough": "Safety LoRA activated only on harmful inputs via learned gate, preserving utility on benign prompts.",
                "mechanism_connection": "Safety intervention is conditional — only harmful inputs trigger the adapter. Global safety tuning modifies the entire model, degrading benign-task performance. Conditional activation separates safety from utility.",
            },
            {
                "title": "Learning When to Think: Adaptive Reasoning",
                "arxiv_id": "2608.20256",
                "breakthrough": "Model chooses reasoning mode (NoThink/Short/Long) as first token, allocating compute conditionally based on problem difficulty.",
                "mechanism_connection": "Reasoning effort is conditional — easy problems get NoThink, hard problems get Long. Fixed budgets over-compute easy problems and under-compute hard ones. Conditional allocation matches compute to need.",
            },
            {
                "title": "Tripwire: Triggering Aligned Refusal via Safety Neurons",
                "arxiv_id": "2608.14392",
                "breakthrough": "Training-free defense that triggers refusal only when harmful input is detected, via statistically certified safety neurons.",
                "mechanism_connection": "Refusal is conditional — only harmful inputs trigger the safety neuron clamp. The model's native reasoning is untouched on benign inputs. The intervention leverages existing refusal pathways rather than suppressing computation.",
            },
        ],
        "structural_insight": (
            "The deep insight is that global interventions cause collateral damage. "
            "When you apply safety training to the entire model, you degrade math "
            "ability. When you apply long reasoning to every problem, you waste "
            "compute on easy ones. When you suppress harmful computation globally, "
            "you suppress benign computation too. The solution in each case is to "
            "make the intervention CONDITIONAL: a gate, a router, or a detector "
            "that activates the mechanism only when the input requires it. The "
            "general principle: SEPARATE CONCERNS BY CONDITION. Safety and utility "
            "are not opposed — they're orthogonal when safety is conditional. "
            "Reasoning and efficiency are not traded — they're matched when "
            "reasoning is adaptive. The key engineering challenge is building a "
            "reliable condition detector (gate, router, neuron identifier) that "
            "doesn't itself become a failure point."
        ),
        "mechanism_keywords": ["conditional", "gate", "activate", "input", "global", "intervention", "adaptive", "trigger"],
        "connection_keywords": ["safety", "reasoning", "refusal", "utility", "compute", "benign", "harmful"],
    },
    {
        "mechanism_name": "Self-Generated Supervision",
        "mechanism_description": (
            "Several breakthroughs exploit the model's own internal signals as "
            "supervision, eliminating the need for external reward models, critics, "
            "or human annotations. The model becomes its own teacher."
        ),
        "papers": [
            {
                "title": "SRPO: Self-Reflective Policy Optimization",
                "arxiv_id": "2608.23493",
                "breakthrough": "Model analyzes own failed trajectories, synthesizes reflection patches as dense training signal.",
                "mechanism_connection": "The model generates its own process supervision by reflecting on failures. No external PRM or critic needed — the model's self-reflection IS the reward signal.",
            },
            {
                "title": "uPRM: Unsupervised Process Reward Models",
                "arxiv_id": "2605.10158",
                "breakthrough": "Trains PRMs without human supervision using LLM next-token probabilities to identify first erroneous steps.",
                "mechanism_connection": "The model's own next-token distribution encodes implicit judgments about step correctness. By comparing distributions across trajectories, uPRM extracts process supervision from the model itself.",
            },
            {
                "title": "EP-GRPO: Entropy-Progress Aligned GRPO",
                "arxiv_id": "2605.04960",
                "breakthrough": "Extracts implicit process signals from policy divergence anchored to outcome advantages for token-level feedback.",
                "mechanism_connection": "The model's entropy patterns reveal which tokens are genuine decision points. By mining this intrinsic information flow, EP-GRPO generates dense guidance without external reward models.",
            },
        ],
        "structural_insight": (
            "The deep insight is that models already encode the supervision they "
            "need — it's just not in the form we usually look for. The model's "
            "next-token probabilities encode step-quality judgments. Its entropy "
            "patterns encode decision-point importance. Its self-reflection "
            "encodes error identification. In each case, the signal is IMPLICIT "
            "in the model's own computation, and the breakthrough is EXTRACTING "
            "it. This eliminates the expensive pipeline of training external "
            "reward models, collecting human annotations, or building separate "
            "critic networks. The general principle: THE MODEL KNOWS MORE THAN "
            "ITS OUTPUT REVEALS. The next-token distribution, the entropy "
            "landscape, the attention patterns — all of these contain information "
            "about the quality of reasoning that the final answer doesn't capture. "
            "The engineering challenge is designing extractors that pull the right "
            "signal from the right part of the computation without introducing "
            "noise. The risk is bootstrapping: if the model's self-generated "
            "supervision is wrong, it trains itself on noise. The solution is "
            "anchoring self-generated signals to outcome verification — the "
            "self-supervision guides, but the outcome validates."
        ),
        "mechanism_keywords": ["self", "internal", "implicit", "extract", "supervision", "own", "signal", "model"],
        "connection_keywords": ["reflection", "PRM", "entropy", "reward", "process", "token", "probability"],
    },
    {
        "mechanism_name": "Distributional Reframing",
        "mechanism_description": (
            "Several breakthroughs reframe a point-estimation problem as a "
            "distributional problem, revealing structure that was invisible when "
            "looking at scalar outputs."
        ),
        "papers": [
            {
                "title": "Representation-Aware Advantage Estimation (GraphAE)",
                "arxiv_id": "2606.10528",
                "breakthrough": "Used reward model hidden states (not just scalar outputs) for advantage estimation via graph propagation.",
                "mechanism_connection": "Scalar rewards discard the rich distributional information in RM hidden states. By treating the group as a graph in representation space, GraphAE captures relationships that scalar comparison misses.",
            },
            {
                "title": "The Distributional View of Knowledge Distillation",
                "arxiv_id": "2608.15215",
                "breakthrough": "Represented teacher as multi-temperature views with geometry-aware aggregation, revealing three laws of distillation.",
                "mechanism_connection": "Standard KD compares pointwise distributions via KL, which is blind to which wrong token gets mass. The distributional view with embedding-based ground costs captures the geometry of the output space.",
            },
            {
                "title": "SoftmaxGRPO: Softmax Advantage Group Estimation",
                "arxiv_id": "2608.09271",
                "breakthrough": "Replaced z-score normalization with temperature-scaled softmax advantages, preventing divergent weighting on easy prompts.",
                "mechanism_connection": "Z-score normalization treats the advantage as a scalar statistic, ignoring the distributional structure of rewards. Softmax weighting respects the distributional shape, keeping advantages bounded regardless of difficulty.",
            },
        ],
        "structural_insight": (
            "The deep insight is that reducing rich distributions to scalars loses "
            "information that matters. A scalar reward tells you 'this is better "
            "than that' but not 'these are similar in a different way than those.' "
            "A scalar KL divergence tells you 'these distributions differ' but not "
            "'they differ in the wrong-token assignment.' A scalar z-score tells "
            "you 'this is above average' but not 'the average is unstable.' By "
            "reframing each as a distributional problem — using hidden states, "
            "multi-temperature views, or softmax weighting — these methods recover "
            "the structural information that scalar reduction discards. The general "
            "principle: WHEN A SCALAR SEEMS SUFFICIENT, ASK WHAT IT'S HIDING. The "
            "scalar is always a projection of a higher-dimensional object, and the "
            "projection direction determines what information is lost. Choosing a "
            "better projection (or avoiding projection entirely) can reveal structure "
            "that makes the problem tractable. The engineering challenge is that "
            "distributional methods are computationally more expensive — graph "
            "propagation, transport-based aggregation, and softmax weighting all "
            "add overhead. The payoff is that they capture relationships that scalar "
            "methods structurally cannot."
        ),
        "mechanism_keywords": ["distribution", "scalar", "representation", "geometry", "projection", "structure", "hidden", "reframe"],
        "connection_keywords": ["reward", "distillation", "advantage", "KL", "softmax", "graph", "temperature"],
    },
    {
        "mechanism_name": "Adversarial Self-Improvement",
        "mechanism_description": (
            "Several breakthroughs use adversarial structures where the model "
            "generates challenges for itself, turning weaknesses into training "
            "signals through self-play or co-evolution."
        ),
        "papers": [
            {
                "title": "Seirenes: Adversarial Self-Play with Evolving Distractions",
                "arxiv_id": "2605.11636",
                "breakthrough": "Model constructs distracting contexts that expose its own blind spots, then learns to solve despite them.",
                "mechanism_connection": "The model's vulnerability to distraction becomes the training signal. By constructing adversarial distractors, the model discovers its own weaknesses and trains to overcome them.",
            },
            {
                "title": "Mendel Gödel Machine: Recursive Self-Improving Agents",
                "arxiv_id": "2608.07645",
                "breakthrough": "Cross-lineage hybridization where agents learn from each other's trajectories on the same task.",
                "mechanism_connection": "Different agent lineages expose different failure modes. By cross-pollinating solutions, agents discover improvements that single-lineage self-modification would miss.",
            },
            {
                "title": "Debate Training Reduces Reward Hacking in RLAIF",
                "arxiv_id": "2608.17776",
                "breakthrough": "Two-player adversarial game between generator and critic reduces reward hacking compared to single-player RLAIF.",
                "mechanism_connection": "The critic exposes the generator's reward-hacking strategies, turning the vulnerability into a training signal. The adversarial structure makes hacking harder because both players must be robust.",
            },
        ],
        "structural_insight": (
            "The deep insight is that a system's weaknesses are more informative "
            "than its strengths. When a model succeeds, you learn it CAN do "
            "something. When it fails, you learn WHAT IT CAN'T DO — and that's "
            "where improvement happens. Adversarial self-improvement operationalizes "
            "this: the system (or a copy of it) is tasked with finding its own "
            "weaknesses, then those weaknesses become training targets. The key "
            "design challenge is maintaining the adversarial pressure without "
            "collapse. In Seirenes, parameter sharing prevents the distractor from "
            "becoming too strong (it's limited by the same capability as the solver). "
            "In Mendel, cross-lineage hybridization prevents any single lineage from "
            "getting stuck. In debate, the critic's word limit prevents it from "
            "hacking the judge. The general principle: WEAKNESSES ARE TRAINING "
            "SIGNALS. Rather than avoiding failure modes, engineer systems that "
            "discover and exploit them. The risk is that the adversarial structure "
            "can be co-opted — if the 'adversary' becomes too weak, there's no "
            "pressure; if too strong, the system can't learn. The solution is "
            "co-evolutionary dynamics where both sides improve together, maintaining "
            "productive tension."
        ),
        "mechanism_keywords": ["adversarial", "weakness", "self-play", "co-evolution", "challenge", "failure", "pressure", "tension"],
        "connection_keywords": ["distraction", "lineage", "debate", "hacking", "critic", "generator", "hybridization"],
    },
    {
        "mechanism_name": "Spectral Structure Exploitation",
        "mechanism_description": (
            "Several breakthroughs exploit the spectral (eigenvalue/singular value) "
            "structure of neural network components, revealing that what appears "
            "high-dimensional actually lives in a low-dimensional spectral subspace."
        ),
        "papers": [
            {
                "title": "Spectral Compact Training: Permanent Truncated SVD",
                "arxiv_id": "2604.00733",
                "breakthrough": "Replaced dense weight matrices with permanent SVD factors, 199x memory reduction, never materializing dense matrix.",
                "mechanism_connection": "Weight matrices have low-rank spectral structure during training. By working in the factored form and retracting to Stiefel manifold, the method exploits this structure throughout training.",
            },
            {
                "title": "Intrinsic Structure: Spectral Identifiability for Interpretability",
                "arxiv_id": "2608.10172",
                "breakthrough": "Proved Koopman operator spectrum of forward pass is recoverable and identifiable, providing model-intrinsic fingerprint.",
                "mechanism_connection": "The forward pass, viewed as a dynamical system, has a spectral structure (Koopman operator) that is coordinate-free and identifiable. This spectral fingerprint is a property of the model, not the decomposition method.",
            },
            {
                "title": "PRAC: Principal-Random Subspace for Activation Compression",
                "arxiv_id": "2602.23111",
                "breakthrough": "Decomposed activations into principal (SVD) + random subspace, proving unbiased gradient estimator with minimum variance.",
                "mechanism_connection": "Activations have spectral structure — dominant singular values capture most information. By combining SVD (deterministic) with random projection (stochastic), PRAC exploits the spectrum while maintaining unbiased estimation.",
            },
        ],
        "structural_insight": (
            "The deep insight is that neural network components have rich spectral "
            "structure that is invisible when you look at them as dense matrices or "
            "unstructured activations. Weight matrices are approximately low-rank. "
            "The forward pass has a Koopman spectrum that characterizes its dynamics. "
            "Activations have a principal subspace that captures most variance. In "
            "each case, the spectral decomposition reveals that the 'high-dimensional' "
            "object actually lives in a much lower-dimensional spectral subspace, and "
            "working in that subspace is both more efficient and more informative. "
            "The general principle: LOOK AT THE SPECTRUM, NOT THE MATRIX. The "
            "eigenvalues and singular values tell you where the information is "
            "concentrated, and operating in the spectral domain preserves that "
            "information while reducing dimensionality. The engineering challenges "
            "are: (1) the spectrum changes during training, requiring adaptive "
            "methods (Stiefel retraction, online subspace tracking); (2) spectral "
            "methods add computational overhead (SVD, eigendecomposition) that must "
            "be amortized; (3) the low-rank assumption may not hold for all layers "
            "or all training stages. The payoff is that spectral methods provide "
            "theoretical guarantees (identifiability, unbiasedness) that unstructured "
            "methods cannot."
        ),
        "mechanism_keywords": ["spectral", "spectrum", "eigenvalue", "singular", "low-rank", "subspace", "SVD", "Koopman"],
        "connection_keywords": ["weight", "forward", "activation", "compression", "identifiable", "principal", "memory"],
    },
    {
        "mechanism_name": "Non-Stationarity Embrace",
        "mechanism_description": (
            "Several breakthroughs stop fighting non-stationarity in RL training "
            "and instead EMBRACE it — designing methods that adapt to the changing "
            "landscape rather than assuming a fixed target."
        ),
        "papers": [
            {
                "title": "Actor-Curator: Co-adaptive Curriculum Learning",
                "arxiv_id": "2602.20532",
                "breakthrough": "Neural curator formulates problem selection as non-stationary bandit, adapting curriculum as policy improves.",
                "mechanism_connection": "The optimal curriculum changes as the policy improves. Rather than using a fixed curriculum, Actor-Curator embraces non-stationarity by treating problem selection as a non-stationary bandit that adapts.",
            },
            {
                "title": "VCPO: Variance-Controlled Off-Policy RL",
                "arxiv_id": "2602.17616",
                "breakthrough": "Scales learning rate based on effective sample size to handle stale rollouts in async training.",
                "mechanism_connection": "In async RL, rollouts become stale as the policy updates. Rather than discarding stale data or ignoring the problem, VCPO adapts the learning rate based on how stale the data is.",
            },
            {
                "title": "Pursuit-Evasion Curriculum",
                "arxiv_id": "2608.16156",
                "breakthrough": "Adversarial difficulty positioning that maintains 50% capture rate, adapting as the agent improves.",
                "mechanism_connection": "The optimal difficulty changes as the agent learns. Rather than a fixed schedule, the curriculum adapts to maintain the frontier where the agent is challenged but not overwhelmed.",
            },
        ],
        "structural_insight": (
            "The deep insight is that non-stationarity is not a bug to be fixed "
            "but a feature to be exploited. In RL training, the landscape changes "
            "as the policy improves — what was hard becomes easy, what was useful "
            "becomes trivial. Methods that assume stationarity (fixed curricula, "
            "fixed learning rates, fixed difficulty) fight this change. Methods "
            "that embrace non-stationarity (adaptive curricula, ESS-based learning "
            "rates, pursuit-evasion frontiers) use the change as a signal. The "
            "general principle: WHEN THE TARGET MOVES, MOVE WITH IT. The engineering "
            "challenge is tracking the non-stationarity without adding too much "
            "overhead. Actor-Curator uses a bandit formulation. VCPO uses effective "
            "sample size. Pursuit-evasion uses capture rate. In each case, a "
            "lightweight signal tracks the changing landscape and adapts the method. "
            "The theoretical challenge is providing guarantees under non-stationarity "
            "— standard convergence proofs assume fixed distributions. The solution "
            "is using tools from non-stationary bandit theory and adaptive control."
        ),
        "mechanism_keywords": ["non-stationary", "adapt", "change", "dynamic", "track", "moving", "embrace", "frontier"],
        "connection_keywords": ["curriculum", "stale", "difficulty", "learning-rate", "capture", "bandit", "async"],
    },
    {
        "mechanism_name": "Falsification Over Verification",
        "mechanism_description": (
            "Several breakthroughs exploit the epistemological asymmetry between "
            "falsification and verification: it's easier to prove something is "
            "wrong than to prove it is right."
        ),
        "papers": [
            {
                "title": "Via Negativa for AI Alignment",
                "arxiv_id": "2603.16417",
                "breakthrough": "Theoretical framework showing negative constraints (what's wrong) are structurally superior to positive preferences (what's better).",
                "mechanism_connection": "Negative constraints are discrete and verifiable (is this wrong?). Positive preferences are continuous and context-dependent (what's best?). Falsification is easier than verification.",
            },
            {
                "title": "VERGE: Formal Refinement Engine for Verifiable Reasoning",
                "arxiv_id": "2601.20055",
                "breakthrough": "Neurosymbolic framework using SMT solvers to verify LLM outputs, with precise error localization via Minimal Correction Subsets.",
                "mechanism_connection": "Rather than verifying the entire output is correct, VERGE identifies the minimal subset that needs correction. Finding errors (falsification) is more tractable than proving correctness (verification).",
            },
            {
                "title": "SignCert-PO: Advantage Sign Robustness",
                "arxiv_id": "2604.02986",
                "breakthrough": "Down-weights completions where the advantage sign is fragile to RM perturbation, preventing reward hacking.",
                "mechanism_connection": "Rather than verifying the reward is correct, SignCert-PO checks if the advantage SIGN is robust. The sign (positive/negative) is a falsifiable property — if it flips under small perturbation, the signal is untrustworthy.",
            },
        ],
        "structural_insight": (
            "The deep insight is Popperian: you can never prove a theory is true, "
            "but you can prove it's false. In ML, this means: you can never verify "
            "a response is 'good' (the space of good is infinite and context-"
            "dependent), but you can verify it's 'not wrong' (the space of wrong "
            "is finite and checkable). Via Negativa applies this to alignment: "
            "prohibit specific bad behaviors rather than trying to specify all "
            "good ones. VERGE applies this to reasoning: find the minimal error "
            "rather than verifying the whole proof. SignCert-PO applies this to "
            "RL: check the sign is robust rather than verifying the reward is "
            "accurate. The general principle: PROVE WHAT'S WRONG, DON'T TRY TO "
            "PROVE WHAT'S RIGHT. This is more tractable (finite vs infinite "
            "search space), more robust (wrong is context-independent, right is "
            "context-dependent), and more aligned with how science actually works. "
            "The limitation is that falsification alone doesn't tell you what TO "
            "do — it only tells you what NOT to do. The solution is combining "
            "falsification with minimal positive guidance: prohibit the bad, "
            "suggest the good, let the model fill the gap."
        ),
        "mechanism_keywords": ["falsif", "negative", "wrong", "verify", "prohibit", "sign", "correction", "Popper"],
        "connection_keywords": ["alignment", "reasoning", "advantage", "constraint", "error", "robust", "reward"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def breakthrough_mechanism_mapping_generator(seed: int) -> Problem:
    """Generate a BreakthroughMechanismMapping problem."""
    rng = random.Random(seed)
    problem_data = rng.choice(_MECHANISM_PROBLEMS)

    papers_text = ""
    for i, paper in enumerate(problem_data["papers"], 1):
        papers_text += (
            f"### Paper {i}: {paper['title']}\n"
            f"**arXiv:** {paper['arxiv_id']}\n"
            f"**Breakthrough:** {paper['breakthrough']}\n\n"
        )

    prompt = (
        f"## Breakthrough Mechanism Mapping\n\n"
        f"You are given {len(problem_data['papers'])} papers from 2026 that share a "
        f"DEEP MECHANISM — a structural insight that explains why each breakthrough "
        f"works. Your task is to identify this mechanism.\n\n"
        f"### Papers\n{papers_text}\n"
        f"### Your Task\n"
        f"1. **MECHANISM**: What is the deep mechanism that connects these papers? "
        f"What structural insight do they all exploit?\n\n"
        f"2. **CONNECTIONS**: For each paper, explain how it instantiates the "
        f"mechanism. What specific aspect of the breakthrough exploits the insight?\n\n"
        f"3. **STRUCTURAL_INSIGHT**: Articulate the general principle. If you were "
        f"writing a position paper, what would the thesis be? How would you predict "
        f"which future papers would succeed based on this mechanism?\n\n"
        f"4. **MECHANISM_REASONING**: Explain your reasoning process. How did you "
        f"identify the common mechanism? What alternatives did you consider and reject?\n\n"
        f"Format your answer as:\n"
        f"MECHANISM:\n<the deep mechanism>\n\n"
        f"CONNECTIONS:\n- Paper 1: <how it exploits the mechanism>\n- Paper 2: <how>\n...\n\n"
        f"STRUCTURAL_INSIGHT:\n<the general principle and predictions>\n\n"
        f"MECHANISM_REASONING:\n<your reasoning process>"
    )

    difficulty = 0.85 + len(problem_data["papers"]) * 0.02
    difficulty = min(0.95, difficulty)

    return Problem(
        id=f"breakthrough_mechanism_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "mechanism_name": problem_data["mechanism_name"],
            "mechanism_description": problem_data["mechanism_description"],
            "papers": problem_data["papers"],
            "structural_insight": problem_data["structural_insight"],
            "mechanism_keywords": problem_data["mechanism_keywords"],
            "connection_keywords": problem_data["connection_keywords"],
        },
        token_budget=4096,
        source="breakthrough_mechanism_mapping_generator",
    )


breakthrough_mechanism_mapping_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class BreakthroughMechanismVerifier(Verifier):
    """Verify a breakthrough mechanism mapping.

    Checks:
    1. Mechanism keywords matched
    2. Per-paper connections identified
    3. Structural insight articulated
    4. Reasoning depth

    reward = mechanism_match * 0.35 + connection_match * 0.25 + insight_quality * 0.25 + depth * 0.15
    """

    def __init__(
        self,
        papers: list[dict],
        mechanism_keywords: list[str],
        connection_keywords: list[str],
        structural_insight: str,
    ):
        super().__init__()
        self._papers = papers
        self._mechanism_keywords = [k.lower() for k in mechanism_keywords]
        self._connection_keywords = [k.lower() for k in connection_keywords]
        self._structural_insight = structural_insight.lower()

    def verify(self, response: str) -> VerifierResult:
        response_lower = response.lower()

        # --- Mechanism ---
        mech_section = self._extract_section(response, "MECHANISM")
        mech_text = mech_section.lower() if mech_section else response_lower
        mech_kw_found = sum(1 for kw in self._mechanism_keywords if kw in mech_text)
        mechanism_match = mech_kw_found / len(self._mechanism_keywords) if self._mechanism_keywords else 0.0

        # --- Connections ---
        conn_section = self._extract_section(response, "CONNECTIONS")
        conn_text = conn_section.lower() if conn_section else response_lower
        conn_items = len(re.findall(r"^[-*]\s*(?:paper\s*)?\d", conn_section, re.MULTILINE | re.IGNORECASE))
        conn_kw_found = sum(1 for kw in self._connection_keywords if kw in conn_text)
        connection_match = (
            min(1.0, conn_items / max(1, len(self._papers))) * 0.5
            + (conn_kw_found / len(self._connection_keywords) if self._connection_keywords else 0.0) * 0.5
        )

        # --- Structural insight ---
        insight_section = self._extract_section(response, "STRUCTURAL_INSIGHT")
        insight_text = insight_section.lower() if insight_section else response_lower

        # Check for general principle language
        principle_indicators = [
            "general principle", "the key insight", "the deep insight",
            "in each case", "the pattern", "the common", "all exploit",
            "the fundamental", "the structural", "predict",
        ]
        principle_found = sum(1 for ind in principle_indicators if ind in insight_text)
        principle_score = min(1.0, principle_found / 4.0)

        insight_length = len(insight_section.split()) if insight_section else 0
        insight_length_score = min(1.0, insight_length / 80.0)

        # Check for prediction language (forward-looking)
        has_prediction = bool(re.search(r"predict|future|would|could|next", insight_text))
        prediction_score = 1.0 if has_prediction else 0.0

        insight_quality = principle_score * 0.4 + insight_length_score * 0.3 + prediction_score * 0.3

        # --- Reasoning depth ---
        reason_section = self._extract_section(response, "MECHANISM_REASONING")
        reason_text = reason_section.lower() if reason_section else response_lower

        reasoning_indicators = [
            "i considered", "alternatively", "at first", "then i realized",
            "the connection is", "i noticed", "the pattern", "rejected",
            "considered and rejected", "initially thought", "but then",
        ]
        reasoning_found = sum(1 for ind in reasoning_indicators if ind in reason_text)
        reasoning_score = min(1.0, reasoning_found / 3.0)

        reason_length = len(reason_section.split()) if reason_section else 0
        reason_length_score = min(1.0, reason_length / 60.0)

        depth = reasoning_score * 0.5 + reason_length_score * 0.5

        # --- Final score ---
        score = (
            mechanism_match * 0.35
            + connection_match * 0.25
            + insight_quality * 0.25
            + depth * 0.15
        )
        correct = score >= 0.55

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "mechanism_match": mechanism_match,
                "connection_match": connection_match,
                "insight_quality": insight_quality,
                "depth": depth,
                "mechanism_keywords_found": float(mech_kw_found),
                "connection_keywords_found": float(conn_kw_found),
                "principle_indicators_found": float(principle_found),
                "has_prediction": float(has_prediction),
                "reasoning_indicators_found": float(reasoning_found),
            },
            diagnostics=(
                f"mechanism={mechanism_match:.2f} connections={connection_match:.2f} "
                f"insight={insight_quality:.2f} depth={depth:.2f} "
                f"(mech_kw={mech_kw_found}/{len(self._mechanism_keywords)} "
                f"conn_kw={conn_kw_found}/{len(self._connection_keywords)} "
                f"principle={principle_found}/4 predict={has_prediction} "
                f"reasoning={reasoning_found}/3)"
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


class BreakthroughMechanismMappingEnv(BatchEnvBase):
    """BreakthroughMechanismMapping: identify deep mechanisms across papers.

    Batch-aware: N parallel attempts; reward = best mechanism identification.
    Trains the highest level of research reasoning — seeing the structural
    insight that connects multiple breakthroughs.
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
            problem_generator = breakthrough_mechanism_mapping_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return BreakthroughMechanismVerifier(
            papers=problem.metadata["papers"],
            mechanism_keywords=problem.metadata["mechanism_keywords"],
            connection_keywords=problem.metadata["connection_keywords"],
            structural_insight=problem.metadata["structural_insight"],
        )

    def _check_format(self, response: str) -> float:
        sections = ["MECHANISM", "CONNECTIONS", "STRUCTURAL_INSIGHT", "MECHANISM_REASONING"]
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
        match = re.search(r"STRUCTURAL_INSIGHT\s*:\s*(.+?)(?:\n[A-Z_]|\Z)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
