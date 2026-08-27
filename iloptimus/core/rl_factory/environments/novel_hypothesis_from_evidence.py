"""
NovelHypothesisFromEvidence: Synthesize new research hypotheses from evidence.

Environment concept:
  The model is given evidence from multiple 2026 arxiv papers — findings,
  results, and observations — and must synthesize a NOVEL research hypothesis
  that connects them in a way no individual paper proposed. This trains the
  creative reasoning that generates new research directions.

  Why: the hardest part of research is not reading papers or running experiments
  — it's having the insight that connects disparate findings into a new idea.
  This environment trains that creative synthesis using real evidence from 2026
  papers with manually authored novel hypotheses and reasoning traces.

  All reasoning traces are manually authored — no scripts generate them.

Verification:
  - Hypothesis is novel (not stated in any individual paper)
  - Hypothesis connects multiple evidence pieces
  - Hypothesis is testable and falsifiable
  - Reasoning chain is clear and creative

Reward design:
  novelty * 0.3 + connection * 0.25 + testability * 0.25 + reasoning_quality * 0.2
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — evidence clusters with manually authored novel hypotheses
# ---------------------------------------------------------------------------

_HYPOTHESIS_PROBLEMS = [
    {
        "cluster_name": "Self-knowledge extraction across modalities",
        "evidence": [
            {
                "source": "uPRM (2605.10158)",
                "finding": "LLM next-token probabilities can identify first erroneous steps in reasoning trajectories without any human supervision, achieving 15% absolute improvement over LLM-as-Judge on ProcessBench.",
            },
            {
                "source": "EP-GRPO (2605.04960)",
                "finding": "Token entropy patterns reveal which tokens are genuine decision points vs deterministic progress. High-entropy tokens are more informative for learning than low-entropy tokens.",
            },
            {
                "source": "GRPO-VPS (2604.20659)",
                "finding": "The model's conditional probability of the correct answer at intermediate steps encodes its confidence in partial progress — this can serve as dense process supervision.",
            },
            {
                "source": "LLM Reasoning Is Latent (2604.15726)",
                "finding": "LLM reasoning should be studied as latent-state trajectory formation, not as surface chain-of-thought. The primary object of reasoning is in latent states, not explicit tokens.",
            },
        ],
        "novel_hypothesis": (
            "If LLM reasoning lives in latent states (not surface CoT), and the "
            "model's own next-token probabilities encode step-quality judgments, "
            "then we can extract a UNIVERSAL process reward signal by tracking "
            "how the latent state trajectory's predictability changes at each step. "
            "Specifically: a step is 'good' if it increases the predictability of "
            "the correct answer's latent representation, and 'bad' if it decreases "
            "it. This would unify uPRM (next-token probability), EP-GRPO (entropy), "
            "and GRPO-VPS (conditional probability) into a single latent-space "
            "process reward that requires no external supervision and operates on "
            "the actual reasoning substrate (latent states) rather than surface "
            "tokens. The hypothesis predicts that latent-state predictability "
            "tracking will outperform all three methods individually because it "
            "captures the common signal they each approximate partially."
        ),
        "hypothesis_keywords": ["latent", "predictability", "process", "universal", "unify", "trajectory", "state"],
        "connection_keywords": ["uPRM", "entropy", "probability", "reasoning", "token", "step"],
        "testability": (
            "Testable by: (1) extracting latent states at each reasoning step via "
            "hooked forward passes, (2) measuring how the latent representation of "
            "the correct answer changes in predictability (e.g., via probing "
            "accuracy or cosine similarity), (3) using this as a process reward in "
            "GRPO and comparing against uPRM, EP-GRPO, and GRPO-VPS individually. "
            "Falsifiable if latent predictability tracking does not outperform the "
            "three individual methods."
        ),
        "reasoning_trace": (
            "The four papers each extract a different signal from the model's own "
            "computation: uPRM uses next-token probabilities, EP-GRPO uses entropy, "
            "GRPO-VPS uses conditional probability of the correct answer, and the "
            "latent reasoning paper tells us the actual reasoning happens in latent "
            "states. The novel insight is that all three signals (probability, "
            "entropy, conditional probability) are PROJECTIONS of the same "
            "underlying quantity: how the latent state trajectory's predictability "
            "evolves. Next-token probability is a projection of latent state onto "
            "the vocabulary space. Entropy measures the spread of that projection. "
            "Conditional probability measures the projection onto a specific answer. "
            "If we could measure predictability directly in latent space — before "
            "the projection onto tokens — we'd capture the common signal that all "
            "three methods partially observe. This is analogous to how PCA captures "
            "the common variance that individual features partially express. The "
            "hypothesis is creative because it takes a finding from interpretability "
            "(reasoning is latent) and applies it to RL training (process rewards "
            "should operate on latent states, not tokens). The testability comes "
            "from the fact that latent states are extractable via hooked forward "
            "passes, and the comparison against three existing methods provides "
            "clear baselines."
        ),
    },
    {
        "cluster_name": "Conditional computation as universal alignment primitive",
        "evidence": [
            {
                "source": "CLEAR (2608.21278)",
                "finding": "Conditional safety adapter activation via learned gate preserves utility while maintaining safety. ASR 32.3% -> 0.5%, 7.1pp higher GSM8K than global tuning.",
            },
            {
                "source": "Learning When to Think (2608.20256)",
                "finding": "First-token routing between NoThink/Short/Long modes enables 41% token reduction with minimal accuracy loss. Modes emerge without collapsing.",
            },
            {
                "source": "Tripwire (2608.14392)",
                "finding": "Safety neurons identified via statistical testing can trigger refusal conditionally, with only 0.5-5.3% utility drop.",
            },
            {
                "source": "Steering Vector Fields (2602.01654)",
                "finding": "Context-dependent steering (vector fields, not static vectors) provides stronger control because the effective direction varies with activation context.",
            },
        ],
        "novel_hypothesis": (
            "Conditional computation is not just a technique for safety or efficiency "
            "— it is the universal primitive for ALL alignment interventions. The "
            "hypothesis: any alignment objective (safety, helpfulness, honesty, "
            "creativity) can be implemented as a CONDITIONAL ROUTING problem where "
            "a learned gate determines which adapter/expert/mode to activate based "
            "on the input's position in a multi-dimensional alignment space. "
            "Specifically, we can build a SINGLE unified routing layer that "
            "simultaneously handles safety (CLEAR), reasoning depth (Learning When "
            "to Think), refusal (Tripwire), and behavioral steering (SVF) — because "
            "all four are instances of the same pattern: conditional activation of "
            "a specialized computation. The hypothesis predicts that a unified "
            "conditional routing layer will outperform separate conditional "
            "mechanisms because it can learn the INTERACTIONS between alignment "
            "dimensions (e.g., a query that is both safety-relevant AND requires "
            "deep reasoning needs both adapters, and a unified router can learn "
            "this joint activation pattern)."
        ),
        "hypothesis_keywords": ["conditional", "routing", "unified", "alignment", "primitive", "gate", "joint", "interaction"],
        "connection_keywords": ["safety", "reasoning", "refusal", "steering", "adapter", "mode", "activation"],
        "testability": (
            "Testable by: (1) defining a multi-dimensional alignment space (safety, "
            "reasoning depth, honesty, creativity), (2) training a unified routing "
            "layer that activates multiple adapters simultaneously based on input "
            "position in this space, (3) comparing against separate conditional "
            "mechanisms for each dimension. Falsifiable if the unified router does "
            "not outperform separate mechanisms, particularly on inputs requiring "
            "multiple alignment dimensions simultaneously."
        ),
        "reasoning_trace": (
            "The four papers each implement conditional computation for a different "
            "purpose: CLEAR for safety, Learning When to Think for efficiency, "
            "Tripwire for refusal, SVF for steering. But they all share the same "
            "structure: a gate/detector/router determines whether to activate a "
            "specialized computation based on the input. The novel insight is that "
            "these are not four different techniques — they're four instances of "
            "the SAME technique applied to different alignment dimensions. If "
            "that's true, then a unified routing layer should work because the "
            "underlying problem is the same: given an input, determine which "
            "specialized computations to activate. The creative leap is recognizing "
            "that alignment dimensions INTERACT: a query about dangerous chemistry "
            "that requires deep reasoning needs BOTH safety AND reasoning-depth "
            "conditioning. Separate mechanisms can't learn this interaction because "
            "they make independent decisions. A unified router can learn joint "
            "activation patterns. The hypothesis is testable because we can "
            "construct evaluation sets that require multiple alignment dimensions "
            "simultaneously and measure whether the unified router handles them "
            "better than separate mechanisms. The risk is that the alignment space "
            "is too high-dimensional for a single router to learn — but SVF's "
            "vector fields suggest that context-dependent routing is tractable."
        ),
    },
    {
        "cluster_name": "Spectral structure as training signal",
        "evidence": [
            {
                "source": "Spectral Compact Training (2604.00733)",
                "finding": "Weight matrices have low-rank spectral structure during training. Permanent SVD factoring with Stiefel retraction achieves 199x memory reduction. All ranks converge to same loss floor.",
            },
            {
                "source": "Intrinsic Structure (2608.10172)",
                "finding": "The Koopman operator spectrum of the forward pass is identifiable and recoverable at M^(-1/2) rate. The spectrum is a coordinate-free property of the model.",
            },
            {
                "source": "PRAC (2602.23111)",
                "finding": "Activations decompose into principal (SVD) + random subspace. The principal subspace captures dominant information, and combining with random projection gives unbiased gradient estimation.",
            },
            {
                "source": "OASIS (2604.09406)",
                "finding": "Online tracking of activation subspace during training enables gradients and optimizer states to be maintained in the subspace. Projection-aware optimizer transports states across subspace updates.",
            },
        ],
        "novel_hypothesis": (
            "The spectral structure of a neural network is not just a compression "
            "opportunity — it is a TRAINING SIGNAL. The hypothesis: the rate of "
            "change of the spectral structure (how the singular value distribution "
            "evolves during training) predicts training health and can be used as "
            "a meta-learning signal. Specifically: when the spectral structure is "
            "STABLE (singular values not changing much), the model is in a "
            "productive learning phase; when it's UNSTABLE (rapid spectral shifts), "
            "the model is either about to break through a capability barrier or "
            "about to diverge. By tracking spectral stability, we can build a "
            "spectral-aware learning rate scheduler that accelerates during stable "
            "periods and decelerates during unstable ones — the opposite of "
            "standard schedulers that use loss curves. The hypothesis predicts "
            "that spectral stability is a LEADING indicator (it changes before "
            "the loss does), making it more useful than loss-based scheduling."
        ),
        "hypothesis_keywords": ["spectral", "stability", "training", "signal", "singular", "rate", "predict", "leading"],
        "connection_keywords": ["weight", "Koopman", "activation", "subspace", "SVD", "low-rank", "gradient"],
        "testability": (
            "Testable by: (1) tracking the singular value distribution of weight "
            "matrices at each training step, (2) computing a spectral stability "
            "metric (e.g., rate of change of top-k singular values), (3) correlating "
            "spectral stability with training outcomes (loss, capability acquisition, "
            "divergence), (4) building a spectral-aware learning rate scheduler and "
            "comparing against loss-based schedulers. Falsifiable if spectral "
            "stability does not predict training outcomes better than loss curves."
        ),
        "reasoning_trace": (
            "The four papers all exploit spectral structure but for different "
            "purposes: compression (Spectral Compact), interpretability (Intrinsic "
            "Structure), memory efficiency (PRAC, OASIS). The novel insight is that "
            "they're all observing the SAME phenomenon — the spectral structure of "
            "neural networks — but at different levels (weights, forward pass, "
            "activations). If the spectral structure is a fundamental property "
            "(Intrinsic Structure says it's identifiable), then its EVOLUTION during "
            "training should be informative. The creative leap is connecting the "
            "static spectral observations to dynamic training dynamics: if the "
            "spectrum is stable, the model has found a productive representation; "
            "if it's shifting, something is changing. This is analogous to how "
            "seismologists use spectral analysis of vibrations to predict earthquakes "
            "— the spectrum changes before the big event. The 'leading indicator' "
            "prediction is the key testable claim: if spectral stability changes "
            "BEFORE the loss does, it's a more useful signal for scheduling. The "
            "Stiefel retraction in Spectral Compact Training is relevant here — it "
            "MAINTAINS spectral structure, suggesting that spectral stability is "
            "important for training health. The OASIS projection-aware optimizer "
            "transports states across subspace updates, which is a form of spectral "
            "tracking — but it's reactive, not predictive. The hypothesis says we "
            "should be PROACTIVE: use spectral stability to predict and prevent "
            "training problems before they manifest in the loss."
        ),
    },
    {
        "cluster_name": "Adversarial structure as alignment guarantee",
        "evidence": [
            {
                "source": "Debate Training Reduces Reward Hacking (2608.17776)",
                "finding": "Two-player adversarial debate between generator and critic reduces reward hacking. Critic word limits balance the game. 45% performance gap recovered.",
            },
            {
                "source": "Seirenes (2605.11636)",
                "finding": "Adversarial self-play where model constructs distractors for itself improves reasoning robustness. +10.2 points across 7 benchmarks. Distractors transfer to GPT/Gemini.",
            },
            {
                "source": "Calibrated Collective Oversight (2605.28807)",
                "finding": "Aggregating diverse auxiliary scorers into a conservatism penalty with conformal calibration provides finite-time bounds on undesirable outcomes.",
            },
            {
                "source": "Knowledge Divergence and Debate (2603.05293)",
                "finding": "Debate's value is characterized by principal angles between models' representation subspaces. Phase transition from quadratic to linear regime where debate becomes essential.",
            },
        ],
        "novel_hypothesis": (
            "Adversarial structure provides alignment guarantees that are "
            "PROVABLY STRONGER than any single-agent method, and the strength of "
            "the guarantee is determined by the KNOWLEDGE DIVERGENCE between the "
            "adversarial agents. The hypothesis: we can build a tunable alignment "
            "guarantee by controlling the knowledge divergence between generator "
            "and critic. Specifically: if we train the critic on a DIFFERENT data "
            "distribution than the generator (creating controlled knowledge "
            "divergence), the adversarial game provides a alignment guarantee "
            "proportional to the divergence. Low divergence = weak guarantee "
            "(critic knows what generator knows). High divergence = strong guarantee "
            "(critic catches what generator misses). This unifies debate training "
            "(adversarial structure), Seirenes (self-play with shared parameters = "
            "low divergence), CCO (diverse scorers = high divergence), and knowledge "
            "divergence theory (principal angles characterize guarantee strength). "
            "The hypothesis predicts an OPTIMAL divergence: too low and the critic "
            "can't catch errors; too high and the critic can't understand the "
            "generator's outputs. The optimal point is where the critic's knowledge "
            "is complementary, not overlapping or alien."
        ),
        "hypothesis_keywords": ["adversarial", "guarantee", "divergence", "knowledge", "tunable", "optimal", "complementary", "proportional"],
        "connection_keywords": ["debate", "self-play", "oversight", "critic", "generator", "scorer", "calibration"],
        "testability": (
            "Testable by: (1) training generators and critics with controlled "
            "knowledge divergence (different training data, different model sizes, "
            "different architectures), (2) measuring alignment guarantee strength "
            "(reward hacking rate, safety violation rate) as a function of "
            "divergence, (3) fitting the predicted optimal-divergence curve. "
            "Falsifiable if guarantee strength is not proportional to divergence, "
            "or if there is no optimal point (monotonically increasing or decreasing)."
        ),
        "reasoning_trace": (
            "The four papers all use adversarial structure but with different "
            "divergence levels: debate training uses different players (moderate "
            "divergence), Seirenes uses parameter-shared self-play (low divergence), "
            "CCO uses diverse scorers (high divergence), and the knowledge divergence "
            "paper provides the theory (principal angles characterize divergence). "
            "The novel insight is that these are all points on a SPECTRUM of "
            "adversarial divergence, and the alignment guarantee strength is a "
            "function of where you sit on that spectrum. The creative leap is "
            "making the divergence TUNABLE: rather than accepting whatever divergence "
            "your architecture gives you, deliberately control it by choosing how "
            "different the critic's training is from the generator's. This turns "
            "alignment from a binary property (aligned or not) into a continuous "
            "one (how strong is the guarantee), which is more useful for deployment. "
            "The 'optimal divergence' prediction is the key testable claim: if "
            "there's a sweet spot where the critic is complementary but not alien, "
            "that's the practical design point. The knowledge divergence paper's "
            "phase transition (quadratic to linear) suggests there IS a critical "
            "point — the hypothesis predicts this critical point is the optimal "
            "divergence for alignment. The risk is that the optimal divergence is "
            "task-dependent, making it hard to calibrate in practice."
        ),
    },
    {
        "cluster_name": "Falsification as universal verification",
        "evidence": [
            {
                "source": "Via Negativa (2603.16417)",
                "finding": "Negative constraints are structurally superior to positive preferences because falsification (is this wrong?) is discrete and verifiable while verification (is this best?) is continuous and context-dependent.",
            },
            {
                "source": "VERGE (2601.20055)",
                "finding": "SMT solver verification with Minimal Correction Subsets provides precise error localization. 18.7% average uplift via iterative refinement guided by formal error finding.",
            },
            {
                "source": "SignCert-PO (2604.02986)",
                "finding": "Down-weighting completions with fragile advantage signs (certified sign-preservation radius) reduces reward hacking. The sign is a falsifiable property.",
            },
            {
                "source": "CHERRL (2606.04923)",
                "finding": "Injecting known biases into LLM-as-Judge creates a controllable hacking environment where hacking onset can be precisely identified and studied.",
            },
        ],
        "novel_hypothesis": (
            "All reward hacking is a falsification failure: the reward signal "
            "cannot distinguish between 'genuinely good' and 'appears good but "
            "isn't.' The hypothesis: we can eliminate reward hacking entirely by "
            "building reward functions that are FALSIFICATION-FIRST — they check "
            "for specific failure modes rather than measuring positive quality. "
            "Specifically, instead of R(response) = quality_score, we use "
            "R(response) = 1 - max(failure_mode_penalty(response)) for a "
            "COMPREHENSIVE set of failure modes. The reward is 1 only if NO "
            "failure mode is detected. This makes the reward a conjunction of "
            "falsification checks, each of which is discrete and verifiable. The "
            "hypothesis predicts that falsification-first rewards will be "
            "structurally immune to hacking because hacking requires finding a "
            "response that passes ALL falsification checks, which is equivalent "
            "to being genuinely good. The key insight is that CHERRL's "
            "controllable bias injection gives us a way to ENUMERATE failure "
            "modes systematically, making the comprehensive set tractable."
        ),
        "hypothesis_keywords": ["falsification", "reward", "hacking", "failure", "conjunction", "immune", "comprehensive", "penalty"],
        "connection_keywords": ["negative", "constraint", "verification", "sign", "bias", "judge", "error"],
        "testability": (
            "Testable by: (1) using CHERRL's bias injection to enumerate common "
            "failure modes for a task, (2) building falsification-first rewards "
            "as conjunctions of failure-mode checks, (3) training with these "
            "rewards and measuring hacking rate vs standard positive rewards. "
            "Falsifiable if falsification-first rewards do not reduce hacking, "
            "or if the comprehensive failure-mode set is too expensive to evaluate."
        ),
        "reasoning_trace": (
            "The four papers all relate to falsification but at different levels: "
            "Via Negativa provides the theory (falsification > verification), VERGE "
            "provides the tool (SMT-based error finding), SignCert-PO provides the "
            "RL application (sign robustness), and CHERRL provides the experimental "
            "infrastructure (controlled bias injection). The novel insight is that "
            "they can be COMPOSED into a complete anti-hacking framework: Via "
            "Negativa says use falsification, VERGE says how to find errors, "
            "SignCert-PO says how to use falsification in RL, and CHERRL says how "
            "to enumerate what to falsify. The creative leap is the conjunction "
            "structure: R = 1 - max(penalty) means the reward is 1 ONLY if every "
            "single failure mode check passes. This is structurally different from "
            "weighted sum rewards where one strong dimension can compensate for "
            "weakness in another. In a conjunction, ALL dimensions must pass — "
            "there's no compensation. This is why it's immune to hacking: hacking "
            "requires finding a shortcut that passes the reward, but in a "
            "conjunction of falsification checks, there IS no shortcut — you have "
            "to genuinely pass every check. The risk is evaluation cost: a "
            "comprehensive set of failure modes might be expensive to check at "
            "every training step. The solution is caching and incremental "
            "evaluation — only re-check failure modes that could have changed."
        ),
    },
    {
        "cluster_name": "Non-stationarity as curriculum signal",
        "evidence": [
            {
                "source": "Actor-Curator (2602.20532)",
                "finding": "Non-stationary bandit formulation of curriculum selection achieves 28.6% gain on AIME. The curator adapts as the policy improves.",
            },
            {
                "source": "Curriculum RL Beyond Base Model (2606.22317)",
                "finding": "Standard RLVR reallocates sampling probabilities among existing trajectories (improves pass@1 but not pass@k). Boundary-aware curriculum expands reasoning capacity.",
            },
            {
                "source": "Autocurriculum Theory (2603.18325)",
                "finding": "Autocurriculum requires exponentially fewer demonstrations than non-adaptive fine-tuning. Theoretical proof via boosting and learning from counterexamples.",
            },
            {
                "source": "Pursuit-Evasion Curriculum (2608.16156)",
                "finding": "Adversarial difficulty positioning at 50% capture rate creates a frontier that adapts as the agent improves, maintaining productive challenge.",
            },
        ],
        "novel_hypothesis": (
            "The non-stationarity of the learning landscape is itself the optimal "
            "curriculum signal, and we can extract it WITHOUT any external curator. "
            "The hypothesis: the model's OWN performance trajectory contains "
            "sufficient information to construct an optimal curriculum, because "
            "the rate of improvement on each problem type reveals which problems "
            "are at the learning frontier. Specifically: problems where the model "
            "is rapidly improving are at the frontier (productive); problems where "
            "it's stagnant are either too easy (mastered) or too hard (beyond "
            "reach). By tracking per-problem-type improvement rates and selecting "
            "problems at the maximum improvement rate, the model constructs its "
            "OWN curriculum without a separate curator network. This unifies "
            "Actor-Curator (bandit selection), boundary-aware curriculum (frontier "
            "targeting), autocurriculum theory (exponential efficiency), and "
            "pursuit-evasion (50% capture rate = maximum improvement rate). The "
            "hypothesis predicts that self-extracted curriculum from improvement "
            "rate tracking will match Actor-Curator's performance without the "
            "curator's training cost, because the improvement rate IS the signal "
            "the curator is trying to estimate."
        ),
        "hypothesis_keywords": ["non-stationarity", "curriculum", "improvement", "rate", "frontier", "self", "trajectory", "extract"],
        "connection_keywords": ["bandit", "boundary", "autocurriculum", "pursuit", "capture", "pass@k", "adaptive"],
        "testability": (
            "Testable by: (1) tracking per-problem-type improvement rates during "
            "standard RL training, (2) constructing a curriculum that samples "
            "problems at the maximum improvement rate, (3) comparing against "
            "Actor-Curator (with curator) and uniform sampling (no curriculum). "
            "Falsifiable if self-extracted curriculum does not match Actor-Curator "
            "performance, or if improvement rate is not a good proxy for the "
            "curator's selection signal."
        ),
        "reasoning_trace": (
            "The four papers all deal with adaptive curricula but require different "
            "external components: Actor-Curator needs a neural curator, boundary-"
            "aware curriculum needs pass@k sampling, autocurriculum needs a teacher, "
            "and pursuit-evasion needs an adversary. The novel insight is that the "
            "SIGNAL they're all trying to estimate — which problems are at the "
            "learning frontier — is already present in the model's own performance "
            "trajectory. You don't need a curator to tell you which problems are "
            "productive; you just need to track how fast the model is improving on "
            "each problem type. The creative leap is recognizing that improvement "
            "rate is a SUFFICIENT statistic for curriculum value: problems with "
            "high improvement rates are at the frontier (productive), problems with "
            "low rates are either mastered or unreachable. This is essentially "
            "the pursuit-evasion insight (50% capture rate = maximum learning) "
            "applied to the model's own trajectory rather than an external "
            "adversary. The connection to autocurriculum theory is that improvement-"
            "rate tracking is a form of counterexample-focused learning (focus on "
            "problems where you're changing = where you're being challenged). The "
            "hypothesis is testable because improvement rates are trivially "
            "measurable during training — no extra model needed. The risk is that "
            "improvement rates are noisy (single-problem improvement can be "
            "negative due to optimization noise), requiring smoothing or Bayesian "
            "estimation."
        ),
    },
    {
        "cluster_name": "Transience as design principle for efficient systems",
        "evidence": [
            {
                "source": "FORGE (2606.22932)",
                "finding": "Gradients are transient (computed once, consumed once). Fusing computation and consumption eliminates gradient memory. 16-33% memory reduction, 1.5x faster.",
            },
            {
                "source": "Scroll (2608.21690)",
                "finding": "Context tokens are transient in agent attention. Keeping history external and projecting on demand achieves 86.7% on LOCA256K, +37.4 points over best prior.",
            },
            {
                "source": "FlashOptim (2602.23349)",
                "finding": "Optimizer states can be quantized to 8 bits with companding functions, reducing AdamW from 16 to 7 bytes per parameter. States are 'semi-transient' — accessed once per step.",
            },
            {
                "source": "LazyTrain (2608.11919)",
                "finding": "Checkpoint selection, activation placement, and communication overlap formulated as mixed-integer scheduling problem. 1.24x TFLOPS improvement from zero-waste scheduling.",
            },
        ],
        "novel_hypothesis": (
            "Every component of an ML system has a TRANSIENCE PROFILE — a "
            "characterization of how long each artifact lives and how many times "
            "it's accessed. By analyzing the transience profile, we can determine "
            "the optimal storage strategy for each component: artifacts accessed "
            "exactly once should be streamed (FORGE), artifacts accessed rarely "
            "should be externalized (Scroll), artifacts accessed every step but "
            "with low precision needs should be quantized (FlashOptim), and "
            "artifacts with complex access patterns should be scheduled (LazyTrain). "
            "The hypothesis: a TRANSIENCE-AWARE COMPILER that automatically "
            "analyzes the transience profile of each tensor and assigns the "
            "optimal storage strategy will achieve the combined memory savings of "
            "all four methods without manual engineering. The compiler would "
            "analyze the computation graph, determine access frequency and "
            "precision requirements for each tensor, and automatically apply "
            "streaming, externalization, quantization, or scheduling. The "
            "hypothesis predicts that transience-aware compilation will achieve "
            "near-optimal memory efficiency because it addresses the ROOT CAUSE "
            "(unnecessary storage of transient artifacts) rather than individual "
            "symptoms."
        ),
        "hypothesis_keywords": ["transience", "profile", "compiler", "storage", "stream", "externalize", "quantize", "schedule"],
        "connection_keywords": ["gradient", "context", "optimizer", "checkpoint", "memory", "scheduling", "activation"],
        "testability": (
            "Testable by: (1) defining a transience profile representation for "
            "computation graphs (access count, frequency, precision requirement), "
            "(2) implementing a compiler pass that assigns storage strategies "
            "based on the profile, (3) measuring memory efficiency vs manual "
            "application of FORGE + Scroll + FlashOptim + LazyTrain. Falsifiable "
            "if the compiler does not achieve combined savings, or if the "
            "transience profile is insufficient to determine optimal storage."
        ),
        "reasoning_trace": (
            "The four papers each identify a specific transience pattern: gradients "
            "are accessed once (FORGE), context tokens are rarely re-accessed "
            "(Scroll), optimizer states are accessed every step but tolerate "
            "quantization (FlashOptim), and activations have complex access "
            "patterns requiring scheduling (LazyTrain). The novel insight is that "
            "these are all instances of the same meta-problem: given an artifact "
            "with a specific access pattern, what's the optimal storage strategy? "
            "The creative leap is building a COMPILER that automates this decision. "
            "Current ML systems require manual application of each technique — "
            "you decide to use FORGE for gradients, Scroll for context, etc. A "
            "transience-aware compiler would make these decisions automatically by "
            "analyzing the computation graph. This is analogous to how register "
            "allocation compilers analyze variable lifetimes to determine which "
            "variables go in registers vs memory. The transience profile is the "
            "ML equivalent of variable lifetime. The hypothesis is testable because "
            "computation graphs are analyzable and the four strategies cover the "
            "main storage options. The risk is that the optimal strategy depends "
            "on hardware-specific factors (cache sizes, bandwidth) that are hard "
            "to model in a compiler. The solution is hardware-aware compilation "
            "that takes the target GPU's specifications as input."
        ),
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def novel_hypothesis_from_evidence_generator(seed: int) -> Problem:
    """Generate a NovelHypothesisFromEvidence problem."""
    rng = random.Random(seed)
    problem_data = rng.choice(_HYPOTHESIS_PROBLEMS)

    evidence_text = ""
    for i, ev in enumerate(problem_data["evidence"], 1):
        evidence_text += f"### Evidence {i}\n**Source:** {ev['source']}\n**Finding:** {ev['finding']}\n\n"

    prompt = (
        f"## Novel Hypothesis from Evidence\n\n"
        f"You are given {len(problem_data['evidence'])} pieces of evidence from "
        f"2026 AI research papers. Your task is to synthesize a NOVEL research "
        f"hypothesis that connects them in a way no individual paper proposed.\n\n"
        f"### Evidence Cluster: {problem_data['cluster_name']}\n\n"
        f"{evidence_text}\n"
        f"### Your Task\n"
        f"1. **HYPOTHESIS**: State a novel, specific, testable hypothesis that "
        f"connects the evidence. This should NOT be stated in any individual paper "
        f"— it should be a NEW insight that emerges from connecting them.\n\n"
        f"2. **CONNECTIONS**: For each piece of evidence, explain how it supports "
        f"or relates to your hypothesis. What specific finding connects?\n\n"
        f"3. **TESTABILITY**: How would you test this hypothesis? What experiment "
        f"would confirm or falsify it? What are the specific predictions?\n\n"
        f"4. **REASONING**: Walk through your creative process. How did you "
        f"connect the evidence? What insight emerged? What alternative hypotheses "
        f"did you consider and reject?\n\n"
        f"Format your answer as:\n"
        f"HYPOTHESIS:\n<your novel hypothesis>\n\n"
        f"CONNECTIONS:\n- Evidence 1: <connection>\n- Evidence 2: <connection>\n...\n\n"
        f"TESTABILITY:\n<how to test, predictions, falsification conditions>\n\n"
        f"REASONING:\n<your creative process and alternative hypotheses>"
    )

    difficulty = 0.9
    return Problem(
        id=f"novel_hypothesis_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "cluster_name": problem_data["cluster_name"],
            "evidence": problem_data["evidence"],
            "novel_hypothesis": problem_data["novel_hypothesis"],
            "testability": problem_data["testability"],
            "reasoning_trace": problem_data["reasoning_trace"],
            "hypothesis_keywords": problem_data["hypothesis_keywords"],
            "connection_keywords": problem_data["connection_keywords"],
        },
        token_budget=4096,
        source="novel_hypothesis_from_evidence_generator",
    )


novel_hypothesis_from_evidence_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class NovelHypothesisVerifier(Verifier):
    """Verify a novel hypothesis from evidence.

    Checks:
    1. Novelty (hypothesis is not stated in any individual paper)
    2. Connections to evidence
    3. Testability and falsifiability
    4. Reasoning quality (creative process)

    reward = novelty * 0.3 + connection * 0.25 + testability * 0.25 + reasoning_quality * 0.2
    """

    def __init__(
        self,
        evidence: list[dict],
        hypothesis_keywords: list[str],
        connection_keywords: list[str],
        novel_hypothesis: str,
    ):
        super().__init__()
        self._evidence = evidence
        self._hypothesis_keywords = [k.lower() for k in hypothesis_keywords]
        self._connection_keywords = [k.lower() for k in connection_keywords]
        self._novel_hypothesis = novel_hypothesis.lower()
        # Extract individual paper findings for novelty check
        self._paper_findings = [ev["finding"].lower() for ev in evidence]

    def verify(self, response: str) -> VerifierResult:
        response_lower = response.lower()

        # --- Hypothesis ---
        hyp_section = self._extract_section(response, "HYPOTHESIS")
        hyp_text = hyp_section.lower() if hyp_section else response_lower

        # Keyword match
        hyp_kw_found = sum(1 for kw in self._hypothesis_keywords if kw in hyp_text)
        keyword_score = hyp_kw_found / len(self._hypothesis_keywords) if self._hypothesis_keywords else 0.0

        # Novelty check: hypothesis should contain synthesis language
        novelty_indicators = [
            "if", "then", "because", "this means", "the insight",
            "we can", "by connecting", "unify", "combine",
            "the hypothesis", "predict", "specifically",
        ]
        novelty_found = sum(1 for ind in novelty_indicators if ind in hyp_text)
        novelty_language = min(1.0, novelty_found / 4.0)

        # Check hypothesis is not just restating a paper finding
        # (heuristic: if >60% of hypothesis words appear in a single finding, it's not novel)
        # Trivially short responses get zero originality — they're not novel hypotheses.
        hyp_words = set(hyp_text.split())
        if len(hyp_words) < 15:
            originality = 0.0
        else:
            max_overlap = 0.0
            for finding in self._paper_findings:
                finding_words = set(finding.split())
                if hyp_words:
                    overlap = len(hyp_words & finding_words) / len(hyp_words)
                    max_overlap = max(max_overlap, overlap)
            originality = 1.0 - min(1.0, max_overlap / 0.6)  # 1.0 if no overlap, 0.0 if >60% overlap

        novelty = keyword_score * 0.3 + novelty_language * 0.3 + originality * 0.4

        # --- Connections ---
        conn_section = self._extract_section(response, "CONNECTIONS")
        conn_text = conn_section.lower() if conn_section else response_lower
        conn_items = len(re.findall(r"^[-*]\s*(?:evidence\s*)?\d", conn_section, re.MULTILINE | re.IGNORECASE))
        conn_kw_found = sum(1 for kw in self._connection_keywords if kw in conn_text)
        connection = (
            min(1.0, conn_items / max(1, len(self._evidence))) * 0.5
            + (conn_kw_found / len(self._connection_keywords) if self._connection_keywords else 0.0) * 0.5
        )

        # --- Testability ---
        test_section = self._extract_section(response, "TESTABILITY")
        test_text = test_section.lower() if test_section else response_lower

        testability_indicators = [
            "test", "experiment", "measure", "compare", "falsif",
            "if", "then", "predict", "baseline", "control",
        ]
        test_found = sum(1 for ind in testability_indicators if ind in test_text)
        testability_score = min(1.0, test_found / 5.0)

        # Falsifiability check
        has_falsification = bool(re.search(r"falsif|if.*not|if.*fail|would.*not|refute", test_text))
        has_prediction = bool(re.search(r"predict|expect|hypothes", test_text))
        falsifiability = (1.0 if has_falsification else 0.0) * 0.5 + (1.0 if has_prediction else 0.0) * 0.5

        test_length = len(test_section.split()) if test_section else 0
        test_length_score = min(1.0, test_length / 60.0)

        testability = testability_score * 0.4 + falsifiability * 0.3 + test_length_score * 0.3

        # --- Reasoning quality ---
        reason_section = self._extract_section(response, "REASONING")
        reason_text = reason_section.lower() if reason_section else response_lower

        creative_indicators = [
            "i noticed", "the insight", "the connection", "creative",
            "alternative", "considered", "rejected", "at first",
            "then i realized", "the leap", "the key", "novel",
        ]
        creative_found = sum(1 for ind in creative_indicators if ind in reason_text)
        creative_score = min(1.0, creative_found / 3.0)

        # Check for alternative hypotheses (shows creative process)
        has_alternatives = bool(re.search(r"alternative|considered.*reject|instead|other.*hypothes", reason_text))
        alternative_score = 1.0 if has_alternatives else 0.0

        reason_length = len(reason_section.split()) if reason_section else 0
        reason_length_score = min(1.0, reason_length / 80.0)

        reasoning_quality = creative_score * 0.4 + alternative_score * 0.3 + reason_length_score * 0.3

        # --- Final score ---
        score = (
            novelty * 0.3
            + connection * 0.25
            + testability * 0.25
            + reasoning_quality * 0.2
        )
        correct = score >= 0.55

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "novelty": novelty,
                "connection": connection,
                "testability": testability,
                "reasoning_quality": reasoning_quality,
                "hypothesis_keywords_found": float(hyp_kw_found),
                "connection_keywords_found": float(conn_kw_found),
                "originality_score": originality,
                "testability_indicators_found": float(test_found),
                "has_falsification": float(has_falsification),
                "has_prediction": float(has_prediction),
                "has_alternatives": float(has_alternatives),
                "creative_indicators_found": float(creative_found),
            },
            diagnostics=(
                f"novelty={novelty:.2f} connection={connection:.2f} "
                f"testability={testability:.2f} reasoning={reasoning_quality:.2f} "
                f"(hyp_kw={hyp_kw_found}/{len(self._hypothesis_keywords)} "
                f"originality={originality:.2f} "
                f"test_ind={test_found}/5 falsif={has_falsification} "
                f"predict={has_prediction} alternatives={has_alternatives} "
                f"creative={creative_found}/3)"
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


class NovelHypothesisFromEvidenceEnv(BatchEnvBase):
    """NovelHypothesisFromEvidence: synthesize novel hypotheses from evidence.

    Batch-aware: N parallel attempts; reward = best novel hypothesis.
    Trains creative research reasoning — the ability to connect disparate
    findings into new, testable hypotheses that no individual paper proposed.
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
            problem_generator = novel_hypothesis_from_evidence_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return NovelHypothesisVerifier(
            evidence=problem.metadata["evidence"],
            hypothesis_keywords=problem.metadata["hypothesis_keywords"],
            connection_keywords=problem.metadata["connection_keywords"],
            novel_hypothesis=problem.metadata["novel_hypothesis"],
        )

    def _check_format(self, response: str) -> float:
        sections = ["HYPOTHESIS", "CONNECTIONS", "TESTABILITY", "REASONING"]
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
        match = re.search(r"HYPOTHESIS\s*:\s*(.+?)(?:\n[A-Z_]|\Z)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
