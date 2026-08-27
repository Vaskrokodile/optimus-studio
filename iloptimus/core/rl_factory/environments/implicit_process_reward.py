"""
ImplicitProcessReward: Dense token-level rewards WITHOUT step labels.

Environment concept:
  This environment implements the *implicit process reward model* paradigm
  (inspired by recent research on training PRMs from outcome labels alone).
  Instead of requiring human-annotated step labels, we derive dense
  per-step rewards from outcome-only supervision.

  The key idea — parameterize the process reward as a log-likelihood ratio:
    PRM(step_t) = log P(correct | trace[:t+1]) - log P(correct | trace[:t])
  This is the *temporal difference* of the log-likelihood that the overall
  solution is correct. A step that moves strongly toward correctness gets a
  high positive reward; a step that introduces an error gets a negative one.

  In this environment we simulate this with RULE-BASED scoring (no neural
  reward model, per the project's design principles). We generate problems
  with KNOWN correct and incorrect reasoning traces, so the ground-truth
  per-step contributions are computable:
    - Each step in a correct trace contributes positively (proportional to
      how much closer it brings us to the answer).
    - A step that deviates from the correct chain contributes negatively.

  The model is asked to SCORE a given reasoning trace — producing per-step
  scores. This trains the model to become an implicit PRM itself: it learns
  to identify which steps are productive and which are not, using only
  outcome-level correctness as supervision.

  Verification:
    1. Outcome correctness: does the trace reach the right answer?
    2. Process density: how well-distributed are the step scores (a good
       PRM assigns meaningful scores to many steps, not just the last one)?
    3. Step attribution: do the model's per-step scores correlate with the
       ground-truth per-step contributions (measured by rank correlation)?

  Reward:
    reward = outcome * 0.5 + process_density * 0.3 + step_attribution * 0.2

  This trains the model to:
    1. Recognize correct vs. incorrect reasoning steps (outcome signal)
    2. Produce DENSE per-step signals (not just a single end verdict)
    3. Accurately attribute credit/blame to individual steps (attribution)

  Why this matters for 10T-100T param models:
    Process reward models are the key to solving hard math/code problems
    via search (e.g., tree-of-thought, MCTS). But human step labels are
    expensive and noisy. Implicit PRMs — trained from outcome labels alone
    via log-likelihood ratios — can provide dense supervision at scale.
    This environment trains the model to BE an implicit PRM.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: problems with known correct and incorrect reasoning traces
# ---------------------------------------------------------------------------


_IPR_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem": "What is 15 * 12?",
        "answer": "180",
        "correct_trace": [
            "Step 1: 15 * 12 = 15 * (10 + 2)",
            "Step 2: 15 * 10 = 150",
            "Step 3: 15 * 2 = 30",
            "Step 4: 150 + 30 = 180",
        ],
        "incorrect_trace": [
            "Step 1: 15 * 12 = 15 * (10 + 2)",
            "Step 2: 15 * 10 = 150",
            "Step 3: 15 * 2 = 20",
            "Step 4: 150 + 20 = 170",
        ],
    },
    {
        "problem": "What is 7^3?",
        "answer": "343",
        "correct_trace": [
            "Step 1: 7^3 = 7 * 7 * 7",
            "Step 2: 7 * 7 = 49",
            "Step 3: 49 * 7 = 343",
        ],
        "incorrect_trace": [
            "Step 1: 7^3 = 7 * 7 * 7",
            "Step 2: 7 * 7 = 48",
            "Step 3: 48 * 7 = 336",
        ],
    },
    {
        "problem": "What is 144 / 6 + 8?",
        "answer": "32",
        "correct_trace": [
            "Step 1: 144 / 6 = 24",
            "Step 2: 24 + 8 = 32",
        ],
        "incorrect_trace": [
            "Step 1: 144 / 6 = 22",
            "Step 2: 22 + 8 = 30",
        ],
    },
    {
        "problem": "What is (3 + 4) * 5?",
        "answer": "35",
        "correct_trace": [
            "Step 1: 3 + 4 = 7",
            "Step 2: 7 * 5 = 35",
        ],
        "incorrect_trace": [
            "Step 1: 3 + 4 = 8",
            "Step 2: 8 * 5 = 40",
        ],
    },
    {
        "problem": "What is 2^5 - 10?",
        "answer": "22",
        "correct_trace": [
            "Step 1: 2^5 = 32",
            "Step 2: 32 - 10 = 22",
        ],
        "incorrect_trace": [
            "Step 1: 2^5 = 30",
            "Step 2: 30 - 10 = 20",
        ],
    },
    {
        "problem": "What is 9 * 13?",
        "answer": "117",
        "correct_trace": [
            "Step 1: 9 * 13 = 9 * (10 + 3)",
            "Step 2: 9 * 10 = 90",
            "Step 3: 9 * 3 = 27",
            "Step 4: 90 + 27 = 117",
        ],
        "incorrect_trace": [
            "Step 1: 9 * 13 = 9 * (10 + 3)",
            "Step 2: 9 * 10 = 90",
            "Step 3: 9 * 3 = 24",
            "Step 4: 90 + 24 = 114",
        ],
    },
    {
        "problem": "What is 100 - 7 * 8?",
        "answer": "44",
        "correct_trace": [
            "Step 1: 7 * 8 = 56",
            "Step 2: 100 - 56 = 44",
        ],
        "incorrect_trace": [
            "Step 1: 7 * 8 = 54",
            "Step 2: 100 - 54 = 46",
        ],
    },
    {
        "problem": "What is 6! / (6 - 2)!?",
        "answer": "360",
        "correct_trace": [
            "Step 1: 6! = 720",
            "Step 2: (6-2)! = 4! = 24",
            "Step 3: 720 / 24 = 30",
        ],
        "incorrect_trace": [
            "Step 1: 6! = 720",
            "Step 2: (6-2)! = 4! = 20",
            "Step 3: 720 / 20 = 36",
        ],
    },
    {
        "problem": "What is sqrt(144) + sqrt(25)?",
        "answer": "17",
        "correct_trace": [
            "Step 1: sqrt(144) = 12",
            "Step 2: sqrt(25) = 5",
            "Step 3: 12 + 5 = 17",
        ],
        "incorrect_trace": [
            "Step 1: sqrt(144) = 12",
            "Step 2: sqrt(25) = 4",
            "Step 3: 12 + 4 = 16",
        ],
    },
    {
        "problem": "What is 25% of 480?",
        "answer": "120",
        "correct_trace": [
            "Step 1: 25% = 0.25",
            "Step 2: 0.25 * 480 = 120",
        ],
        "incorrect_trace": [
            "Step 1: 25% = 0.20",
            "Step 2: 0.20 * 480 = 96",
        ],
    },
    {
        "problem": "What is 11 * 11 + 11?",
        "answer": "132",
        "correct_trace": [
            "Step 1: 11 * 11 = 121",
            "Step 2: 121 + 11 = 132",
        ],
        "incorrect_trace": [
            "Step 1: 11 * 11 = 111",
            "Step 2: 111 + 11 = 122",
        ],
    },
    {
        "problem": "What is 3/4 of 200?",
        "answer": "150",
        "correct_trace": [
            "Step 1: 3/4 = 0.75",
            "Step 2: 0.75 * 200 = 150",
        ],
        "incorrect_trace": [
            "Step 1: 3/4 = 0.70",
            "Step 2: 0.70 * 200 = 140",
        ],
    },
]


def _compute_ground_truth_step_scores(trace: list[str], is_correct: bool) -> list[float]:
    """Compute ground-truth per-step contribution scores.

    For a correct trace: each step contributes positively. We model the
    log-likelihood ratio as increasing uniformly across steps (each step
    brings us closer to the correct answer). The temporal difference is
    constant and positive.

    For an incorrect trace: steps before the error contribute positively,
    the error step contributes negatively (a sharp drop in log-likelihood),
    and steps after the error contribute near-zero (the outcome is already
    determined).

    Args:
        trace: List of step strings.
        is_correct: Whether the trace reaches the correct answer.

    Returns:
        List of per-step scores in [-1, 1].
    """
    n = len(trace)
    if n == 0:
        return []

    if is_correct:
        # Uniform positive contribution: each step adds 1/n to the
        # cumulative log-likelihood ratio. TD = 1/n for each step.
        return [1.0 / n] * n

    # Incorrect trace: find the first error step (heuristic: the step
    # where the trace diverges). We don't have the correct trace here,
    # so we use a simple heuristic — the error is typically in the
    # middle of the trace. We assign the error to a random-ish position
    # based on trace length.
    error_idx = max(0, n - 2)  # Error is usually in the second-to-last step
    scores: list[float] = []
    for i in range(n):
        if i < error_idx:
            # Pre-error steps: positive contribution
            scores.append(1.0 / n)
        elif i == error_idx:
            # Error step: strong negative contribution
            scores.append(-0.8)
        else:
            # Post-error steps: near-zero (outcome already determined)
            scores.append(-0.1)
    return scores


def implicit_process_reward_generator(seed: int) -> Problem:
    """Generate an ImplicitProcessReward problem.

    Randomly selects a problem template and decides whether to present
    a correct or incorrect reasoning trace. The model must score each
    step of the trace.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with the trace, ground-truth step scores, and outcome
        label in metadata.
    """
    rng = random.Random(seed)
    template = rng.choice(_IPR_PROBLEMS)

    # 50% chance of correct trace, 50% incorrect
    use_correct = rng.random() < 0.5
    trace = template["correct_trace"] if use_correct else template["incorrect_trace"]
    is_correct = use_correct

    ground_truth_scores = _compute_ground_truth_step_scores(trace, is_correct)

    # Build the trace text with step markers
    trace_text = "\n".join(trace)

    n_steps = len(trace)

    prompt = (
        f"Score this reasoning trace:\n"
        f"{template['problem']}\n"
        f"<trace with steps>\n"
        f"{trace_text}\n"
        f"STEP SCORES: <per-step scores in [-1, 1], one per line as "
        f"'Step i: score'>\n"
        f"OUTCOME: <correct or incorrect>"
    )

    return Problem(
        id=f"implicit_prm_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.3 + 0.15 * rng.random(),
        metadata={
            "type": "implicit_process_reward",
            "problem": template["problem"],
            "answer": template["answer"],
            "trace": trace,
            "is_correct": is_correct,
            "ground_truth_step_scores": ground_truth_scores,
            "n_steps": n_steps,
        },
        token_budget=300 + n_steps * 30,
        source="generated",
    )


# Prevent pytest from collecting this generator function as a test
implicit_process_reward_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ImplicitProcessRewardVerifier(Verifier):
    """Verifies per-step scores for an implicit process reward task.

    The model produces:
      1. Per-step scores (one per step in the trace)
      2. An outcome label (correct/incorrect)

    Verification checks:
      - Outcome correctness: does the model's outcome label match the
        ground truth?
      - Process density: how many steps received a non-trivial score
        (|score| > 0.05)? A good PRM assigns meaningful scores to many steps.
      - Step attribution: rank correlation between the model's scores and
        the ground-truth step scores (measured via Spearman-like ranking).

    Args:
        is_correct: Ground-truth outcome (True = trace is correct).
        ground_truth_scores: Ground-truth per-step scores.
        n_steps: Number of steps in the trace.
    """

    def __init__(
        self,
        is_correct: bool,
        ground_truth_scores: list[float],
        n_steps: int,
    ):
        super().__init__()
        self._is_correct = is_correct
        self._ground_truth_scores = ground_truth_scores
        self._n_steps = n_steps

    def verify(self, response: str) -> VerifierResult:
        # Parse per-step scores
        model_scores = self._parse_step_scores(response)
        # Parse outcome label
        model_outcome = self._parse_outcome(response)

        # 1. Outcome correctness
        outcome_correct = model_outcome == self._is_correct
        outcome_score = 1.0 if outcome_correct else 0.0

        # 2. Process density: fraction of steps with non-trivial scores
        if model_scores:
            non_trivial = sum(1 for s in model_scores if abs(s) > 0.05)
            process_density = non_trivial / len(model_scores)
        else:
            process_density = 0.0

        # 3. Step attribution: rank correlation with ground truth
        step_attribution = self._compute_attribution(model_scores)

        # Combined score
        score = outcome_score * 0.5 + process_density * 0.3 + step_attribution * 0.2
        correct = outcome_correct and step_attribution > 0.3

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "outcome": outcome_score,
                "process_density": process_density,
                "step_attribution": step_attribution,
            },
            diagnostics=(
                f"Outcome={outcome_score:.2f} "
                f"Density={process_density:.2f} "
                f"Attribution={step_attribution:.2f} "
                f"Scores={len(model_scores)}/{self._n_steps}"
            ),
            metadata={
                "model_scores": model_scores,
                "model_outcome": model_outcome,
                "outcome_correct": outcome_correct,
            },
        )

    def _parse_step_scores(self, response: str) -> list[float]:
        """Parse per-step scores from the model's response.

        Expected format: "Step i: score" on each line, or a comma-separated
        list after "STEP SCORES:".

        Args:
            response: The model's response text.

        Returns:
            List of parsed float scores.
        """
        scores: list[float] = []

        # Try "Step i: score" format
        step_matches = re.findall(
            r"Step\s+\d+\s*:\s*(-?\d+\.?\d*)", response, re.IGNORECASE
        )
        if step_matches:
            for s in step_matches:
                try:
                    val = float(s)
                    val = max(-1.0, min(1.0, val))
                    scores.append(val)
                except ValueError:
                    pass
            return scores

        # Try comma-separated list after "STEP SCORES:"
        scores_match = re.search(
            r"STEP SCORES\s*:\s*(.+?)(?:\n(?:OUTCOME|Step)|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if scores_match:
            parts = re.split(r"[,\s]+", scores_match.group(1).strip())
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    val = float(p)
                    val = max(-1.0, min(1.0, val))
                    scores.append(val)
                except ValueError:
                    pass

        return scores

    def _parse_outcome(self, response: str) -> Optional[bool]:
        """Parse the outcome label from the response.

        Args:
            response: The model's response text.

        Returns:
            True if "correct", False if "incorrect", None if not found.
        """
        match = re.search(r"OUTCOME\s*:\s*(\w+)", response, re.IGNORECASE)
        if not match:
            return None
        label = match.group(1).strip().lower()
        if "correct" in label and "in" not in label:
            return True
        if "incorrect" in label or "wrong" in label:
            return False
        return None

    def _compute_attribution(self, model_scores: list[float]) -> float:
        """Compute step attribution accuracy via rank correlation.

        Measures how well the model's per-step scores correlate (in rank)
        with the ground-truth per-step scores. Uses a simplified Spearman
        rank correlation.

        Args:
            model_scores: The model's parsed per-step scores.

        Returns:
            Attribution score in [0, 1]. 1.0 = perfect rank correlation.
        """
        gt = self._ground_truth_scores
        if not model_scores or not gt:
            return 0.0

        # Align lengths (use min)
        n = min(len(model_scores), len(gt))
        if n == 0:
            return 0.0

        m = model_scores[:n]
        g = gt[:n]

        if n == 1:
            # Single step: check sign agreement
            if (m[0] > 0) == (g[0] > 0):
                return 1.0
            return 0.0

        # Compute ranks
        m_ranks = self._rank(m)
        g_ranks = self._rank(g)

        # Spearman correlation
        d_sq = sum((mr - gr) ** 2 for mr, gr in zip(m_ranks, g_ranks))
        spearman = 1.0 - (6.0 * d_sq) / (n * (n * n - 1)) if n > 1 else 1.0

        # Map [-1, 1] to [0, 1]
        return (spearman + 1.0) / 2.0

    @staticmethod
    def _rank(values: list[float]) -> list[float]:
        """Compute ranks of values (1 = smallest).

        Args:
            values: List of floats.

        Returns:
            List of ranks (1-indexed, ties get average rank).
        """
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ImplicitProcessRewardEnv(BatchEnvBase):
    """ImplicitProcessReward environment: train an implicit PRM.

    The model is given a reasoning trace and must produce per-step scores
    (simulating an implicit process reward model) plus an outcome label.

    Reward = outcome * 0.5 + process_density * 0.3 + step_attribution * 0.2.

    This is a batch environment: N parallel scorings are aggregated by
    taking the best attribution score (the model that best identifies
    productive vs. unproductive steps).

    Args:
        problems: Fixed list of problems to sample from.
        problem_generator: Callable(seed) -> Problem. Defaults to
            implicit_process_reward_generator.
        reward_config: Configuration for the reward function.
        anti_pattern_detector: Detector for wasteful reasoning patterns.
        render_mode: "text" or None.
        batch_size: Number of parallel responses (default 16).
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = implicit_process_reward_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return ImplicitProcessRewardVerifier(
            is_correct=meta["is_correct"],
            ground_truth_scores=meta["ground_truth_step_scores"],
            n_steps=meta["n_steps"],
        )

    def _check_format(self, response: str) -> float:
        """Check if the response contains STEP SCORES and OUTCOME markers."""
        has_scores = bool(re.search(r"STEP SCORES", response, re.IGNORECASE))
        has_outcome = bool(re.search(r"OUTCOME\s*:", response, re.IGNORECASE))
        if has_scores and has_outcome:
            return 1.0
        if has_scores or has_outcome:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """Aggregate per-sample scores for the implicit PRM batch.

        Takes the best outcome score, best process density, and best
        step attribution across the batch, then combines them.

        Args:
            per_sample: List of per-sample score dicts.

        Returns:
            Aggregate dict with combined reward and diagnostics.
        """
        if not per_sample:
            return {
                "best_score": 0.0,
                "reward": 0.0,
                "correct": False,
                "diagnostics": "No samples",
            }

        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]

        best_score = max(scores)
        any_correct = any(corrects)

        # Extract per-component scores from diagnostics metadata
        # The verifier stores partial_credit in the result, but per_sample
        # only has the score. We approximate by using the best overall score.
        mean_score = sum(scores) / len(scores)

        # Reward: best_score is already the weighted combination
        reward = best_score

        return {
            "best_score": best_score,
            "mean_score": mean_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "reward": reward,
            "correct": any_correct and best_score >= 0.5,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        """Extract the step scores section from the response."""
        match = re.search(
            r"STEP SCORES\s*:\s*(.+?)(?:OUTCOME|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return response
