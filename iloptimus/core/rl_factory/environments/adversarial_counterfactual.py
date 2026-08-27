"""
AdversarialCounterfactual: Generate breaking counterfactuals, then defend.

Environment concept:
  The model is given a solution and must (1) generate a counterfactual
  scenario that would break it, then (2) explain why the solution survives
  (or acknowledge it breaks). This trains adversarial self-critique.

  Why: robust solutions survive counterfactual stress tests. This
  environment trains the model to think adversarially about its own
  solutions — generating worst cases and verifying resilience.

Verification:
  - Counterfactual is valid (mentions relevant keywords / scenario).
  - Defense is coherent (mentions relevant defense keywords).

Reward design:
  counterfactual_valid * 0.5 + defense_coherent * 0.5
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — solutions with counterfactual and defense keywords
# ---------------------------------------------------------------------------

_COUNTERFACTUAL_PROBLEMS = [
    {
        "solution": (
            "Binary search works because the array is sorted: at each step "
            "we eliminate half the search space by comparing to the middle "
            "element."
        ),
        "counterfactual_keywords": ["unsorted", "not sorted", "duplicates", "empty", "one element"],
        "defense_keywords": ["sort", "sorted", "precondition", "assume", "handle", "check"],
        "pattern": "binary search",
    },
    {
        "solution": (
            "A hash set provides O(1) lookup because it uses a hash function "
            "to map keys to buckets, avoiding linear scanning."
        ),
        "counterfactual_keywords": ["collision", "hash", "worst case", "adversarial", "many"],
        "defense_keywords": ["resize", "rehash", "load factor", "chaining", "balanced", "amortized"],
        "pattern": "hash set",
    },
    {
        "solution": (
            "Quicksort sorts by picking a pivot and partitioning: elements "
            "smaller go left, larger go right, then recurse."
        ),
        "counterfactual_keywords": ["sorted", "already sorted", "worst case", "pivot", "duplicates", "equal"],
        "defense_keywords": ["random", "median", "pivot", "three", "shuffle", "balanced"],
        "pattern": "quicksort",
    },
    {
        "solution": (
            "Memoization speeds up recursion by caching results of "
            "subproblems, so each subproblem is solved only once."
        ),
        "counterfactual_keywords": ["memory", "space", "large", "cache", "infinite", "overflow"],
        "defense_keywords": ["tabulation", "iterative", "bounded", "limit", "trade-off", "space"],
        "pattern": "memoization",
    },
    {
        "solution": (
            "Gradient descent converges by following the negative gradient "
            "direction, reducing the loss at each step."
        ),
        "counterfactual_keywords": ["local minimum", "saddle", "learning rate", "diverge", "oscillate"],
        "defense_keywords": ["momentum", "adaptive", "learning rate", "schedule", "second order", "restart"],
        "pattern": "gradient descent",
    },
    {
        "solution": (
            "A two-pointer technique finds pairs in a sorted array by "
            "moving one pointer from each end toward the middle."
        ),
        "counterfactual_keywords": ["unsorted", "not sorted", "duplicates", "three", "k-sum"],
        "defense_keywords": ["sort", "sorted", "precondition", "extend", "generalize", "k"],
        "pattern": "two-pointer",
    },
    {
        "solution": (
            "Dynamic programming solves problems by breaking them into "
            "overlapping subproblems and building up the solution."
        ),
        "counterfactual_keywords": ["no overlap", "independent", "exponential", "state", "large"],
        "defense_keywords": ["subproblem", "overlap", "optimal substructure", "table", "state"],
        "pattern": "dynamic programming",
    },
    {
        "solution": (
            "A Bloom filter tests set membership probabilistically using "
            "multiple hash functions and a bit array."
        ),
        "counterfactual_keywords": ["false positive", "no deletion", "counting", "saturate", "full"],
        "defense_keywords": ["counting bloom", "false positive rate", "resize", "multiple", "probability"],
        "pattern": "bloom filter",
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def adversarial_counterfactual_generator(seed: int) -> Problem:
    """Generate an AdversarialCounterfactual problem."""
    rng = random.Random(seed)
    problem = rng.choice(_COUNTERFACTUAL_PROBLEMS)

    solution = problem["solution"]
    counterfactual_keywords = problem["counterfactual_keywords"]
    defense_keywords = problem["defense_keywords"]

    prompt = (
        f"Consider the following solution:\n\n"
        f"\"{solution}\"\n\n"
        f"Generate a counterfactual scenario that could break this solution, "
        f"then explain whether the solution survives (and how).\n\n"
        f"Format your answer as:\n"
        f"COUNTERFACTUAL: <scenario>\n"
        f"DEFENSE: <explanation>"
    )

    difficulty = 0.6

    return Problem(
        id=f"adversarial_counterfactual_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "solution": solution,
            "counterfactual_keywords": counterfactual_keywords,
            "defense_keywords": defense_keywords,
            "pattern": problem["pattern"],
        },
        token_budget=768,
        source="adversarial_counterfactual_generator",
    )


adversarial_counterfactual_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class AdversarialCounterfactualVerifier(Verifier):
    """Verify an adversarial counterfactual: valid scenario + coherent defense.

    reward = counterfactual_valid * 0.5 + defense_coherent * 0.5
    """

    def __init__(self, counterfactual_keywords: list[str], defense_keywords: list[str]):
        super().__init__()
        self._cf_keywords = [k.lower() for k in counterfactual_keywords]
        self._def_keywords = [k.lower() for k in defense_keywords]

    def verify(self, response: str) -> VerifierResult:
        # Parse COUNTERFACTUAL
        cf_match = re.search(r"COUNTERFACTUAL\s*:\s*(.+?)(?:\nDEFENSE|\Z)", response, re.IGNORECASE | re.DOTALL)
        cf_text = cf_match.group(1).lower().strip() if cf_match else ""

        # Parse DEFENSE
        def_match = re.search(r"DEFENSE\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        def_text = def_match.group(1).lower().strip() if def_match else ""

        # Check counterfactual validity: at least one keyword OR substantial text
        cf_found = sum(1 for kw in self._cf_keywords if kw in cf_text)
        cf_valid = cf_found > 0 or len(cf_text) > 30
        cf_score = min(1.0, cf_found / 2.0) if self._cf_keywords else 0.0
        if cf_valid and cf_score == 0.0:
            cf_score = 0.5  # partial for substantial text without keywords

        # Check defense coherence
        def_found = sum(1 for kw in self._def_keywords if kw in def_text)
        def_coherent = def_found > 0 or len(def_text) > 30
        def_score = min(1.0, def_found / 2.0) if self._def_keywords else 0.0
        if def_coherent and def_score == 0.0:
            def_score = 0.5

        score = cf_score * 0.5 + def_score * 0.5
        correct = cf_valid and def_coherent and score >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "counterfactual_valid": cf_score,
                "defense_coherent": def_score,
                "cf_keywords_found": float(cf_found),
                "def_keywords_found": float(def_found),
            },
            diagnostics=(
                f"cf_score={cf_score:.2f} (kw={cf_found}) "
                f"def_score={def_score:.2f} (kw={def_found})"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AdversarialCounterfactualEnv(BatchEnvBase):
    """AdversarialCounterfactual: generate counterfactuals and defend.

    Batch-aware: N parallel attempts; reward = best counterfactual+defense.
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
            problem_generator = adversarial_counterfactual_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return AdversarialCounterfactualVerifier(
            counterfactual_keywords=problem.metadata["counterfactual_keywords"],
            defense_keywords=problem.metadata["defense_keywords"],
        )

    def _check_format(self, response: str) -> float:
        has_cf = bool(re.search(r"COUNTERFACTUAL\s*:", response, re.IGNORECASE))
        has_def = bool(re.search(r"DEFENSE\s*:", response, re.IGNORECASE))
        if has_cf and has_def:
            return 1.0
        if has_cf or has_def:
            return 0.5
        return 0.0

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
        cf_match = re.search(r"COUNTERFACTUAL\s*:\s*(.+?)(?:\nDEFENSE|\Z)", response, re.IGNORECASE | re.DOTALL)
        def_match = re.search(r"DEFENSE\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        parts = []
        if cf_match:
            parts.append(f"CF: {cf_match.group(1).strip()[:80]}")
        if def_match:
            parts.append(f"DEF: {def_match.group(1).strip()[:80]}")
        return " | ".join(parts) if parts else response
