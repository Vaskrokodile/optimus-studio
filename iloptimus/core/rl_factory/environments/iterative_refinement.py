"""
IterativeRefinement: Generate → score → improve → score, keep best across K rounds.

Environment concept:
  The model goes through K rounds of refinement:
    Round 1: Generate a solution → get score
    Round 2: See the score + feedback → generate an improved solution → get score
    ...
    Round K: Final improvement → get score

  The reward is based on:
    1. Best score across all K rounds (did any round produce a correct solution?)
    2. Improvement trajectory (did scores increase over rounds?)
    3. Final score (did the model converge on a good solution?)

    reward = best_score * 0.5 + final_score * 0.3 + improvement * 0.2

  Where:
    - best_score = max(scores across rounds)
    - final_score = score of the last round
    - improvement = (final_score - first_score) / max(first_score, 0.01)

  This trains the model to:
    1. Recognize when its solution is wrong (from the score feedback)
    2. Make TARGETED improvements (not just re-rolling)
    3. Converge on a correct solution over multiple rounds
    4. NOT regress (improvement should be positive, not negative)

  The key insight: iterative refinement is how agents actually work in
  practice — try, get feedback, improve. This environment trains the
  model to be GOOD at this loop, not just capable of it.

Why this environment is worth using for 10T-100T param models:
  Iterative refinement is the most practical test-time compute method.
  A 100T model that can improve its solution over 4 rounds can match a
  400T model that generates 1 solution. This environment trains the
  model to use feedback effectively — the core skill for agentic loops.

Problem types:
  - Code problems (with test feedback)
  - Math problems (with answer verification)
  - Bug fixes (with error messages)

Reward design:
  - best_score: 0.5 weight (at least one round must be correct)
  - final_score: 0.3 weight (the model should converge, not just spike)
  - improvement: 0.2 weight (positive trajectory is rewarded)
  - If best_score = 0: reward = 0 (all rounds failed)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_REFINE_CODE = [
    {
        "task": "Write `is_palindrome(s)` — True if s == s[::-1].",
        "tests": [("racecar", True), ("hello", False), ("", True), ("a", True)],
        "buggy_hint": "Common bug: forgetting to handle empty strings or case sensitivity.",
    },
    {
        "task": "Write `fibonacci(n)` — F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).",
        "tests": [(0, 0), (1, 1), (5, 5), (10, 55)],
        "buggy_hint": "Common bug: off-by-one in the loop or wrong base cases.",
    },
    {
        "task": "Write `factorial(n)` — returns n!.",
        "tests": [(0, 1), (1, 1), (5, 120), (3, 6)],
        "buggy_hint": "Common bug: starting the product from 0 instead of 1.",
    },
    {
        "task": "Write `is_prime(n)` — True if n is prime.",
        "tests": [(2, True), (3, True), (4, False), (1, False), (17, True)],
        "buggy_hint": "Common bug: not handling n < 2 or checking up to n instead of sqrt(n).",
    },
    {
        "task": "Write `count_vowels(s)` — count vowels in s (case-insensitive).",
        "tests": [("hello", 2), ("", 0), ("AEIOU", 5), ("xyz", 0)],
        "buggy_hint": "Common bug: not lowercasing the string first.",
    },
    {
        "task": "Write `max_of_list(lst)` — return max of non-empty list.",
        "tests": [([1, 2, 3], 3), ([5], 5), ([-1, -2, -3], -1)],
        "buggy_hint": "Common bug: initializing max to 0 instead of lst[0].",
    },
    {
        "task": "Write `sum_range(a, b)` — sum of integers from a to b inclusive.",
        "tests": [(1, 5, 15), (0, 0, 0), (3, 3, 3), (1, 10, 55)],
        "buggy_hint": "Common bug: using range(a, b) instead of range(a, b+1).",
    },
    {
        "task": "Write `reverse_list(lst)` — return reversed list.",
        "tests": [([1, 2, 3], [3, 2, 1]), ([], []), (["a"], ["a"])],
        "buggy_hint": "Common bug: modifying in place instead of returning a new list.",
    },
]

_REFINE_MATH = [
    {"question": "What is 17 * 23?", "answer": "391"},
    {"question": "What is 15^2 + 7?", "answer": "232"},
    {"question": "What is 144 / 8 + 3?", "answer": "21"},
    {"question": "What is 2^8 - 10?", "answer": "246"},
    {"question": "What is 37 * 3 - 11?", "answer": "100"},
    {"question": "What is 9^3 / 9?", "answer": "81"},
    {"question": "What is 13 * 7 + 9?", "answer": "100"},
    {"question": "What is 5^4 - 25?", "answer": "600"},
]


def iterative_refinement_generator(seed: int) -> Problem:
    """Generate an IterativeRefinement problem."""
    rng = random.Random(seed)
    if rng.random() < 0.6:
        template = rng.choice(_REFINE_CODE)
        return Problem(
            id=f"iterative_refine_code_{rng.randint(0, 99999)}",
            prompt=(
                f"Task: {template['task']}\n"
                f"Hint: {template['buggy_hint']}\n\n"
                f"Output ONLY the code in a ```python block."
            ),
            difficulty=0.25 + 0.1 * rng.random(),
            metadata={
                "type": "iterative_refinement",
                "subtype": "code",
                "task": template["task"],
                "tests": template["tests"],
            },
            token_budget=400,
            source="generated",
        )
    else:
        template = rng.choice(_REFINE_MATH)
        return Problem(
            id=f"iterative_refine_math_{rng.randint(0, 99999)}",
            prompt=(
                f"Question: {template['question']}\n\n"
                f"End with: ANSWER: <value>"
            ),
            difficulty=0.2 + 0.1 * rng.random(),
            metadata={
                "type": "iterative_refinement",
                "subtype": "math",
                "answer": template["answer"],
            },
            token_budget=200,
            source="generated",
        )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class IterativeRefinementVerifier(Verifier):
    """Verifies a single solution for IterativeRefinement."""

    def __init__(self, subtype: str, answer: str = "", tests: Optional[list] = None):
        super().__init__()
        self._subtype = subtype
        self._answer = answer
        self._tests = tests or []

    def verify(self, response: str) -> VerifierResult:
        if self._subtype == "math":
            match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if not match:
                return VerifierResult(correct=False, score=0.0, diagnostics="No ANSWER: found")
            submitted = match.group(1).strip()
            if submitted == self._answer:
                return VerifierResult(correct=True, score=1.0, diagnostics="Correct")
            return VerifierResult(correct=False, score=0.0, diagnostics=f"Wrong: {submitted} != {self._answer}")
        else:
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
            if not code_match:
                return VerifierResult(correct=False, score=0.0, diagnostics="No code block")

            code = code_match.group(1).strip()
            namespace: dict[str, Any] = {}
            try:
                exec(code, namespace)
            except Exception as e:
                return VerifierResult(correct=False, score=0.0, diagnostics=f"Exec error: {type(e).__name__}: {e}")

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name != "__builtins__":
                    func = obj
                    break
            if func is None:
                return VerifierResult(correct=False, score=0.0, diagnostics="No function found")

            passed = 0
            total = len(self._tests)
            failures = []
            for test in self._tests:
                try:
                    if len(test) == 2:
                        args, expected = test
                        result = func(args) if not isinstance(args, tuple) else func(*args)
                    elif len(test) == 3:
                        a, b, expected = test
                        result = func(a, b)
                    else:
                        args = test[:-1]
                        expected = test[-1]
                        result = func(*args)
                    if result == expected:
                        passed += 1
                    else:
                        failures.append(f"input={test[:-1]}: got {result}, expected {expected}")
                except Exception as e:
                    failures.append(f"input={test[:-1]}: {type(e).__name__}")

            score = passed / total if total > 0 else 0.0
            diag = f"Tests: {passed}/{total}"
            if failures:
                diag += f". Failures: {failures[0]}"
            return VerifierResult(correct=passed == total, score=score, diagnostics=diag)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class IterativeRefinementEnv(BatchEnvBase):
    """
    IterativeRefinement environment: K rounds of generate → score → improve.

    The model submits K solutions (one per round). Each is scored.
    Reward = best_score * 0.5 + final_score * 0.3 + improvement * 0.2.

    The batch_size parameter controls K (number of refinement rounds).
    Each "sample" in the batch is one round's solution.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 4,  # K rounds (smaller than other batch envs)
    ):
        if problems is None and problem_generator is None:
            problem_generator = iterative_refinement_generator
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
        return IterativeRefinementVerifier(
            subtype=meta["subtype"],
            answer=meta.get("answer", ""),
            tests=meta.get("tests", []),
        )

    def _check_format(self, response: str) -> float:
        if "ANSWER:" in response:
            return 1.0
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]

        if not scores:
            return {
                "best_score": 0.0,
                "final_score": 0.0,
                "improvement": 0.0,
                "reward": 0.0,
                "correct": False,
                "diagnostics": "No samples",
            }

        best_score = max(scores)
        final_score = scores[-1]
        first_score = scores[0]

        # Improvement: how much the score improved from first to last
        if first_score > 0:
            improvement = (final_score - first_score) / first_score
        else:
            improvement = final_score  # if first was 0, any positive final is improvement

        # Clamp improvement to [0, 1] — we don't want to reward huge spikes
        improvement = max(0.0, min(1.0, improvement))

        # Reward
        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.5 + final_score * 0.3 + improvement * 0.2
            reward = min(1.0, reward)

        # Check if the trajectory is monotonically improving (bonus signal)
        monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

        return {
            "best_score": best_score,
            "final_score": final_score,
            "first_score": first_score,
            "improvement": improvement,
            "monotonic_improvement": monotonic,
            "scores_trajectory": scores,
            "any_correct": any(corrects),
            "correct_count": sum(corrects),
            "reward": reward,
            "correct": best_score >= 0.8,
            "diagnostics": f"Best={best_score:.2f} Final={final_score:.2f} Improvement={improvement:.2f} Trajectory={[f'{s:.1f}' for s in scores]}",
        }
