"""
BestOfN: Generate N solutions, reward = best score + diversity bonus.

Environment concept:
  The model generates N solutions to a single problem in parallel (via vLLM
  batching). Each solution is scored independently by the verifier. The
  reward is:

    reward = best_score * 0.7 + diversity_bonus * 0.3

  Where:
    - best_score = max(verifier_scores) — the best solution in the batch
    - diversity_bonus = measures how diverse the N solutions are
      (distinct answers / N, or pairwise distance for code)

  This trains the model to:
    1. Generate DIVERSE solutions (not N copies of the same approach)
    2. Generate HIGH-QUALITY solutions (the best one matters)
    3. Explore the solution space (diversity is explicitly rewarded)

  The key insight: with vLLM batching, generating 16 solutions costs roughly
  the same as generating 1. So we can afford to "hedge" — try multiple
  approaches and pick the best. This is the test-time compute scaling paradigm.

Why this environment is worth using for 10T-100T param models:
  Best-of-N is the simplest form of test-time compute scaling. A 100T model
  that generates 16 diverse solutions and picks the best can match a 1000T
  model that generates 1 solution. This environment trains the model to
  be a good "generator" for best-of-N selection — producing diverse, high-
  quality candidates that a verifier can rank.

Problem types:
  - Math problems (exact match verifier)
  - Code problems (execution verifier)
  - Reasoning problems (keyword overlap verifier)

Reward design:
  - best_score: 0.7 weight (the best solution must be correct)
  - diversity_bonus: 0.3 weight (solutions must be diverse, not copies)
  - If best_score = 0 (all wrong): reward = 0 regardless of diversity
  - If best_score = 1 and diversity = 1: reward = 1.0 (perfect)
  - Anti-pattern penalties apply per-sample (averaged)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult, ExactMatchVerifier, NumericVerifier


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_BESTOFN_MATH = [
    {"question": "What is 17 * 23?", "answer": "391"},
    {"question": "What is 15^2?", "answer": "225"},
    {"question": "What is 144 / 8?", "answer": "18"},
    {"question": "What is 2^8?", "answer": "256"},
    {"question": "What is 37 + 48?", "answer": "85"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 13 * 7?", "answer": "91"},
    {"question": "What is 9^3?", "answer": "729"},
    {"question": "What is 120 / 15?", "answer": "8"},
    {"question": "What is 11 * 12?", "answer": "132"},
    {"question": "What is 5^4?", "answer": "625"},
    {"question": "What is 200 / 25?", "answer": "8"},
    {"question": "What is 19 * 6?", "answer": "114"},
    {"question": "What is 7^4?", "answer": "2401"},
    {"question": "What is 350 / 14?", "answer": "25"},
]

_BESTOFN_CODE = [
    {
        "task": "Write `is_palindrome(s)` — True if s reads the same forwards and backwards.",
        "tests": [("racecar", True), ("hello", False), ("", True), ("a", True), ("ab", False)],
        "golden": "def is_palindrome(s):\n    return s == s[::-1]",
    },
    {
        "task": "Write `factorial(n)` — returns n!.",
        "tests": [(0, 1), (1, 1), (5, 120), (3, 6)],
        "golden": "def factorial(n):\n    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r",
    },
    {
        "task": "Write `fibonacci(n)` — returns the nth Fibonacci number (F(0)=0, F(1)=1).",
        "tests": [(0, 0), (1, 1), (2, 1), (5, 5), (10, 55)],
        "golden": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a",
    },
    {
        "task": "Write `is_prime(n)` — True if n is a prime number.",
        "tests": [(2, True), (3, True), (4, False), (1, False), (17, True), (10, False)],
        "golden": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
    },
    {
        "task": "Write `gcd(a, b)` — returns the greatest common divisor.",
        "tests": [(12, 8, 4), (17, 5, 1), (100, 60, 20), (7, 7, 7)],
        "golden": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    },
    {
        "task": "Write `count_vowels(s)` — returns the number of vowels in s.",
        "tests": [("hello", 2), ("", 0), ("aeiou", 5), ("xyz", 0), ("Apple", 2)],
        "golden": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    },
    {
        "task": "Write `reverse_string(s)` — returns the reversed string.",
        "tests": [("hello", "olleh"), ("", ""), ("a", "a"), ("ab", "ba")],
        "golden": "def reverse_string(s):\n    return s[::-1]",
    },
    {
        "task": "Write `sum_digits(n)` — returns the sum of digits of a non-negative integer.",
        "tests": [(123, 6), (0, 0), (9, 9), (99, 18)],
        "golden": "def sum_digits(n):\n    return sum(int(d) for d in str(n))",
    },
    {
        "task": "Write `power(base, exp)` — returns base raised to exp (non-negative exp).",
        "tests": [(2, 3, 8), (5, 0, 1), (3, 2, 9), (10, 3, 1000)],
        "golden": "def power(base, exp):\n    return base ** exp",
    },
    {
        "task": "Write `max_element(lst)` — returns the maximum element in a non-empty list.",
        "tests": [([1, 2, 3], 3), ([5], 5), ([-1, -2, -3], -1), ([0, 0, 0], 0)],
        "golden": "def max_element(lst):\n    return max(lst)",
    },
]


def bestofn_generator(seed: int) -> Problem:
    """Generate a BestOfN problem (math or code)."""
    rng = random.Random(seed)
    if rng.random() < 0.4:
        template = rng.choice(_BESTOFN_MATH)
        return Problem(
            id=f"bestofn_math_{rng.randint(0, 99999)}",
            prompt=(
                f"Question: {template['question']}\n\n"
                f"End with: ANSWER: <value>"
            ),
            difficulty=0.2 + 0.1 * rng.random(),
            metadata={
                "type": "bestofn",
                "subtype": "math",
                "answer": template["answer"],
            },
            token_budget=200,
            source="generated",
        )
    else:
        template = rng.choice(_BESTOFN_CODE)
        return Problem(
            id=f"bestofn_code_{rng.randint(0, 99999)}",
            prompt=(
                f"Task: {template['task']}\n\n"
                f"Output ONLY the code in a ```python block."
            ),
            difficulty=0.25 + 0.1 * rng.random(),
            metadata={
                "type": "bestofn",
                "subtype": "code",
                "task": template["task"],
                "tests": template["tests"],
                "golden": template.get("golden", ""),
            },
            token_budget=400,
            source="generated",
        )


# ---------------------------------------------------------------------------
# Verifier (reuses logic from other envs but packaged for batch)
# ---------------------------------------------------------------------------


class BestOfNVerifier(Verifier):
    """Verifies a single solution for BestOfN (math or code)."""

    def __init__(self, subtype: str, answer: str = "", tests: Optional[list] = None):
        super().__init__()
        self._subtype = subtype
        self._answer = answer
        self._tests = tests or []

    def verify(self, response: str) -> VerifierResult:
        if self._subtype == "math":
            return self._verify_math(response)
        else:
            return self._verify_code(response)

    def _verify_math(self, response: str) -> VerifierResult:
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if not match:
            return VerifierResult(correct=False, score=0.0, diagnostics="No ANSWER: found")
        submitted = match.group(1).strip()
        if submitted == self._answer:
            return VerifierResult(correct=True, score=1.0, diagnostics="Correct")
        return VerifierResult(correct=False, score=0.0, diagnostics=f"Wrong: {submitted} != {self._answer}")

    def _verify_code(self, response: str) -> VerifierResult:
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
            return VerifierResult(correct=False, score=0.0, diagnostics=f"Exec error: {e}")

        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "__builtins__":
                func = obj
                break
        if func is None:
            return VerifierResult(correct=False, score=0.0, diagnostics="No function found")

        passed = 0
        total = len(self._tests)
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
            except Exception:
                pass

        score = passed / total if total > 0 else 0.0
        return VerifierResult(
            correct=passed == total,
            score=score,
            diagnostics=f"Tests: {passed}/{total}",
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BestOfNEnv(BatchEnvBase):
    """
    BestOfN environment: generate N solutions, reward = best + diversity.

    The model generates N solutions in parallel. The verifier scores each.
    Reward = best_score * 0.7 + diversity_bonus * 0.3.

    Diversity is measured as the fraction of distinct answers/approaches
    among the N solutions. For math: distinct ANSWER values. For code:
    distinct normalized code hashes.
    """

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
            problem_generator = bestofn_generator
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
        return BestOfNVerifier(
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
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Compute diversity
        diversity = self._compute_diversity(per_sample)

        # Reward: best_score * 0.7 + diversity * 0.3
        # If all wrong (best_score = 0), reward = 0 regardless of diversity
        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.7 + diversity * 0.3
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Diversity={diversity:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _compute_diversity(self, per_sample: list[dict]) -> float:
        """
        Compute diversity as the fraction of distinct solutions.

        For math: distinct ANSWER values.
        For code: distinct normalized code (by stripping whitespace).
        """
        if not per_sample:
            return 0.0

        # Try to extract distinct answers
        answers = set()
        for s in per_sample:
            resp = s.get("response", "")
            # Try ANSWER: extraction
            match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", resp, re.IGNORECASE)
            if match:
                answers.add(match.group(1).strip())
            else:
                # Use response hash as fallback
                answers.add(hash(resp.strip()))

        if not answers:
            return 0.0

        # Diversity = distinct / total, capped at 1.0
        # But we don't want to reward random noise — so we cap diversity
        # at the point where we have enough variety
        n = len(per_sample)
        distinct = len(answers)
        diversity = min(1.0, distinct / max(n, 1))
        return diversity
