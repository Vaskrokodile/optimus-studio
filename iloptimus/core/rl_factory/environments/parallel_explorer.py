"""
ParallelExplorer: Generate N different approaches, reward = best + coverage.

Environment concept:
  The model generates N solutions that must use DIFFERENT approaches to the
  same problem. The environment:
    1. Scores each solution with the verifier
    2. Detects which "approach" each solution uses (via keyword/heuristic analysis)
    3. Rewards both the BEST solution and the COVERAGE of distinct approaches

    reward = best_score * 0.4 + coverage * 0.3 + correct_fraction * 0.3

  Where:
    - best_score = max(verifier_scores)
    - coverage = fraction of expected approaches that were attempted
    - correct_fraction = fraction of solutions that are correct

  This is different from BestOfN (which rewards diversity generically) and
  SelfConsistency (which rewards agreement). ParallelExplorer rewards
  EXPLORING DIFFERENT APPROACHES — e.g., for a sorting problem:
    - Approach 1: Bubble sort
    - Approach 2: Merge sort
    - Approach 3: Quick sort
    - Approach 4: Built-in sort

  The model learns to explore the solution space systematically, not just
  randomly vary outputs.

Why this environment is worth using for 10T-100T param models:
  Systematic exploration is the highest-level reasoning skill. A 100T model
  that can generate 4 genuinely different approaches to a problem and pick
  the best one is more robust than one that generates 4 variations of the
  same approach. This trains the model to think in terms of STRATEGIES, not
  just solutions.

Problem types:
  - Code problems with known approach categories (iterative, recursive, builtin)
  - Math problems with multiple solution methods
  - Algorithm problems with different complexity classes

Reward design:
  - best_score: 0.4 weight (at least one must be correct)
  - coverage: 0.3 weight (different approaches must be explored)
  - correct_fraction: 0.3 weight (multiple correct solutions = robustness)
  - If best_score = 0: reward = 0 (all approaches failed)
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


_EXPLORER_CODE = [
    {
        "task": "Write a function `factorial(n)` that returns n!.",
        "tests": [(0, 1), (1, 1), (5, 120), (3, 6)],
        "approaches": ["iterative", "recursive", "builtin", "functional"],
        "approach_keywords": {
            "iterative": ["for", "while", "loop", "range"],
            "recursive": ["def", "self", "call", "return", "n - 1", "n-1"],
            "builtin": ["math", "factorial", "import", "prod"],
            "functional": ["reduce", "lambda", "map", "filter", "operator"],
        },
    },
    {
        "task": "Write a function `fibonacci(n)` that returns the nth Fibonacci number.",
        "tests": [(0, 0), (1, 1), (2, 1), (5, 5), (10, 55)],
        "approaches": ["iterative", "recursive", "memoized", "formula"],
        "approach_keywords": {
            "iterative": ["for", "while", "loop", "range"],
            "recursive": ["def", "n - 1", "n-1", "self"],
            "memoized": ["cache", "memo", "dict", "lru"],
            "formula": ["sqrt", "golden", "5", "**", "phi"],
        },
    },
    {
        "task": "Write a function `reverse_string(s)` that reverses a string.",
        "tests": [("hello", "olleh"), ("", ""), ("a", "a"), ("ab", "ba")],
        "approaches": ["slice", "loop", "recursive", "builtin"],
        "approach_keywords": {
            "slice": ["[::-1]", "slice", "-1"],
            "loop": ["for", "while", "append", "+= "],
            "recursive": ["def", "s[0]", "s[1:]", "self", "call"],
            "builtin": ["reversed", "join", "list"],
        },
    },
    {
        "task": "Write a function `sum_list(lst)` that returns the sum of list elements.",
        "tests": [([1, 2, 3], 6), ([], 0), ([5], 5), ([-1, -2, 3], 0)],
        "approaches": ["builtin", "loop", "recursive", "functional"],
        "approach_keywords": {
            "builtin": ["sum("],
            "loop": ["for", "while", "+=", "total"],
            "recursive": ["def", "lst[0]", "lst[1:]", "self"],
            "functional": ["reduce", "lambda", "operator", "add"],
        },
    },
    {
        "task": "Write a function `max_element(lst)` that returns the max of a non-empty list.",
        "tests": [([1, 2, 3], 3), ([5], 5), ([-1, -2, -3], -1)],
        "approaches": ["builtin", "loop", "recursive", "sort"],
        "approach_keywords": {
            "builtin": ["max("],
            "loop": ["for", "while", "if", ">", "current"],
            "recursive": ["def", "lst[0]", "lst[1:]", "self"],
            "sort": ["sort", "sorted", "[-1]"],
        },
    },
    {
        "task": "Write a function `count_vowels(s)` that counts vowels (case-insensitive).",
        "tests": [("hello", 2), ("", 0), ("AEIOU", 5), ("xyz", 0)],
        "approaches": ["loop", "comprehension", "functional", "regex"],
        "approach_keywords": {
            "loop": ["for", "while", "if", "in", "count"],
            "comprehension": ["sum(", "for ", "in ", "if "],
            "functional": ["filter", "lambda", "map", "len"],
            "regex": ["re", "findall", "pattern", "compile"],
        },
    },
    {
        "task": "Write a function `is_palindrome(s)` — True if s reads the same both ways.",
        "tests": [("racecar", True), ("hello", False), ("", True), ("a", True)],
        "approaches": ["slice", "loop", "recursive", "two_pointer"],
        "approach_keywords": {
            "slice": ["[::-1]", "slice"],
            "loop": ["for", "while", "if", "=="],
            "recursive": ["def", "s[0]", "s[1:]", "self"],
            "two_pointer": ["left", "right", "i", "j", "while"],
        },
    },
    {
        "task": "Write a function `power(base, exp)` — base raised to exp (non-negative).",
        "tests": [(2, 3, 8), (5, 0, 1), (3, 2, 9), (10, 3, 1000)],
        "approaches": ["builtin", "loop", "recursive", "fast"],
        "approach_keywords": {
            "builtin": ["**", "pow("],
            "loop": ["for", "while", "range", "*="],
            "recursive": ["def", "exp - 1", "exp-1", "self"],
            "fast": ["//", "exp % 2", "exp//2", "square", "half"],
        },
    },
]


def parallel_explorer_generator(seed: int) -> Problem:
    """Generate a ParallelExplorer problem."""
    rng = random.Random(seed)
    template = rng.choice(_EXPLORER_CODE)

    return Problem(
        id=f"parallel_explorer_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Generate {len(template['approaches'])} DIFFERENT solutions, each using a "
            f"different approach. Label each solution:\n"
            f"APPROACH: <name>\n```python\n<code>\n```\n\n"
            f"Possible approaches: {', '.join(template['approaches'])}\n"
            f"Each solution must use a genuinely different strategy."
        ),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "parallel_explorer",
            "task": template["task"],
            "tests": template["tests"],
            "approaches": template["approaches"],
            "approach_keywords": template["approach_keywords"],
        },
        token_budget=1200,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ParallelExplorerVerifier(Verifier):
    """Verifies a single solution block for ParallelExplorer."""

    def __init__(self, tests: list, approach_keywords: dict[str, list[str]]):
        super().__init__()
        self._tests = tests
        self._approach_keywords = approach_keywords

    def verify(self, response: str) -> VerifierResult:
        # Extract approach label
        approach_match = re.search(r"APPROACH:\s*(\w+)", response, re.IGNORECASE)
        approach = approach_match.group(1).lower() if approach_match else "unknown"

        # Extract code
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if not code_match:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No code block",
                metadata={"approach": approach},
            )

        code = code_match.group(1).strip()

        # Execute
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Exec error: {e}",
                metadata={"approach": approach},
            )

        # Find function
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "__builtins__":
                func = obj
                break
        if func is None:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No function found",
                metadata={"approach": approach},
            )

        # Run tests
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

        # Verify the approach matches the keywords
        approach_detected = self._detect_approach(code)
        approach_match_score = 1.0 if approach_detected == approach else 0.5

        return VerifierResult(
            correct=passed == total,
            score=score,
            diagnostics=f"Approach={approach} (detected={approach_detected}) Tests: {passed}/{total}",
            metadata={"approach": approach, "approach_detected": approach_detected, "approach_match": approach_match_score},
        )

    def _detect_approach(self, code: str) -> str:
        """Detect which approach the code uses based on keywords."""
        code_lower = code.lower()
        best_approach = "unknown"
        best_score = 0

        for approach, keywords in self._approach_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in code_lower)
            score = matches / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_approach = approach

        return best_approach


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ParallelExplorerEnv(BatchEnvBase):
    """
    ParallelExplorer environment: N different approaches, reward = best + coverage.

    The model generates N solutions, each labeled with an approach name.
    Reward = best_score * 0.4 + coverage * 0.3 + correct_fraction * 0.3.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 8,  # Typically 4-8 approaches
    ):
        if problems is None and problem_generator is None:
            problem_generator = parallel_explorer_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ParallelExplorerVerifier(
            tests=problem.metadata["tests"],
            approach_keywords=problem.metadata["approach_keywords"],
        )

    def _check_format(self, response: str) -> float:
        if "APPROACH:" in response and "```python" in response:
            return 1.0
        if "```python" in response:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        n = len(per_sample)
        if n == 0:
            return {"best_score": 0.0, "reward": 0.0, "correct": False, "diagnostics": "No samples"}

        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        correct_fraction = sum(corrects) / n

        # Compute coverage: how many of the expected approaches were attempted
        expected_approaches = set(self._current_problem.metadata["approaches"])
        attempted_approaches = set()
        for s in per_sample:
            # The approach is stored in the verifier metadata, but per_sample
            # doesn't have it. We need to re-extract from the response.
            resp = s.get("response", "")
            match = re.search(r"APPROACH:\s*(\w+)", resp, re.IGNORECASE)
            if match:
                attempted_approaches.add(match.group(1).lower())

        coverage = len(attempted_approaches & expected_approaches) / len(expected_approaches) if expected_approaches else 0.0

        # Reward
        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.4 + coverage * 0.3 + correct_fraction * 0.3
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "coverage": coverage,
            "correct_fraction": correct_fraction,
            "attempted_approaches": list(attempted_approaches),
            "expected_approaches": list(expected_approaches),
            "any_correct": any(corrects),
            "correct_count": sum(corrects),
            "reward": reward,
            "correct": best_score >= 0.8 and coverage >= 0.5,
            "diagnostics": f"Best={best_score:.2f} Coverage={coverage:.0%} ({len(attempted_approaches & expected_approaches)}/{len(expected_approaches)}) Correct={sum(corrects)}/{n}",
        }
