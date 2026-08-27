"""
RefactorArena: Produce equivalent but more concise code.

Environment concept:
  The model is given a piece of verbose, correct code and must produce a
  refactor that is:
    1. Functionally equivalent (passes all the same test cases)
    2. Shorter (fewer tokens / fewer lines)
    3. Still readable (no golf-style obfuscation)

  This trains the model to:
    - Identify redundant logic that can be collapsed
    - Use language idioms instead of verbose patterns
    - Remove unnecessary intermediate variables
    - Replace verbose conditionals with concise equivalents
    - Eliminate dead code and unreachable branches

Why this environment is unique:
  Most code RL environments ask the model to WRITE code from scratch or
  FIX bugs. RefactorArena asks the model to COMPRESS existing code while
  preserving behavior. This directly trains the "code density" skill —
  saying the same thing in fewer tokens — which transfers to reasoning
  conciseness.

Problem types:
  - Verbose loops that can be comprehensions
  - Repeated if/elif chains that can be dict dispatch
  - Unnecessary intermediate variables
  - Manual implementations of built-in operations
  - Verbose string formatting that can use f-strings
  - Redundant null checks
  - Unwrapped repeated patterns

Verification:
  Each problem includes:
    - The original verbose code
    - A set of test cases
    - The original token count
  The verifier:
    1. Executes the refactored code against all test cases
    2. Counts the tokens in the refactored code
    3. Computes compression_ratio = original_tokens / refactored_tokens

Reward design:
  - Correctness gate: all tests must pass
  - If correct: reward = 1.0 * compression_factor
    where compression_factor = min(1.5, refactored_tokens / original_tokens)
    (capped at 1.5x to prevent over-compression rewards)
  - Penalty if the refactored code is LONGER than the original
  - Anti-pattern penalties for verbose explanations
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank
# ---------------------------------------------------------------------------


_REFACTOR_PROBLEMS = [
    {
        "verbose": """def get_evens(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num)
    return result""",
        "concise": """def get_evens(numbers):
    return [n for n in numbers if n % 2 == 0]""",
        "tests": [
            ([1, 2, 3, 4, 5, 6], [2, 4, 6]),
            ([], []),
            ([1, 3, 5], []),
            ([2, 4, 6], [2, 4, 6]),
        ],
        "difficulty": 0.2,
    },
    {
        "verbose": """def double_all(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result""",
        "concise": """def double_all(items):
    return [x * 2 for x in items]""",
        "tests": [
            ([1, 2, 3], [2, 4, 6]),
            ([], []),
            ([0], [0]),
            ([-1, -2], [-2, -4]),
        ],
        "difficulty": 0.15,
    },
    {
        "verbose": """def count_words(sentence):
    words = sentence.split()
    count = 0
    for word in words:
        count = count + 1
    return count""",
        "concise": """def count_words(sentence):
    return len(sentence.split())""",
        "tests": [
            ("hello world", 2),
            ("", 0),
            ("one", 1),
            ("a b c d e", 5),
        ],
        "difficulty": 0.2,
    },
    {
        "verbose": """def get_value(mapping, key, default):
    if key in mapping:
        return mapping[key]
    else:
        return default""",
        "concise": """def get_value(mapping, key, default):
    return mapping.get(key, default)""",
        "tests": [
            ({"a": 1}, "a", 0, 1),
            ({"a": 1}, "b", 0, 0),
            ({}, "x", "none", "none"),
        ],
        "difficulty": 0.25,
    },
    {
        "verbose": """def is_empty(lst):
    if len(lst) == 0:
        return True
    else:
        return False""",
        "concise": """def is_empty(lst):
    return len(lst) == 0""",
        "tests": [
            ([], True),
            ([1], False),
            ([1, 2, 3], False),
        ],
        "difficulty": 0.15,
    },
    {
        "verbose": """def merge_lists(a, b):
    result = []
    for item in a:
        result.append(item)
    for item in b:
        result.append(item)
    return result""",
        "concise": """def merge_lists(a, b):
    return a + b""",
        "tests": [
            ([1, 2], [3, 4], [1, 2, 3, 4]),
            ([], [], []),
            ([1], [], [1]),
            ([], [2], [2]),
        ],
        "difficulty": 0.2,
    },
    {
        "verbose": """def square_if_positive(numbers):
    result = []
    for n in numbers:
        if n > 0:
            result.append(n * n)
        else:
            result.append(n)
    return result""",
        "concise": """def square_if_positive(numbers):
    return [n*n if n > 0 else n for n in numbers]""",
        "tests": [
            ([1, -2, 3], [1, -2, 9]),
            ([], []),
            ([-1, -2], [-1, -2]),
            ([2, 4], [4, 16]),
        ],
        "difficulty": 0.35,
    },
    {
        "verbose": """def to_uppercase(strings):
    result = []
    for s in strings:
        result.append(s.upper())
    return result""",
        "concise": """def to_uppercase(strings):
    return [s.upper() for s in strings]""",
        "tests": [
            (["a", "b"], ["A", "B"]),
            ([], []),
            (["hello"], ["HELLO"]),
        ],
        "difficulty": 0.15,
    },
    {
        "verbose": """def filter_and_square(numbers):
    result = []
    for n in numbers:
        if n > 0:
            squared = n * n
            result.append(squared)
    return result""",
        "concise": """def filter_and_square(numbers):
    return [n*n for n in numbers if n > 0]""",
        "tests": [
            ([1, -2, 3, -4], [1, 9]),
            ([], []),
            ([-1, -2], []),
            ([2, 3], [4, 9]),
        ],
        "difficulty": 0.3,
    },
    {
        "verbose": """def sum_positive(numbers):
    total = 0
    for n in numbers:
        if n > 0:
            total = total + n
    return total""",
        "concise": """def sum_positive(numbers):
    return sum(n for n in numbers if n > 0)""",
        "tests": [
            ([1, -2, 3], 4),
            ([], 0),
            ([-1, -2], 0),
            ([5, 10, -3], 15),
        ],
        "difficulty": 0.25,
    },
    {
        "verbose": """def classify_number(n):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero\"""",
        "concise": """def classify_number(n):
    return "positive" if n > 0 else "negative" if n < 0 else "zero\"""",
        "tests": [
            (5, "positive"),
            (-3, "negative"),
            (0, "zero"),
        ],
        "difficulty": 0.3,
    },
    {
        "verbose": """def create_pairs(list1, list2):
    result = []
    for i in range(len(list1)):
        pair = (list1[i], list2[i])
        result.append(pair)
    return result""",
        "concise": """def create_pairs(list1, list2):
    return list(zip(list1, list2))""",
        "tests": [
            ([1, 2], [3, 4], [(1, 3), (2, 4)]),
            ([], [], []),
            (["a"], ["b"], [("a", "b")]),
        ],
        "difficulty": 0.35,
    },
    {
        "verbose": """def all_positive(numbers):
    for n in numbers:
        if n <= 0:
            return False
    return True""",
        "concise": """def all_positive(numbers):
    return all(n > 0 for n in numbers)""",
        "tests": [
            ([1, 2, 3], True),
            ([1, -2, 3], False),
            ([], True),
            ([0], False),
        ],
        "difficulty": 0.3,
    },
    {
        "verbose": """def any_even(numbers):
    for n in numbers:
        if n % 2 == 0:
            return True
    return False""",
        "concise": """def any_even(numbers):
    return any(n % 2 == 0 for n in numbers)""",
        "tests": [
            ([1, 2, 3], True),
            ([1, 3, 5], False),
            ([], False),
            ([2], True),
        ],
        "difficulty": 0.3,
    },
    {
        "verbose": """def reverse_string(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result = result + s[i]
    return result""",
        "concise": """def reverse_string(s):
    return s[::-1]""",
        "tests": [
            ("hello", "olleh"),
            ("", ""),
            ("a", "a"),
            ("ab", "ba"),
        ],
        "difficulty": 0.3,
    },
]


def refactor_arena_generator(seed: int) -> Problem:
    """Generate a RefactorArena problem from the bank."""
    rng = random.Random(seed)
    entry = rng.choice(_REFACTOR_PROBLEMS)

    verbose_tokens = estimate_tokens(entry["verbose"])
    concise_tokens = estimate_tokens(entry["concise"])

    return Problem(
        id=f"refactor_{seed}_{rng.randint(0, 99999)}",
        prompt=_format_prompt(entry["verbose"], entry["tests"], verbose_tokens),
        difficulty=entry["difficulty"],
        metadata={
            "type": "refactor",
            "verbose_code": entry["verbose"],
            "concise_code": entry["concise"],
            "tests": entry["tests"],
            "verbose_tokens": verbose_tokens,
            "concise_tokens": concise_tokens,
        },
        token_budget=1200,
        source="generated",
    )


def _format_prompt(verbose_code: str, tests: list, verbose_tokens: int) -> str:
    """Format the problem prompt."""
    test_str = "\n".join(
        f"  Input: {t[:-1]} → Expected: {t[-1]}" for t in tests
    )
    return (
        f"Refactor the following code to be SHORTER while preserving behavior.\n\n"
        f"```python\n{verbose_code}\n```\n\n"
        f"Test cases (your refactor must pass all):\n{test_str}\n\n"
        f"Original code: {verbose_tokens} tokens\n"
        f"Rules:\n"
        f"  - Output ONLY the refactored code in a ```python block\n"
        f"  - The refactored code must pass ALL test cases\n"
        f"  - Shorter = better (fewer tokens)\n"
        f"  - Keep it readable — no golf-style obfuscation\n"
        f"  - Do NOT explain — just output the code"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class RefactorVerifier(Verifier):
    """
    Verifies a refactored code by:
      1. Running it against all test cases
      2. Comparing token count to the original
    """

    def __init__(self, tests: list, original_tokens: int):
        super().__init__()
        self._tests = tests
        self._original_tokens = original_tokens

    def verify(self, response: str) -> VerifierResult:
        # Extract code from ```python block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)

        if not code_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No code block found in response",
            )

        refactored_code = code_match.group(1).strip()

        # Execute the refactored code
        namespace: dict[str, Any] = {}
        try:
            exec(refactored_code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Code execution error: {type(e).__name__}: {e}",
            )

        # Find the function (exclude builtins and imports)
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "__builtins__":
                if hasattr(obj, "__module__") and obj.__module__ in ("builtins",):
                    continue
                func = obj
                break

        if func is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No function found in the code",
            )

        # Run tests
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
                    failures.append(f"Input {test[:-1]}: expected {expected}, got {result}")
            except Exception as e:
                failures.append(f"Input {test[:-1]}: raised {type(e).__name__}: {e}")

        if passed < total:
            return VerifierResult(
                correct=False,
                score=passed / total,
                diagnostics=f"Tests: {passed}/{total} passed. Failures: {'; '.join(failures[:3])}",
            )

        # All tests pass — compute compression
        refactored_tokens = estimate_tokens(refactored_code)

        if refactored_tokens < self._original_tokens:
            # Compression achieved — score based on ratio
            compression_ratio = self._original_tokens / max(refactored_tokens, 1)
            # Cap at 1.5x to prevent over-compression rewards
            score = min(1.0, (compression_ratio - 1.0) / 0.5) * 0.5 + 0.5
        elif refactored_tokens == self._original_tokens:
            score = 0.5  # same size, no improvement
        else:
            # Refactored code is LONGER — penalty
            expansion_ratio = refactored_tokens / self._original_tokens
            score = max(0.0, 0.5 - (expansion_ratio - 1.0) * 0.5)

        return VerifierResult(
            correct=True,
            score=score,
            partial_credit={
                "test_pass_rate": 1.0,
                "compression_ratio": self._original_tokens / max(refactored_tokens, 1),
            },
            diagnostics=f"All {total} tests passed. Tokens: {refactored_tokens} (original: {self._original_tokens}). Score: {score:.2f}",
            metadata={"refactored_tokens": refactored_tokens, "original_tokens": self._original_tokens},
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RefactorArenaEnv(BaseReasoningEnv):
    """
    RefactorArena environment: produce equivalent but more concise code.

    The model receives verbose code + test cases and must produce a shorter
    equivalent. Reward = correctness * compression_factor.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
    ):
        if problems is None and problem_generator is None:
            problem_generator = refactor_arena_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return RefactorVerifier(
            tests=problem.metadata["tests"],
            original_tokens=problem.metadata["verbose_tokens"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if re.search(r"```\n.*?```", response, re.DOTALL):
            return 0.5
        return 0.0

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response
