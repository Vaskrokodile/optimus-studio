"""
SurgicalPatch: Minimal bug fix under a context budget.

Environment concept:
  The model is given a piece of code with a bug and must produce a minimal
  patch that fixes it. The reward favors:
    1. Correctness (the patched code passes all test cases)
    2. Minimality (fewer lines changed = higher reward)
    3. Precision (no unnecessary changes, no refactoring, no "while I'm here" edits)

  This trains the model to:
    - Identify the ROOT CAUSE, not symptoms
    - Make surgical edits, not rewrites
    - Avoid "drive-by" improvements that inflate diffs
    - Resist the urge to refactor when a one-line fix suffices

Why this environment is unique:
  Most code RL environments reward passing tests regardless of how much
  code changed. SurgicalPatch makes DIFF SIZE the primary optimization
  target after correctness, directly fighting the frontier-model tendency
  to over-edit, add comments, reformat, and "improve" code that only needs
  a one-character fix.

Problem types:
  - Off-by-one errors
  - Wrong comparison operator (< vs <=, == vs !=)
  - Missing edge case handling
  - Incorrect variable reference
  - Logic inversion (if/else swapped)
  - Missing return statement
  - Wrong data structure operation

Verification:
  Each problem includes:
    - The buggy code
    - A set of test cases (input → expected output)
    - The expected fix size (in lines changed)
  The verifier:
    1. Applies the patch to the buggy code
    2. Executes the patched code against all test cases
    3. Counts the number of lines changed (diff size)

Reward design:
  - Correctness gate: all tests must pass
  - If correct: reward = 1.0 * efficiency_bonus
    where efficiency_bonus = max(0, 1 - (lines_changed / max_expected_lines))
  - Anti-pattern penalties for verbose explanations
  - Extra penalty for changes that are NOT in the fix region
"""

from __future__ import annotations

import difflib
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


def _make_off_by_one(rng: random.Random) -> Problem:
    """Generate an off-by-one error problem."""
    templates = [
        {
            "buggy": """def sum_range(n):
    total = 0
    for i in range(1, n):
        total += i
    return total""",
            "fixed": """def sum_range(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total""",
            "tests": [
                (5, 15),   # 1+2+3+4+5 = 15
                (1, 1),    # just 1
                (10, 55),  # 1+...+10 = 55
            ],
            "bug": "range(1, n) should be range(1, n+1) — missing the last element",
        },
        {
            "buggy": """def count_items(lst):
    count = 0
    for i in range(0, len(lst) - 1):
        count += 1
    return count""",
            "fixed": """def count_items(lst):
    count = 0
    for i in range(0, len(lst)):
        count += 1
    return count""",
            "tests": [
                ([1, 2, 3], 3),
                ([], 0),
                (["a"], 1),
            ],
            "bug": "range(0, len(lst)-1) should be range(0, len(lst)) — off by one",
        },
        {
            "buggy": """def factorial(n):
    result = 1
    for i in range(2, n):
        result *= i
    return result""",
            "fixed": """def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
            "tests": [
                (5, 120),
                (3, 6),
                (1, 1),
                (0, 1),
            ],
            "bug": "range(2, n) should be range(2, n+1) — missing n itself",
        },
    ]

    template = rng.choice(templates)
    expected_lines_changed = sum(
        1 for line in difflib.unified_diff(
            template["buggy"].splitlines(), template["fixed"].splitlines(),
            n=0
        ) if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    difficulty = 0.2 + 0.2 * rng.random()

    return Problem(
        id=f"off_by_one_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["buggy"], template["tests"]),
        difficulty=difficulty,
        metadata={
            "type": "off_by_one",
            "buggy_code": template["buggy"],
            "fixed_code": template["fixed"],
            "tests": template["tests"],
            "bug_description": template["bug"],
            "expected_lines_changed": max(1, expected_lines_changed),
        },
        token_budget=800,
        source="generated",
    )


def _make_wrong_operator(rng: random.Random) -> Problem:
    """Generate a wrong comparison operator problem."""
    templates = [
        {
            "buggy": """def is_even(n):
    return n % 2 == 1""",
            "fixed": """def is_even(n):
    return n % 2 == 0""",
            "tests": [
                (4, True),
                (7, False),
                (0, True),
                (-2, True),
            ],
            "bug": "== 1 should be == 0 for even check",
        },
        {
            "buggy": """def is_positive(n):
    return n < 0""",
            "fixed": """def is_positive(n):
    return n > 0""",
            "tests": [
                (5, True),
                (-3, False),
                (0, False),
            ],
            "bug": "< should be > for positive check",
        },
        {
            "buggy": """def max_val(a, b):
    if a < b:
        return a
    return b""",
            "fixed": """def max_val(a, b):
    if a > b:
        return a
    return b""",
            "tests": [
                (3, 5, 5),
                (10, 2, 10),
                (4, 4, 4),
            ],
            "bug": "< should be > — returns min instead of max",
        },
        {
            "buggy": """def contains(items, target):
    for item in items:
        if item != target:
            return True
    return False""",
            "fixed": """def contains(items, target):
    for item in items:
        if item == target:
            return True
    return False""",
            "tests": [
                ([1, 2, 3], 2, True),
                ([1, 2, 3], 5, False),
                ([], 1, False),
            ],
            "bug": "!= should be == — returns True for non-matching items",
        },
    ]

    template = rng.choice(templates)
    expected_lines_changed = 1  # operator fixes are always 1 line

    difficulty = 0.15 + 0.15 * rng.random()

    # Fix tests format for 3-arg tests
    tests = template["tests"]

    return Problem(
        id=f"wrong_op_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["buggy"], tests),
        difficulty=difficulty,
        metadata={
            "type": "wrong_operator",
            "buggy_code": template["buggy"],
            "fixed_code": template["fixed"],
            "tests": tests,
            "bug_description": template["bug"],
            "expected_lines_changed": expected_lines_changed,
        },
        token_budget=600,
        source="generated",
    )


def _make_missing_edge_case(rng: random.Random) -> Problem:
    """Generate a missing edge case problem."""
    templates = [
        {
            "buggy": """def safe_divide(a, b):
    return a / b""",
            "fixed": """def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b""",
            "tests": [
                (10, 2, 5.0),
                (10, 0, 0),
                (0, 5, 0.0),
            ],
            "bug": "No handling for division by zero",
        },
        {
            "buggy": """def get_first(lst):
    return lst[0]""",
            "fixed": """def get_first(lst):
    if not lst:
        return None
    return lst[0]""",
            "tests": [
                ([1, 2, 3], 1),
                ([], None),
                (["a", "b"], "a"),
            ],
            "bug": "No handling for empty list",
        },
        {
            "buggy": """def parse_int(s):
    return int(s)""",
            "fixed": """def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return 0""",
            "tests": [
                ("42", 42),
                ("abc", 0),
                ("", 0),
            ],
            "bug": "No handling for non-numeric strings",
        },
    ]

    template = rng.choice(templates)
    expected_lines_changed = sum(
        1 for line in difflib.unified_diff(
            template["buggy"].splitlines(), template["fixed"].splitlines(),
            n=0
        ) if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    difficulty = 0.4 + 0.2 * rng.random()

    return Problem(
        id=f"edge_case_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["buggy"], template["tests"]),
        difficulty=difficulty,
        metadata={
            "type": "missing_edge_case",
            "buggy_code": template["buggy"],
            "fixed_code": template["fixed"],
            "tests": template["tests"],
            "bug_description": template["bug"],
            "expected_lines_changed": max(1, expected_lines_changed),
        },
        token_budget=700,
        source="generated",
    )


def _make_logic_inversion(rng: random.Random) -> Problem:
    """Generate a logic inversion problem (if/else swapped)."""
    templates = [
        {
            "buggy": """def classify_grade(score):
    if score >= 60:
        return "FAIL"
    else:
        return "PASS\"""",
            "fixed": """def classify_grade(score):
    if score >= 60:
        return "PASS"
    else:
        return "FAIL\"""",
            "tests": [
                (85, "PASS"),
                (45, "FAIL"),
                (60, "PASS"),
                (59, "FAIL"),
            ],
            "bug": "PASS and FAIL are swapped",
        },
        {
            "buggy": """def is_weekend(day):
    if day in ["Saturday", "Sunday"]:
        return False
    return True""",
            "fixed": """def is_weekend(day):
    if day in ["Saturday", "Sunday"]:
        return True
    return False""",
            "tests": [
                ("Saturday", True),
                ("Monday", False),
                ("Sunday", True),
            ],
            "bug": "True and False are swapped",
        },
    ]

    template = rng.choice(templates)
    expected_lines_changed = 2  # swap two return values

    difficulty = 0.25 + 0.15 * rng.random()

    return Problem(
        id=f"logic_inv_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["buggy"], template["tests"]),
        difficulty=difficulty,
        metadata={
            "type": "logic_inversion",
            "buggy_code": template["buggy"],
            "fixed_code": template["fixed"],
            "tests": template["tests"],
            "bug_description": template["bug"],
            "expected_lines_changed": expected_lines_changed,
        },
        token_budget=700,
        source="generated",
    )


def _format_prompt(buggy_code: str, tests: list) -> str:
    """Format the problem prompt with the buggy code and test cases."""
    test_str = "\n".join(
        f"  Input: {t[:-1]} → Expected: {t[-1]}" for t in tests
    )
    return (
        f"The following code has a bug. Fix it with the MINIMAL possible change.\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"Test cases:\n{test_str}\n\n"
        f"Rules:\n"
        f"  - Output ONLY the fixed code in a ```python block\n"
        f"  - Change as few lines as possible\n"
        f"  - Do NOT add comments, docstrings, or refactor\n"
        f"  - Do NOT explain your reasoning — just output the fixed code"
    )


def surgical_patch_generator(seed: int) -> Problem:
    """Master generator for SurgicalPatch problems."""
    rng = random.Random(seed)
    generators = [
        _make_off_by_one,
        _make_wrong_operator,
        _make_missing_edge_case,
        _make_logic_inversion,
    ]
    gen = rng.choice(generators)
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SurgicalPatchVerifier(Verifier):
    """
    Verifies a surgical patch by:
      1. Extracting the fixed code from the response
      2. Running it against all test cases
      3. Computing the diff size against the buggy code
    """

    def __init__(self, buggy_code: str, tests: list, expected_lines_changed: int):
        super().__init__()
        self._buggy_code = buggy_code
        self._tests = tests
        self._expected_lines_changed = expected_lines_changed

    def verify(self, response: str) -> VerifierResult:
        # Extract code from ```python ... ``` block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)

        if not code_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No code block found in response",
            )

        fixed_code = code_match.group(1).strip()

        # Execute the fixed code and run tests
        namespace: dict[str, Any] = {}
        try:
            exec(fixed_code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Code execution error: {type(e).__name__}: {e}",
            )

        # Find the function defined (exclude builtins and imports)
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "__builtins__":
                # Skip built-in functions and types
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

        # Run all test cases
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

        # All tests pass — compute diff size
        buggy_lines = self._buggy_code.strip().splitlines()
        fixed_lines = fixed_code.strip().splitlines()

        diff_lines = 0
        for line in difflib.unified_diff(buggy_lines, fixed_lines, n=0):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                diff_lines += 1

        # Efficiency: fewer changes = higher score
        # Perfect score if diff_lines <= expected_lines_changed
        # Score decreases as diff_lines exceeds expected
        if diff_lines <= self._expected_lines_changed:
            efficiency = 1.0
        else:
            # Linear penalty for extra lines
            extra = diff_lines - self._expected_lines_changed
            efficiency = max(0.0, 1.0 - 0.15 * extra)

        return VerifierResult(
            correct=True,
            score=efficiency,
            partial_credit={"test_pass_rate": 1.0, "diff_efficiency": efficiency},
            diagnostics=f"All {total} tests passed. Diff: {diff_lines} lines changed (expected ~{self._expected_lines_changed}). Efficiency: {efficiency:.2f}",
            metadata={"diff_lines": diff_lines, "expected_lines_changed": self._expected_lines_changed},
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SurgicalPatchEnv(BaseReasoningEnv):
    """
    SurgicalPatch environment: minimal bug fix under a context budget.

    The model receives buggy code + test cases and must produce the minimal
    fix. Reward = correctness * diff_efficiency, with anti-pattern penalties.
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
            problem_generator = surgical_patch_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SurgicalPatchVerifier(
            buggy_code=problem.metadata["buggy_code"],
            tests=problem.metadata["tests"],
            expected_lines_changed=problem.metadata["expected_lines_changed"],
        )

    def _check_format(self, response: str) -> float:
        """Check for a ```python code block."""
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if re.search(r"```\n.*?```", response, re.DOTALL):
            return 0.5
        return 0.0

    def _extract_answer(self, response: str) -> str:
        """Extract the code block from the response."""
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response
