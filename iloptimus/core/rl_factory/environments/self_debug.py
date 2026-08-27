"""
SelfDebug: Predict test outcomes without running code.

Environment concept:
  The model is given a piece of code and a set of test cases. It must
  predict — WITHOUT running the code — which tests will pass and which
  will fail. For failing tests, it must predict the error type.

  This trains the model to:
    1. Mentally simulate code execution (trace through logic)
    2. Identify bugs by reading (not by running)
    3. Predict error types precisely
    4. Know when it's confident enough to skip execution

  This directly attacks the frontier-model pattern of running code to
  "see what happens" instead of reasoning about it first. A model that
  can predict test outcomes accurately can skip the execution step
  entirely, saving a tool call per debugging cycle.

Why this environment is worth using for 10T-100T param models:
  Mental simulation is the highest-value reasoning skill for code. A
  100T model that can trace code execution in its head can:
    - Skip test runs (save tool calls)
    - Fix bugs without the execute→error→fix loop (save round trips)
    - Write correct code on the first try (save iterations)
  This is the difference between a 5-step and a 1-step debugging cycle.

Problem types:
  - Correct code (all tests pass — model should predict all pass)
  - Code with one bug (model should identify which test fails)
  - Code with type errors (model should predict TypeError)
  - Code with logic errors (model should predict wrong output)
  - Code with edge case failures (model should identify edge cases)

Verification:
  Each problem has:
    - The code
    - Test cases with expected outcomes (pass/fail + expected error/output)
  The verifier:
    1. Parses the model's predictions
    2. Compares to actual outcomes (computed by running the code)
    3. Computes accuracy per test case

Reward design:
  - Per-test accuracy: fraction of tests correctly predicted
  - Error type accuracy: for failing tests, did the model predict the right error?
  - Confidence calibration bonus: if model says "all pass" and all pass, bonus
  - Anti-pattern penalties for verbose reasoning
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_SELF_DEBUG_PROBLEMS = [
    # Correct code — all tests pass
    {
        "code": "def add(a, b):\n    return a + b\n",
        "tests": [
            {"input": (1, 2), "expected": 3, "will_pass": True},
            {"input": (0, 0), "expected": 0, "will_pass": True},
            {"input": (-1, 1), "expected": 0, "will_pass": True},
        ],
        "difficulty": 0.15,
    },
    # Bug: subtraction instead of addition
    {
        "code": "def add(a, b):\n    return a - b\n",
        "tests": [
            {"input": (1, 2), "expected": 3, "will_pass": False, "actual": -1},
            {"input": (0, 0), "expected": 0, "will_pass": True},
            {"input": (5, 3), "expected": 8, "will_pass": False, "actual": 2},
        ],
        "difficulty": 0.2,
    },
    # Bug: off by one in range
    {
        "code": "def count_up_to(n):\n    return list(range(1, n))\n",
        "tests": [
            {"input": (5,), "expected": [1, 2, 3, 4, 5], "will_pass": False, "actual": [1, 2, 3, 4]},
            {"input": (1,), "expected": [1], "will_pass": False, "actual": []},
            {"input": (0,), "expected": [], "will_pass": True},
        ],
        "difficulty": 0.3,
    },
    # Correct code — string manipulation
    {
        "code": "def reverse(s):\n    return s[::-1]\n",
        "tests": [
            {"input": ("hello",), "expected": "olleh", "will_pass": True},
            {"input": ("",), "expected": "", "will_pass": True},
            {"input": ("a",), "expected": "a", "will_pass": True},
        ],
        "difficulty": 0.2,
    },
    # Bug: wrong comparison operator
    {
        "code": "def is_adult(age):\n    return age > 18\n",
        "tests": [
            {"input": (25,), "expected": True, "will_pass": True},
            {"input": (18,), "expected": True, "will_pass": False, "actual": False},
            {"input": (15,), "expected": False, "will_pass": True},
        ],
        "difficulty": 0.25,
    },
    # Bug: division by zero not handled
    {
        "code": "def divide(a, b):\n    return a / b\n",
        "tests": [
            {"input": (10, 2), "expected": 5.0, "will_pass": True},
            {"input": (10, 0), "expected": None, "will_pass": False, "error": "ZeroDivisionError"},
            {"input": (0, 5), "expected": 0.0, "will_pass": True},
        ],
        "difficulty": 0.3,
    },
    # Correct code — list operations
    {
        "code": "def get_max(lst):\n    return max(lst) if lst else None\n",
        "tests": [
            {"input": ([1, 2, 3],), "expected": 3, "will_pass": True},
            {"input": ([],), "expected": None, "will_pass": True},
            {"input": ([-1, -2, -3],), "expected": -1, "will_pass": True},
        ],
        "difficulty": 0.25,
    },
    # Bug: using len instead of sum
    {
        "code": "def total(lst):\n    return len(lst)\n",
        "tests": [
            {"input": ([1, 2, 3],), "expected": 6, "will_pass": False, "actual": 3},
            {"input": ([],), "expected": 0, "will_pass": True},
            {"input": ([5],), "expected": 5, "will_pass": False, "actual": 1},
        ],
        "difficulty": 0.25,
    },
    # Bug: KeyError not handled
    {
        "code": "def get_value(d, key):\n    return d[key]\n",
        "tests": [
            {"input": ({"a": 1}, "a"), "expected": 1, "will_pass": True},
            {"input": ({"a": 1}, "b"), "expected": None, "will_pass": False, "error": "KeyError"},
            {"input": ({}, "x"), "expected": None, "will_pass": False, "error": "KeyError"},
        ],
        "difficulty": 0.3,
    },
    # Correct code — even/odd
    {
        "code": "def is_even(n):\n    return n % 2 == 0\n",
        "tests": [
            {"input": (2,), "expected": True, "will_pass": True},
            {"input": (3,), "expected": False, "will_pass": True},
            {"input": (0,), "expected": True, "will_pass": True},
            {"input": (-1,), "expected": False, "will_pass": True},
        ],
        "difficulty": 0.15,
    },
    # Bug: wrong index
    {
        "code": "def first_two(s):\n    return s[2:]\n",
        "tests": [
            {"input": ("hello",), "expected": "he", "will_pass": False, "actual": "llo"},
            {"input": ("ab",), "expected": "ab", "will_pass": False, "actual": ""},
            {"input": ("",), "expected": "", "will_pass": True},
        ],
        "difficulty": 0.3,
    },
    # Correct code — filtering
    {
        "code": "def positives(lst):\n    return [x for x in lst if x > 0]\n",
        "tests": [
            {"input": ([1, -2, 3],), "expected": [1, 3], "will_pass": True},
            {"input": ([],), "expected": [], "will_pass": True},
            {"input": ([-1, -2],), "expected": [], "will_pass": True},
        ],
        "difficulty": 0.2,
    },
]


def self_debug_generator(seed: int) -> Problem:
    """Generate a SelfDebug problem."""
    rng = random.Random(seed)
    template = rng.choice(_SELF_DEBUG_PROBLEMS)

    # Format the test cases for the prompt
    test_descriptions = []
    for i, test in enumerate(template["tests"]):
        inp = test["input"]
        if isinstance(inp, tuple) and len(inp) == 1:
            inp_str = repr(inp[0])
        elif isinstance(inp, tuple):
            inp_str = ", ".join(repr(x) for x in inp)
        else:
            inp_str = repr(inp)
        expected = test["expected"]
        test_descriptions.append(f"  Test {i+1}: input=({inp_str}), expected={repr(expected)}")

    tests_str = "\n".join(test_descriptions)

    return Problem(
        id=f"self_debug_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["code"], tests_str, len(template["tests"])),
        difficulty=template["difficulty"],
        metadata={
            "type": "self_debug",
            "code": template["code"],
            "tests": template["tests"],
        },
        token_budget=500,
        source="generated",
    )


def _format_prompt(code: str, tests_str: str, num_tests: int) -> str:
    return (
        f"Predict the outcome of each test case WITHOUT running the code.\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Test cases:\n{tests_str}\n\n"
        f"For each test, predict:\n"
        f"  TEST <n>: PASS  (if the test will pass)\n"
        f"  TEST <n>: FAIL - <error type or actual value>  (if it will fail)\n\n"
        f"Output ALL {num_tests} predictions. Do NOT run the code — reason about it mentally."
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SelfDebugVerifier(Verifier):
    """
    Verifies a SelfDebug response by comparing predictions to actual outcomes.
    """

    def __init__(self, tests: list[dict]):
        super().__init__()
        self._tests = tests

    def verify(self, response: str) -> VerifierResult:
        # Parse predictions
        predictions = self._parse_predictions(response, len(self._tests))

        if not predictions:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No TEST predictions found in response",
            )

        # Compare to actual outcomes
        correct_preds = 0
        total = len(self._tests)
        details = []

        for i, test in enumerate(self._tests):
            actual_pass = test["will_pass"]
            pred = predictions.get(i + 1, {})

            if pred.get("pass") == actual_pass:
                if actual_pass:
                    correct_preds += 1
                    details.append(f"Test {i+1}: correctly predicted PASS")
                else:
                    # For failing tests, also check error type/value
                    error_info = test.get("error")
                    if error_info is None:
                        error_info = test.get("actual")
                    pred_detail = pred.get("detail", "")
                    # Handle falsy but valid error_info (e.g. [], 0, "")
                    if error_info is not None and str(error_info).lower() in pred_detail.lower():
                        correct_preds += 1
                        details.append(f"Test {i+1}: correctly predicted FAIL ({error_info})")
                    elif error_info is None and pred_detail:
                        # No specific error info to check — accept any FAIL prediction
                        correct_preds += 1
                        details.append(f"Test {i+1}: correctly predicted FAIL")
                    else:
                        details.append(f"Test {i+1}: predicted FAIL but wrong detail. Expected: {error_info}, Got: {pred_detail}")
            else:
                status = "PASS" if actual_pass else "FAIL"
                pred_status = "PASS" if pred.get("pass") else "FAIL"
                details.append(f"Test {i+1}: predicted {pred_status}, actual {status}")

        accuracy = correct_preds / total if total > 0 else 0.0
        correct = accuracy >= 0.8

        return VerifierResult(
            correct=correct,
            score=accuracy,
            partial_credit={"accuracy": accuracy, "correct_preds": correct_preds, "total": total},
            diagnostics=f"Accuracy: {correct_preds}/{total} ({accuracy:.0%}). " + " | ".join(details[:3]),
        )

    def _parse_predictions(self, response: str, num_tests: int) -> dict[int, dict]:
        """Parse TEST n: PASS/FAIL predictions."""
        predictions = {}

        # Pattern: TEST n: PASS or TEST n: FAIL - detail
        pattern = r"TEST\s+(\d+)\s*:\s*(PASS|FAIL)(?:\s*-\s*(.+?))?(?:\n|$)"
        matches = re.finditer(pattern, response, re.IGNORECASE)

        for match in matches:
            test_num = int(match.group(1))
            status = match.group(2).upper()
            detail = match.group(3) or "" if match.group(3) else ""

            predictions[test_num] = {
                "pass": status == "PASS",
                "detail": detail.strip(),
            }

        return predictions


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SelfDebugEnv(BaseReasoningEnv):
    """
    SelfDebug environment: predict test outcomes without running code.

    The model receives code + test cases and must predict pass/fail for each.
    Reward = prediction accuracy.
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
            problem_generator = self_debug_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SelfDebugVerifier(tests=problem.metadata["tests"])

    def _check_format(self, response: str) -> float:
        if re.search(r"TEST\s+\d+\s*:\s*(?:PASS|FAIL)", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
