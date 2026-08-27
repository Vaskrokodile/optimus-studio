"""
DebuggingRace: Find the bug and fix it.

Environment concept:
  The model is given buggy code and must identify the bug's location
  (line number) and provide a fix. This trains precise debugging —
  not just "find a bug" but pinpoint the exact line and correct it.

  Why: debugging is a core engineering skill. This environment trains
  the model to localize faults precisely and produce minimal fixes,
  rather than rewriting entire functions.

Verification:
  - Bug line identified correctly.
  - Fix produces correct output on test inputs.

Reward design:
  line_correct * 0.4 + fix_correct * 0.6
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — correct code, buggy code (with bug line), test inputs
# ---------------------------------------------------------------------------

_DEBUG_PROBLEMS = [
    {
        "correct_code": (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        "buggy_code": (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        ),
        "bug_line": 3,
        "func_name": "sum_range",
        "test_inputs": [5, 10, 1],
        "expected_outputs": [15, 55, 1],
        "bug_description": "range should be range(1, n+1) not range(1, n)",
    },
    {
        "correct_code": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "buggy_code": (
            "def factorial(n):\n"
            "    result = 0\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "bug_line": 2,
        "func_name": "factorial",
        "test_inputs": [5, 3, 1],
        "expected_outputs": [120, 6, 1],
        "bug_description": "result should be initialized to 1 not 0",
    },
    {
        "correct_code": (
            "def max_element(arr):\n"
            "    best = arr[0]\n"
            "    for x in arr:\n"
            "        if x > best:\n"
            "            best = x\n"
            "    return best\n"
        ),
        "buggy_code": (
            "def max_element(arr):\n"
            "    best = arr[0]\n"
            "    for x in arr:\n"
            "        if x < best:\n"
            "            best = x\n"
            "    return best\n"
        ),
        "bug_line": 4,
        "func_name": "max_element",
        "test_inputs": [[3, 1, 4, 1, 5], [10, 20, 5]],
        "expected_outputs": [5, 20],
        "bug_description": "should be x > best not x < best",
    },
    {
        "correct_code": (
            "def count_down(n):\n"
            "    result = []\n"
            "    for i in range(n, 0, -1):\n"
            "        result.append(i)\n"
            "    return result\n"
        ),
        "buggy_code": (
            "def count_down(n):\n"
            "    result = []\n"
            "    for i in range(n, 0, 1):\n"
            "        result.append(i)\n"
            "    return result\n"
        ),
        "bug_line": 3,
        "func_name": "count_down",
        "test_inputs": [5, 3],
        "expected_outputs": [[5, 4, 3, 2, 1], [3, 2, 1]],
        "bug_description": "step should be -1 not 1",
    },
    {
        "correct_code": (
            "def is_even(n):\n"
            "    return n % 2 == 0\n"
        ),
        "buggy_code": (
            "def is_even(n):\n"
            "    return n % 2 == 1\n"
        ),
        "bug_line": 2,
        "func_name": "is_even",
        "test_inputs": [4, 7, 0],
        "expected_outputs": [True, False, True],
        "bug_description": "should check == 0 not == 1",
    },
    {
        "correct_code": (
            "def reverse_list(arr):\n"
            "    result = []\n"
            "    for x in arr:\n"
            "        result.insert(0, x)\n"
            "    return result\n"
        ),
        "buggy_code": (
            "def reverse_list(arr):\n"
            "    result = []\n"
            "    for x in arr:\n"
            "        result.append(x)\n"
            "    return result\n"
        ),
        "bug_line": 4,
        "func_name": "reverse_list",
        "test_inputs": [[1, 2, 3], [5, 6]],
        "expected_outputs": [[3, 2, 1], [6, 5]],
        "bug_description": "should insert at 0 not append",
    },
]


def _extract_code_block(response: str) -> str:
    """Extract Python code from a markdown code block or the FIX line."""
    # Try FIX: ```python ... ```
    fix_match = re.search(r"FIX\s*:\s*```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if fix_match:
        return fix_match.group(1).strip()
    # Try FIX: <code>
    fix_match = re.search(r"FIX\s*:\s*(.+?)(?:\n```|\nBUG_LINE|\Z)", response, re.DOTALL | re.IGNORECASE)
    if fix_match:
        code = fix_match.group(1).strip()
        if code.startswith("```"):
            code = re.sub(r"^```(?:python)?\s*\n", "", code)
            code = re.sub(r"```$", "", code).strip()
        return code
    # Try any code block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def debugging_race_generator(seed: int) -> Problem:
    """Generate a DebuggingRace problem: find the bug and fix it."""
    rng = random.Random(seed)
    problem = rng.choice(_DEBUG_PROBLEMS)

    buggy_code = problem["buggy_code"]
    correct_code = problem["correct_code"]
    bug_line = problem["bug_line"]
    func_name = problem["func_name"]
    test_inputs = problem["test_inputs"]
    expected_outputs = problem["expected_outputs"]

    # Show buggy code with line numbers
    buggy_lines = buggy_code.split("\n")
    numbered = "\n".join(f"{i+1}: {ln}" for i, ln in enumerate(buggy_lines))

    prompt = (
        f"The following code has a bug. Find the buggy line and provide a fix.\n\n"
        f"```python\n{numbered}\n```\n\n"
        f"Identify the bug's line number and write the corrected function.\n"
        f"Format your answer as:\n"
        f"BUG_LINE: <line number>\n"
        f"FIX: <corrected code in a python code block>"
    )

    difficulty = 0.5

    return Problem(
        id=f"debugging_race_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "correct_code": correct_code,
            "buggy_code": buggy_code,
            "bug_line": bug_line,
            "func_name": func_name,
            "test_inputs": test_inputs,
            "expected_outputs": expected_outputs,
            "bug_description": problem["bug_description"],
        },
        token_budget=1024,
        source="debugging_race_generator",
    )


debugging_race_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class DebuggingRaceVerifier(Verifier):
    """Verify a debugging race: bug line correct + fix produces correct output.

    reward = line_correct * 0.4 + fix_correct * 0.6
    """

    def __init__(
        self,
        bug_line: int,
        func_name: str,
        test_inputs: list,
        expected_outputs: list,
    ):
        super().__init__()
        self._bug_line = bug_line
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._expected_outputs = expected_outputs

    def verify(self, response: str) -> VerifierResult:
        # Parse bug line
        line_match = re.search(r"BUG_LINE\s*:\s*(\d+)", response, re.IGNORECASE)
        given_line = int(line_match.group(1)) if line_match else -1
        line_correct = given_line == self._bug_line

        # Parse and test fix
        fix_code = _extract_code_block(response)
        fix_correct = False
        fix_error = ""

        if fix_code:
            namespace: dict[str, Any] = {}
            try:
                exec(fix_code, namespace)
                if self._func_name in namespace and callable(namespace[self._func_name]):
                    func = namespace[self._func_name]
                    correct_count = 0
                    total = len(self._test_inputs)
                    for ti, expected in zip(self._test_inputs, self._expected_outputs):
                        try:
                            result = func(ti)
                            if result == expected:
                                correct_count += 1
                            else:
                                fix_error = f"got {result!r}, expected {expected!r}"
                        except Exception as e:
                            fix_error = f"{type(e).__name__}: {e}"
                    fix_correct = correct_count == total
                else:
                    fix_error = f"function '{self._func_name}' not found"
            except Exception as e:
                fix_error = f"exec error: {type(e).__name__}: {e}"

        score = (1.0 if line_correct else 0.0) * 0.4 + (1.0 if fix_correct else 0.0) * 0.6
        correct = line_correct and fix_correct

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "line_correct": 1.0 if line_correct else 0.0,
                "fix_correct": 1.0 if fix_correct else 0.0,
            },
            diagnostics=(
                f"line={given_line} (expected {self._bug_line}) "
                f"fix_correct={fix_correct} {fix_error}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DebuggingRaceEnv(BatchEnvBase):
    """DebuggingRace: find the bug and fix it.

    Batch-aware: N parallel attempts; reward = best debug attempt.
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
            problem_generator = debugging_race_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return DebuggingRaceVerifier(
            bug_line=problem.metadata["bug_line"],
            func_name=problem.metadata["func_name"],
            test_inputs=problem.metadata["test_inputs"],
            expected_outputs=problem.metadata["expected_outputs"],
        )

    def _check_format(self, response: str) -> float:
        has_bug = bool(re.search(r"BUG_LINE\s*:", response, re.IGNORECASE))
        has_fix = bool(re.search(r"FIX\s*:", response, re.IGNORECASE))
        if has_bug and has_fix:
            return 1.0
        if has_bug or has_fix:
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
        line_match = re.search(r"BUG_LINE\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        fix_match = re.search(r"FIX\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        parts = []
        if line_match:
            parts.append(f"BUG_LINE: {line_match.group(1).strip()}")
        if fix_match:
            parts.append(f"FIX: {fix_match.group(1).strip()[:100]}")
        return " | ".join(parts) if parts else response
