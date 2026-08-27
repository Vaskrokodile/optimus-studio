"""
DiffLevelRepair: Fine-grained diff-level code repair.

Environment concept:
  The model is given buggy code plus a hint diff and must produce the fixed
  code. The verifier compares the fixed code's output to the expected (correct)
  output on a set of test inputs.

  Verification is rule-based and FAST:
    - exec() the fixed code to load the function.
    - Run it on test inputs and compare to expected outputs.

  reward = fraction of test inputs that produce the correct output.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: correct code + buggy variant + hint diff
# ---------------------------------------------------------------------------


_DIFF_REPAIR_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Fix an off-by-one error in a sum function.",
        "func_name": "sum_range",
        "correct": (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        "buggy": (
            "def sum_range(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        ),
        "hint": "-    for i in range(1, n):\n+    for i in range(1, n + 1):",
        "test_inputs": [1, 5, 10, 0],
        "expected_outputs": [1, 15, 55, 0],
    },
    {
        "task": "Fix a wrong comparison operator in a max function.",
        "func_name": "find_max",
        "correct": (
            "def find_max(lst):\n"
            "    if not lst:\n"
            "        return None\n"
            "    m = lst[0]\n"
            "    for x in lst[1:]:\n"
            "        if x > m:\n"
            "            m = x\n"
            "    return m\n"
        ),
        "buggy": (
            "def find_max(lst):\n"
            "    if not lst:\n"
            "        return None\n"
            "    m = lst[0]\n"
            "    for x in lst[1:]:\n"
            "        if x < m:\n"
            "            m = x\n"
            "    return m\n"
        ),
        "hint": "-        if x < m:\n+        if x > m:",
        "test_inputs": [[3, 1, 4, 1, 5], [10], [-1, -5, -2], []],
        "expected_outputs": [5, 10, -1, None],
    },
    {
        "task": "Fix a wrong base case in factorial.",
        "func_name": "factorial",
        "correct": (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        "buggy": (
            "def factorial(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    return n * factorial(n - 1)\n"
        ),
        "hint": "-    if n <= 0:\n-        return 0\n+    if n <= 1:\n+        return 1",
        "test_inputs": [0, 1, 5, 3],
        "expected_outputs": [1, 1, 120, 6],
    },
    {
        "task": "Fix a missing return in a string duplicate function.",
        "func_name": "duplicate",
        "correct": (
            "def duplicate(s):\n"
            "    result = s + s\n"
            "    return result\n"
        ),
        "buggy": (
            "def duplicate(s):\n"
            "    result = s + s\n"
            "    return s\n"
        ),
        "hint": "-    return s\n+    return result",
        "test_inputs": ["hello", "", "a", "ab"],
        "expected_outputs": ["hellohello", "", "aa", "abab"],
    },
    {
        "task": "Fix a wrong list index in an average function.",
        "func_name": "average",
        "correct": (
            "def average(lst):\n"
            "    if not lst:\n"
            "        return 0\n"
            "    return sum(lst) / len(lst)\n"
        ),
        "buggy": (
            "def average(lst):\n"
            "    if not lst:\n"
            "        return 0\n"
            "    return sum(lst) / (len(lst) - 1)\n"
        ),
        "hint": "-    return sum(lst) / (len(lst) - 1)\n+    return sum(lst) / len(lst)",
        "test_inputs": [[1, 2, 3], [10, 20], [5], []],
        "expected_outputs": [2.0, 15.0, 5.0, 0],
    },
]


def diff_level_repair_generator(seed: int) -> Problem:
    """Generate a DiffLevelRepair problem."""
    rng = random.Random(seed)
    template = rng.choice(_DIFF_REPAIR_PROBLEMS)

    prompt = (
        f"Fix the buggy code using the provided hint diff.\n\n"
        f"Buggy code:\n"
        f"```python\n{template['buggy']}```\n\n"
        f"Hint diff:\n```\n{template['hint']}\n```\n\n"
        f"Task: {template['task']}\n"
        f"Requirements:\n"
        f"  - The fixed code must produce correct output on all test inputs\n"
        f"  - Keep the same function name: {template['func_name']}\n\n"
        f"Format your answer as:\n"
        f"```python\n<fixed code>\n```"
    )

    return Problem(
        id=f"diff_repair_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.3 + 0.15 * rng.random(),
        metadata={
            "type": "diff_level_repair",
            "task": template["task"],
            "func_name": template["func_name"],
            "buggy_code": template["buggy"],
            "correct_code": template["correct"],
            "hint": template["hint"],
            "test_inputs": template["test_inputs"],
            "expected_outputs": template["expected_outputs"],
        },
        token_budget=1000,
        source="generated",
    )


diff_level_repair_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class DiffLevelRepairVerifier(Verifier):
    """Verify fixed code by comparing output to expected on test inputs."""

    def __init__(
        self,
        correct_code: str,
        func_name: str,
        test_inputs: list[Any],
        expected_outputs: list[Any],
    ):
        super().__init__()
        self._correct_code = correct_code
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._expected_outputs = expected_outputs

    def verify(self, response: str) -> VerifierResult:
        code = self._extract_code(response)
        if not code:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No python code block found in response",
            )

        ns: dict[str, Any] = {}
        try:
            exec(code, ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Fixed code failed to execute: {type(e).__name__}: {e}",
            )

        func = ns.get(self._func_name)
        if not callable(func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in fixed code",
            )

        correct_count = 0
        total = len(self._test_inputs)
        diag: list[str] = []

        for inp, expected in zip(self._test_inputs, self._expected_outputs):
            try:
                result = func(inp)
            except Exception as e:
                diag.append(f"error on {inp!r}: {e}")
                continue
            if isinstance(expected, float):
                ok = isinstance(result, (int, float)) and abs(result - expected) < 1e-6
            else:
                ok = result == expected
            if ok:
                correct_count += 1
            else:
                diag.append(f"mismatch on {inp!r}: {result!r} != {expected!r}")

        score = correct_count / total if total > 0 else 0.0
        correct = correct_count == total and total > 0

        full_diag = (
            f"Correctness: {correct_count}/{total} "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "correct_inputs": correct_count,
                "total_inputs": total,
                "pass_rate": score,
            },
            diagnostics=full_diag,
        )

    def _extract_code(self, response: str) -> str:
        matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        if "def " in response and "```" not in response:
            return response.strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DiffLevelRepairEnv(BatchEnvBase):
    """DiffLevelRepair: fine-grained diff-level code repair."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = diff_level_repair_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return DiffLevelRepairVerifier(
            correct_code=md["correct_code"],
            func_name=md["func_name"],
            test_inputs=md["test_inputs"],
            expected_outputs=md["expected_outputs"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if "def " in response:
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
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Mean={sum(scores)/len(scores) if scores else 0:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
