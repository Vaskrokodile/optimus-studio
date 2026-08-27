"""
KernelBuildCheckpoints: Build a simple program through verifiable checkpoints.

Environment concept:
  The model must implement a sequence of checkpoint functions in order, each
  building on the previous. For example: "implement add", "implement multiply
  using add", "implement power using multiply".

  Verification is rule-based and FAST:
    - Parse the response for ``CHECKPOINT <n>:`` blocks.
    - exec() each checkpoint function and test it against expected outputs.

  reward = fraction of checkpoints that pass all their tests.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: sequences of checkpoint functions
# ---------------------------------------------------------------------------


_CHECKPOINT_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Build arithmetic from scratch: add -> multiply -> power.",
        "checkpoints": [
            {
                "name": "my_add",
                "signature": "def my_add(a, b):",
                "description": "Return a + b.",
                "test_inputs": [(2, 3), (0, 0), (-1, 5), (10, -4)],
                "expected_outputs": [5, 0, 4, 6],
            },
            {
                "name": "my_multiply",
                "signature": "def my_multiply(a, b):",
                "description": "Return a * b using only my_add (repeated addition). b >= 0.",
                "test_inputs": [(2, 3), (0, 5), (4, 0), (-2, 3)],
                "expected_outputs": [6, 0, 0, -6],
            },
            {
                "name": "my_power",
                "signature": "def my_power(base, exp):",
                "description": "Return base**exp using only my_multiply (repeated multiplication). exp >= 0.",
                "test_inputs": [(2, 3), (5, 0), (3, 2), (1, 10)],
                "expected_outputs": [8, 1, 9, 1],
            },
        ],
    },
    {
        "task": "Build list operations: sum -> mean -> variance.",
        "checkpoints": [
            {
                "name": "my_sum",
                "signature": "def my_sum(lst):",
                "description": "Return the sum of a list of numbers.",
                "test_inputs": [([1, 2, 3],), ([0],), ([],), ([-1, -2, -3],)],
                "expected_outputs": [6, 0, 0, -6],
            },
            {
                "name": "my_mean",
                "signature": "def my_mean(lst):",
                "description": "Return the mean of a list using my_sum. Return 0 for empty list.",
                "test_inputs": [([1, 2, 3],), ([5],), ([2, 4, 6],)],
                "expected_outputs": [2.0, 5.0, 4.0],
            },
            {
                "name": "my_variance",
                "signature": "def my_variance(lst):",
                "description": "Return the population variance using my_mean. Return 0 for empty list.",
                "test_inputs": [([1, 2, 3],), ([5, 5, 5],), ([0, 10],)],
                "expected_outputs": [2.0 / 3.0, 0.0, 25.0],
            },
        ],
    },
    {
        "task": "Build string operations: reverse -> is_palindrome.",
        "checkpoints": [
            {
                "name": "my_reverse",
                "signature": "def my_reverse(s):",
                "description": "Return the reversed string.",
                "test_inputs": [("hello",), ("",), ("a",), ("ab",)],
                "expected_outputs": ["olleh", "", "a", "ba"],
            },
            {
                "name": "is_palindrome",
                "signature": "def is_palindrome(s):",
                "description": "Return True if s is a palindrome, using my_reverse.",
                "test_inputs": [("racecar",), ("hello",), ("",), ("a",), ("abba",)],
                "expected_outputs": [True, False, True, True, True],
            },
        ],
    },
]


def kernel_build_checkpoints_generator(seed: int) -> Problem:
    """Generate a KernelBuildCheckpoints problem."""
    rng = random.Random(seed)
    template = rng.choice(_CHECKPOINT_PROBLEMS)
    checkpoints = template["checkpoints"]
    n = len(checkpoints)

    lines = [f"Build a program through {n} verifiable checkpoints.\n"]
    lines.append(f"Task: {template['task']}\n")
    for i, cp in enumerate(checkpoints):
        lines.append(f"CHECKPOINT {i + 1}: {cp['name']}")
        lines.append(f"  Signature: {cp['signature']}")
        lines.append(f"  Description: {cp['description']}")
        lines.append("")
    lines.append("Requirements:")
    lines.append("  - Implement each checkpoint function in order.")
    lines.append("  - Each function must pass all its tests.")
    lines.append("  - Later checkpoints may use earlier ones.\n")
    lines.append("Format your answer as:")
    for i in range(n):
        lines.append(f"CHECKPOINT {i + 1}:")
        lines.append("```python")
        lines.append(f"<{checkpoints[i]['name']} implementation>")
        lines.append("```")

    prompt = "\n".join(lines)

    return Problem(
        id=f"kernel_build_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "kernel_build_checkpoints",
            "task": template["task"],
            "checkpoints": checkpoints,
            "n_checkpoints": n,
        },
        token_budget=2000,
        source="generated",
    )


kernel_build_checkpoints_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class KernelBuildCheckpointsVerifier(Verifier):
    """Verify checkpoint implementations by exec()ing and testing each."""

    def __init__(self, checkpoints: list, n_checkpoints: int):
        super().__init__()
        self._checkpoints = checkpoints
        self._n = n_checkpoints

    def verify(self, response: str) -> VerifierResult:
        # Parse CHECKPOINT <n>: ```python ... ``` blocks
        # Use a regex that captures the checkpoint number and code block.
        pattern = re.compile(
            r"CHECKPOINT\s+(\d+)\s*:\s*```(?:python)?\n(.*?)```",
            re.DOTALL,
        )
        matches = pattern.findall(response)

        if not matches:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No CHECKPOINT blocks found in response",
            )

        # Build a map: checkpoint_number -> code
        code_by_cp: dict[int, str] = {}
        for num_str, code in matches:
            code_by_cp[int(num_str)] = code.strip()

        # Shared namespace so later checkpoints can use earlier ones
        ns: dict[str, Any] = {}

        checkpoints_passed = 0
        total_tests = 0
        tests_passed = 0
        diag: list[str] = []

        for i, cp in enumerate(self._checkpoints):
            cp_num = i + 1
            code = code_by_cp.get(cp_num)
            if code is None:
                diag.append(f"checkpoint {cp_num} ({cp['name']}) missing")
                total_tests += len(cp["test_inputs"])
                continue

            try:
                exec(code, ns)
            except Exception as e:
                diag.append(f"checkpoint {cp_num} exec error: {type(e).__name__}: {e}")
                total_tests += len(cp["test_inputs"])
                continue

            func = ns.get(cp["name"])
            if not callable(func):
                diag.append(f"checkpoint {cp_num}: function '{cp['name']}' not found")
                total_tests += len(cp["test_inputs"])
                continue

            cp_all_pass = True
            for inp, expected in zip(cp["test_inputs"], cp["expected_outputs"]):
                total_tests += 1
                try:
                    result = func(*inp)
                except Exception as e:
                    diag.append(f"cp{cp_num} {cp['name']}({inp!r}) raised: {e}")
                    cp_all_pass = False
                    continue
                # Use approximate comparison for floats
                if isinstance(expected, float):
                    ok = abs(result - expected) < 1e-6
                else:
                    ok = result == expected
                if ok:
                    tests_passed += 1
                else:
                    cp_all_pass = False
                    diag.append(f"cp{cp_num} {cp['name']}({inp!r}) = {result!r}, expected {expected!r}")

            if cp_all_pass:
                checkpoints_passed += 1

        cp_score = checkpoints_passed / self._n if self._n > 0 else 0.0
        test_score = tests_passed / total_tests if total_tests > 0 else 0.0
        # Weight checkpoint completion more heavily
        score = cp_score * 0.6 + test_score * 0.4
        correct = checkpoints_passed == self._n

        full_diag = (
            f"Checkpoints: {checkpoints_passed}/{self._n} "
            f"Tests: {tests_passed}/{total_tests} "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "checkpoints_passed": checkpoints_passed,
                "n_checkpoints": self._n,
                "tests_passed": tests_passed,
                "total_tests": total_tests,
                "checkpoint_score": cp_score,
                "test_score": test_score,
            },
            diagnostics=full_diag,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class KernelBuildCheckpointsEnv(BatchEnvBase):
    """KernelBuildCheckpoints: build a program through verifiable checkpoints."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = kernel_build_checkpoints_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return KernelBuildCheckpointsVerifier(
            checkpoints=md["checkpoints"],
            n_checkpoints=md["n_checkpoints"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"CHECKPOINT\s+\d+\s*:\s*```", response, re.DOTALL):
            return 1.0
        if "CHECKPOINT" in response:
            return 0.5
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 0.3
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
