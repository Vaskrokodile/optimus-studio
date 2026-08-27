"""
AlgorithmInvention: Invent a better algorithm than an O(n^2) baseline.

Environment concept:
  The model is given a slow O(n^2) algorithm and must write a faster one
  that produces the same output. This trains algorithmic creativity —
  finding asymptotic improvements rather than incremental tweaks.

  Why: most coding tasks ask "implement X." This one asks "make X faster,"
  which requires understanding the algorithm's bottleneck and inventing
  a fundamentally different approach (e.g. hash set instead of nested loops).

Verification:
  - The submitted code compiles (exec without error).
  - The submitted code produces correct output on test inputs.
  - The submitted code is different from the baseline (not a copy).

Reward design:
  compiles * 0.2 + correctness * 0.6 + is_different * 0.2
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — each: baseline O(n^2) code, func name, test inputs, expected outputs
# ---------------------------------------------------------------------------

_ALGO_PROBLEMS = [
    {
        "baseline_code": (
            "def find_duplicates(arr):\n"
            "    result = []\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i + 1, len(arr)):\n"
            "            if arr[i] == arr[j] and arr[i] not in result:\n"
            "                result.append(arr[i])\n"
            "    return result\n"
        ),
        "func_name": "find_duplicates",
        "test_inputs": [[1, 2, 3, 2, 4, 1], [5, 5, 5, 5], [1, 2, 3, 4]],
        "expected_outputs": [[1, 2], [5], []],
        "description": "find duplicate elements in a list",
    },
    {
        "baseline_code": (
            "def two_sum(arr, target):\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i + 1, len(arr)):\n"
            "            if arr[i] + arr[j] == target:\n"
            "                return [i, j]\n"
            "    return []\n"
        ),
        "func_name": "two_sum",
        "test_inputs": [[2, 7, 11, 15], 9],
        "expected_outputs": [[0, 1]],
        "description": "find indices of two numbers that sum to target",
    },
    {
        "baseline_code": (
            "def count_pairs(arr, target):\n"
            "    count = 0\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i + 1, len(arr)):\n"
            "            if arr[i] + arr[j] == target:\n"
            "                count += 1\n"
            "    return count\n"
        ),
        "func_name": "count_pairs",
        "test_inputs": [[1, 2, 3, 4, 5], 6],
        "expected_outputs": [2],
        "description": "count pairs that sum to target",
    },
    {
        "baseline_code": (
            "def has_common(a, b):\n"
            "    for x in a:\n"
            "        for y in b:\n"
            "            if x == y:\n"
            "                return True\n"
            "    return False\n"
        ),
        "func_name": "has_common",
        "test_inputs": [[1, 2, 3], [3, 4, 5]],
        "expected_outputs": [True],
        "description": "check if two lists have a common element",
    },
    {
        "baseline_code": (
            "def intersection(a, b):\n"
            "    result = []\n"
            "    for x in a:\n"
            "        for y in b:\n"
            "            if x == y and x not in result:\n"
            "                result.append(x)\n"
            "    return result\n"
        ),
        "func_name": "intersection",
        "test_inputs": [[1, 2, 3, 4], [3, 4, 5, 6]],
        "expected_outputs": [[3, 4]],
        "description": "find intersection of two lists",
    },
    {
        "baseline_code": (
            "def max_difference(arr):\n"
            "    best = 0\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i + 1, len(arr)):\n"
            "            diff = arr[j] - arr[i]\n"
            "            if diff > best:\n"
            "                best = diff\n"
            "    return best\n"
        ),
        "func_name": "max_difference",
        "test_inputs": [[7, 1, 5, 3, 6, 4]],
        "expected_outputs": [5],
        "description": "find max difference (buy low sell high)",
    },
]


def _extract_code_block(response: str) -> str:
    """Extract Python code from a markdown code block or the response."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def algorithm_invention_generator(seed: int) -> Problem:
    """Generate an AlgorithmInvention problem: write a faster algorithm."""
    rng = random.Random(seed)
    problem = rng.choice(_ALGO_PROBLEMS)
    baseline = problem["baseline_code"]
    func_name = problem["func_name"]
    test_inputs = problem["test_inputs"]
    expected_outputs = problem["expected_outputs"]

    prompt = (
        f"The following algorithm is correct but slow (O(n^2)).\n\n"
        f"```python\n{baseline}\n```\n\n"
        f"Write a FASTER algorithm that produces the same output for the "
        f"function `{func_name}`. Your solution must be more efficient than "
        f"the baseline (avoid nested loops).\n\n"
        f"Format your answer as a Python code block:\n"
        f"```python\n<your code>\n```"
    )

    difficulty = 0.6

    return Problem(
        id=f"algorithm_invention_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "baseline_code": baseline,
            "func_name": func_name,
            "test_inputs": test_inputs,
            "expected_outputs": expected_outputs,
            "description": problem["description"],
        },
        token_budget=1024,
        source="algorithm_invention_generator",
    )


algorithm_invention_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class AlgorithmInventionVerifier(Verifier):
    """Verify an algorithm invention: compiles + correct + different from baseline.

    reward = compiles * 0.2 + correctness * 0.6 + is_different * 0.2
    """

    def __init__(
        self,
        baseline_code: str,
        func_name: str,
        test_inputs: list,
        expected_outputs: list,
    ):
        super().__init__()
        self._baseline_code = baseline_code
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._expected_outputs = expected_outputs

    def verify(self, response: str) -> VerifierResult:
        code = _extract_code_block(response)
        if not code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No code block found",
            )

        # Check it's different from baseline
        is_different = code.strip() != self._baseline_code.strip()
        # Normalize for comparison: remove whitespace
        norm_code = re.sub(r"\s+", "", code)
        norm_baseline = re.sub(r"\s+", "", self._baseline_code)
        if norm_code == norm_baseline:
            is_different = False

        # Try to compile and run
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                partial_credit={
                    "compiles": 0.0,
                    "correctness": 0.0,
                    "is_different": 1.0 if is_different else 0.0,
                },
                diagnostics=f"Execution error: {type(e).__name__}: {e}",
            )

        compiles = 1.0

        # Check the function exists
        if self._func_name not in namespace or not callable(namespace[self._func_name]):
            return VerifierResult(
                correct=False,
                score=0.2,
                partial_credit={
                    "compiles": compiles,
                    "correctness": 0.0,
                    "is_different": 1.0 if is_different else 0.0,
                },
                diagnostics=f"Function '{self._func_name}' not found",
            )

        func = namespace[self._func_name]
        correct_count = 0
        total = len(self._test_inputs)
        error_msg = ""

        for i, (test_input, expected) in enumerate(zip(self._test_inputs, self._expected_outputs)):
            try:
                if isinstance(test_input, list) and len(test_input) == 2 and isinstance(test_input[1], (int, float)) and not isinstance(test_input[0], (int, float)):
                    # two_sum style: (arr, target)
                    result = func(test_input[0], test_input[1])
                else:
                    result = func(test_input)
                if result == expected:
                    correct_count += 1
                else:
                    error_msg = f"Test {i}: got {result!r}, expected {expected!r}"
            except Exception as e:
                error_msg = f"Test {i}: {type(e).__name__}: {e}"

        correctness = correct_count / total if total > 0 else 0.0
        is_correct = correctness == 1.0 and is_different

        score = compiles * 0.2 + correctness * 0.6 + (1.0 if is_different else 0.0) * 0.2

        return VerifierResult(
            correct=is_correct,
            score=score,
            partial_credit={
                "compiles": compiles,
                "correctness": correctness,
                "is_different": 1.0 if is_different else 0.0,
            },
            diagnostics=(
                f"compiles={compiles:.1f} correctness={correctness:.2f} "
                f"different={is_different} {error_msg}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AlgorithmInventionEnv(BatchEnvBase):
    """AlgorithmInvention: invent a faster algorithm.

    Batch-aware: N parallel attempts; reward = best correct algorithm.
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
            problem_generator = algorithm_invention_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return AlgorithmInventionVerifier(
            baseline_code=problem.metadata["baseline_code"],
            func_name=problem.metadata["func_name"],
            test_inputs=problem.metadata["test_inputs"],
            expected_outputs=problem.metadata["expected_outputs"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python", response, re.IGNORECASE):
            return 1.0
        if re.search(r"```\w*", response):
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
        return _extract_code_block(response)
