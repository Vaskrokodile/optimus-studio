"""
SpecToVerifiedCode: Generate code from formal specifications.

Environment concept:
  The model is given a formal specification (input/output constraints) and
  must implement a function that satisfies the spec. The verifier runs the
  code on test inputs and checks that outputs satisfy the spec constraints.

  Verification is rule-based and FAST:
    - exec() the implementation to load the function.
    - Run it on test inputs and compare to expected outputs.
    - Additionally check that constraint predicates hold.

  reward = fraction of test inputs that produce correct + constraint-satisfying output.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: spec + constraints + test inputs/outputs
# ---------------------------------------------------------------------------


def _constraint_returns_int(result, inp):
    return isinstance(result, int)


def _constraint_returns_list(result, inp):
    return isinstance(result, list)


def _constraint_non_empty_if_input_nonempty(result, inp):
    if len(inp) > 0:
        return len(result) > 0
    return True


def _constraint_sorted(result, inp):
    return result == sorted(result)


def _constraint_same_length(result, inp):
    return len(result) == len(inp)


_SPEC_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Find the maximum element in a list of ints.",
        "func_name": "find_max",
        "signature": "def find_max(lst):",
        "spec": "Takes a non-empty list of ints, returns the maximum element.",
        "constraints": [
            ("returns_int", "output is an int", _constraint_returns_int),
        ],
        "test_inputs": [[3, 1, 4, 1, 5, 9], [10], [-5, -1, -10], [0, 0, 0]],
        "expected_outputs": [9, 10, -1, 0],
    },
    {
        "task": "Sort a list of ints in ascending order.",
        "func_name": "spec_sort",
        "signature": "def spec_sort(lst):",
        "spec": "Takes a list of ints, returns a new list sorted ascending.",
        "constraints": [
            ("returns_list", "output is a list", _constraint_returns_list),
            ("sorted", "output is sorted ascending", _constraint_sorted),
            ("same_length", "output length == input length", _constraint_same_length),
        ],
        "test_inputs": [[3, 1, 2], [1], [], [5, 4, 3, 2, 1]],
        "expected_outputs": [[1, 2, 3], [1], [], [1, 2, 3, 4, 5]],
    },
    {
        "task": "Count the number of even integers in a list.",
        "func_name": "count_evens",
        "signature": "def count_evens(lst):",
        "spec": "Takes a list of ints, returns the count of even numbers.",
        "constraints": [
            ("returns_int", "output is an int", _constraint_returns_int),
            ("non_negative", "output >= 0", lambda r, i: r >= 0),
        ],
        "test_inputs": [[1, 2, 3, 4], [1, 3, 5], [2, 4, 6], []],
        "expected_outputs": [2, 0, 3, 0],
    },
    {
        "task": "Return the first non-negative element in a list.",
        "func_name": "first_nonneg",
        "signature": "def first_nonneg(lst):",
        "spec": "Takes a list of ints, returns the first element >= 0, or None if none.",
        "constraints": [
            ("nonneg_or_none", "output >= 0 or None", lambda r, i: r is None or r >= 0),
        ],
        "test_inputs": [[-1, -2, 3, 4], [-5, -1], [0, 1, 2], []],
        "expected_outputs": [3, None, 0, None],
    },
]


def spec_to_verified_code_generator(seed: int) -> Problem:
    """Generate a SpecToVerifiedCode problem."""
    rng = random.Random(seed)
    template = rng.choice(_SPEC_PROBLEMS)

    constraint_lines = "\n".join(
        f"  - {name}: {desc}" for name, desc, _ in template["constraints"]
    )

    prompt = (
        f"Implement a function from the following formal specification.\n\n"
        f"Signature: {template['signature']}\n"
        f"Specification: {template['spec']}\n"
        f"Task: {template['task']}\n\n"
        f"Constraints (must hold for all valid inputs):\n{constraint_lines}\n\n"
        f"Requirements:\n"
        f"  - The function must be named '{template['func_name']}'\n"
        f"  - It must satisfy the spec and all constraints\n\n"
        f"Format your answer as:\n"
        f"```python\n<implementation>\n```"
    )

    constraints = [(name, desc, check) for name, desc, check in template["constraints"]]

    return Problem(
        id=f"spec_code_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "spec_to_verified_code",
            "task": template["task"],
            "func_name": template["func_name"],
            "signature": template["signature"],
            "spec": template["spec"],
            "constraints": constraints,
            "test_inputs": template["test_inputs"],
            "expected_outputs": template["expected_outputs"],
        },
        token_budget=1200,
        source="generated",
    )


spec_to_verified_code_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SpecToVerifiedCodeVerifier(Verifier):
    """Verify an implementation satisfies the spec on all test inputs."""

    def __init__(
        self,
        func_name: str,
        constraints: list,
        test_inputs: list[Any],
        expected_outputs: list[Any],
    ):
        super().__init__()
        self._func_name = func_name
        self._constraints = constraints
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
                diagnostics=f"Code failed to execute: {type(e).__name__}: {e}",
            )

        func = ns.get(self._func_name)
        if not callable(func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in code",
            )

        total = len(self._test_inputs)
        correct_count = 0
        constraint_ok_count = 0
        diag: list[str] = []

        for inp, expected in zip(self._test_inputs, self._expected_outputs):
            try:
                result = func(inp)
            except Exception as e:
                diag.append(f"error on {inp!r}: {e}")
                continue

            # Check output correctness
            output_ok = result == expected
            if output_ok:
                correct_count += 1

            # Check constraints
            all_constraints_ok = True
            for name, desc, check in self._constraints:
                try:
                    if not check(result, inp):
                        all_constraints_ok = False
                        diag.append(f"constraint '{name}' failed on {inp!r}")
                except Exception as e:
                    all_constraints_ok = False
                    diag.append(f"constraint '{name}' raised on {inp!r}: {e}")

            if all_constraints_ok:
                constraint_ok_count += 1

            if not output_ok:
                diag.append(f"mismatch on {inp!r}: {result!r} != {expected!r}")

        output_score = correct_count / total if total > 0 else 0.0
        constraint_score = constraint_ok_count / total if total > 0 else 0.0
        score = output_score * 0.6 + constraint_score * 0.4
        correct = correct_count == total and constraint_ok_count == total and total > 0

        full_diag = (
            f"Output: {correct_count}/{total} "
            f"Constraints: {constraint_ok_count}/{total} "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "output_correct": correct_count,
                "constraint_ok": constraint_ok_count,
                "total_inputs": total,
                "output_score": output_score,
                "constraint_score": constraint_score,
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


class SpecToVerifiedCodeEnv(BatchEnvBase):
    """SpecToVerifiedCode: generate code from formal specifications."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = spec_to_verified_code_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return SpecToVerifiedCodeVerifier(
            func_name=md["func_name"],
            constraints=md["constraints"],
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
