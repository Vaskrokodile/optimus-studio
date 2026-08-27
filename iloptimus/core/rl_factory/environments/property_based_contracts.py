"""
PropertyBasedContracts: Satisfy property-based test invariants.

Environment concept:
  The model is given a function signature and a set of properties (invariants)
  that the implementation must satisfy. It must implement the function so that
  all properties hold across multiple test inputs.

  Verification is rule-based and FAST:
    - exec() the implementation to load the function.
    - Run the function on multiple test inputs.
    - Check that each property (expressed as a Python check function) holds.

  reward = fraction of (input, property) pairs that pass.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates
# ---------------------------------------------------------------------------


def _gen_int_list(n: int, rng: random.Random) -> list[int]:
    return [rng.randint(-50, 50) for _ in range(n)]


# Each property is (name, description, check_fn).
# check_fn(func, input) -> bool
def _check_idempotent(func, inp):
    return func(func(inp)) == func(inp)


def _check_preserves_length(func, inp):
    return len(func(inp)) == len(inp)


def _check_sorted_output(func, inp):
    out = func(inp)
    return out == sorted(out)


def _check_returns_int(func, inp):
    return isinstance(func(inp), int)


def _check_non_negative(func, inp):
    return func(inp) >= 0


def _check_preserves_elements(func, inp):
    return sorted(func(inp)) == sorted(inp)


_PROPERTY_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Implement a sort function.",
        "func_name": "my_sort",
        "signature": "def my_sort(lst):",
        "properties": [
            ("idempotent", "sort(sort(x)) == sort(x)", _check_idempotent),
            ("preserves_length", "len(sort(x)) == len(x)", _check_preserves_length),
            ("sorted_output", "output is sorted", _check_sorted_output),
            ("preserves_elements", "output has same elements as input", _check_preserves_elements),
        ],
        "input_gen": _gen_int_list,
        "sizes": [0, 1, 5, 10, 20],
    },
    {
        "task": "Implement an absolute-value function.",
        "func_name": "my_abs",
        "signature": "def my_abs(n):",
        "properties": [
            ("returns_int", "output is an int", _check_returns_int),
            ("non_negative", "output >= 0", _check_non_negative),
        ],
        "input_gen": lambda n, rng: [rng.randint(-100, 100) for _ in range(n)],
        "sizes": [1, 5, 10, 20],
        "scalar": True,
    },
    {
        "task": "Implement a deduplicate function (preserving order).",
        "func_name": "dedup",
        "signature": "def dedup(lst):",
        "properties": [
            ("idempotent", "dedup(dedup(x)) == dedup(x)", _check_idempotent),
            ("preserves_elements", "output elements are subset of input", lambda f, i: all(e in i for e in f(i))),
        ],
        "input_gen": _gen_int_list,
        "sizes": [0, 1, 5, 10, 20],
    },
    {
        "task": "Implement a reverse function.",
        "func_name": "my_reverse",
        "signature": "def my_reverse(lst):",
        "properties": [
            ("idempotent_double", "reverse(reverse(x)) == x", lambda f, i: f(f(i)) == i),
            ("preserves_length", "len(reverse(x)) == len(x)", _check_preserves_length),
            ("preserves_elements", "output has same elements as input", _check_preserves_elements),
        ],
        "input_gen": _gen_int_list,
        "sizes": [0, 1, 5, 10, 20],
    },
]


def property_based_contracts_generator(seed: int) -> Problem:
    """Generate a PropertyBasedContracts problem."""
    rng = random.Random(seed)
    template = rng.choice(_PROPERTY_PROBLEMS)

    test_inputs: list[Any] = []
    for size in template["sizes"]:
        test_inputs.append(template["input_gen"](size, rng))

    # Serialize property descriptions for the prompt
    prop_lines = "\n".join(
        f"  - {name}: {desc}" for name, desc, _ in template["properties"]
    )

    prompt = (
        f"Implement a function satisfying the following properties.\n\n"
        f"Signature: {template['signature']}\n"
        f"Task: {template['task']}\n\n"
        f"Properties (must hold for all valid inputs):\n{prop_lines}\n\n"
        f"Requirements:\n"
        f"  - The function must be named '{template['func_name']}'\n"
        f"  - All properties must hold for every test input\n\n"
        f"Format your answer as:\n"
        f"```python\n<implementation>\n```"
    )

    # Serialize check functions into metadata as lambdas (kept in-memory)
    properties = [(name, desc, check) for name, desc, check in template["properties"]]

    return Problem(
        id=f"property_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "property_based_contracts",
            "task": template["task"],
            "func_name": template["func_name"],
            "signature": template["signature"],
            "properties": properties,
            "test_inputs": test_inputs,
            "scalar": template.get("scalar", False),
        },
        token_budget=1200,
        source="generated",
    )


property_based_contracts_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PropertyBasedContractsVerifier(Verifier):
    """Verify an implementation satisfies all properties on all test inputs."""

    def __init__(
        self,
        func_name: str,
        properties: list,
        test_inputs: list[Any],
        scalar: bool = False,
    ):
        super().__init__()
        self._func_name = func_name
        self._properties = properties
        self._test_inputs = test_inputs
        self._scalar = scalar

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

        total_checks = 0
        passed_checks = 0
        diag: list[str] = []

        for inp in self._test_inputs:
            # For scalar problems, test each element individually
            inputs_to_test = inp if self._scalar else [inp]
            for single in inputs_to_test:
                for name, desc, check in self._properties:
                    total_checks += 1
                    try:
                        if check(func, single):
                            passed_checks += 1
                        else:
                            diag.append(f"property '{name}' failed on {single!r}")
                    except Exception as e:
                        diag.append(f"property '{name}' raised on {single!r}: {e}")

        score = passed_checks / total_checks if total_checks > 0 else 0.0
        correct = passed_checks == total_checks and total_checks > 0

        full_diag = (
            f"Properties: {passed_checks}/{total_checks} passed "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "passed_checks": passed_checks,
                "total_checks": total_checks,
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


class PropertyBasedContractsEnv(BatchEnvBase):
    """PropertyBasedContracts: implement functions satisfying property invariants."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = property_based_contracts_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return PropertyBasedContractsVerifier(
            func_name=md["func_name"],
            properties=md["properties"],
            test_inputs=md["test_inputs"],
            scalar=md.get("scalar", False),
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
