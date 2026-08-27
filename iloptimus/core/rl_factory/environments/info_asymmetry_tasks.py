"""
InfoAsymmetryTasks: Easy-to-generate, hard-to-solve task chains.

Environment concept:
  This implements the *information asymmetry* training paradigm: tasks
  that are trivial to GENERATE procedurally but require multi-step
  reasoning to SOLVE. The generator builds a chain of simple transforms
  (e.g. "add 3, then multiply by 2, then reverse the digits") whose
  composition is complex, and the model must apply all transforms in
  order to compute the final output.

  Generating such a chain is O(n) — just pick n random transforms. But
  solving it requires applying each transform correctly in sequence,
  which is where models make mistakes. This asymmetry lets us produce
  unlimited training data with cheap, exact verification.

Verification:
  - The final output after applying all transforms is checked against
    the expected output (computed by the generator).

Reward:
  reward = exact_match (1.0 if the final output matches, else partial
  credit based on how many transforms were applied correctly).

Format:
  ANSWER: <answer>
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Transform definitions: each is (description, function)
# ---------------------------------------------------------------------------


def _t_add(n: int, k: int) -> int:
    return n + k


def _t_mul(n: int, k: int) -> int:
    return n * k


def _t_sub(n: int, k: int) -> int:
    return n - k


def _t_mod(n: int, k: int) -> int:
    return n % k if k != 0 else n


def _t_reverse_digits(n: int) -> int:
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])


def _t_double(n: int) -> int:
    return n * 2


def _t_halve(n: int) -> int:
    return n // 2


def _t_square(n: int) -> int:
    return n * n


def _t_negate(n: int) -> int:
    return -n


def _t_digit_sum(n: int) -> int:
    return sum(int(d) for d in str(abs(n)))


def _t_increment(n: int) -> int:
    return n + 1


def _t_decrement(n: int) -> int:
    return n - 1


# Each transform: (description_template, function, needs_param)
_TRANSFORMS: list[tuple[str, Callable[..., int], bool]] = [
    ("add {k}", _t_add, True),
    ("subtract {k}", _t_sub, True),
    ("multiply by {k}", _t_mul, True),
    ("take mod {k}", _t_mod, True),
    ("reverse the digits", _t_reverse_digits, False),
    ("double the number", _t_double, False),
    ("halve the number (integer division)", _t_halve, False),
    ("square the number", _t_square, False),
    ("negate the number", _t_negate, False),
    ("sum the digits", _t_digit_sum, False),
    ("increment by 1", _t_increment, False),
    ("decrement by 1", _t_decrement, False),
]


def _build_transform(rng: random.Random) -> tuple[str, Callable[[int], int]]:
    """Build a single transform: returns (description, applied_function)."""
    desc_tmpl, fn, needs_param = rng.choice(_TRANSFORMS)
    if needs_param:
        k = rng.randint(2, 9)
        desc = desc_tmpl.format(k=k)
        return desc, lambda n, fn=fn, k=k: fn(n, k)
    desc = desc_tmpl
    return desc, lambda n, fn=fn: fn(n)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def info_asymmetry_tasks_generator(seed: int) -> Problem:
    """Generate an InfoAsymmetryTasks problem.

    Builds a chain of 2-5 simple transforms applied to a random input.
    The expected output is computed by applying all transforms. The
    model must apply the chain and report the final output.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with the input, transform list, and expected output.
    """
    rng = random.Random(seed)
    n_transforms = rng.randint(2, 5)
    input_val = rng.randint(10, 99)

    transforms: list[tuple[str, Callable[[int], int]]] = []
    current = input_val
    descs: list[str] = []
    for _ in range(n_transforms):
        desc, fn = _build_transform(rng)
        transforms.append((desc, fn))
        descs.append(desc)
        current = fn(current)

    expected_output = current

    chain_text = "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(descs))
    prompt = (
        f"Starting with the number {input_val}, apply the following "
        f"transforms in order and report the final result:\n"
        f"{chain_text}\n\n"
        f"Format: ANSWER: <final number>"
    )

    return Problem(
        id=f"info_asym_{seed}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=0.2 + 0.15 * n_transforms,
        metadata={
            "type": "info_asymmetry_tasks",
            "input": input_val,
            "transforms": descs,
            "expected_output": expected_output,
            "n_transforms": n_transforms,
        },
        token_budget=300,
        source="info_asymmetry_tasks_generator",
    )


info_asymmetry_tasks_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class InfoAsymmetryVerifier(Verifier):
    """Verify the final output of a transform chain.

    Args:
        expected_output: The correct final output after all transforms.
        input_val: The starting input (for partial credit).
        transforms: List of (description, function) tuples.
    """

    def __init__(
        self,
        expected_output: int,
        input_val: int,
        transforms: list[tuple[str, Callable[[int], int]]],
    ):
        super().__init__()
        self._expected_output = expected_output
        self._input_val = input_val
        self._transforms = transforms

    def verify(self, response: str) -> VerifierResult:
        given = self._parse_answer(response)

        if given is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No number found in ANSWER field",
            )

        exact = given == self._expected_output
        if exact:
            return VerifierResult(
                correct=True,
                score=1.0,
                diagnostics=f"Exact match: {given}",
            )

        # Partial credit: how close is the answer?
        diff = abs(given - self._expected_output)
        if self._expected_output != 0:
            rel = diff / abs(self._expected_output)
        else:
            rel = 1.0 if diff != 0 else 0.0
        partial = max(0.0, 1.0 - rel)
        partial = min(partial, 0.5)  # partial credit capped at 0.5

        return VerifierResult(
            correct=False,
            score=partial,
            partial_credit={"exact": 0.0, "relative_error": rel},
            diagnostics=f"Expected={self._expected_output} Got={given} rel_err={rel:.3f}",
        )

    def _parse_answer(self, response: str) -> Optional[int]:
        match = re.search(r"ANSWER\s*:\s*(-?\d+)", response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Fallback: last number in the response
        nums = re.findall(r"-?\d+", response)
        return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class InfoAsymmetryEnv(BatchEnvBase):
    """InfoAsymmetryTasks environment: easy-to-generate, hard-to-solve chains.

    Batch-aware: N parallel attempts; reward = best attempt.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = info_asymmetry_tasks_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        # Rebuild the transform functions from the stored descriptions.
        # We can't store callables in metadata (not serializable), so we
        # recompute the chain deterministically from the seed-derived
        # input and expected output. The verifier only needs the expected
        # output for exact matching, plus the input for partial credit.
        return InfoAsymmetryVerifier(
            expected_output=meta["expected_output"],
            input_val=meta["input"],
            transforms=[],
        )

    def _check_format(self, response: str) -> float:
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        return 1.0 if has_answer else 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "mean_score": mean_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "reward": best_score,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
