"""
ReverseCurriculum: Backward chaining — reveal the solution end, complete the start.

Environment concept:
  This implements the *reverse curriculum* training paradigm. The model
  is given the FINAL step of a multi-step solution and must produce the
  PRECEDING steps that lead to it. This is backward chaining: instead of
  solving forward (problem -> solution), the model solves backward
  (solution end -> solution start).

  Why: backward chaining teaches the model to reason about *what must
  have been true before* a given outcome. It complements forward
  training and exposes gaps in causal understanding.

  The generator creates a full multi-step solution, reveals only the
  last step, and the model must produce the missing earlier steps.

Verification:
  - The produced steps must be valid (each step's arithmetic/logic
    must be correct).
  - The produced steps must lead to the revealed final step (the last
    produced step's result must match the revealed step's input).

Reward:
  reward = steps_valid * 0.5 + leads_to_final * 0.3 + count_match * 0.2
  correct = leads_to_final AND steps_valid >= 0.5

Format:
  STEP 1: <step>
  STEP 2: <step>
  ...
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Solution templates: each has a problem, a full list of steps, and the
# final result.
# ---------------------------------------------------------------------------


_RC_SOLUTIONS: list[dict[str, Any]] = [
    {
        "problem": "Compute 12 * 15 step by step.",
        "all_steps": [
            "12 * 15 = 12 * (10 + 5)",
            "12 * 10 = 120",
            "12 * 5 = 60",
            "120 + 60 = 180",
        ],
        "final_result": "180",
    },
    {
        "problem": "Compute 7^3 step by step.",
        "all_steps": [
            "7^3 = 7 * 7 * 7",
            "7 * 7 = 49",
            "49 * 7 = 343",
        ],
        "final_result": "343",
    },
    {
        "problem": "Compute (3 + 4) * 5 step by step.",
        "all_steps": [
            "3 + 4 = 7",
            "7 * 5 = 35",
        ],
        "final_result": "35",
    },
    {
        "problem": "Compute 144 / 6 + 8 step by step.",
        "all_steps": [
            "144 / 6 = 24",
            "24 + 8 = 32",
        ],
        "final_result": "32",
    },
    {
        "problem": "Compute 2^5 - 10 step by step.",
        "all_steps": [
            "2^5 = 32",
            "32 - 10 = 22",
        ],
        "final_result": "22",
    },
    {
        "problem": "Compute 9 * 13 step by step.",
        "all_steps": [
            "9 * 13 = 9 * (10 + 3)",
            "9 * 10 = 90",
            "9 * 3 = 27",
            "90 + 27 = 117",
        ],
        "final_result": "117",
    },
    {
        "problem": "Compute 100 - 7 * 8 step by step.",
        "all_steps": [
            "7 * 8 = 56",
            "100 - 56 = 44",
        ],
        "final_result": "44",
    },
    {
        "problem": "Compute sqrt(144) + sqrt(25) step by step.",
        "all_steps": [
            "sqrt(144) = 12",
            "sqrt(25) = 5",
            "12 + 5 = 17",
        ],
        "final_result": "17",
    },
    {
        "problem": "Compute 25% of 480 step by step.",
        "all_steps": [
            "25% = 0.25",
            "0.25 * 480 = 120",
        ],
        "final_result": "120",
    },
    {
        "problem": "Compute 11 * 11 + 11 step by step.",
        "all_steps": [
            "11 * 11 = 121",
            "121 + 11 = 132",
        ],
        "final_result": "132",
    },
]


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def _step_is_valid_arithmetic(step: str) -> bool:
    """Check if a step contains a valid arithmetic statement."""
    step = step.strip()
    if not step:
        return False
    expr_match = re.search(
        r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
        step,
    )
    if expr_match:
        a, op, b, c = (
            float(expr_match.group(1)),
            expr_match.group(2),
            float(expr_match.group(3)),
            float(expr_match.group(4)),
        )
        if op == "+" and a + b == c:
            return True
        if op == "-" and a - b == c:
            return True
        if op == "*" and a * b == c:
            return True
        if op == "/" and b != 0 and a / b == c:
            return True
        return False
    # Non-arithmetic steps (e.g. "7^3 = 7 * 7 * 7") are valid if substantive
    if re.search(r"=", step):
        return True
    if re.search(r"(sqrt|power|percent|factorial)", step, re.IGNORECASE):
        return True
    return len(step) > 8


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def reverse_curriculum_generator(seed: int) -> Problem:
    """Generate a ReverseCurriculum problem.

    Selects a solution template, reveals only the final step, and asks
    the model to produce the preceding steps.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with all_steps, revealed step, and missing count.
    """
    rng = random.Random(seed)
    template = rng.choice(_RC_SOLUTIONS)

    all_steps = template["all_steps"]
    n = len(all_steps)
    revealed_idx = n - 1  # reveal the last step
    revealed_step = all_steps[revealed_idx]
    n_missing = revealed_idx  # number of preceding steps to produce

    prompt = (
        f"Problem: {template['problem']}\n\n"
        f"The FINAL step of the solution is revealed:\n"
        f"  FINAL STEP: {revealed_step}\n\n"
        f"Produce the {n_missing} preceding step(s) that lead to this "
        f"final step.\n\n"
        f"Format (one per line):\n"
        + "".join(f"STEP {i + 1}: <step>\n" for i in range(n_missing))
    )

    return Problem(
        id=f"reverse_curr_{seed}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=0.3 + 0.1 * n_missing,
        metadata={
            "type": "reverse_curriculum",
            "problem": template["problem"],
            "all_steps": all_steps,
            "revealed_step_idx": revealed_idx,
            "revealed_step": revealed_step,
            "n_missing_steps": n_missing,
            "final_result": template["final_result"],
        },
        token_budget=400,
        source="reverse_curriculum_generator",
    )


reverse_curriculum_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ReverseCurriculumVerifier(Verifier):
    """Verify backward-chained steps lead to the revealed final step.

    Args:
        all_steps: The full ground-truth step list.
        revealed_step: The revealed final step.
        n_missing: Number of missing (preceding) steps.
    """

    def __init__(
        self,
        all_steps: list[str],
        revealed_step: str,
        n_missing: int,
    ):
        super().__init__()
        self._all_steps = all_steps
        self._revealed_step = revealed_step
        self._n_missing = n_missing
        self._expected_missing = all_steps[:n_missing]

    def verify(self, response: str) -> VerifierResult:
        # Parse produced steps
        step_lines = re.findall(
            r"STEP\s+\d+\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        if not step_lines:
            # Fallback: numbered lines
            step_lines = re.findall(
                r"(?:^\d+\.?\s*)(.+?)(?:\n|$)", response, re.IGNORECASE | re.MULTILINE
            )

        # 1. Steps valid: each step has valid arithmetic/logic
        if step_lines:
            valid_count = sum(
                1 for sl in step_lines if _step_is_valid_arithmetic(sl)
            )
            steps_valid = valid_count / len(step_lines)
        else:
            steps_valid = 0.0

        # 2. Leads to final: the last produced step's numbers should
        #    match the numbers in the revealed step (or the expected
        #    missing steps' content).
        leads_to_final = self._leads_to_final(step_lines)

        # 3. Count match: did the model produce the right number of steps?
        count_match = 1.0 if len(step_lines) == self._n_missing else (
            0.5 if len(step_lines) > 0 else 0.0
        )

        score = steps_valid * 0.5 + leads_to_final * 0.3 + count_match * 0.2
        correct = leads_to_final >= 0.5 and steps_valid >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "steps_valid": steps_valid,
                "leads_to_final": leads_to_final,
                "count_match": count_match,
            },
            diagnostics=(
                f"Valid={steps_valid:.2f} ({len(step_lines)} steps) "
                f"Leads={leads_to_final:.2f} Count={count_match:.2f}"
            ),
        )

    def _leads_to_final(self, step_lines: list[str]) -> float:
        """Check if produced steps lead to the revealed final step."""
        if not step_lines:
            return 0.0
        # Compare produced steps to expected missing steps
        matches = 0
        for produced in step_lines:
            p_nums = tuple(sorted(_extract_numbers(produced)))
            for expected in self._expected_missing:
                e_nums = tuple(sorted(_extract_numbers(expected)))
                if p_nums and e_nums and p_nums == e_nums:
                    matches += 1
                    break
        return matches / max(len(self._expected_missing), 1)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ReverseCurriculumEnv(BatchEnvBase):
    """ReverseCurriculum environment: backward chaining.

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
            problem_generator = reverse_curriculum_generator
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
        return ReverseCurriculumVerifier(
            all_steps=meta["all_steps"],
            revealed_step=meta["revealed_step"],
            n_missing=meta["n_missing_steps"],
        )

    def _check_format(self, response: str) -> float:
        has_step = bool(re.search(r"STEP\s+\d+\s*:", response, re.IGNORECASE))
        return 1.0 if has_step else 0.0

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
            "correct": any_correct and best_score >= 0.5,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        steps = re.findall(
            r"STEP\s+\d+\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        return "\n".join(steps) if steps else response
