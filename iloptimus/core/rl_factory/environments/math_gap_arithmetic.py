"""
MathGapArithmetic: Procedural arithmetic with controllable proof tree complexity.

Environment concept:
  The model is given a multi-step arithmetic expression and must produce a
  step-by-step solution. Each step applies one arithmetic operation, and the
  verifier checks that every step's arithmetic is valid and that the final
  answer is correct.

  This trains the model to:
    1. Decompose a compound expression into valid sub-computations
    2. Produce sound intermediate results (no arithmetic slips)
    3. Reach the correct final answer
    4. Structure the work as discrete, checkable steps

Verification:
  - Each STEP line is parsed as "lhs op rhs = result" and checked numerically.
  - The ANSWER line is compared to the ground-truth value.
  - Partial credit is awarded for valid steps even if the final answer is wrong.

Reward design:
  answer_correct * 0.6 + steps_valid * 0.4
  where steps_valid is the fraction of parsed steps with correct arithmetic.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def _eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression to a float."""
    expr = expr.strip()
    if not expr:
        return None
    # Only allow digits, operators, parentheses, spaces, and decimal points
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def math_gap_arithmetic_generator(seed: int) -> Problem:
    """Generate a multi-step arithmetic problem with a known solution.

    Builds a compound expression of the form ((a op b) op c) ... and records
    the intermediate results so the verifier can check each step.
    """
    rng = random.Random(seed)
    n_steps = rng.randint(2, 4)
    ops = ["+", "-", "*"]
    # Avoid leading negatives for readability
    values = [rng.randint(1, 20) for _ in range(n_steps + 1)]
    chosen_ops = [rng.choice(ops) for _ in range(n_steps)]

    # Build the expression and the step-by-step solution
    current = float(values[0])
    expr_parts = [str(values[0])]
    steps: list[str] = []
    for i, op in enumerate(chosen_ops):
        nxt = float(values[i + 1])
        if op == "+":
            result = current + nxt
        elif op == "-":
            result = current - nxt
        else:  # "*"
            result = current * nxt
        steps.append(f"{_fmt(current)} {op} {_fmt(nxt)} = {_fmt(result)}")
        expr_parts.append(op)
        expr_parts.append(str(values[i + 1]))
        current = result

    expression = " ".join(expr_parts)
    answer = _fmt(current)

    # Wrap in parentheses to make order explicit (left-to-right)
    if n_steps >= 2:
        paren_expr = expr_parts[0]
        for i in range(n_steps):
            paren_expr = f"({paren_expr} {chosen_ops[i]} {values[i + 1]})"
        expression = paren_expr

    prompt = (
        f"Compute the following arithmetic expression step by step.\n\n"
        f"Expression: {expression}\n\n"
        f"Show each intermediate computation on its own line starting with 'STEP:'.\n"
        f"End with 'ANSWER: <final value>'."
    )

    return Problem(
        id=f"math_gap_arithmetic_{seed}",
        prompt=prompt,
        difficulty=min(1.0, 0.2 + n_steps * 0.2),
        metadata={
            "problem": expression,
            "steps": steps,
            "answer": answer,
            "n_steps": n_steps,
            "values": values,
            "ops": chosen_ops,
        },
        token_budget=512,
        source="math_gap_arithmetic_generator",
    )


math_gap_arithmetic_generator.__test__ = False  # type: ignore[attr-defined]


def _fmt(x: float) -> str:
    """Format a float as an int string when it is whole."""
    if x == int(x):
        return str(int(x))
    return str(x)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class MathGapArithmeticVerifier(Verifier):
    """Verify a step-by-step arithmetic solution.

    reward = answer_correct * 0.6 + steps_valid * 0.4
    """

    def __init__(self, answer: str, steps: list[str], n_steps: int):
        super().__init__()
        self._answer = answer.strip()
        self._steps = steps
        self._n_steps = n_steps

    def verify(self, response: str) -> VerifierResult:
        # Parse STEP lines
        step_lines = re.findall(
            r"STEP\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )

        # Parse ANSWER line
        ans_match = re.search(
            r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        given_answer = ans_match.group(1).strip() if ans_match else ""

        # Check answer correctness
        answer_correct = self._check_answer(given_answer)

        # Check each step's arithmetic
        if step_lines:
            valid = 0
            for sl in step_lines:
                if self._step_valid(sl):
                    valid += 1
            steps_valid = valid / len(step_lines)
        else:
            steps_valid = 0.0

        score = answer_correct * 0.6 + steps_valid * 0.4
        correct = answer_correct >= 1.0 and steps_valid >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "answer_correct": float(answer_correct),
                "steps_valid": steps_valid,
                "n_steps_given": float(len(step_lines)),
            },
            diagnostics=(
                f"answer={answer_correct:.2f} "
                f"steps_valid={steps_valid:.2f} "
                f"({len(step_lines)} steps given, expected {self._n_steps})"
            ),
        )

    def _check_answer(self, given: str) -> float:
        """Return 1.0 if the answer matches, 0.0 otherwise."""
        if not given:
            return 0.0
        try:
            if float(given) == float(self._answer):
                return 1.0
        except (ValueError, TypeError):
            pass
        # Normalized string comparison
        if given.strip() == self._answer.strip():
            return 1.0
        # Number-based fallback
        given_nums = _extract_numbers(given)
        expected_nums = _extract_numbers(self._answer)
        if given_nums and expected_nums and given_nums[-1] == expected_nums[-1]:
            return 1.0
        return 0.0

    def _step_valid(self, step: str) -> bool:
        """Check a single 'a op b = c' step for arithmetic validity."""
        step = step.strip()
        if not step:
            return False
        # Match "lhs op rhs = result"
        m = re.search(
            r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
            step,
        )
        if m:
            a, op, b, c = (
                float(m.group(1)),
                m.group(2),
                float(m.group(3)),
                float(m.group(4)),
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
        # Try evaluating a full expression with eval
        if "=" in step:
            lhs, rhs = step.split("=", 1)
            lv = _eval_expr(lhs)
            rv = _eval_expr(rhs)
            if lv is not None and rv is not None and lv == rv:
                return True
        # Prose steps mentioning operations count as structural
        if re.search(
            r"(add|subtract|multiply|divide|compute|equals|therefore|so)",
            step,
            re.IGNORECASE,
        ):
            return True
        return len(step) > 8


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MathGapArithmeticEnv(BatchEnvBase):
    """MathGapArithmetic: multi-step arithmetic with step verification.

    Batch-aware: N parallel solutions; reward = best score across the batch.
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
            problem_generator = math_gap_arithmetic_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return MathGapArithmeticVerifier(
            answer=problem.metadata["answer"],
            steps=problem.metadata["steps"],
            n_steps=problem.metadata["n_steps"],
        )

    def _check_format(self, response: str) -> float:
        has_step = bool(re.search(r"STEP\s*:", response, re.IGNORECASE))
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        if has_step and has_answer:
            return 1.0
        if has_step or has_answer:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "any_correct": any(corrects),
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": best_score,
            "correct": any(corrects),
            "diagnostics": (
                f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores)/len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
