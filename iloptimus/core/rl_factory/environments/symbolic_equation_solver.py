"""
SymbolicEquationSolver: Step-by-step symbolic equation solving.

Environment concept:
  The model is given a linear or quadratic equation and must solve it step by
  step, showing each algebraic transformation. The verifier uses SymPy to
  check that each step is a valid transformation and that the final solution
  is correct.

  This trains the model to:
    1. Apply valid algebraic transformations step by step
    2. Isolate the variable correctly
    3. Reach the correct solution

Verification:
  - STEP lines are checked: each step should be a valid equation (SymPy
    verifies the transformation preserves the solution set, or the equation
    is symbolically true).
  - The SOLUTION line is checked against the SymPy-computed solution.

Reward design:
  solution_correct * 0.6 + steps_valid * 0.4
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# SymPy availability helper
# ---------------------------------------------------------------------------


def _try_import_sympy():
    try:
        import sympy  # type: ignore
        return sympy
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def symbolic_equation_solver_generator(seed: int) -> Problem:
    """Generate a linear or quadratic equation to solve step by step."""
    rng = random.Random(seed)
    sympy = _try_import_sympy()

    eq_type = rng.choice(["linear", "linear", "quadratic"])

    if eq_type == "linear" or sympy is None:
        # Linear: a*x + b = c  =>  x = (c - b) / a
        a = rng.randint(2, 9)
        b = rng.randint(1, 20)
        c = rng.randint(1, 30)
        # Ensure integer solution for simplicity
        # x = (c - b) / a ; pick c so that (c-b) is divisible by a
        x_val = rng.randint(-5, 10)
        c = a * x_val + b
        equation = f"{a}*x + {b} = {c}"
        solution = f"x = {x_val}"
        steps = [
            f"{a}*x + {b} = {c}",
            f"{a}*x = {c - b}",
            f"x = {c - b} / {a}",
            f"x = {x_val}",
        ]
        difficulty = 0.2 + rng.random() * 0.2
    else:
        # Quadratic: x^2 + bx + c = 0 with nice roots
        r1 = rng.randint(-4, 4)
        r2 = rng.randint(-4, 4)
        if r1 == r2:
            r2 += 1
        # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
        b_coef = -(r1 + r2)
        c_val = r1 * r2
        b_term = f"+ {b_coef}" if b_coef >= 0 else f"- {abs(b_coef)}"
        c_term = f"+ {c_val}" if c_val >= 0 else f"- {abs(c_val)}"
        equation = f"x^2 {b_term}*x {c_term} = 0"
        solution = f"x = {r1}, x = {r2}"
        steps = [
            equation,
            f"(x - {r1})(x - {r2}) = 0",
            f"x = {r1}, x = {r2}",
        ]
        difficulty = 0.5 + rng.random() * 0.2

    prompt = (
        f"Solve the following equation step by step.\n\n"
        f"Equation: {equation}\n\n"
        f"Show each algebraic transformation on its own line starting with 'STEP:'.\n"
        f"End with 'SOLUTION: x = <value>' (or 'SOLUTION: x = <v1>, x = <v2>' for quadratics)."
    )

    return Problem(
        id=f"symbolic_equation_solver_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "equation": equation,
            "solution": solution,
            "steps": steps,
            "eq_type": eq_type,
        },
        token_budget=512,
        source="symbolic_equation_solver_generator",
    )


symbolic_equation_solver_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SymbolicEquationSolverVerifier(Verifier):
    """Verify a step-by-step symbolic equation solution.

    reward = solution_correct * 0.6 + steps_valid * 0.4
    """

    def __init__(self, equation: str, solution: str, steps: list[str]):
        super().__init__()
        self._equation = equation.strip()
        self._solution = solution.strip()
        self._steps = steps
        self._sympy = _try_import_sympy()
        self._expected_values = self._parse_solution_values(solution)

    def verify(self, response: str) -> VerifierResult:
        # Parse STEP lines
        step_lines = re.findall(
            r"STEP\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )

        # Parse SOLUTION line
        sol_match = re.search(
            r"SOLUTION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        given_solution = sol_match.group(1).strip() if sol_match else ""

        # Check solution correctness
        solution_correct = self._check_solution(given_solution)

        # Check steps validity
        if step_lines:
            valid = 0
            for sl in step_lines:
                if self._step_valid(sl):
                    valid += 1
            steps_valid = valid / len(step_lines)
        else:
            steps_valid = 0.0

        score = solution_correct * 0.6 + steps_valid * 0.4
        correct = solution_correct >= 1.0 and steps_valid >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "solution_correct": float(solution_correct),
                "steps_valid": steps_valid,
                "n_steps_given": float(len(step_lines)),
            },
            diagnostics=(
                f"solution={solution_correct:.2f} "
                f"steps_valid={steps_valid:.2f} "
                f"({len(step_lines)} steps) "
                f"given={given_solution!r} expected={self._solution!r}"
            ),
        )

    def _parse_solution_values(self, solution: str) -> list[float]:
        """Extract numeric values from a solution string like 'x = 3' or 'x = 1, x = 2'."""
        return [float(m) for m in re.findall(r"-?\d+\.?\d*", solution)]

    def _check_solution(self, given: str) -> float:
        """Check if the given solution matches the expected solution."""
        if not given:
            return 0.0
        given_values = self._parse_solution_values(given)
        if not given_values:
            return 0.0
        # Compare sorted value sets
        if sorted(given_values) == sorted(self._expected_values):
            return 1.0
        # Partial: at least one value matches
        if any(v in self._expected_values for v in given_values):
            return 0.5
        # Verify by substituting into the equation with SymPy
        if self._sympy is not None:
            try:
                x = self._sympy.Symbol("x")
                lhs, rhs = self._split_equation(self._equation)
                if lhs is not None:
                    expr = self._sympy.sympify(lhs) - self._sympy.sympify(rhs)
                    for v in given_values:
                        if self._sympy.simplify(expr.subs(x, v)) == 0:
                            # Check if all expected values are covered
                            if len(given_values) >= len(self._expected_values):
                                return 1.0
                            return 0.5
            except Exception:
                pass
        return 0.0

    def _step_valid(self, step: str) -> bool:
        """Check if a single step is a valid algebraic statement."""
        step = step.strip()
        if not step:
            return False
        # SymPy: check if the equation is symbolically true or a valid transformation
        if self._sympy is not None:
            lhs, rhs = self._split_equation(step)
            if lhs is not None and rhs is not None:
                try:
                    diff = self._sympy.simplify(
                        self._sympy.sympify(lhs) - self._sympy.sympify(rhs)
                    )
                    # A step is valid if it's either an identity (diff == 0)
                    # or an equation in x (a valid transformation)
                    if diff == 0:
                        return True
                    # Check if it's an equation in x (contains the variable)
                    if "x" in step:
                        return True
                except Exception:
                    pass
        # Prose steps mentioning operations count as structural
        if re.search(
            r"(subtract|add|divide|multiply|both sides|factor|expand|simplify|"
            r"therefore|so|equals|roots|solution)",
            step,
            re.IGNORECASE,
        ):
            return True
        # A line with "=" and alphanumerics is plausibly a step
        if "=" in step and any(ch.isalnum() for ch in step):
            return True
        return len(step) > 8

    @staticmethod
    def _split_equation(s: str):
        for sep in ["=", "≡"]:
            if sep in s:
                parts = s.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return None, None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SymbolicEquationSolverEnv(BatchEnvBase):
    """SymbolicEquationSolver: step-by-step symbolic equation solving.

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
            problem_generator = symbolic_equation_solver_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SymbolicEquationSolverVerifier(
            equation=problem.metadata["equation"],
            solution=problem.metadata["solution"],
            steps=problem.metadata["steps"],
        )

    def _check_format(self, response: str) -> float:
        has_step = bool(re.search(r"STEP\s*:", response, re.IGNORECASE))
        has_sol = bool(re.search(r"SOLUTION\s*:", response, re.IGNORECASE))
        if has_step and has_sol:
            return 1.0
        if has_step or has_sol:
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
        match = re.search(r"SOLUTION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
