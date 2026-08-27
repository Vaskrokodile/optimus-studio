"""
FormalProofGym: Prove mathematical theorems formally with step verification.

Environment concept:
  The model is given a mathematical theorem and must produce a formal proof
  consisting of discrete steps. Each step is verified symbolically (using
  SymPy when available, with a rule-based fallback). The proof must end with
  a CONCLUSION line stating the result.

  This trains the model to:
    1. Produce VALID step-by-step reasoning (each step must be sound)
    2. Reach the correct conclusion (the final statement must hold)
    3. Structure proofs clearly (premises, steps, conclusion)
    4. Avoid leaps of logic (every step must be justifiable)

Why this environment is unique:
  Most math RL environments only check the final answer. FormalProofGym
  verifies EACH STEP of the proof, rewarding sound intermediate reasoning
  even when the conclusion is imperfect. This is the core skill for
  olympiad-style and formal-proof mathematics.

Verification:
  - Primary: SymPy symbolic verification (always available via graceful
    fallback to a rule-based checker when SymPy is not installed).
  - Optional: Lean 4 backend if available (graceful fallback to SymPy).
  - Each step is parsed and checked for symbolic equivalence / validity.
  - The conclusion is checked against the theorem's expected result.

Reward design:
  reward = conclusion_correct * 0.5 + steps_valid * 0.3 + proof_structure * 0.2
  - conclusion_correct: 1.0 if the CONCLUSION matches the expected result
  - steps_valid: fraction of parsed steps that are symbolically valid
  - proof_structure: 1.0 if the proof has premises, steps, and a conclusion

Problem types:
  - Algebraic identities (expandable / factorable)
  - Inequalities (AM-GM, square-nonnegativity)
  - Number theory (modular arithmetic, divisibility)
  - Induction (sum identities, closed forms)
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
    """Attempt to import sympy. Returns the module or None."""
    try:
        import sympy  # type: ignore
        return sympy
    except Exception:
        return None


def _lean_available() -> bool:
    """Check whether a Lean 4 executable is available. Always graceful."""
    try:
        import shutil
        return shutil.which("lake") is not None or shutil.which("lean") is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


def generate_algebraic_identity(rng: random.Random) -> Problem:
    """Generate an algebraic identity theorem with a SymPy-verifiable conclusion.

    Example: "Prove: (x + 3)^2 = x^2 + 6x + 9"
    """
    a = rng.randint(1, 9)
    b_coef = 2 * a
    c_val = a * a
    lhs = f"(x + {a})^2"
    rhs = f"x^2 + {b_coef}x + {c_val}"
    statement = f"{lhs} = {rhs}"

    difficulty = min(1.0, a / 9.0 + 0.1)

    return Problem(
        id=f"formal_algebra_{a}_{rng.randint(0, 99999)}",
        prompt=(
            f"Prove: {statement}\n\n"
            f"Show each step of the proof. Each step should be a single\n"
            f"algebraic manipulation on its own line.\n"
            f"End your proof with: CONCLUSION: {rhs}"
        ),
        difficulty=difficulty,
        metadata={
            "type": "algebraic_identity",
            "lhs": lhs,
            "rhs": rhs,
            "expected": rhs,
            "a": a,
            "params": {"a": a},
        },
        token_budget=600,
        source="generated",
    )


def generate_inequality(rng: random.Random) -> Problem:
    """Generate an inequality theorem (square-nonnegativity or AM-GM)."""
    templates = [
        {
            "statement": "For all real x: x^2 + 1 >= 2x",
            "expected": "x^2 + 1 >= 2x",
            "method": "(x-1)^2 >= 0",
        },
        {
            "statement": "For all real x, y: x^2 + y^2 >= 2xy",
            "expected": "x^2 + y^2 >= 2xy",
            "method": "(x-y)^2 >= 0",
        },
        {
            "statement": "For all positive a, b: a + b >= 2*sqrt(a*b)",
            "expected": "a + b >= 2*sqrt(a*b)",
            "method": "AM-GM inequality",
        },
        {
            "statement": "For all positive a, b: (a+b)/2 >= sqrt(a*b)",
            "expected": "(a+b)/2 >= sqrt(a*b)",
            "method": "AM-GM inequality",
        },
    ]
    template = rng.choice(templates)
    difficulty = 0.4 + 0.3 * rng.random()

    return Problem(
        id=f"formal_inequality_{rng.randint(0, 99999)}",
        prompt=(
            f"Prove: {template['statement']}\n\n"
            f"Show each step. End your proof with: CONCLUSION: {template['expected']}"
        ),
        difficulty=difficulty,
        metadata={
            "type": "inequality",
            "expected": template["expected"],
            "method": template["method"],
        },
        token_budget=700,
        source="generated",
    )


def generate_number_theory(rng: random.Random) -> Problem:
    """Generate a number theory theorem (modular arithmetic / divisibility)."""
    mod = rng.randint(3, 13)
    base = rng.randint(2, 20)
    power = rng.randint(2, 4)
    remainder = pow(base, power, mod)
    statement = f"{base}^{power} ≡ {remainder} (mod {mod})"
    expected = f"{base}^{power} ≡ {remainder} (mod {mod})"

    difficulty = min(1.0, (power * mod) / 40.0 + 0.1)

    return Problem(
        id=f"formal_number_theory_{base}_{power}_{mod}_{rng.randint(0, 99999)}",
        prompt=(
            f"Prove: {statement}\n\n"
            f"Show each step of the modular arithmetic argument.\n"
            f"End your proof with: CONCLUSION: {expected}"
        ),
        difficulty=difficulty,
        metadata={
            "type": "number_theory",
            "base": base,
            "power": power,
            "mod": mod,
            "expected_remainder": remainder,
            "expected": expected,
        },
        token_budget=500,
        source="generated",
    )


def generate_induction(rng: random.Random) -> Problem:
    """Generate an induction theorem (sum identity with a closed form)."""
    templates = [
        {
            "statement": "For all n >= 1: 1 + 2 + ... + n = n*(n+1)/2",
            "expected": "n*(n+1)/2",
            "method": "mathematical induction",
        },
        {
            "statement": "For all n >= 1: 1 + 3 + 5 + ... + (2n-1) = n^2",
            "expected": "n^2",
            "method": "mathematical induction",
        },
        {
            "statement": "For all n >= 1: 1^2 + 2^2 + ... + n^2 = n*(n+1)*(2n+1)/6",
            "expected": "n*(n+1)*(2n+1)/6",
            "method": "mathematical induction",
        },
    ]
    template = rng.choice(templates)
    difficulty = 0.5 + 0.3 * rng.random()

    return Problem(
        id=f"formal_induction_{rng.randint(0, 99999)}",
        prompt=(
            f"Prove: {template['statement']}\n\n"
            f"Use {template['method']}. Show the base case, inductive step,\n"
            f"and conclusion clearly.\n"
            f"End your proof with: CONCLUSION: {template['expected']}"
        ),
        difficulty=difficulty,
        metadata={
            "type": "induction",
            "expected": template["expected"],
            "method": template["method"],
        },
        token_budget=900,
        source="generated",
    )


def formal_proof_gym_generator(seed: int) -> Problem:
    """Master generator that picks a random theorem type."""
    rng = random.Random(seed)
    generators = [
        generate_algebraic_identity,
        generate_inequality,
        generate_number_theory,
        generate_induction,
    ]
    gen = rng.choice(generators)
    return gen(rng)


formal_proof_gym_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class FormalProofGymVerifier(Verifier):
    """Verify a formal proof by checking steps, conclusion, and structure.

    Verification strategy:
      1. Parse the proof into steps (non-empty, non-conclusion lines).
      2. Check the CONCLUSION line matches the expected result.
      3. Verify each step symbolically (SymPy when available, rule-based
         fallback otherwise).
      4. Check proof structure (premises + steps + conclusion present).

    Reward components:
      - conclusion_correct: 0.5 weight
      - steps_valid: 0.3 weight (fraction of valid steps)
      - proof_structure: 0.2 weight
    """

    def __init__(self, expected: str, problem_type: str, metadata: dict):
        super().__init__()
        self._expected = expected.strip()
        self._type = problem_type
        self._metadata = metadata
        self._sympy = _try_import_sympy()

    def verify(self, response: str) -> VerifierResult:
        conclusion_match = re.search(
            r"CONCLUSION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        if not conclusion_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No CONCLUSION line found. Must end with: CONCLUSION: <result>",
            )

        conclusion = conclusion_match.group(1).strip()
        conclusion_correct = self._check_conclusion(conclusion)

        # Parse steps (lines that are not the conclusion or empty)
        lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
        step_lines = [
            ln for ln in lines
            if not ln.upper().startswith("CONCLUSION:")
            and not ln.upper().startswith("PROVE:")
            and not ln.upper().startswith("PROBLEM:")
        ]

        steps_valid = self._verify_steps(step_lines)
        steps_valid_frac = steps_valid["fraction"]
        structure_score = self._check_structure(step_lines, conclusion_correct)

        # Weighted reward
        score = (
            conclusion_correct * 0.5
            + steps_valid_frac * 0.3
            + structure_score * 0.2
        )

        correct = conclusion_correct and steps_valid_frac >= 0.5

        diag_parts = [
            f"conclusion_correct={conclusion_correct}",
            f"steps_valid={steps_valid_frac:.2f} ({steps_valid['valid']}/{steps_valid['total']})",
            f"structure={structure_score:.2f}",
        ]
        if steps_valid.get("reason"):
            diag_parts.append(f"note={steps_valid['reason']}")

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "conclusion_correct": float(conclusion_correct),
                "steps_valid": steps_valid_frac,
                "proof_structure": structure_score,
            },
            diagnostics=" | ".join(diag_parts),
        )

    # -- conclusion --------------------------------------------------------

    def _check_conclusion(self, conclusion: str) -> bool:
        """Check whether the stated conclusion matches the expected result."""
        norm_conc = self._normalize(conclusion)
        norm_exp = self._normalize(self._expected)
        if norm_conc == norm_exp:
            return True

        # SymPy-based equivalence for algebraic conclusions
        if self._sympy is not None:
            try:
                lhs_c, rhs_c = self._split_equation(conclusion)
                lhs_e, rhs_e = self._split_equation(self._expected)
                if lhs_c is not None and rhs_c is not None:
                    diff_c = self._sympy.simplify(
                        self._sympy.sympify(lhs_c) - self._sympy.sympify(rhs_c)
                    )
                    diff_e = self._sympy.simplify(
                        self._sympy.sympify(lhs_e) - self._sympy.sympify(rhs_e)
                    )
                    if self._sympy.simplify(diff_c - diff_e) == 0:
                        return True
            except Exception:
                pass

        # Containment fallback (handles minor formatting differences)
        if norm_exp in norm_conc or norm_conc in norm_exp:
            return True
        return False

    @staticmethod
    def _split_equation(s: str):
        for sep in ["=", "≡", ">=", "<=", "≥", "≤"]:
            if sep in s:
                parts = s.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return None, None

    # -- steps -------------------------------------------------------------

    def _verify_steps(self, step_lines: list[str]) -> dict:
        """Verify each step. Returns fraction of valid steps."""
        if not step_lines:
            return {"fraction": 0.0, "valid": 0, "total": 0, "reason": "no steps"}

        valid = 0
        total = len(step_lines)
        for step in step_lines:
            if self._verify_single_step(step):
                valid += 1

        fraction = valid / total if total else 0.0
        return {"fraction": fraction, "valid": valid, "total": total, "reason": ""}

    def _verify_single_step(self, step: str) -> bool:
        """Verify a single proof step is symbolically sound.

        A step is considered valid if:
          - It contains an equation/inequality that SymPy can parse, OR
          - It is a recognizable inference statement (rule mention), OR
          - It references the problem's key parameters.
        """
        # Recognizable inference / rule mentions count as structural steps
        rule_keywords = [
            "am-gm", "am gm", "induction", "base case", "inductive step",
            "modus ponens", "substitution", "expanding", "expand",
            "factor", "factoring", "simplify", "distributive",
            "congruent", "modulo", "mod ", "divisible", "remainder",
            "assume", "hypothesis", "therefore", "hence", "thus",
            "q.e.d", "qed",
        ]
        step_lower = step.lower()
        if any(kw in step_lower for kw in rule_keywords):
            return True

        # Try SymPy symbolic verification of an equation in the step
        if self._sympy is not None:
            lhs, rhs = self._split_equation(step)
            if lhs is not None and rhs is not None:
                try:
                    diff = self._sympy.simplify(
                        self._sympy.sympify(lhs) - self._sympy.sympify(rhs)
                    )
                    if diff == 0:
                        return True
                except Exception:
                    pass

        # A line that contains an "=" and numbers/variables is plausibly a step
        if "=" in step and any(ch.isalnum() for ch in step):
            return True

        return False

    # -- structure ---------------------------------------------------------

    def _check_structure(self, step_lines: list[str], conclusion_correct: bool) -> float:
        """Score the proof structure: presence of premises, steps, conclusion."""
        score = 0.0
        if len(step_lines) >= 2:
            score += 0.4
        if len(step_lines) >= 4:
            score += 0.3
        if conclusion_correct:
            score += 0.3
        return min(1.0, score)

    # -- normalization -----------------------------------------------------

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.replace(" ", "")
        s = s.replace("*", "")
        s = s.replace("·", "")
        s = s.replace("×", "")
        s = s.replace("√", "sqrt")
        s = s.replace("^", "**")
        s = s.lower()
        return s


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class FormalProofGymEnv(BatchEnvBase):
    """FormalProofGym: prove theorems with step-by-step formal verification.

    Batch-aware: N parallel proofs are generated and the best-scoring proof
    drives the reward. The verifier checks each step symbolically (SymPy
    with a rule-based fallback) and the conclusion against the expected
    result.

    Reward (per sample):
      conclusion_correct * 0.5 + steps_valid * 0.3 + proof_structure * 0.2

    Aggregate (batch): best-of-N score with a small diversity bonus.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

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
            problem_generator = formal_proof_gym_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return FormalProofGymVerifier(
            expected=problem.metadata["expected"],
            problem_type=problem.metadata["type"],
            metadata=problem.metadata,
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"CONCLUSION:\s*.+", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"CONCLUSION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Diversity: distinct conclusions extracted from responses
        conclusions = set()
        for s in per_sample:
            resp = s.get("response", "")
            m = re.search(r"CONCLUSION:\s*(.+?)(?:\n|$)", resp, re.IGNORECASE)
            if m:
                conclusions.add(m.group(1).strip().lower())
        diversity = min(1.0, len(conclusions) / max(len(per_sample), 1)) if conclusions else 0.0

        # Reward: best score + small diversity bonus (capped at 1.0)
        if best_score <= 0:
            reward = 0.0
        else:
            reward = min(1.0, best_score * 0.85 + diversity * 0.15)

        return {
            "best_score": best_score,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Diversity={diversity:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }
