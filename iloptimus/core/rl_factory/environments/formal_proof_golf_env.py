"""
FormalProofGolf: Shortest formal proof wins.

Environment concept:
  The model is given an algebraic identity or simple theorem to prove and must
  produce a formal proof that is both VALID and CONCISE. Like proof golf, the
  reward rewards shorter proofs (fewer tokens) that still reach the correct
  conclusion with valid steps.

  This trains the model to:
    1. Produce valid step-by-step formal reasoning
    2. Compress proofs to their essential steps
    3. Reach the correct conclusion efficiently

Verification:
  - STEP lines are verified symbolically (SymPy when available).
  - The CONCLUSION line is checked against the expected result.
  - Compression ratio = max(0, 1 - proof_tokens / reference_tokens).

Reward design:
  validity * (0.5 + 0.5 * compression)
  where validity = conclusion_correct * 0.6 + steps_valid * 0.4
  and compression = max(0, 1 - proof_tokens / reference_tokens)
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
# Reference proof bank
# Each entry: (theorem statement, expected conclusion, reference steps)
# ---------------------------------------------------------------------------


_REFERENCES = [
    {
        "theorem": "Prove: (x + 2)^2 = x^2 + 4x + 4",
        "expected_conclusion": "x^2 + 4x + 4",
        "reference_steps": [
            "(x + 2)^2 = (x + 2)(x + 2)",
            "(x + 2)(x + 2) = x^2 + 2x + 2x + 4",
            "x^2 + 2x + 2x + 4 = x^2 + 4x + 4",
        ],
    },
    {
        "theorem": "Prove: (x + 5)^2 = x^2 + 10x + 25",
        "expected_conclusion": "x^2 + 10x + 25",
        "reference_steps": [
            "(x + 5)^2 = (x + 5)(x + 5)",
            "(x + 5)(x + 5) = x^2 + 5x + 5x + 25",
            "x^2 + 5x + 5x + 25 = x^2 + 10x + 25",
        ],
    },
    {
        "theorem": "Prove: (x + 3)(x - 3) = x^2 - 9",
        "expected_conclusion": "x^2 - 9",
        "reference_steps": [
            "(x + 3)(x - 3) = x^2 - 3x + 3x - 9",
            "x^2 - 3x + 3x - 9 = x^2 - 9",
        ],
    },
    {
        "theorem": "Prove: (2x + 1)^2 = 4x^2 + 4x + 1",
        "expected_conclusion": "4x^2 + 4x + 1",
        "reference_steps": [
            "(2x + 1)^2 = (2x + 1)(2x + 1)",
            "(2x + 1)(2x + 1) = 4x^2 + 2x + 2x + 1",
            "4x^2 + 2x + 2x + 1 = 4x^2 + 4x + 1",
        ],
    },
    {
        "theorem": "Prove: (x + 4)^2 = x^2 + 8x + 16",
        "expected_conclusion": "x^2 + 8x + 16",
        "reference_steps": [
            "(x + 4)^2 = (x + 4)(x + 4)",
            "(x + 4)(x + 4) = x^2 + 4x + 4x + 16",
            "x^2 + 4x + 4x + 16 = x^2 + 8x + 16",
        ],
    },
    {
        "theorem": "Prove: (x + 1)(x + 2) = x^2 + 3x + 2",
        "expected_conclusion": "x^2 + 3x + 2",
        "reference_steps": [
            "(x + 1)(x + 2) = x^2 + 2x + x + 2",
            "x^2 + 2x + x + 2 = x^2 + 3x + 2",
        ],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def formal_proof_golf_generator(seed: int) -> Problem:
    """Generate a formal proof golf problem: prove concisely."""
    rng = random.Random(seed)
    ref = rng.choice(_REFERENCES)
    theorem = ref["theorem"]
    expected_conclusion = ref["expected_conclusion"]
    reference_steps = ref["reference_steps"]
    n_reference_steps = len(reference_steps)

    prompt = (
        f"{theorem}\n\n"
        f"Prove this as concisely as possible. Each step should be a single\n"
        f"algebraic manipulation on its own line starting with 'STEP:'.\n"
        f"End with 'CONCLUSION: {expected_conclusion}'.\n"
        f"Shorter valid proofs score higher."
    )

    return Problem(
        id=f"formal_proof_golf_{seed}",
        prompt=prompt,
        difficulty=0.3 + n_reference_steps * 0.1,
        metadata={
            "theorem": theorem,
            "expected_conclusion": expected_conclusion,
            "reference_steps": reference_steps,
            "n_reference_steps": n_reference_steps,
        },
        token_budget=512,
        source="formal_proof_golf_generator",
    )


formal_proof_golf_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class FormalProofGolfVerifier(Verifier):
    """Verify a formal proof for validity and compression.

    reward = validity * (0.5 + 0.5 * compression)
    where validity = conclusion_correct * 0.6 + steps_valid * 0.4
    and compression = max(0, 1 - proof_tokens / reference_tokens)
    """

    def __init__(
        self,
        expected_conclusion: str,
        reference_steps: list[str],
        n_reference_steps: int,
    ):
        super().__init__()
        self._expected = expected_conclusion.strip()
        self._reference_steps = reference_steps
        self._n_reference_steps = n_reference_steps
        self._sympy = _try_import_sympy()

    def verify(self, response: str) -> VerifierResult:
        # Parse CONCLUSION
        conc_match = re.search(
            r"CONCLUSION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        if not conc_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No CONCLUSION line found.",
            )
        conclusion = conc_match.group(1).strip()
        conclusion_correct = self._check_conclusion(conclusion)

        # Parse STEP lines
        step_lines = re.findall(
            r"STEP\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        steps_valid = self._verify_steps(step_lines)
        steps_valid_frac = steps_valid["fraction"]

        validity = conclusion_correct * 0.6 + steps_valid_frac * 0.4

        # Compression: compare proof tokens to reference tokens
        proof_tokens = self._count_tokens(response)
        reference_tokens = self._count_reference_tokens()
        if reference_tokens > 0:
            compression = max(0.0, 1.0 - proof_tokens / reference_tokens)
        else:
            compression = 0.0

        score = validity * (0.5 + 0.5 * compression)
        correct = conclusion_correct and steps_valid_frac >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "conclusion_correct": float(conclusion_correct),
                "steps_valid": steps_valid_frac,
                "compression": compression,
                "proof_tokens": float(proof_tokens),
                "reference_tokens": float(reference_tokens),
                "validity": float(validity),
            },
            diagnostics=(
                f"validity={validity:.2f} "
                f"conclusion={conclusion_correct:.2f} "
                f"steps_valid={steps_valid_frac:.2f} "
                f"compression={compression:.2f} "
                f"({proof_tokens}/{reference_tokens} tokens)"
            ),
        )

    def _count_tokens(self, text: str) -> int:
        """Approximate token count as word count."""
        return max(1, len(text.split()))

    def _count_reference_tokens(self) -> int:
        ref_text = "\n".join(
            f"STEP: {s}" for s in self._reference_steps
        ) + f"\nCONCLUSION: {self._expected}"
        return self._count_tokens(ref_text)

    def _check_conclusion(self, conclusion: str) -> bool:
        norm_conc = self._normalize(conclusion)
        norm_exp = self._normalize(self._expected)
        if norm_conc == norm_exp:
            return True
        # SymPy equivalence
        if self._sympy is not None:
            try:
                diff = self._sympy.simplify(
                    self._sympy.sympify(norm_conc) - self._sympy.sympify(norm_exp)
                )
                if diff == 0:
                    return True
            except Exception:
                pass
        # Containment fallback
        if norm_exp in norm_conc or norm_conc in norm_exp:
            return True
        return False

    def _verify_steps(self, step_lines: list[str]) -> dict:
        if not step_lines:
            return {"fraction": 0.0, "valid": 0, "total": 0}
        valid = 0
        for step in step_lines:
            if self._verify_single_step(step):
                valid += 1
        fraction = valid / len(step_lines) if step_lines else 0.0
        return {"fraction": fraction, "valid": valid, "total": len(step_lines)}

    def _verify_single_step(self, step: str) -> bool:
        step = step.strip()
        if not step:
            return False
        # SymPy symbolic check for equations
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
        # Rule keywords count as structural steps
        rule_keywords = [
            "expand", "expanding", "factor", "factoring", "simplify",
            "distributive", "substitution", "foil", "therefore", "hence",
            "thus", "so", "equals",
        ]
        step_lower = step.lower()
        if any(kw in step_lower for kw in rule_keywords):
            return True
        # A line with "=" and alphanumerics is plausibly a step
        if "=" in step and any(ch.isalnum() for ch in step):
            return True
        return False

    @staticmethod
    def _split_equation(s: str):
        for sep in ["=", "≡"]:
            if sep in s:
                parts = s.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return None, None

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.replace(" ", "")
        s = s.replace("*", "")
        s = s.replace("^", "**")
        s = s.lower()
        return s


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class FormalProofGolfEnv(BatchEnvBase):
    """FormalProofGolf: shortest valid formal proof wins.

    Batch-aware: N parallel proofs; reward = best (most compressed valid).
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
            problem_generator = formal_proof_golf_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return FormalProofGolfVerifier(
            expected_conclusion=problem.metadata["expected_conclusion"],
            reference_steps=problem.metadata["reference_steps"],
            n_reference_steps=problem.metadata["n_reference_steps"],
        )

    def _check_format(self, response: str) -> float:
        has_step = bool(re.search(r"STEP\s*:", response, re.IGNORECASE))
        has_conc = bool(re.search(r"CONCLUSION\s*:", response, re.IGNORECASE))
        if has_step and has_conc:
            return 1.0
        if has_step or has_conc:
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
        match = re.search(r"CONCLUSION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
