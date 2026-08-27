"""
ProofCompression: Compress a verbose proof to minimal steps.

Environment concept:
  The model is given a verbose, multi-step proof and must compress it to
  the minimal number of steps while preserving correctness. This trains
  proof elegance — not just correctness, but conciseness.

  Why: verbose proofs waste tokens and obscure the key insight. Training
  the model to compress proofs teaches it to identify which steps are
  essential and which are redundant — a core reasoning skill.

Problem types:
  - Arithmetic proofs with redundant intermediate steps
  - Algebraic derivations with unnecessary expansions
  - Logical arguments with restated premises

Verification:
  - Proof validity: the compressed proof reaches the correct conclusion.
  - Compression ratio: fewer steps than the original = higher reward.

Reward design:
  conclusion_correct * (0.4 + 0.3 * steps_valid + 0.3 * compression_ratio)
  where compression_ratio = max(0, 1 - compressed_steps / original_steps)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — verbose proofs to compress
# Each entry: (verbose_steps, conclusion, min_compressed_steps)
# ---------------------------------------------------------------------------

_PROOFS = [
    {
        "verbose_steps": [
            "We want to compute 6 * 7.",
            "We know that 6 * 7 = 6 * (5 + 2).",
            "6 * 5 = 30.",
            "6 * 2 = 12.",
            "30 + 12 = 42.",
            "Therefore 6 * 7 = 42.",
        ],
        "conclusion": "42",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to solve x + 3 = 10.",
            "We subtract 3 from both sides of the equation.",
            "x + 3 - 3 = 10 - 3.",
            "x = 10 - 3.",
            "10 - 3 = 7.",
            "Therefore x = 7.",
        ],
        "conclusion": "x = 7",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to compute 15 * 4.",
            "We can write 15 * 4 = 15 * (2 * 2).",
            "15 * 2 = 30.",
            "30 * 2 = 60.",
            "Therefore 15 * 4 = 60.",
        ],
        "conclusion": "60",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to solve 2x = 14.",
            "We divide both sides by 2.",
            "2x / 2 = 14 / 2.",
            "x = 14 / 2.",
            "14 / 2 = 7.",
            "Therefore x = 7.",
        ],
        "conclusion": "x = 7",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to compute 9 * 12.",
            "We write 9 * 12 = 9 * (10 + 2).",
            "9 * 10 = 90.",
            "9 * 2 = 18.",
            "90 + 18 = 108.",
            "Therefore 9 * 12 = 108.",
        ],
        "conclusion": "108",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to compute 100 - 45.",
            "We can write 100 - 45 = 100 - 40 - 5.",
            "100 - 40 = 60.",
            "60 - 5 = 55.",
            "Therefore 100 - 45 = 55.",
        ],
        "conclusion": "55",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to solve x - 8 = 12.",
            "We add 8 to both sides.",
            "x - 8 + 8 = 12 + 8.",
            "x = 12 + 8.",
            "12 + 8 = 20.",
            "Therefore x = 20.",
        ],
        "conclusion": "x = 20",
        "min_steps": 2,
    },
    {
        "verbose_steps": [
            "We want to compute 7 * 8.",
            "We write 7 * 8 = 7 * (4 * 2).",
            "7 * 4 = 28.",
            "28 * 2 = 56.",
            "Therefore 7 * 8 = 56.",
        ],
        "conclusion": "56",
        "min_steps": 2,
    },
]


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z=.\-+*/x\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def proof_compression_generator(seed: int) -> Problem:
    """Generate a ProofCompression problem: compress a verbose proof."""
    rng = random.Random(seed)
    proof_data = rng.choice(_PROOFS)

    verbose_steps = proof_data["verbose_steps"]
    conclusion = proof_data["conclusion"]
    min_steps = proof_data["min_steps"]

    verbose_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(verbose_steps))

    prompt = (
        f"Compress the following verbose proof to the minimal number of steps "
        f"while preserving correctness.\n\n"
        f"Verbose proof ({len(verbose_steps)} steps):\n{verbose_text}\n\n"
        f"Write a compressed proof. Format each step on its own line starting "
        f"with 'STEP:'\n"
        f"End with 'CONCLUSION: {conclusion}'"
    )

    return Problem(
        id=f"proof_compression_{seed}",
        prompt=prompt,
        difficulty=0.3 + len(verbose_steps) * 0.05,
        metadata={
            "type": "compression",
            "verbose_steps": verbose_steps,
            "n_verbose_steps": len(verbose_steps),
            "conclusion": conclusion,
            "min_steps": min_steps,
        },
        token_budget=384,
        source="proof_compression_generator",
    )


proof_compression_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ProofCompressionVerifier(Verifier):
    """Verify a compressed proof: validity + compression ratio.

    reward = conclusion_correct * (0.4 + 0.3 * steps_valid + 0.3 * compression)
    """

    def __init__(self, conclusion: str, n_verbose_steps: int, min_steps: int):
        super().__init__()
        self._conclusion = _normalize(conclusion)
        self._n_verbose = n_verbose_steps
        self._min_steps = min_steps

    def verify(self, response: str) -> VerifierResult:
        # Parse steps
        step_lines = re.findall(r"STEP\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if not step_lines:
            step_lines = re.findall(r"(?:^\d+\.?\s*)(.+?)(?:\n|$)", response, re.MULTILINE)

        # Parse conclusion
        conc_match = re.search(r"CONCLUSION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if conc_match:
            given_conc = _normalize(conc_match.group(1))
        else:
            lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
            given_conc = _normalize(lines[-1]) if lines else ""

        conclusion_correct = 1.0 if given_conc == self._conclusion else 0.0
        # Partial: number match
        if conclusion_correct == 0.0:
            conc_nums = _extract_numbers(self._conclusion)
            given_nums = _extract_numbers(given_conc)
            if conc_nums and given_nums and conc_nums[-1] == given_nums[-1]:
                conclusion_correct = 0.5

        # Steps valid: each step should contain a valid arithmetic statement
        if step_lines:
            valid = 0
            for sl in step_lines:
                if self._step_valid(sl):
                    valid += 1
            steps_valid = valid / len(step_lines)
        else:
            steps_valid = 0.0

        # Compression ratio
        n_compressed = len(step_lines) if step_lines else 1
        if self._n_verbose > 0:
            compression = max(0.0, 1.0 - n_compressed / self._n_verbose)
        else:
            compression = 0.0

        score = conclusion_correct * (0.4 + 0.3 * steps_valid + 0.3 * compression)
        correct = conclusion_correct >= 0.5 and steps_valid >= 0.5 and n_compressed < self._n_verbose

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "conclusion_correct": conclusion_correct,
                "steps_valid": steps_valid,
                "compression": compression,
                "n_compressed_steps": float(n_compressed),
            },
            diagnostics=(
                f"conclusion={conclusion_correct:.2f} "
                f"steps_valid={steps_valid:.2f} "
                f"compression={compression:.2f} "
                f"({n_compressed}/{self._n_verbose} steps)"
            ),
        )

    def _step_valid(self, step: str) -> bool:
        step = step.strip()
        if not step:
            return False
        expr = re.search(r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", step)
        if expr:
            a, op, b, c = float(expr.group(1)), expr.group(2), float(expr.group(3)), float(expr.group(4))
            if op == "+" and a + b == c:
                return True
            if op == "-" and a - b == c:
                return True
            if op == "*" and a * b == c:
                return True
            if op == "/" and b != 0 and a / b == c:
                return True
            return False
        if re.search(r"(subtract|add|divide|multiply|both sides|therefore|so|=)", step, re.IGNORECASE):
            return True
        return len(step) > 8


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ProofCompressionEnv(BatchEnvBase):
    """ProofCompression: compress a verbose proof to minimal steps.

    Batch-aware: N parallel compressions; reward = best (most compressed valid).
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
            problem_generator = proof_compression_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ProofCompressionVerifier(
            conclusion=problem.metadata["conclusion"],
            n_verbose_steps=problem.metadata["n_verbose_steps"],
            min_steps=problem.metadata["min_steps"],
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
        match = re.search(r"CONCLUSION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
