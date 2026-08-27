"""
AnswerFirst: Given the answer, generate backward reasoning.

Environment concept:
  The model is given a problem AND its final answer, then must produce
  step-by-step reasoning that *arrives at* that answer. This trains
  backward reasoning (conclusions -> premises) — the inverse of the
  standard forward-reasoning setup.

  Why: backward reasoning forces the model to justify *why* an answer is
  correct rather than mechanically deriving it. It exposes gaps in
  understanding that forward-only training misses.

Problem types:
  - Arithmetic: "What is 17 * 23? Answer: 391" — show the steps.
  - Algebra: "Solve x^2 - 5x + 6 = 0. Answer: x = 2 or x = 3"
  - Logic: "Is the statement true? Answer: Yes" — justify.

Verification:
  - Each step is checked for arithmetic/logical validity (rule-based).
  - The final step must reach the given answer.
  - Steps must form a coherent chain (no non-sequiturs).

Reward design:
  endpoint_match * 0.5 + steps_valid * 0.3 + chain_coherence * 0.2
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — each entry: (prompt_template, answer, valid_steps)
# valid_steps is a list of step strings the model should produce.
# ---------------------------------------------------------------------------

_ARITH_PROBLEMS = [
    {
        "question": "What is 12 * 15?",
        "answer": "180",
        "steps": [
            "12 * 15 = 12 * (10 + 5)",
            "12 * 10 = 120",
            "12 * 5 = 60",
            "120 + 60 = 180",
        ],
    },
    {
        "question": "What is 7 * 13?",
        "answer": "91",
        "steps": [
            "7 * 13 = 7 * (10 + 3)",
            "7 * 10 = 70",
            "7 * 3 = 21",
            "70 + 21 = 91",
        ],
    },
    {
        "question": "What is 25 * 8?",
        "answer": "200",
        "steps": [
            "25 * 8 = 25 * (4 * 2)",
            "25 * 4 = 100",
            "100 * 2 = 200",
        ],
    },
    {
        "question": "What is 144 / 12?",
        "answer": "12",
        "steps": [
            "144 / 12: 12 * 10 = 120",
            "144 - 120 = 24",
            "12 * 2 = 24",
            "So 144 / 12 = 10 + 2 = 12",
        ],
    },
    {
        "question": "What is 17 + 26?",
        "answer": "43",
        "steps": [
            "17 + 26: 17 + 20 = 37",
            "37 + 6 = 43",
        ],
    },
    {
        "question": "What is 9 * 16?",
        "answer": "144",
        "steps": [
            "9 * 16 = 9 * (10 + 6)",
            "9 * 10 = 90",
            "9 * 6 = 54",
            "90 + 54 = 144",
        ],
    },
    {
        "question": "What is 13 * 14?",
        "answer": "182",
        "steps": [
            "13 * 14 = 13 * (10 + 4)",
            "13 * 10 = 130",
            "13 * 4 = 52",
            "130 + 52 = 182",
        ],
    },
    {
        "question": "What is 100 - 37?",
        "answer": "63",
        "steps": [
            "100 - 37: 100 - 30 = 70",
            "70 - 7 = 63",
        ],
    },
]

_ALGEBRA_PROBLEMS = [
    {
        "question": "Solve x + 7 = 15 for x.",
        "answer": "x = 8",
        "steps": [
            "x + 7 = 15",
            "Subtract 7 from both sides: x = 15 - 7",
            "x = 8",
        ],
    },
    {
        "question": "Solve 2x = 18 for x.",
        "answer": "x = 9",
        "steps": [
            "2x = 18",
            "Divide both sides by 2: x = 18 / 2",
            "x = 9",
        ],
    },
    {
        "question": "Solve x - 5 = 11 for x.",
        "answer": "x = 16",
        "steps": [
            "x - 5 = 11",
            "Add 5 to both sides: x = 11 + 5",
            "x = 16",
        ],
    },
    {
        "question": "Solve 3x = 21 for x.",
        "answer": "x = 7",
        "steps": [
            "3x = 21",
            "Divide both sides by 3: x = 21 / 3",
            "x = 7",
        ],
    },
    {
        "question": "Solve x^2 = 49 for x (positive root).",
        "answer": "x = 7",
        "steps": [
            "x^2 = 49",
            "Take square root: x = sqrt(49)",
            "x = 7",
        ],
    },
]


def _normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z=.\-+*/x\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def answer_first_generator(seed: int) -> Problem:
    """Generate an AnswerFirst problem: given the answer, produce reasoning."""
    rng = random.Random(seed)

    pool = _ARITH_PROBLEMS + _ALGEBRA_PROBLEMS
    problem_data = rng.choice(pool)
    ptype = "arithmetic" if problem_data in _ARITH_PROBLEMS else "algebra"

    question = problem_data["question"]
    answer = problem_data["answer"]
    steps = problem_data["steps"]

    prompt = (
        f"Problem: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide step-by-step reasoning that arrives at this answer.\n"
        f"Format each step on its own line starting with 'STEP:'\n"
        f"End with 'ANSWER: {answer}'"
    )

    difficulty = 0.3 + len(steps) * 0.08
    difficulty = min(0.9, difficulty)

    return Problem(
        id=f"answer_first_{ptype}_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": ptype,
            "question": question,
            "answer": answer,
            "expected_steps": steps,
            "n_expected_steps": len(steps),
        },
        token_budget=512,
        source="answer_first_generator",
    )


answer_first_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class AnswerFirstVerifier(Verifier):
    """Verify backward reasoning: steps valid + endpoint matches answer.

    reward = endpoint_match * 0.5 + steps_valid * 0.3 + chain_coherence * 0.2
    """

    def __init__(self, answer: str, expected_steps: list[str]):
        super().__init__()
        self._answer = _normalize_answer(answer)
        self._expected_steps = expected_steps
        self._expected_numbers = [
            tuple(sorted(_extract_numbers(s))) for s in expected_steps
        ]

    def verify(self, response: str) -> VerifierResult:
        # Parse steps
        step_lines = re.findall(r"STEP\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if not step_lines:
            # Also accept numbered steps: "1. ..." or "Step 1: ..."
            step_lines = re.findall(r"(?:^\d+\.?\s*|Step\s+\d+\s*:\s*)(.+?)(?:\n|$)", response, re.IGNORECASE | re.MULTILINE)

        # Parse final answer
        answer_match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if answer_match:
            given_answer = _normalize_answer(answer_match.group(1))
        else:
            given_answer = _normalize_answer(response.split("\n")[-1])

        # 1. Endpoint match
        endpoint_match = 1.0 if given_answer == self._answer else 0.0
        # Partial: if the answer number appears anywhere
        if endpoint_match == 0.0:
            ans_nums = _extract_numbers(self._answer)
            given_nums = _extract_numbers(given_answer)
            if ans_nums and given_nums and ans_nums[-1] == given_nums[-1]:
                endpoint_match = 0.5

        # 2. Steps valid: check that each step contains valid arithmetic
        if step_lines:
            valid_count = 0
            for sl in step_lines:
                if self._step_is_valid(sl):
                    valid_count += 1
            steps_valid = valid_count / len(step_lines)
        else:
            steps_valid = 0.0

        # 3. Chain coherence: do the steps reference numbers from expected steps?
        if step_lines and self._expected_steps:
            response_numbers = [tuple(sorted(_extract_numbers(s))) for s in step_lines]
            # Check overlap with expected step numbers
            matches = 0
            for rn in response_numbers:
                for en in self._expected_numbers:
                    if rn and en and rn == en:
                        matches += 1
                        break
            chain_coherence = matches / max(len(self._expected_numbers), 1)
        else:
            chain_coherence = 0.0

        score = endpoint_match * 0.5 + steps_valid * 0.3 + chain_coherence * 0.2
        correct = endpoint_match >= 0.5 and steps_valid >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "endpoint_match": endpoint_match,
                "steps_valid": steps_valid,
                "chain_coherence": chain_coherence,
            },
            diagnostics=(
                f"endpoint={endpoint_match:.2f} "
                f"steps_valid={steps_valid:.2f} ({len(step_lines)} steps) "
                f"coherence={chain_coherence:.2f}"
            ),
        )

    def _step_is_valid(self, step: str) -> bool:
        """Check if a step contains a valid arithmetic statement."""
        step = step.strip()
        if not step:
            return False
        # Check for arithmetic expressions like "a + b = c" or "a * b = c"
        expr_match = re.search(r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", step)
        if expr_match:
            a, op, b, c = float(expr_match.group(1)), expr_match.group(2), float(expr_match.group(3)), float(expr_match.group(4))
            if op == "+" and a + b == c:
                return True
            if op == "-" and a - b == c:
                return True
            if op == "*" and a * b == c:
                return True
            if op == "/" and b != 0 and a / b == c:
                return True
            return False
        # Non-arithmetic steps (e.g. "Subtract 7 from both sides") are valid
        # if they describe an operation or contain an equation
        if re.search(r"(subtract|add|divide|multiply|factor|expand|simplify|take|both sides)", step, re.IGNORECASE):
            return True
        if re.search(r"=\s*-?\d+\.?\d*", step):
            return True
        return len(step) > 10  # accept any substantive step


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AnswerFirstEnv(BatchEnvBase):
    """AnswerFirst: given the answer, produce backward reasoning.

    Batch-aware: N parallel reasoning attempts; reward = best attempt.
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
            problem_generator = answer_first_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return AnswerFirstVerifier(
            answer=problem.metadata["answer"],
            expected_steps=problem.metadata["expected_steps"],
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
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
