"""
SpeedRun: Solve correctly with a decaying token budget.

Environment concept:
  The model must solve a problem correctly AND concisely. The token budget
  decays with each step, so verbose reasoning is penalized. This explicitly
  trains token efficiency — the essence of intelligence density.

  Why: most RL environments reward correctness but ignore efficiency.
  SpeedRun makes token count a first-class reward signal, teaching the
  model to find the shortest correct path to an answer.

Problem types:
  - Arithmetic: "What is 15 * 17?" — answer in as few tokens as possible.
  - Logic: "If all A are B, and all B are C, are all A C?" — yes/no.
  - Sequence: "What comes next: 2, 4, 8, 16, ?" — answer = 32.

Verification:
  - Correctness: exact answer match.
  - Efficiency: reward scales with tokens_remaining / token_budget.

Reward design:
  correctness * (0.5 + 0.5 * efficiency)
  i.e. a correct answer using 10% of budget gets 0.95, using 90% gets 0.55.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank
# ---------------------------------------------------------------------------

_ARITH = [
    ("What is 15 * 17?", "255"),
    ("What is 12 * 13?", "156"),
    ("What is 9 * 16?", "144"),
    ("What is 7 * 19?", "133"),
    ("What is 11 * 14?", "154"),
    ("What is 8 * 21?", "168"),
    ("What is 13 * 15?", "195"),
    ("What is 6 * 24?", "144"),
    ("What is 18 * 5?", "90"),
    ("What is 22 * 7?", "154"),
]

_LOGIC = [
    ("If all cats are mammals, and all mammals are animals, are all cats animals?", "yes"),
    ("If all squares are rectangles, and all rectangles have 4 sides, do all squares have 4 sides?", "yes"),
    ("If some birds can fly, and penguins are birds, can all penguins fly?", "no"),
    ("If A > B and B > C, is A > C?", "yes"),
    ("If all primes > 2 are odd, and 17 is prime, is 17 odd?", "yes"),
    ("If no reptiles are mammals, and snakes are reptiles, are snakes mammals?", "no"),
]

_SEQUENCE = [
    ("What comes next: 2, 4, 8, 16, ?", "32"),
    ("What comes next: 1, 1, 2, 3, 5, 8, ?", "13"),
    ("What comes next: 3, 6, 12, 24, ?", "48"),
    ("What comes next: 1, 4, 9, 16, 25, ?", "36"),
    ("What comes next: 2, 6, 12, 20, 30, ?", "42"),
    ("What comes next: 1, 3, 6, 10, 15, ?", "21"),
]


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]", "", s)
    return s


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def speed_run_generator(seed: int) -> Problem:
    """Generate a SpeedRun problem: solve correctly and concisely."""
    rng = random.Random(seed)

    category = rng.choice(["arithmetic", "logic", "sequence"])
    if category == "arithmetic":
        question, answer = rng.choice(_ARITH)
    elif category == "logic":
        question, answer = rng.choice(_LOGIC)
    else:
        question, answer = rng.choice(_SEQUENCE)

    # Token budget varies — smaller budget = harder
    budget = rng.choice([128, 256, 384, 512])
    difficulty = 1.0 - (budget / 512.0)  # smaller budget = higher difficulty

    prompt = (
        f"Problem: {question}\n"
        f"Token budget: {budget} tokens (fewer is better)\n"
        f"Answer concisely. End with 'ANSWER: <answer>'"
    )

    return Problem(
        id=f"speed_run_{category}_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": category,
            "question": question,
            "answer": answer,
            "token_budget": budget,
        },
        token_budget=budget,
        source="speed_run_generator",
    )


speed_run_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SpeedRunVerifier(Verifier):
    """Verify a SpeedRun response: correctness + token efficiency.

    reward = correctness * (0.5 + 0.5 * efficiency)
    where efficiency = max(0, 1 - tokens_used / token_budget)
    """

    def __init__(self, answer: str, token_budget: int):
        super().__init__()
        self._answer = _normalize(answer)
        self._token_budget = token_budget

    def verify(self, response: str) -> VerifierResult:
        # Parse answer
        answer_match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if answer_match:
            given = _normalize(answer_match.group(1))
        else:
            # Take last non-empty line
            lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
            given = _normalize(lines[-1]) if lines else ""

        correct = given == self._answer
        correctness = 1.0 if correct else 0.0

        # Token efficiency
        tokens_used = estimate_tokens(response)
        if self._token_budget > 0:
            efficiency = max(0.0, 1.0 - tokens_used / self._token_budget)
        else:
            efficiency = 0.0

        score = correctness * (0.5 + 0.5 * efficiency)
        is_correct = correct and score > 0

        return VerifierResult(
            correct=is_correct,
            score=score,
            partial_credit={
                "correctness": correctness,
                "efficiency": efficiency,
                "tokens_used": float(tokens_used),
            },
            diagnostics=(
                f"correct={correctness:.2f} "
                f"efficiency={efficiency:.2f} "
                f"tokens={tokens_used}/{self._token_budget}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SpeedRunEnv(BatchEnvBase):
    """SpeedRun: solve correctly with a decaying token budget.

    Batch-aware: N parallel attempts; reward = best (most efficient correct).
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
            problem_generator = speed_run_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SpeedRunVerifier(
            answer=problem.metadata["answer"],
            token_budget=problem.metadata["token_budget"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"ANSWER\s*:", response, re.IGNORECASE):
            return 1.0
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
