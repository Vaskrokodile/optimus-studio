"""
ContradictionDetection: Find contradictory statements in a set.

Environment concept:
  The model is given a set of logical statements. Exactly one pair is
  contradictory. The model must identify which two statements contradict
  each other and explain why.

  Why: contradiction detection is a core reasoning skill. It trains the
  model to carefully compare statements and identify logical conflicts —
  essential for self-consistency, fact-checking, and formal verification.

Problem types:
  - Direct contradiction: "All A are B" vs "Some A are not B"
  - Numeric contradiction: "X = 5" vs "X = 7"
  - Implication contradiction: "If A then B" vs "A and not B"
  - Quantifier contradiction: "All S are P" vs "No S are P"

Verification:
  - The identified pair matches the ground-truth contradictory pair.
  - The explanation mentions the type of contradiction.

Reward design:
  pair_correct * 0.6 + explanation_valid * 0.4
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — each entry: (statements, contradiction_indices, explanation_keywords)
# ---------------------------------------------------------------------------

_PROBLEMS = [
    {
        "statements": [
            "All birds can fly.",
            "Penguins are birds.",
            "Some birds cannot fly.",
            "Eagles are birds.",
        ],
        "contradiction": (0, 2),
        "keywords": ["quantifier", "all", "some", "cannot", "fly"],
    },
    {
        "statements": [
            "The temperature is 25 degrees.",
            "It is a warm day.",
            "The temperature is 10 degrees.",
            "The sun is shining.",
        ],
        "contradiction": (0, 2),
        "keywords": ["temperature", "25", "10", "different", "value", "number"],
    },
    {
        "statements": [
            "If it rains, the ground gets wet.",
            "It is raining.",
            "The ground is dry.",
            "The sky is cloudy.",
        ],
        "contradiction": (0, 2),
        "keywords": ["implication", "rain", "wet", "dry", "if", "then"],
    },
    {
        "statements": [
            "All squares have four sides.",
            "A square is a type of rectangle.",
            "No square has four sides.",
            "Rectangles have four sides.",
        ],
        "contradiction": (0, 2),
        "keywords": ["all", "no", "four", "sides", "quantifier"],
    },
    {
        "statements": [
            "X is greater than 10.",
            "X is a positive number.",
            "X is less than 5.",
            "X is an integer.",
        ],
        "contradiction": (0, 2),
        "keywords": ["greater", "less", "10", "5", "cannot", "both"],
    },
    {
        "statements": [
            "Every student passed the exam.",
            "There are 30 students.",
            "Some students failed the exam.",
            "The exam was difficult.",
        ],
        "contradiction": (0, 2),
        "keywords": ["every", "some", "passed", "failed", "all"],
    },
    {
        "statements": [
            "The car is red.",
            "The car is a vehicle.",
            "The car is blue.",
            "The car has four wheels.",
        ],
        "contradiction": (0, 2),
        "keywords": ["red", "blue", "color", "different", "cannot"],
    },
    {
        "statements": [
            "All mammals are warm-blooded.",
            "Dolphins are mammals.",
            "Some mammals are cold-blooded.",
            "Warm-blooded animals regulate their temperature.",
        ],
        "contradiction": (0, 2),
        "keywords": ["all", "some", "warm", "cold", "blooded"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def contradiction_detection_generator(seed: int) -> Problem:
    """Generate a ContradictionDetection problem."""
    rng = random.Random(seed)
    problem_data = rng.choice(_PROBLEMS)

    statements = problem_data["statements"]
    contra_idx = problem_data["contradiction"]
    keywords = problem_data["keywords"]

    # Shuffle statements but track the contradiction pair
    indices = list(range(len(statements)))
    rng.shuffle(indices)
    shuffled = [statements[i] for i in indices]
    new_contra = tuple(sorted([indices.index(contra_idx[0]), indices.index(contra_idx[1])]))

    # Build prompt with labeled statements
    labeled = "\n".join(f"  ({chr(65 + i)}) {s}" for i, s in enumerate(shuffled))
    prompt = (
        f"Find the two contradictory statements among the following:\n\n"
        f"{labeled}\n\n"
        f"Identify the contradictory pair and explain why.\n"
        f"Format: CONTRADICTION: <letter> and <letter>\n"
        f"EXPLANATION: <why they contradict>"
    )

    return Problem(
        id=f"contradiction_detection_{seed}",
        prompt=prompt,
        difficulty=0.4 + len(statements) * 0.05,
        metadata={
            "type": "contradiction",
            "statements": shuffled,
            "contradiction_indices": new_contra,
            "explanation_keywords": keywords,
            "n_statements": len(shuffled),
        },
        token_budget=384,
        source="contradiction_detection_generator",
    )


contradiction_detection_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ContradictionDetectionVerifier(Verifier):
    """Verify contradiction detection: pair match + explanation.

    reward = pair_correct * 0.6 + explanation_valid * 0.4
    """

    def __init__(
        self,
        contradiction_indices: tuple[int, int],
        explanation_keywords: list[str],
    ):
        super().__init__()
        self._contra_indices = set(contradiction_indices)
        self._keywords = [k.lower() for k in explanation_keywords]

    def verify(self, response: str) -> VerifierResult:
        # Parse contradiction pair: "CONTRADICTION: A and C" or "(A) and (C)"
        pair_match = re.search(
            r"CONTRADICTION\s*:\s*\(?([A-Z])\)?\s*(?:and|&|,)\s*\(?([A-Z])\)?",
            response,
            re.IGNORECASE,
        )

        pair_correct = 0.0
        if pair_match:
            idx_a = ord(pair_match.group(1).upper()) - ord("A")
            idx_b = ord(pair_match.group(2).upper()) - ord("A")
            given = {idx_a, idx_b}
            if given == self._contra_indices:
                pair_correct = 1.0
            elif len(given & self._contra_indices) == 1:
                pair_correct = 0.3  # one correct

        # Parse explanation
        expl_match = re.search(r"EXPLANATION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        explanation = expl_match.group(1).lower() if expl_match else response.lower()

        keyword_hits = sum(1 for kw in self._keywords if kw in explanation)
        explanation_valid = min(1.0, keyword_hits / max(len(self._keywords) // 2, 1))

        score = pair_correct * 0.6 + explanation_valid * 0.4
        correct = pair_correct >= 1.0 and explanation_valid >= 0.3

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "pair_correct": pair_correct,
                "explanation_valid": explanation_valid,
            },
            diagnostics=(
                f"pair={pair_correct:.2f} "
                f"explanation={explanation_valid:.2f} "
                f"keywords_hit={keyword_hits}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ContradictionDetectionEnv(BatchEnvBase):
    """ContradictionDetection: find the contradictory pair in a statement set.

    Batch-aware: N parallel attempts; reward = best attempt.
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
            problem_generator = contradiction_detection_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ContradictionDetectionVerifier(
            contradiction_indices=problem.metadata["contradiction_indices"],
            explanation_keywords=problem.metadata["explanation_keywords"],
        )

    def _check_format(self, response: str) -> float:
        has_pair = bool(re.search(r"CONTRADICTION\s*:", response, re.IGNORECASE))
        has_expl = bool(re.search(r"EXPLANATION\s*:", response, re.IGNORECASE))
        if has_pair and has_expl:
            return 1.0
        if has_pair:
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
        match = re.search(r"CONTRADICTION\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
