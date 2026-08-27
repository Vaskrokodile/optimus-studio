"""
MinimumInformation: Identify the minimal missing information to solve a problem.

Environment concept:
  The model is given a problem with some information provided and some
  deliberately withheld. It must identify the MINIMAL set of missing
  information needed to solve the problem — no more, no less.

  Why: identifying what information is necessary vs. sufficient trains
  the model to understand problem structure. It's a meta-reasoning skill:
  knowing what you need to know is harder than knowing the answer.

Problem types:
  - Arithmetic: "If x + y = 10, what is x?" — missing: value of y (or x - y relation)
  - Geometry: "What is the area of a rectangle?" — missing: length and width
  - Logic: "Is the statement true?" — missing: definitions or premises

Verification:
  - Necessity: each identified item is actually needed (not redundant).
  - Sufficiency: the identified items together are enough to solve it.
  - Minimality: no unnecessary items identified.

Reward design:
  necessity * 0.4 + sufficiency * 0.4 + minimality * 0.2
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank
# Each entry: (problem_text, given_info, needed_info, distractor_info)
# needed_info: the minimal set of missing items required to solve
# distractor_info: plausible but unnecessary items
# ---------------------------------------------------------------------------

_PROBLEMS = [
    {
        "problem": "What is the area of a rectangle?",
        "given": ["The shape is a rectangle"],
        "needed": ["length of the rectangle", "width of the rectangle"],
        "distractors": ["color of the rectangle", "perimeter of the rectangle"],
    },
    {
        "problem": "If x + y = 10, what is the value of x?",
        "given": ["x + y = 10"],
        "needed": ["value of y"],
        "distractors": ["value of z", "the color of x"],
    },
    {
        "problem": "How long does it take to travel 200 km?",
        "given": ["The distance is 200 km"],
        "needed": ["speed of travel"],
        "distractors": ["color of the vehicle", "number of passengers"],
    },
    {
        "problem": "What is the perimeter of a square?",
        "given": ["The shape is a square"],
        "needed": ["side length of the square"],
        "distractors": ["area of the square", "diagonal of the square"],
    },
    {
        "problem": "What is the total cost of 5 items?",
        "given": ["There are 5 items"],
        "needed": ["price per item"],
        "distractors": ["weight of each item", "color of each item"],
    },
    {
        "problem": "What is the average of a set of numbers?",
        "given": ["There is a set of numbers"],
        "needed": ["the numbers in the set"],
        "distractors": ["the median of the set", "the range of the set"],
    },
    {
        "problem": "If 2x = 14, what is x + 3?",
        "given": ["2x = 14"],
        "needed": [],  # actually solvable with given info — trick question
        "distractors": ["value of x", "value of 3"],
    },
    {
        "problem": "What is the speed of a car?",
        "given": ["The car traveled 150 km in 3 hours"],
        "needed": [],  # solvable with given info
        "distractors": ["distance traveled", "time taken"],
    },
    {
        "problem": "How many days are in February?",
        "given": ["The month is February"],
        "needed": ["whether it is a leap year"],
        "distractors": ["the day of the week", "the temperature"],
    },
    {
        "problem": "What is the volume of a box?",
        "given": ["The shape is a rectangular box"],
        "needed": ["length of the box", "width of the box", "height of the box"],
        "distractors": ["color of the box", "weight of the box"],
    },
]


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def minimum_information_generator(seed: int) -> Problem:
    """Generate a MinimumInformation problem."""
    rng = random.Random(seed)
    problem_data = rng.choice(_PROBLEMS)

    problem = problem_data["problem"]
    given = problem_data["given"]
    needed = problem_data["needed"]
    distractors = problem_data["distractors"]

    # Present the problem and given info
    given_text = "\n".join(f"  - {g}" for g in given)
    prompt = (
        f"Problem: {problem}\n\n"
        f"Given information:\n{given_text}\n\n"
        f"What is the MINIMAL missing information needed to solve this problem?\n"
        f"List each item on its own line starting with 'MISSING:'\n"
        f"If the problem can be solved with the given information, "
        f"respond 'MISSING: none'"
    )

    is_trick = len(needed) == 0

    return Problem(
        id=f"minimum_information_{seed}",
        prompt=prompt,
        difficulty=0.3 + len(needed) * 0.1 + (0.2 if is_trick else 0),
        metadata={
            "type": "minimum_info",
            "problem": problem,
            "given": given,
            "needed": needed,
            "distractors": distractors,
            "is_trick": is_trick,
        },
        token_budget=256,
        source="minimum_information_generator",
    )


minimum_information_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class MinimumInformationVerifier(Verifier):
    """Verify minimal missing info: necessity + sufficiency + minimality.

    reward = necessity * 0.4 + sufficiency * 0.4 + minimality * 0.2
    """

    def __init__(
        self,
        needed: list[str],
        distractors: list[str],
        is_trick: bool,
    ):
        super().__init__()
        self._needed = [_normalize(n) for n in needed]
        self._distractors = [_normalize(d) for d in distractors]
        self._is_trick = is_trick

    def verify(self, response: str) -> VerifierResult:
        # Parse missing items
        missing_items = re.findall(r"MISSING\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        given_items = [_normalize(m.strip()) for m in missing_items if m.strip().lower() != "none"]

        if self._is_trick:
            # Trick question: answer should be "none"
            if not given_items or all(
                "none" in m for m in [_normalize(m) for m in missing_items]
            ):
                return VerifierResult(
                    correct=True,
                    score=1.0,
                    partial_credit={
                        "necessity": 1.0,
                        "sufficiency": 1.0,
                        "minimality": 1.0,
                    },
                    diagnostics="Correctly identified no missing info needed (trick question)",
                )
            # Identified unnecessary items
            return VerifierResult(
                correct=False,
                score=0.2,
                partial_credit={
                    "necessity": 0.0,
                    "sufficiency": 1.0,
                    "minimality": 0.0,
                },
                diagnostics=f"Problem is solvable with given info, but listed {len(given_items)} items",
            )

        # Non-trick: check necessity, sufficiency, minimality
        given_normalized = given_items

        # Necessity: how many given items match needed items?
        needed_matched = 0
        for g in given_normalized:
            for n in self._needed:
                if g == n or n in g or g in n:
                    needed_matched += 1
                    break
        necessity = needed_matched / max(len(given_normalized), 1) if given_normalized else 0.0

        # Sufficiency: are all needed items covered?
        needed_covered = 0
        for n in self._needed:
            for g in given_normalized:
                if g == n or n in g or g in n:
                    needed_covered += 1
                    break
        sufficiency = needed_covered / max(len(self._needed), 1) if self._needed else 1.0

        # Minimality: penalize distractor matches
        distractor_hits = 0
        for g in given_normalized:
            for d in self._distractors:
                if g == d or d in g or g in d:
                    distractor_hits += 1
                    break
        minimality = 1.0 - min(1.0, distractor_hits / max(len(given_normalized), 1)) if given_normalized else 0.0

        score = necessity * 0.4 + sufficiency * 0.4 + minimality * 0.2
        correct = sufficiency >= 1.0 and minimality >= 0.8 and necessity >= 0.8

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "necessity": necessity,
                "sufficiency": sufficiency,
                "minimality": minimality,
            },
            diagnostics=(
                f"necessity={necessity:.2f} "
                f"sufficiency={sufficiency:.2f} "
                f"minimality={minimality:.2f} "
                f"items={len(given_normalized)}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MinimumInformationEnv(BatchEnvBase):
    """MinimumInformation: identify minimal missing info to solve a problem.

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
            problem_generator = minimum_information_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return MinimumInformationVerifier(
            needed=problem.metadata["needed"],
            distractors=problem.metadata["distractors"],
            is_trick=problem.metadata["is_trick"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"MISSING\s*:", response, re.IGNORECASE):
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
        matches = re.findall(r"MISSING\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return "; ".join(m.strip() for m in matches) if matches else response
