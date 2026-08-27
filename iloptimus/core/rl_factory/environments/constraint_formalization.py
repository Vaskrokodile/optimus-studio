"""
ConstraintFormalization: Natural language -> constraint solver DSL.

Environment concept:
  The model is given a word problem describing relationships between entities
  and must formalize it as a set of constraints. The verifier checks that the
  formalized constraints match the ground-truth constraints.

  This trains the model to:
    1. Parse natural-language relationships into formal constraints
    2. Express ordering, equality, and inequality relationships
    3. Cover all the relationships stated in the problem

Verification:
  - Each CONSTRAINT line is parsed and normalized.
  - The set of formalized constraints is compared to the ground-truth set.
  - Partial credit for each ground-truth constraint that is matched.

Reward design:
  coverage * 0.7 + format_score * 0.3
  where coverage = matched_constraints / total_ground_truth_constraints
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
# ---------------------------------------------------------------------------


_PROBLEMS = [
    {
        "problem": (
            "Alice is taller than Bob. Bob is taller than Carol. "
            "Carol is taller than Dave. Who is the tallest?"
        ),
        "constraints": [
            "height(Alice) > height(Bob)",
            "height(Bob) > height(Carol)",
            "height(Carol) > height(Dave)",
        ],
        "solution": "Alice",
        "entities": ["Alice", "Bob", "Carol", "Dave"],
    },
    {
        "problem": (
            "Alice is older than Bob. Bob is older than Carol. "
            "Carol is older than Dave. Who is the oldest?"
        ),
        "constraints": [
            "age(Alice) > age(Bob)",
            "age(Bob) > age(Carol)",
            "age(Carol) > age(Dave)",
        ],
        "solution": "Alice",
        "entities": ["Alice", "Bob", "Carol", "Dave"],
    },
    {
        "problem": (
            "The red car is faster than the blue car. The blue car is faster "
            "than the green car. The green car is faster than the yellow car. "
            "Which car is the fastest?"
        ),
        "constraints": [
            "speed(red) > speed(blue)",
            "speed(blue) > speed(green)",
            "speed(green) > speed(yellow)",
        ],
        "solution": "red",
        "entities": ["red", "blue", "green", "yellow"],
    },
    {
        "problem": (
            "Alice earns more than Bob. Bob earns more than Carol. "
            "Carol earns the same as Dave. Who earns the most?"
        ),
        "constraints": [
            "salary(Alice) > salary(Bob)",
            "salary(Bob) > salary(Carol)",
            "salary(Carol) = salary(Dave)",
        ],
        "solution": "Alice",
        "entities": ["Alice", "Bob", "Carol", "Dave"],
    },
    {
        "problem": (
            "The apple costs more than the banana. The banana costs more than "
            "the cherry. The cherry costs less than the date. Which fruit "
            "costs the most?"
        ),
        "constraints": [
            "price(apple) > price(banana)",
            "price(banana) > price(cherry)",
            "price(cherry) < price(date)",
        ],
        "solution": "apple",
        "entities": ["apple", "banana", "cherry", "date"],
    },
    {
        "problem": (
            "Alice is taller than Bob. Carol is taller than Alice. "
            "Bob is taller than Dave. Who is the tallest?"
        ),
        "constraints": [
            "height(Alice) > height(Bob)",
            "height(Carol) > height(Alice)",
            "height(Bob) > height(Dave)",
        ],
        "solution": "Carol",
        "entities": ["Alice", "Bob", "Carol", "Dave"],
    },
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_constraint(s: str) -> str:
    """Normalize a constraint string for comparison."""
    s = s.strip().lower()
    # Remove spaces around operators and parentheses
    s = re.sub(r"\s*([><=])\s*", r"\1", s)
    # Collapse remaining whitespace
    s = re.sub(r"\s+", " ", s)
    # Normalize operator variants
    s = s.replace(">=", ">=").replace("<=", "<=")
    return s.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def constraint_formalization_generator(seed: int) -> Problem:
    """Generate a constraint formalization problem."""
    rng = random.Random(seed)
    data = rng.choice(_PROBLEMS)
    problem_text = data["problem"]
    constraints = data["constraints"]
    solution = data["solution"]

    prompt = (
        f"Formalize the following word problem as a set of constraints.\n\n"
        f"Problem: {problem_text}\n\n"
        f"Write each constraint on its own line starting with 'CONSTRAINT:'.\n"
        f"Use the form: attribute(entity) > attribute(entity), "
        f"attribute(entity) = attribute(entity), etc.\n"
        f"End with 'ANSWER: <answer to the question>'."
    )

    return Problem(
        id=f"constraint_formalization_{seed}",
        prompt=prompt,
        difficulty=0.3 + len(constraints) * 0.1,
        metadata={
            "problem": problem_text,
            "constraints": constraints,
            "solution": solution,
            "entities": data["entities"],
        },
        token_budget=512,
        source="constraint_formalization_generator",
    )


constraint_formalization_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ConstraintFormalizationVerifier(Verifier):
    """Verify formalized constraints against ground truth.

    reward = coverage * 0.7 + format_score * 0.3
    """

    def __init__(self, constraints: list[str], solution: str):
        super().__init__()
        self._gt_constraints = [_normalize_constraint(c) for c in constraints]
        self._solution = solution.strip().lower()
        self._n_gt = len(self._gt_constraints)

    def verify(self, response: str) -> VerifierResult:
        # Parse CONSTRAINT lines
        constraint_lines = re.findall(
            r"CONSTRAINT\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        given_constraints = [_normalize_constraint(c) for c in constraint_lines if c.strip()]

        # Parse ANSWER line
        ans_match = re.search(
            r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        given_answer = ans_match.group(1).strip().lower() if ans_match else ""

        # Format score
        format_score = 0.0
        if constraint_lines:
            format_score += 0.5
        if given_answer:
            format_score += 0.5

        # Coverage: how many ground-truth constraints are matched?
        matched = 0
        for gt in self._gt_constraints:
            if gt in given_constraints:
                matched += 1
            else:
                # Try reversed direction (a < b matches b > a)
                rev = self._reverse_constraint(gt)
                if rev in given_constraints:
                    matched += 1
        coverage = matched / self._n_gt if self._n_gt > 0 else 0.0

        # Answer correctness
        answer_correct = 1.0 if given_answer == self._solution else 0.0

        score = coverage * 0.7 + format_score * 0.3
        # Boost if answer is also correct
        score = min(1.0, score + answer_correct * 0.1)
        correct = coverage >= 1.0 and answer_correct >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "coverage": coverage,
                "format_score": format_score,
                "answer_correct": answer_correct,
                "matched": float(matched),
                "total": float(self._n_gt),
            },
            diagnostics=(
                f"coverage={coverage:.2f} ({matched}/{self._n_gt}) "
                f"format={format_score:.2f} "
                f"answer={answer_correct:.2f} "
                f"given_answer={given_answer!r} expected={self._solution!r}"
            ),
        )

    @staticmethod
    def _reverse_constraint(c: str) -> str:
        """Reverse the direction of a comparison constraint.

        e.g. 'a>b' -> 'b<a', 'a<b' -> 'b>a'.
        """
        for op, rev in [(">", "<"), ("<", ">")]:
            if op in c:
                parts = c.split(op, 1)
                if len(parts) == 2:
                    return f"{parts[1].strip()}{rev}{parts[0].strip()}"
        return c


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ConstraintFormalizationEnv(BatchEnvBase):
    """ConstraintFormalization: natural language -> constraints.

    Batch-aware: N parallel formalizations; reward = best score across batch.
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
            problem_generator = constraint_formalization_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ConstraintFormalizationVerifier(
            constraints=problem.metadata["constraints"],
            solution=problem.metadata["solution"],
        )

    def _check_format(self, response: str) -> float:
        has_constraint = bool(re.search(r"CONSTRAINT\s*:", response, re.IGNORECASE))
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        if has_constraint and has_answer:
            return 1.0
        if has_constraint or has_answer:
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
