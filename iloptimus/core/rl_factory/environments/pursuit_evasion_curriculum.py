"""
PursuitEvasionCurriculum: Adversarial difficulty positioning at ~50% capture rate.

Environment concept:
  This implements the *pursuit-evasion curriculum* paradigm: the
  environment generates problems at a difficulty level and ADAPTIVELY
  adjusts the level to maintain approximately a 50% success rate. This
  keeps the model at the frontier of its capability — neither too easy
  (boring, no learning) nor too hard (frustrating, no signal).

  The generator produces problems at a specified difficulty level (1-5).
  The environment tracks the success rate per difficulty level and
  exposes an `adjust_difficulty()` method that increases difficulty if
  the success rate > 0.6 and decreases it if < 0.4.

  Problem types: arithmetic with increasing operand sizes / operation
  complexity keyed to the difficulty level.

Verification:
  - The answer is checked for exact correctness at the given difficulty.

Reward:
  reward = 1.0 if correct, 0.0 otherwise (with partial credit for
  near-misses on harder levels).

Format:
  ANSWER: <answer>
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Difficulty-keyed problem generation
# ---------------------------------------------------------------------------

# Level 1: single-digit + / -
# Level 2: two-digit + / -
# Level 3: two-digit *
# Level 4: three-digit * / mixed
# Level 5: exponent / multi-step


def _generate_problem_at_level(level: int, rng: random.Random) -> dict[str, Any]:
    """Generate a problem at the given difficulty level (1-5)."""
    level = max(1, min(5, level))
    if level == 1:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        answer = a + b if op == "+" else a - b
        problem = f"What is {a} {op} {b}?"
    elif level == 2:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        answer = a + b if op == "+" else a - b
        problem = f"What is {a} {op} {b}?"
    elif level == 3:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        answer = a * b
        problem = f"What is {a} * {b}?"
    elif level == 4:
        a, b = rng.randint(10, 99), rng.randint(2, 12)
        op = rng.choice(["*", "+", "-"])
        if op == "*":
            answer = a * b
            problem = f"What is {a} * {b}?"
        elif op == "+":
            answer = a + b
            problem = f"What is {a} + {b}?"
        else:
            if b > a:
                a, b = b, a
            answer = a - b
            problem = f"What is {a} - {b}?"
    else:  # level 5
        base = rng.randint(2, 5)
        exp = rng.randint(2, 4)
        answer = base ** exp
        problem = f"What is {base}^{exp}?"
    return {
        "problem": problem,
        "answer": str(answer),
        "answer_int": answer,
        "difficulty_level": level,
        "problem_type": f"level_{level}",
    }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def pursuit_evasion_curriculum_generator(
    seed: int, difficulty_level: int = 3
) -> Problem:
    """Generate a PursuitEvasionCurriculum problem at a given difficulty.

    Args:
        seed: Random seed for reproducible problem generation.
        difficulty_level: Difficulty level 1-5 (default 3).

    Returns:
        A Problem with the answer and difficulty level.
    """
    rng = random.Random(seed)
    data = _generate_problem_at_level(difficulty_level, rng)

    prompt = (
        f"{data['problem']}\n\n"
        f"Format: ANSWER: <answer>"
    )

    return Problem(
        id=f"pursuit_evade_{seed}_lv{difficulty_level}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=difficulty_level / 5.0,
        metadata={
            "type": "pursuit_evasion_curriculum",
            "problem": data["problem"],
            "answer": data["answer"],
            "answer_int": data["answer_int"],
            "difficulty_level": difficulty_level,
            "problem_type": data["problem_type"],
        },
        token_budget=200,
        source="pursuit_evasion_curriculum_generator",
    )


pursuit_evasion_curriculum_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PursuitEvasionVerifier(Verifier):
    """Verify a pursuit-evasion answer at a given difficulty level.

    Args:
        answer_int: The correct numeric answer.
        difficulty_level: The difficulty level (for partial credit).
    """

    def __init__(self, answer_int: int, difficulty_level: int):
        super().__init__()
        self._answer_int = answer_int
        self._difficulty_level = difficulty_level

    def verify(self, response: str) -> VerifierResult:
        given = self._parse_answer(response)
        if given is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No number found in ANSWER field",
            )

        exact = given == self._answer_int
        if exact:
            return VerifierResult(
                correct=True,
                score=1.0,
                diagnostics=f"Exact match: {given}",
            )

        # Partial credit for near-miss on harder levels
        diff = abs(given - self._answer_int)
        if self._answer_int != 0:
            rel = diff / abs(self._answer_int)
        else:
            rel = 1.0 if diff != 0 else 0.0
        partial = max(0.0, 1.0 - rel)
        partial = min(partial, 0.4)

        return VerifierResult(
            correct=False,
            score=partial,
            partial_credit={"relative_error": rel},
            diagnostics=f"Expected={self._answer_int} Got={given} rel_err={rel:.3f}",
        )

    def _parse_answer(self, response: str) -> Optional[int]:
        match = re.search(r"ANSWER\s*:\s*(-?\d+)", response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        nums = re.findall(r"-?\d+", response)
        return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PursuitEvasionCurriculumEnv(BatchEnvBase):
    """PursuitEvasionCurriculum environment: adaptive difficulty at ~50%.

    Tracks success rate per difficulty level and exposes
    `adjust_difficulty()` to adapt the level. The current difficulty
    level is stored in `self._current_difficulty`.

    Batch-aware: N parallel attempts; reward = best attempt.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
        initial_difficulty: int = 3,
    ):
        if problems is None and problem_generator is None:
            # Use a closure that passes the current difficulty level
            self._current_difficulty = initial_difficulty
            problem_generator = self._make_level_generator()
        else:
            self._current_difficulty = initial_difficulty
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )
        # Success tracking per difficulty level
        self._success_history: dict[int, list[bool]] = {}

    def _make_level_generator(self):
        """Create a generator closure that uses the current difficulty."""
        current_difficulty = self._current_difficulty

        def _gen(seed: int) -> Problem:
            return pursuit_evasion_curriculum_generator(
                seed, difficulty_level=current_difficulty
            )

        _gen.__test__ = False  # type: ignore[attr-defined]
        return _gen

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return PursuitEvasionVerifier(
            answer_int=meta["answer_int"],
            difficulty_level=meta["difficulty_level"],
        )

    def _check_format(self, response: str) -> float:
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        return 1.0 if has_answer else 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Record success for the current difficulty level
        level = self._current_problem.metadata.get("difficulty_level", 3)
        correct = info.get("correct", False)
        self._success_history.setdefault(level, []).append(correct)
        return obs, reward, terminated, truncated, info

    def success_rate(self, level: Optional[int] = None) -> float:
        """Return the success rate for a difficulty level.

        Args:
            level: Difficulty level. If None, uses the current level.

        Returns:
            Success rate in [0, 1]. Returns 0.5 if no history.
        """
        if level is None:
            level = self._current_difficulty
        history = self._success_history.get(level, [])
        if not history:
            return 0.5
        return sum(history) / len(history)

    def adjust_difficulty(self) -> int:
        """Adjust the difficulty level based on recent success rate.

        Increases difficulty if success rate > 0.6, decreases if < 0.4.
        Returns the new difficulty level.
        """
        rate = self.success_rate()
        if rate > 0.6:
            self._current_difficulty = min(5, self._current_difficulty + 1)
        elif rate < 0.4:
            self._current_difficulty = max(1, self._current_difficulty - 1)
        # Update the generator to use the new level
        self._problem_generator = self._make_level_generator()
        return self._current_difficulty

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "mean_score": mean_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "reward": best_score,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)} "
                f"Level={self._current_difficulty}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
