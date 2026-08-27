"""
IncrementalComplexity: Solve with increasing complexity, no forgetting.

Environment concept:
  The model is given a sequence of problems at increasing complexity levels.
  It must solve the current level AND all previous levels correctly. This
  trains progressive skill building with retention — the model cannot just
  solve the hardest problem, it must maintain competence at all levels.

  Why: models often catastrophically forget simpler skills when trained on
  harder problems. IncrementalComplexity explicitly tests for retention
  alongside progression, ensuring robust capability across difficulty levels.

Problem types:
  - Arithmetic sequences: add 1-digit, 2-digit, 3-digit, 4-digit numbers
  - Multiplication tables: 2x, 3x, ..., up to Nx
  - Pattern extension: simple → complex sequences

Verification:
  - Current level answer correct.
  - All previous level answers correct.
  - Partial credit for getting some levels right.

Reward design:
  (n_correct_levels / n_total_levels) * 0.7 + current_correct * 0.3
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------


def _gen_arithmetic_levels(rng: random.Random, n_levels: int) -> list[tuple[str, str]]:
    """Generate arithmetic problems at increasing complexity."""
    levels = []
    for level in range(1, n_levels + 1):
        if level == 1:
            a, b = rng.randint(1, 9), rng.randint(1, 9)
        elif level == 2:
            a, b = rng.randint(10, 99), rng.randint(1, 9)
        elif level == 3:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        elif level == 4:
            a, b = rng.randint(100, 999), rng.randint(10, 99)
        else:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
        op = rng.choice(["+", "-"] if level <= 2 else ["+", "-", "*"])
        if op == "+":
            ans = str(a + b)
        elif op == "-":
            if a < b:
                a, b = b, a
            ans = str(a - b)
        else:
            ans = str(a * b)
        levels.append((f"What is {a} {op} {b}?", ans))
    return levels


def _gen_sequence_levels(rng: random.Random, n_levels: int) -> list[tuple[str, str]]:
    """Generate sequence-extension problems at increasing complexity."""
    levels = []
    for level in range(1, n_levels + 1):
        start = rng.randint(1, 5)
        step = rng.randint(1, 4)
        seq_len = 3 + level  # longer sequences at higher levels
        seq = [start + i * step for i in range(seq_len)]
        next_val = start + seq_len * step
        levels.append((f"What comes next: {', '.join(map(str, seq))}, ?", str(next_val)))
    return levels


def _gen_multiplication_levels(rng: random.Random, n_levels: int) -> list[tuple[str, str]]:
    """Generate multiplication problems at increasing complexity."""
    levels = []
    for level in range(1, n_levels + 1):
        max_val = min(12, 2 + level * 2)
        a = rng.randint(2, max_val)
        b = rng.randint(2, max_val)
        levels.append((f"What is {a} * {b}?", str(a * b)))
    return levels


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z=.\-+*/x\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def incremental_complexity_generator(seed: int) -> Problem:
    """Generate an IncrementalComplexity problem set."""
    rng = random.Random(seed)

    category = rng.choice(["arithmetic", "sequence", "multiplication"])
    n_levels = rng.randint(3, 5)

    if category == "arithmetic":
        levels = _gen_arithmetic_levels(rng, n_levels)
    elif category == "sequence":
        levels = _gen_sequence_levels(rng, n_levels)
    else:
        levels = _gen_multiplication_levels(rng, n_levels)

    current_level = n_levels  # must solve all levels up to current

    # Build prompt: show all levels, ask for answers to all
    level_text = "\n".join(
        f"  Level {i+1}: {q}" for i, (q, _) in enumerate(levels)
    )
    prompt = (
        f"Solve ALL {n_levels} levels below. Each level increases in complexity.\n"
        f"You must get ALL levels correct (including easier earlier ones).\n\n"
        f"{level_text}\n\n"
        f"Format each answer as: LEVEL <n>: <answer>"
    )

    return Problem(
        id=f"incremental_complexity_{category}_{seed}",
        prompt=prompt,
        difficulty=0.2 + n_levels * 0.15,
        metadata={
            "type": category,
            "levels": levels,
            "n_levels": n_levels,
            "current_level": current_level,
            "answers": [ans for _, ans in levels],
        },
        token_budget=512,
        source="incremental_complexity_generator",
    )


incremental_complexity_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class IncrementalComplexityVerifier(Verifier):
    """Verify incremental complexity: all levels correct + current level.

    reward = (n_correct / n_total) * 0.7 + current_correct * 0.3
    """

    def __init__(self, answers: list[str], n_levels: int):
        super().__init__()
        self._answers = [_normalize(a) for a in answers]
        self._n_levels = n_levels

    def verify(self, response: str) -> VerifierResult:
        # Parse level answers: "LEVEL 1: 42" or "Level 1: 42"
        level_answers: dict[int, str] = {}
        for m in re.finditer(r"LEVEL\s+(\d+)\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE):
            idx = int(m.group(1))
            level_answers[idx] = _normalize(m.group(2))

        # Check each level
        n_correct = 0
        per_level = []
        for i, expected in enumerate(self._answers):
            given = level_answers.get(i + 1, "")
            is_correct = given == expected
            if is_correct:
                n_correct += 1
            per_level.append({"level": i + 1, "correct": is_correct})

        current_correct = per_level[-1]["correct"] if per_level else False
        ratio = n_correct / self._n_levels if self._n_levels > 0 else 0.0

        score = ratio * 0.7 + (1.0 if current_correct else 0.0) * 0.3
        correct = n_correct == self._n_levels

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "levels_correct_ratio": ratio,
                "current_correct": 1.0 if current_correct else 0.0,
                "n_correct": float(n_correct),
            },
            diagnostics=(
                f"correct={n_correct}/{self._n_levels} "
                f"current={'yes' if current_correct else 'no'} "
                f"score={score:.2f}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class IncrementalComplexityEnv(BatchEnvBase):
    """IncrementalComplexity: solve with increasing complexity, no forgetting.

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
            problem_generator = incremental_complexity_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return IncrementalComplexityVerifier(
            answers=problem.metadata["answers"],
            n_levels=problem.metadata["n_levels"],
        )

    def _check_format(self, response: str) -> float:
        n_level_lines = len(re.findall(r"LEVEL\s+\d+\s*:", response, re.IGNORECASE))
        if n_level_lines == 0:
            return 0.0
        return min(1.0, n_level_lines / self._current_problem.metadata["n_levels"])

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
        matches = re.findall(r"LEVEL\s+\d+\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return "; ".join(m.strip() for m in matches) if matches else response
