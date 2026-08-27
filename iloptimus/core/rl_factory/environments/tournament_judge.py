"""
TournamentJudge: Generate N solutions, pairwise compare, tournament ranking.

Environment concept:
  The model generates N solutions to a problem. The environment then runs a
  round-robin tournament: every pair of solutions is compared, and the
  solution with the most wins is the champion.

  Comparison is done by the verifier (not a neural judge) — each solution
  is scored independently, and the higher-scored solution wins the pairwise
  match. Ties are broken by conciseness (shorter = better).

  The reward is:
    reward = champion_score * 0.5 + tournament_consistency * 0.3 + diversity * 0.2

  Where:
    - champion_score = verifier score of the tournament champion
    - tournament_consistency = how decisively the champion won
      (champion_wins / (N-1))
    - diversity = fraction of distinct solutions (same as BestOfN)

  This trains the model to:
    1. Generate solutions that DOMINATE other solutions (not just pass)
    2. Generate diverse candidates (so the tournament has variety)
    3. Produce solutions that are robust across different comparison criteria

  The key insight: tournament selection is how evolutionary algorithms work.
  By training the model to produce "tournament-dominant" solutions, we're
  training it to produce solutions that are robustly good — not just barely
  correct, but clearly better than alternatives.

Why this environment is worth using for 10T-100T param models:
  Tournament selection rewards DOMINANCE, not just correctness. A 100T model
  that produces solutions which consistently win against alternatives is
  more reliable than one that produces barely-correct solutions. This is
  the difference between "works" and "works well" — critical for production
  agentic systems.

Problem types:
  - Code problems (execution-based comparison)
  - Math problems (exact match + conciseness tiebreaker)
  - Refactoring problems (correctness + brevity)

Reward design:
  - champion_score: 0.5 weight (the champion must be correct)
  - tournament_consistency: 0.3 weight (decisive wins are better)
  - diversity: 0.2 weight (diverse candidates make better tournaments)
  - If champion_score = 0: reward = 0 (the champion is wrong)
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
# Problem generators (reuse BestOfN-style problems)
# ---------------------------------------------------------------------------


_TOURNAMENT_MATH = [
    {"question": "What is 19 * 7?", "answer": "133"},
    {"question": "What is 14^2?", "answer": "196"},
    {"question": "What is 156 / 12?", "answer": "13"},
    {"question": "What is 3^6?", "answer": "729"},
    {"question": "What is 47 + 56?", "answer": "103"},
    {"question": "What is 100 - 38?", "answer": "62"},
    {"question": "What is 11 * 13?", "answer": "143"},
    {"question": "What is 8^3?", "answer": "512"},
    {"question": "What is 180 / 15?", "answer": "12"},
    {"question": "What is 23 * 4?", "answer": "92"},
]

_TOURNAMENT_CODE = [
    {
        "task": "Write `is_even(n)` — True if n is even.",
        "tests": [(2, True), (3, False), (0, True), (-1, False), (100, True)],
    },
    {
        "task": "Write `square(n)` — returns n^2.",
        "tests": [(2, 4), (0, 0), (-3, 9), (5, 25)],
    },
    {
        "task": "Write `is_positive(n)` — True if n > 0.",
        "tests": [(1, True), (0, False), (-1, False), (100, True)],
    },
    {
        "task": "Write `double(n)` — returns n * 2.",
        "tests": [(1, 2), (0, 0), (-5, -10), (50, 100)],
    },
    {
        "task": "Write `abs_val(n)` — returns absolute value.",
        "tests": [(5, 5), (-5, 5), (0, 0), (-100, 100)],
    },
    {
        "task": "Write `max_of_two(a, b)` — returns the larger value.",
        "tests": [(1, 2, 2), (5, 3, 5), (0, 0, 0), (-1, -2, -1)],
    },
    {
        "task": "Write `count_vowels(s)` — count vowels (case-insensitive).",
        "tests": [("hello", 2), ("", 0), ("AEIOU", 5), ("xyz", 0)],
    },
    {
        "task": "Write `sum_list(lst)` — sum of list elements.",
        "tests": [([1, 2, 3], 6), ([], 0), ([5], 5), ([-1, -2, 3], 0)],
    },
]


def tournament_judge_generator(seed: int) -> Problem:
    """Generate a TournamentJudge problem."""
    rng = random.Random(seed)
    if rng.random() < 0.4:
        template = rng.choice(_TOURNAMENT_MATH)
        return Problem(
            id=f"tournament_math_{rng.randint(0, 99999)}",
            prompt=(
                f"Question: {template['question']}\n\n"
                f"End with: ANSWER: <value>"
            ),
            difficulty=0.2 + 0.1 * rng.random(),
            metadata={
                "type": "tournament_judge",
                "subtype": "math",
                "answer": template["answer"],
            },
            token_budget=200,
            source="generated",
        )
    else:
        template = rng.choice(_TOURNAMENT_CODE)
        return Problem(
            id=f"tournament_code_{rng.randint(0, 99999)}",
            prompt=(
                f"Task: {template['task']}\n\n"
                f"Output ONLY the code in a ```python block."
            ),
            difficulty=0.25 + 0.1 * rng.random(),
            metadata={
                "type": "tournament_judge",
                "subtype": "code",
                "task": template["task"],
                "tests": template["tests"],
            },
            token_budget=300,
            source="generated",
        )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TournamentVerifier(Verifier):
    """Verifies a single solution for TournamentJudge."""

    def __init__(self, subtype: str, answer: str = "", tests: Optional[list] = None):
        super().__init__()
        self._subtype = subtype
        self._answer = answer
        self._tests = tests or []

    def verify(self, response: str) -> VerifierResult:
        if self._subtype == "math":
            match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if not match:
                return VerifierResult(correct=False, score=0.0, diagnostics="No ANSWER: found")
            submitted = match.group(1).strip()
            if submitted == self._answer:
                return VerifierResult(correct=True, score=1.0, diagnostics="Correct")
            return VerifierResult(correct=False, score=0.0, diagnostics=f"Wrong: {submitted} != {self._answer}")
        else:
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
            if not code_match:
                return VerifierResult(correct=False, score=0.0, diagnostics="No code block")

            code = code_match.group(1).strip()
            namespace: dict[str, Any] = {}
            try:
                exec(code, namespace)
            except Exception as e:
                return VerifierResult(correct=False, score=0.0, diagnostics=f"Exec error: {e}")

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name != "__builtins__":
                    func = obj
                    break
            if func is None:
                return VerifierResult(correct=False, score=0.0, diagnostics="No function found")

            passed = 0
            total = len(self._tests)
            for test in self._tests:
                try:
                    if len(test) == 2:
                        args, expected = test
                        result = func(args) if not isinstance(args, tuple) else func(*args)
                    elif len(test) == 3:
                        a, b, expected = test
                        result = func(a, b)
                    else:
                        args = test[:-1]
                        expected = test[-1]
                        result = func(*args)
                    if result == expected:
                        passed += 1
                except Exception:
                    pass

            score = passed / total if total > 0 else 0.0
            return VerifierResult(correct=passed == total, score=score, diagnostics=f"Tests: {passed}/{total}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TournamentJudgeEnv(BatchEnvBase):
    """
    TournamentJudge environment: N solutions, round-robin tournament.

    The model generates N solutions. Every pair is compared (higher verifier
    score wins, ties broken by conciseness). The champion is the solution
    with the most wins.

    Reward = champion_score * 0.5 + consistency * 0.3 + diversity * 0.2.
    """

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
            problem_generator = tournament_judge_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return TournamentVerifier(
            subtype=meta["subtype"],
            answer=meta.get("answer", ""),
            tests=meta.get("tests", []),
        )

    def _check_format(self, response: str) -> float:
        if "ANSWER:" in response:
            return 1.0
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        n = len(per_sample)
        if n == 0:
            return {"best_score": 0.0, "reward": 0.0, "correct": False, "diagnostics": "No samples"}

        scores = [s["verifier_score"] for s in per_sample]
        tokens = [s["tokens_used"] for s in per_sample]

        # Run round-robin tournament
        wins = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._compare(i, j, scores, tokens)
                wins[winner] += 1

        # Champion = most wins (ties broken by score, then by conciseness)
        champion = max(range(n), key=lambda i: (wins[i], scores[i], -tokens[i]))
        champion_score = scores[champion]
        champion_wins = wins[champion]

        # Consistency: how decisively the champion won
        consistency = champion_wins / max(n - 1, 1)

        # Diversity
        diversity = self._compute_diversity(per_sample)

        # Reward
        if champion_score <= 0:
            reward = 0.0
        else:
            reward = champion_score * 0.5 + consistency * 0.3 + diversity * 0.2
            reward = min(1.0, reward)

        return {
            "best_score": champion_score,
            "champion_index": champion,
            "champion_wins": champion_wins,
            "consistency": consistency,
            "diversity": diversity,
            "wins": wins,
            "scores": scores,
            "any_correct": any(s["correct"] for s in per_sample),
            "correct_count": sum(1 for s in per_sample if s["correct"]),
            "reward": reward,
            "correct": champion_score >= 0.8,
            "diagnostics": f"Champion=#{champion} (score={champion_score:.2f}, wins={champion_wins}/{n-1}) Consistency={consistency:.2f} Diversity={diversity:.2f}",
        }

    def _compare(self, i: int, j: int, scores: list[float], tokens: list[int]) -> int:
        """Compare two solutions. Returns the index of the winner."""
        # Higher score wins
        if scores[i] > scores[j]:
            return i
        if scores[j] > scores[i]:
            return j
        # Tie on score — shorter (more concise) wins
        if tokens[i] <= tokens[j]:
            return i
        return j

    def _compute_diversity(self, per_sample: list[dict]) -> float:
        """Compute diversity as fraction of distinct solutions."""
        if not per_sample:
            return 0.0

        answers = set()
        for s in per_sample:
            resp = s.get("response", "")
            match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", resp, re.IGNORECASE)
            if match:
                answers.add(match.group(1).strip())
            else:
                answers.add(hash(resp.strip()))

        if not answers:
            return 0.0

        return min(1.0, len(answers) / len(per_sample))
