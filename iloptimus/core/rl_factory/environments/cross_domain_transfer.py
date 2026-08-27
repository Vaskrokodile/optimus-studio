"""
CrossDomainTransfer: Solve analogous problems across math/code/logic domains.

Environment concept:
  The model is given a problem in one domain and an analogous problem in
  another domain. It must solve BOTH, demonstrating that it can transfer
  understanding across representations.

  Why: true understanding is domain-invariant. A model that can solve
  "sum 1..n" in math but not as a code function has memorized, not
  understood. Cross-domain transfer tests for genuine abstraction.

Verification:
  - Both answers must be correct.
  - Partial credit for getting one of the two right.

Reward design:
  (answer1_correct * 0.5 + answer2_correct * 0.5)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — pairs of analogous problems across domains
# ---------------------------------------------------------------------------

_DOMAIN_PAIRS = [
    {
        "domain1": "math",
        "domain2": "code",
        "problem1": "What is the sum of all integers from 1 to 10?",
        "problem2": "Write a function sum_range(n) that returns the sum of integers from 1 to n. What does sum_range(10) return?",
        "answer1": "55",
        "answer2": "55",
        "concept": "sum of arithmetic series",
    },
    {
        "domain1": "math",
        "domain2": "logic",
        "problem1": "If x > 5 and x < 10, what are the integer values of x?",
        "problem2": "All A are greater than 5. All A are less than 10. What integer values can A be?",
        "answer1": "6, 7, 8, 9",
        "answer2": "6, 7, 8, 9",
        "concept": "bounded integer range",
    },
    {
        "domain1": "code",
        "domain2": "math",
        "problem1": "A function returns the length of a list. If the list has 7 elements, what does len() return?",
        "problem2": "A set has 7 elements. What is the cardinality of the set?",
        "answer1": "7",
        "answer2": "7",
        "concept": "cardinality / count",
    },
    {
        "domain1": "math",
        "domain2": "code",
        "problem1": "What is 2^5?",
        "problem2": "Write a function power(base, exp). What does power(2, 5) return?",
        "answer1": "32",
        "answer2": "32",
        "concept": "exponentiation",
    },
    {
        "domain1": "math",
        "domain2": "logic",
        "problem1": "Is 15 divisible by 3?",
        "problem2": "If A divides B and B = 15 and A = 3, is the statement 'A divides B' true?",
        "answer1": "yes",
        "answer2": "yes",
        "concept": "divisibility",
    },
    {
        "domain1": "code",
        "domain2": "math",
        "problem1": "A list [3, 1, 4, 1, 5] is sorted. What is the first element?",
        "problem2": "What is the minimum of the set {3, 1, 4, 1, 5}?",
        "answer1": "1",
        "answer2": "1",
        "concept": "minimum / sorting",
    },
    {
        "domain1": "math",
        "domain2": "code",
        "problem1": "What is the factorial of 5?",
        "problem2": "A function factorial(n) computes n!. What does factorial(5) return?",
        "answer1": "120",
        "answer2": "120",
        "concept": "factorial",
    },
    {
        "domain1": "logic",
        "domain2": "math",
        "problem1": "If all even numbers are divisible by 2, and 8 is even, is 8 divisible by 2?",
        "problem2": "Is 8 mod 2 equal to 0?",
        "answer1": "yes",
        "answer2": "yes",
        "concept": "evenness / divisibility by 2",
    },
]


def _normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z,\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def cross_domain_transfer_generator(seed: int) -> Problem:
    """Generate a CrossDomainTransfer problem: solve two analogous problems."""
    rng = random.Random(seed)
    pair = rng.choice(_DOMAIN_PAIRS)

    domain1 = pair["domain1"]
    domain2 = pair["domain2"]
    problem1 = pair["problem1"]
    problem2 = pair["problem2"]
    answer1 = pair["answer1"]
    answer2 = pair["answer2"]

    label1 = domain1.upper()
    label2 = domain2.upper()

    prompt = (
        f"You are given two analogous problems from different domains.\n"
        f"Solve BOTH to demonstrate cross-domain understanding.\n\n"
        f"{label1} PROBLEM: {problem1}\n\n"
        f"{label2} PROBLEM: {problem2}\n\n"
        f"Format your answer as:\n"
        f"{label1}_ANSWER: <answer1>\n"
        f"{label2}_ANSWER: <answer2>"
    )

    difficulty = 0.5

    return Problem(
        id=f"cross_domain_transfer_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "domain1": domain1,
            "domain2": domain2,
            "problem1": problem1,
            "problem2": problem2,
            "answer1": answer1,
            "answer2": answer2,
            "label1": label1,
            "label2": label2,
            "concept": pair["concept"],
        },
        token_budget=512,
        source="cross_domain_transfer_generator",
    )


cross_domain_transfer_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CrossDomainTransferVerifier(Verifier):
    """Verify a cross-domain transfer: both answers must be correct.

    reward = answer1_correct * 0.5 + answer2_correct * 0.5
    """

    def __init__(self, answer1: str, answer2: str, label1: str, label2: str):
        super().__init__()
        self._answer1 = _normalize_answer(answer1)
        self._answer2 = _normalize_answer(answer2)
        self._label1 = label1
        self._label2 = label2

    def _extract_labeled_answer(self, response: str, label: str) -> str:
        pattern = rf"{label}_ANSWER\s*:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def verify(self, response: str) -> VerifierResult:
        given1 = _normalize_answer(self._extract_labeled_answer(response, self._label1))
        given2 = _normalize_answer(self._extract_labeled_answer(response, self._label2))

        a1_correct = given1 == self._answer1 if given1 else False
        a2_correct = given2 == self._answer2 if given2 else False

        # Partial: check if the answer number appears
        if not a1_correct and given1:
            ans_nums = re.findall(r"\d+", self._answer1)
            giv_nums = re.findall(r"\d+", given1)
            if ans_nums and giv_nums and ans_nums == giv_nums:
                a1_correct = True
        if not a2_correct and given2:
            ans_nums = re.findall(r"\d+", self._answer2)
            giv_nums = re.findall(r"\d+", given2)
            if ans_nums and giv_nums and ans_nums == giv_nums:
                a2_correct = True

        score = (1.0 if a1_correct else 0.0) * 0.5 + (1.0 if a2_correct else 0.0) * 0.5
        correct = a1_correct and a2_correct

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "answer1_correct": 1.0 if a1_correct else 0.0,
                "answer2_correct": 1.0 if a2_correct else 0.0,
            },
            diagnostics=(
                f"a1={a1_correct} ({given1!r} vs {self._answer1!r}) "
                f"a2={a2_correct} ({given2!r} vs {self._answer2!r})"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CrossDomainTransferEnv(BatchEnvBase):
    """CrossDomainTransfer: solve analogous problems across domains.

    Batch-aware: N parallel attempts; reward = best dual-answer.
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
            problem_generator = cross_domain_transfer_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CrossDomainTransferVerifier(
            answer1=problem.metadata["answer1"],
            answer2=problem.metadata["answer2"],
            label1=problem.metadata["label1"],
            label2=problem.metadata["label2"],
        )

    def _check_format(self, response: str) -> float:
        has_a1 = bool(re.search(r"_ANSWER\s*:", response, re.IGNORECASE))
        if has_a1:
            count = len(re.findall(r"_ANSWER\s*:", response, re.IGNORECASE))
            if count >= 2:
                return 1.0
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
        matches = re.findall(r"_ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return " | ".join(m.strip() for m in matches) if matches else response
