"""
SelfConsistency: Generate N solutions, majority vote, reward = vote + agreement.

Environment concept:
  The model generates N solutions to a problem in parallel. Each solution
  produces an answer. The environment:
    1. Extracts the answer from each solution
    2. Performs majority voting (the most common answer wins)
    3. Rewards based on:
       - Vote correctness: is the majority-voted answer correct?
       - Agreement fraction: what fraction of solutions agree with the vote?

    reward = vote_correct * 0.6 + agreement_fraction * 0.2 + best_individual * 0.2

  This trains the model to:
    1. Generate DIVERSE reasoning paths that converge on the SAME answer
    2. Be confident when multiple paths agree (high agreement = high reward)
    3. Explore different approaches (diversity in reasoning, consistency in answer)

  The key insight: self-consistency is the most reliable test-time compute
  scaling method. If 15 out of 16 solutions agree, the answer is almost
  certainly correct — even if no single solution is provably correct.

Why this environment is worth using for 10T-100T param models:
  Self-consistency voting can boost accuracy by 10-20% over single-sample
  decoding with zero model changes. A 100T model with self-consistency can
  match a 300T model without it. This environment trains the model to
  produce solutions that AGREE when correct and DIVERGE when uncertain —
  the ideal distribution for voting.

Problem types:
  - Math problems (numeric answers — easy to vote on)
  - Logic problems (yes/no or multiple choice)
  - Code output prediction (numeric/string outputs)

Reward design:
  - vote_correct: 0.6 weight (the majority-voted answer must be right)
  - agreement_fraction: 0.2 weight (higher agreement = higher confidence)
  - best_individual: 0.2 weight (at least one solution should be correct)
  - If vote is wrong: reward capped at 0.2 (only best_individual can save it)
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_SC_MATH = [
    {"question": "What is 24 * 7?", "answer": "168"},
    {"question": "What is 15 + 38?", "answer": "53"},
    {"question": "What is 100 - 42?", "answer": "58"},
    {"question": "What is 9 * 13?", "answer": "117"},
    {"question": "What is 144 / 12?", "answer": "12"},
    {"question": "What is 6^3?", "answer": "216"},
    {"question": "What is 37 * 3?", "answer": "111"},
    {"question": "What is 200 / 8?", "answer": "25"},
    {"question": "What is 11 * 14?", "answer": "154"},
    {"question": "What is 5^5?", "answer": "3125"},
    {"question": "What is 18 * 9?", "answer": "162"},
    {"question": "What is 7 * 8 + 3?", "answer": "59"},
    {"question": "What is 1000 / 25?", "answer": "40"},
    {"question": "What is 13^2?", "answer": "169"},
    {"question": "What is 45 + 67?", "answer": "112"},
]

_SC_LOGIC = [
    {
        "question": "All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
        "answer": "Yes",
        "choices": ["Yes", "No"],
    },
    {
        "question": "If it rains, the ground gets wet. The ground is dry. Did it rain?",
        "answer": "No",
        "choices": ["Yes", "No"],
    },
    {
        "question": "All squares have 4 sides. A rectangle has 4 sides. Is a rectangle a square?",
        "answer": "No",
        "choices": ["Yes", "No"],
    },
    {
        "question": "5 > 3 and 3 > 1. Is 5 > 1?",
        "answer": "Yes",
        "choices": ["Yes", "No"],
    },
    {
        "question": "If x = 5 and y = x + 3, what is y? (Answer with the number)",
        "answer": "8",
        "choices": ["7", "8", "9", "15"],
    },
    {
        "question": "A triangle has angles 60, 60, 60. Is it equilateral?",
        "answer": "Yes",
        "choices": ["Yes", "No"],
    },
    {
        "question": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
        "answer": "Yes",
        "choices": ["Yes", "No"],
    },
    {
        "question": "Is 17 a prime number?",
        "answer": "Yes",
        "choices": ["Yes", "No"],
    },
]


def self_consistency_generator(seed: int) -> Problem:
    """Generate a SelfConsistency problem (math or logic)."""
    rng = random.Random(seed)
    if rng.random() < 0.6:
        template = rng.choice(_SC_MATH)
        return Problem(
            id=f"selfconsistency_math_{rng.randint(0, 99999)}",
            prompt=(
                f"Question: {template['question']}\n\n"
                f"Solve this step by step.\n"
                f"End with: ANSWER: <value>"
            ),
            difficulty=0.2 + 0.1 * rng.random(),
            metadata={
                "type": "self_consistency",
                "subtype": "math",
                "answer": template["answer"],
            },
            token_budget=300,
            source="generated",
        )
    else:
        template = rng.choice(_SC_LOGIC)
        choices_str = " / ".join(template["choices"])
        return Problem(
            id=f"selfconsistency_logic_{rng.randint(0, 99999)}",
            prompt=(
                f"Question: {template['question']}\n"
                f"Choices: {choices_str}\n\n"
                f"End with: ANSWER: <choice>"
            ),
            difficulty=0.2 + 0.1 * rng.random(),
            metadata={
                "type": "self_consistency",
                "subtype": "logic",
                "answer": template["answer"],
                "choices": template["choices"],
            },
            token_budget=200,
            source="generated",
        )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SelfConsistencyVerifier(Verifier):
    """Verifies a single solution for SelfConsistency."""

    def __init__(self, answer: str):
        super().__init__()
        self._answer = answer

    def verify(self, response: str) -> VerifierResult:
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if not match:
            return VerifierResult(correct=False, score=0.0, diagnostics="No ANSWER: found")
        submitted = match.group(1).strip()
        if submitted.lower() == self._answer.lower():
            return VerifierResult(correct=True, score=1.0, diagnostics="Correct")
        return VerifierResult(correct=False, score=0.0, diagnostics=f"Wrong: {submitted} != {self._answer}")

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the answer from a response (for voting)."""
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SelfConsistencyEnv(BatchEnvBase):
    """
    SelfConsistency environment: N solutions, majority vote.

    The model generates N solutions. Answers are extracted and voted on.
    Reward = vote_correct * 0.6 + agreement * 0.2 + best_individual * 0.2.
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
            problem_generator = self_consistency_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SelfConsistencyVerifier(answer=problem.metadata["answer"])

    def _check_format(self, response: str) -> float:
        if "ANSWER:" in response:
            return 1.0
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        # We need the raw responses to extract answers for voting
        # per_sample only has truncated responses, so we need to re-extract
        # from the verifier diagnostics (which contain the submitted answer)
        answers = []
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]

        # Extract answers from diagnostics
        for s in per_sample:
            diag = s.get("verifier_diagnostics", "")
            # Parse "Wrong: X != Y" or "Correct"
            if diag.startswith("Correct"):
                # The answer matches — we need to get it from the problem
                answers.append(self._current_problem.metadata["answer"])
            elif diag.startswith("Wrong:"):
                match = re.match(r"Wrong:\s*(.+?)\s*!=\s*.+", diag)
                if match:
                    answers.append(match.group(1).strip())
                else:
                    answers.append("__NO_ANSWER__")
            else:
                answers.append("__NO_ANSWER__")

        # Majority vote
        if not answers:
            return {
                "best_score": 0.0,
                "vote_answer": None,
                "vote_correct": False,
                "agreement": 0.0,
                "reward": 0.0,
                "correct": False,
                "diagnostics": "No answers extracted",
            }

        answer_counts = Counter(a.lower() for a in answers if a != "__NO_ANSWER__")
        if not answer_counts:
            return {
                "best_score": max(scores) if scores else 0.0,
                "vote_answer": None,
                "vote_correct": False,
                "agreement": 0.0,
                "reward": 0.0,
                "correct": False,
                "diagnostics": "No valid answers to vote on",
            }

        vote_answer, vote_count = answer_counts.most_common(1)[0]
        agreement = vote_count / len(answers)

        # Check if the vote is correct
        correct_answer = self._current_problem.metadata["answer"].lower()
        vote_correct = vote_answer == correct_answer

        best_individual = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Reward computation
        if vote_correct:
            reward = 0.6 + agreement * 0.2 + best_individual * 0.2
        else:
            # Vote is wrong — only best_individual can contribute (capped at 0.2)
            reward = best_individual * 0.2

        reward = min(1.0, reward)

        return {
            "best_score": best_individual,
            "vote_answer": vote_answer,
            "vote_count": vote_count,
            "vote_correct": vote_correct,
            "agreement": agreement,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "answer_distribution": dict(answer_counts),
            "reward": reward,
            "correct": vote_correct,
            "diagnostics": f"Vote='{vote_answer}' ({vote_count}/{len(answers)} agree) Correct={vote_correct} Agreement={agreement:.0%}",
        }
