"""
CalibratedQA: Buy hints vs commit, calibrated confidence.

Environment concept:
  The model is given a question and a "hint budget" — it can spend tokens
  to request hints (each hint costs budget and reduces reward) or commit
  to an answer immediately. The key tradeoff:

    - Commit too early with low confidence → wrong answer → zero reward
    - Buy too many hints → correct answer but wasted budget → reduced reward
    - Perfect calibration → commit at exactly the right confidence level

  This trains the model to:
    1. Know what it knows (calibrated confidence)
    2. Ask for help only when genuinely uncertain
    3. Avoid hedging and over-verification ("let me double-check...")
    4. Make decisive commitments instead of endless deliberation

Why this environment is unique:
  Most QA environments are single-shot: answer the question, get a reward.
  CalibratedQA introduces a MULTI-STEP decision process where the model
  must actively manage its information budget. This directly attacks the
  frontier-model tendency to over-verify and hedge — the model that
  commits confidently and correctly gets the highest reward.

Mechanics:
  - The model can output one of three action types per step:
    1. "HINT" — request a hint (costs budget, reveals partial info)
    2. "ANSWER: <answer>" — commit to a final answer
    3. "PASS" — skip (costs budget, no benefit — penalized)
  - Up to 3 hints are available, each progressively more revealing
  - The reward = correctness * (1 - hint_cost) * confidence_bonus
  - Confidence bonus: if the model commits without hints and is correct,
    it gets a large bonus. If it uses all hints, the bonus is minimal.

Problem types:
  - Factual questions with varying difficulty
  - Math problems with progressive hints
  - Logic puzzles with progressive hints
  - Code output prediction with progressive hints
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank
# ---------------------------------------------------------------------------


_PROBLEM_BANK = [
    {
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
        "hints": [
            "It is not Sydney or Melbourne.",
            "It is a purpose-built capital city.",
            "It starts with 'C'.",
        ],
        "difficulty": 0.2,
    },
    {
        "question": "What is 17 * 23?",
        "answer": "391",
        "hints": [
            "Think about it as (20-3)*(20+3) = 400-9.",
            "The answer is between 350 and 450.",
            "The last digit is 1.",
        ],
        "difficulty": 0.3,
    },
    {
        "question": "What is the time complexity of binary search?",
        "answer": "O(log n)",
        "hints": [
            "It involves repeatedly halving the search space.",
            "It is logarithmic, not linear.",
            "The base of the log is 2.",
        ],
        "difficulty": 0.25,
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "Au",
        "hints": [
            "It comes from the Latin word 'aurum'.",
            "It is a two-letter symbol.",
            "The first letter is 'A'.",
        ],
        "difficulty": 0.15,
    },
    {
        "question": "How many planets are in our solar system?",
        "answer": "8",
        "hints": [
            "Pluto was reclassified as a dwarf planet in 2006.",
            "The number is between 7 and 9.",
            "It is an even number.",
        ],
        "difficulty": 0.15,
    },
    {
        "question": "What is the derivative of x^3?",
        "answer": "3x^2",
        "hints": [
            "Use the power rule: d/dx(x^n) = n*x^(n-1).",
            "The exponent decreases by 1.",
            "The coefficient becomes the old exponent.",
        ],
        "difficulty": 0.3,
    },
    {
        "question": "What is the output of: print(len('hello world'))?",
        "answer": "11",
        "hints": [
            "Count the characters including the space.",
            "The string has 5 + 1 + 5 = 11 characters.",
            "len() counts all characters including spaces.",
        ],
        "difficulty": 0.2,
    },
    {
        "question": "What is the largest prime less than 20?",
        "answer": "19",
        "hints": [
            "Check the numbers from 19 downward.",
            "19 is prime, 18 is not (divisible by 2).",
            "It ends in 9.",
        ],
        "difficulty": 0.35,
    },
    {
        "question": "What does ACID stand for in databases?",
        "answer": "Atomicity, Consistency, Isolation, Durability",
        "hints": [
            "Each letter is a property of transactions.",
            "A = Atomicity, C = Consistency.",
            "I = Isolation, D = Durability.",
        ],
        "difficulty": 0.4,
    },
    {
        "question": "What is 2^10?",
        "answer": "1024",
        "hints": [
            "It is a power of 2 commonly used in computing.",
            "2^8 = 256, so 2^10 = 256 * 4.",
            "It is slightly over 1000.",
        ],
        "difficulty": 0.25,
    },
    {
        "question": "What is the square root of 144?",
        "answer": "12",
        "hints": [
            "It is an integer.",
            "It is between 10 and 15.",
            "12 * 12 = 144.",
        ],
        "difficulty": 0.2,
    },
    {
        "question": "What sorting algorithm has O(n log n) average time and is divide-and-conquer?",
        "answer": "Merge sort",
        "hints": [
            "It is not quicksort (though quicksort is also O(n log n) average).",
            "It guarantees O(n log n) worst case.",
            "It merges sorted subarrays.",
        ],
        "difficulty": 0.45,
    },
    {
        "question": "What is the factorial of 5?",
        "answer": "120",
        "hints": [
            "5! = 5 * 4 * 3 * 2 * 1.",
            "The result is between 100 and 150.",
            "5 * 24 = 120.",
        ],
        "difficulty": 0.2,
    },
    {
        "question": "In Python, what does the 'self' parameter represent in a class method?",
        "answer": "The instance of the class",
        "hints": [
            "It is a reference to the current object.",
            "It is automatically passed when you call a method on an instance.",
            "It allows access to instance attributes and other methods.",
        ],
        "difficulty": 0.3,
    },
    {
        "question": "What is the Fibonacci number at position 10 (0-indexed)?",
        "answer": "55",
        "hints": [
            "The sequence starts 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...",
            "Position 10 means the 11th number (0-indexed).",
            "It is the sum of positions 8 and 9: 21 + 34.",
        ],
        "difficulty": 0.35,
    },
    {
        "question": "What is the output of: print(type([]))?",
        "answer": "<class 'list'>",
        "hints": [
            "[] is an empty list literal.",
            "type() returns the class of an object.",
            "The output format is <class 'typename'>.",
        ],
        "difficulty": 0.3,
    },
    {
        "question": "What is the HTTP status code for 'Not Found'?",
        "answer": "404",
        "hints": [
            "It is in the 4xx range (client errors).",
            "It is one of the most well-known status codes.",
            "It is a three-digit number starting with 4.",
        ],
        "difficulty": 0.15,
    },
    {
        "question": "What is the sum of angles in a triangle (in degrees)?",
        "answer": "180",
        "hints": [
            "It is the same for all triangles.",
            "It is a multiple of 90.",
            "60 + 60 + 60 = 180 for an equilateral triangle.",
        ],
        "difficulty": 0.15,
    },
    {
        "question": "What is the output of: print(2 ** 3 ** 2)?",
        "answer": "512",
        "hints": [
            "Exponentiation is right-associative in Python.",
            "It is 2 ** (3 ** 2) = 2 ** 9.",
            "2^9 = 512.",
        ],
        "difficulty": 0.45,
    },
    {
        "question": "What data structure uses LIFO (Last In, First Out) ordering?",
        "answer": "Stack",
        "hints": [
            "Think of a stack of plates.",
            "The last item added is the first removed.",
            "Operations are called push and pop.",
        ],
        "difficulty": 0.2,
    },
]


def calibrated_qa_generator(seed: int) -> Problem:
    """Generate a CalibratedQA problem from the bank."""
    rng = random.Random(seed)
    entry = rng.choice(_PROBLEM_BANK)

    return Problem(
        id=f"calibrated_qa_{seed}_{rng.randint(0, 99999)}",
        prompt=_format_prompt(entry),
        difficulty=entry["difficulty"],
        metadata={
            "type": "calibrated_qa",
            "question": entry["question"],
            "answer": entry["answer"],
            "hints": entry["hints"],
            "max_hints": len(entry["hints"]),
        },
        token_budget=1000,
        source="generated",
    )


def _format_prompt(entry: dict) -> str:
    """Format the problem prompt."""
    return (
        f"Answer the following question.\n\n"
        f"Question: {entry['question']}\n\n"
        f"You can take the following actions:\n"
        f"  - Type HINT to request a hint (max {len(entry['hints'])} hints available)\n"
        f"  - Type ANSWER: <your answer> to commit your final answer\n\n"
        f"Scoring:\n"
        f"  - Correct answer with 0 hints: maximum reward\n"
        f"  - Each hint used reduces your reward\n"
        f"  - Wrong answer: zero reward regardless of hints used\n"
        f"  - Be decisive — don't hedge or over-verify"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CalibratedQAVerifier(Verifier):
    """Verifies the final answer against the expected answer."""

    def __init__(self, answer: str):
        super().__init__()
        self._answer = answer

    def verify(self, response: str) -> VerifierResult:
        # The response should be the extracted answer
        norm_response = self._normalize(response)
        norm_expected = self._normalize(self._answer)

        correct = norm_response == norm_expected

        # Also check for partial matches (for longer answers)
        if not correct:
            # Check if the expected answer is contained in the response
            if norm_expected in norm_response:
                correct = True

        return VerifierResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            diagnostics=f"Expected: {self._answer!r}, Got: {response!r}",
        )

    def _normalize(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s


# ---------------------------------------------------------------------------
# Environment (multi-turn)
# ---------------------------------------------------------------------------


class CalibratedQAEnv(BaseReasoningEnv):
    """
    CalibratedQA environment: buy hints vs commit.

    This is a MULTI-TURN environment. The model can:
      - Request hints (up to max_hints)
      - Commit an answer
      - The episode ends when the model commits or runs out of budget

    The reward accounts for:
      - Correctness (gate)
      - Number of hints used (fewer = better)
      - Decisiveness (committing quickly = bonus)
      - Anti-patterns (hedging, over-verification)
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
    ):
        if problems is None and problem_generator is None:
            problem_generator = calibrated_qa_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

        # Multi-turn state
        self._hints_used: int = 0
        self._hints_revealed: list[str] = []
        self._committed: bool = False
        self._final_answer: str = ""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._hints_used = 0
        self._hints_revealed = []
        self._committed = False
        self._final_answer = ""
        return obs, info

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CalibratedQAVerifier(answer=problem.metadata["answer"])

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """
        Multi-turn step: parse the action and respond accordingly.

        Actions:
          - "HINT" → reveal next hint
          - "ANSWER: <answer>" → commit and end episode
          - anything else → treated as a pass (wastes budget)
        """
        if self._current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        action = action.strip()
        self._step_count += 1
        tokens_used = self._budget.consume(action)
        self._episode_responses.append(action)

        truncated = self._budget.remaining <= 0
        terminated = False
        reward = 0.0

        # Parse the action
        answer_match = re.match(r"ANSWER:\s*(.+)", action, re.IGNORECASE)

        if answer_match:
            # Model commits an answer
            self._final_answer = answer_match.group(1).strip()
            self._committed = True
            terminated = True

            # Verify
            verifier_result = self._verifier.verify(self._final_answer)
            correct = verifier_result.correct

            # Compute hint penalty
            max_hints = self._current_problem.metadata["max_hints"]
            hint_penalty = self._hints_used / max(max_hints, 1) * 0.5  # up to 50% penalty

            # Decisiveness bonus: committing in fewer steps
            decisiveness_bonus = max(0, 1.0 - 0.1 * (self._step_count - 1)) * 0.2

            # Format bonus
            format_bonus = 1.0  # correct format (ANSWER: ...)

            # Efficiency
            from iloptimus.core.rl_factory.core.reward import compute_efficiency_bonus
            efficiency = compute_efficiency_bonus(
                self._budget, self._reward_config, self._current_problem.difficulty
            )

            # Anti-pattern detection on all responses combined
            full_trace = "\n".join(self._episode_responses)
            ap_report = self._ap_detector.analyze(full_trace, prompt=self._current_problem.prompt)

            # Difficulty bonus
            difficulty_bonus = self._current_problem.difficulty if correct else 0.0

            if correct:
                base_reward = 1.0 - hint_penalty + decisiveness_bonus
                base_reward = max(self._reward_config.correct_floor, base_reward)
            else:
                base_reward = 0.0

            # Anti-pattern penalty
            ap_penalty = -ap_report.total_penalty * self._reward_config.w_anti_pattern

            # Efficiency bonus
            eff_bonus = efficiency * self._reward_config.w_efficiency if correct else 0.0

            reward = base_reward + eff_bonus + ap_penalty
            reward = max(0.0, min(2.0, reward))

            info = {
                "problem_id": self._current_problem.id,
                "correct": correct,
                "hints_used": self._hints_used,
                "final_answer": self._final_answer,
                "expected_answer": self._current_problem.metadata["answer"],
                "hint_penalty": hint_penalty,
                "decisiveness_bonus": decisiveness_bonus,
                "tokens_used": self._budget.used,
                "budget_used_fraction": self._budget.fraction_used,
                "step_count": self._step_count,
                "anti_pattern_hits": ap_report.total_hits,
                "anti_pattern_counts": ap_report.counts_by_type,
                "reward": reward,
                "verifier_diagnostics": verifier_result.diagnostics,
            }

        elif action.upper().startswith("HINT"):
            # Model requests a hint
            max_hints = self._current_problem.metadata["max_hints"]
            hints = self._current_problem.metadata["hints"]

            if self._hints_used < max_hints:
                self._hints_revealed.append(hints[self._hints_used])
                self._hints_used += 1
                # Small negative reward for using a hint
                reward = -0.05
                terminated = False
            else:
                # No more hints available
                reward = -0.1  # penalty for wasting a turn
                terminated = False

            info = {
                "problem_id": self._current_problem.id,
                "hints_used": self._hints_used,
                "hint_revealed": self._hints_revealed[-1] if self._hints_revealed else None,
                "tokens_used": self._budget.used,
                "budget_used_fraction": self._budget.fraction_used,
                "step_count": self._step_count,
                "reward": reward,
            }

        else:
            # Unrecognized action — treat as a wasted turn
            reward = -0.1
            terminated = False
            info = {
                "problem_id": self._current_problem.id,
                "action_type": "unrecognized",
                "tokens_used": self._budget.used,
                "budget_used_fraction": self._budget.fraction_used,
                "step_count": self._step_count,
                "reward": reward,
            }

        # If budget is exhausted and no answer committed, end with zero reward
        if truncated and not self._committed:
            terminated = True
            reward = 0.0
            info["budget_exhausted"] = True

        obs = self._make_observation()
        return obs, reward, terminated, truncated, info

    def _make_observation(self) -> dict:
        """Build observation including revealed hints."""
        base_obs = super()._make_observation()

        if self._current_problem is None:
            return base_obs

        # Add hint info to the prompt
        prompt = self._current_problem.prompt
        if self._hints_revealed:
            hints_text = "\n\nHints revealed so far:\n"
            for i, hint in enumerate(self._hints_revealed, 1):
                hints_text += f"  Hint {i}: {hint}\n"
            prompt += hints_text

        remaining_hints = self._current_problem.metadata["max_hints"] - self._hints_used
        prompt += f"\n\nHints remaining: {remaining_hints}"

        base_obs["prompt"] = prompt
        return base_obs

    def _check_format(self, response: str) -> float:
        if re.match(r"ANSWER:\s*.+", response, re.IGNORECASE):
            return 1.0
        if response.upper().startswith("HINT"):
            return 0.5
        return 0.0

    def _extract_answer(self, response: str) -> str:
        match = re.match(r"ANSWER:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response
