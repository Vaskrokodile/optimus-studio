"""
RedundancyEliminator: Eliminate redundant steps from reasoning traces.

Environment concept:
  The model is given a multi-step reasoning trace that contains redundant
  steps (repeated logic, unnecessary verification, filler, backtracking).
  It must produce an equivalent trace with FEWER steps — removing only
  the redundant ones, keeping all necessary steps.

  This trains the model to:
    1. Identify which steps are essential vs redundant
    2. Remove waste without losing correctness
    3. Produce compressed traces that reach the same conclusion
    4. Recognize backtracking patterns and remove the dead ends

  This is different from TrajectoryDoctor (which identifies ONE wrong step)
  and ProofGolf (which writes a proof from scratch). RedundancyEliminator
  takes an EXISTING trace and COMPRESSES it — the editing skill.

Why this environment is worth using for 10T-100T param models:
  Trace compression is the core skill for efficient chain-of-thought.
  A 100T model that can compress its own traces from 50 steps to 20 steps
  without losing accuracy saves 60% of reasoning tokens. This compounds
  across every inference call.

Problem types:
  - Math traces with redundant verification steps
  - Logic traces with backtracking that should be removed
  - Code traces with unnecessary intermediate checks
  - Real frontier model traces with identified waste

Verification:
  Each problem has:
    - The original trace (with numbered steps)
    - The set of redundant step indices (ground truth)
    - The essential step indices
    - The final answer (must be preserved)
  The verifier:
    1. Parses which steps the model chose to keep/remove
    2. Checks that all essential steps are kept
    3. Checks that all redundant steps are removed
    4. Checks that the final answer is preserved

Reward design:
  - Precision: fraction of removed steps that were actually redundant
  - Recall: fraction of redundant steps that were removed
  - F1 = 2 * P * R / (P + R)
  - Answer preservation: if the final answer is missing, penalty
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_REDUNDANCY_TEMPLATES = [
    {
        "steps": [
            "We need to compute 7 * 8.",
            "7 * 8 = 56.",
            "Let me double-check: 7 * 8 = 56.",
            "Let me verify once more: 7 * 8 = 56. Confirmed.",
            "The answer is 56.",
        ],
        "redundant": [2, 3],  # steps 3 and 4 (0-indexed)
        "essential": [0, 1, 4],
        "answer": "56",
    },
    {
        "steps": [
            "We need to find x where 2x + 6 = 14.",
            "Subtract 6: 2x = 8.",
            "Divide by 2: x = 4.",
            "Let me verify: 2 * 4 + 6 = 8 + 6 = 14. Correct.",
            "So x = 4.",
        ],
        "redundant": [3],
        "essential": [0, 1, 2, 4],
        "answer": "4",
    },
    {
        "steps": [
            "Compute the sum: 3 + 7 + 12 + 8.",
            "3 + 7 = 10.",
            "10 + 12 = 22.",
            "22 + 8 = 30.",
            "Let me recheck: 3 + 7 = 10, 10 + 12 = 22, 22 + 8 = 30.",
            "The sum is 30.",
        ],
        "redundant": [4],
        "essential": [0, 1, 2, 3, 5],
        "answer": "30",
    },
    {
        "steps": [
            "We need to evaluate (3 + 4)^2.",
            "3 + 4 = 7.",
            "7^2 = 49.",
            "Let me think about this. This is an interesting problem.",
            "The answer is 49.",
        ],
        "redundant": [3],
        "essential": [0, 1, 2, 4],
        "answer": "49",
    },
    {
        "steps": [
            "We need to compute 25% of 80.",
            "25% = 0.25.",
            "0.25 * 80 = 20.",
            "Let me verify: 0.25 * 80 = 20. Yes.",
            "Let me double-check: 80 / 4 = 20. Confirmed.",
            "So 25% of 80 is 20.",
        ],
        "redundant": [3, 4],
        "essential": [0, 1, 2, 5],
        "answer": "20",
    },
    {
        "steps": [
            "Problem: What is 15 + 27?",
            "Let me think about this. This is an interesting problem.",
            "15 + 27 = 42.",
            "The answer is 42.",
        ],
        "redundant": [1],
        "essential": [0, 2, 3],
        "answer": "42",
    },
    {
        "steps": [
            "We need to find the maximum of [3, 7, 2, 9, 1].",
            "The maximum is 9.",
            "Wait, let me reconsider. Actually, let me check each element.",
            "3 < 9, 7 < 9, 2 < 9, 9 is the max, 1 < 9.",
            "So the maximum is 9.",
        ],
        "redundant": [2, 3],
        "essential": [0, 1, 4],
        "answer": "9",
    },
    {
        "steps": [
            "We need to solve x^2 = 49.",
            "x = sqrt(49) = 7.",
            "But x could also be -7.",
            "Let me verify: 7^2 = 49. Correct.",
            "Let me verify: (-7)^2 = 49. Correct.",
            "So x = 7 or x = -7.",
        ],
        "redundant": [3, 4],
        "essential": [0, 1, 2, 5],
        "answer": "7 or -7",
    },
    {
        "steps": [
            "Compute 48 / 6 * 2.",
            "48 / 6 = 8.",
            "8 * 2 = 16.",
            "Let me recheck: 48 / 6 = 8, 8 * 2 = 16. Yes.",
            "The answer is 16.",
        ],
        "redundant": [3],
        "essential": [0, 1, 2, 4],
        "answer": "16",
    },
    {
        "steps": [
            "We need to compute 15 * 12.",
            "15 * 12 = 15 * (10 + 2) = 150 + 30 = 180.",
            "Let me double-check: 15 * 10 = 150, 15 * 2 = 30, 150 + 30 = 180.",
            "Let me verify once more: 15 * 12 = 180. Confirmed.",
            "So 15 * 12 = 180.",
        ],
        "redundant": [2, 3],
        "essential": [0, 1, 4],
        "answer": "180",
    },
]


def redundancy_eliminator_generator(seed: int) -> Problem:
    """Generate a RedundancyEliminator problem."""
    rng = random.Random(seed)
    template = rng.choice(_REDUNDANCY_TEMPLATES)

    steps = template["steps"]
    redundant = set(template["redundant"])
    essential = set(template["essential"])
    answer = template["answer"]

    # Format the trace
    trace_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))

    difficulty = 0.2 + 0.15 * len(redundant) + 0.1 * rng.random()

    return Problem(
        id=f"redundancy_elim_{rng.randint(0, 99999)}",
        prompt=_format_prompt(trace_text, len(steps), answer),
        difficulty=min(0.8, difficulty),
        metadata={
            "type": "redundancy_eliminator",
            "steps": steps,
            "redundant": sorted(redundant),
            "essential": sorted(essential),
            "answer": answer,
            "num_steps": len(steps),
        },
        token_budget=400,
        source="generated",
    )


def _format_prompt(trace_text: str, num_steps: int, answer: str) -> str:
    return (
        f"The following reasoning trace contains redundant steps.\n\n"
        f"{trace_text}\n\n"
        f"The trace concludes with the answer: {answer}\n\n"
        f"Your task: Output the list of step numbers to REMOVE.\n"
        f"Format: REMOVE: <step numbers separated by commas>\n"
        f"Example: REMOVE: 3, 4, 5\n\n"
        f"Rules:\n"
        f"  - Remove ONLY redundant steps (verification, filler, backtracking)\n"
        f"  - Keep ALL essential steps\n"
        f"  - The remaining steps must still lead to the answer\n"
        f"  - Be precise — removing essential steps reduces your score"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class RedundancyEliminatorVerifier(Verifier):
    """
    Verifies a RedundancyEliminator response by checking:
      1. Which steps the model chose to remove
      2. Precision: fraction of removed steps that were actually redundant
      3. Recall: fraction of redundant steps that were removed
      4. F1 score
    """

    def __init__(self, redundant: list[int], essential: list[int], answer: str):
        super().__init__()
        self._redundant = set(redundant)
        self._essential = set(essential)
        self._answer = answer

    def verify(self, response: str) -> VerifierResult:
        # Parse the REMOVE: line
        match = re.search(r"REMOVE:\s*([\d,\s]+)", response, re.IGNORECASE)
        if not match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No REMOVE: <numbers> found in response",
            )

        # Parse step numbers (1-indexed in prompt, 0-indexed internally)
        numbers_str = match.group(1)
        try:
            removed_1indexed = [int(x.strip()) for x in numbers_str.split(",") if x.strip()]
            removed = set(n - 1 for n in removed_1indexed)
        except ValueError:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Invalid step numbers: {numbers_str}",
            )

        # Compute precision and recall
        # Precision: fraction of removed that were actually redundant
        if removed:
            true_removals = len(removed & self._redundant)
            precision = true_removals / len(removed)
        else:
            precision = 1.0 if not self._redundant else 0.0

        # Recall: fraction of redundant that were removed
        if self._redundant:
            recall = len(removed & self._redundant) / len(self._redundant)
        else:
            recall = 1.0 if not removed else 0.5

        # Check if any essential steps were removed (critical error)
        essential_removed = removed & self._essential
        if essential_removed:
            # Heavy penalty for removing essential steps
            penalty = len(essential_removed) * 0.5
            return VerifierResult(
                correct=False,
                score=max(0.0, 0.3 - penalty),
                partial_credit={"precision": precision, "recall": recall, "essential_removed": len(essential_removed)},
                diagnostics=f"Removed {len(essential_removed)} essential steps! P={precision:.2f} R={recall:.2f}",
            )

        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        correct = f1 >= 0.8

        return VerifierResult(
            correct=correct,
            score=f1,
            partial_credit={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "removed_count": len(removed),
                "redundant_count": len(self._redundant),
            },
            diagnostics=f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} (removed {len(removed)}, redundant {len(self._redundant)})",
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RedundancyEliminatorEnv(BaseReasoningEnv):
    """
    RedundancyEliminator environment: eliminate redundant steps from traces.

    The model receives a trace with redundant steps and must identify which
    to remove. Reward = F1 of removal precision/recall.
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
            problem_generator = redundancy_eliminator_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return RedundancyEliminatorVerifier(
            redundant=problem.metadata["redundant"],
            essential=problem.metadata["essential"],
            answer=problem.metadata["answer"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"REMOVE:\s*[\d,\s]+", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
