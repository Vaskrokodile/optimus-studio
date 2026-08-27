"""
CausalSurgery: Repair broken causal chains.

Environment concept:
  The model is given a causal chain (A -> B -> C -> D) with one broken
  link. It must identify the broken link and provide the correct link.

  Why: causal reasoning is fundamental to understanding systems. This
  environment trains the model to identify where a chain breaks and
  repair it — a skill transferable to debugging, scientific reasoning,
  and system design.

Verification:
  - Broken link identified correctly.
  - Correct link provided.

Reward design:
  broken_link_correct * 0.5 + correct_link_correct * 0.5
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — causal chains with one broken link
# ---------------------------------------------------------------------------

_CAUSAL_CHAINS = [
    {
        "chain": ["rain", "wet_ground", "puddles", "slippery", "fall"],
        "broken_link": ("wet_ground", "puddles"),
        "correct_link": ("wet_ground", "puddles"),
        "wrong_link": ("wet_ground", "dry"),
        "description": "rain causes wet ground causes puddles causes slippery causes fall",
    },
    {
        "chain": ["spark", "fire", "smoke", "alarm", "evacuation"],
        "broken_link": ("fire", "smoke"),
        "correct_link": ("fire", "smoke"),
        "wrong_link": ("fire", "water"),
        "description": "spark causes fire causes smoke causes alarm causes evacuation",
    },
    {
        "chain": ["study", "knowledge", "good_grades", "graduation", "job"],
        "broken_link": ("knowledge", "good_grades"),
        "correct_link": ("knowledge", "good_grades"),
        "wrong_link": ("knowledge", "ignorance"),
        "description": "study causes knowledge causes good grades causes graduation causes job",
    },
    {
        "chain": ["seed", "sprout", "plant", "flower", "fruit"],
        "broken_link": ("sprout", "plant"),
        "correct_link": ("sprout", "plant"),
        "wrong_link": ("sprout", "stone"),
        "description": "seed causes sprout causes plant causes flower causes fruit",
    },
    {
        "chain": ["heat", "evaporation", "clouds", "rain", "growth"],
        "broken_link": ("evaporation", "clouds"),
        "correct_link": ("evaporation", "clouds"),
        "wrong_link": ("evaporation", "ice"),
        "description": "heat causes evaporation causes clouds causes rain causes growth",
    },
    {
        "chain": ["virus", "infection", "symptoms", "diagnosis", "treatment"],
        "broken_link": ("infection", "symptoms"),
        "correct_link": ("infection", "symptoms"),
        "wrong_link": ("infection", "health"),
        "description": "virus causes infection causes symptoms causes diagnosis causes treatment",
    },
    {
        "chain": ["friction", "heat", "melting", "liquid", "flow"],
        "broken_link": ("heat", "melting"),
        "correct_link": ("heat", "melting"),
        "wrong_link": ("heat", "freezing"),
        "description": "friction causes heat causes melting causes liquid causes flow",
    },
    {
        "chain": ["input", "process", "output", "feedback", "improvement"],
        "broken_link": ("output", "feedback"),
        "correct_link": ("output", "feedback"),
        "wrong_link": ("output", "nothing"),
        "description": "input causes process causes output causes feedback causes improvement",
    },
]


def _normalize_node(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def causal_surgery_generator(seed: int) -> Problem:
    """Generate a CausalSurgery problem: repair a broken causal chain."""
    rng = random.Random(seed)
    chain_data = rng.choice(_CAUSAL_CHAINS)
    chain = chain_data["chain"]
    broken_link = chain_data["broken_link"]
    correct_link = chain_data["correct_link"]
    wrong_link = chain_data["wrong_link"]

    # Build the chain display with the broken link shown as wrong
    chain_display = []
    for i in range(len(chain) - 1):
        if (chain[i], chain[i + 1]) == broken_link:
            chain_display.append(f"{chain[i]} -> {wrong_link[1]} [BROKEN]")
        else:
            chain_display.append(f"{chain[i]} -> {chain[i + 1]}")

    chain_text = " -> ".join(chain)
    broken_display = "\n".join(chain_display)

    prompt = (
        f"The following causal chain has one broken link:\n\n"
        f"Chain: {chain_text}\n\n"
        f"Links:\n{broken_display}\n\n"
        f"One link is broken (marked [BROKEN]). Identify the broken link "
        f"and provide the correct link.\n"
        f"Format your answer as:\n"
        f"BROKEN_LINK: <from> -> <to>\n"
        f"FIX: <from> -> <to>"
    )

    difficulty = 0.4 + len(chain) * 0.05
    difficulty = min(0.8, difficulty)

    return Problem(
        id=f"causal_surgery_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "chain": chain,
            "broken_link": broken_link,
            "correct_link": correct_link,
            "wrong_link": wrong_link,
            "description": chain_data["description"],
        },
        token_budget=512,
        source="causal_surgery_generator",
    )


causal_surgery_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CausalSurgeryVerifier(Verifier):
    """Verify a causal surgery: broken link identified + correct link provided.

    reward = broken_link_correct * 0.5 + correct_link_correct * 0.5
    """

    def __init__(self, broken_link: tuple, correct_link: tuple):
        super().__init__()
        self._broken_from = _normalize_node(broken_link[0])
        self._broken_to = _normalize_node(broken_link[1])
        self._correct_from = _normalize_node(correct_link[0])
        self._correct_to = _normalize_node(correct_link[1])

    def _parse_link(self, text: str) -> Optional[tuple[str, str]]:
        match = re.search(r"(\w+)\s*->\s*(\w+)", text)
        if match:
            return (_normalize_node(match.group(1)), _normalize_node(match.group(2)))
        return None

    def verify(self, response: str) -> VerifierResult:
        # Parse BROKEN_LINK
        broken_match = re.search(r"BROKEN_LINK\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        broken_link = self._parse_link(broken_match.group(1)) if broken_match else None

        # Parse FIX
        fix_match = re.search(r"FIX\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        fix_link = self._parse_link(fix_match.group(1)) if fix_match else None

        # Check broken link: the "from" must match, and the "to" must be the wrong one
        broken_correct = False
        if broken_link:
            broken_correct = (
                broken_link[0] == self._broken_from
                and broken_link[1] != self._correct_to
            )

        # Check fix link: must match the correct link
        fix_correct = False
        if fix_link:
            fix_correct = (
                fix_link[0] == self._correct_from
                and fix_link[1] == self._correct_to
            )

        score = (1.0 if broken_correct else 0.0) * 0.5 + (1.0 if fix_correct else 0.0) * 0.5
        correct = broken_correct and fix_correct

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "broken_link_correct": 1.0 if broken_correct else 0.0,
                "fix_correct": 1.0 if fix_correct else 0.0,
            },
            diagnostics=(
                f"broken={broken_link} (expected from={self._broken_from}) "
                f"fix={fix_link} (expected={self._correct_from}->{self._correct_to})"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CausalSurgeryEnv(BatchEnvBase):
    """CausalSurgery: repair broken causal chains.

    Batch-aware: N parallel attempts; reward = best repair.
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
            problem_generator = causal_surgery_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CausalSurgeryVerifier(
            broken_link=problem.metadata["broken_link"],
            correct_link=problem.metadata["correct_link"],
        )

    def _check_format(self, response: str) -> float:
        has_broken = bool(re.search(r"BROKEN_LINK\s*:", response, re.IGNORECASE))
        has_fix = bool(re.search(r"FIX\s*:", response, re.IGNORECASE))
        if has_broken and has_fix:
            return 1.0
        if has_broken or has_fix:
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
        broken_match = re.search(r"BROKEN_LINK\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        fix_match = re.search(r"FIX\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        parts = []
        if broken_match:
            parts.append(f"BROKEN_LINK: {broken_match.group(1).strip()}")
        if fix_match:
            parts.append(f"FIX: {fix_match.group(1).strip()}")
        return " | ".join(parts) if parts else response
