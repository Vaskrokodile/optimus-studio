"""
CheckpointPlanner: Split a task into checkpoints with success criteria.

Environment concept:
  The model receives a complex task and must decompose it into a sequence
  of checkpoints. Each checkpoint must have:
    1. A name (short label)
    2. Success criteria (what must be true to pass this checkpoint)
    3. An audit method (how to verify the checkpoint passed)

  This trains the model to think in terms of verifiable sub-goals rather
  than monolithic solutions. The checkpoint decomposition is the FIRST
  step of any complex reasoning task — if the plan is wrong, execution
  will fail.

  This is a single-turn environment (not batch) because planning happens
  before execution. But it produces the checkpoint structure that the
  batch environments (22-25) consume.

Reward design:
  - Coverage: fraction of expected checkpoints that have a match
  - Quality: does each checkpoint have clear success criteria?
  - Granularity: not too few (under-decomposed) or too many (over-decomposed)
  - reward = coverage * 0.5 + quality * 0.3 + granularity * 0.2
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult
from iloptimus.core.rl_factory.data.checkpoint_traces import get_checkpoint_problem


def checkpoint_planner_generator(seed: int) -> Problem:
    """Generate a CheckpointPlanner problem from real or synthetic traces."""
    rng = random.Random(seed)
    cp_problem = get_checkpoint_problem(rng)

    expected_count = len(cp_problem["checkpoints"])
    expected_names = [cp["name"] for cp in cp_problem["checkpoints"]]

    return Problem(
        id=f"checkpoint_planner_{rng.randint(0, 99999)}",
        prompt=_format_prompt(cp_problem["prompt"], expected_count),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "checkpoint_planner",
            "task": cp_problem["prompt"],
            "expected_count": expected_count,
            "expected_names": expected_names,
            "expected_checkpoints": cp_problem["checkpoints"],
            "source": cp_problem.get("source", "unknown"),
        },
        token_budget=600,
        source="generated",
    )


def _format_prompt(task: str, expected_count: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"Decompose this task into {expected_count} checkpoints.\n"
        f"For each checkpoint, provide:\n"
        f"  CHECKPOINT <n>: <name>\n"
        f"  CRITERIA: <what must be true to pass>\n"
        f"  AUDIT: <how to verify it passed>\n\n"
        f"Rules:\n"
        f"  - Each checkpoint should be a distinct, verifiable sub-goal\n"
        f"  - Checkpoints should be ordered (each builds on the previous)\n"
        f"  - Success criteria must be concrete and testable\n"
        f"  - Do NOT over-decompose (keep it to ~{expected_count} checkpoints)"
    )


class CheckpointPlannerVerifier(Verifier):
    """Verifies a checkpoint plan against expected checkpoints."""

    def __init__(self, expected_checkpoints: list[dict], expected_count: int):
        super().__init__()
        self._expected = expected_checkpoints
        self._expected_count = expected_count

    def verify(self, response: str) -> VerifierResult:
        # Parse checkpoints from the response
        parsed = self._parse_checkpoints(response)

        if not parsed:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No checkpoints found in response",
            )

        # Match parsed checkpoints to expected
        matched = 0
        for exp_cp in self._expected:
            for p_cp in parsed:
                if self._keyword_overlap(exp_cp["name"] + " " + exp_cp.get("content", ""),
                                        p_cp["name"] + " " + p_cp.get("criteria", "")) > 0.2:
                    matched += 1
                    break

        coverage = matched / len(self._expected) if self._expected else 1.0

        # Quality: does each checkpoint have criteria and audit?
        quality_scores = []
        for p_cp in parsed:
            q = 0.0
            if p_cp.get("criteria"):
                q += 0.5
            if p_cp.get("audit"):
                q += 0.5
            quality_scores.append(q)
        quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Granularity: how close to expected count?
        actual_count = len(parsed)
        if actual_count == 0:
            granularity = 0.0
        else:
            ratio = actual_count / self._expected_count
            if 0.5 <= ratio <= 2.0:
                granularity = 1.0
            elif 0.3 <= ratio <= 3.0:
                granularity = 0.5
            else:
                granularity = 0.2

        score = coverage * 0.5 + quality * 0.3 + granularity * 0.2
        correct = score >= 0.6

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "coverage": coverage,
                "quality": quality,
                "granularity": granularity,
                "actual_count": actual_count,
                "expected_count": self._expected_count,
            },
            diagnostics=f"Coverage={coverage:.0%} Quality={quality:.0%} Granularity={granularity:.0%} Count={actual_count}/{self._expected_count}",
        )

    def _parse_checkpoints(self, response: str) -> list[dict]:
        """Parse CHECKPOINT n: name / CRITERIA: ... / AUDIT: ... blocks."""
        checkpoints = []
        # Split by CHECKPOINT markers
        blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE)

        for block in blocks[1:]:  # skip the part before the first CHECKPOINT
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""

            criteria = ""
            audit = ""
            for line in lines[1:]:
                crit_match = re.match(r"\s*CRITERIA:\s*(.+)", line, re.IGNORECASE)
                audit_match = re.match(r"\s*AUDIT:\s*(.+)", line, re.IGNORECASE)
                if crit_match:
                    criteria = crit_match.group(1).strip()
                elif audit_match:
                    audit = audit_match.group(1).strip()

            if name:
                checkpoints.append({"name": name, "criteria": criteria, "audit": audit})

        return checkpoints

    def _keyword_overlap(self, text1: str, text2: str) -> float:
        def extract_kw(s: str) -> set[str]:
            words = re.findall(r"\b[a-z]{3,}\b", s.lower())
            common = {"the", "and", "for", "with", "that", "this", "from", "into", "then", "must"}
            return set(w for w in words if w not in common)

        kw1 = extract_kw(text1)
        kw2 = extract_kw(text2)
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / len(kw1 | kw2)


class CheckpointPlannerEnv(BatchEnvBase):
    """CheckpointPlanner environment: decompose a task into checkpoints.

    Batch-aware: N parallel plans are scored, and the best plan wins.
    This enables the model to explore different decompositions and pick
    the best one — important because there are often multiple valid ways
    to decompose a task.
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = checkpoint_planner_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CheckpointPlannerVerifier(
            expected_checkpoints=problem.metadata["expected_checkpoints"],
            expected_count=problem.metadata["expected_count"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Diversity: different decompositions are valuable
        import re as _re
        plans = set()
        for s in per_sample:
            resp = s.get("response", "")
            cps = _re.findall(r"CHECKPOINT\s+\d+\s*:\s*(.+?)(?:\n|$)", resp, _re.IGNORECASE)
            plans.add(tuple(c.lower().strip() for c in cps))
        diversity = len(plans) / len(per_sample) if per_sample else 0.0

        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.8 + diversity * 0.2
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Diversity={diversity:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
