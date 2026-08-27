"""
CheckpointRecovery: Fix a failed checkpoint without breaking others (batch).

Environment concept:
  The model receives a task with checkpoints, where ONE checkpoint has
  FAILED (the work is incorrect). The model must:
    1. Identify which checkpoint failed
    2. Fix ONLY that checkpoint's work
    3. NOT break any other checkpoint (the fix must be localized)
    4. Re-audit to confirm the fix works

  This is a batch environment: N parallel fix attempts are scored.
  The reward measures:
    - Fix correctness: did the fix address the failed checkpoint?
    - Non-regression: did the fix NOT break other checkpoints?
    - Minimality: was the fix small and targeted (not a rewrite)?

  reward = fix_correct * 0.5 + non_regression * 0.3 + minimality * 0.2

  This trains the model to:
    1. Make SURGICAL fixes (not rewrite everything)
    2. Verify that fixes don't introduce regressions
    3. Target the specific failure, not symptoms
    4. Re-audit after fixing

  The key insight: this is the most practical agentic skill. When a test
  fails, the model must fix the specific issue without breaking everything
  else. Models that rewrite entire files on every bug are 10x more expensive
  than models that make surgical fixes.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult
from iloptimus.core.rl_factory.data.checkpoint_traces import get_checkpoint_problem


def checkpoint_recovery_generator(seed: int) -> Problem:
    """Generate a CheckpointRecovery problem."""
    rng = random.Random(seed)
    cp_problem = get_checkpoint_problem(rng)

    checkpoints = cp_problem["checkpoints"]
    if len(checkpoints) < 2:
        # Need at least 2 checkpoints for this to make sense
        cp_problem = get_synthetic_for_recovery(rng)
        checkpoints = cp_problem["checkpoints"]

    # Choose which checkpoint is broken
    broken_idx = rng.randint(0, len(checkpoints) - 1)

    # Build the "broken solution"
    solution_parts = []
    for i, cp in enumerate(checkpoints):
        if i == broken_idx:
            solution_parts.append(f"CHECKPOINT {i+1}: {cp['name']}\nWORK: <BROKEN: incorrect work>")
        else:
            solution_parts.append(f"CHECKPOINT {i+1}: {cp['name']}\nWORK: {cp['content']}")

    solution_text = "\n---\n".join(solution_parts)
    broken_cp = checkpoints[broken_idx]

    return Problem(
        id=f"checkpoint_recovery_{rng.randint(0, 99999)}",
        prompt=_format_prompt(cp_problem["prompt"], solution_text, broken_idx + 1, broken_cp),
        difficulty=0.35 + 0.1 * rng.random(),
        metadata={
            "type": "checkpoint_recovery",
            "task": cp_problem["prompt"],
            "checkpoints": checkpoints,
            "broken_idx": broken_idx,
            "broken_cp": broken_cp,
            "num_checkpoints": len(checkpoints),
            "source": cp_problem.get("source", "unknown"),
        },
        token_budget=800,
        source="generated",
    )


def get_synthetic_for_recovery(rng: random.Random) -> dict:
    """Get a synthetic problem with enough checkpoints for recovery."""
    from iloptimus.core.rl_factory.data.checkpoint_traces import get_synthetic_checkpoint_problem
    problem = get_synthetic_checkpoint_problem(rng)
    # Ensure at least 2 checkpoints
    if len(problem["checkpoints"]) < 2:
        problem["checkpoints"].append({
            "name": "Final verification",
            "content": "Verify the solution is complete and correct.",
            "success_criteria": "Must verify completeness",
            "audit_keywords": ["verify", "complete", "correct"],
            "status": "passed",
        })
    return problem


def _format_prompt(task: str, solution: str, broken_num: int, broken_cp: dict) -> str:
    return (
        f"Task: {task}\n\n"
        f"A solution has a bug in checkpoint {broken_num}. Fix it.\n\n"
        f"Solution:\n{solution}\n\n"
        f"The broken checkpoint is: CHECKPOINT {broken_num}: {broken_cp['name']}\n"
        f"Expected criteria: {broken_cp['success_criteria']}\n\n"
        f"Output:\n"
        f"  IDENTIFY: <which checkpoint is broken and why>\n"
        f"  FIX: <your fix for the broken checkpoint>\n"
        f"  NON_REGRESSION: <confirm other checkpoints still work>\n"
        f"  RE-AUDIT: <audit the fix>\n\n"
        f"Rules:\n"
        f"  - Fix ONLY the broken checkpoint\n"
        f"  - Do NOT rewrite other checkpoints\n"
        f"  - Verify your fix doesn't break anything else\n"
        f"  - Be concise and surgical"
    )


class CheckpointRecoveryVerifier(Verifier):
    """Verifies a checkpoint recovery response."""

    def __init__(self, checkpoints: list[dict], broken_idx: int, broken_cp: dict):
        super().__init__()
        self._checkpoints = checkpoints
        self._broken_idx = broken_idx
        self._broken_cp = broken_cp

    def verify(self, response: str) -> VerifierResult:
        # Parse the response
        identify = self._extract_section(response, "IDENTIFY")
        fix = self._extract_section(response, "FIX")
        non_regression = self._extract_section(response, "NON_REGRESSION")
        re_audit = self._extract_section(response, "RE-AUDIT")

        if not fix:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No FIX section found",
            )

        # Check fix correctness: does the fix address the broken checkpoint's keywords?
        fix_lower = fix.lower()
        audit_keywords = self._broken_cp.get("audit_keywords", [])
        kw_matches = sum(1 for kw in audit_keywords if kw.lower() in fix_lower)
        fix_correct = kw_matches / max(len(audit_keywords), 1) >= 0.2

        # Check identification: did the model correctly identify the broken checkpoint?
        identify_correct = str(self._broken_idx + 1) in identify or \
                          self._broken_cp["name"].lower() in identify.lower()

        # Non-regression: did the model mention other checkpoints?
        non_regression_ok = len(non_regression) > 10  # at least some acknowledgment

        # Minimality: is the fix short? (shorter = more surgical)
        fix_tokens = estimate_tokens(fix)
        if fix_tokens < 50:
            minimality = 1.0
        elif fix_tokens < 100:
            minimality = 0.7
        elif fix_tokens < 200:
            minimality = 0.4
        else:
            minimality = 0.2

        # Re-audit: did the model re-audit?
        re_audit_ok = len(re_audit) > 5

        # Score
        fix_score = 1.0 if fix_correct else 0.0
        non_reg_score = 1.0 if non_regression_ok else 0.5
        min_score = minimality

        score = fix_score * 0.5 + non_reg_score * 0.3 + min_score * 0.2
        correct = fix_correct and non_regression_ok

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "fix_correct": fix_correct,
                "identify_correct": identify_correct,
                "non_regression": non_reg_score,
                "minimality": minimality,
                "re_audit": re_audit_ok,
                "fix_tokens": fix_tokens,
            },
            diagnostics=f"Fix={'OK' if fix_correct else 'FAIL'} Identify={'OK' if identify_correct else 'FAIL'} NonReg={'OK' if non_regression_ok else 'MISS'} Minimality={minimality:.0%} Tokens={fix_tokens}",
        )

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a named section from the response."""
        pattern = rf"{section_name}\s*:\s*(.*?)(?=\n\s*[A-Z][-A-Z]+:|\Z)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""


class CheckpointRecoveryEnv(BatchEnvBase):
    """
    CheckpointRecovery environment: fix a failed checkpoint without regressions.

    Batch-aware: N parallel fix attempts, reward = best fix.
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = checkpoint_recovery_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CheckpointRecoveryVerifier(
            checkpoints=problem.metadata["checkpoints"],
            broken_idx=problem.metadata["broken_idx"],
            broken_cp=problem.metadata["broken_cp"],
        )

    def _check_format(self, response: str) -> float:
        if "FIX:" in response.upper():
            return 1.0
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
            "reward": best_score if best_score > 0 else 0.0,
            "correct": any_correct,
            "diagnostics": f"Best fix score={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
