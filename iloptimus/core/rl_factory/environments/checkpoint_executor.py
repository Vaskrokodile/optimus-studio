"""
CheckpointExecutor: Execute checkpoints one by one with self-audit.

Environment concept:
  The model receives a task with predefined checkpoints. It must:
    1. Execute each checkpoint in order
    2. After each checkpoint, AUDIT its own work (did it pass?)
    3. Only move to the next checkpoint if the audit passes
    4. If the audit fails, fix the issue before moving on

  The response format is:
    CHECKPOINT 1: <name>
    WORK: <the work done for this checkpoint>
    AUDIT: <self-audit result — PASS or FAIL with reason>
    [FIX: <fix if audit failed>]
    RE-AUDIT: <result after fix>
    ---
    CHECKPOINT 2: <name>
    ...

  This is a batch environment: N parallel executions are scored, and the
  best one (most checkpoints passed + best audit quality) wins.

  reward = checkpoints_passed * 0.5 + audit_accuracy * 0.3 + best_score * 0.2

  This trains the model to:
    1. Work in verifiable increments (not one big blob)
    2. Self-audit after each increment (catch errors early)
    3. Fix errors before proceeding (not ignore them)
    4. Be honest in self-audits (not always say "PASS")
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult
from iloptimus.core.rl_factory.data.checkpoint_traces import get_checkpoint_problem


def checkpoint_executor_generator(seed: int) -> Problem:
    """Generate a CheckpointExecutor problem."""
    rng = random.Random(seed)
    cp_problem = get_checkpoint_problem(rng)

    checkpoints = cp_problem["checkpoints"]
    # Format checkpoints for the prompt
    cp_text = []
    for i, cp in enumerate(checkpoints):
        cp_text.append(
            f"CHECKPOINT {i+1}: {cp['name']}\n"
            f"  CRITERIA: {cp['success_criteria']}"
        )

    return Problem(
        id=f"checkpoint_executor_{rng.randint(0, 99999)}",
        prompt=_format_prompt(cp_problem["prompt"], "\n".join(cp_text), len(checkpoints)),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "checkpoint_executor",
            "task": cp_problem["prompt"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
            "source": cp_problem.get("source", "unknown"),
        },
        token_budget=1500,
        source="generated",
    )


def _format_prompt(task: str, checkpoints_str: str, num_cps: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"Checkpoints:\n{checkpoints_str}\n\n"
        f"Execute each checkpoint in order. After each, AUDIT your work.\n\n"
        f"Format:\n"
        f"CHECKPOINT <n>: <name>\n"
        f"WORK: <your work>\n"
        f"AUDIT: PASS or FAIL - <reason>\n"
        f"[If FAIL: FIX: <fix> / RE-AUDIT: <result>]\n"
        f"---\n\n"
        f"Rules:\n"
        f"  - Do ALL {num_cps} checkpoints\n"
        f"  - Audit honestly — don't always say PASS\n"
        f"  - If audit fails, fix before moving on\n"
        f"  - Be concise in work and audit"
    )


class CheckpointExecutorVerifier(Verifier):
    """Verifies a single checkpoint execution response."""

    def __init__(self, checkpoints: list[dict]):
        super().__init__()
        self._checkpoints = checkpoints

    def verify(self, response: str) -> VerifierResult:
        # Parse the executed checkpoints
        executed = self._parse_execution(response)

        if not executed:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No checkpoint executions found",
            )

        total_cps = len(self._checkpoints)
        cps_passed = 0
        audit_honest = 0
        audit_total = 0

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            audit_total += 1

            # Check if the work addresses the checkpoint's keywords
            work_text = exe_cp.get("work", "").lower()
            audit_keywords = exp_cp.get("audit_keywords", [])
            keyword_matches = sum(1 for kw in audit_keywords if kw.lower() in work_text)
            keyword_score = keyword_matches / max(len(audit_keywords), 1)

            # Did the model's audit match reality?
            model_audit = exe_cp.get("audit_result", "unknown")
            actual_pass = keyword_score >= 0.2  # heuristic: if keywords match, it probably passed

            if actual_pass:
                cps_passed += 1

            # Audit honesty: did the model say PASS when it should, FAIL when it should?
            if (actual_pass and "pass" in model_audit.lower()) or \
               (not actual_pass and "fail" in model_audit.lower()):
                audit_honest += 1

        checkpoint_score = cps_passed / total_cps if total_cps > 0 else 0.0
        audit_accuracy = audit_honest / audit_total if audit_total > 0 else 0.0

        # Overall score
        score = checkpoint_score * 0.5 + audit_accuracy * 0.3 + min(1.0, len(executed) / total_cps) * 0.2
        correct = checkpoint_score >= 0.6 and audit_accuracy >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "checkpoints_passed": cps_passed,
                "total_checkpoints": total_cps,
                "checkpoint_score": checkpoint_score,
                "audit_accuracy": audit_accuracy,
                "executed_count": len(executed),
            },
            diagnostics=f"CPs passed: {cps_passed}/{total_cps} Audit accuracy: {audit_accuracy:.0%} Executed: {len(executed)}/{total_cps}",
        )

    def _parse_execution(self, response: str) -> list[dict]:
        """Parse CHECKPOINT n / WORK / AUDIT blocks."""
        executed = []
        # Split by CHECKPOINT markers
        blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE)

        for block in blocks[1:]:
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""
            work = ""
            audit = ""
            audit_result = "unknown"
            fix = ""

            for line in lines[1:]:
                work_match = re.match(r"\s*WORK:\s*(.+)", line, re.IGNORECASE)
                audit_match = re.match(r"\s*AUDIT:\s*(.+)", line, re.IGNORECASE)
                fix_match = re.match(r"\s*FIX:\s*(.+)", line, re.IGNORECASE)

                if work_match:
                    work = work_match.group(1).strip()
                elif audit_match:
                    audit = audit_match.group(1).strip()
                    if "pass" in audit.lower():
                        audit_result = "pass"
                    elif "fail" in audit.lower():
                        audit_result = "fail"
                elif fix_match:
                    fix = fix_match.group(1).strip()

            if name:
                executed.append({
                    "name": name,
                    "work": work,
                    "audit": audit,
                    "audit_result": audit_result,
                    "fix": fix,
                })

        return executed


class CheckpointExecutorEnv(BatchEnvBase):
    """
    CheckpointExecutor environment: execute checkpoints with self-audit.

    Batch-aware: N parallel executions, reward = best execution.
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = checkpoint_executor_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CheckpointExecutorVerifier(checkpoints=problem.metadata["checkpoints"])

    def _check_format(self, response: str) -> float:
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE) and "AUDIT:" in response.upper():
            return 1.0
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0

        # Extract checkpoint pass counts from diagnostics
        best_cp_passed = 0
        best_audit = 0.0
        for s in per_sample:
            pc = s.get("verifier_diagnostics", "")
            if s["verifier_score"] == best_score:
                # Parse from partial_credit if available (not in per_sample)
                break

        any_correct = any(corrects)

        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score  # Already includes checkpoint + audit components

        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Mean={sum(scores)/len(scores) if scores else 0:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
