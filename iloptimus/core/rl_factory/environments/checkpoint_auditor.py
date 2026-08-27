"""
CheckpointAuditor: Audit a solution against checkpoints (batch).

Environment concept:
  The model receives a task, a set of checkpoints, and a SOLUTION (produced
  by another model or a previous round). It must AUDIT the solution:
    1. For each checkpoint, determine if it PASS or FAIL
    2. For FAIL, explain what's wrong
    3. For PASS, confirm what's correct

  This is a batch environment: N parallel audits are scored, and the most
  accurate audit wins. The reward measures:
    - Audit accuracy: did the model correctly identify pass/fail for each CP?
    - Explanation quality: for failures, did it explain what's wrong?
    - Honesty: did it avoid false positives (saying PASS when it should FAIL)?

  reward = audit_accuracy * 0.6 + explanation_quality * 0.2 + honesty_bonus * 0.2

  This trains the model to:
    1. Read solutions critically (not passively)
    2. Check each sub-goal independently
    3. Be honest about failures (not rubber-stamp everything)
    4. Provide actionable feedback (not vague "looks good")

  The key insight: a model that can audit OTHERS' solutions well can audit
  ITS OWN solutions — the self-monitoring skill that prevents error cascades.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult
from iloptimus.core.rl_factory.data.checkpoint_traces import get_checkpoint_problem


def checkpoint_auditor_generator(seed: int) -> Problem:
    """Generate a CheckpointAuditor problem."""
    rng = random.Random(seed)
    cp_problem = get_checkpoint_problem(rng)

    checkpoints = cp_problem["checkpoints"]

    # Generate a "solution" to audit — sometimes correct, sometimes with bugs
    # We'll create a solution that passes some checkpoints and fails others
    solution_parts = []
    expected_audits = []

    bug_checkpoint = rng.randint(0, len(checkpoints) - 1) if rng.random() < 0.6 else -1

    for i, cp in enumerate(checkpoints):
        if i == bug_checkpoint:
            # Introduce a bug in this checkpoint's work
            solution_parts.append(f"CHECKPOINT {i+1}: {cp['name']}\nWORK: <incorrect work that doesn't meet criteria>")
            expected_audits.append({"checkpoint": i, "expected": "FAIL", "keywords": cp.get("audit_keywords", [])})
        else:
            solution_parts.append(f"CHECKPOINT {i+1}: {cp['name']}\nWORK: {cp['content']}")
            expected_audits.append({"checkpoint": i, "expected": "PASS", "keywords": cp.get("audit_keywords", [])})

    solution_text = "\n---\n".join(solution_parts)

    return Problem(
        id=f"checkpoint_auditor_{rng.randint(0, 99999)}",
        prompt=_format_prompt(cp_problem["prompt"], solution_text, len(checkpoints)),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "checkpoint_auditor",
            "task": cp_problem["prompt"],
            "checkpoints": checkpoints,
            "expected_audits": expected_audits,
            "num_checkpoints": len(checkpoints),
            "bug_checkpoint": bug_checkpoint,
            "source": cp_problem.get("source", "unknown"),
        },
        token_budget=800,
        source="generated",
    )


def _format_prompt(task: str, solution: str, num_cps: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"A solution has been submitted. Audit it checkpoint by checkpoint.\n\n"
        f"Solution:\n{solution}\n\n"
        f"For each checkpoint, output:\n"
        f"  AUDIT <n>: PASS - <what's correct>\n"
        f"  OR\n"
        f"  AUDIT <n>: FAIL - <what's wrong>\n\n"
        f"Rules:\n"
        f"  - Audit ALL {num_cps} checkpoints\n"
        f"  - Be honest — don't rubber-stamp\n"
        f"  - For FAIL, explain what's specifically wrong\n"
        f"  - For PASS, confirm what's correct"
    )


class CheckpointAuditorVerifier(Verifier):
    """Verifies a checkpoint audit response."""

    def __init__(self, expected_audits: list[dict]):
        super().__init__()
        self._expected = expected_audits

    def verify(self, response: str) -> VerifierResult:
        # Parse audit results
        parsed = self._parse_audits(response)

        if not parsed:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No AUDIT entries found",
            )

        correct_audits = 0
        total = len(self._expected)
        explanations_quality = []

        for exp in self._expected:
            cp_idx = exp["checkpoint"]
            expected_result = exp["expected"]
            keywords = exp.get("keywords", [])

            # Find the model's audit for this checkpoint
            model_audit = parsed.get(cp_idx + 1, {})  # 1-indexed
            model_result = model_audit.get("result", "unknown")
            model_explanation = model_audit.get("explanation", "")

            # Check if the audit result matches
            if model_result == expected_result.lower():
                correct_audits += 1

            # Check explanation quality (keyword overlap for FAILs)
            if expected_result == "FAIL" and model_result == "fail":
                kw_matches = sum(1 for kw in keywords if kw.lower() in model_explanation.lower())
                explanations_quality.append(kw_matches / max(len(keywords), 1))
            elif expected_result == "PASS" and model_result == "pass":
                explanations_quality.append(1.0)
            else:
                explanations_quality.append(0.0)

        audit_accuracy = correct_audits / total if total > 0 else 0.0
        explanation_quality = sum(explanations_quality) / len(explanations_quality) if explanations_quality else 0.0

        # Honesty bonus: did the model avoid false positives?
        # (saying PASS when it should be FAIL)
        false_positives = 0
        for exp in self._expected:
            if exp["expected"] == "FAIL":
                model_audit = parsed.get(exp["checkpoint"] + 1, {})
                if model_audit.get("result") == "pass":
                    false_positives += 1

        expected_fails = sum(1 for exp in self._expected if exp["expected"] == "FAIL")
        honesty = 1.0 - (false_positives / max(expected_fails, 1)) if expected_fails > 0 else 1.0

        score = audit_accuracy * 0.6 + explanation_quality * 0.2 + honesty * 0.2
        correct = audit_accuracy >= 0.7

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "audit_accuracy": audit_accuracy,
                "explanation_quality": explanation_quality,
                "honesty": honesty,
                "correct_audits": correct_audits,
                "total": total,
                "false_positives": false_positives,
            },
            diagnostics=f"Accuracy={audit_accuracy:.0%} ({correct_audits}/{total}) Explanation={explanation_quality:.0%} Honesty={honesty:.0%} FP={false_positives}",
        )

    def _parse_audits(self, response: str) -> dict[int, dict]:
        """Parse AUDIT n: PASS/FAIL - explanation."""
        results = {}
        pattern = r"AUDIT\s+(\d+)\s*:\s*(PASS|FAIL)\s*[-:]?\s*(.*?)(?:\n\s*AUDIT|\Z)"
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)

        for match in matches:
            cp_num = int(match.group(1))
            result = match.group(2).lower()
            explanation = match.group(3).strip()
            results[cp_num] = {"result": result, "explanation": explanation}

        return results


class CheckpointAuditorEnv(BatchEnvBase):
    """
    CheckpointAuditor environment: audit a solution against checkpoints.

    Batch-aware: N parallel audits, reward = most accurate audit.
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = checkpoint_auditor_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CheckpointAuditorVerifier(expected_audits=problem.metadata["expected_audits"])

    def _check_format(self, response: str) -> float:
        if re.search(r"AUDIT\s+\d+\s*:\s*(?:PASS|FAIL)", response, re.IGNORECASE):
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
            "diagnostics": f"Best audit score={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
