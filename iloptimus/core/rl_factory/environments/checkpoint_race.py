"""
CheckpointRace: Race through checkpoints, reward = speed + audit quality (batch).

Environment concept:
  The model must complete ALL checkpoints as fast as possible (in terms of
  tokens used), while maintaining audit quality. This is a batch environment:
  N parallel "racers" compete, and the reward combines:
    1. Completion: did the racer finish all checkpoints?
    2. Speed: fewer tokens = faster (token efficiency)
    3. Audit quality: did the racer audit honestly at each checkpoint?
    4. Correctness: are all checkpoints actually correct?

  reward = completion * 0.3 + correctness * 0.3 + speed_bonus * 0.2 + audit_quality * 0.2

  Where:
    - completion = checkpoints_attempted / total_checkpoints
    - correctness = checkpoints_correct / total_checkpoints
    - speed_bonus = 1.0 - (tokens_used / token_budget), clamped to [0, 1]
    - audit_quality = fraction of checkpoints with honest audits

  The batch dimension adds a RACE element: the fastest correct racer wins.
  Among the N parallel attempts, the one that finishes all checkpoints
  correctly with the fewest tokens gets a bonus.

  This trains the model to:
    1. Work FAST through checkpoints (token efficiency)
    2. Still AUDIT at each checkpoint (don't skip for speed)
    3. Be CORRECT (speed without correctness = 0)
    4. Balance speed vs thoroughness (the key agentic tradeoff)

  The key insight: in production, agents have token budgets and time limits.
  A model that races through checkpoints correctly and efficiently is worth
  10x more than one that is thorough but slow. This environment trains the
  speed-thoroughness balance that production agents need.
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


def checkpoint_race_generator(seed: int) -> Problem:
    """Generate a CheckpointRace problem."""
    rng = random.Random(seed)
    cp_problem = get_checkpoint_problem(rng)

    checkpoints = cp_problem["checkpoints"]
    cp_text = []
    for i, cp in enumerate(checkpoints):
        cp_text.append(
            f"CHECKPOINT {i+1}: {cp['name']}\n"
            f"  CRITERIA: {cp['success_criteria']}"
        )

    token_budget = 200 + len(checkpoints) * 150  # tight budget for racing

    return Problem(
        id=f"checkpoint_race_{rng.randint(0, 99999)}",
        prompt=_format_prompt(cp_problem["prompt"], "\n".join(cp_text), len(checkpoints), token_budget),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "checkpoint_race",
            "task": cp_problem["prompt"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
            "source": cp_problem.get("source", "unknown"),
        },
        token_budget=token_budget,
        source="generated",
    )


def _format_prompt(task: str, checkpoints_str: str, num_cps: int, token_budget: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"Checkpoints:\n{checkpoints_str}\n\n"
        f"RACE through all {num_cps} checkpoints as fast as possible.\n"
        f"Token budget: {token_budget} tokens. Fewer tokens = higher score.\n\n"
        f"Format (be concise!):\n"
        f"CP<n>: <work> | AUDIT: <PASS/FAIL - brief reason>\n\n"
        f"Rules:\n"
        f"  - Complete ALL {num_cps} checkpoints\n"
        f"  - Audit each (briefly)\n"
        f"  - Be as concise as possible\n"
        f"  - Correctness > speed, but both matter"
    )


class CheckpointRaceVerifier(Verifier):
    """Verifies a single checkpoint race response."""

    def __init__(self, checkpoints: list[dict], token_budget: int):
        super().__init__()
        self._checkpoints = checkpoints
        self._token_budget = token_budget

    def verify(self, response: str) -> VerifierResult:
        # Parse checkpoints (compact format: CP<n>: <work> | AUDIT: <result>)
        executed = self._parse_race(response)

        total_cps = len(self._checkpoints)
        cps_attempted = len(executed)
        cps_correct = 0
        audits_honest = 0
        audit_count = 0

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            work_text = exe_cp.get("work", "").lower()
            audit_keywords = exp_cp.get("audit_keywords", [])

            kw_matches = sum(1 for kw in audit_keywords if kw.lower() in work_text)
            is_correct = kw_matches / max(len(audit_keywords), 1) >= 0.2

            if is_correct:
                cps_correct += 1

            # Check audit
            if exe_cp.get("audit_result"):
                audit_count += 1
                model_audit = exe_cp["audit_result"]
                actual = "pass" if is_correct else "fail"
                if actual in model_audit.lower():
                    audits_honest += 1

        completion = cps_attempted / total_cps if total_cps > 0 else 0.0
        correctness = cps_correct / total_cps if total_cps > 0 else 0.0
        audit_quality = audits_honest / audit_count if audit_count > 0 else 0.0

        # Speed bonus
        tokens_used = estimate_tokens(response)
        speed_bonus = max(0.0, 1.0 - (tokens_used / self._token_budget))

        # Score
        score = completion * 0.3 + correctness * 0.3 + speed_bonus * 0.2 + audit_quality * 0.2
        correct = correctness >= 0.6 and completion >= 0.8

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "completion": completion,
                "correctness": correctness,
                "audit_quality": audit_quality,
                "speed_bonus": speed_bonus,
                "tokens_used": tokens_used,
                "cps_attempted": cps_attempted,
                "cps_correct": cps_correct,
                "total_cps": total_cps,
            },
            diagnostics=f"Completion={completion:.0%} Correct={cps_correct}/{total_cps} Audit={audit_quality:.0%} Speed={speed_bonus:.0%} Tokens={tokens_used}",
        )

    def _parse_race(self, response: str) -> list[dict]:
        """Parse compact CP<n>: <work> | AUDIT: <result> format."""
        executed = []
        # Try compact format first: CP1: ... | AUDIT: ...
        pattern = r"CP\s*(\d+)\s*:\s*(.*?)(?:\s*\|\s*AUDIT\s*:\s*(.*?))?(?:\n\s*CP\s*\d+|\Z)"
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)

        for match in matches:
            cp_num = int(match.group(1))
            work = match.group(2).strip()
            audit = match.group(3).strip() if match.group(3) else ""

            audit_result = "unknown"
            if "pass" in audit.lower():
                audit_result = "pass"
            elif "fail" in audit.lower():
                audit_result = "fail"

            executed.append({
                "cp_num": cp_num,
                "work": work,
                "audit": audit,
                "audit_result": audit_result,
            })

        # Fallback: try CHECKPOINT n: format
        if not executed:
            blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, re.IGNORECASE)
            for block in blocks[1:]:
                lines = block.strip().split("\n")
                name = lines[0].strip() if lines else ""
                work = ""
                audit_result = "unknown"
                for line in lines[1:]:
                    if "AUDIT:" in line.upper():
                        if "pass" in line.lower():
                            audit_result = "pass"
                        elif "fail" in line.lower():
                            audit_result = "fail"
                    else:
                        work += line.strip() + " "
                executed.append({"cp_num": len(executed) + 1, "work": work.strip(), "audit": "", "audit_result": audit_result})

        return executed


class CheckpointRaceEnv(BatchEnvBase):
    """
    CheckpointRace environment: race through checkpoints, speed + audit quality.

    Batch-aware: N parallel racers, reward = best racer (completion + speed + audit).
    The fastest correct racer gets a bonus.
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = checkpoint_race_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CheckpointRaceVerifier(
            checkpoints=problem.metadata["checkpoints"],
            token_budget=problem.token_budget,
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"CP\s*\d+\s*:", response, re.IGNORECASE) or \
           re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        tokens = [s["tokens_used"] for s in per_sample]

        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Find the fastest correct racer
        fastest_correct_tokens = None
        for i, s in enumerate(per_sample):
            if s["correct"]:
                if fastest_correct_tokens is None or tokens[i] < fastest_correct_tokens:
                    fastest_correct_tokens = tokens[i]

        # Speed bonus: if there's a correct racer, bonus = how much faster than average
        if fastest_correct_tokens is not None:
            avg_tokens = sum(tokens) / len(tokens) if tokens else 0
            speed_bonus = max(0.0, (avg_tokens - fastest_correct_tokens) / max(avg_tokens, 1))
        else:
            speed_bonus = 0.0

        # Reward: best score + speed bonus for the fastest correct racer
        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score + speed_bonus * 0.1  # small bonus
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "fastest_correct_tokens": fastest_correct_tokens,
            "speed_bonus": speed_bonus,
            "mean_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Fastest={fastest_correct_tokens}tk SpeedBonus={speed_bonus:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
