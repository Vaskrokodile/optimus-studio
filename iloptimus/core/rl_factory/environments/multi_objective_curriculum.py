"""
MultiObjectiveCurriculum: Multi-reward adaptive curriculum.

Environment concept:
  This implements the *multi-objective curriculum* paradigm. The model
  is given a problem that tests MULTIPLE capabilities simultaneously —
  e.g. correctness, conciseness, and clarity. The model must optimize
  ALL objectives, not just correctness. This trains the model to
  produce answers that are both right AND well-communicated.

  The generator creates problems with multiple objectives (each with a
  name, type, and target). The verifier checks each objective
  separately and combines them into a single reward.

Verification:
  - correctness: is the answer numerically correct?
  - conciseness: is the response within a token budget (fewer = better)?
  - clarity: does the response have good structure (keywords, steps)?

Reward:
  reward = correctness * 0.5 + conciseness * 0.25 + clarity * 0.25
  correct = correctness >= 0.5 AND clarity >= 0.5

Format:
  ANSWER: <answer>
  EXPLANATION: <text>
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: each has a problem, answer, and objectives.
# ---------------------------------------------------------------------------


_MO_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem": "What is 17 * 23?",
        "answer": "391",
        "answer_int": 391,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 391},
            {"name": "conciseness", "type": "token_budget", "target": 80},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 144 / 8?",
        "answer": "18",
        "answer_int": 18,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 18},
            {"name": "conciseness", "type": "token_budget", "target": 60},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 7^3?",
        "answer": "343",
        "answer_int": 343,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 343},
            {"name": "conciseness", "type": "token_budget", "target": 90},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 15% of 200?",
        "answer": "30",
        "answer_int": 30,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 30},
            {"name": "conciseness", "type": "token_budget", "target": 70},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is sqrt(169)?",
        "answer": "13",
        "answer_int": 13,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 13},
            {"name": "conciseness", "type": "token_budget", "target": 50},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 12 * 15?",
        "answer": "180",
        "answer_int": 180,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 180},
            {"name": "conciseness", "type": "token_budget", "target": 80},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 9 * 13?",
        "answer": "117",
        "answer_int": 117,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 117},
            {"name": "conciseness", "type": "token_budget", "target": 80},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
    {
        "problem": "What is 100 - 37?",
        "answer": "63",
        "answer_int": 63,
        "objectives": [
            {"name": "correctness", "type": "numeric", "target": 63},
            {"name": "conciseness", "type": "token_budget", "target": 50},
            {"name": "clarity", "type": "structure", "target": "has_explanation"},
        ],
    },
]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def multi_objective_curriculum_generator(seed: int) -> Problem:
    """Generate a MultiObjectiveCurriculum problem.

    Selects a problem template with multiple objectives. The model must
    optimize all objectives: correctness, conciseness, and clarity.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with objectives metadata.
    """
    rng = random.Random(seed)
    template = rng.choice(_MO_PROBLEMS)

    objectives = template["objectives"]
    obj_descs = "\n".join(
        f"  - {o['name']}: {o['type']} (target: {o['target']})"
        for o in objectives
    )

    prompt = (
        f"Problem: {template['problem']}\n\n"
        f"You must optimize ALL of the following objectives:\n"
        f"{obj_descs}\n\n"
        f"Format:\n"
        f"ANSWER: <answer>\n"
        f"EXPLANATION: <concise explanation of your approach>"
    )

    return Problem(
        id=f"multi_obj_{seed}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=0.4,
        metadata={
            "type": "multi_objective_curriculum",
            "problem": template["problem"],
            "answer": template["answer"],
            "answer_int": template["answer_int"],
            "objectives": objectives,
            "n_objectives": len(objectives),
        },
        token_budget=300,
        source="multi_objective_curriculum_generator",
    )


multi_objective_curriculum_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class MultiObjectiveVerifier(Verifier):
    """Verify a multi-objective response.

    Checks each objective separately and combines them.

    Args:
        answer_int: The correct numeric answer.
        objectives: List of objective dicts.
    """

    def __init__(self, answer_int: int, objectives: list[dict]):
        super().__init__()
        self._answer_int = answer_int
        self._objectives = objectives

    def verify(self, response: str) -> VerifierResult:
        answer_text = self._parse_answer(response)
        explanation = self._parse_explanation(response)

        # Gate all scoring on having the ANSWER format marker
        if not answer_text:
            return VerifierResult(
                correct=False,
                score=0.0,
                partial_credit={o["name"]: 0.0 for o in self._objectives},
                diagnostics="No ANSWER field found",
            )

        scores: dict[str, float] = {}
        for obj in self._objectives:
            name = obj["name"]
            otype = obj["type"]
            target = obj["target"]
            if otype == "numeric":
                scores[name] = self._check_correctness(answer_text, target)
            elif otype == "token_budget":
                scores[name] = self._check_conciseness(response, target)
            elif otype == "structure":
                scores[name] = self._check_clarity(explanation)
            else:
                scores[name] = 0.0

        correctness = scores.get("correctness", 0.0)
        conciseness = scores.get("conciseness", 0.0)
        clarity = scores.get("clarity", 0.0)

        score = correctness * 0.5 + conciseness * 0.25 + clarity * 0.25
        correct = correctness >= 0.5 and clarity >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit=scores,
            diagnostics=(
                f"Correct={correctness:.2f} Concise={conciseness:.2f} "
                f"Clear={clarity:.2f}"
            ),
            metadata={"answer_text": answer_text, "scores": scores},
        )

    def _parse_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_explanation(self, response: str) -> str:
        match = re.search(
            r"EXPLANATION\s*:\s*(.+?)(?:\n[A-Z]+:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _check_correctness(self, answer_text: str, target: int) -> float:
        nums = re.findall(r"-?\d+", answer_text)
        if not nums:
            return 0.0
        given = int(nums[-1])
        if given == target:
            return 1.0
        diff = abs(given - target)
        if target != 0:
            rel = diff / abs(target)
        else:
            rel = 1.0 if diff != 0 else 0.0
        return max(0.0, min(0.5, 1.0 - rel))

    def _check_conciseness(self, response: str, budget: int) -> float:
        tokens = _estimate_tokens(response)
        if tokens <= budget:
            return 1.0
        # Linear decay: 0 score at 2x budget
        excess = (tokens - budget) / budget
        return max(0.0, 1.0 - excess)

    def _check_clarity(self, explanation: str) -> float:
        if not explanation:
            return 0.0
        words = len(explanation.split())
        score = 0.0
        # Has reasonable length
        if words >= 5:
            score += 0.4
        elif words >= 2:
            score += 0.2
        # Has structure keywords
        if re.search(r"(because|since|therefore|so|thus|first|then|step)", explanation, re.IGNORECASE):
            score += 0.3
        # Has an equation or operation
        if re.search(r"[+\-*/=^]|sqrt|percent", explanation, re.IGNORECASE):
            score += 0.3
        return min(1.0, score)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MultiObjectiveCurriculumEnv(BatchEnvBase):
    """MultiObjectiveCurriculum environment: optimize multiple objectives.

    Batch-aware: N parallel attempts; reward = best attempt.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = multi_objective_curriculum_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return MultiObjectiveVerifier(
            answer_int=meta["answer_int"],
            objectives=meta["objectives"],
        )

    def _check_format(self, response: str) -> float:
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        has_explanation = bool(re.search(r"EXPLANATION\s*:", response, re.IGNORECASE))
        if has_answer and has_explanation:
            return 1.0
        if has_answer or has_explanation:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "mean_score": mean_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "reward": best_score,
            "correct": any_correct and best_score >= 0.5,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
