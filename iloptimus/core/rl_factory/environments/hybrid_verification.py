"""
HybridVerification: Rule-based hard constraints + LLM soft constraints.

Environment concept:
  The model is given a problem with both *hard* constraints (exact,
  rule-checkable requirements, e.g. "answer must be a prime number > 10")
  and *soft* constraints (quality / style requirements, e.g. "explain
  your reasoning clearly"). The model must satisfy BOTH to receive full
  reward.

  This implements the *hybrid verification* training paradigm: hard
  constraints are verified by deterministic rules (no reward hacking),
  while soft constraints are scored by lightweight heuristics
  (length, structure, keywords). Combining the two trains the model to
  produce answers that are both correct AND well-explained.

Verification:
  1. Hard constraints: checked by deterministic rules (is the number
     prime? is it > 10? etc.). All hard constraints must pass.
  2. Soft constraints: scored by heuristics (length, structure,
     keywords). Partial credit is awarded.

Reward:
  reward = hard_score * 0.6 + soft_score * 0.4
  correct = (all hard constraints satisfied) AND (soft_score >= 0.5)

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
# Problem templates: each has a problem, hard constraints, soft constraints,
# and a correct answer.
# ---------------------------------------------------------------------------


_HV_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem": "Give a prime number greater than 10.",
        "hard_constraints": [
            {"desc": "answer must be a prime number", "check": "is_prime"},
            {"desc": "answer must be greater than 10", "check": "gt_10"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention primality", "check": "mentions_prime"},
        ],
        "correct_answer": "11",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give an even number between 20 and 30.",
        "hard_constraints": [
            {"desc": "answer must be even", "check": "is_even"},
            {"desc": "answer must be between 20 and 30", "check": "between_20_30"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention evenness", "check": "mentions_even"},
        ],
        "correct_answer": "22",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give a perfect square greater than 50.",
        "hard_constraints": [
            {"desc": "answer must be a perfect square", "check": "is_square"},
            {"desc": "answer must be greater than 50", "check": "gt_50"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention square root", "check": "mentions_sqrt"},
        ],
        "correct_answer": "64",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give a multiple of 7 greater than 40.",
        "hard_constraints": [
            {"desc": "answer must be a multiple of 7", "check": "mult_7"},
            {"desc": "answer must be greater than 40", "check": "gt_40"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention divisibility", "check": "mentions_div"},
        ],
        "correct_answer": "42",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give a Fibonacci number greater than 20.",
        "hard_constraints": [
            {"desc": "answer must be a Fibonacci number", "check": "is_fib"},
            {"desc": "answer must be greater than 20", "check": "gt_20"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention the sequence", "check": "mentions_sequence"},
        ],
        "correct_answer": "21",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give a power of 2 greater than 100.",
        "hard_constraints": [
            {"desc": "answer must be a power of 2", "check": "is_pow2"},
            {"desc": "answer must be greater than 100", "check": "gt_100"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention exponent", "check": "mentions_exponent"},
        ],
        "correct_answer": "128",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give an odd number divisible by 3.",
        "hard_constraints": [
            {"desc": "answer must be odd", "check": "is_odd"},
            {"desc": "answer must be divisible by 3", "check": "div_3"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention divisibility", "check": "mentions_div"},
        ],
        "correct_answer": "9",
        "constraint_type": "numeric",
    },
    {
        "problem": "Give a palindrome number greater than 100.",
        "hard_constraints": [
            {"desc": "answer must be a palindrome", "check": "is_palindrome"},
            {"desc": "answer must be greater than 100", "check": "gt_100"},
        ],
        "soft_constraints": [
            {"desc": "explain your reasoning clearly", "check": "has_explanation"},
            {"desc": "mention palindrome", "check": "mentions_palindrome"},
        ],
        "correct_answer": "101",
        "constraint_type": "numeric",
    },
]


# ---------------------------------------------------------------------------
# Hard-constraint check functions
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _is_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(round(n ** 0.5))
    return r * r == n


def _is_pow2(n: int) -> bool:
    if n < 1:
        return False
    return (n & (n - 1)) == 0


_FIB_SET = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597}


def _is_fib(n: int) -> bool:
    # Extend the set for larger values
    a, b = 1, 2
    fibs = set()
    while a <= 100000:
        fibs.add(a)
        a, b = b, a + b
    return n in fibs


def _is_palindrome(n: int) -> bool:
    s = str(n)
    return s == s[::-1]


_HARD_CHECKS = {
    "is_prime": lambda n: _is_prime(n),
    "gt_10": lambda n: n > 10,
    "is_even": lambda n: n % 2 == 0,
    "between_20_30": lambda n: 20 <= n <= 30,
    "is_square": lambda n: _is_square(n),
    "gt_50": lambda n: n > 50,
    "mult_7": lambda n: n % 7 == 0,
    "gt_40": lambda n: n > 40,
    "is_fib": lambda n: _is_fib(n),
    "gt_20": lambda n: n > 20,
    "is_pow2": lambda n: _is_pow2(n),
    "gt_100": lambda n: n > 100,
    "is_odd": lambda n: n % 2 == 1,
    "div_3": lambda n: n % 3 == 0,
    "is_palindrome": lambda n: _is_palindrome(n),
}


# ---------------------------------------------------------------------------
# Soft-constraint check functions (heuristic)
# ---------------------------------------------------------------------------


def _has_explanation(text: str) -> float:
    """Score based on explanation length and substance."""
    words = len(text.split())
    if words >= 15:
        return 1.0
    if words >= 8:
        return 0.7
    if words >= 3:
        return 0.4
    return 0.0


def _mentions_prime(text: str) -> float:
    return 1.0 if re.search(r"prime", text, re.IGNORECASE) else 0.0


def _mentions_even(text: str) -> float:
    return 1.0 if re.search(r"even", text, re.IGNORECASE) else 0.0


def _mentions_sqrt(text: str) -> float:
    return 1.0 if re.search(r"sqrt|square root|root", text, re.IGNORECASE) else 0.0


def _mentions_div(text: str) -> float:
    return 1.0 if re.search(r"divisib|multiple|factor", text, re.IGNORECASE) else 0.0


def _mentions_sequence(text: str) -> float:
    return 1.0 if re.search(r"sequence|fibonacci|series", text, re.IGNORECASE) else 0.0


def _mentions_exponent(text: str) -> float:
    return 1.0 if re.search(r"exponent|power|2\^", text, re.IGNORECASE) else 0.0


def _mentions_palindrome(text: str) -> float:
    return 1.0 if re.search(r"palindrome|same.*backward|reads.*same", text, re.IGNORECASE) else 0.0


_SOFT_CHECKS = {
    "has_explanation": _has_explanation,
    "mentions_prime": _mentions_prime,
    "mentions_even": _mentions_even,
    "mentions_sqrt": _mentions_sqrt,
    "mentions_div": _mentions_div,
    "mentions_sequence": _mentions_sequence,
    "mentions_exponent": _mentions_exponent,
    "mentions_palindrome": _mentions_palindrome,
}


def _extract_number(text: str) -> Optional[int]:
    """Extract the first integer from text."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def hybrid_verification_generator(seed: int) -> Problem:
    """Generate a HybridVerification problem.

    Selects a problem template with hard and soft constraints. The model
    must produce an answer satisfying the hard constraints and an
    explanation satisfying the soft constraints.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with hard/soft constraint metadata.
    """
    rng = random.Random(seed)
    template = rng.choice(_HV_PROBLEMS)

    hard_descs = [c["desc"] for c in template["hard_constraints"]]
    soft_descs = [c["desc"] for c in template["soft_constraints"]]

    prompt = (
        f"Problem: {template['problem']}\n\n"
        f"Hard constraints (must ALL be satisfied):\n"
        + "".join(f"  - {d}\n" for d in hard_descs)
        + "Soft constraints (quality / style):\n"
        + "".join(f"  - {d}\n" for d in soft_descs)
        + "\nFormat:\n"
        f"ANSWER: <answer>\n"
        f"EXPLANATION: <text>"
    )

    return Problem(
        id=f"hybrid_ver_{seed}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=0.4 + 0.1 * len(hard_descs),
        metadata={
            "type": "hybrid_verification",
            "problem": template["problem"],
            "hard_constraints": hard_descs,
            "soft_constraints": soft_descs,
            "hard_checks": [c["check"] for c in template["hard_constraints"]],
            "soft_checks": [c["check"] for c in template["soft_constraints"]],
            "correct_answer": template["correct_answer"],
            "constraint_type": template["constraint_type"],
        },
        token_budget=400,
        source="hybrid_verification_generator",
    )


hybrid_verification_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class HybridVerificationVerifier(Verifier):
    """Verify a hybrid-constraint response.

    Hard constraints are checked by deterministic rules; soft constraints
    are scored by heuristics. The final score combines both.

    Args:
        hard_checks: List of hard-constraint check names.
        soft_checks: List of soft-constraint check names.
        correct_answer: A known-correct answer (used for partial credit).
    """

    def __init__(
        self,
        hard_checks: list[str],
        soft_checks: list[str],
        correct_answer: str,
    ):
        super().__init__()
        self._hard_checks = hard_checks
        self._soft_checks = soft_checks
        self._correct_answer = correct_answer

    def verify(self, response: str) -> VerifierResult:
        answer_text = self._parse_answer(response)
        explanation = self._parse_explanation(response)

        # Extract numeric answer
        n = _extract_number(answer_text) if answer_text else None

        # Hard constraints
        hard_results: list[bool] = []
        for check_name in self._hard_checks:
            fn = _HARD_CHECKS.get(check_name)
            if fn is None or n is None:
                hard_results.append(False)
                continue
            try:
                hard_results.append(bool(fn(n)))
            except Exception:
                hard_results.append(False)

        hard_score = sum(hard_results) / len(hard_results) if hard_results else 0.0
        all_hard = all(hard_results) if hard_results else False

        # Soft constraints
        soft_results: list[float] = []
        for check_name in self._soft_checks:
            fn = _SOFT_CHECKS.get(check_name)
            if fn is None:
                soft_results.append(0.0)
                continue
            try:
                soft_results.append(float(fn(explanation)))
            except Exception:
                soft_results.append(0.0)

        soft_score = sum(soft_results) / len(soft_results) if soft_results else 0.0

        score = hard_score * 0.6 + soft_score * 0.4
        correct = all_hard and soft_score >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "hard_score": hard_score,
                "soft_score": soft_score,
                "all_hard": 1.0 if all_hard else 0.0,
            },
            diagnostics=(
                f"Hard={hard_score:.2f} ({sum(hard_results)}/{len(hard_results)}) "
                f"Soft={soft_score:.2f} Answer={n}"
            ),
            metadata={
                "answer": n,
                "hard_results": hard_results,
                "soft_results": soft_results,
            },
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class HybridVerificationEnv(BatchEnvBase):
    """HybridVerification environment: hard + soft constraints.

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
            problem_generator = hybrid_verification_generator
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
        return HybridVerificationVerifier(
            hard_checks=meta["hard_checks"],
            soft_checks=meta["soft_checks"],
            correct_answer=meta["correct_answer"],
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
