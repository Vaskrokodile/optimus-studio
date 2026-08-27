"""
TraceSurgery: Find the surgically injected error in a reasoning trace.

Environment concept:
  The model is given a reasoning trace (math proof, code explanation, or
  logic chain) that was originally correct but has had ONE subtle error
  surgically injected at a random position. The model must:
    1. Identify WHICH step contains the error
    2. Explain WHAT the error is
    3. Provide the FIX (the corrected step)

  This trains the model to carefully read and verify each step of a
  reasoning chain — a critical skill for self-correction and debugging.

Why this environment is unique:
  Unlike standard debugging (find a bug in code), trace surgery operates
  on REASONING TRACES. The error is a logical or mathematical mistake,
  not a syntax error. The model must understand the entire chain of
  reasoning to find where it breaks.

Problem types:
  - Math proofs with an incorrect arithmetic or algebraic step
  - Code logic explanations with a wrong claim about program behavior
  - Logical deduction chains with an invalid inference

Verification:
  - Error step position match (exact or adjacent)
  - Error description validity (mentions the correct issue)
  - Fix correctness (the corrected step produces the right result)

Reward design:
  position_correct * 0.4 + error_identified * 0.3 + fix_correct * 0.3
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Base trace definitions — each is a list of (step_text, step_value) pairs
# where step_value is the correct intermediate result.
# ---------------------------------------------------------------------------

_MATH_TRACES = [
    {
        "title": "Sum of first n integers",
        "steps": [
            ("Step 1: We want to find the sum S = 1 + 2 + 3 + ... + n.", "intro"),
            ("Step 2: Write S = n(n+1)/2 by the known formula.", "formula"),
            ("Step 3: For n=10, S = 10 * 11 / 2 = 55.", "55"),
            ("Step 4: For n=100, S = 100 * 101 / 2 = 5050.", "5050"),
            ("Step 5: Therefore the sum of the first n integers is n(n+1)/2.", "conclusion"),
        ],
        "injectable_steps": [2, 3],  # 0-indexed steps that can be corrupted
        "injections": {
            2: [
                ("For n=10, S = 10 * 11 / 2 = 60.", "55", "arithmetic error: 10*11/2=55 not 60"),
            ],
            3: [
                ("For n=100, S = 100 * 101 / 2 = 5000.", "5050", "arithmetic error: 100*101/2=5050 not 5000"),
                ("For n=100, S = 100 * 100 / 2 = 5000.", "5050", "wrong formula: should use n+1 not n"),
            ],
        },
    },
    {
        "title": "Quadratic formula derivation",
        "steps": [
            ("Step 1: Start with ax^2 + bx + c = 0.", "intro"),
            ("Step 2: Divide by a: x^2 + (b/a)x + c/a = 0.", "divide"),
            ("Step 3: Complete the square: (x + b/2a)^2 = b^2/4a^2 - c/a.", "complete"),
            ("Step 4: Simplify RHS: (b^2 - 4ac) / 4a^2.", "simplify"),
            ("Step 5: Take square root: x + b/2a = ±sqrt(b^2-4ac) / 2a.", "sqrt"),
            ("Step 6: Solve for x: x = (-b ± sqrt(b^2-4ac)) / 2a.", "conclusion"),
        ],
        "injectable_steps": [3, 4],
        "injections": {
            3: [
                ("Simplify RHS: (b^2 + 4ac) / 4a^2.", "(b^2 - 4ac) / 4a^2", "sign error: should be -4ac not +4ac"),
            ],
            4: [
                ("Take square root: x + b/2a = sqrt(b^2-4ac) / 2a.", "±sqrt(b^2-4ac) / 2a", "missing ± sign in square root"),
            ],
        },
    },
    {
        "title": "Modular arithmetic: 3^4 mod 7",
        "steps": [
            ("Step 1: Compute 3^1 mod 7 = 3.", "3"),
            ("Step 2: Compute 3^2 mod 7 = 9 mod 7 = 2.", "2"),
            ("Step 3: Compute 3^3 mod 7 = 3*2 mod 7 = 6.", "6"),
            ("Step 4: Compute 3^4 mod 7 = 3*6 mod 7 = 18 mod 7 = 4.", "4"),
            ("Step 5: Therefore 3^4 ≡ 4 (mod 7).", "conclusion"),
        ],
        "injectable_steps": [2, 3],
        "injections": {
            2: [
                ("Compute 3^3 mod 7 = 3*2 mod 7 = 5.", "6", "arithmetic error: 3*2=6 not 5"),
            ],
            3: [
                ("Compute 3^4 mod 7 = 3*6 mod 7 = 18 mod 7 = 3.", "4", "arithmetic error: 18 mod 7 = 4 not 3"),
            ],
        },
    },
    {
        "title": "Pythagorean theorem application",
        "steps": [
            ("Step 1: Given a right triangle with legs a=3 and b=4.", "intro"),
            ("Step 2: By Pythagorean theorem: c^2 = a^2 + b^2.", "theorem"),
            ("Step 3: Compute a^2 = 9 and b^2 = 16.", "squares"),
            ("Step 4: c^2 = 9 + 16 = 25.", "25"),
            ("Step 5: c = sqrt(25) = 5.", "5"),
        ],
        "injectable_steps": [3, 4],
        "injections": {
            3: [
                ("c^2 = 9 + 16 = 24.", "25", "arithmetic error: 9+16=25 not 24"),
            ],
            4: [
                ("c = sqrt(25) = 6.", "5", "arithmetic error: sqrt(25)=5 not 6"),
            ],
        },
    },
]

_CODE_TRACES = [
    {
        "title": "Binary search explanation",
        "steps": [
            ("Step 1: Binary search works on a sorted array by repeatedly dividing the search interval in half.", "intro"),
            ("Step 2: Start with lo=0 and hi=len(arr)-1.", "init"),
            ("Step 3: Compute mid = (lo + hi) // 2.", "mid"),
            ("Step 4: If arr[mid] == target, return mid (found).", "found"),
            ("Step 5: If arr[mid] < target, search the right half: lo = mid + 1.", "right"),
            ("Step 6: If arr[mid] > target, search the left half: hi = mid - 1.", "left"),
            ("Step 7: Repeat until lo > hi, meaning the target is not found.", "end"),
        ],
        "injectable_steps": [4, 5],
        "injections": {
            4: [
                ("If arr[mid] < target, search the left half: lo = mid + 1.", "search the right half: lo = mid + 1", "direction error: should search right half when arr[mid] < target"),
            ],
            5: [
                ("If arr[mid] > target, search the right half: hi = mid - 1.", "search the left half: hi = mid - 1", "direction error: should search left half when arr[mid] > target"),
            ],
        },
    },
    {
        "title": "Merge sort explanation",
        "steps": [
            ("Step 1: Merge sort divides the array into two halves recursively.", "intro"),
            ("Step 2: Base case: if the array has 0 or 1 elements, it is already sorted.", "base"),
            ("Step 3: Split the array into left and right halves.", "split"),
            ("Step 4: Recursively sort both halves.", "recurse"),
            ("Step 5: Merge the two sorted halves by comparing front elements.", "merge"),
            ("Step 6: The merged result is the fully sorted array.", "conclusion"),
        ],
        "injectable_steps": [2, 4],
        "injections": {
            2: [
                ("Base case: if the array has 0 or 2 elements, it is already sorted.", "0 or 1 elements", "base case error: arrays of size 2 are not necessarily sorted"),
            ],
            4: [
                ("Merge the two sorted halves by comparing back elements.", "comparing front elements", "merge error: merge compares front elements, not back elements"),
            ],
        },
    },
]

_LOGIC_TRACES = [
    {
        "title": "Syllogism: All men are mortal",
        "steps": [
            ("Step 1: Premise 1: All men are mortal.", "p1"),
            ("Step 2: Premise 2: Socrates is a man.", "p2"),
            ("Step 3: By universal instantiation: if Socrates is a man, then Socrates is mortal.", "instantiate"),
            ("Step 4: By modus ponens: Socrates is mortal.", "conclusion"),
        ],
        "injectable_steps": [2, 3],
        "injections": {
            2: [
                ("By universal instantiation: if Socrates is a man, then Socrates is immortal.", "mortal", "logic error: should conclude mortal, not immortal"),
            ],
            3: [
                ("By modus tollens: Socrates is mortal.", "modus ponens", "rule error: should use modus ponens, not modus tollens"),
            ],
        },
    },
    {
        "title": "If it rains, the ground is wet",
        "steps": [
            ("Step 1: Premise: If it rains, then the ground is wet (R → W).", "p1"),
            ("Step 2: Observation: The ground is wet (W is true).", "p2"),
            ("Step 3: This does NOT necessarily mean it rained — affirming the consequent is a fallacy.", "fallacy"),
            ("Step 4: The ground could be wet from other causes (sprinkler, etc.).", "conclusion"),
        ],
        "injectable_steps": [2],
        "injections": {
            2: [
                ("Therefore it rained (R is true) by modus ponens.", "does NOT necessarily mean it rained", "fallacy error: affirming the consequent is invalid, not modus ponens"),
            ],
        },
    },
]


def _inject_error(trace: dict, rng: random.Random) -> dict:
    """Inject a single error into a trace at a random injectable position."""
    injectable = trace["injectable_steps"]
    step_idx = rng.choice(injectable)
    injection_options = trace["injections"][step_idx]
    corrupted_text, correct_text, error_desc = rng.choice(injection_options)

    steps = list(trace["steps"])
    original_step = steps[step_idx][0]
    steps[step_idx] = (corrupted_text, correct_text)

    return {
        "title": trace["title"],
        "steps": steps,
        "error_step_index": step_idx,
        "error_description": error_desc,
        "correct_step_text": original_step,
        "corrupted_step_text": corrupted_text,
        "correct_value": correct_text,
    }


def generate_math_trace_surgery(rng: random.Random) -> Problem:
    """Generate a trace surgery problem from a math proof trace."""
    trace = rng.choice(_MATH_TRACES)
    injected = _inject_error(trace, rng)

    steps_text = "\n".join(s[0] for s in injected["steps"])
    num_steps = len(injected["steps"])

    prompt = (
        f"The following reasoning trace contains exactly ONE error that was "
        f"surgically injected. Find it.\n\n"
        f"Title: {injected['title']}\n\n"
        f"{steps_text}\n\n"
        f"Identify the error:\n"
        f"ERROR STEP: <step number, 1-{num_steps}>\n"
        f"WHAT'S WRONG: <explain the error>\n"
        f"FIX: <the corrected step>"
    )

    return Problem(
        id=f"trace_surgery_math_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.3 * rng.random(),
        metadata={
            "type": "trace_surgery",
            "trace_type": "math",
            "error_step_index": injected["error_step_index"],
            "error_step_number": injected["error_step_index"] + 1,
            "error_description": injected["error_description"],
            "correct_step_text": injected["correct_step_text"],
            "correct_value": injected["correct_value"],
            "num_steps": num_steps,
        },
        token_budget=600,
        source="generated",
    )


def generate_code_trace_surgery(rng: random.Random) -> Problem:
    """Generate a trace surgery problem from a code logic trace."""
    trace = rng.choice(_CODE_TRACES)
    injected = _inject_error(trace, rng)

    steps_text = "\n".join(s[0] for s in injected["steps"])
    num_steps = len(injected["steps"])

    prompt = (
        f"The following code logic explanation contains exactly ONE error "
        f"that was surgically injected. Find it.\n\n"
        f"Title: {injected['title']}\n\n"
        f"{steps_text}\n\n"
        f"Identify the error:\n"
        f"ERROR STEP: <step number, 1-{num_steps}>\n"
        f"WHAT'S WRONG: <explain the error>\n"
        f"FIX: <the corrected step>"
    )

    return Problem(
        id=f"trace_surgery_code_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.3 * rng.random(),
        metadata={
            "type": "trace_surgery",
            "trace_type": "code",
            "error_step_index": injected["error_step_index"],
            "error_step_number": injected["error_step_index"] + 1,
            "error_description": injected["error_description"],
            "correct_step_text": injected["correct_step_text"],
            "correct_value": injected["correct_value"],
            "num_steps": num_steps,
        },
        token_budget=600,
        source="generated",
    )


def generate_logic_trace_surgery(rng: random.Random) -> Problem:
    """Generate a trace surgery problem from a logic deduction trace."""
    trace = rng.choice(_LOGIC_TRACES)
    injected = _inject_error(trace, rng)

    steps_text = "\n".join(s[0] for s in injected["steps"])
    num_steps = len(injected["steps"])

    prompt = (
        f"The following logical deduction contains exactly ONE error "
        f"that was surgically injected. Find it.\n\n"
        f"Title: {injected['title']}\n\n"
        f"{steps_text}\n\n"
        f"Identify the error:\n"
        f"ERROR STEP: <step number, 1-{num_steps}>\n"
        f"WHAT'S WRONG: <explain the error>\n"
        f"FIX: <the corrected step>"
    )

    return Problem(
        id=f"trace_surgery_logic_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.3 * rng.random(),
        metadata={
            "type": "trace_surgery",
            "trace_type": "logic",
            "error_step_index": injected["error_step_index"],
            "error_step_number": injected["error_step_index"] + 1,
            "error_description": injected["error_description"],
            "correct_step_text": injected["correct_step_text"],
            "correct_value": injected["correct_value"],
            "num_steps": num_steps,
        },
        token_budget=600,
        source="generated",
    )


def trace_surgery_generator(seed: int) -> Problem:
    """Master generator that picks a random trace type."""
    rng = random.Random(seed)
    generators = [
        generate_math_trace_surgery,
        generate_code_trace_surgery,
        generate_logic_trace_surgery,
    ]
    gen = rng.choice(generators)
    return gen(rng)


# Prevent pytest from collecting this generator function as a test
trace_surgery_generator.__test__ = False  # type: ignore[attr-defined]
generate_math_trace_surgery.__test__ = False  # type: ignore[attr-defined]
generate_code_trace_surgery.__test__ = False  # type: ignore[attr-defined]
generate_logic_trace_surgery.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TraceSurgeryVerifier(Verifier):
    """
    Verifies a trace surgery response by checking:
      1. ERROR STEP matches the actual error position (exact or adjacent)
      2. WHAT'S WRONG description mentions the correct issue
      3. FIX produces the correct step
    """

    def __init__(
        self,
        error_step_number: int,
        error_description: str,
        correct_step_text: str,
        correct_value: str,
        num_steps: int,
    ):
        super().__init__()
        self._error_step_number = error_step_number
        self._error_description = error_description.lower()
        self._correct_step_text = correct_step_text.lower()
        self._correct_value = correct_value.lower()
        self._num_steps = num_steps

    def verify(self, response: str) -> VerifierResult:
        # Parse ERROR STEP
        step_match = re.search(r"ERROR\s+STEP:\s*(\d+)", response, re.IGNORECASE)
        if not step_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No ERROR STEP found. Must specify: ERROR STEP: <number>",
            )
        try:
            identified_step = int(step_match.group(1))
        except ValueError:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="ERROR STEP must be a number",
            )

        # Position check: exact match = 1.0, adjacent = 0.5
        if identified_step == self._error_step_number:
            position_score = 1.0
        elif abs(identified_step - self._error_step_number) == 1:
            position_score = 0.5
        else:
            position_score = 0.0

        # Parse WHAT'S WRONG
        wrong_match = re.search(
            r"WHAT'?S\s+WRONG:\s*(.+?)(?:\n\s*(?:FIX:|$))",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if not wrong_match:
            return VerifierResult(
                correct=False,
                score=position_score * 0.4,
                diagnostics="No WHAT'S WRONG section found",
            )
        wrong_text = wrong_match.group(1).strip().lower()

        # Check if the explanation mentions key elements of the error
        error_keywords = self._extract_keywords(self._error_description)
        wrong_keywords = self._extract_keywords(wrong_text)
        keyword_overlap = len(error_keywords & wrong_keywords) / max(len(error_keywords), 1)
        # Also check if the correct value appears in the explanation or fix
        value_mentioned = self._correct_value in wrong_text
        error_identified_score = max(keyword_overlap, 0.5 if value_mentioned else 0.0)

        # Parse FIX
        fix_match = re.search(r"FIX:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        if not fix_match:
            return VerifierResult(
                correct=False,
                score=position_score * 0.4 + error_identified_score * 0.3,
                diagnostics="No FIX section found",
            )
        fix_text = fix_match.group(1).strip().lower()

        # Check if fix contains the correct value
        fix_correct = self._correct_value in fix_text
        fix_score = 1.0 if fix_correct else 0.0

        # Compute weighted score
        total_score = position_score * 0.4 + error_identified_score * 0.3 + fix_score * 0.3
        correct = position_score >= 0.5 and error_identified_score >= 0.3 and fix_correct

        return VerifierResult(
            correct=correct,
            score=total_score,
            partial_credit={
                "position_correct": position_score,
                "error_identified": error_identified_score,
                "fix_correct": fix_score,
                "identified_step": identified_step,
                "actual_step": self._error_step_number,
            },
            diagnostics=(
                f"Step: identified={identified_step}, actual={self._error_step_number} "
                f"(score={position_score:.1f}), error_id={error_identified_score:.2f}, "
                f"fix={fix_score:.1f}"
            ),
        )

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from text, filtering stop words."""
        stop_words = {"the", "a", "an", "is", "are", "not", "should", "be", "to", "in", "of", "it"}
        words = re.findall(r"\b\w+\b", text)
        return {w for w in words if w not in stop_words and len(w) > 2}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TraceSurgeryEnv(BatchEnvBase):
    """
    TraceSurgery: find the surgically injected error in a reasoning trace.

    Batch-aware: N parallel attempts, reward = best attempt.
    The model must identify the error step, explain it, and provide a fix.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

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
            problem_generator = trace_surgery_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return TraceSurgeryVerifier(
            error_step_number=problem.metadata["error_step_number"],
            error_description=problem.metadata["error_description"],
            correct_step_text=problem.metadata["correct_step_text"],
            correct_value=problem.metadata["correct_value"],
            num_steps=problem.metadata["num_steps"],
        )

    def _check_format(self, response: str) -> float:
        has_step = bool(re.search(r"ERROR\s+STEP:\s*\d+", response, re.IGNORECASE))
        has_wrong = bool(re.search(r"WHAT'?S\s+WRONG:", response, re.IGNORECASE))
        has_fix = bool(re.search(r"FIX:", response, re.IGNORECASE))
        if has_step and has_wrong and has_fix:
            return 1.0
        if has_step and has_fix:
            return 0.5
        if has_step:
            return 0.3
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
                f"Mean={sum(scores) / len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ERROR\s+STEP:\s*(\d+)", response, re.IGNORECASE)
        if match:
            return f"Step {match.group(1)}"
        return response
