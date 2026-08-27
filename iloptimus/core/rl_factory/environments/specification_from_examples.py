"""
SpecificationFromExamples: Write a spec from I/O examples (not code).

Environment concept:
  The model is given input/output examples and must describe the
  specification (the rule) that transforms input to output. This trains
  inductive specification — the inverse of code-from-spec.

  Why: writing a spec from examples requires abstracting the pattern.
  The model must identify the *rule*, not just memorize the examples.
  This is the core of program synthesis understanding.

Verification:
  - Check the spec mentions the correct transformation rule keywords.
  - Partial credit for each keyword found.

Reward design:
  fraction of rule_keywords found in the specification.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — I/O examples + rule description + keywords
# ---------------------------------------------------------------------------

_SPEC_PROBLEMS = [
    {
        "examples": [("hello", "HELLO"), ("world", "WORLD"), ("python", "PYTHON")],
        "rule_description": "Convert each string to uppercase.",
        "rule_keywords": ["uppercase", "upper", "capital", "case"],
    },
    {
        "examples": [([1, 2, 3], 6), ([4, 5, 6], 15), ([10, 20], 30)],
        "rule_description": "Sum all elements in the list.",
        "rule_keywords": ["sum", "add", "total", "elements"],
    },
    {
        "examples": [("hello", 5), ("world", 5), ("hi", 2)],
        "rule_description": "Return the length of the string.",
        "rule_keywords": ["length", "count", "size", "characters"],
    },
    {
        "examples": [([3, 1, 2], [1, 2, 3]), ([9, 5, 7], [5, 7, 9])],
        "rule_description": "Sort the list in ascending order.",
        "rule_keywords": ["sort", "ascending", "order", "arrange"],
    },
    {
        "examples": [("hello world", "world hello"), ("a b", "b a")],
        "rule_description": "Reverse the order of words in the string.",
        "rule_keywords": ["reverse", "words", "order", "swap"],
    },
    {
        "examples": [([1, 2, 3, 4], 2), ([2, 4, 6], 3), ([10, 20, 30], 3)],
        "rule_description": "Count the even numbers in the list.",
        "rule_keywords": ["count", "even", "divisible", "2"],
    },
    {
        "examples": [("racecar", True), ("hello", False), ("level", True)],
        "rule_description": "Check if the string is a palindrome.",
        "rule_keywords": ["palindrome", "reverse", "same", "reads"],
    },
    {
        "examples": [([1, 2, 3], 3), ([5, 5, 5, 5], 4), ([7], 1)],
        "rule_description": "Return the number of elements in the list.",
        "rule_keywords": ["count", "length", "size", "number", "elements"],
    },
]


def _format_example(inp: Any, out: Any) -> str:
    return f"  Input: {inp!r} -> Output: {out!r}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def specification_from_examples_generator(seed: int) -> Problem:
    """Generate a SpecificationFromExamples problem: describe the rule."""
    rng = random.Random(seed)
    problem = rng.choice(_SPEC_PROBLEMS)
    examples = problem["examples"]

    examples_text = "\n".join(_format_example(inp, out) for inp, out in examples)

    prompt = (
        f"Given the following input/output examples, describe the "
        f"specification (the rule) that transforms input to output.\n\n"
        f"Examples:\n{examples_text}\n\n"
        f"Describe the transformation rule clearly.\n"
        f"Format your answer as: SPEC: <description>"
    )

    difficulty = 0.4 + len(problem["rule_keywords"]) * 0.05
    difficulty = min(0.8, difficulty)

    return Problem(
        id=f"specification_from_examples_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "examples": examples,
            "rule_description": problem["rule_description"],
            "rule_keywords": problem["rule_keywords"],
        },
        token_budget=512,
        source="specification_from_examples_generator",
    )


specification_from_examples_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SpecificationFromExamplesVerifier(Verifier):
    """Verify a specification: check for rule keywords.

    reward = fraction of rule_keywords found in the spec.
    """

    def __init__(self, rule_keywords: list[str], rule_description: str):
        super().__init__()
        self._keywords = [k.lower() for k in rule_keywords]
        self._description = rule_description.lower()

    def verify(self, response: str) -> VerifierResult:
        spec_match = re.search(r"SPEC\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if spec_match:
            text = spec_match.group(1).lower()
        else:
            text = response.lower()

        found = 0
        for kw in self._keywords:
            if kw in text:
                found += 1

        fraction = found / len(self._keywords) if self._keywords else 0.0
        correct = fraction >= 0.5

        return VerifierResult(
            correct=correct,
            score=fraction,
            partial_credit={
                "keywords_found": float(found),
                "keywords_total": float(len(self._keywords)),
                "keyword_fraction": fraction,
            },
            diagnostics=(
                f"keywords={found}/{len(self._keywords)} "
                f"fraction={fraction:.2f}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SpecificationFromExamplesEnv(BatchEnvBase):
    """SpecificationFromExamples: write a spec from I/O examples.

    Batch-aware: N parallel attempts; reward = best specification.
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
            problem_generator = specification_from_examples_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SpecificationFromExamplesVerifier(
            rule_keywords=problem.metadata["rule_keywords"],
            rule_description=problem.metadata["rule_description"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"SPEC\s*:", response, re.IGNORECASE):
            return 1.0
        if len(response.strip()) > 20:
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
        match = re.search(r"SPEC\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
