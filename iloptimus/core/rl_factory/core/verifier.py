"""
Verifier framework for rl-factory environments.

A Verifier checks whether a model's response is correct. Each environment
provides its own verifier; this module defines the common interface and
utility verifiers that can be composed.

Design principles (from the research):
  - Rule-based verifiers are more reliable than neural reward models
    (DeepSeek-R1's approach — avoid reward hacking).
  - Verifiers should be deterministic and fast.
  - A verifier returns a VerifierResult with correctness, partial credit,
    and diagnostic info.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class VerifierResult:
    """
    Result of verifying a model response.

    Attributes:
        correct: Whether the response fully satisfies the task.
        score: Continuous score in [0, 1]. 1.0 = perfect, 0.0 = wrong.
        partial_credit: Breakdown of partial credit components.
        diagnostics: Human-readable diagnostic info for analysis.
        metadata: Arbitrary structured data for logging/analysis.
    """
    correct: bool
    score: float
    partial_credit: dict[str, float] = field(default_factory=dict)
    diagnostics: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp score to [0, 1]
        self.score = max(0.0, min(1.0, self.score))
        # correct implies score >= some threshold
        if self.correct and self.score < 0.5:
            self.score = 1.0


class Verifier:
    """
    Base verifier. Subclasses implement `verify()`.

    A verifier is initialized with the problem's ground truth and any
    parameters needed for checking. The `verify(response)` method returns
    a VerifierResult.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def verify(self, response: str) -> VerifierResult:
        """Verify a model response. Must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, response: str) -> VerifierResult:
        return self.verify(response)


# ---------------------------------------------------------------------------
# Utility verifiers (composable building blocks)
# ---------------------------------------------------------------------------


class ExactMatchVerifier(Verifier):
    """Check if the response exactly matches the expected answer (after normalization)."""

    def __init__(self, expected: str, case_sensitive: bool = False, strip_whitespace: bool = True):
        super().__init__()
        self._expected = expected
        self._case_sensitive = case_sensitive
        self._strip = strip_whitespace

    def _normalize(self, s: str) -> str:
        s = s.strip() if self._strip else s
        if not self._case_sensitive:
            s = s.lower()
        return s

    def verify(self, response: str) -> VerifierResult:
        norm_response = self._normalize(response)
        norm_expected = self._normalize(self._expected)
        correct = norm_response == norm_expected
        return VerifierResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            diagnostics=f"Expected: {self._expected!r}, Got: {response!r}",
        )


class NumericVerifier(Verifier):
    """
    Check if the response contains a number matching the expected value
    within a tolerance. Extracts the last number from the response.
    """

    def __init__(self, expected: float, tolerance: float = 1e-6):
        super().__init__()
        self._expected = expected
        self._tolerance = tolerance

    def verify(self, response: str) -> VerifierResult:
        # Find all numbers in the response
        numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", response)
        if not numbers:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No number found in response",
            )

        # Try the last number (models usually state the answer last)
        try:
            value = float(numbers[-1])
        except ValueError:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Could not parse number from: {numbers[-1]}",
            )

        correct = abs(value - self._expected) <= self._tolerance
        return VerifierResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            diagnostics=f"Expected: {self._expected}, Got: {value}",
        )


class RegexVerifier(Verifier):
    """Check if the response matches a regex pattern."""

    def __init__(self, pattern: str, group: int = 0, expected: Optional[str] = None):
        super().__init__()
        self._pattern = re.compile(pattern)
        self._group = group
        self._expected = expected

    def verify(self, response: str) -> VerifierResult:
        match = self._pattern.search(response)
        if not match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Pattern not found: {self._pattern.pattern}",
            )

        if self._expected is not None:
            extracted = match.group(self._group)
            correct = extracted.strip() == self._expected.strip()
            return VerifierResult(
                correct=correct,
                score=1.0 if correct else 0.0,
                diagnostics=f"Extracted: {extracted!r}, Expected: {self._expected!r}",
            )

        return VerifierResult(correct=True, score=1.0, diagnostics="Pattern matched")


class CodeExecutionVerifier(Verifier):
    """
    Verify a code response by executing it and checking the output.

    Runs the code in a restricted namespace and compares the result
    (return value or stdout) against an expected value.
    """

    def __init__(
        self,
        expected_output: Any,
        setup_code: str = "",
        extract_function: Optional[Callable[[str], str]] = None,
        timeout: float = 5.0,
    ):
        super().__init__()
        self._expected_output = expected_output
        self._setup_code = setup_code
        self._extract = extract_function
        self._timeout = timeout

    def verify(self, response: str) -> VerifierResult:
        code = response if self._extract is None else self._extract(response)

        # Build the execution context
        namespace: dict[str, Any] = {}
        full_code = self._setup_code + "\n" + code

        try:
            # Execute with a timeout using signal (Unix) or simple exec
            # For cross-platform safety, we use a simple exec with a check
            exec(full_code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Execution error: {type(e).__name__}: {e}",
            )

        # Check for a 'result' variable or the last expression
        result = namespace.get("result", None)
        if result is None:
            # Try calling a 'solve' function if it exists
            if "solve" in namespace and callable(namespace["solve"]):
                try:
                    result = namespace["solve"]()
                except Exception as e:
                    return VerifierResult(
                        correct=False,
                        score=0.0,
                        diagnostics=f"solve() error: {type(e).__name__}: {e}",
                    )

        if result is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No 'result' variable or 'solve()' function found",
            )

        correct = result == self._expected_output
        return VerifierResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            diagnostics=f"Expected: {self._expected_output!r}, Got: {result!r}",
        )


class CompositeVerifier(Verifier):
    """
    Combine multiple verifiers with weights.

    Each sub-verifier checks a different aspect; the final score is the
    weighted average. correct=True only if all required verifiers pass.
    """

    def __init__(self, verifiers: list[tuple[Verifier, float, bool]]):
        """
        Args:
            verifiers: List of (verifier, weight, required) tuples.
                weight: contribution to the final score.
                required: if True, this verifier must pass for correct=True.
        """
        super().__init__()
        self._verifiers = verifiers

    def verify(self, response: str) -> VerifierResult:
        total_weight = sum(w for _, w, _ in self._verifiers)
        if total_weight == 0:
            return VerifierResult(correct=False, score=0.0, diagnostics="No verifiers with weight > 0")

        weighted_score = 0.0
        all_required_pass = True
        partial = {}
        diagnostics_parts = []

        for verifier, weight, required in self._verifiers:
            result = verifier.verify(response)
            weighted_score += result.score * weight
            partial[verifier.__class__.__name__] = result.score
            if required and not result.correct:
                all_required_pass = False
            if result.diagnostics:
                diagnostics_parts.append(f"[{verifier.__class__.__name__}] {result.diagnostics}")

        final_score = weighted_score / total_weight
        return VerifierResult(
            correct=all_required_pass,
            score=final_score,
            partial_credit=partial,
            diagnostics=" | ".join(diagnostics_parts),
        )
