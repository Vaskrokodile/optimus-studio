"""EIG and Intelligence Density computations for the adaptive RL curriculum.

Pure-Python implementation (no scipy dependency). ``digamma`` is implemented
via the standard asymptotic expansion with recurrence to push the argument
above the expansion threshold, so ``beta_entropy`` is exact enough for the
information-gain ranking used by the frontier sampler.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass


def bernoulli_entropy(p: float) -> float:
    """Binary entropy H(p) in bits, clamped away from 0/1."""
    p = max(1e-6, min(1 - 1e-6, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def digamma(x: float) -> float:
    """Pure-Python digamma (psi) function.

    Uses recurrence to lift ``x`` to >= 6, then the asymptotic series:
        psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    Accurate to ~1e-9 for x >= 6.
    """
    if x <= 0.0:
        # Reflection: psi(1-x) - psi(x) = pi * cot(pi*x)
        # Avoid poles; for our use x = alpha/beta > 0 so this branch is rare.
        if abs(x - round(x)) < 1e-9:
            return float("inf")
        return digamma(1.0 - x) - math.pi / math.tan(math.pi * x)

    result = 0.0
    # Recurrence: psi(x) = psi(x+1) - 1/x  ->  lift to >= 6
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0

    # Asymptotic expansion
    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv
    # Bernoulli-number series in inv2: -1/12, +1/120, -1/252, +1/240, ...
    result -= inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
    return result


def beta_entropy(alpha: float, beta: float) -> float:
    """Differential entropy of Beta(alpha, beta) in nats.

    H = ln B(a,b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)
    Falls back to the Bernoulli entropy at the posterior mean if inputs are
    degenerate (non-positive parameters).
    """
    if alpha <= 0.0 or beta <= 0.0:
        return bernoulli_entropy(alpha / (alpha + beta)) if (alpha + beta) > 0 else 0.0
    try:
        return (
            math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
            - (alpha - 1.0) * digamma(alpha)
            - (beta - 1.0) * digamma(beta)
            + (alpha + beta - 2.0) * digamma(alpha + beta)
        )
    except (ValueError, OverflowError):
        return bernoulli_entropy(alpha / (alpha + beta))


def information_gain_binary(
    alpha_before: float,
    beta_before: float,
    successes: int,
    failures: int,
) -> float:
    """Expected reduction in posterior entropy from observing ``successes``/
    ``failures`` more Bernoulli draws. Non-negative (entropy can only drop in
    expectation for a Beta prior under i.i.d. Bernoulli observations)."""
    alpha_after = alpha_before + successes
    beta_after = beta_before + failures
    return max(0.0, beta_entropy(alpha_before, beta_before) - beta_entropy(alpha_after, beta_after))


def tool_selection_entropy(calls: list[dict]) -> float:
    """Shannon entropy (bits) over the empirical tool-name distribution."""
    names = [c.get("name") for c in calls if c.get("name") and c.get("name") != "__parse_error__"]
    if not names:
        return 0.0
    total = len(names)
    h = 0.0
    for count in Counter(names).values():
        p = count / total
        h -= p * math.log2(p)
    return h


def capability_gain(score_history: deque[float] | list[float], window: int = 20) -> float:
    """Difference between the recent half-window mean and the preceding half.

    Positive => the model is improving on this task; negative => regressing.
    Returns 0.0 when there is not enough history.
    """
    if len(score_history) < window:
        return 0.0
    seq = list(score_history)[-window:]
    half = window // 2
    recent = seq[-half:]
    older = seq[-window:-half] if half else []
    if not older:
        return 0.0
    return sum(recent) / len(recent) - sum(older) / len(older)


def intelligence_density(
    capability_gain: float,
    token_cost: float,
    flops: float,
) -> float:
    """Capability gain per unit of token-FLOP cost.

    ID = gain / (tokens * flops). Higher is a better training instrument.
    Robust to zero/negative cost (returns 0.0) and to negative gain (a task
    where the model is regressing is not a high-quality instrument).
    """
    denom = token_cost * flops
    if denom <= 0:
        return 0.0
    return max(0.0, capability_gain) / denom


def learning_progress(
    short_window: list[bool] | deque[bool],
    long_window: list[bool] | deque[bool],
) -> float:
    """Recent success rate minus long-run success rate. Positive => teachable."""
    short = list(short_window)
    long = list(long_window)
    if not short or not long:
        return 0.0
    return sum(short) / len(short) - sum(long) / len(long)


@dataclass
class RolloutMetrics:
    score: float
    correctness: float
    tokens: int
    flops: float
    ig_binary: float
    ig_tool: float
    intelligence_density: float
    capability_gain: float


__all__ = [
    "bernoulli_entropy",
    "digamma",
    "beta_entropy",
    "information_gain_binary",
    "tool_selection_entropy",
    "capability_gain",
    "intelligence_density",
    "learning_progress",
    "RolloutMetrics",
]
