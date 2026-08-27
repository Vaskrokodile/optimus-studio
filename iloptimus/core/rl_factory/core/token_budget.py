"""
Token budget management for reasoning environments.

Provides token counting (character-based approximation + optional tiktoken integration)
and budget tracking that environments use to penalize verbosity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Character-to-token ratio. The empirical average for English text + code is
# ~4 characters per token. We use a slightly conservative 3.5 to avoid
# under-counting dense code snippets.
_CHARS_PER_TOKEN = 3.5

# Optional: use tiktoken if available for exact counts
_tiktoken_enc = None


def _try_load_tiktoken():
    """Lazily load tiktoken if installed."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = False  # marker: not available
    return _tiktoken_enc


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in *text*.

    Uses tiktoken (cl100k_base) if available for an exact count, otherwise
    falls back to a character-based heuristic calibrated for mixed English+code.
    """
    if not text:
        return 0

    enc = _try_load_tiktoken()
    if enc:
        return len(enc.encode(text))

    # Heuristic: split on whitespace + punctuation, count chunks
    # This is more accurate than pure char-count for code with lots of symbols.
    # Each match contributes ~len(match)//4 tokens (clamped to >=1) so that long
    # runs of repeated characters (e.g. "xxx...x") don't collapse to a single
    # token — a known failure mode of pure match-counting.
    tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
    # Calibrate: our regex produces ~1.3x the token count vs cl100k_base
    count = sum(max(1, len(m) // 4) for m in tokens)
    return max(1, int(count / 1.3))


@dataclass
class TokenBudget:
    """
    Tracks token usage against a budget.

    Environments create a TokenBudget per episode. Each action (or sub-action)
    consumes tokens. When the budget is exhausted, the environment truncates.
    """
    initial: int
    used: int = 0
    history: list[int] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.initial - self.used)

    @property
    def fraction_used(self) -> float:
        """Return fraction of budget consumed, in [0, 1]."""
        if self.initial <= 0:
            return 1.0
        return min(1.0, self.used / self.initial)

    @property
    def fraction_remaining(self) -> float:
        return 1.0 - self.fraction_used

    def consume(self, text: str) -> int:
        """Consume tokens for *text*. Returns the number of tokens consumed."""
        n = estimate_tokens(text)
        self.used += n
        self.history.append(n)
        return n

    def can_spend(self, text: str) -> bool:
        """Check whether *text* would fit within the remaining budget."""
        return estimate_tokens(text) <= self.remaining

    def reset(self, initial: Optional[int] = None) -> None:
        """Reset the budget, optionally to a new *initial* value."""
        if initial is not None:
            self.initial = initial
        self.used = 0
        self.history.clear()
