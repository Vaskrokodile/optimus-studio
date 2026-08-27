"""
SpeculativeRollout — speculative decoding with dynamic tuning.

Uses a draft model to propose tokens, then verifies with the target
model. Tracks acceptance rate and dynamically adjusts max_draft_tokens
(ReSpec-style). Works WITHOUT vllm — draft_fn and target_fn are plain
callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_model_name: str = "draft"
    target_model_name: str = "target"
    acceptance_threshold: float = 0.5
    max_draft_tokens: int = 8
    dynamic_tuning: bool = True
    min_draft_tokens: int = 1
    max_draft_tokens_cap: int = 32


class SpeculativeRollout:
    """
    Speculative decoding rollout.

    draft_fn(prompt, n_tokens) -> list[int]: proposes n_tokens token ids.
    target_fn(prompt, n_tokens) -> list[int]: returns the target model's
        token ids for the same position (for verification).

    During generate(), the draft model proposes a block of tokens, the
    target model verifies them, and accepted tokens are kept. The
    acceptance rate is tracked and used to dynamically tune
    max_draft_tokens.

    Usage:
        config = SpeculativeConfig()
        rollout = SpeculativeRollout(config, draft_fn, target_fn)
        tokens = rollout.generate("Hello", max_tokens=32)
        stats = rollout.get_stats()
    """

    def __init__(
        self,
        config: SpeculativeConfig,
        draft_fn: Callable[[str, int], list[int]],
        target_fn: Callable[[str, int], list[int]],
    ):
        self.config = config
        self.draft_fn = draft_fn
        self.target_fn = target_fn

        # Acceptance tracking
        self._total_accepted = 0
        self._total_proposed = 0
        self._n_rounds = 0
        self._draft_token_history: list[int] = []

    def generate(self, prompt: str, max_tokens: int) -> list[int]:
        """
        Simulate speculative decoding.

        Repeatedly: draft proposes max_draft_tokens, target verifies,
        keep accepted prefix. Continue until max_tokens reached.
        """
        output_tokens: list[int] = []
        current_prompt = prompt

        while len(output_tokens) < max_tokens:
            n_draft = min(self.config.max_draft_tokens, max_tokens - len(output_tokens))
            if n_draft <= 0:
                break

            draft_tokens = self.draft_fn(current_prompt, n_draft)
            target_tokens = self.target_fn(current_prompt, n_draft)

            # Verify: accept tokens while draft == target
            accepted = 0
            for d, t in zip(draft_tokens, target_tokens):
                if d == t:
                    accepted += 1
                    output_tokens.append(d)
                else:
                    # Mismatch — use target token, stop this round
                    output_tokens.append(t)
                    break
            else:
                # All matched — if target produced an extra token, append it
                if len(target_tokens) > len(draft_tokens):
                    output_tokens.append(target_tokens[len(draft_tokens)])

            self.update_acceptance_rate(accepted, n_draft)
            self._n_rounds += 1
            self._draft_token_history.append(n_draft)

            # Update prompt context for next round
            current_prompt = prompt + " " + " ".join(str(t) for t in output_tokens)

            if self.config.dynamic_tuning and self._n_rounds % 4 == 0:
                self.tune()

        return output_tokens[:max_tokens]

    def update_acceptance_rate(self, accepted: int, total: int) -> None:
        """Update rolling acceptance rate."""
        self._total_accepted += accepted
        self._total_proposed += total

    def tune(self) -> None:
        """
        Dynamically adjust max_draft_tokens based on acceptance rate
        (ReSpec-style).

        High acceptance -> increase draft tokens (more speculation).
        Low acceptance -> decrease draft tokens (less wasted compute).
        """
        if self._total_proposed == 0:
            return
        rate = self._total_accepted / self._total_proposed

        if rate > self.config.acceptance_threshold:
            # Increase draft tokens
            new_val = min(
                self.config.max_draft_tokens + 2,
                self.config.max_draft_tokens_cap,
            )
        else:
            # Decrease draft tokens
            new_val = max(
                self.config.max_draft_tokens - 2,
                self.config.min_draft_tokens,
            )
        self.config.max_draft_tokens = new_val

    def get_stats(self) -> dict:
        """Return speculative decoding statistics."""
        rate = (
            self._total_accepted / self._total_proposed
            if self._total_proposed > 0
            else 0.0
        )
        avg_draft = (
            float(np.mean(self._draft_token_history))
            if self._draft_token_history
            else 0.0
        )
        # Speedup estimate: with acceptance rate r and k draft tokens,
        # expected speedup ~ 1 + r * k (rough)
        speedup = 1.0 + rate * avg_draft
        return {
            "acceptance_rate": rate,
            "avg_draft_tokens": avg_draft,
            "speedup_estimate": speedup,
            "current_max_draft_tokens": self.config.max_draft_tokens,
        }
