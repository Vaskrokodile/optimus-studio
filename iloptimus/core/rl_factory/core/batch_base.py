"""
Batch-aware environment base class for multi-attempt RL environments.

These environments are designed for vLLM-batched inference, where generating
N responses in parallel costs roughly the same as generating 1. The key
difference from single-turn environments:

  - `step()` accepts a LIST of N responses (the batch), not a single string
  - The verifier scores ALL N responses
  - The reward is computed from the AGGREGATE (best, vote, tournament, etc.)
  - Additional signals: diversity, coverage, agreement, self-improvement

This enables training paradigms that are impossible with single-turn envs:
  - Best-of-N selection (generate many, pick the best)
  - Self-consistency voting (generate many, majority vote)
  - Iterative refinement (generate → score → improve → score)
  - Tournament judging (generate many, pairwise compare)
  - Parallel exploration (generate diverse approaches, score each)

The batch size N is configurable per environment (default: 16, range: 8-32).
"""

from __future__ import annotations

import random
from typing import Any, Optional, Union

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
from iloptimus.core.rl_factory.core.reward import RewardConfig, compute_combined_reward, RewardComponents
from iloptimus.core.rl_factory.core.token_budget import TokenBudget, estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


class BatchEnvBase(BaseReasoningEnv):
    """
    Base class for batch-aware multi-attempt environments.

    The key difference: `step()` accepts either:
      - A list of N strings (the batch of responses from vLLM)
      - A single string (for backward compatibility, treated as N=1)

    Subclasses must implement:
      - `_make_verifier(problem)` → Verifier (same as base)
      - `_aggregate_scores(scores: list[VerifierResult]) → dict`
        Returns the aggregate reward computation (best, vote, tournament, etc.)
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config: Optional[RewardConfig] = None,
        anti_pattern_detector: Optional[AntiPatternDetector] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )
        self._batch_size = batch_size
        self._batch_results: list[dict] = []

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def step(self, action: Union[str, list[str]]) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one batch step.

        Args:
            action: Either a list of N response strings (the batch), or a
                    single string (treated as N=1 for backward compatibility).

        Returns:
            (observation, reward, terminated, truncated, info)
            The info dict contains per-sample scores plus aggregate metrics.
        """
        # Normalize action to a list
        if isinstance(action, str):
            responses = [action]
        else:
            responses = list(action)

        n = len(responses)
        self._step_count += 1

        # Score each response with the verifier
        per_sample: list[dict] = []
        for i, resp in enumerate(responses):
            verifier_result = self._verifier.verify(resp)
            tokens_used = estimate_tokens(resp)
            ap_report = self._ap_detector.analyze(
                resp, prompt=self._current_problem.prompt
            )
            format_bonus = self._check_format(resp)

            per_sample.append({
                "index": i,
                "response": resp[:200],  # truncated for logging
                "correct": verifier_result.correct,
                "verifier_score": verifier_result.score,
                "verifier_diagnostics": verifier_result.diagnostics,
                "tokens_used": tokens_used,
                "format_bonus": format_bonus,
                "anti_pattern_hits": ap_report.total_hits,
                "anti_pattern_penalty": ap_report.total_penalty,
            })

        # Aggregate scores using the subclass's strategy
        aggregate = self._aggregate_scores(per_sample)

        # Compute the final reward from the aggregate
        reward = self._compute_batch_reward(aggregate, per_sample)

        # Build info dict
        info = {
            "problem_id": self._current_problem.id,
            "difficulty": self._current_problem.difficulty,
            "batch_size": n,
            "per_sample": per_sample,
            "aggregate": aggregate,
            "reward": reward,
            "correct": aggregate.get("correct", False),
            "verifier_score": aggregate.get("best_score", 0.0),
            "verifier_diagnostics": aggregate.get("diagnostics", ""),
            "tokens_used": sum(s["tokens_used"] for s in per_sample),
            "tokens_per_sample": [s["tokens_used"] for s in per_sample],
            "anti_pattern_hits": sum(s["anti_pattern_hits"] for s in per_sample),
            "step_count": self._step_count,
        }

        terminated = True
        truncated = False
        obs = self._make_observation()

        return obs, reward, terminated, truncated, info

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """
        Aggregate per-sample scores into a single reward signal.

        Override in subclasses to implement best-of-N, voting, tournament, etc.
        """
        raise NotImplementedError("Subclasses must implement _aggregate_scores")

    def _compute_batch_reward(self, aggregate: dict, per_sample: list[dict]) -> float:
        """
        Convert the aggregate metrics into a scalar reward.

        Default: use the aggregate's 'reward' field if present, else
        use the best score. Subclasses can override.
        """
        if "reward" in aggregate:
            return float(aggregate["reward"])
        return float(aggregate.get("best_score", 0.0))

    def _check_format(self, response: str) -> float:
        """Default format check — subclasses can override."""
        return 1.0 if len(response.strip()) > 10 else 0.0

    def _extract_answer(self, response: str) -> str:
        return response
