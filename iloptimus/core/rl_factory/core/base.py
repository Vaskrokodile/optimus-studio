"""
Base environment class for rl-factory.

All five environments inherit from BaseReasoningEnv, which provides:
  - Gymnasium API compliance (reset, step, render, close)
  - Token budget management
  - Anti-pattern detection
  - Reward computation via the shared RewardConfig
  - A clean separation between problem generation, verification, and reward

The environment is designed for text-in / text-out interaction:
  - observation: a dict with "prompt" (the problem), "budget_remaining", "step", "difficulty"
  - action: a string (the model's reasoning + answer)
  - reward: scalar from compute_combined_reward
  - terminated: True when the action is submitted (single-turn) or the task is done
  - truncated: True when the token budget is exhausted

This is a SINGLE-TURN environment by default: the model gets the problem,
produces one response, and gets a reward. Multi-turn variants can be built
by subclassing and overriding step().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
from iloptimus.core.rl_factory.core.reward import (
    RewardComponents,
    RewardConfig,
    compute_combined_reward,
    compute_efficiency_bonus,
)
from iloptimus.core.rl_factory.core.token_budget import TokenBudget, estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


@dataclass
class Problem:
    """
    A single problem instance for an environment.

    Attributes:
        id: Unique identifier for this problem.
        prompt: The problem text shown to the model.
        difficulty: Difficulty in [0, 1] (0=easy, 1=hard).
        metadata: Environment-specific data (ground truth, test cases, etc.).
        token_budget: Maximum tokens the model can use for this problem.
        source: Where the problem came from (generator name, dataset, etc.).
    """
    id: str
    prompt: str
    difficulty: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    token_budget: int = 2048
    source: str = "generated"


class BaseReasoningEnv(gym.Env):
    """
    Base class for all rl-factory reasoning environments.

    Subclasses must implement:
      - _make_verifier(problem) -> Verifier
      - _make_prompt(problem) -> str  (usually just problem.prompt)
      - _check_format(response) -> float  (format bonus, 0-1)

    Subclasses can override:
      - _generate_problem(seed) -> Problem
      - _extract_answer(response) -> str  (extract the final answer from the response)
    """

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[RewardConfig] = None,
        anti_pattern_detector: Optional[AntiPatternDetector] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            problems: A fixed list of problems to sample from. If None, uses problem_generator.
            problem_generator: A callable that takes a seed and returns a Problem.
            reward_config: Configuration for the reward function.
            anti_pattern_detector: Detector for wasteful reasoning patterns.
            render_mode: "text" or None.
        """
        super().__init__()

        if problems is None and problem_generator is None:
            raise ValueError("Must provide either 'problems' or 'problem_generator'")

        self._problems = problems
        self._problem_generator = problem_generator
        self._reward_config = reward_config or RewardConfig()
        self._ap_detector = anti_pattern_detector or AntiPatternDetector()
        self.render_mode = render_mode

        # Action space: the model produces a text response
        # We use a text-based space (represented as a string)
        self.action_space = spaces.Text(max_length=100_000)

        # Observation space: dict with problem info
        self.observation_space = spaces.Dict({
            "prompt": spaces.Text(max_length=50_000),
            "budget_remaining": spaces.Box(low=0, high=1e6, shape=(), dtype=np.float32),
            "step": spaces.Discrete(100),
            "difficulty": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })

        # Episode state
        self._current_problem: Optional[Problem] = None
        self._verifier: Optional[Verifier] = None
        self._budget: Optional[TokenBudget] = None
        self._step_count: int = 0
        self._episode_responses: list[str] = []
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for problem selection.
            options: Optional dict with "problem_id" to select a specific problem,
                     or "problem_index" for direct indexing.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Select a problem
        if options and "problem_id" in options:
            problem_id = options["problem_id"]
            self._current_problem = next(
                (p for p in self._problems if p.id == problem_id), None
            )
            if self._current_problem is None:
                raise ValueError(f"Problem {problem_id} not found")
        elif options and "problem_index" in options:
            self._current_problem = self._problems[options["problem_index"]]
        elif self._problems is not None:
            idx = self._rng.integers(len(self._problems))
            self._current_problem = self._problems[idx]
        else:
            gen_seed = int(self._rng.integers(0, 2**31 - 1))
            self._current_problem = self._problem_generator(gen_seed)

        # Create the verifier for this problem
        self._verifier = self._make_verifier(self._current_problem)

        # Initialize token budget
        self._budget = TokenBudget(initial=self._current_problem.token_budget)

        # Reset episode state
        self._step_count = 0
        self._episode_responses = []

        obs = self._make_observation()
        info = {
            "problem_id": self._current_problem.id,
            "difficulty": self._current_problem.difficulty,
            "token_budget": self._current_problem.token_budget,
            "source": self._current_problem.source,
        }

        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """
        Process the model's action (a text response).

        For single-turn environments, this verifies the response and returns
        the reward. For multi-turn environments, subclasses can override this
        to support multiple steps.

        Args:
            action: The model's text response.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        # Consume tokens from the budget
        tokens_used = self._budget.consume(action)
        self._episode_responses.append(action)
        self._step_count += 1

        # Check if budget is exhausted
        truncated = self._budget.remaining <= 0

        # Verify correctness — pass the FULL response to the verifier.
        # Each verifier handles its own answer extraction internally.
        verifier_result = self._verifier.verify(action)
        correct = verifier_result.correct

        # Check format
        format_bonus = self._check_format(action)

        # Detect anti-patterns
        ap_report = self._ap_detector.analyze(action, prompt=self._current_problem.prompt)

        # Compute efficiency bonus
        efficiency = compute_efficiency_bonus(
            self._budget, self._reward_config, self._current_problem.difficulty
        )

        # Difficulty bonus: solving harder problems gives a small bonus
        difficulty_bonus = self._current_problem.difficulty if correct else 0.0

        # Assemble reward components
        components = RewardComponents(
            correctness=verifier_result.score,
            efficiency=efficiency,
            anti_pattern_penalty=-ap_report.total_penalty,
            format_bonus=format_bonus,
            difficulty_bonus=difficulty_bonus,
            token_count=tokens_used,
            budget_used_fraction=self._budget.fraction_used,
            anti_pattern_report=ap_report,
            verifier_result=verifier_result,
        )

        # Compute final reward
        reward, reward_info = compute_combined_reward(
            components, self._reward_config, self._current_problem.difficulty
        )

        # Single-turn: always terminated after one action
        terminated = True

        # Build observation (mostly empty since episode is over)
        obs = self._make_observation()

        # Build info dict
        # NOTE: reward_info may contain a "correct" key (derived from score > 0),
        # but the authoritative correctness comes from the verifier. We spread
        # reward_info first, then override "correct" with the verifier's verdict.
        info = {
            "problem_id": self._current_problem.id,
            "difficulty": self._current_problem.difficulty,
            "tokens_used": tokens_used,
            "budget_remaining": self._budget.remaining,
            "budget_used_fraction": self._budget.fraction_used,
            "verifier_score": verifier_result.score,
            "verifier_diagnostics": verifier_result.diagnostics,
            "format_bonus": format_bonus,
            "efficiency": efficiency,
            "anti_pattern_hits": ap_report.total_hits,
            "anti_pattern_counts": ap_report.counts_by_type,
            "anti_pattern_penalty": ap_report.total_penalty,
            "difficulty_bonus": difficulty_bonus,
            "step_count": self._step_count,
            **reward_info,
            # Override with authoritative verifier correctness
            "correct": correct,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "text":
            print(f"Problem: {self._current_problem.prompt[:200]}...")
            print(f"Budget: {self._budget.remaining}/{self._budget.initial} tokens remaining")
            print(f"Step: {self._step_count}")
            if self._episode_responses:
                print(f"Last response: {self._episode_responses[-1][:200]}...")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Methods for subclasses to implement
    # ------------------------------------------------------------------

    def _make_verifier(self, problem: Problem) -> Verifier:
        """Create a verifier for the given problem. Must be implemented by subclasses."""
        raise NotImplementedError

    def _check_format(self, response: str) -> float:
        """
        Check if the response follows the expected format.
        Returns a bonus in [0, 1]. Default: no format check.
        """
        return 0.0

    def _extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the model's response.
        Default: return the full response. Subclasses can override to parse
        a specific format (e.g., \\boxed{} for math).
        """
        return response

    def _generate_problem(self, seed: int) -> Problem:
        """Generate a problem from the generator. Called when no fixed problem list."""
        if self._problem_generator is None:
            raise RuntimeError("No problem generator provided")
        return self._problem_generator(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> dict:
        """Build the observation dict from current state."""
        if self._current_problem is None:
            return {
                "prompt": "",
                "budget_remaining": np.float32(0),
                "step": 0,
                "difficulty": np.float32(0),
            }

        return {
            "prompt": self._current_problem.prompt,
            "budget_remaining": np.float32(self._budget.remaining if self._budget else 0),
            "step": self._step_count,
            "difficulty": np.float32(self._current_problem.difficulty),
        }

    @property
    def current_problem(self) -> Optional[Problem]:
        return self._current_problem

    @property
    def budget(self) -> Optional[TokenBudget]:
        return self._budget
