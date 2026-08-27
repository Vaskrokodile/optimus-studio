"""
Intelligence Density Tracker (Idea 19).

Measures the true metric for RL environment efficiency:

    Intelligence Density = Δcapability / (tokens × FLOPs)

Current RL optimization focuses on tokens/sec, rollouts/sec, GPU utilization.
But the real metric should be: capability gain per unit of environment compute.

An environment producing 1% improvement using 10M tokens could be vastly
better than one producing 1.1% improvement using 1B tokens.

This module provides:
  - IntelligenceDensityTracker: Logs episodes, computes density
  - DensityTrackedEnv: Wrapper that auto-logs any environment
  - EnvironmentBenchmark: Compares multiple environments on density

Usage:
    tracker = IntelligenceDensityTracker()
    env = DensityTrackedEnv(SomeEnv(), tracker)

    # Measure baseline capability
    baseline = tracker.evaluate(model, eval_tasks)

    # Train
    for episode in range(1000):
        obs, info = env.reset()
        obs, reward, _, _, info = env.step(action)
        # tracker logs automatically

    # Measure post-training capability
    post = tracker.evaluate(model, eval_tasks)

    # Get density report
    report = tracker.compute_density(baseline, post)
    print(f"Intelligence Density: {report.intelligence_density:.6f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from iloptimus.core.rl_factory.core.token_budget import estimate_tokens


@dataclass
class EpisodeRecord:
    """A single training episode record."""
    env_name: str
    tokens_used: int
    reward: float
    correct: bool
    timestamp: float = field(default_factory=time.time)
    problem_type: str = ""
    difficulty: float = 0.0


@dataclass
class IntelligenceDensityReport:
    """Report on intelligence density for an environment or set of environments."""
    environment_name: str
    capability_delta: float
    baseline_capability: float
    post_training_capability: float
    total_tokens: int
    estimated_flops: float
    intelligence_density: float
    episodes: int
    mean_reward: float
    success_rate: float
    tokens_per_episode: float
    capability_per_million_tokens: float

    def summary(self) -> str:
        return (
            f"Environment: {self.environment_name}\n"
            f"  Capability: {self.baseline_capability:.1%} -> {self.post_training_capability:.1%} "
            f"(delta={self.capability_delta:+.1%})\n"
            f"  Compute: {self.total_tokens:,} tokens, ~{self.estimated_flops:.2e} FLOPs\n"
            f"  Intelligence Density: {self.intelligence_density:.6f} per token-FLOP\n"
            f"  Capability per 1M tokens: {self.capability_per_million_tokens:.4f}\n"
            f"  Episodes: {self.episodes}, Success rate: {self.success_rate:.1%}\n"
            f"  Mean reward: {self.mean_reward:.3f}, Tokens/episode: {self.tokens_per_episode:.0f}"
        )


class IntelligenceDensityTracker:
    """
    Tracks compute usage and capability to compute intelligence density.

    Intelligence Density = Δcapability / (tokens × FLOPs)

    This is the metric that matters for RL environment design — not
    tokens/sec or rollouts/sec, but how much capability the model
    gains per unit of compute invested.
    """

    def __init__(self, model_parameters: int = 1_000_000_000):
        """
        Args:
            model_parameters: Number of model parameters (for FLOPs estimation).
                Default 1B. FLOPs ≈ parameters × tokens × 6 (transformer approximation).
        """
        self._model_parameters = model_parameters
        self._episodes: list[EpisodeRecord] = []
        self._eval_scores: list[float] = []

    def log_episode(
        self,
        env_name: str,
        tokens_used: int,
        reward: float,
        correct: bool,
        problem_type: str = "",
        difficulty: float = 0.0,
    ) -> None:
        """Log a single training episode."""
        self._episodes.append(EpisodeRecord(
            env_name=env_name,
            tokens_used=tokens_used,
            reward=reward,
            correct=correct,
            problem_type=problem_type,
            difficulty=difficulty,
        ))

    def log_evaluation(self, score: float) -> None:
        """Log an evaluation score (capability measurement)."""
        self._eval_scores.append(score)

    def evaluate(self, eval_fn) -> float:
        """Run an evaluation function and log the result.

        Args:
            eval_fn: A callable that returns a capability score (0.0-1.0).

        Returns:
            The capability score.
        """
        score = eval_fn()
        self.log_evaluation(score)
        return score

    def compute_density(
        self,
        baseline_capability: Optional[float] = None,
        post_training_capability: Optional[float] = None,
        env_name: Optional[str] = None,
    ) -> IntelligenceDensityReport:
        """Compute intelligence density from logged data.

        Args:
            baseline_capability: Pre-training capability score. If None, uses
                first logged eval score.
            post_training_capability: Post-training capability score. If None,
                uses last logged eval score.
            env_name: Environment name for the report. If None, uses the most
                common env name in logged episodes.

        Returns:
            IntelligenceDensityReport with all metrics.
        """
        if baseline_capability is None:
            if not self._eval_scores:
                raise ValueError("No eval scores logged. Provide baseline_capability or call log_evaluation.")
            baseline_capability = self._eval_scores[0]

        if post_training_capability is None:
            if not self._eval_scores:
                raise ValueError("No eval scores logged. Provide post_training_capability or call log_evaluation.")
            post_training_capability = self._eval_scores[-1]

        if not self._episodes:
            raise ValueError("No episodes logged. Call log_episode during training.")

        # Filter by env_name if specified
        episodes = self._episodes
        if env_name:
            episodes = [e for e in self._episodes if e.env_name == env_name]
            if not episodes:
                raise ValueError(f"No episodes found for environment '{env_name}'")

        total_tokens = sum(e.tokens_used for e in episodes)
        capability_delta = post_training_capability - baseline_capability

        # FLOPs estimation: ~6 × parameters × tokens (forward + backward)
        estimated_flops = 6 * self._model_parameters * total_tokens

        # Intelligence density: capability gain per token-FLOP
        if total_tokens > 0 and estimated_flops > 0:
            intelligence_density = capability_delta / (total_tokens * estimated_flops / 1e18)  # Normalized
        else:
            intelligence_density = 0.0

        # Capability per million tokens (more intuitive metric)
        capability_per_million = capability_delta / (total_tokens / 1_000_000) if total_tokens > 0 else 0.0

        # Determine env name
        if env_name is None:
            env_names = [e.env_name for e in episodes]
            env_name = max(set(env_names), key=env_names.count) if env_names else "unknown"

        mean_reward = sum(e.reward for e in episodes) / len(episodes)
        success_rate = sum(1 for e in episodes if e.correct) / len(episodes)
        tokens_per_episode = total_tokens / len(episodes)

        return IntelligenceDensityReport(
            environment_name=env_name,
            capability_delta=capability_delta,
            baseline_capability=baseline_capability,
            post_training_capability=post_training_capability,
            total_tokens=total_tokens,
            estimated_flops=estimated_flops,
            intelligence_density=intelligence_density,
            episodes=len(episodes),
            mean_reward=mean_reward,
            success_rate=success_rate,
            tokens_per_episode=tokens_per_episode,
            capability_per_million_tokens=capability_per_million,
        )

    def compare_environments(self, baseline: float, post: float) -> list[IntelligenceDensityReport]:
        """Compute density reports for each environment type in the log.

        Args:
            baseline: Baseline capability score.
            post: Post-training capability score.

        Returns:
            List of reports, sorted by intelligence density (descending).
        """
        env_names = set(e.env_name for e in self._episodes)
        reports = []
        for name in env_names:
            try:
                report = self.compute_density(baseline, post, env_name=name)
                reports.append(report)
            except ValueError:
                continue
        reports.sort(key=lambda r: r.intelligence_density, reverse=True)
        return reports

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self._episodes)

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    def reset(self) -> None:
        """Clear all logged data."""
        self._episodes.clear()
        self._eval_scores.clear()


class EnvironmentBenchmark:
    """
    Benchmark multiple environments on intelligence density.

    Runs each environment for N episodes, measures capability before/after,
    and ranks environments by intelligence density.
    """

    def __init__(self, model_parameters: int = 1_000_000_000):
        self._tracker = IntelligenceDensityTracker(model_parameters)

    def benchmark_environment(
        self,
        env_name: str,
        run_fn,
        eval_fn,
        n_episodes: int = 100,
    ) -> IntelligenceDensityReport:
        """Benchmark a single environment.

        Args:
            env_name: Name of the environment.
            run_fn: Callable(episode_idx) -> (tokens, reward, correct, problem_type, difficulty).
                Runs one episode and returns metrics.
            eval_fn: Callable() -> float. Measures capability (0.0-1.0).
            n_episodes: Number of episodes to run.

        Returns:
            IntelligenceDensityReport for this environment.
        """
        # Baseline
        baseline = self._tracker.evaluate(eval_fn)

        # Run episodes
        for i in range(n_episodes):
            tokens, reward, correct, ptype, diff = run_fn(i)
            self._tracker.log_episode(env_name, tokens, reward, correct, ptype, diff)

        # Post-training
        post = self._tracker.evaluate(eval_fn)

        return self._tracker.compute_density(baseline, post, env_name=env_name)

    def rank_environments(
        self,
        environments: dict[str, tuple],
        eval_fn,
        n_episodes: int = 100,
    ) -> list[IntelligenceDensityReport]:
        """Rank multiple environments by intelligence density.

        Args:
            environments: Dict of {name: (run_fn, baseline_capability)}.
                run_fn(episode_idx) -> (tokens, reward, correct, type, difficulty).
            eval_fn: Callable() -> float. Measures capability.
            n_episodes: Episodes per environment.

        Returns:
            List of reports sorted by intelligence density (best first).
        """
        reports = []
        for name, (run_fn, _) in environments.items():
            self._tracker.reset()
            report = self.benchmark_environment(name, run_fn, eval_fn, n_episodes)
            reports.append(report)

        reports.sort(key=lambda r: r.intelligence_density, reverse=True)
        return reports
