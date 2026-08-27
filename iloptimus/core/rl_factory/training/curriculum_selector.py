"""
CurriculumSelector — adaptive task selection for RL training.

Implements learned curriculum selection inspired by Actor-Curator (2025):
  - Neural curator that selects training problems by optimizing expected policy improvement
  - Formulated as non-stationary stochastic bandit
  - UCB-based sampling as simpler alternative
  - Co-adaptive: curator selects → actor trains → improvement feeds curator

Two modes:
  1. UCB-based (simple, no training needed): Select tasks using Upper Confidence Bound
     on the reward signal. Balances exploitation (high-reward tasks) with exploration
     (under-sampled tasks).

  2. Sliding-window adaptive: Track recent performance per task type, select tasks
     at the "frontier" of model capability (50-70% success rate).

Usage:
    selector = CurriculumSelector(mode="ucb", num_task_types=10)
    task_idx = selector.select()
    selector.update(task_idx, reward=0.7, correct=True)

    # Or with sliding window:
    selector = CurriculumSelector(mode="frontier", num_task_types=10)
    task_idx = selector.select()
    selector.update(task_idx, reward=0.5, correct=False)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TaskStats:
    """Statistics for a single task type."""
    task_type: str
    total_attempts: int = 0
    total_correct: int = 0
    total_reward: float = 0.0
    recent_rewards: list[float] = field(default_factory=list)
    recent_correct: list[bool] = field(default_factory=list)
    last_selected: float = 0.0
    selection_count: int = 0

    @property
    def success_rate(self) -> float:
        """Recent success rate (sliding window)."""
        if not self.recent_correct:
            return 0.0
        return sum(self.recent_correct) / len(self.recent_correct)

    @property
    def mean_reward(self) -> float:
        """Recent mean reward (sliding window)."""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_correct / self.total_attempts

    @property
    def overall_mean_reward(self) -> float:
        """Overall mean reward."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_reward / self.total_attempts


class CurriculumSelector:
    """
    Adaptive curriculum selector for RL training.

    Selects which task type to train on next, based on:
      - UCB: Upper Confidence Bound on reward (exploration-exploitation balance)
      - Frontier: Select tasks at model's capability frontier (~50-70% success)
      - Difficulty: Select tasks matching target difficulty level
      - Weakness: Target the weakest-performing task types

    Modes:
      "ucb": UCB-based selection. Balances high-reward (exploitation) with
             under-sampled (exploration) tasks. Good for general use.
      "frontier": Select tasks where recent success rate is 40-70%. This is
                  the "zone of proximal development" — challenging but solvable.
      "weakness": Target task types with lowest recent success rate.
      "difficulty": Match task difficulty to model's current capability.
    """

    def __init__(
        self,
        mode: str = "ucb",
        task_types: Optional[list[str]] = None,
        num_task_types: int = 10,
        window_size: int = 50,
        ucb_c: float = 2.0,
        frontier_min: float = 0.3,
        frontier_max: float = 0.7,
        target_difficulty: float = 0.5,
        min_attempts_before_select: int = 3,
    ):
        """
        Args:
            mode: Selection mode ("ucb", "frontier", "weakness", "difficulty").
            task_types: List of task type names. If None, uses indices 0..num_task_types-1.
            num_task_types: Number of task types (if task_types not provided).
            window_size: Sliding window size for recent performance tracking.
            ucb_c: UCB exploration constant (higher = more exploration).
            frontier_min: Minimum success rate for "frontier" mode.
            frontier_max: Maximum success rate for "frontier" mode.
            target_difficulty: Target difficulty for "difficulty" mode.
            min_attempts: Minimum attempts before a task is eligible for selection.
        """
        self.mode = mode
        self.window_size = window_size
        self.ucb_c = ucb_c
        self.frontier_min = frontier_min
        self.frontier_max = frontier_max
        self.target_difficulty = target_difficulty
        self.min_attempts = min_attempts_before_select

        if task_types is None:
            task_types = [f"task_{i}" for i in range(num_task_types)]

        self.task_types = task_types
        self.stats: dict[str, TaskStats] = {
            t: TaskStats(task_type=t) for t in task_types
        }
        self.total_selections = 0
        self.start_time = time.time()

    def select(self) -> str:
        """
        Select the next task type to train on.

        Returns:
            Task type name.
        """
        self.total_selections += 1

        if self.mode == "ucb":
            return self._select_ucb()
        elif self.mode == "frontier":
            return self._select_frontier()
        elif self.mode == "weakness":
            return self._select_weakness()
        elif self.mode == "difficulty":
            return self._select_difficulty()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def select_index(self) -> int:
        """Select and return the index of the chosen task type."""
        task_type = self.select()
        return self.task_types.index(task_type)

    def update(self, task_type: str, reward: float, correct: bool) -> None:
        """
        Update statistics after a training episode.

        Args:
            task_type: The task type that was trained on.
            reward: The reward received.
            correct: Whether the response was correct.
        """
        if task_type not in self.stats:
            self.stats[task_type] = TaskStats(task_type=task_type)
            self.task_types.append(task_type)

        stats = self.stats[task_type]
        stats.total_attempts += 1
        stats.total_reward += reward
        if correct:
            stats.total_correct += 1

        # Update sliding window
        stats.recent_rewards.append(reward)
        stats.recent_correct.append(correct)
        if len(stats.recent_rewards) > self.window_size:
            stats.recent_rewards.pop(0)
            stats.recent_correct.pop(0)

        stats.selection_count += 1
        stats.last_selected = time.time()

    def _select_ucb(self) -> str:
        """UCB-based selection: balance exploitation and exploration."""
        total_n = sum(s.total_attempts for s in self.stats.values())
        if total_n == 0:
            # No data yet — select randomly
            return self.task_types[np.random.randint(len(self.task_types))]

        ucb_values = {}
        for task_type, stats in self.stats.items():
            n = stats.total_attempts
            if n < self.min_attempts:
                # Under-explored task — give it high priority
                ucb_values[task_type] = float('inf')
                continue

            # UCB1 formula: mean_reward + c * sqrt(ln(total_n) / n)
            mean_r = stats.overall_mean_reward
            exploration = self.ucb_c * math.sqrt(math.log(total_n) / n)
            ucb_values[task_type] = mean_r + exploration

        # Select task with highest UCB value
        return max(ucb_values, key=ucb_values.get)

    def _select_frontier(self) -> str:
        """Select tasks at the capability frontier (40-70% success rate)."""
        # Prefer tasks with success rate in [frontier_min, frontier_max]
        frontier_tasks = []
        for task_type, stats in self.stats.items():
            if stats.total_attempts < self.min_attempts:
                # Under-explored — include with high priority
                frontier_tasks.append((task_type, 1.0))
                continue

            sr = stats.success_rate
            if self.frontier_min <= sr <= self.frontier_max:
                # In the frontier zone — high priority
                # Prefer tasks closer to 50% success (maximum learning signal)
                distance_from_optimal = abs(sr - 0.5)
                priority = 1.0 - distance_from_optimal
                frontier_tasks.append((task_type, priority))

        if frontier_tasks:
            # Select from frontier tasks, weighted by priority
            task_names = [t for t, _ in frontier_tasks]
            priorities = np.array([p for _, p in frontier_tasks])
            priorities = priorities / priorities.sum()
            return np.random.choice(task_names, p=priorities)

        # No frontier tasks — select the task closest to the frontier
        best_task = None
        best_distance = float('inf')
        for task_type, stats in self.stats.items():
            if stats.total_attempts < self.min_attempts:
                return task_type
            sr = stats.success_rate
            # Distance to the center of the frontier zone
            target = (self.frontier_min + self.frontier_max) / 2
            distance = abs(sr - target)
            if distance < best_distance:
                best_distance = distance
                best_task = task_type

        return best_task or self.task_types[0]

    def _select_weakness(self) -> str:
        """Select the task type with the lowest recent success rate."""
        min_success = float('inf')
        weakest_task = None

        for task_type, stats in self.stats.items():
            if stats.total_attempts < self.min_attempts:
                # Under-explored — high priority
                return task_type

            sr = stats.success_rate
            if sr < min_success:
                min_success = sr
                weakest_task = task_type

        return weakest_task or self.task_types[0]

    def _select_difficulty(self) -> str:
        """Select tasks matching the target difficulty level."""
        # This mode requires task types to have associated difficulty scores
        # We use the mean reward as a proxy for difficulty alignment
        best_task = None
        best_distance = float('inf')

        for task_type, stats in self.stats.items():
            if stats.total_attempts < self.min_attempts:
                return task_type

            # Distance from target difficulty
            mean_r = stats.mean_reward
            distance = abs(mean_r - self.target_difficulty)
            if distance < best_distance:
                best_distance = distance
                best_task = task_type

        return best_task or self.task_types[0]

    def get_stats(self) -> dict[str, dict]:
        """Get current statistics for all task types."""
        return {
            t: {
                "attempts": s.total_attempts,
                "success_rate": s.success_rate,
                "overall_success_rate": s.overall_success_rate,
                "mean_reward": s.mean_reward,
                "overall_mean_reward": s.overall_mean_reward,
                "selection_count": s.selection_count,
            }
            for t, s in self.stats.items()
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of curriculum state."""
        lines = [f"Curriculum Selector (mode={self.mode}, selections={self.total_selections})"]
        for task_type, stats in sorted(self.stats.items()):
            sr = stats.success_rate
            mr = stats.mean_reward
            lines.append(
                f"  {task_type}: attempts={stats.total_attempts}, "
                f"success={sr:.1%}, reward={mr:.3f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        for stats in self.stats.values():
            stats.total_attempts = 0
            stats.total_correct = 0
            stats.total_reward = 0.0
            stats.recent_rewards.clear()
            stats.recent_correct.clear()
            stats.selection_count = 0
        self.total_selections = 0
