"""
ParallelRewardComputation — parallel reward computation and adaptive
rollout allocation.

Provides:
  - ParallelRewardComputer: compute rewards for a batch in parallel
    using ThreadPoolExecutor, overlapping reward computation with
    next-batch generation.
  - AdaptiveRolloutAllocator (SARA): allocate more rollouts to
    environments at the capability frontier (~50% success rate).

Works WITHOUT torch — uses concurrent.futures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


@dataclass
class ParallelRewardConfig:
    """Configuration for parallel reward computation."""
    num_workers: int = 4
    timeout: float = 30.0
    use_processes: bool = False  # kept for API compat; tests use threads
    max_queue_size: int = 64


class ParallelRewardComputer:
    """
    Compute rewards for a batch of responses in parallel.

    Uses ThreadPoolExecutor (to avoid pickling issues). Supports both
    synchronous batch computation and async computation with a callback.

    Usage:
        config = ParallelRewardConfig(num_workers=4)
        computer = ParallelRewardComputer(config, [reward_fn1, reward_fn2])
        rewards = computer.compute_batch(responses)
    """

    def __init__(
        self,
        config: ParallelRewardConfig,
        reward_fns: list[Callable[[str], float]],
    ):
        self.config = config
        self.reward_fns = reward_fns
        self._executor: Optional[ThreadPoolExecutor] = None
        self._times: list[float] = []
        self._n_computed = 0

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        return self._executor

    def compute_batch(self, responses: list[str]) -> list[float]:
        """
        Compute rewards for a batch of responses in parallel.

        Each reward function is applied to each response. If multiple
        reward functions are provided, their results are averaged per
        response.
        """
        executor = self._get_executor()
        start = time.time()

        # Submit all (response, fn) pairs
        futures = {}
        for i, resp in enumerate(responses):
            for j, fn in enumerate(self.reward_fns):
                fut = executor.submit(fn, resp)
                futures[fut] = (i, j)

        # Collect results
        results: dict[int, list[float]] = {i: [0.0] * len(self.reward_fns) for i in range(len(responses))}
        for fut in as_completed(futures, timeout=self.config.timeout):
            i, j = futures[fut]
            results[i][j] = fut.result()

        # Average across reward functions
        rewards = [float(np.mean(results[i])) for i in range(len(responses))]

        elapsed = time.time() - start
        self._times.append(elapsed)
        self._n_computed += len(responses)
        return rewards

    def compute_async(
        self,
        responses: list[str],
        callback: Callable[[list[float]], None],
    ) -> None:
        """
        Start async reward computation. Calls callback with results
        when done.
        """
        executor = self._get_executor()
        executor.submit(self._async_wrapper, responses, callback)

    def _async_wrapper(
        self,
        responses: list[str],
        callback: Callable[[list[float]], None],
    ) -> None:
        rewards = self.compute_batch(responses)
        callback(rewards)

    def get_stats(self) -> dict:
        """Return computation statistics."""
        avg_time = float(np.mean(self._times)) if self._times else 0.0
        throughput = self._n_computed / avg_time if avg_time > 0 else 0.0
        return {
            "avg_time": avg_time,
            "throughput": throughput,
            "n_computed": self._n_computed,
        }

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


class AdaptiveRolloutAllocator:
    """
    SARA-style adaptive rollout allocation.

    Allocates more rollouts to environments with success rates near 0.5
    (the frontier — maximum learning signal), and fewer to environments
    with very high (>0.8) or very low (<0.2) success rates.

    Usage:
        allocator = AdaptiveRolloutAllocator(n_envs=4, min_rollouts=1, max_rollouts=16)
        allocation = allocator.allocate([0.1, 0.5, 0.9, 0.3])
    """

    def __init__(self, n_envs: int, min_rollouts: int = 1, max_rollouts: int = 16):
        self.n_envs = n_envs
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts
        self._history: list[list[int]] = []

    def allocate(self, success_rates: list[float]) -> list[int]:
        """
        Allocate rollouts per environment based on success rates.

        Environments near 0.5 success get the most rollouts.
        """
        rates = np.asarray(success_rates, dtype=np.float64)
        if len(rates) != self.n_envs:
            raise ValueError(
                f"Expected {self.n_envs} success rates, got {len(rates)}"
            )

        # Frontier weight: peaks at 0.5, falls off towards 0 and 1
        # Using a Gaussian-like kernel centered at 0.5
        weights = np.exp(-((rates - 0.5) ** 2) / (2 * 0.2 ** 2))

        # Normalize and scale to [min_rollouts, max_rollouts]
        total_weight = weights.sum()
        if total_weight <= 0:
            # Uniform fallback
            allocation = [self.min_rollouts] * self.n_envs
        else:
            fractions = weights / total_weight
            # Scale: total rollouts = n_envs * (min + max) / 2 (midpoint)
            total_rollouts = self.n_envs * (self.min_rollouts + self.max_rollouts) // 2
            raw = fractions * total_rollouts
            allocation = [
                int(np.clip(round(r), self.min_rollouts, self.max_rollouts))
                for r in raw
            ]

        self._history.append(allocation)
        return allocation

    def get_allocation_stats(self) -> dict:
        """Return allocation statistics."""
        if not self._history:
            return {"n_allocations": 0}
        last = self._history[-1]
        return {
            "n_allocations": len(self._history),
            "last_allocation": last,
            "total_rollouts": sum(last),
            "mean_rollouts": float(np.mean(last)),
            "max_rollouts_env": int(np.argmax(last)),
            "min_rollouts_env": int(np.argmin(last)),
        }
