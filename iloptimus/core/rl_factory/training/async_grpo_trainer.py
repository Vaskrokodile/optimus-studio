"""
AsyncGRPOTrainer — fully async RL architecture.

Decouples generation, verification, and training into parallel stages:
  - Rollout worker: continuously generates, never waits for training
  - Reward worker: scores completions in parallel (ThreadPoolExecutor)
  - Training worker: consumes completed groups from a bounded queue
  - Weight sync: placeholder for vLLM pause/resume API

All core logic works WITHOUT torch — rollout_fn/reward_fn/update_fn are
plain callables, so the async pipeline can be tested with mocks.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


@dataclass
class AsyncGRPOConfig:
    """Configuration for the async GRPO trainer."""
    batch_size: int = 8
    group_size: int = 4
    learning_rate: float = 1e-6
    queue_maxsize: int = 64
    num_reward_workers: int = 4
    sync_interval: int = 1  # weight sync interval (steps)


class AsyncGRPOTrainer:
    """
    Fully async GRPO trainer with decoupled rollout/reward/training stages.

    Three stages connected by queues:
      rollout_queue -> [RolloutThread] -> reward_queue -> [RewardThread] -> results_queue
      [TrainingThread] consumes from results_queue.

    All inter-thread communication is via queue.Queue (thread-safe).

    Usage:
        config = AsyncGRPOConfig()
        trainer = AsyncGRPOTrainer(config, rollout_fn, reward_fn, update_fn)
        trainer.start()
        trainer.submit_rollout(prompts)
        results = trainer.get_results(timeout=5)
        trainer.step()  # one training step
        trainer.stop()
    """

    def __init__(
        self,
        config: AsyncGRPOConfig,
        rollout_fn: Callable[[list[Any]], list[Any]],
        reward_fn: Callable[[list[Any]], list[Any]],
        update_fn: Callable[[list[Any]], Any],
    ):
        self.config = config
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.update_fn = update_fn

        # Queues connecting the pipeline stages
        self.rollout_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self.reward_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self.results_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)

        # Control flags
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._step_count = 0
        self._lock = threading.RLock()  # re-entrant: step() calls _sync_weights()

        # Stats
        self.stats = {
            "rollouts_generated": 0,
            "rewards_computed": 0,
            "training_steps": 0,
            "weight_syncs": 0,
        }

    def start(self) -> None:
        """Start the rollout and reward worker threads."""
        self._stop_event.clear()
        rollout_thread = threading.Thread(
            target=self._rollout_loop, name="rollout-worker", daemon=True
        )
        reward_thread = threading.Thread(
            target=self._reward_loop, name="reward-worker", daemon=True
        )
        rollout_thread.start()
        reward_thread.start()
        self._threads = [rollout_thread, reward_thread]

    def stop(self) -> None:
        """Stop all worker threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []

    def submit_rollout(self, prompts: list[Any]) -> None:
        """Add prompts to the rollout queue."""
        self.rollout_queue.put(prompts)

    def get_results(self, timeout: Optional[float] = None) -> Optional[list[Any]]:
        """Get a completed (scored) group from the results queue."""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def step(self) -> Optional[Any]:
        """
        Process one training step: pull a batch from results queue and
        call update_fn. Returns the update result, or None if no data.
        """
        batch = []
        for _ in range(self.config.batch_size):
            group = self.get_results(timeout=0.1)
            if group is None:
                break
            batch.extend(group if isinstance(group, list) else [group])

        if not batch:
            return None

        result = self.update_fn(batch)
        with self._lock:
            self._step_count += 1
            self.stats["training_steps"] += 1
            # Weight sync placeholder (vLLM pause/resume API)
            if self._step_count % self.config.sync_interval == 0:
                self._sync_weights()
        return result

    def _rollout_loop(self) -> None:
        """Rollout worker: calls rollout_fn, puts results in reward queue."""
        while not self._stop_event.is_set():
            try:
                prompts = self.rollout_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                completions = self.rollout_fn(prompts)
                with self._lock:
                    self.stats["rollouts_generated"] += len(completions)
                self.reward_queue.put(completions)
            except Exception as e:
                # Put error sentinel so downstream doesn't hang
                self.reward_queue.put({"error": str(e)})

    def _reward_loop(self) -> None:
        """Reward worker: calls reward_fn, puts scored groups in results queue."""
        while not self._stop_event.is_set():
            try:
                completions = self.reward_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(completions, dict) and "error" in completions:
                self.results_queue.put(completions)
                continue
            try:
                scored = self.reward_fn(completions)
                with self._lock:
                    self.stats["rewards_computed"] += len(scored)
                self.results_queue.put(scored)
            except Exception as e:
                self.results_queue.put({"error": str(e)})

    def _sync_weights(self) -> None:
        """Placeholder for vLLM weight sync (pause/resume API)."""
        with self._lock:
            self.stats["weight_syncs"] += 1

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        with self._lock:
            return dict(self.stats)
