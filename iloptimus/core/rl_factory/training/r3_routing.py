"""
R3Routing — Rollout Routing Replay for MoE stability.

Records expert routing decisions during rollout, then replays the exact
routing during training. This eliminates the train/inference routing
mismatch that causes KL divergence in MoE models.

Provides:
  - RoutingRecorder: records routing decisions during rollout.
  - RoutingReplayer: replays recorded routing during training.

Works WITHOUT torch — pure data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class R3Config:
    """Configuration for R3 routing replay."""
    enabled: bool = True
    replay_strictness: float = 1.0  # 0.0 = loose, 1.0 = exact
    overhead_estimate: float = 0.019  # 1.9% recording overhead


class RoutingRecorder:
    """
    Records expert routing decisions during rollout.

    Stores a routing table: (token_idx, layer_idx) -> expert_idx.

    Usage:
        recorder = RoutingRecorder()
        recorder.record(token_idx=0, expert_idx=3, layer_idx=0)
        table = recorder.get_routing_table()
    """

    def __init__(self, config: Optional[R3Config] = None):
        self.config = config or R3Config()
        # (token_idx, layer_idx) -> expert_idx
        self._table: dict[tuple[int, int], int] = {}
        self._n_recorded = 0

    def record(self, token_idx: int, expert_idx: int, layer_idx: int) -> None:
        """Record a routing decision."""
        self._table[(token_idx, layer_idx)] = expert_idx
        self._n_recorded += 1

    def get_routing_table(self) -> dict:
        """
        Return the recorded routing table.

        Returns a dict mapping (token_idx, layer_idx) -> expert_idx.
        """
        return dict(self._table)

    def clear(self) -> None:
        """Clear recorded routing."""
        self._table.clear()
        self._n_recorded = 0

    def get_overhead(self) -> float:
        """Estimate the recording overhead (default 1.9%)."""
        return self.config.overhead_estimate


class RoutingReplayer:
    """
    Replays recorded routing during training.

    Usage:
        replayer = RoutingReplayer(routing_table, replay_strictness=1.0)
        expected = replayer.get_expected_routing(token_idx=0, layer_idx=0)
        ok = replayer.check_replay(token_idx=0, layer_idx=0, actual_expert=3)
        accuracy = replayer.get_replay_accuracy()
    """

    def __init__(
        self,
        routing_table: dict,
        replay_strictness: float = 1.0,
    ):
        self.routing_table = dict(routing_table)
        self.replay_strictness = replay_strictness
        self._checks = 0
        self._matches = 0

    def get_expected_routing(self, token_idx: int, layer_idx: int) -> int:
        """Return the recorded expert index for (token_idx, layer_idx)."""
        return self.routing_table.get((token_idx, layer_idx), -1)

    def check_replay(
        self, token_idx: int, layer_idx: int, actual_expert: int
    ) -> bool:
        """
        Check if actual routing matches the replayed routing.

        With strictness < 1.0, allows probabilistic acceptance of
        mismatches (loose replay). With strictness = 1.0, requires
        exact match.
        """
        expected = self.get_expected_routing(token_idx, layer_idx)
        self._checks += 1

        if expected == actual_expert:
            self._matches += 1
            return True

        # Loose replay: with probability (1 - strictness), accept mismatch
        if self.replay_strictness < 1.0:
            accept_prob = 1.0 - self.replay_strictness
            if np.random.random() < accept_prob:
                self._matches += 1
                return True

        return False

    def get_replay_accuracy(self) -> float:
        """Return fraction of correctly replayed routings."""
        if self._checks == 0:
            return 0.0
        return self._matches / self._checks

    def get_kl_reduction_estimate(self) -> float:
        """Estimate KL reduction factor (default 10x)."""
        # With exact replay (strictness=1.0), KL reduction is maximal (~10x).
        # With loose replay, reduction scales with strictness.
        base_reduction = 10.0
        return base_reduction * self.replay_strictness
