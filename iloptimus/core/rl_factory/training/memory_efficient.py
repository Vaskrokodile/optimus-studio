"""
MemoryEfficientTraining — memory-efficient training utilities.

Provides:
  - MemoryConfig: KV cache dtype, gradient checkpointing, CPU offloading,
    partial-token gradients (NAT — Not All Tokens Needed).
  - TokenSelector: NAT importance-based token selection with
    Horvitz-Thompson reweighting.
  - MemoryTracker: memory usage estimates (without torch).

Works WITHOUT torch — uses numpy for all math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


# Bytes per element for common dtypes
_DTYPE_BYTES = {
    "fp8": 1,
    "int8": 1,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp64": 8,
}


@dataclass
class MemoryConfig:
    """Configuration for memory-efficient training."""
    kv_cache_dtype: str = "fp16"  # "fp8" or "fp16"
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    offload_ratio: float = 0.5
    partial_token_ratio: float = 0.5  # NAT: keep 50% of tokens


class TokenSelector:
    """
    NAT (Not All Tokens Needed) — select a subset of tokens to compute
    gradients for, using importance sampling.

    Tokens with larger loss magnitude are more likely to be selected.
    Horvitz-Thompson reweighting ensures the gradient estimate remains
    unbiased.

    Usage:
        selector = TokenSelector(ratio=0.5)
        mask = selector.select_tokens(losses)
        reweighted = selector.reweight_losses(losses, mask)
    """

    def __init__(self, ratio: float = 0.5):
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio

    def select_tokens(self, losses: np.ndarray) -> np.ndarray:
        """
        Select tokens using importance sampling.

        Selection probability is proportional to loss magnitude (so
        high-loss tokens are more likely to be kept), then normalized
        so the expected number selected equals ratio * len(losses).

        Returns:
            Boolean mask (True = keep this token).
        """
        losses = np.asarray(losses, dtype=np.float64)
        n = len(losses)
        if n == 0:
            return np.array([], dtype=bool)

        k = max(1, int(round(self.ratio * n)))

        # Importance weights: proportional to |loss| (plus epsilon to
        # avoid zero probability for zero-loss tokens)
        eps = 1e-8
        weights = np.abs(losses) + eps
        probs = weights / weights.sum()

        # Sample k indices without replacement, weighted by importance
        rng = np.random.default_rng()
        selected_indices = rng.choice(n, size=k, replace=False, p=probs)

        mask = np.zeros(n, dtype=bool)
        mask[selected_indices] = True
        return mask

    def reweight_losses(
        self, losses: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply Horvitz-Thompson reweighting to selected tokens.

        Each selected token's loss is scaled by 1/p_i where p_i is its
        inclusion probability. This ensures the sum of reweighted losses
        is an unbiased estimator of the sum of all losses.

        Non-selected tokens get loss 0.
        """
        losses = np.asarray(losses, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        n = len(losses)
        if n == 0:
            return np.array([], dtype=np.float64)

        eps = 1e-8
        weights = np.abs(losses) + eps
        probs = weights / weights.sum()

        # Inclusion probability for sampling k from n with weights p:
        # approximate p_i_inclusion ~ k * p_i (capped at 1)
        k = int(mask.sum())
        inclusion_probs = np.clip(k * probs, eps, 1.0)

        reweighted = np.zeros(n, dtype=np.float64)
        reweighted[mask] = losses[mask] / inclusion_probs[mask]
        return reweighted


class MemoryTracker:
    """
    Tracks memory usage estimates (without actually using torch memory).

    Usage:
        tracker = MemoryTracker(MemoryConfig(kv_cache_dtype="fp8"))
        kv_mem = tracker.estimate_kv_cache_memory(seq_len=2048, num_layers=24, hidden_size=1024)
        report = tracker.get_memory_report()
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._estimates: dict = {}

    def estimate_kv_cache_memory(
        self,
        seq_len: int,
        num_layers: int,
        hidden_size: int,
        dtype: Optional[str] = None,
    ) -> float:
        """
        Estimate KV cache memory in bytes.

        KV cache stores K and V for each layer: 2 * seq_len * num_layers * hidden_size * bytes_per_element
        """
        dtype = dtype or self.config.kv_cache_dtype
        bytes_per = _DTYPE_BYTES.get(dtype, 2)
        # 2 (K and V) * seq_len * num_layers * hidden_size * bytes
        mem = 2 * seq_len * num_layers * hidden_size * bytes_per
        self._estimates["kv_cache_memory"] = mem
        return float(mem)

    def estimate_offload_savings(
        self, activation_memory: float, offload_ratio: Optional[float] = None
    ) -> float:
        """
        Estimate CPU offload savings (bytes saved from GPU -> CPU).
        """
        ratio = offload_ratio if offload_ratio is not None else self.config.offload_ratio
        savings = activation_memory * ratio
        self._estimates["offload_savings"] = savings
        return float(savings)

    def get_memory_report(self) -> dict:
        """Return a dict of all memory estimates."""
        report = dict(self._estimates)
        report["kv_cache_dtype"] = self.config.kv_cache_dtype
        report["gradient_checkpointing"] = self.config.gradient_checkpointing
        report["cpu_offload"] = self.config.cpu_offload
        report["offload_ratio"] = self.config.offload_ratio
        report["partial_token_ratio"] = self.config.partial_token_ratio
        return report
