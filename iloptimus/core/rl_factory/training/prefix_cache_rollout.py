"""
PrefixCacheRollout — prefix caching for shared system prompts.

Computes KV cache once for a shared prefix, reuses it for all prompts
sharing that prefix. This module provides the orchestration layer:
grouping prompts by prefix and tracking cache hit/miss statistics.
The actual KV cache is managed by the underlying inference engine
(vLLM), accessed via base_rollout_fn.

Works WITHOUT vllm — base_rollout_fn is a plain callable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix-cached rollout."""
    enable_prefix_caching: bool = True
    cache_size: int = 128  # max number of cached prefixes
    prewarm: bool = True


class PrefixCacheRollout:
    """
    Prefix-caching rollout wrapper.

    Groups prompts by their shared prefix, so the underlying engine can
    reuse the KV cache for prompts sharing a prefix. Tracks cache
    hit/miss statistics.

    Usage:
        config = PrefixCacheConfig()
        rollout = PrefixCacheRollout(config, base_rollout_fn)
        rollout.cache_prefix("You are a helpful assistant.")
        responses = rollout.generate(prompts, params={})
        stats = rollout.get_cache_stats()
    """

    def __init__(
        self,
        config: PrefixCacheConfig,
        base_rollout_fn: Callable[[list[str], dict], list[str]],
    ):
        self.config = config
        self.base_rollout_fn = base_rollout_fn
        # prefix_hash -> placeholder (real KV cache lives in the engine)
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def cache_prefix(self, prefix: str) -> str:
        """
        Pre-compute and cache a prefix.

        Returns the cache key (hash) for the prefix.
        """
        key = self._hash_prefix(prefix)
        self._cache[key] = {
            "prefix": prefix,
            "length": len(prefix),
            "prewarmed": self.config.prewarm,
        }
        # Evict oldest entries if over capacity
        if len(self._cache) > self.config.cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return key

    def generate(self, prompts: list[str], params: Optional[dict] = None) -> list[str]:
        """
        Generate responses, grouping prompts by shared prefix.

        Prompts are expected to be full strings. If a prompt starts with
        a cached prefix, it's a cache hit. Prompts are grouped by their
        matching cached prefix (or "no prefix" group) and each group is
        sent to base_rollout_fn in one call.
        """
        params = params or {}
        if not self.config.enable_prefix_caching:
            return self.base_rollout_fn(prompts, params)

        # Group prompts by matching cached prefix
        groups: dict[str, list[int]] = {}  # prefix_key -> list of prompt indices
        for i, prompt in enumerate(prompts):
            matched_key = self._match_prefix(prompt)
            if matched_key is not None:
                self._hits += 1
            else:
                self._misses += 1
            key = matched_key or "__no_prefix__"
            groups.setdefault(key, []).append(i)

        # Call base_rollout_fn per group, reassemble in original order
        responses: list[Optional[str]] = [None] * len(prompts)
        for key, indices in groups.items():
            group_prompts = [prompts[i] for i in indices]
            group_responses = self.base_rollout_fn(group_prompts, params)
            for idx, resp in zip(indices, group_responses):
                responses[idx] = resp

        return [r if r is not None else "" for r in responses]

    def get_cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear the prefix cache (does not reset stats)."""
        self._cache.clear()

    def _hash_prefix(self, prefix: str) -> str:
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]

    def _match_prefix(self, prompt: str) -> Optional[str]:
        """Return the cache key of the longest cached prefix that matches."""
        best_key = None
        best_len = -1
        for key, entry in self._cache.items():
            prefix = entry["prefix"]
            if prompt.startswith(prefix) and len(prefix) > best_len:
                best_key = key
                best_len = len(prefix)
        return best_key
