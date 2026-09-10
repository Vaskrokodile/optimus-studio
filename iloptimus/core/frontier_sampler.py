"""Capability Frontier Sampler — adaptive RL curriculum over (domain, task) arms.

Each ``(domain, task_idx)`` pair is an independent Bernoulli arm with a Beta
posterior. The sampler draws a Thompson sample per arm, keeps arms whose sample
falls in the 5-30% frontier band (or that are under-explored), adds a UCB
exploration bonus and an information-gain multiplier, weights by per-domain
learning progress, and samples proportionally. A Page-Hinkley-style drift test
re-centers a domain's posterior when the model improves.

See ``RESEARCH_frontier_sampler.md``.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .capability_metrics import information_gain_binary, learning_progress
from .storage import atomic_write_json, profiles_dir

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MIN_ATTEMPTS = 2
FRONTIER_BAND: tuple[float, float] = (0.05, 0.30)
UCB_C = 1.414
IG_WEIGHT = 0.5
DRIFT_THRESHOLD = 3.0
WINDOW_SHORT = 20
WINDOW_LONG = 100


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RolloutKey:
    domain: str
    task_idx: int
    rollout_idx: int = 0
    taskset_id: str = ""


@dataclass
class RolloutRecord:
    key: RolloutKey
    timestamp: float
    score: float
    correctness: float
    tokens: int = 0
    flops: float = 0.0
    tool_entropy: float = 0.0


@dataclass
class TaskProfile:
    taskset_id: str = ""
    domain: str = ""
    task_idx: int = 0
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    successes: int = 0
    score_history: deque = field(default_factory=lambda: deque(maxlen=100))
    token_costs: deque = field(default_factory=lambda: deque(maxlen=100))
    tool_entropies: deque = field(default_factory=lambda: deque(maxlen=100))
    last_selected: float = 0.0

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, score: float, correct: bool, tokens: int, tool_entropy: float = 0.0) -> None:
        self.attempts += 1
        if correct:
            self.successes += 1
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.score_history.append(score)
        self.token_costs.append(tokens)
        self.tool_entropies.append(tool_entropy)
        self.last_selected = time.time()


@dataclass
class DomainProfile:
    domain: str
    tasks: dict = field(default_factory=dict)
    short_window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SHORT))
    long_window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_LONG))
    drift_state: str = "stable"

    def update(self, task_idx: int, correct: bool) -> None:
        self.short_window.append(correct)
        self.long_window.append(correct)
        if self._upward_drift():
            self.drift_state = "frontier_up"
            keep = list(self.long_window)[-WINDOW_LONG // 2:]
            self.long_window = deque(keep, maxlen=WINDOW_LONG)

    def _upward_drift(self) -> bool:
        if len(self.long_window) < WINDOW_LONG:
            return False
        mu_long = _mean(self.long_window)
        std_long = _std(self.long_window) or 1e-6
        mu_short = _mean(self.short_window) if self.short_window else 0.0
        return mu_short > mu_long + DRIFT_THRESHOLD * std_long / math.sqrt(WINDOW_LONG)


@dataclass
class FrontierProfile:
    model_id: str
    checkpoint: str = ""
    version: int = 1
    domains: dict = field(default_factory=dict)
    frontier_band: list = field(default_factory=lambda: list(FRONTIER_BAND))
    total_attempts: int = 0
    records: list = field(default_factory=list)

    def ensure_task(self, taskset_id: str, domain: str, task_idx: int) -> TaskProfile:
        dprof = self.domains.setdefault(domain, DomainProfile(domain=domain))
        if task_idx not in dprof.tasks:
            dprof.tasks[task_idx] = TaskProfile(
                taskset_id=taskset_id, domain=domain, task_idx=task_idx
            )
        return dprof.tasks[task_idx]

    @property
    def band(self) -> tuple:
        return (float(self.frontier_band[0]), float(self.frontier_band[1]))


# ---------------------------------------------------------------------------
# Statistics helpers (no hard numpy dependency)
# ---------------------------------------------------------------------------


def _mean(seq) -> float:
    s = list(seq)
    return sum(s) / len(s) if s else 0.0


def _std(seq) -> float:
    s = list(seq)
    if not s:
        return 0.0
    m = sum(s) / len(s)
    var = sum((x - m) ** 2 for x in s) / len(s)
    return math.sqrt(var)


def _weighted_choice(items: list) -> Any:
    """Sample one item proportional to its (non-negative) weight.

    ``items`` is a list of ``(value, weight)`` pairs.
    """
    weights = [max(0.0, w) for _, w in items]
    total = sum(weights)
    if total <= 0:
        return random.choice(items)[0]
    r = random.random() * total
    acc = 0.0
    for value, w in items:
        acc += w
        if r <= acc:
            return value
    return items[-1][0]


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class FrontierSampler:
    """Selects the next (domain, task_idx) arm from a ``FrontierProfile``."""

    def __init__(self, profile: FrontierProfile) -> None:
        self.profile = profile

    def sample(self) -> RolloutKey:
        total_attempts = self.profile.total_attempts
        band = self.profile.band

        # Forced cold-start exploration: any arm with fewer than MIN_ATTEMPTS
        # observations is tried first. We pick the least-tried under-explored
        # arm (round-robin) rather than uniformly at random, so every arm is
        # guaranteed coverage within ~MIN_ATTEMPTS * num_arms draws (the doc's
        # "infinite UCB exploration bonus" for cold start) instead of hitting
        # the coupon-collector bound.
        under_explored: list[tuple[int, RolloutKey]] = []
        for domain, dprof in self.profile.domains.items():
            for task_idx, tprof in dprof.tasks.items():
                if tprof.attempts < MIN_ATTEMPTS:
                    under_explored.append((tprof.attempts, RolloutKey(domain=domain, task_idx=task_idx)))
        if under_explored:
            min_attempts = min(a for a, _ in under_explored)
            tied = [k for a, k in under_explored if a == min_attempts]
            return random.choice(tied)

        candidates: list = []

        for domain, dprof in self.profile.domains.items():
            dp = self._domain_progress(dprof)
            for task_idx, tprof in dprof.tasks.items():
                p_thompson = self._thompson_sample(tprof)
                in_band = band[0] <= p_thompson <= band[1]
                under_explored = tprof.attempts < MIN_ATTEMPTS

                if not (in_band or under_explored):
                    continue

                n = max(1, tprof.attempts)
                exploration = UCB_C * math.sqrt(math.log(total_attempts + 1) / n)
                ucb = p_thompson + exploration
                ig = self._estimated_information_gain(tprof)
                score = ucb * (1.0 + IG_WEIGHT * max(0.0, ig))
                score *= dp
                candidates.append((domain, task_idx, score))

        if not candidates:
            target = (band[0] + band[1]) / 2.0
            best = None
            best_dist = float("inf")
            for domain, dprof in self.profile.domains.items():
                for task_idx, tprof in dprof.tasks.items():
                    p = tprof.posterior_mean
                    dist = abs(p - target)
                    if dist < best_dist:
                        best_dist = dist
                        best = (domain, task_idx)
            if best:
                return RolloutKey(domain=best[0], task_idx=best[1])
            domains = list(self.profile.domains.keys())
            if not domains:
                return RolloutKey(domain="", task_idx=0)
            domain = random.choice(domains)
            task_idx = random.choice(list(self.profile.domains[domain].tasks.keys()))
            return RolloutKey(domain=domain, task_idx=task_idx)

        chosen = _weighted_choice(
            [(RolloutKey(domain=d, task_idx=t), s) for d, t, s in candidates]
        )
        return chosen

    def observe(
        self,
        key: RolloutKey,
        graded,
        tokens: int,
        flops: float,
        tool_entropy: float = 0.0,
    ) -> None:
        tprof = self.profile.ensure_task(key.taskset_id, key.domain, key.task_idx)
        if isinstance(graded, dict):
            score = float(graded.get("score", 0.0))
            correctness_raw = float(graded.get("correctness", 0.0))
        else:
            score = float(getattr(graded, "score", 0.0) or 0.0)
            correctness_raw = float(getattr(graded, "correctness", 0.0) or 0.0)
        correct = bool(correctness_raw >= 1.0)
        tprof.update(score, correct, tokens, tool_entropy)

        dprof = self.profile.domains[key.domain]
        dprof.update(key.task_idx, correct)

        self.profile.total_attempts += 1
        self.profile.records.append(
            RolloutRecord(
                key=key,
                timestamp=time.time(),
                score=score,
                correctness=float(correct),
                tokens=tokens,
                flops=flops,
                tool_entropy=tool_entropy,
            )
        )

    # -- internals --------------------------------------------------------

    def _thompson_sample(self, tprof: TaskProfile) -> float:
        try:
            return random.betavariate(tprof.alpha, tprof.beta)
        except (ValueError, ZeroDivisionError):
            return tprof.posterior_mean

    def _estimated_information_gain(self, tprof: TaskProfile) -> float:
        # Expected IG of one more observation from the current posterior.
        p = tprof.posterior_mean
        ig_success = information_gain_binary(tprof.alpha, tprof.beta, 1, 0)
        ig_fail = information_gain_binary(tprof.alpha, tprof.beta, 0, 1)
        return p * ig_success + (1.0 - p) * ig_fail

    def _domain_progress(self, dprof: DomainProfile) -> float:
        lp = learning_progress(dprof.short_window, dprof.long_window)
        saturation = _mean(dprof.long_window) if dprof.long_window else 0.0
        return max(0.1, 1.0 + lp) * (1.0 - 0.5 * saturation)

    # -- persistence ------------------------------------------------------

    def save(self) -> None:
        atomic_write_json(_frontier_path(self.profile.model_id), _serialize_profile(self.profile))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _frontier_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace("\\", "_")
    return profiles_dir() / safe / "frontier.json"


def _serialize_profile(profile: FrontierProfile) -> dict:
    """JSON-safe view: deques become lists, nested dataclasses become dicts."""
    domains = {}
    for dname, dprof in profile.domains.items():
        domains[dname] = {
            "domain": dprof.domain,
            "tasks": {
                str(idx): {
                    "taskset_id": t.taskset_id,
                    "domain": t.domain,
                    "task_idx": t.task_idx,
                    "alpha": t.alpha,
                    "beta": t.beta,
                    "attempts": t.attempts,
                    "successes": t.successes,
                    "score_history": list(t.score_history),
                    "token_costs": list(t.token_costs),
                    "tool_entropies": list(t.tool_entropies),
                    "last_selected": t.last_selected,
                }
                for idx, t in dprof.tasks.items()
            },
            "short_window": list(dprof.short_window),
            "long_window": list(dprof.long_window),
            "drift_state": dprof.drift_state,
        }
    records = [
        {
            "key": asdict(r.key),
            "timestamp": r.timestamp,
            "score": r.score,
            "correctness": r.correctness,
            "tokens": r.tokens,
            "flops": r.flops,
            "tool_entropy": r.tool_entropy,
        }
        for r in profile.records
    ]
    return {
        "model_id": profile.model_id,
        "checkpoint": profile.checkpoint,
        "version": profile.version,
        "frontier_band": list(profile.frontier_band),
        "total_attempts": profile.total_attempts,
        "domains": domains,
        "records": records,
    }


def deserialize_profile(data: dict) -> FrontierProfile:
    profile = FrontierProfile(
        model_id=data["model_id"],
        checkpoint=data.get("checkpoint", ""),
        version=data.get("version", 1),
        frontier_band=list(data.get("frontier_band", list(FRONTIER_BAND))),
        total_attempts=data.get("total_attempts", 0),
    )
    for dname, dd in data.get("domains", {}).items():
        dprof = DomainProfile(
            domain=dname,
            short_window=deque(dd.get("short_window", []), maxlen=WINDOW_SHORT),
            long_window=deque(dd.get("long_window", []), maxlen=WINDOW_LONG),
            drift_state=dd.get("drift_state", "stable"),
        )
        for idx_str, td in dd.get("tasks", {}).items():
            tprof = TaskProfile(
                taskset_id=td.get("taskset_id", ""),
                domain=td.get("domain", ""),
                task_idx=int(td.get("task_idx", idx_str)),
                alpha=td.get("alpha", 1.0),
                beta=td.get("beta", 1.0),
                attempts=td.get("attempts", 0),
                successes=td.get("successes", 0),
                score_history=deque(td.get("score_history", []), maxlen=100),
                token_costs=deque(td.get("token_costs", []), maxlen=100),
                tool_entropies=deque(td.get("tool_entropies", []), maxlen=100),
                last_selected=td.get("last_selected", 0.0),
            )
            dprof.tasks[int(idx_str)] = tprof
        profile.domains[dname] = dprof
    for rd in data.get("records", []):
        key = rd.get("key", {})
        profile.records.append(
            RolloutRecord(
                key=RolloutKey(
                    domain=key.get("domain", ""),
                    task_idx=int(key.get("task_idx", 0)),
                    rollout_idx=int(key.get("rollout_idx", 0)),
                    taskset_id=key.get("taskset_id", ""),
                ),
                timestamp=rd.get("timestamp", 0.0),
                score=rd.get("score", 0.0),
                correctness=rd.get("correctness", 0.0),
                tokens=rd.get("tokens", 0),
                flops=rd.get("flops", 0.0),
                tool_entropy=rd.get("tool_entropy", 0.0),
            )
        )
    return profile


def load_frontier_profile(model_id: str) -> FrontierProfile:
    path = _frontier_path(model_id)
    if path.exists():
        try:
            return deserialize_profile(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return FrontierProfile(model_id=model_id)


__all__ = [
    "FRONTIER_BAND",
    "RolloutKey",
    "RolloutRecord",
    "TaskProfile",
    "DomainProfile",
    "FrontierProfile",
    "FrontierSampler",
    "load_frontier_profile",
    "deserialize_profile",
]
