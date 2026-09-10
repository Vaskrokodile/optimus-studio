# Capability Frontier Sampler + EIG / Intelligence Density
## Design doc for the Optimus Studio adaptive RL curriculum system

**Scope:** This doc specifies the `Capability Frontier Sampler` and the `EIG / Intelligence Density` measurement for the RSI / RL harness. It is written against the existing tasksets, grader, tool-calling engine, RSI loop store, and storage layer already in `iloptimus/`.

**Repo anchors used:**
- `iloptimus/core/tasksets.py:8-286` — `TasksetInfo`, `TASKSET_REGISTRY` (11 `tool-*` tasksets, eval configs).
- `iloptimus/core/grader.py:133-573` — `GradedResult`, `grade_response(domain, task_idx, response)`, `build_prompt(domain, task_idx)`, `_TOOL_DOMAINS`.
- `iloptimus/core/rsi_loops.py:120-233` — `RsiLoop`, `RsiLoopStore`, persistent under `~/.iloptimus/rsi-loops/`.
- `iloptimus/core/storage.py:12-75` — `app_home()`, `atomic_write_json()`.
- `il_toolcalling_core/il_toolcalling_core/engine.py:42-197` — `ToolCallTrace`, `score_trajectory(task, response)` with `distractor_hits`, `errors`, `calls_made`, `optimal_calls`, `selection`, `efficiency`.
- `iloptimus/core/rl_factory/training/curriculum_selector.py:39-331` — existing UCB/frontier selector (noted below; the new modules should absorb or supersede it).

---

## 1. Prior art summary

**ACCEL** (Parker-Holder et al., ICML 2022) — *Evolving Curricula with Regret-Based Environment Design*.
- Mechanism: keeps a replay buffer of previously generated levels and repeatedly **edits/mutates the highest-regret levels** to produce new levels at the frontier. Regret is approximated as the gap between an estimated optimal return and the current student's return.
- Signal it optimizes: **regret** (expected improvement from training on that level).
- What to steal: maintain a **frontier buffer of task instances** and, instead of always sampling the same static tasks, **mutate difficulty knobs** (e.g. add distractor tools, increase pipeline length, raise error-injection rate) on tasks that show high regret / learning progress.

**PLR / Prioritized Level Replay** (Jiang et al., ICML 2021).
- Mechanism: for each procedurally-generated level, maintain a score of **future learning potential** (originally the TD-error or positive-value surprise from the last rollout on that level). Sample the next training level proportionally to that score.
- Signal it optimizes: **TD-error / value surprise** as a proxy for how much the policy will learn from replaying that level.
- What to steal: use a **per-task rolling score** based on the magnitude of the change in the model's success probability after training on that task. Update sampling weights after every rollout.

**PAIRED** (Dennis et al., NeurIPS 2020) — *Unsupervised Environment Design*.
- Mechanism: a three-agent game — a **protagonist** (the model being trained), an **antagonist** (a strong reference solver), and an **adversary/teacher** that generates environment parameters. The teacher maximizes **regret = antagonist return − protagonist return**, which prevents the teacher from generating impossible environments.
- Signal it optimizes: **regret gap** constrained by solvability.
- What to steal: use the **expert trajectories** already present in every IL taskset (`task.expert`) as the antagonist. For any task we can compute an approximate regret = `1 − correctness` (since the expert solves it). Tasks with high regret but non-zero success are frontier tasks.

**LLM self-play / RL for reasoning** (DeepSeek-R1, OpenAI o1, Open-RS, ReST-RL, SkyRL-Gym).
- **DeepSeek-R1** (DeepSeek-AI, 2025): trains base model with **GRPO** on verifiable correctness rewards only, no SFT warm-up for the first stage. The model naturally develops reflection, verification, and long CoT.
- **OpenAI o1-style**: inference-time search / self-play with a reward model; the curriculum is the distribution of hard, verifiable problems.
- **Open-RS** (Dang & Ngo, 2026): GRPO on small models with a compact curated math set; uses **two-stage context-length curriculum** (4k → 8k) and DAPO-style optimizations.
- **ReST-RL** (THUDM, 2025): two-stage pipeline — (1) **ReST-GRPO** filters and assembles high-reward data, (2) **VM-MCTS** trains a value model for test-time decoding guidance.
- **SkyRL-Gym** (NovaSky, 2025): a Gymnasium API for text/tool-based LLM RL environments with multi-turn rollouts.
- What to steal:
  - Use **rule-based correctness reward only** (we already have `graded_result.correctness` from `grade_response`).
  - Adopt a **two-stage rollout budget** (short rollouts for selection, longer rollouts for training).
  - Wrap IL tool tasksets into a Gymnasium-compatible `BaseTextEnv` (optional future work).
  - Use **GRPO / group-relative advantage** for policy updates, not a learned value model for V1.

**Regret / learning progress / entropy of success rate.**
- **Regret**: `R(task) = V*(task) − Vπ(task)`. With binary correctness and an expert solver, `V* = 1`, so `R(task) = 1 − success_rate(task)`.
- **Learning progress**: `LP(task, t) = success_rate_t(task) − success_rate_{t−w}(task)`. Positive LP means the task is currently teachable; negative LP means it may be too hard or the model is regressing.
- **Entropy of success rate**: for a task with success probability `p`, Bernoulli entropy `H(p) = −p log₂ p − (1−p) log₂(1−p)` is maximized at `p = 0.5`. At `p ≈ 0.05–0.30` (our target frontier band) entropy is still high, meaning the outcome is informative.
- What to steal: use **learning progress as the primary curriculum signal**, and use **entropy / uncertainty as an exploration bonus** so the sampler does not collapse on already-mastered tasks.

**Thompson / UCB sampling over difficulty bands.**
- **Thompson Sampling**: maintain a Beta posterior over each task's success probability; sample `p̂ ~ Beta(α, β)` and pick the task with the highest sample in the target band.
- **UCB1**: `score = mean + c · sqrt(ln(N) / n)` where `N` is total attempts and `n` is task attempts.
- What to steal: combine them — **sample from the Beta posterior (Thompson)**, then add a **UCB exploration bonus** for tasks that have been attempted fewer than `min_attempts`.

---

## 2. Frontier estimation algorithm

### 2.1 Per-task success-rate model
Every `(domain, task_idx)` pair is an independent Bernoulli arm. Maintain a **Beta posterior**:

```
prior:    α₀ = 1, β₀ = 1     # uniform prior, cold-start friendly
after s successes and f failures:
          α = α₀ + s
          β = β₀ + f
posterior mean:     p̂ = α / (α + β)
Thompson sample:     p̃ ~ Beta(α, β)
```

This is maintained in `TaskProfile` (see Section 6).

### 2.2 Target band
The requested **5–30% success band** is stored as a tunable tuple:

```python
FRONTIER_BAND: tuple[float, float] = (0.05, 0.30)
```

A task is in the frontier when its **Thompson sample** `p̃` falls inside `[0.05, 0.30]`. We use the Thompson sample, not the posterior mean, for selection because it naturally encodes uncertainty and avoids premature exploitation.

### 2.3 Per-domain vs global
- **Per-domain frontier**: for each `tool-*` domain, independently estimate which `task_idx` arms lie in the frontier.
- **Global scheduling**: do not let a single domain dominate. Maintain a **domain weight** `w_d` proportional to the domain's recent learning progress (`LP_d`) and inversely proportional to its saturation (`1 − max_d sr`). This ensures all 11 tool domains get sampled even as some saturate.

### 2.4 Cold start
- For any task with `attempts < MIN_ATTEMPTS` (default `2`), force it into the candidate set with an **infinite UCB exploration bonus** and a high Thompson sample drawn from `Beta(1,1)`.
- Initial global fallback: if no task has been attempted at all, sample **uniformly** over all `(domain, task_idx)` arms for the first `K` rollouts.

### 2.5 Drift detection (model improved → frontier moved)
Maintain two correctness windows per domain:

```python
short_window: deque[bool]  # maxlen = 20
long_window:  deque[bool]  # maxlen = 100
```

Detect upward drift with a Page-Hinkley-style test:

```
μ_long  = mean(long_window)
σ_long  = std(long_window)
μ_short = mean(short_window)
if μ_short > μ_long + DRIFT_THRESHOLD * σ_long / sqrt(len(long_window)):
    declare "frontier_up"
```

`DRIFT_THRESHOLD` default `3.0`. When drift is declared:
1. Raise the band midpoint slightly (optional; leave band fixed for V1, but reset the long window so old failures do not mask new competence).
2. Reset the `long_window` for that domain to the last 50 observations so the posterior re-centers on the improved model.
3. Emit a `frontier_up` event to the loop store.

---

## 3. EIG / Intelligence Density metric

### 3.1 "Capability gain"
For a single task, capability gain over a rolling window `W = 20` is:

```python
def capability_gain(score_history: deque[float]) -> float:
    if len(score_history) < W:
        return 0.0
    recent = list(score_history)[-W//2:]    # last 10
    older  = list(score_history)[-W:-W//2]   # preceding 10
    return mean(recent) - mean(older)
```

`score_history` stores the `GradedResult.score` (0..1) for every rollout on that task.

### 3.2 Token cost
From every rollout record:

```python
token_cost = prompt_tokens + completion_tokens
```

For tool-calling tasks, `completion_tokens` must include every `<tool>{...}</tool>` block emitted by the model.

### 3.3 FLOPs
Use a cheap approximation. For a model with `P` parameters generating `T` tokens:

```python
flops_per_token = 2 * P
flops = T * flops_per_token
```

Use `params_b` from the `ModelInfo` record (already available in the web backend) converted to integer parameter count.

### 3.4 Exact Intelligence Density (ID)

```python
def intelligence_density(
    gain: float,
    token_cost: float,
    flops: float,
) -> float:
    denominator = token_cost * flops
    if denominator <= 0:
        return 0.0
    return gain / denominator
```

Per-domain aggregate:

```python
ID_domain = sum(task.gain for task in domain) / sum(task.tokens * task.flops for task in domain)
```

Interpretation: how much the model's success probability improves **per token-FLOP**. Benchmark environments can be ranked by `ID`; a task with high `ID` is a high-quality training instrument.

---

## 4. Information Gain metric

### 4.1 IG over binary success/fail per task
For a `TaskProfile` with Beta posterior `Beta(α, β)`, the differential entropy of the Beta distribution is:

```python
import math
from scipy.special import digamma  # or implement manually

def beta_entropy(alpha: float, beta: float) -> float:
    return (
        math.lbeta(alpha, beta)
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (alpha + beta - 2) * digamma(alpha + beta)
    )
```

When a new batch of `b` observations (successes/failures) is added:

```python
IG = beta_entropy(alpha_old, beta_old) - beta_entropy(alpha_old + b_success, beta_old + b_fail)
```

If `scipy` is unavailable, use the empirical Bernoulli approximation:

```python
def bernoulli_entropy(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
```

### 4.2 IG over tool-selection distribution
For tool-calling domains, `ToolCallTrace.calls` gives the sequence of tool names. Compute the empirical entropy over tool-name frequencies in the rollout:

```python
def tool_selection_entropy(calls: list[dict]) -> float:
    names = [c["name"] for c in calls if c.get("name") != "__parse_error__"]
    if not names:
        return 0.0
    counts = Counter(names)
    total = len(names)
    H = 0.0
    for count in counts.values():
        p = count / total
        H -= p * math.log2(p)
    return H
```

To estimate `IG_tool`:
- Compute `H_before` from the **previous window** of rollouts on this task.
- Compute `H_after` from the **current window**.
- `IG_tool = H_before - H_after`.

A large drop means the model is converging to a sharper tool-selection policy (useful signal), while an increase means exploration. For the curriculum, prefer tasks where `IG_tool` is high in magnitude (model is still learning the tool policy).

### 4.3 Combined IG score
For selection weighting:

```python
IG_score(task) = w_bin * IG_binary(task) + w_tool * abs(IG_tool(task))
```

Default `w_bin = 0.7`, `w_tool = 0.3`. This score is used as a multiplier in the sampler (Section 5).

---

## 5. Sampler algorithm

### 5.1 Inputs
- `profile: FrontierProfile`
- `frontier_band = (0.05, 0.30)`
- `min_attempts = 2`
- `ucb_c = 1.414`
- `ig_weight = 0.5`

### 5.2 Pseudocode

```python
def sample_next_task(profile: FrontierProfile) -> RolloutKey:
    total_attempts = profile.total_attempts
    candidates: list[tuple[str, int, float]] = []

    for domain, dprof in profile.domains.items():
        for task_idx, tprof in dprof.tasks.items():
            p_thompson = random.betavariate(tprof.alpha, tprof.beta)

            in_band = profile.frontier_band[0] <= p_thompson <= profile.frontier_band[1]
            under_explored = tprof.attempts < min_attempts

            if in_band or under_explored:
                n = max(1, tprof.attempts)
                exploration = ucb_c * math.sqrt(math.log(total_attempts + 1) / n)
                ucb = p_thompson + exploration

                ig = estimated_information_gain(tprof)
                score = ucb * (1.0 + ig_weight * max(0.0, ig))

                # domain progress multiplier
                dp = domain_progress(dprof)
                score *= dp

                candidates.append((domain, task_idx, score))

    if not candidates:
        # Fallback: no task estimated in frontier (all saturated or all impossible)
        # Pick the task whose Thompson sample is closest to the band center
        target = (profile.frontier_band[0] + profile.frontier_band[1]) / 2
        best = None
        best_dist = float("inf")
        for domain, dprof in profile.domains.items():
            for task_idx, tprof in dprof.tasks.items():
                p = tprof.alpha / (tprof.alpha + tprof.beta)
                dist = abs(p - target)
                if dist < best_dist:
                    best_dist = dist
                    best = (domain, task_idx)
        if best:
            return RolloutKey(domain=best[0], task_idx=best[1])
        # ultimate fallback: uniform random
        domain = random.choice(list(profile.domains.keys()))
        task_idx = random.choice(list(profile.domains[domain].tasks.keys()))
        return RolloutKey(domain=domain, task_idx=task_idx)

    # Sample proportional to score
    domains, tasks, scores = zip(*candidates)
    probs = np.array(scores, dtype=float)
    probs /= probs.sum()
    idx = np.random.choice(len(candidates), p=probs)
    return RolloutKey(domain=domains[idx], task_idx=tasks[idx])
```

### 5.3 Domain progress multiplier
```python
def domain_progress(dprof: DomainProfile) -> float:
    lp = learning_progress(list(dprof.short_window), list(dprof.long_window))
    saturation = mean(list(dprof.long_window)) if dprof.long_window else 0.0
    # Encourage domains that are improving and not fully saturated
    return max(0.1, 1.0 + lp) * (1.0 - 0.5 * saturation)
```

---

## 6. Integration plan

### 6.1 New files to create

#### `iloptimus/core/frontier_sampler.py`
Core dataclasses and the sampler.

```python
"""Capability Frontier Sampler for adaptive RL curricula."""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .storage import app_home, atomic_write_json


MIN_ATTEMPTS = 2
FRONTIER_BAND: tuple[float, float] = (0.05, 0.30)
UCB_C = 1.414
IG_WEIGHT = 0.5
DRIFT_THRESHOLD = 3.0
WINDOW_SHORT = 20
WINDOW_LONG = 100


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
    score_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    token_costs: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    tool_entropies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
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
    tasks: dict[int, TaskProfile] = field(default_factory=dict)
    short_window: deque[bool] = field(default_factory=lambda: deque(maxlen=WINDOW_SHORT))
    long_window: deque[bool] = field(default_factory=lambda: deque(maxlen=WINDOW_LONG))
    drift_state: str = "stable"

    def update(self, task_idx: int, correct: bool) -> None:
        self.short_window.append(correct)
        self.long_window.append(correct)
        if self._upward_drift():
            self.drift_state = "frontier_up"
            # Trim long window so posterior recentering is faster
            keep = list(self.long_window)[-WINDOW_LONG // 2:]
            self.long_window = deque(keep, maxlen=WINDOW_LONG)

    def _upward_drift(self) -> bool:
        if len(self.long_window) < WINDOW_LONG:
            return False
        mu_long = float(np.mean(self.long_window))
        std_long = float(np.std(self.long_window)) or 1e-6
        mu_short = float(np.mean(self.short_window)) if self.short_window else 0.0
        return mu_short > mu_long + DRIFT_THRESHOLD * std_long / math.sqrt(WINDOW_LONG)


@dataclass
class FrontierProfile:
    model_id: str
    checkpoint: str = ""
    version: int = 1
    domains: dict[str, DomainProfile] = field(default_factory=dict)
    frontier_band: tuple[float, float] = FRONTIER_BAND
    total_attempts: int = 0
    records: list[RolloutRecord] = field(default_factory=list)

    def ensure_task(self, taskset_id: str, domain: str, task_idx: int) -> TaskProfile:
        dprof = self.domains.setdefault(domain, DomainProfile(domain=domain))
        if task_idx not in dprof.tasks:
            dprof.tasks[task_idx] = TaskProfile(
                taskset_id=taskset_id, domain=domain, task_idx=task_idx
            )
        return dprof.tasks[task_idx]


class FrontierSampler:
    def __init__(self, profile: FrontierProfile):
        self.profile = profile

    def sample(self) -> RolloutKey:
        # ... implementation from Section 5
        pass

    def observe(
        self,
        key: RolloutKey,
        graded: dict[str, Any],
        tokens: int,
        flops: float,
        tool_entropy: float = 0.0,
    ) -> None:
        tprof = self.profile.ensure_task(key.taskset_id, key.domain, key.task_idx)
        score = float(graded.get("score", 0.0))
        correct = bool(graded.get("correctness", 0.0) >= 1.0)
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

    def save(self) -> None:
        path = app_home() / "profiles" / self.profile.model_id / "frontier.json"
        atomic_write_json(path, asdict(self.profile))
```

#### `iloptimus/core/capability_metrics.py`
Metrics implementations.

```python
"""EIG and Intelligence Density computations."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from scipy.special import digamma  # fallback to Bernoulli approximation if missing


def bernoulli_entropy(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def beta_entropy(alpha: float, beta: float) -> float:
    try:
        return (
            math.lbeta(alpha, beta)
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (alpha + beta - 2) * digamma(alpha + beta)
        )
    except Exception:
        return bernoulli_entropy(alpha / (alpha + beta))


def information_gain_binary(
    alpha_before: float,
    beta_before: float,
    successes: int,
    failures: int,
) -> float:
    alpha_after = alpha_before + successes
    beta_after = beta_before + failures
    return max(0.0, beta_entropy(alpha_before, beta_before) - beta_entropy(alpha_after, beta_after))


def tool_selection_entropy(calls: list[dict]) -> float:
    names = [c.get("name") for c in calls if c.get("name") and c.get("name") != "__parse_error__"]
    if not names:
        return 0.0
    total = len(names)
    H = 0.0
    for count in Counter(names).values():
        p = count / total
        H -= p * math.log2(p)
    return H


def capability_gain(score_history: deque[float], window: int = 20) -> float:
    if len(score_history) < window:
        return 0.0
    seq = list(score_history)
    half = window // 2
    recent = seq[-half:]
    older = seq[-window:-half]
    return sum(recent) / len(recent) - sum(older) / len(older)


def intelligence_density(
    capability_gain: float,
    token_cost: int,
    flops: float,
) -> float:
    denom = token_cost * flops
    return capability_gain / denom if denom > 0 else 0.0


@dataclass
class RolloutMetrics:
    score: float
    correctness: float
    tokens: int
    flops: float
    ig_binary: float
    ig_tool: float
    intelligence_density: float
    capability_gain: float
```

### 6.2 Modify `iloptimus/core/rsi_loops.py`

Add frontier-aware fields and methods to `RsiLoop`:

```python
@dataclass
class RsiLoop:
    # ... existing fields ...
    frontier_profile_id: str = ""       # usually == model_id
    active_domains: list[str] = field(default_factory=list)
    status: str = "draft"
    ...
```

Add a new method (or standalone function) `next_task` in `rsi_loops.py` that owns the sampling step:

```python
from .frontier_sampler import FrontierProfile, FrontierSampler

def next_task(loop: RsiLoop, profile: FrontierProfile | None = None) -> tuple[FrontierProfile, dict]:
    """Pick the next (domain, task_idx) for this loop."""
    if profile is None:
        profile = load_frontier_profile(loop.frontier_profile_id or loop.model_id)
    sampler = FrontierSampler(profile)
    key = sampler.sample()
    return profile, {
        "domain": key.domain,
        "task_idx": key.task_idx,
        "taskset_id": key.taskset_id,
    }
```

Also wire the **run step**. The existing `rsi_worker.py` is a generic file-editing agent, not a taskset grader. For the new curriculum loop, add an execution path in `rsi_loops.py` (or a new `FrontierLoopRunner`):

```python
def run_frontier_iteration(
    loop: RsiLoop,
    model_generate: Callable[[str], tuple[str, int]],
) -> dict:
    profile = load_frontier_profile(loop.model_id)
    profile, selection = next_task(loop, profile)

    domain = selection["domain"]
    task_idx = selection["task_idx"]

    prompt = build_prompt(domain, task_idx)
    response, tokens = model_generate(prompt)
    flops = tokens * 2 * estimate_params(loop.model_id)

    graded = grade_response(domain, task_idx, response)
    tool_entropy = 0.0
    if domain.startswith("tool-"):
        # Re-parse tool calls for IG
        from il_toolcalling_core.engine import parse_tool_calls
        tool_entropy = tool_selection_entropy(parse_tool_calls(response))

    sampler = FrontierSampler(profile)
    sampler.observe(
        RolloutKey(domain=domain, task_idx=task_idx, taskset_id=_taskset_id_for_domain(domain)),
        graded={"score": graded.score, "correctness": graded.correctness},
        tokens=tokens,
        flops=flops,
        tool_entropy=tool_entropy,
    )
    sampler.save()

    loop.iterations_done += 1
    loop.score_history.append(graded.score)
    loop.best_score = max(loop.best_score, graded.score)
    return {"selection": selection, "graded": graded.public(), "tokens": tokens}
```

### 6.3 Persistence

Use the existing `storage.atomic_write_json`:

```python
FRONTIER_DIR = app_home() / "profiles"
FRONTIER_DIR.mkdir(parents=True, exist_ok=True)

def frontier_profile_path(model_id: str) -> Path:
    return FRONTIER_DIR / model_id / "frontier.json"

def capability_metrics_path(model_id: str) -> Path:
    return FRONTIER_DIR / model_id / "capability_metrics.json"
```

`FrontierProfile.save()` already calls `atomic_write_json` in the new module.

### 6.4 Existing `curriculum_selector.py` note

`iloptimus/core/rl_factory/training/curriculum_selector.py` already implements a UCB/frontier selector for the RL training path. For V1:
- Keep it as a **thin wrapper** over the new `FrontierSampler` class, OR
- Mark it deprecated and update `rl_factory/training/__init__.py` to export `FrontierSampler` instead.

Recommended: **replace `CurriculumSelector` internals with `FrontierSampler`** and keep the same public API (`select`, `update`) so existing training code does not break.

### 6.5 Tool-calling integration

The 11 `tool-*` domains are enumerated in `iloptimus/core/grader.py:423-434` and `tests/smoke_toolcalling.py:10-20`. Each taskset has 4 tasks. The sampler treats each `(domain, task_idx)` as an arm. The domain list for the profile should be populated from `_TOOL_DOMAINS` keys.

---

## 7. Validation plan

### 7.1 Test target
Run the sampler end-to-end against the 11 IL tool-calling tasksets **without a real model** by using a configurable stub model.

### 7.2 Stub model
`tests/test_frontier_sampler.py`:

```python
import random
from iloptimus.core.grader import grade_response, build_prompt, _TOOL_DOMAINS
from iloptimus.core.frontier_sampler import FrontierProfile, FrontierSampler
from iloptimus.core.capability_metrics import (
    capability_gain,
    intelligence_density,
    information_gain_binary,
    tool_selection_entropy,
)

TARGET_SUCCESS = {
    "tool-fs":       0.25,
    "tool-sql":      0.10,
    "tool-web":      0.05,
    "tool-booking":  0.60,
    "tool-pipeline": 0.15,
    "tool-recovery": 0.20,
    "tool-distractor": 0.30,
    "tool-parallel": 0.35,
    "tool-interpreter": 0.45,
    "tool-devops":   0.08,
    "tool-api":      0.12,
}

def stub_generate(domain: str, task_idx: int) -> tuple[str, int]:
    """Return a response that succeeds with the configured domain rate."""
    target = TARGET_SUCCESS[domain]
    roll = random.random()
    if roll < target:
        # Return the expert response (deterministically correct)
        from tests.smoke_toolcalling import expert_response, _taskset_tasks
        task = _taskset_tasks(domain)[task_idx]
        return expert_response(task), 120
    else:
        # Return a broken response
        bad = "<reasoning>bad</reasoning>\n<tool>{\"name\":\"nonexistent_tool\",\"args\":{}}</tool>\n<answer>wrong</answer>"
        return bad, 80
```

For `tests/smoke_toolcalling.py`, refactor the task-loading helper:

```python
def _taskset_tasks(domain: str):
    pkg = _TOOL_DOMAINS[domain]
    tasks_mod = _load_module(f"{pkg}_tasks", str(_taskset_path(pkg, "tasks.py")))
    return tasks_mod.TASKS
```

### 7.3 Assertions
After `N = 500` selections:

1. **Frontier concentration**: at least `50%` of selected `(domain, task_idx)` arms have true success rate in `[0.05, 0.30]`.
2. **Cold-start coverage**: every `(domain, task_idx)` arm is selected at least once in the first `100` iterations.
3. **Drift migration**: after running `100` iterations, raise `TARGET_SUCCESS` for `tool-fs` from `0.25` to `0.90` and run `100` more iterations; assert the sampler reduces selection frequency for `tool-fs` tasks.
4. **Metric sanity**:
   - `information_gain_binary(1, 1, 3, 7) > information_gain_binary(1, 1, 9, 1)` (uncertain posteriors yield higher IG).
   - `intelligence_density(0.2, 100, 1e12) > intelligence_density(0.05, 1000, 1e12)` (higher gain and lower cost is better).
   - `capability_gain` is positive when a task's recent success window improves.
5. **Persistence round-trip**: save `FrontierProfile`, reload via `json.loads(path.read_text())`, assert `total_attempts` unchanged.
6. **Tool entropy**: for a tool domain rollout with one correct tool call, `tool_selection_entropy` is `0.0`; for a rollout with three distinct distractors and one correct call, entropy is `> 1.5`.

### 7.4 Acceptance criteria
- `pytest tests/test_frontier_sampler.py` passes.
- The sampler does not collapse onto a single task or domain.
- The 5–30% frontier band is respected after the cold-start phase.

---

## 8. Immediate next steps

1. Create `iloptimus/core/frontier_sampler.py` and `iloptimus/core/capability_metrics.py` with the dataclasses and functions above.
2. Add `next_task` / `run_frontier_iteration` to `iloptimus/core/rsi_loops.py`.
3. Write `tests/test_frontier_sampler.py` using the stub model and the 11 tool domains.
4. Once validated, replace the internals of `iloptimus/core/rl_factory/training/curriculum_selector.py` with `FrontierSampler` to unify the RL training and RSI loop curriculums.
