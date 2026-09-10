# Model Profiler / Skill Graph Design Doc — Optimus Studio

## 1. Prior Art Summary

### Robotics skill graphs / skill libraries
- **RoboSkillFramework** learns visuomotor manipulation skills from teleoperated demonstrations and uses a foundation model (LLM/VLM) for skill selection plus a precondition check before execution.
- **Open-Robot-Skills / Anthropic Agent Skills format** packages skills as bundles with a `SKILL.md` contract, executable scripts, and exit conditions; GaP then composes bundles into executable robot graphs.
- *Steal:* skills should be executable units with explicit pre/post-conditions, and a typed graph decides which unit to run next.

### LLM capability profiling (METR, Apollo, BIG-Bench)
- **METR** evaluates autonomous capability by the length/duration of tasks a model can complete and publishes elicitation guidelines so evals do not underestimate ability.
- **Apollo Research** focuses on behavioral evals of agentic LLMs and model organisms, running thousands of structured environment rollouts (often via the Inspect framework).
- **BIG-Bench** tags each of its 200+ tasks with keywords (`logical-reasoning`, `decomposition`, `computer-code`, `mathematics`, etc.), giving a ready-made skill taxonomy.
- *Steal:* treat each rollout as a noisy observation over multiple skill dimensions and use a keyword/skill ontology to aggregate evidence.

### Skill discovery in RL (DIAYN, Option-Critic, skill chaining)
- **DIAYN** maximizes mutual information between a latent skill and the visited state distribution, discovering diverse options without task reward.
- **Option-Critic** learns both the internal policy of each option and its termination function end-to-end via policy gradients.
- **Skill chaining** (and Deep Skill Chaining / SCaR) builds options whose initiation set is the terminal state distribution of the previous option.
- *Steal:* curriculum should discover ordered sequences where skill A’s output is a precondition for skill B.

### Knowledge tracing (BKT, DKT, PDT)
- **Bayesian Knowledge Tracing** models each skill as a hidden binary state with prior `P(L0)`, learn transition `P(T)`, slip `P(S)`, and guess `P(G)`; it updates `P(L_t)` after each binary response.
- **Deep Knowledge Tracing** replaces the HMM with an RNN, capturing recency, inter-skill similarity, and individual ability at the cost of interpretability.
- **Performance Distribution Tracing (PDT)** is the closest match: it uses a continuous dynamic Bayesian network with **Beta conjugate priors** so skill distributions can be updated analytically online, even for compositions.
- *Steal:* keep a Beta posterior per skill node and update it with fractional pass/fail evidence.

### Compositional skill learning (LAMP, DeCo, Learning to Compose Skills, GraSP / SkillGraph / SkillDAG / HiSkill)
- **LAMP** and **DeCo** show that compositional generalization comes from a discrete skill vocabulary plus a learned composition rule (symbolic PDDL or VLM-retrieved skill schedules).
- **Learning to Compose Skills** trains a differentiable composition function over skill embeddings, enabling recursive, zero-shot composition.
- Recent agent skill graphs (**SkillGraph**, **SkillDAG**, **HiSkill**, **GraSP**) store skills as typed nodes (prerequisite, enhancement, co-occurrence, conflict, precondition-effect) and retrieve a task-relevant subgraph.
- *Steal:* use typed edges and retrieve an ordered subgraph; update edge evidence from rollout traces.

### “What skills compose?” recent work
- **SkillGraph / SkillOrchestra** maintain a skill handbook with agent profiles, routing tasks based on competence and cost.
- **GraSP** compiles flat skill libraries into typed DAGs with precondition-effect edges and locality-bounded repair, reducing replanning from `O(N)` to `O(d^h)`.
- **SkillDAG** exposes the graph as an LLM-callable retrieval interface and lets the model register execution-backed edges so the graph evolves.
- *Steal:* the curriculum generator should query the graph at inference time and write back composition edges after each rollout.

---

## 2. Skill Ontology

Root: **`agentic-intelligence`**
Top-level branches: **`coding`**, **`math`**, **`reasoning`**, **`tool-calling`**, **`agentic`**.

```python
from __future__ import annotations

# DAG as adjacency list.  A child may appear under multiple parents (valid DAG).
SKILL_DAG: dict[str, dict[str, list[str]]] = {
    "agentic-intelligence": {
        "children": ["coding", "math", "reasoning", "tool-calling", "agentic"]
    },
    "coding": {
        "children": [
            "code-implementation", "humaneval", "debugging", "refactoring",
            "edge-case-handling", "test-interpretation",
            "performance-profiling", "architecture-selection",
        ]
    },
    "math": {
        "children": [
            "gsm8k", "aime", "arithmetic", "algebra", "combinatorics",
            "geometry", "word-problems", "competition-math",
        ]
    },
    "reasoning": {
        "children": [
            "reasoning-tasks", "logical-deduction", "constraint-reasoning",
            "probabilistic-reasoning", "multi-step-deduction",
            "cross-module-reasoning",
        ]
    },
    "tool-calling": {
        "children": [
            "tool-selection", "argument-construction", "tool-ordering",
            "efficiency", "error-recovery", "distractor-resistance",
            "tool-fs", "tool-sql", "tool-web", "tool-booking",
            "tool-pipeline", "tool-recovery", "tool-distractor",
            "tool-parallel", "tool-interpreter", "tool-devops", "tool-api",
        ]
    },
    # Tool domain leaf nodes + their narrow skill.
    "tool-fs":        {"children": ["filesystem-navigation"]},
    "tool-sql":       {"children": ["sql-querying"]},
    "tool-web":       {"children": ["web-research"]},
    "tool-booking":   {"children": ["booking-flows"]},
    "tool-pipeline":  {"children": ["pipeline-sequencing"]},
    "tool-recovery":  {"children": ["fault-recovery"]},
    "tool-distractor":{"children": ["distractor-resistance"]},
    "tool-parallel":  {"children": ["parallel-batching"]},
    "tool-interpreter":{"children": ["code-interpretation"]},
    "tool-devops":    {"children": ["devops-triage"]},
    "tool-api":       {"children": ["api-reliability"]},
    # Agentic branch.
    "agentic": {
        "children": ["agentic-reasoning", "agentic-coding"]
    },
    "agentic-reasoning": {
        "children": [
            "sustained-planning", "multi-step-deduction", "cross-module-reasoning"
        ]
    },
    "agentic-coding": {
        "children": [
            "codebase-navigation", "dependency-resolution", "api-discovery",
            "rollback", "failure-diagnosis", "refactoring", "debugging",
            "test-interpretation", "performance-profiling", "architecture-selection",
        ]
    },
}
```

### Taskset domain → active skill nodes

A single task usually exercises several skills. `DOMAIN_TO_SKILLS` maps each registry domain to the leaf/interior nodes that receive evidence.

```python
DOMAIN_TO_SKILLS: dict[str, list[str]] = {
    "coding": [
        "code-implementation", "debugging", "refactoring",
        "edge-case-handling", "test-interpretation", "performance-profiling",
    ],
    "reasoning": [
        "reasoning-tasks", "logical-deduction", "constraint-reasoning",
        "probabilistic-reasoning", "combinatorics",
    ],
    "agentic-reasoning": [
        "agentic-reasoning", "sustained-planning",
        "multi-step-deduction", "cross-module-reasoning",
    ],
    "agentic-coding": [
        "agentic-coding", "codebase-navigation", "debugging", "refactoring",
        "dependency-resolution", "api-discovery", "failure-diagnosis",
        "test-interpretation", "performance-profiling", "architecture-selection",
    ],
    "humaneval": [
        "humaneval", "code-implementation", "edge-case-handling",
        "test-interpretation",
    ],
    "gsm8k": ["gsm8k", "word-problems", "arithmetic", "algebra"],
    "aime": ["aime", "competition-math", "combinatorics", "algebra", "geometry"],
    # Tool domains: domain-specific leaf + cross-cutting tool sub-skills.
    "tool-fs": [
        "tool-fs", "filesystem-navigation",
        "tool-selection", "argument-construction", "efficiency", "distractor-resistance",
    ],
    "tool-sql": [
        "tool-sql", "sql-querying",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-web": [
        "tool-web", "web-research",
        "tool-selection", "argument-construction", "efficiency", "distractor-resistance",
    ],
    "tool-booking": [
        "tool-booking", "booking-flows",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-pipeline": [
        "tool-pipeline", "pipeline-sequencing",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-recovery": [
        "tool-recovery", "fault-recovery", "error-recovery",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-distractor": [
        "tool-distractor", "distractor-resistance",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-parallel": [
        "tool-parallel", "parallel-batching",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-interpreter": [
        "tool-interpreter", "code-interpretation",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-devops": [
        "tool-devops", "devops-triage",
        "error-recovery", "tool-selection", "argument-construction",
        "tool-ordering", "efficiency", "rollback", "failure-diagnosis",
    ],
    "tool-api": [
        "tool-api", "api-reliability",
        "tool-selection", "argument-construction", "efficiency",
    ],
}
```

### Compositional environments
Compositions are not hard-coded in the static DAG; they are generated on demand by the curriculum search. Example high-value combinations to test:

```python
EXAMPLE_COMPOSITIONS = [
    {"debugging", "codebase-navigation"},
    {"debugging", "api-discovery"},
    {"refactoring", "performance-profiling"},
    {"tool-ordering", "error-recovery"},
    {"codebase-navigation", "api-discovery", "dependency-resolution", "debugging"},
]
```

---

## 3. Mastery Estimation

Each skill node maintains a **Beta** posterior over mastery:

```python
from dataclasses import dataclass, field

@dataclass
class SkillNodeState:
    node_id: str
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    successes: float = 0.0
    last_seen: float | None = None
```

### Exact update rule
A rollout produces a graded score `r ∈ [0, 1]`. For every skill node activated by the task:

```
alpha_{t+1} = alpha_t + r
beta_{t+1}  = beta_t + (1 - r)
attempts    += 1
successes   += r
last_seen    = now()
```

Mastery estimate:

```
mu(node) = alpha / (alpha + beta)
```

If the user wants a strict pass/fail threshold, set `r = 1` when `graded.score >= PASS_THRESHOLD` (0.7) else `r = 0`; for partial-credit tasks the continuous `r` is used directly. The prior is `Beta(1, 1)` (uniform).

### Propagation to interior nodes (optional)
When a leaf is updated, also add **a fraction** of the same evidence to each ancestor to keep branch estimates current, e.g. `0.25 * r` per level. This is optional; the curriculum query already computes branch mastery as an aggregate if needed.

### Composition formula
For an untested composition `C = {s1, s2, ..., sk}` we assume conjunctive independence:

```
mu(C) = prod_{s in C} mu(s)
```

This is defensible because a compositional task succeeds only if **all** constituent skills fire. If one skill is weak, the product collapses, which is the desired “weakest-link” behavior. For a **trained** composition node (one that has been explicitly rolled out), we keep its own `CompositionState` with a separate Beta and use that posterior instead of the product; this lets the system learn interaction effects (positive or negative) beyond the independence assumption.

```python
@dataclass
class CompositionState:
    node_ids: frozenset[str]
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    successes: float = 0.0

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)
```

---

## 4. Per-Model Persistence

Profile path:

```
~/.iloptimus/profiles/<model_id>/skill_graph.json
```

Example model IDs: `deepseek-r1-distill-qwen-1.5b`, `boosted-v1-small`, `omnicoder-9b`. Sanitize `/` and `\\` to `_` if a Hugging Face id is ever used directly.

### Profile dataclass schema

```python
from dataclasses import dataclass, field
import time

@dataclass
class SkillGraphProfile:
    model_id: str
    model_fingerprint: str  # e.g. "deepseek-r1-distill-qwen-1.5b+adapter=Akahsizrr/boosted-v1-small"
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    nodes: dict[str, SkillNodeState] = field(default_factory=dict)
    compositions: dict[str, CompositionState] = field(default_factory=dict)
    rollout_count: int = 0
```

`compositions` is keyed by `"+".join(sorted(node_ids))`.

### Multiple models and version drift
- Each `model_id` gets its own directory.
- `model_fingerprint` is computed from the model family, base checkpoint, and adapter repo/checkpoint tag.
- On load, if the fingerprint changed:
  - Archive the old profile to `skill_graph.json.v{timestamp}`.
  - Seed the new profile from the old one with `alpha *= 0.5`, `beta *= 0.5` per node (transfer prior) to reflect uncertainty about the new checkpoint.
  - Increment `version`.
- Writes use `storage.atomic_write_json` to avoid corruption from concurrent processes.

---

## 5. Curriculum Query API

```python
@dataclass(frozen=True)
class SkillComposition:
    node_ids: frozenset[str]
    target_mastery: float = 0.85
    priority_score: float = 0.0
    distance_to_frontier: int = 0
```

```python
def find_unmastered_compositions(
    graph: SkillGraph,
    profile: SkillGraphProfile,
    k: int = 5,
    threshold: float = 0.85,
    max_size: int = 4,
    frontier_distance: int = 2,
) -> list[SkillComposition]:
    ...
```

### Search algorithm
1. `mastered = {n | mu(n) >= threshold}`.
2. `frontier = {unmastered nodes within frontier_distance graph edges of mastered}`.
3. Generate candidate compositions of size `1..max_size` that contain at least one frontier node and whose members are pairwise **adjacent** in the DAG (parent/child, sibling, or linked by a co-occurrence edge from recorded rollouts).
4. Estimate mastery:
   - If the composition has a recorded `CompositionState`, use its posterior.
   - Else use `prod(mu(node) for node in composition)`.
5. Priority score:

```python
gap = max(0.0, threshold - est_mastery)
dist = min_distance_to_mastered(composition, mastered)
novelty = 1.0 / (1.0 + attempts)
size_penalty = 0.05 * (len(node_ids) - 1)

priority = gap - 0.2 * dist - size_penalty + 0.1 * novelty
```

6. Return the top `k` by `priority` (higher = more urgent and closer to the frontier).

The generator calls this and maps the returned `SkillComposition` to one or more taskset tasks (domain + task index) or to a generated compositional episode.

---

## 6. Integration Plan

### New file: `iloptimus/core/skill_graph.py`
- Contains `SKILL_DAG`, `DOMAIN_TO_SKILLS`, `SkillGraph` dataclass, and helper:

```python
def domain_to_active_skills(
    domain: str,
    graded: GradedResult,
) -> list[tuple[str, float]]:
    """Return (skill_node_id, weight) pairs for a rollout."""
```

For tool-calling it also inspects `graded.info` (`tool_selection`, `efficiency`, `errors`, `distractor_hits`) to weight the cross-cutting sub-skills.

### New file: `iloptimus/core/capability_profiler.py`
- `CapabilityProfiler` class:
  - `record_rollout(model_id, domain, task_idx, graded, fingerprint)`
  - `mastery(model_id, node_id) -> float`
  - `composition_mastery(model_id, node_ids) -> float`
  - `suggest_compositions(model_id, loop_kind, k) -> list[SkillComposition]`
  - `load(model_id)`, `save(model_id)`

### Modify `iloptimus/core/grader.py`
- Change signature at line 529:

```python
def grade_response(
    domain: str,
    task_idx: int,
    response: str,
    model_id: str = "",
) -> GradedResult:
```

- At the end of `grade_response` (after line 568) add:

```python
if model_id:
    from .capability_profiler import CapabilityProfiler
    profiler = CapabilityProfiler()
    profiler.record_rollout(
        model_id=model_id,
        domain=domain,
        task_idx=task_idx,
        graded=result,
        fingerprint=...,  # passed in by caller or read from loaded model
    )
```

- Update all call sites that already call `grade_response`:
  - `iloptimus/core/pipeline.py:746`
  - `iloptimus/core/benchmark.py:102`
  - `iloptimus/core/sft.py:100`
  - Any custom-environment path inside `grade_response` itself should forward `model_id` to `score_task` if available.

### Modify `iloptimus/core/rsi_loops.py`
- Add fields to `RsiLoop` dataclass (lines 120–137):

```python
current_focus: list[str] = field(default_factory=list)
focus_history: list[list[str]] = field(default_factory=list)
rollouts_on_focus: int = 0
```

- Add a helper:

```python
def plan_next_composition(
    loop: RsiLoop,
    profiler: CapabilityProfiler,
    k: int = 1,
) -> SkillComposition | None:
    compositions = profiler.suggest_compositions(
        model_id=loop.model_id,
        loop_kind=loop.kind,
        k=k,
    )
    if not compositions:
        return None
    comp = compositions[0]
    loop.current_focus = sorted(comp.node_ids)
    loop.focus_history.append(loop.current_focus)
    loop.rollouts_on_focus = 0
    return comp
```

- `server.py` calls `plan_next_composition` when creating or resuming an RSI loop and embeds the focus in the loop prompt:

```python
focus = plan_next_composition(loop, profiler)
if focus:
    loop_prompt += f"\n\nThis session focus: {', '.join(focus.node_ids)}."
```

### Modify `iloptimus/core/storage.py`
Add:

```python
def profiles_dir() -> Path:
    return app_home() / "profiles"
```

---

## 7. Validation Plan

Create `tests/test_capability_profiler.py`.

### Test 1: Stub model with known per-skill strength
```python
def make_stub_result(score: float) -> GradedResult:
    return GradedResult(
        score=score,
        correctness=float(score >= 0.7),
        reasoning_quality=score,
    )
```

- Configure stub model A: `code-implementation` successes at `p=0.9`, `debugging` at `p=0.1`.
- Run 50 rollouts per domain through `CapabilityProfiler.record_rollout`.
- Assert `mastery("code-implementation") > 0.8` and `mastery("debugging") < 0.3`.
- Assert `suggest_compositions` returns a composition containing `debugging` before one containing `code-implementation`.

### Test 2: Composition interaction detection
- Stub model B: `codebase-navigation` and `api-discovery` both `p=0.9`, but the explicit `{codebase-navigation, api-discovery}` composition succeeds only `p=0.2`.
- Run 20 compositional rollouts.
- Assert `composition_mastery({...})` is near `0.2`, much lower than the product `0.9*0.9`.
- Assert `find_unmastered_compositions` returns that pair.

### Test 3: Persistence roundtrip
- Save a profile, corrupt the file with a write collision, then load and verify `atomic_write_json` leaves a consistent JSON.
- Verify `model_fingerprint` mismatch triggers transfer/reset and version increment.

### Test 4: Curriculum ordering
- Build a minimal graph with three nodes A→B→C.
- Set A mastered, B low, C unobserved.
- Call `find_unmastered_compositions` and assert B (distance 1) is ranked before C (distance 2) and single-node B appears before `{B, C}` due to size penalty.

### End-to-end smoke test
- Launch a short RSI loop with a stub model, run 3 iterations, then open `~/.iloptimus/profiles/<model_id>/skill_graph.json` and verify non-zero `rollout_count` and updated `alpha`/`beta` values for the focus nodes.
