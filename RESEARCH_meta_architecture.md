# META-ARCHITECTURE: Adaptive Self-Evolving RL Orchestrator for Optimus Studio

**Scope:** Design the integration layer that composes the Model Profiler/Skill Graph, Frontier Sampler, Environment Mutator/Adversarial Generator, Counterfactual Engine, Trajectory Recombination, and Behavioral Environments into a single adaptive, self-evolving RL loop. Also designs the **Persistent World (#16)** and the **Environment-as-RL-agent (#26)**.

**Codebase anchors:**
- `iloptimus/core/rsi_loops.py` — existing RSI loop store/kinds (`RsiLoop`, `RsiLoopStore`, `LOOP_KINDS`).
- `iloptimus/core/tasksets.py` — `TASKSET_REGISTRY`.
- `iloptimus/core/grader.py` — `grade_response`, `build_prompt`.
- `iloptimus/core/stateful_environments.py` — declarative `StateMachineRuntime`.
- `il_toolcalling_core/engine.py` — `ToolCallTrace`, `score_trajectory`.
- `iloptimus/core/storage.py` — `app_home()`, `atomic_write_json()`.
- `iloptimus/server.py` — existing `/api/rsi/loops/*` endpoints.
- `iloptimus/core/rl_factory/training/curriculum_selector.py` — `CurriculumSelector` with a `frontier` mode.
- `iloptimus/core/harness_graph.py` — `HarnessGraphManager`, `ActionNode`, `TaskNode`, `ActionEdge`.
- `iloptimus/core/rl_factory/environments/environment_mutation.py` — `DynamicsMutator`, `MUTATION_TYPES`.
- `iloptimus/core/rl_factory/environments/adversarial_counterfactual.py` — counterfactual generator/verifier.
- `iloptimus/core/rl_factory/environments/trajectory_recombination.py` — `TrajectoryRecombinationVerifier`.
- `iloptimus/core/rl_factory/core/base.py` — `Problem`, `BaseReasoningEnv`.
- `iloptimus/core/rl_factory/core/batch_base.py` — `BatchEnvBase`.

---

## 1. Prior art summary

### Open-ended RL / environment generation
- **POET / Enhanced POET (Uber AI, 2019/2020)** pairs a population of environments with a population of agents, mutates environment parameters, and periodically transfers an agent to a new environment if it performs better there. The key mechanism is *co-evolution + transfer* across a growing archive of challenges. We can reuse the idea of maintaining a **task archive** and letting successful model checkpoints (or trajectories) transfer between task specs.
- **DeepMind XLand / Open-Ended Learning (2021)** procedurally generates 3D multi-agent games from a compact rule language and meta-trains a single agent across the entire task distribution. The insight is **one unified world + a task grammar** can produce an unbounded curriculum. Our Persistent World is the Optimus Studio analog: one company sim with a task-derivation grammar.
- **OMNI-EPIC (ICLR 2025)** uses a foundation model to generate Python environment code and reward functions, keeps an archive of learned tasks, and scores novelty/difficulty/interestingness. The mechanism we steal is **LLM-as-environment-builder**: the generator agent writes or mutates task specs in JSON/Python, and a verifier+scorer decides whether the task is admitted.
- **AMIGo (FAIR, 2020)** trains a teacher that proposes *adversarially motivated intrinsic goals* to a student; the teacher reward is a function of whether the goal is reached and how hard it was. The sweet spot is “not too easy, not impossible.” We use this directly for the generator-agent objective: reward comes from model failure **but only if the task is solvable**.
- **PAIRED (Google/UC Berkeley, NeurIPS 2020)** treats the environment generator as a third RL agent in a three-player game: protagonist, antagonist, and adversary. The adversary maximizes **regret** (antagonist best return − protagonist average return), which prevents unsolvable levels. We use regret as the generator reward and implement an “antagonist” as an oracle/expert rollout or a stronger model.
- **AlphaStar league** maintains a continuously updated league of agents, branches new competitors from old ones, and samples from the Nash distribution. We can reuse the league idea as a **population of model checkpoints** and a **population of generator policies** that play against each other.
- **AlphaEvolve / FunSearch** combine LLM code mutation with automated evaluators and an evolutionary population. We steal the pattern of **LLM-mutated code → verifier score → population update**, applied to task-spec and verifier code, not solution code.
- **RAGEN / StarPO** frame multi-turn LLM agents as RL and find that stable training requires trajectory-level rewards, reasoning-aware reward, and diverse initial states. We apply this by making the orchestrator operate on **trajectories**, not single responses, and by using fine-grained verifier signals.
- **Self-Rewarding LMs / Self-Questioning / SCOPE / EvoEnv** show that a model can co-evolve task proposer and solver policies, with the proposer rewarded for difficulty and novelty. We use the same proposer/solver asymmetry, but the proposer is explicitly the *environment generator* and the reward is rule-based (regret, novelty, compute cost).

### Persistent world simulators
- **WebArena** provides self-hostable, stateful web sites (e-commerce, dev, forum, CMS) and tasks that require long-horizon interaction; state persists across HTTP actions. We borrow the concept of a **stateful service with tool endpoints** and a final-state verifier.
- **SWE-bench / SWE-bench-Live** use a real Docker repo checkout as persistent state: each task is a GitHub issue and the repo state evolves as the agent edits files. We use this for the company “code” module: an evolving repo with bug/issue tasks.
- **τ-bench** simulates dynamic tool-agent-user conversations with a database; success is a state-diff. We use this for customer/operations tasks: the model uses policy-aware API tools and the world DB updates.
- **OSWorld / OSWorld 2.0** run agents in a real OS where files, apps, and desktop state persist. We use the same persistence rule: **every tool call changes the world**, not a reset environment.
- **CoreCraftSimulator** (vibrantlabsai) is the closest prior art: a Smallville-style company sim with employees, customers, tickets, and a shared SQLite database, plus a Task Miner that targets a `k/n` fail rate. We adopt the tick engine and task-miner pattern.

---

## 2. The orchestrator loop

### 2.1 Data that flows between steps

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SkillTarget:
    """Output of the profiler / frontier sampler."""
    skill_key: str
    composition: tuple[str, ...]          # e.g. ("tool_call:read_file", "tool_call:grep_search")
    estimated_success_rate: float
    recommended_difficulty: float
    rationale: str

@dataclass
class TaskSpec:
    """A concrete task the model will attempt."""
    task_id: str
    prompt: str
    domain: str                           # "coding", "tool-fs", "agentic", "persistent_world", ...
    source: str                           # "taskset", "mutated", "generated", "persistent_world"
    difficulty: float
    metadata: dict[str, Any] = field(default_factory=dict)
    token_budget: int = 2048
    verifier_config: dict[str, Any] = field(default_factory=dict)

@dataclass
class Rollout:
    """One trajectory produced by the model."""
    trajectory_id: str
    task: TaskSpec
    response: str
    score: float
    success: bool
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    final_state: dict[str, Any] | None
    verifier_info: dict[str, Any]

@dataclass
class CounterfactualBranch:
    """A mutation of a rollout (or the task) used to stress-test robustness."""
    branch_id: str
    parent_id: str
    mutation_type: str                    # "hidden_constraint", "requirement_change", etc.
    mutated_task: TaskSpec
    rollout: Rollout
    delta_score: float

@dataclass
class RecombinedCandidate:
    """A candidate built by splicing pieces of multiple rollouts."""
    candidate_id: str
    source_ids: list[str]
    response: str
    score: float
    passes: bool

@dataclass
class AnalyzedEpisode:
    """Result of scoring rollouts + counterfactuals + recombination."""
    episode_id: str
    task: TaskSpec
    best_score: float
    chosen_response: str
    frontier_info_gain: float
    skill_updates: list[tuple[str, float]]   # (skill_key, outcome)
    failure_spec: dict[str, Any] | None
    training_pairs: list[dict[str, Any]]

@dataclass
class TrainingSignal:
    """Payload handed to the trainer (GRPO/PPO/IL)."""
    episode_id: str
    prompt: str
    chosen: str
    rejected: list[str]
    reward: float
    advantage: float
    metadata: dict[str, Any]
```

### 2.2 One iteration pseudocode

```python
def adaptive_iteration(orchestrator: AdaptiveOrchestrator) -> TrainingSignal:
    # ------------------------------------------------------------------
    # Step 1: Profiler queries the skill graph for unmastered compositions
    # ------------------------------------------------------------------
    # Uses HarnessGraphManager (harness_graph.py:193) which already stores
    # ActionNode, TaskNode and co-occurrence ActionEdge objects.
    frontier_candidates = profiler.find_frontier_compositions(
        min_success=0.05,
        max_success=0.30,
        min_edge_count=3,
    )

    # ------------------------------------------------------------------
    # Step 2: Frontier sampler picks a (skill, difficulty) target
    # ------------------------------------------------------------------
    # Wraps CurriculumSelector in "frontier" mode
    # (rl_factory/training/curriculum_selector.py:80-331).
    target = frontier_sampler.select(
        frontier_candidates,
        target_band=(0.05, 0.30),
    )

    # ------------------------------------------------------------------
    # Step 3: Environment generator produces a task
    # ------------------------------------------------------------------
    # EnvGeneratorAgent is an LLM (see Section 4). It can:
    #   - mutate an existing task
    #   - compose skills into a new task
    #   - pull a task from the persistent world
    task = env_generator.produce_task(
        target=target,
        recent_failures=orchestrator.recent_failures,
        recent_tasks=orchestrator.recent_tasks,
        persistent_world=orchestrator.world,
    )

    # ------------------------------------------------------------------
    # Step 4: Rollouts
    # ------------------------------------------------------------------
    # N trajectories, possibly from a population of model checkpoints.
    # For coding/reasoning we can use the existing grader.py path;
    # for tool-calling we use ToolCallTrace / score_trajectory.
    rollouts: list[Rollout] = rollout_worker.collect(
        task=task,
        n=orchestrator.config.rollouts_per_task,
        temperature=orchestrator.config.temperature,
    )

    # ------------------------------------------------------------------
    # Step 5: Counterfactual engine branches each rollout
    # ------------------------------------------------------------------
    # Reuses adversarial_counterfactual.py and environment_mutation.py.
    branches: list[CounterfactualBranch] = []
    for r in rollouts:
        for mutation in counterfactual_engine.mutations_for(r):
            branch = counterfactual_engine.branch(r, mutation)
            branches.append(branch)

    # ------------------------------------------------------------------
    # Step 6: Trajectory recombination produces spliced candidates
    # ------------------------------------------------------------------
    # Reuses TrajectoryRecombinationVerifier (trajectory_recombination.py:249)
    # for coding tasks; for text tasks we splice at tool-call boundaries.
    candidates: list[RecombinedCandidate] = recombination_engine.splice(
        rollouts + [b.rollout for b in branches],
    )

    # ------------------------------------------------------------------
    # Step 7: Analyzer scores everything and updates the skill graph
    # ------------------------------------------------------------------
    analyzed = analyzer.score_and_learn(
        task=task,
        rollouts=rollouts,
        branches=branches,
        candidates=candidates,
    )
    profiler.apply_updates(analyzed.skill_updates)
    skill_graph.resolve_task(
        task_id=task.task_id,
        success=analyzed.best_score >= 1.0,
        score=analyzed.best_score,
    )

    # ------------------------------------------------------------------
    # Step 8: RL update — emit the training signal
    # ------------------------------------------------------------------
    signal = build_training_signal(analyzed)
    return signal
```

### 2.3 Concrete re-use of existing components

| Component | Existing file | How it is used |
|-----------|---------------|----------------|
| Skill graph / profiler | `iloptimus/core/harness_graph.py` (`HarnessGraphManager`, `ActionNode`, `TaskNode`, `ActionEdge`) | Nodes become skill nodes; `resolve_task()` back-propagates outcomes. |
| Frontier sampler | `iloptimus/core/rl_factory/training/curriculum_selector.py` (`CurriculumSelector` mode `"frontier"`, `_select_frontier` at line 218) | Selects tasks whose recent success rate is 5–30%. |
| Static task source | `iloptimus/core/tasksets.py` (`TASKSET_REGISTRY` line 22, `get_all_tasksets` line 246) | Bootstraps the loop before mutation. |
| Grading | `iloptimus/core/grader.py` (`grade_response` line 529, `build_prompt` line 575) | Scores responses and builds prompts for taskset domains. |
| Tool-calling rollouts | `il_toolcalling_core/engine.py` (`ToolCallTrace` line 52, `score_trajectory` line 138) | Replays `<tool>` blocks through deterministic simulators. |
| State-machine episodes | `iloptimus/core/stateful_environments.py` (`StateMachineRuntime` line 180, `validate_simulator` line 86) | Runs persistent-world task episodes. |
| Environment mutation | `iloptimus/core/rl_factory/environments/environment_mutation.py` (`DynamicsMutator` line 373, `MUTATION_TYPES` line 51) | Mutates a base problem into hidden-constraint / requirement-change / resource-constraint variants. |
| Counterfactual generation | `iloptimus/core/rl_factory/environments/adversarial_counterfactual.py` (generator line 118, verifier line 162) | Generates “break the solution” scenarios. |
| Trajectory recombination | `iloptimus/core/rl_factory/environments/trajectory_recombination.py` (`TrajectoryRecombinationVerifier` line 249, `recombine` line 346) | Splices multi-checkpoint coding solutions. |
| RL environment base | `iloptimus/core/rl_factory/core/base.py` (`Problem` line 45, `BaseReasoningEnv` line 65) and `batch_base.py` (`BatchEnvBase` line 35) | Standardizes task spec and batch rollout API. |

---

## 3. Persistent World design (#16)

### 3.1 World state schema

The world is a single JSON object stored at:

```
~/.iloptimus/worlds/<world_id>/state.json
```

with periodic checkpoints in:

```
~/.iloptimus/worlds/<world_id>/checkpoints/<tick>.json
```

and an append-only event log:

```
~/.iloptimus/worlds/<world_id>/events.jsonl
```

Schema:

```python
WORLD_SCHEMA: dict[str, Any] = {
    "world_id": "company-001",
    "tick": 0,
    "sim_time": "2026-01-01T00:00:00Z",
    "employees": [
        {
            "id": "e1",
            "role": "sre",
            "skill": "devops",
            "morale": 0.80,
            "load": 0.40,
            "salary": 120_000,
            "on_call": True,
        },
    ],
    "customers": [
        {
            "id": "c1",
            "tier": "enterprise",
            "satisfaction": 0.75,
            "churn_risk": 0.20,
            "open_ticket": None,
        },
    ],
    "finances": {
        "cash": 1_000_000,
        "revenue": 100_000,
        "expenses": 80_000,
        "runway_months": 12.0,
    },
    "servers": [
        {
            "id": "srv-3",
            "health": 0.55,
            "cpu_load": 92,
            "disk_usage": 0.78,
            "last_restart_tick": 0,
            "role": "api",
        },
    ],
    "code": {
        "repo": "shop",
        "bug_count": 12,
        "open_prs": 5,
        "test_pass_rate": 0.87,
        "tech_debt_score": 0.30,
    },
    "inventory": {
        "sku_count": 340,
        "low_stock_count": 7,
        "backlog": 23,
    },
    "competitors": [
        {
            "id": "comp-1",
            "market_share": 0.25,
            "price_index": 1.00,
        },
    ],
    "contracts": [
        {
            "id": "ct-7",
            "value": 50_000,
            "deadline_tick": 168,
            "status": "in_negotiation",
            "sla_breach": False,
        },
    ],
    "market": {
        "trend": 0.02,
        "demand_index": 1.0,
        "season": "post_holiday",
    },
    "policies": {
        "max_refund": 500,
        "sla_hours": 24,
    },
    "unresolved_tickets": [],
}
```

### 3.2 Time advancement

Each model action (or each tool call, depending on granularity) advances `tick` by 1 and `sim_time` by one simulated hour. The world evolves deterministically:

```python
def advance_world(world: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    world["tick"] += 1
    world["sim_time"] = add_hours(world["sim_time"], 1)

    for server in world["servers"]:
        # Degrade health when overloaded
        if server["cpu_load"] > 80:
            server["health"] -= 0.01 * (server["cpu_load"] / 100.0)
        server["health"] = clamp(server["health"], 0.0, 1.0)

        # Apply any tool effects (restart, scale, patch)
        for action in actions:
            if action.get("target") == server["id"]:
                apply_server_action(world, server, action)

    for customer in world["customers"]:
        if customer.get("open_ticket"):
            customer["churn_risk"] += 0.02
        else:
            customer["churn_risk"] -= 0.01
        customer["churn_risk"] = clamp(customer["churn_risk"], 0.0, 1.0)

    # Competitors may randomly drop prices, which changes market demand
    for competitor in world["competitors"]:
        if random.random() < 0.02:
            competitor["price_index"] -= 0.05
            world["market"]["demand_index"] += 0.03

    # Finance update once per 24 simulated ticks
    if world["tick"] % 24 == 0:
        revenue = world["finances"]["revenue"] * world["market"]["demand_index"]
        world["finances"]["cash"] += revenue - world["finances"]["expenses"]
        world["finances"]["runway_months"] = (
            world["finances"]["cash"] / world["finances"]["expenses"]
            if world["finances"]["expenses"] > 0 else 999.0
        )
```

### 3.3 Task generation FROM the world

At each orchestrator step, `PersistentWorld.derive_task` scans the world for active triggers and returns a `TaskSpec` whose `initial_state` is the **current world state**, not a fresh reset.

```python
TRIGGER_RULES = [
    ("debugging",       lambda w: [(s["id"], s) for s in w["servers"] if s["health"] < 0.60]),
    ("negotiation",     lambda w: [(c["id"], c) for c in w["customers"] if c["churn_risk"] > 0.70]),
    ("finance",         lambda w: [("runway", w["finances"])] if w["finances"]["runway_months"] < 3.0 else []),
    ("operations",      lambda w: [("inventory", w["inventory"])] if w["inventory"]["low_stock_count"] > 5 else []),
    ("coding",          lambda w: [("repo", w["code"])] if w["code"]["bug_count"] > 10 else []),
    ("strategy",        lambda w: [("pricing", c) for c in w["competitors"] if c["price_index"] < 0.90]),
    ("planning",        lambda w: [("contract", c) for c in w["contracts"] if c["deadline_tick"] - w["tick"] < 48 and c["status"] != "signed"]),
]

def derive_task(world: dict[str, Any], focus_skill: str | None = None) -> TaskSpec:
    triggers = []
    for skill, rule in TRIGGER_RULES:
        for key, obj in rule(world):
            triggers.append((skill, key, obj))

    # Prefer the skill the profiler asked for; otherwise sample weighted by urgency
    if focus_skill:
        matches = [t for t in triggers if t[0] == focus_skill]
        trigger = random.choice(matches) if matches else random.choice(triggers)
    else:
        trigger = random.choice(triggers)

    skill, key, obj = trigger
    prompt = render_task_prompt(world, skill, key, obj)

    return TaskSpec(
        task_id=f"world-{world['world_id']}-{world['tick']}-{skill}-{key}",
        prompt=prompt,
        domain="persistent_world",
        source="persistent_world",
        difficulty=estimate_difficulty(world, skill, key),
        metadata={
            "world_id": world["world_id"],
            "tick": world["tick"],
            "trigger": (skill, key),
            "snapshot": copy.deepcopy(world),
        },
    )
```

Example prompts:

| Trigger | Prompt |
|---------|--------|
| `srv-3` degraded | “Server `srv-3` health is 0.55 and CPU load is 92. Diagnose the cause and return the minimal remediation actions. The company finances and SLA are in the world state.” |
| `c1` high churn risk | “Customer `c1` (enterprise tier) has churn risk 0.82 and an open ticket. Use the support tools to recover them without violating the `max_refund` policy.” |
| Competitor price drop | “Competitor `comp-1` dropped `price_index` to 0.85. Update pricing and marketing while preserving runway > 3 months.” |

### 3.4 Using `stateful_environments.py` as the substrate

`stateful_environments.py` already provides `validate_simulator()` (line 86) and `StateMachineRuntime` (line 180). The Persistent World projects a small slice of the world into a state-machine simulator for each episode, then writes the updated scalar values back to the full world JSON.

```python
class PersistentWorld:
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.root = app_home() / "worlds" / world_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.checkpoint_every = 10
        self.world = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        world = copy.deepcopy(WORLD_SCHEMA)
        world["world_id"] = self.world_id
        return world

    def save(self) -> None:
        atomic_write_json(self.state_path, self.world)
        if self.world["tick"] % self.checkpoint_every == 0:
            checkpoint_dir = self.root / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(checkpoint_dir / f"{self.world['tick']}.json", self.world)

    def log_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def derive_task(self, focus_skill: str | None = None) -> TaskSpec:
        # described above
        ...

    def apply_outcome(self, task: TaskSpec, final_state: dict[str, Any], success: bool) -> None:
        """Write the episode's final scalar state back into the world JSON."""
        trigger = task.metadata["trigger"]  # (skill, key)
        # Reverse projection: map simulator scalars to world objects
        if trigger[0] == "debugging":
            server = next(s for s in self.world["servers"] if s["id"] == trigger[1])
            server["health"] = final_state.get("health", server["health"])
            server["cpu_load"] = final_state.get("cpu_load", server["cpu_load"])
            if not success:
                server["health"] -= 0.10  # failure has consequences
        elif trigger[0] == "negotiation":
            customer = next(c for c in self.world["customers"] if c["id"] == trigger[1])
            customer["churn_risk"] = final_state.get("churn_risk", customer["churn_risk"])
            if success:
                customer["satisfaction"] = min(1.0, customer["satisfaction"] + 0.15)
            else:
                customer["churn_risk"] = min(1.0, customer["churn_risk"] + 0.30)
        # ... other projections ...
        self.save()
        self.log_event({
            "tick": self.world["tick"],
            "task_id": task.task_id,
            "trigger": trigger,
            "success": success,
            "final_state": final_state,
        })
```

A projected simulator for a server-debug episode looks like:

```python
{
    "template_id": "persistent-world-server-debug",
    "observation": "server={server_id}, health={health}, cpu={cpu}, tickets={tickets}, reboots={reboots}",
    "state": {
        "server_id": "srv-3",
        "health": 0.55,
        "cpu": 92,
        "tickets": 3,
        "reboots": 0,
        "fixed": False,
    },
    "actions": [
        {
            "name": "check_logs",
            "description": "Read the server log and reduce unknown ticket count",
            "effects": [{"var": "tickets", "op": "add", "value": -1}],
        },
        {
            "name": "restart_server",
            "description": "Reboot the server to clear high CPU",
            "requires": {"var": "cpu", "op": "gt", "value": 80},
            "effects": [
                {"var": "cpu", "op": "set", "value": 40},
                {"var": "reboots", "op": "add", "value": 1},
            ],
        },
        {
            "name": "scale_up",
            "description": "Provision a new instance",
            "requires": {"var": "tickets", "op": "eq", "value": 0},
            "effects": [
                {"var": "cpu", "op": "set", "value": 30},
                {"var": "fixed", "op": "set", "value": True},
            ],
            "reward": 0.2,
        },
    ],
    "terminals": [
        {"when": {"var": "fixed", "op": "eq", "value": True}, "outcome": "resolved", "success": True, "reward": 1.0},
        {"when": {"var": "health", "op": "lte", "value": 0.0}, "outcome": "downtime", "success": False, "reward": -1.0},
    ],
    "max_steps": 12,
}
```

### 3.5 Plug-in to the orchestrator

In `AdaptiveOrchestrator`:

```python
if target.skill_key.startswith("persistent_world") or random.random() < config.world_task_prob:
    task = persistent_world.derive_task(focus_skill=target.skill_key)
else:
    task = env_generator.produce_task(...)
```

This makes the orchestrator treat the Persistent World as **one more generator source**, but with the crucial property that the task state is the current world state.

---

## 4. Environment-as-RL-agent (#26)

### 4.1 Co-evolutionary design

There are two RL agents:

1. **Trainee (model)**: maximizes task reward.
2. **Generator (environment)**: maximizes regret, i.e. tasks the trainee fails but a stronger reference can solve.

In Optimus Studio the generator is an LLM, not a separately trained neural network. Its “policy” is its prompt context, and its “reward” is computed after the trainee rollouts and fed back into its context (or a small bandit/prompt template selector).

### 4.2 Generator action space

The generator returns one JSON object per call:

```json
{
  "op": "mutate_existing" | "compose_skills" | "synthesize_new" | "persistent_world",
  "params": { },
  "difficulty": 0.45,
  "estimated_success_rate": 0.15
}
```

Supported actions:

| `op` | `params` | Meaning |
|------|----------|---------|
| `mutate_existing` | `{"base_task_id": "...", "mutation_type": "hidden_constraint"}` | Take an existing task and apply one of `MUTATION_TYPES` from `environment_mutation.py:51` using `DynamicsMutator` (`environment_mutation.py:373`). |
| `compose_skills` | `{"skill_keys": ["tool_call:read_file", "tool_call:grep_search"], "difficulty": 0.6}` | Build a new task that requires the listed skills in sequence. |
| `synthesize_new` | `{"domain": "coding", "prompt": "Write a function that ...", "verifier": {"type": "code_execution", "tests": [...]}}` | Generate a brand-new task spec with a deterministic verifier. |
| `persistent_world` | `{"trigger_type": "debugging", "urgency": 0.8}` | Request the Persistent World to emit a task derived from current state. |

### 4.3 Generator observation

The LLM prompt contains:

```python
generator_observation = {
    "frontier_skills": profiler.frontier_skills(limit=20),
    "recent_failures": [f.public() for f in recent_failures[-10:]],
    "recent_task_embeddings": recent_task_embeddings[-50:],
    "trainee_pass_rate_by_skill": pass_rate_by_skill,
    "compute_used": compute_used,
    "compute_budget": compute_budget,
}
```

### 4.4 Generator reward

```python
def generator_reward(
    rollouts: list[Rollout],
    task: TaskSpec,
    recent_tasks: list[TaskSpec],
    compute_used: int,
    compute_budget: int,
) -> float:
    pass_rate = sum(r.success for r in rollouts) / len(rollouts)
    did_fail = 1.0 if pass_rate < 1.0 else 0.0

    # Solvability penalty: prefer 5-30% success. Too easy or impossible is bad.
    if pass_rate < 0.05:
        solvability_penalty = 0.5
    elif pass_rate > 0.30:
        solvability_penalty = pass_rate - 0.30
    else:
        solvability_penalty = 0.0

    # Novelty penalty: discourage repeats.
    max_sim = max_cosine_similarity(task.embedding, [t.embedding for t in recent_tasks])
    novelty_penalty = max(0.0, max_sim - 0.70)

    compute_cost = compute_used / max(compute_budget, 1)

    return did_fail - solvability_penalty - novelty_penalty - 0.05 * compute_cost
```

Equivalently, using regret:

```python
regret = best_expert_score - mean_trainee_score
reward = regret - solvability_penalty - novelty_penalty - 0.05 * compute_cost
```

where `best_expert_score` comes from an oracle rollout or a stronger frozen model.

### 4.5 Generator prompt template

```markdown
You are the Adversarial Environment Generator for an adaptive RL training loop.
Your goal is to create a task that the Trainee model finds difficult but can still solve.

### Observation
- Frontier skills (under-trained): {{frontier_skills}}
- Recent failure patterns: {{recent_failures}}
- Compute budget remaining: {{compute_remaining}}
- Recent tasks (do not repeat): {{recent_task_summaries}}

### Allowed actions
Return exactly one JSON object with this shape:

{
  "op": "mutate_existing" | "compose_skills" | "synthesize_new" | "persistent_world",
  "params": { ... },
  "difficulty": 0.0-1.0,
  "estimated_success_rate": 0.0-1.0
}

### Rules
1. Target a trainee success rate between 5% and 30%.
2. The task must be verifiable (has an objective grader or simulator).
3. Do not repeat a recent task; prefer novel skill combinations.
4. If the trainee is failing on `tool_call:grep_search` followed by `tool_call:write_file`, compose a task that requires that exact sequence.
5. If the Persistent World has an active trigger matching a frontier skill, prefer `persistent_world`.

Return only the JSON object.
```

### 4.6 Why this is not a separate trained network

In Optimus Studio the generator is the same local model (or a stronger local model) invoked through the existing inference path. Its updates are **in-context**: the reward and recent outcomes are appended to its prompt history, and a small template/parameter bandit can select among a few prompt variants. This avoids adding a second model to train and matches the current architecture where environments are generated by `run_inference()` / `run_json_completion()` calls.

---

## 5. Session → environment specification (section 24)

A coding/chat/tool-calling session is a trajectory. We extract skills, failure points, and a causal skill-graph fragment, then emit a `FailureSpec` that the generator uses to build adversarial variations.

### 5.1 Extraction algorithm

```python
class FailureSpec:
    session_id: str
    domain: str
    skill_path: list[str]          # ordered skills used before the stuck point
    stuck_skill: str               # the skill where it got stuck
    missing_skill: str | None      # skill that would have unblocked it
    error: str
    difficulty: str
    grade: GradedResult

def extract_failure_spec(session: dict[str, Any], grade: GradedResult) -> FailureSpec:
    # 1. Parse the trajectory into ToolCallTrace format.
    trace = ToolCallTrace(
        calls=parse_tool_calls(session["response"]),
        errors=session.get("errors", []),
        state=session.get("final_state", {}),
    )

    # 2. Identify skill pattern from tool calls.
    skill_path = []
    for call in trace.calls:
        name = call.get("name", "unknown")
        skill_path.append(f"tool_call:{name}")

    # 3. Find the first stuck point.
    stuck_idx = None
    for i, call in enumerate(trace.calls):
        if call.get("error"):
            stuck_idx = i
            break
        # Repeated identical call = stuck
        if i > 0 and call == trace.calls[i - 1]:
            stuck_idx = i
            break

    # If no explicit error but grade is wrong, the final answer is the stuck point.
    if stuck_idx is None and not grade.correctness:
        stuck_idx = max(0, len(trace.calls) - 1)

    stuck_idx = stuck_idx or 0
    stuck_skill = skill_path[stuck_idx] if stuck_idx < len(skill_path) else "final_answer"

    # 4. Infer the missing skill from the error type.
    error = trace.errors[-1] if trace.errors else str(grade.info)
    missing_skill = infer_missing_skill(error, stuck_skill)

    # 5. Build a causal skill graph fragment and feed it to the graph manager.
    graph = HarnessGraphManager()
    task_id = graph.begin_task(session["id"], kind="session")
    for skill in skill_path:
        graph.record_action("skill_used", key=skill, task_id=task_id)
    graph.record_action("failure", key=stuck_skill, task_id=task_id)
    graph.resolve_task(
        task_id,
        success=grade.correctness >= 1.0,
        score=grade.score,
    )

    return FailureSpec(
        session_id=session["id"],
        domain=session.get("domain", "unknown"),
        skill_path=skill_path[: stuck_idx + 1],
        stuck_skill=stuck_skill,
        missing_skill=missing_skill,
        error=error,
        difficulty=session.get("difficulty", "medium"),
        grade=grade,
    )
```

### 5.2 Example

Session: model reads `server.log`, runs `grep` for `ERROR`, then attempts `restart_server` three times without first checking `load`.

- `skill_path`: `["tool_call:read_file", "tool_call:grep_search", "tool_call:restart_server"]`
- `stuck_skill`: `tool_call:restart_server`
- `missing_skill`: `tool_call:check_load`
- `error`: "restart_server returned 503"

The `FailureSpec` is passed to `EnvGeneratorAgent.produce_task` with instruction: “Create a task where the model must `read_file` → `grep_search` → `check_load` → `restart_server`; failure to call `check_load` should cause a deterministic failure.”

### 5.3 How it plugs into the orchestrator

`session_to_env.py` writes `FailureSpec` objects to:

```
~/.iloptimus/orchestrator/<loop_id>/failure_specs.jsonl
```

`EnvGeneratorAgent` reads this file before producing a task and includes the most recent specs in its prompt.

---

## 6. Integration plan

### 6.1 New files

| File | Responsibility |
|------|----------------|
| `iloptimus/core/adaptive_orchestrator.py` | `AdaptiveOrchestrator` class — the main loop described in Section 2. |
| `iloptimus/core/persistent_world.py` | `PersistentWorld` / `CompanyWorld` classes — Section 3. |
| `iloptimus/core/env_generator_agent.py` | `EnvGeneratorAgent` LLM adversarial generator — Section 4. |
| `iloptimus/core/session_to_env.py` | `FailureSpec` extractor and causal skill-graph fragment builder — Section 5. |

### 6.2 New dataclasses (in `adaptive_orchestrator.py`)

```python
@dataclass
class AdaptiveConfig:
    loop_id: str
    model_id: str
    rollouts_per_task: int = 8
    temperature: float = 0.6
    frontier_min: float = 0.05
    frontier_max: float = 0.30
    max_iterations: int = 50
    checkpoint_every: int = 5
    world_task_prob: float = 0.25

@dataclass
class OrchestratorState:
    loop_id: str
    iteration: int
    best_score: float
    score_history: list[float]
    recent_tasks: list[TaskSpec]
    recent_failures: list[FailureSpec]
    persistent_world_id: str | None
```

### 6.3 Modify `iloptimus/core/rsi_loops.py`

Add `"adaptive"` to `LOOP_KINDS` at line 24 and a template to `LOOP_TEMPLATES` at line 26:

```python
LOOP_KINDS = ["coding", "reasoning", "math", "tool-calling", "agentic", "adaptive"]

LOOP_TEMPLATES.append({
    "id": "loop-adaptive-self-evolve",
    "kind": "adaptive",
    "name": "Adaptive self-evolving loop",
    "objective": (
        "Run the adaptive orchestrator: query the skill graph, sample the frontier, "
        "generate an environment, collect rollouts, run counterfactuals and recombination, "
        "analyze outcomes, and emit a training signal."
    ),
    "default_minutes": 60,
    "default_iterations": 10,
    "needs_sandbox": False,
})
```

The existing `RsiLoop` dataclass at line 120 and `RsiLoopStore.save()` at line 177 already persist loops under `~/.iloptimus/rsi-loops/`; no change needed.

### 6.4 Modify `iloptimus/server.py`

Add orchestrator status and world-state endpoints near the existing RSI loop routes (after line 2714). Example:

```python
# ---- Adaptive orchestrator endpoints ----------------------------------
from .core.adaptive_orchestrator import AdaptiveOrchestrator, OrchestratorStore
from .core.persistent_world import PersistentWorld

orchestrator_store = OrchestratorStore(root=app_home() / "orchestrator")

@app.get("/api/rsi/loops/{loop_id}/adaptive/state")
async def get_adaptive_state(loop_id: str):
    loop = rsi_loops.get(loop_id)
    if not loop or loop.kind != "adaptive":
        raise HTTPException(404, "Adaptive loop not found")
    state = orchestrator_store.load(loop_id)
    return state.public()

@app.post("/api/rsi/loops/{loop_id}/adaptive/step")
async def run_adaptive_step(loop_id: str, request: Request):
    loop = rsi_loops.get(loop_id)
    if not loop or loop.kind != "adaptive":
        raise HTTPException(404, "Adaptive loop not found")
    payload = await request.json()
    return orchestrator_store.step(loop_id, payload)

@app.get("/api/worlds/{world_id}")
async def get_world(world_id: str):
    world = PersistentWorld.load(world_id)
    if not world:
        raise HTTPException(404, "World not found")
    return world.public()

@app.post("/api/worlds/{world_id}/derive_task")
async def derive_world_task(world_id: str, payload: dict[str, Any]):
    world = PersistentWorld.load(world_id)
    if not world:
        raise HTTPException(404, "World not found")
    focus = payload.get("focus_skill")
    return world.derive_task(focus).public()
```

### 6.5 Persistence layout

```
~/.iloptimus/
├── rsi-loops/
│   └── loop-<id>.json              # existing (rsi_loops.py:142-177)
├── orchestrator/
│   └── <loop_id>/
│       ├── state.json              # OrchestratorState
│       ├── episodes.jsonl          # one line per iteration
│       ├── failure_specs.jsonl     # from session_to_env.py
│       └── checkpoints/
│           └── <iteration>.json
└── worlds/
    └── <world_id>/
        ├── state.json              # full company world
        ├── events.jsonl            # append-only log
        └── checkpoints/
            └── <tick>.json
```

---

## 7. Phased rollout

### Phase 1 — Build now: orchestrator + frontier sampler + static taskset sampling

**What is buildable immediately**
- `AdaptiveOrchestrator` skeleton and dataclasses.
- `FrontierSampler` wrapping `CurriculumSelector` (`rl_factory/training/curriculum_selector.py:80`) in `"frontier"` mode with `frontier_min=0.05`, `frontier_max=0.30`.
- Profiler backed by `HarnessGraphManager` (`harness_graph.py:193`).
- Task source = existing `TASKSET_REGISTRY` (`tasksets.py:22`) graded by `grade_response`/`build_prompt` (`grader.py:529`/`575`).

**What to validate**
- The loop can run end-to-end: select a skill → pick a task from a taskset → generate N rollouts → grade them → update the skill graph → emit a `TrainingSignal`.
- `CurriculumSelector` actually selects tasks in the 5–30% success band.
- `HarnessGraphManager` node weights move after `resolve_task()`.

**Signal**
- Per-episode `best_score` and the fraction of selected tasks whose pass rate falls in `[0.05, 0.30]`.

### Phase 2 — Add env mutator + adversarial generator + counterfactual engine

**What is buildable**
- `EnvGeneratorAgent` using the prompt template from Section 4.5, producing `mutate_existing` tasks via `DynamicsMutator` (`environment_mutation.py:373`) and `synthesize_new` tasks.
- Counterfactual branching using `adversarial_counterfactual.py` (generator line 118, verifier line 162).
- Trajectory recombination using `TrajectoryRecombinationVerifier` (`trajectory_recombination.py:249`).

**What to validate**
- Generator tasks are solvable (pass rate > 5%) and not trivial (pass rate < 30%).
- Recombined candidates score higher than the best individual rollout on multi-checkpoint coding tasks.
- Counterfactual branches expose robustness gaps (e.g., a solution that ignores a hidden constraint fails after mutation).

**Signal**
- `regret` per generator task; improvement in recombined score over best individual; failure-mode coverage.

### Phase 3 — Add persistent world + environment-as-agent + session-to-env

**What is buildable**
- `PersistentWorld` / `CompanyWorld` (`persistent_world.py`) with the schema and trigger rules from Section 3.
- Full generator-as-RL-agent loop: generator reward feedback, novelty penalty, compute-cost penalty.
- `session_to_env.py` extracts `FailureSpec` from chat/coding sessions and seeds the generator.

**What to validate**
- Tasks derived from the persistent world have non-trivial pass rates and their outcomes persistently change the world state.
- New failures from sessions produce adversarial variations that the model eventually solves.
- The world generates debugging, planning, finance, negotiation, coding, operations, and strategy tasks from one state.

**Signal**
- World `tick` and `cash/runway/customer_count` trend; cross-skill transfer measured by success rate on tasks from new triggers.

---

## 8. Validation plan

### 8.1 Stub model

```python
class StubModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.i = 0
    def generate(self, prompt: str) -> str:
        r = self.responses[self.i % len(self.responses)]
        self.i += 1
        return r
```

### 8.2 Stub components

- **Stub env generator**: returns a fixed list of `TaskSpec` objects alternating easy/hard so the frontier sampler can be tested.
- **Stub profiler**: returns a synthetic skill graph with known success rates.
- **Stub persistent world**: deterministically sets `server["health"] = 0.5` every 5 ticks.
- **Stub grader**: returns `score = 0.1 * len(response)` for a known prompt set, giving a controllable pass band.

### 8.3 Unit tests to run

1. **Frontier sampler**
   ```python
   selector = CurriculumSelector(mode="frontier", frontier_min=0.05, frontier_max=0.30)
   for t, correct in [("easy", True), ("hard", False), ("medium", False)]:
       selector.update(t, reward=1.0 if correct else 0.0, correct=correct)
   chosen = selector.select()
   assert chosen in ("hard", "medium")  # not the easy one
   ```

2. **State-machine runtime for world**
   ```python
   sim = validate_simulator(PROJECTED_SIM)
   runtime = StateMachineRuntime(sim)
   result = runtime.step("check_logs")
   assert result.state["tickets"] == 2
   ```

3. **Trajectory recombination**
   ```python
   verifier = TrajectoryRecombinationVerifier(checkpoints=CHECKPOINTS)
   result = verifier.recombine([good_partial, other_partial])
   assert result.score > max(good_score, other_score)
   ```

4. **Dataclass invariant**
   ```python
   ep = run_one_stub_iteration()
   assert ep.best_score >= 0.0
   assert ep.task.source in ("taskset", "mutated", "generated", "persistent_world")
   assert Path(f"~/.iloptimus/orchestrator/{loop_id}/episodes.jsonl").exists()
   ```

5. **Session → failure spec**
   ```python
   spec = extract_failure_spec(stub_session, grade=GradedResult(score=0.0, ...))
   assert spec.stuck_skill == "tool_call:restart_server"
   assert "check_load" in (spec.missing_skill or "")
   ```

6. **Generator reward**
   ```python
   reward = generator_reward(
       rollouts=[Rollout(success=False), Rollout(success=False)],
       task=new_task,
       recent_tasks=old_tasks,
       compute_used=1000,
       compute_budget=100000,
   )
   assert reward > generator_reward(
       rollouts=[Rollout(success=True), Rollout(success=True)],
       task=same_task,
       recent_tasks=old_tasks,
       compute_used=1000,
       compute_budget=100000,
   )
   ```

### 8.4 Integration smoke test

```python
def test_adaptive_loop_smoke():
    orch = AdaptiveOrchestrator(
        config=AdaptiveConfig(loop_id="test-loop", model_id="stub"),
        model=StubModel(["<answer>42</answer>"] * 100),
    )
    signal = orch.iteration()
    assert isinstance(signal, TrainingSignal)
    assert signal.reward is not None
    assert orch.state.iteration == 1
```

---

## 9. Actions the parent agent should handle

1. **Write this file** to `E:\optimusstudio\iloptimus\RESEARCH_meta_architecture.md` (the subagent has read-only access).
2. **Create the four new modules** (`adaptive_orchestrator.py`, `persistent_world.py`, `env_generator_agent.py`, `session_to_env.py`) and apply the `rsi_loops.py` / `server.py` modifications.
3. **Add tests** under the existing test directory (or create `iloptimus/core/tests/test_adaptive_orchestrator.py`) implementing the validation plan.
4. **Decide on the inference entry point** for the generator LLM (`run_json_completion`, `run_inference`, or a new `run_chat` wrapper) because `env_generator_agent.py` will call the same local model infrastructure used by `server.py`.
5. **Confirm whether the generator is allowed to synthesize verifier code** (i.e., generate Python `verify.py` scripts) or must stay within the existing `grader.py`/`environment_framework.py` grader set; this affects the `synthesize_new` action implementation.
