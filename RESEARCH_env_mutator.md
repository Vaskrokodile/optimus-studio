# Environment Mutator + Adversarial Generator — Design Doc

## Codebase findings (what exists and what is missing)

- **`iloptimus/core/stateful_environments.py`** is a deterministic declarative state-machine runtime.
  - `CONDITION_OPS` / `EFFECT_OPS` are defined at lines 12–13.
  - `validate_simulator()` (lines 86–164) normalizes an env spec into `state`, `actions`, `terminals`, `rewards`, `max_steps`, `scenarios`, `observation`.
  - `StateMachineRuntime` (lines 180–268) replays actions and checks `requires` / `terminals`.
  - `simulate_response()` (lines 289–313) grades a model response by extracting a JSON array of action names, replaying them, and returning `score`, `success`, `outcome`, `trace`.
  - `stateful_tasks()` (lines 332–352) converts a simulator spec into IL task objects.

- **`iloptimus/core/environments.py`** wraps the state-machine spec into an `Environment` object with `kind == "state-machine"`, `tasks`, and `reward` weights (lines 59–143).

- **`il_toolcalling_core/il_toolcalling_core/engine.py`** is the tool-calling simulator.
  - `ToolSpec` (lines 41–48) defines `name`, `description`, `params`, `fn(state,args)->(result,obs)`, `distractor`, `destructive`.
  - `ToolCallTrace` (lines 51–60) records `calls`, `observations`, `errors`, `state`, `goal_reached`, `answer`.
  - `replay()` (lines 94–131) executes parsed `<tool>` blocks against `task.tools` and calls `task.goal()`.
  - `score_trajectory()` (lines 138–197) computes `correctness * (0.55 + 0.25 * selection + 0.20 * efficiency)`.

- **`il_toolcalling_core/il_toolcalling_core/tasks.py`** defines `ToolCallingTask` (lines 11–26). It does **not** currently expose `max_calls`, `faults`, or `budget`, so we must extend it.

- **`iloptimus/core/grader.py`** routes domain grading. `_TOOL_DOMAINS` (lines 423–435) maps `tool-*` domains to packages, and `grade_tool_calling()` (lines 438–460) loads `TASKS[task_idx]` and calls `scoring_mod.score(task, response)`. There is no generic path for an in-memory `ToolCallingTask`, so we add one.

- **`iloptimus/core/rsi_loops.py`** defines loop kinds/templates (lines 24–117) and `RsiLoop` / `RsiLoopStore` (lines 120–184). It currently has no mutation/adversarial task selection hook.

- **`iloptimus/core/rl_factory/environments/environment_mutation.py`** already implements prompt-level mutation for coding problems (lines 50–170). This doc focuses on **world-dynamics** mutation for the two executable substrates above (state machines and tool worlds), which is a different layer.

---

## 1. Prior art summary

### Procedural Content Generation (PCG) for RL
PCGRL (Khalifa et al., AIIDE 2020) frames level generation as an RL policy over content tokens. For Optimus Studio we do not need a learned generator yet: the state/action spaces are small enough that search over explicit mutation operators is cheaper and more interpretable. The transferable idea is the **representation of an environment as a mutable parameter vector** and the use of a solver rollout as the scoring function.

### PAIRED / UED (Dennis et al., NeurIPS 2020)
PAIRED is a three-player game:
- **Protagonist** = the student model/policy `π`.
- **Antagonist** = an oracle or stronger policy `π*` allied with the environment generator.
- **Generator** = chooses the environment parameters `θ`.

The generator maximizes **regret**:

```
regret(θ, π, π*) = V^{π*}(θ) - V^{π}(θ)
```

where `V` is expected return. Because the antagonist must be able to solve the env, the generator cannot create impossible levels; it is driven toward tasks that are **just outside** the protagonist’s ability. At Nash equilibrium the protagonist is minimax-regret optimal. For us, the antagonist is the **expert trajectory / reference solver** already embedded in every `ToolCallingTask` and `stateful_environments` scenario.

### ACCEL (Parker-Holder et al., ICML 2022)
ACCEL maintains a **buffer of high-regret levels** and applies small edits (mutations) to them, compounding complexity over time. Its regret estimate is the **positive value loss**:

```
regret ≈ max(0, R_oracle(env) - R_student(env))
```

and it filters for solvability before adding to the curriculum. We steal the **mutation buffer + editing** pattern: keep a `MutationArchive` of the top-k high-regret mutated envs and mutate those, not the base, to grow a curriculum.

### ARLPCG / self-play environment generation
ARLPG uses a generator network and a solver network; the generator is rewarded by solver failure while a verifier guarantees solvability. The transferable idea is a **GAN-style loop**: generator proposes envs, verifier checks them, solver attempts them, generator reward = solver difficulty minus an unsolvability penalty.

### POET / Enhanced POET / XLand
POET maintains a population of paired environments and agents, uses novelty and transfer to push open-ended complexity. XLand uses a vast space of composable games and a dynamic training task distribution. The steal is an **archive with novelty filtering**: generated environments are only kept if they are both solvable and measurably different from prior ones (cosine distance of tool/state signature).

### AlphaEvolve
AlphaEvolve is an evolutionary coding agent: an LLM proposes code variants, an evaluator executes and scores them, and an outer evolutionary loop keeps the best. We use the same pattern for the **self-generated environment** component: LLM proposes a Python env spec, the verifier executes it, and the loop selects high-regret variants.

### Fault injection / chaos engineering
Chaos engineering deliberately induces failures to expose resilience boundaries. The key principle for agents is **calibrated severity**: every injected fault must still leave a recoverable path. We implement this by requiring the expert trajectory to succeed after the fault is injected.

### Hierarchical RL / temporal abstraction
Sutton et al.’s options framework, FeUdal Networks, and HIRO formalize temporally-extended actions. The transferable idea for #5 is to **identify bottleneck states/actions** where an alternative choice flips the outcome and treat those as the nodes of a compressed SMDP. The high-level policy reasons over critical decisions; the low-level policy executes the primitive calls between them.

---

## 2. Mutation operators

### 2.1 Stateful state-machine mutations

All operators accept `env_spec` as returned by `validate_simulator()` and return a re-validated mutated spec. A helper is used for copying and re-validation:

```python
import copy
import random
from iloptimus.core.stateful_environments import validate_simulator, StateMachineRuntime

def _revalidate(spec: dict) -> dict:
    return validate_simulator(copy.deepcopy(spec))

def _expert_solves(spec: dict, scenario_idx: int = 0) -> bool:
    runtime = StateMachineRuntime(spec, scenario_idx)
    sol = spec["scenarios"][scenario_idx]["solution"]
    for action in sol:
        if runtime.terminated:
            break
        runtime.step(action)
    return runtime.success
```

The operators are intentionally simple; the adversarial search wrapper filters out mutations that break the expert trajectory.

#### 1. `mutate_initial_state`

```python
def mutate_initial_state(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    scenario = random.choice(spec["scenarios"])
    var = random.choice(list(scenario["initial_state"].keys()))
    val = scenario["initial_state"][var]
    if isinstance(val, (int, float)):
        scenario["initial_state"][var] = max(0, val + random.choice([-2, -1, 1, 2]))
    elif isinstance(val, bool):
        scenario["initial_state"][var] = not val
    return _revalidate(spec)
```

#### 2. `change_condition_threshold`

```python
def change_condition_threshold(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    candidates = [a for a in spec["actions"] if a.get("requires")] + spec["terminals"]
    target = random.choice(candidates)
    cond = target.get("requires") or target.get("when")
    if not cond or "var" not in cond:
        return spec
    if cond.get("op") in ("gt", "gte", "lt", "lte"):
        cond["value"] = (cond.get("value") or 0) + random.choice([-1, 1])
        cond["op"] = random.choice(["gt", "gte", "lt", "lte"])
    return _revalidate(spec)
```

#### 3. `add_missing_dependency`

```python
def add_missing_dependency(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    spec["state"]["auth_token"] = False
    for sc in spec["scenarios"]:
        sc["initial_state"]["auth_token"] = False
    spec["actions"].append({
        "name": "login",
        "description": "Authenticate before sensitive actions",
        "effects": [{"var": "auth_token", "op": "set", "value": True}],
        "reward": 0.0,
    })
    # pick one critical action and gate it on auth_token
    for a in spec["actions"]:
        if a["name"] != "login" and "requires" not in a:
            a["requires"] = {"var": "auth_token", "op": "eq", "value": True}
            for sc in spec["scenarios"]:
                if "login" not in sc["solution"]:
                    sc["solution"].insert(0, "login")
            break
    return _revalidate(spec)
```

#### 4. `change_effect_magnitude`

```python
def change_effect_magnitude(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    action = random.choice(spec["actions"])
    for eff in action["effects"]:
        if eff["op"] in ("add", "subtract"):
            eff["value"] = max(1, eff["value"] + random.choice([-1, 1]))
            eff.setdefault("min", 0)
    return _revalidate(spec)
```

#### 5. `add_distractor_action`

```python
def add_distractor_action(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    var = random.choice(list(spec["state"].keys()))
    spec["actions"].append({
        "name": f"reset_{var}",
        "description": f"Reset {var} to default — looks helpful but undoes progress",
        "effects": [{"var": var, "op": "set", "value": spec["state"][var]}],
        "reward": -0.1,
    })
    return _revalidate(spec)
```

#### 6. `corrupt_observation`

```python
def corrupt_observation(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    keys = [f"{{{k}}}" for k in spec["state"].keys()]
    if keys:
        to_drop = random.choice(keys)
        spec["observation"] = spec["observation"].replace(to_drop, "[redacted]")
    return _revalidate(spec)
```

#### 7. `reduce_step_budget`

```python
def reduce_step_budget(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    min_needed = max(len(sc["solution"]) for sc in spec["scenarios"]) if spec["scenarios"] else 1
    spec["max_steps"] = max(min_needed, spec["max_steps"] - random.choice([1, 2, 3]))
    return _revalidate(spec)
```

#### 8. `add_fault_transition`

```python
def add_fault_transition(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    var = random.choice(list(spec["state"].keys()))
    spec["actions"].append({
        "name": "inject_corruption",
        "description": "Simulate a fault (do not call unless you can recover)",
        "effects": [{"var": var, "op": "toggle"}],
        "reward": -0.5,
    })
    return _revalidate(spec)
```

#### 9. `add_misleading_terminal`

```python
def add_misleading_terminal(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    var = random.choice(list(spec["state"].keys()))
    spec["terminals"].append({
        "when": {"var": var, "op": "eq", "value": spec["state"][var]},
        "outcome": "premature_completion",
        "success": False,
        "reward": 0.0,
    })
    return _revalidate(spec)
```

#### 10. `add_partial_observability`

```python
def add_partial_observability(env_spec: dict) -> dict:
    spec = _revalidate(env_spec)
    # hide all but one state variable
    keep = random.choice(list(spec["state"].keys()))
    spec["observation"] = f"partial state: {keep}={{{keep}}}"
    return _revalidate(spec)
```

### 2.2 Tool-calling mutations

All operators accept a `ToolCallingTask` and return a new `ToolCallingTask`. We first extend the dataclass in `il_toolcalling_core/il_toolcalling_core/tasks.py` to support `max_calls` and `faults`:

```python
# il_toolcalling_core/il_toolcalling_core/tasks.py
@dataclass
class ToolCallingTask:
    ...
    max_calls: int | None = None
    faults: list[dict] = field(default_factory=list)
```

We also patch `engine.py/replay()` to honor `max_calls` (truncation at line 97 after `trace.calls = parse_tool_calls(response)`):

```python
if getattr(task, "max_calls", None) is not None:
    trace.calls = trace.calls[: task.max_calls]
```

Helpers:

```python
import copy
import dataclasses
import json
import random
from il_toolcalling_core.tasks import ToolCallingTask
from il_toolcalling_core.engine import ToolSpec, replay, score_trajectory

def _copy_task(t: ToolCallingTask) -> ToolCallingTask:
    return ToolCallingTask(
        idx=t.idx, name=t.name, spec=t.spec,
        tools=copy.copy(t.tools),
        init=copy.deepcopy(t.init),
        goal=t.goal, expert=list(t.expert),
        verify_answer=t.verify_answer,
        expected_concepts=list(t.expected_concepts),
        token_budget=t.token_budget,
        difficulty=t.difficulty,
        scenario=t.scenario,
        expert_answer=t.expert_answer,
        max_calls=getattr(t, "max_calls", None),
        faults=list(getattr(t, "faults", [])),
    )

def _expert_response(t: ToolCallingTask) -> str:
    parts = [
        f'<tool>{{"name": {json.dumps(n)}, "args": {json.dumps(a)}}}</tool>'
        for n, a in t.expert
    ]
    return "".join(parts) + f"<answer>{t.expert_answer}</answer>"

def _expert_score(t: ToolCallingTask) -> float:
    return score_trajectory(t, _expert_response(t))[0]
```

#### 1. `mutate_init_state`

```python
def mutate_init_state(task: ToolCallingTask) -> ToolCallingTask:
    t = _copy_task(task)
    if t.init:
        k = random.choice(list(t.init.keys()))
        v = t.init[k]
        if isinstance(v, (int, float)):
            t.init[k] = max(0, v + random.choice([-2, -1, 1, 2]))
        elif isinstance(v, bool):
            t.init[k] = not v
        elif isinstance(v, str) and v:
            t.init[k] = v + "_mutated"
    return t
```

#### 2. `inject_fault`

```python
def inject_fault(task: ToolCallingTask, tool_name: str | None = None,
                 fail_first: int = 1, fail_code: str = "500") -> ToolCallingTask:
    t = _copy_task(task)
    candidates = [n for n, sp in t.tools.items() if not sp.distractor]
    if not candidates:
        return t
    tool_name = tool_name or random.choice(candidates)
    base = t.tools[tool_name].fn
    call_count = [0]

    def fn(state, args, base=base):
        call_count[0] += 1
        if call_count[0] <= fail_first:
            return None, f"HTTP {fail_code} on {tool_name}"
        return base(state, args)

    t.tools[tool_name] = dataclasses.replace(t.tools[tool_name], fn=fn)

    # expert must retry the faulted tool
    new_expert = []
    for n, a in t.expert:
        new_expert.append((n, a))
        if n == tool_name and fail_first > 0:
            new_expert.extend([(n, a) for _ in range(fail_first)])
    t.expert = new_expert
    return t
```

#### 3. `add_distractor_tool`

```python
def add_distractor_tool(task: ToolCallingTask) -> ToolCallingTask:
    t = _copy_task(task)
    name = f"noop_{random.randint(0,999)}"
    t.tools[name] = ToolSpec(
        name=name,
        description="Call this first to initialize the session.",
        params={"x": "str"},
        fn=lambda s, a: (None, "no-op"),
        distractor=True,
    )
    return t
```

#### 4. `change_max_calls`

```python
def change_max_calls(task: ToolCallingTask, delta: int | None = None) -> ToolCallingTask:
    t = _copy_task(task)
    base = len(t.expert)
    current = getattr(t, "max_calls", base + 3)
    if delta is None:
        delta = -random.randint(1, 3)
    t.max_calls = max(base, current + delta)
    return t
```

#### 5. `corrupt_observations`

```python
def corrupt_observations(task: ToolCallingTask, tool_name: str | None = None) -> ToolCallingTask:
    t = _copy_task(task)
    tool_name = tool_name or random.choice(list(t.tools.keys()))
    base = t.tools[tool_name].fn
    def fn(state, args, base=base, tool_name=tool_name):
        r, obs = base(state, args)
        if r is not None and isinstance(obs, str):
            obs = obs.replace("25", "30").replace("report-2026-09", "report-OLD")
        return r, obs
    t.tools[tool_name] = dataclasses.replace(t.tools[tool_name], fn=fn)
    return t
```

#### 6. `add_latency_cost`

```python
def add_latency_cost(task: ToolCallingTask, per_call: int = 1) -> ToolCallingTask:
    t = _copy_task(task)
    t.init["_budget"] = getattr(t, "max_calls", len(t.expert) + 3)
    for name, tool in t.tools.items():
        base = tool.fn
        def fn(state, args, base=base, name=name):
            state["_budget"] = state.get("_budget", 0) - per_call
            if state["_budget"] < 0:
                return None, f"{name}: budget exhausted"
            return base(state, args)
        t.tools[name] = dataclasses.replace(tool, fn=fn)
    return t
```

#### 7. `change_goal_predicate`

```python
def change_goal_predicate(task: ToolCallingTask) -> ToolCallingTask:
    t = _copy_task(task)
    base_goal = t.goal
    def goal(state):
        ok, info = base_goal(state)
        return ok and state.get("reported") is True, info
    t.goal = goal
    return t
```

#### 8. `add_missing_dependency`

```python
def add_missing_dependency(task: ToolCallingTask, prereq: str = "login") -> ToolCallingTask:
    t = _copy_task(task)
    candidates = [n for n, sp in t.tools.items() if not sp.distractor]
    if not candidates:
        return t
    target = random.choice(candidates)
    # add auth tool
    t.tools[prereq] = ToolSpec(
        prereq, "Authenticate before privileged calls", {},
        lambda s, a: (s.update({"token": True}) or (True, "authenticated")), distractor=False
    )
    base = t.tools[target].fn
    def fn(state, args, base=base, target=target):
        if not state.get("token"):
            return None, f"{target}: authentication required"
        return base(state, args)
    t.tools[target] = dataclasses.replace(t.tools[target], fn=fn)
    t.expert.insert(0, (prereq, {}))
    return t
```

#### 9. `add_hostile_test`

```python
def add_hostile_test(task: ToolCallingTask) -> ToolCallingTask:
    t = _copy_task(task)
    base_verify = t.verify_answer
    def verify(answer, trace, base=base_verify):
        ok, info = base(answer, trace)
        if ok and "confirmed" not in answer.lower():
            return False, "must confirm result in answer"
        return ok, info
    t.verify_answer = verify
    return t
```

#### 10. `add_resource_constraint`

```python
def add_resource_constraint(task: ToolCallingTask, limit: int | None = None) -> ToolCallingTask:
    t = _copy_task(task)
    limit = limit or len(t.expert)
    t.init["_calls_left"] = limit
    for name, tool in t.tools.items():
        base = tool.fn
        def fn(state, args, base=base, name=name):
            state["_calls_left"] = state.get("_calls_left", limit) - 1
            if state["_calls_left"] < 0:
                return None, f"{name}: resource exhausted"
            return base(state, args)
        t.tools[name] = dataclasses.replace(tool, fn=fn)
    return t
```

---

## 3. Adversarial search

The search wrapper is PAIRED-style: it tries to maximize **regret** while keeping the environment solvable by the expert/oracle.

```python
from typing import Callable

class AdversarialSearch:
    def __init__(
        self,
        base: dict | ToolCallingTask,
        model_fn: Callable[[dict | ToolCallingTask], float],
        oracle_fn: Callable[[dict | ToolCallingTask], float],
        operators: list[Callable],
        budget: int = 64,
        success_threshold: float = 0.3,
        oracle_threshold: float = 0.8,
        rng=None,
    ):
        self.base = base
        self.model_fn = model_fn
        self.oracle_fn = oracle_fn
        self.operators = operators
        self.budget = budget
        self.success_threshold = success_threshold
        self.oracle_threshold = oracle_threshold
        self.rng = rng or random.Random(0)

    def _evaluate(self, candidate) -> tuple[float | None, dict]:
        oracle_scores = [self.oracle_fn(candidate) for _ in range(2)]
        if min(oracle_scores) < self.oracle_threshold:
            return None, {"reason": "unsolvable"}
        model_scores = [self.model_fn(candidate) for _ in range(4)]
        prot = sum(model_scores) / len(model_scores)
        ant = sum(oracle_scores) / len(oracle_scores)
        regret = ant - prot
        return regret, {"prot": prot, "ant": ant}

    def search(self) -> tuple[object | None, float, dict]:
        best = None
        best_regret = -float("inf")
        best_info = {}
        archive = []

        # Phase 1: random exploration
        for _ in range(max(1, self.budget // 4)):
            op = self.rng.choice(self.operators)
            cand = op(self.base)
            regret, info = self._evaluate(cand)
            if regret is None:
                continue
            archive.append((regret, cand))
            if regret > best_regret:
                best_regret, best, best_info = regret, cand, info

        # Phase 2: greedy hill-climb from best
        for _ in range(max(1, self.budget // 2)):
            if best is None:
                break
            op = self.rng.choice(self.operators)
            cand = op(best)
            regret, info = self._evaluate(cand)
            if regret is None:
                continue
            if regret > best_regret:
                best_regret, best, best_info = regret, cand, info
            if best_info.get("prot", 1.0) < self.success_threshold and best_info.get("ant", 0.0) >= self.oracle_threshold:
                return best, best_regret, best_info

        # Phase 3: evolutionary recombination
        archive = sorted([x for x in archive if x[0] is not None], key=lambda x: -x[0])[:8]
        for _ in range(max(1, self.budget - self.budget // 4 - self.budget // 2)):
            if len(archive) < 2:
                break
            r1, c1 = archive[0]
            r2, c2 = self.rng.choice(archive)
            cand = self._crossover(c1, c2)
            regret, info = self._evaluate(cand)
            if regret is None:
                continue
            if regret > best_regret:
                best_regret, best, best_info = regret, cand, info
            if best_info.get("prot", 1.0) < self.success_threshold and best_info.get("ant", 0.0) >= self.oracle_threshold:
                return best, best_regret, best_info

        return best, best_regret, best_info
```

Crossover is substrate-specific. For state machines, crossover swaps a subset of `actions`/`scenarios` between two specs. For tool tasks, crossover swaps the `tools` dicts and `init` states, then re-validates that the hybrid task is still solvable. If not, it is filtered by `_evaluate`.

### Degeneracy guard
- **Solvability**: every candidate must pass the oracle/expert with score ≥ `oracle_threshold` (default 0.8).
- **Not impossible**: the candidate is also evaluated by a slightly stronger baseline (e.g., a larger model or more rollouts). If that baseline passes 0%, the mutation is discarded.
- **No trivial path**: if the model already passes, regret is low; the search naturally ignores those.
- **Diversity**: archive is de-duplicated by a hash of the mutated spec to avoid the same mutation repeatedly.

---

## 4. Self-generated environments (#13)

### Generator output schema

The generator LLM emits a JSON object with this schema:

```json
{
  "domain": "tool-fs",
  "name": "self_gen_config_patch",
  "spec": "You operate a simulated filesystem...",
  "tools": [
    {"name": "list_dir", "description": "...", "params": {"path": "str"}, "distractor": false, "destructive": false}
  ],
  "init": {"tree": {...}, "files": {...}},
  "goal_check_code": "def goal(state):\n    return state.get('patched') == '/etc/app/app.conf' and 'port = 8080' in state.get('patched_content',''), 'patched'",
  "expert": [["list_dir", {"path": "/etc/app"}], ["read_file", {"path": "/etc/app/app.conf"}], ["write_file", {"path": "/etc/app/app.conf", "content": "[server]\nport = 8080"}]],
  "verify_answer_code": "def verify(answer, trace):\n    return '8080' in answer.lower(), 'missing port'",
  "constraints": {"max_calls": 6}
}
```

For state machines the schema is the `validate_simulator` payload:

```json
{
  "template_id": "custom-state-machine-v1",
  "observation": "...",
  "state": {...},
  "actions": [...],
  "terminals": [...],
  "max_steps": 8,
  "scenarios": [{"name": "...", "initial_state": {...}, "solution": [...]}]
}
```

### Verifier builder

`VerifierBuilder` compiles the generated code in a restricted namespace:

```python
import ast
import types

SAFE_BUILTINS = {"len": len, "sum": sum, "sorted": sorted, "min": min, "max": max, "abs": abs, "any": any, "all": all, "enumerate": enumerate, "zip": zip}

class VerifierBuilder:
    def build(self, proposal: dict) -> tuple[ToolCallingTask | dict, bool]:
        # 1. syntax check
        try:
            ast.parse(proposal["goal_check_code"])
            ast.parse(proposal["verify_answer_code"])
        except SyntaxError:
            return None, False

        # 2. compile functions
        ns = {"__builtins__": SAFE_BUILTINS}
        exec(proposal["goal_check_code"], ns)
        exec(proposal["verify_answer_code"], ns)
        goal = ns["goal"]
        verify = ns["verify"]

        # 3. construct task/spec
        if proposal["domain"] == "state-machine":
            spec = validate_simulator(proposal)
            return spec, self._expert_solves(spec)
        else:
            tools = {
                d["name"]: ToolSpec(
                    name=d["name"],
                    description=d["description"],
                    params=d["params"],
                    fn=self._build_tool_fn(d),  # or code string compiled similarly
                    distractor=d.get("distractor", False),
                    destructive=d.get("destructive", False),
                )
                for d in proposal["tools"]
            }
            task = ToolCallingTask(
                idx=0,
                name=proposal["name"],
                spec=proposal["spec"],
                tools=tools,
                init=proposal["init"],
                goal=goal,
                expert=[(n, a) for n, a in proposal["expert"]],
                verify_answer=verify,
                expected_concepts=[t["name"] for t in proposal["tools"]],
                max_calls=proposal["constraints"].get("max_calls"),
            )
            return task, self._expert_solves(task)

    def _expert_solves(self, task_or_spec) -> bool:
        # for tools: replay expert; for state machines: run scenario solution
        ...
```

### Generator reward

```python
def generator_reward(
    task: ToolCallingTask,
    solver_responses: list[str],
    expert_pass: bool,
    archive: list[ToolCallingTask],
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.0,
    delta: float = 0.1,
) -> float:
    scores = [score_trajectory(task, r)[0] for r in solver_responses]
    pass_rate = sum(1 for s in scores if s >= 0.8) / len(solver_responses)

    novelty = _novelty_score(task, archive)  # 0..1

    unsolvable_penalty = 0.0 if expert_pass else 1.0
    trivial_penalty = 1.0 if pass_rate == 1.0 else 0.0

    return (
        alpha * (1.0 - pass_rate)
        - beta * unsolvable_penalty
        - gamma * trivial_penalty
        + delta * novelty
    )
```

### Anti-degeneracy constraints

1. **Verifier soundness**: `goal()` and `verify_answer()` must be syntactically valid, run without exception, and return `(bool, str)`.
2. **Expert solvability**: the generator’s own `expert` trajectory must reach `goal_reached == True` and pass `verify_answer`.
3. **Not impossible**: a stronger baseline (more inference-time compute or larger model) must solve it with pass rate > 0. If not, discard and penalize.
4. **Not trivial**: solver pass rate of 100% triggers a trivial penalty.
5. **No code injection**: `SAFE_BUILTINS` block `__import__`, `open`, `eval`, `exec`, etc.
6. **Novelty**: keep only if Jaccard distance of tool names + state keys to the nearest archive task is > 0.2.

### GAN loop

```python
class SelfGenEnvLoop:
    def __init__(self, generator_llm, solver_llm, archive_size=256):
        self.generator = generator_llm
        self.solver = solver_llm
        self.archive = []

    def iteration(self):
        prompt = self._build_generator_prompt(self.archive)
        proposal = self.generator.generate(prompt)
        task, ok = VerifierBuilder().build(proposal)
        if not ok:
            return None, 0.0

        # run solver
        responses = [self.solver.generate(task.spec) for _ in range(8)]
        expert_pass = self._expert_solves(task)
        reward = generator_reward(task, responses, expert_pass, self.archive)

        if reward > 0.0 and expert_pass:
            self.archive.append(task)
            self.archive = sorted(self.archive, key=lambda t: -t.difficulty)[:self.archive_size]

        return task, reward
```

---

## 5. Temporal compression (#5)

Given a `ToolCallTrace` with `N` calls, identify the `K` critical decision points by counterfactual probing.

```python
from il_toolcalling_core.engine import replay

def _run_calls(task: ToolCallingTask, calls: list[dict]) -> bool:
    resp = "".join(
        f'<tool>{{"name": {json.dumps(c["name"])}, "args": {json.dumps(c.get("args", {}))}}}</tool>'
        for c in calls
    )
    trace = replay(task, resp + f"<answer>{task.expert_answer}</answer>")
    return trace.goal_reached

def critical_decision_points(task: ToolCallingTask, trace) -> list[int]:
    n = len(trace.calls)
    original = trace.goal_reached
    critical = []
    for i in range(n):
        prefix = trace.calls[:i]
        suffix = trace.calls[i+1:]
        actual = trace.calls[i]["name"]
        is_critical = False
        alternatives = list(task.tools.keys()) + ["__skip__"]
        for alt in alternatives:
            if alt == actual:
                continue
            if alt == "__skip__":
                candidate = prefix + suffix
            else:
                candidate = prefix + [{"name": alt, "args": {}}] + suffix
            if _run_calls(task, candidate) != original:
                is_critical = True
                break
        if is_critical:
            critical.append(i)
    return critical
```

If `len(critical) > K`, prune by impact magnitude:

```python
def impact(task, trace, i: int) -> float:
    prefix = trace.calls[:i]
    suffix = trace.calls[i+1:]
    original = trace.goal_reached
    worst = original
    for alt in task.tools.keys():
        if alt == trace.calls[i]["name"]:
            continue
        candidate = prefix + [{"name": alt, "args": {}}] + suffix
        reached = _run_calls(task, candidate)
        if reached != original:
            worst = not original
    return float(abs(int(original) - int(worst)))

def build_compressed_task(task: ToolCallingTask, trace, K: int = 12) -> ToolCallingTask:
    critical = critical_decision_points(task, trace)
    if len(critical) > K:
        critical = sorted(critical, key=lambda i: -impact(task, trace, i))[:K]
    new_expert = [(trace.calls[i]["name"], trace.calls[i].get("args", {})) for i in critical]
    t = _copy_task(task)
    t.expert = new_expert
    t.max_calls = len(new_expert) + 2
    t.spec = (
        f"{task.spec}\n\n"
        f"[COMPRESSED TRAJECTORY] Only {len(new_expert)} critical decision points are required. "
        f"Fill the gaps between them with the appropriate primitive tool calls."
    )
    return t
```

### Hierarchical RL interpretation
- **Options** = each critical decision point + the low-level sequence that carries the state from the previous critical point to this one.
- **High-level policy** selects the next option (critical tool call).
- **Low-level policy** executes the primitive calls between critical points.
- This maps directly to the options/SMDP framework: an option is `(initiation set, intra-option policy, termination condition)`.

---

## 6. Integration plan

### New files

**`iloptimus/core/env_mutator.py`**
- `StatefulMutator` class exposing the 10 state-machine operators.
- `ToolMutator` class exposing the 10 tool-calling operators.
- `AdversarialSearch` class (section 3).
- `MutationArchive` for ACCEL-style high-regret mutation replay.
- Exports `STATEFUL_OPERATORS`, `TOOL_OPERATORS`, `adversarial_search`.

**`iloptimus/core/self_gen_env.py`**
- `SelfGenEnvLoop`, `VerifierBuilder`, `generator_reward`, `novelty_score`.
- Supports both `ToolCallingTask` and `validate_simulator` outputs.

**`iloptimus/core/temporal_compression.py`**
- `critical_decision_points`, `build_compressed_task`, `impact`.
- Optional `HierarchicalTask` wrapper that pairs a compressed high-level task with low-level segment tasks.

### File modifications

**`il_toolcalling_core/il_toolcalling_core/tasks.py`** (line 11)
Extend `ToolCallingTask`:

```python
@dataclass
class ToolCallingTask:
    ...
    max_calls: int | None = None
    faults: list[dict] = field(default_factory=list)
```

**`il_toolcalling_core/il_toolcalling_core/engine.py`** (line 94–131)
Update `replay()` to honor `max_calls` and `faults`:

```python
def replay(task, response: str) -> ToolCallTrace:
    trace = ToolCallTrace(state=dict(task.init))
    trace.calls = parse_tool_calls(response)
    if getattr(task, "max_calls", None) is not None:
        trace.calls = trace.calls[: task.max_calls]
    # apply any task.faults wrappers here or rely on operator-wrapped fn
    ...
```

**`iloptimus/iloptimus/core/rsi_loops.py`**
- Add a loop kind around line 24:
  ```python
  LOOP_KINDS = ["coding", "reasoning", "math", "tool-calling", "agentic", "adversarial"]
  ```
- Add a template around line 117:
  ```python
  {
      "id": "loop-adversarial-env",
      "kind": "adversarial",
      "name": "Adversarial environment gym",
      "objective": "Given a base tool-calling or state-machine task, run an adversarial search to find a world-dynamics mutation the current model fails but the expert solves. Train on the resulting curriculum.",
      "default_minutes": 30,
      "default_iterations": 6,
      "needs_sandbox": False,
  },
  ```
- Add a selection hook after `runnability()` (line 231):
  ```python
  def maybe_mutate_task(loop: RsiLoop, base_task, model_fn, oracle_fn):
      if loop.kind != "adversarial":
          return base_task
      from iloptimus.core.env_mutator import AdversarialSearch, TOOL_OPERATORS, STATEFUL_OPERATORS
      ops = TOOL_OPERATORS if isinstance(base_task, ToolCallingTask) else STATEFUL_OPERATORS
      search = AdversarialSearch(base_task, model_fn, oracle_fn, ops)
      mutated, regret, info = search.search()
      return mutated if mutated is not None else base_task
  ```

**`iloptimus/iloptimus/core/grader.py`**
Add a generic tool-task grader after `grade_tool_calling()` (line 460):

```python
def grade_tool_calling_task(response: str, task: ToolCallingTask) -> GradedResult:
    from il_toolcalling_core.engine import score_trajectory
    score, breakdown = score_trajectory(task, response)
    return GradedResult(
        score=score,
        correctness=breakdown["correctness"],
        reasoning_quality=breakdown["tool_selection"],
        coverage=breakdown["efficiency"],
        verification=1.0 if not breakdown["errors"] else 0.0,
        info=breakdown,
    )
```

**`iloptimus/iloptimus/core/environments.py`**
No structural changes required; mutated state-machine specs are passed through `validate_environment()` and `build_stateful_prompt()` already supports arbitrary spec shapes (lines 59–143, 316–329).

### Reuse of existing substrates
- The state-machine mutations operate on the exact dict schema produced by `validate_simulator()` in `stateful_environments.py`.
- The tool mutations produce `ToolCallingTask` objects compatible with `il_toolcalling_core.engine.replay` and `score_trajectory`.
- Both substrates already provide an **expert trajectory** (`scenario["solution"]` and `task.expert`), which serves as the PAIRED antagonist.

---

## 7. Validation plan

### Unit tests for mutation operators

Create `tests/test_env_mutator.py`:

```python
from iloptimus.core.env_mutator import StatefulMutator, ToolMutator
from iloptimus.core.stateful_environments import validate_simulator
from il_toolcalling_core.tasks import ToolCallingTask

def test_stateful_operators_preserve_solvability():
    base = validate_simulator({
        "observation": "p={position}, e={energy}, g={goal}",
        "state": {"position": 0, "energy": 4, "goal": 3},
        "actions": [{"name": "move_forward", "effects": [{"var": "position", "op": "add", "value": 1}, {"var": "energy", "op": "subtract", "value": 1}]}],
        "terminals": [{"when": {"var": "position", "op": "gte", "value_from": "goal"}, "success": True}],
        "max_steps": 8,
        "scenarios": [{"name": "reach 3", "initial_state": {"position": 0, "energy": 4, "goal": 3}, "solution": ["move_forward"] * 3}],
    })
    for op in StatefulMutator.OPERATORS:
        mutated = op(base)
        assert StatefulMutator().expert_solves(mutated), f"{op.__name__} broke solvability"
```

```python
def test_tool_operators_preserve_solvability(fs_task):
    for op in ToolMutator.OPERATORS:
        mutated = op(fs_task)
        score, _ = ToolMutator().expert_score(mutated)
        assert score >= 0.8, f"{op.__name__} broke expert score"
```

### Adversarial search test

Create `tests/test_adversarial_search.py`:

```python
def test_adversarial_search_flips_stub_model(grid_task):
    # stub model always replays the original expert
    def model_fn(task):
        resp = "<answer>" + json.dumps(grid_task["scenarios"][0]["solution"]) + "</answer>"
        from iloptimus.core.stateful_environments import simulate_response
        return simulate_response({"simulator": task, "goal": "grid"}, 0, resp)["score"]

    def oracle_fn(task):
        return model_fn(task)  # optimistic oracle; in real code use stronger model

    search = AdversarialSearch(grid_task, model_fn, oracle_fn, StatefulMutator.OPERATORS)
    mutated, regret, info = search.search()
    assert info["prot"] < 0.3
    assert info["ant"] >= 0.8
```

For tool tasks:

```python
def test_adversarial_search_tool(fs_task):
    def model_fn(task):
        from il_toolcalling_core.engine import score_trajectory
        return score_trajectory(task, _expert_response(task))[0]
    def oracle_fn(task):
        return model_fn(task) + 0.2  # stronger oracle
    search = AdversarialSearch(fs_task, model_fn, oracle_fn, ToolMutator.OPERATORS)
    mutated, regret, info = search.search()
    assert info["prot"] < 0.3
```

### Self-generated environment test

```python
def test_self_gen_one_iteration():
    loop = SelfGenEnvLoop(generator_llm=mock_generator, solver_llm=mock_solver)
    task, reward = loop.iteration()
    assert task is not None
    assert reward > 0.0
    assert VerifierBuilder()._expert_solves(task)
```

### Temporal compression test

```python
def test_temporal_compression_removes_fillers():
    # start with api-reliability expert and inject 5 no-op get_task_info calls
    base = _load_tool_task("il_tool_api_reliability_v1")
    noisy_expert = [("get_task_info", {})] * 5 + base.expert
    base.expert = noisy_expert
    from iloptimus.core.temporal_compression import build_compressed_task
    full_resp = _expert_response(base)
    trace = replay(base, full_resp)
    compressed = build_compressed_task(base, trace, K=12)
    assert len(compressed.expert) < len(noisy_expert)
    # critical submit and pagination calls are preserved
    assert any(n == "submit_results" for n, _ in compressed.expert)
```

### RSI integration test

```python
def test_rsi_loop_requests_mutated_task():
    loop = RsiLoop(kind="adversarial", model_id="stub", ...)
    base = _load_tool_task("il_tool_fs_navigator_v1")
    mutated = maybe_mutate_task(loop, base, model_fn=lambda t: 0.0, oracle_fn=lambda t: 1.0)
    assert isinstance(mutated, ToolCallingTask)
    assert getattr(mutated, "max_calls", None) is not None or mutated is base
```

---

## Concrete deliverables

1. `iloptimus/core/env_mutator.py` — mutation operators + PAIRED-style adversarial search.
2. `iloptimus/core/self_gen_env.py` — GAN-style generator-solver-verifier loop.
3. `iloptimus/core/temporal_compression.py` — critical decision compression and hierarchical task builder.
4. Patches to:
   - `il_toolcalling_core/il_toolcalling_core/tasks.py` (add `max_calls`, `faults`)
   - `il_toolcalling_core/il_toolcalling_core/engine.py` (enforce `max_calls`)
   - `iloptimus/iloptimus/core/rsi_loops.py` (new loop kind + `maybe_mutate_task`)
   - `iloptimus/iloptimus/core/grader.py` (generic `grade_tool_calling_task`)
