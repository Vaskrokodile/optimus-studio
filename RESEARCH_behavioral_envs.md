# Behavioral Environments — Concrete Design Doc

## 0. Codebase anchors used throughout

- Tool-calling substrate: `il_toolcalling_core/il_toolcalling_core/engine.py`
  - `ToolSpec` dataclass: L41-L49
  - `ToolCallTrace`: L52-L61
  - `score_trajectory` / `replay`: L94-L197
- Tool task dataclass: `il_toolcalling_core/il_toolcalling_core/tasks.py`
  - `ToolCallingTask`: L11-L26
  - `answer_contains`: L31-L43
- Stateful runtime: `iloptimus/iloptimus/core/stateful_environments.py`
  - `StateMachineRuntime`: L180-L269
  - `validate_simulator`: L86-L164
  - `evaluate_condition`: L53-L67
  - `StepResult`: L167-L177
- RL factory base classes:
  - `Problem`, `BaseReasoningEnv`: `iloptimus/core/rl_factory/core/base.py` L44-L62, L65-L347
  - `BatchEnvBase`: `iloptimus/core/rl_factory/core/batch_base.py` L35-L168
  - `Verifier`, `VerifierResult`: `iloptimus/core/rl_factory/core/verifier.py` L23-L67
  - `RewardComponents`, `RewardConfig`: `iloptimus/core/rl_factory/core/reward.py` L30-L95
- Existing prototype environments:
  - `SkillCollisionEnv`: `iloptimus/core/rl_factory/environments/skill_collision.py` L1-L416
  - `FailureFirstEnv`: `iloptimus/core/rl_factory/environments/failure_first.py` L1-L1114
  - `TrajectoryRecombinationEnv`: `iloptimus/core/rl_factory/environments/trajectory_recombination.py` L750-L952
  - `DynamicsMutator` (env mutation): `iloptimus/core/rl_factory/environments/environment_mutation.py` L373-L572
- Registry / grader:
  - `TASKSET_REGISTRY`: `iloptimus/core/tasksets.py` L23-L243
  - `_TOOL_DOMAINS`, `_GRADERS`, `_INSTRUCTIONS`: `iloptimus/core/grader.py` L423-L526
  - `LOOP_KINDS`, `LOOP_TEMPLATES`: `iloptimus/core/rsi_loops.py` L24-L117

---

## 1. Prior art summary

| Area | Mechanism to steal |
|------|-------------------|
| **Skill collision / tradeoffs** | Multi-objective RL and constrained MDPs treat task reward and each constraint as separate objectives; the Pareto-optimal set is searched and a preference vector selects the operating point (Huang et al. *LP3*, CMORL, CoMOGA). **Steal:** return a 2-D reward vector per episode, scalarize with a scenario-conditioned preference `α`, and add a small Pareto-dominance bonus computed over the batch. |
| **Hidden-rule / system ID** | Hidden-Parameter MDPs (HiP-MDP, GHP-MDP) and model-based world-model learning assume a low-dimensional latent vector `θ` conditions the dynamics; the agent infers `θ` from transition data and then plans. **Steal:** make the latent variables explicit in `state`, hide them from the observation string, and force the model to call probe tools to estimate them. |
| **Experimentation / active inference** | Bayesian optimal experimental design and active inference choose actions that maximize expected information gain (EIG) about hidden states; BED-LLM/ASIG amortize this into LLM policies. **Steal:** maintain a categorical belief `b(θ)`, update it with `P(o | θ)` after every probe, and reward `R = R_goal + λ·(H(b) − H(b'))`. |
| **Failure-first / recovery RL** | Recovery RL (Thananjeyan et al.) separates a task policy from a recovery policy and learns recovery zones from offline constraint violations. **Steal:** inject deterministic `Fault`s into `ToolSpec.fn`, score detection + corrective action as `R_recovery`, and penalize destructive calls on bad data as `R_catastrophic`. |
| **Multi-agent evolutionary** | PBT jointly optimizes a population and hyperparameter schedules; AlphaStar’s league uses exploiters and Nash self-play; RAGEN/StarPO train LLM agents with population-based multi-turn RL. **Steal:** keep a lightweight *population of trajectories* (not full agents), select by verifier score, mutate problems with `DynamicsMutator`, and recombine responses with `TrajectoryRecombinationEnv`. |

---

## 2. Skill collision environments (#7)

All three tasks are implemented as `ToolCallingTask` instances scored by `score_trajectory` (`engine.py L138`). The verifier returns a **reward vector**, and the environment (`SkillCollisionEnv`) scalarizes it in a Pareto-aware way.

### 2.1 Shared Pareto-aware scalarization

For a batch of `N` responses, each response `i` has a vector `r_i = [r_i^A, r_i^B]`.  
Scalarization:

```
weighted_sum_i = α · r_i^A + (1 − α) · r_i^B
pareto_bonus_i = 1.0 if no other j ≠ i has (r_j^A ≥ r_i^A and r_j^B ≥ r_i^B
                                          and at least one strict)
                 else 0.0
R_i = 0.95 · weighted_sum_i + 0.05 · pareto_bonus_i
```

`α` is sampled per scenario (e.g. `α ∈ {0.3, 0.5, 0.7}`) and exposed in the prompt as “this is a HIGH-STAKES / LOW-STAKES task”.

### 2.2 Task A — Fast lookup vs. robust lookup

```python
def _make_db_tools(state, rng):
    def quick_lookup(s, args):
        # 30% chance stale; otherwise correct
        if rng.random() < 0.30:
            return s["cached_wrong"], "quick_lookup: cached result"
        return s["true_value"], "quick_lookup: cached result"

    def verified_lookup(s, args):
        # consumes 3 internal "calls" by touching three tables
        s["call_cost"] = s.get("call_cost", 0) + 3
        return s["true_value"], "verified_lookup: cross-checked"

    def submit_answer(s, args):
        s["submitted"] = args["value"]
        return "ok", f"submitted {args['value']}"

    return {
        "quick_lookup": ToolSpec("quick_lookup", "Fast, possibly stale lookup.", {}, quick_lookup),
        "verified_lookup": ToolSpec("verified_lookup", "Slow exact lookup (cost=3 calls).", {}, verified_lookup),
        "submit_answer": ToolSpec("submit_answer", "Submit final answer.", {"value": "str"}, submit_answer),
    }
```

- **Skill A:** speed / low call budget.  
- **Skill B:** correctness / robust verification.  
- **Conflict:** `max_calls = 5`. One `verified_lookup` plus `submit_answer` uses 4 calls and always succeeds. `quick_lookup` + `submit_answer` uses 2 calls but may be wrong.  
- **Reward vector:** `[accuracy, efficiency]`, where `accuracy = 1` iff submitted value equals `true_value`, `efficiency = 1 − calls_used / max_calls`.  
- **Pareto scalarization:** `α = 0.7` for high-stakes, `0.3` for low-stakes.

### 2.3 Task B — Greedy route vs. optimal TSP

```python
def greedy_route(s, args):
    # nearest-neighbor; returns route length 1.15–1.30× optimal
    s["calls"] = s.get("calls", 0) + 1
    return s["greedy_length"], f"greedy route length: {s['greedy_length']}"

def exact_tsp(s, args):
    # held-karp or brute force; exact but costs 3 calls
    s["calls"] = s.get("calls", 0) + 3
    return s["optimal_length"], f"optimal route length: {s['optimal_length']}"

def submit_route(s, args):
    s["submitted_length"] = float(args["length"])
    return "ok", f"submitted {args['length']}"
```

- **Skill A:** greedy local optimization (fast, suboptimal).  
- **Skill B:** global exact planning (slow, optimal).  
- **Conflict:** `max_calls = 6`. Greedy uses 1 call; exact uses 3 calls. The prompt gives 6 cities and a deadline.  
- **Reward vector:** `[optimality, budget_left]`, `optimality = optimal_length / submitted_length`, `budget_left = 1 − calls / max_calls`.  
- **Pareto scalarization:** `α = 0.6` by default, `0.9` when the prompt says “minimize fuel cost”.

### 2.4 Task C — Flash promotion vs. brand investment

```python
def flash_promo(s, args):
    s["revenue_today"] += s["base_demand"] * s["price"] * 1.4
    s["brand"] -= 0.15
    s["calls"] = s.get("calls", 0) + 1

def brand_invest(s, args):
    spend = float(args["amount"])
    s["cash"] -= spend
    s["brand"] += spend / 1000.0
    s["calls"] = s.get("calls", 0) + 1

def end_quarter(s, args):
    # multi-day rollout; hidden brand multiplier
    for day in range(5):
        multiplier = max(0.5, 1.0 + s["brand"])
        s["revenue_total"] += s["base_demand"] * s["price"] * multiplier / 5.0
    s["calls"] = s.get("calls", 0) + 1

def submit_quarter(s, args):
    s["submitted_revenue"] = float(args["revenue"])
```

- **Skill A:** short-term revenue (`flash_promo`).  
- **Skill B:** long-term revenue (`brand_invest`).  
- **Conflict:** cash is limited; flash sales hurt the hidden `brand` variable. `end_quarter` reveals cumulative revenue.  
- **Reward vector:** `[r_short, r_long]`, `r_short = revenue_today / target`, `r_long = revenue_total / target`.  
- **Pareto scalarization:** `R = 0.2·r_short + 0.7·r_long + 0.1·pareto_bonus` (long-term is weighted higher by default).

---

## 3. Hidden-rule environments (#8)

`env_hidden_rule.py` wraps `StateMachineRuntime` (`stateful_environments.py L180`) but exposes the world through `ToolSpec` tools. The latent variables live in `runtime.state`; the `observation()` template omits them.

### 3.1 Fully-specified example: SimulatedCompany

```python
from iloptimus.core.stateful_environments import StateMachineRuntime, validate_simulator
from il_toolcalling_core.engine import ToolSpec, score_trajectory
from dataclasses import dataclass
import math, random


@dataclass
class CompanyLatent:
    price_sensitivity: float = 1.2        # hidden
    employee_motivation: float = 1.0     # hidden
    competitor_aggressiveness: float = 0.5 # hidden
    market_trend: float = 0.05            # hidden
    target_revenue: float = 15_000.0      # known target


class SimulatedCompanyHiddenRuleEnv:
    """
    Observation does NOT include the latent CompanyLatent variables.
    The model must call survey/morale/spy/research tools to estimate them.
    """
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.latent = CompanyLatent()
        self.state = {
            "day": 1,
            "cash": 10_000.0,
            "inventory": 500,
            "price": 10.0,
            "ad_spend": 0.0,
            "staff": 5,
            "revenue": 0.0,
            "satisfaction": 1.0,
            "retention": 1.0,
            "competitor_price": 9.0,
            # hidden
            "price_sensitivity": self.latent.price_sensitivity,
            "employee_motivation": self.latent.employee_motivation,
            "competitor_aggressiveness": self.latent.competitor_aggressiveness,
            "market_trend": self.latent.market_trend,
            "target_revenue": self.latent.target_revenue,
            "base_demand": 1000,
            "max_days": 7,
            "submitted": False,
        }
        self.observation_template = (
            "Day {day}/{max_days}  cash={cash:.0f}  revenue={revenue:.0f}  "
            "inventory={inventory}  price={price}  competitor_price={competitor_price}  "
            "satisfaction={satisfaction:.2f}  retention={retention:.2f}"
        )
        self.tools = self._make_tools()

    def _make_tools(self):
        def noisy(v, sigma=0.1):
            return v + self.rng.gauss(0, sigma)

        def survey_customers(s, _):
            return None, f"customers are sensitive to price changes (sensitivity≈{noisy(s['price_sensitivity'],0.15):.2f})"

        def check_morale(s, _):
            return None, f"employee morale≈{noisy(s['employee_motivation'],0.12):.2f}"

        def spy_competitor(s, _):
            return None, (f"competitor price≈{noisy(s['competitor_price'],0.20):.2f}, "
                          f"aggressiveness≈{noisy(s['competitor_aggressiveness'],0.15):.2f}")

        def market_research(s, _):
            return None, f"market trend≈{noisy(s['market_trend'],0.03):.2f}"

        def set_price(s, args):
            s["price"] = float(args["price"])
            return s["price"], f"price set to {s['price']}"

        def run_ad_spend(s, args):
            amount = float(args["amount"])
            s["cash"] -= amount
            s["ad_spend"] += amount
            return amount, f"spent {amount} on ads"

        def hire(s, args):
            n = int(args["count"])
            cost = n * 500
            s["cash"] -= cost
            s["staff"] += n
            return n, f"hired {n} staff for {cost}"

        def order_inventory(s, args):
            units = int(args["units"])
            cost = units * 5
            s["cash"] -= cost
            s["inventory"] += units
            return units, f"ordered {units} units for {cost}"

        def end_day(s, _):
            # --- latent dynamics ---
            ad_effect = math.sqrt(max(0, s["ad_spend"])) / 30.0
            price_pressure = s["price_sensitivity"] * (s["price"] - 8.0) / 8.0
            comp_effect = s["competitor_aggressiveness"] * (s["competitor_price"] / s["price"] - 1.0)
            trend_effect = s["market_trend"] * s["day"]
            demand = s["base_demand"] * max(0.0, 1.0 - price_pressure - comp_effect + trend_effect + ad_effect)
            production = s["staff"] * 80
            s["inventory"] = min(s["inventory"] + production, 2000)
            sold = min(demand, s["inventory"])
            s["inventory"] -= int(sold)
            daily_revenue = s["price"] * sold
            s["revenue"] += daily_revenue
            s["cash"] += daily_revenue
            # brand / satisfaction effects
            s["satisfaction"] = max(0.0, 1.0 - abs(s["price"] - 9.5) * s["price_sensitivity"] / 10.0)
            s["retention"] = max(0.0, min(1.0, s["retention"] + (s["employee_motivation"] - 1.0) * 0.05))
            # competitor reacts
            s["competitor_price"] *= 1.0 + s["competitor_aggressiveness"] * 0.02
            s["day"] += 1
            s["ad_spend"] = 0.0
            return daily_revenue, f"day {s['day']-1}: sold {int(sold)}, revenue {daily_revenue:.0f}"

        def submit_plan(s, _):
            s["submitted"] = True
            return "ok", "plan submitted"

        return {
            "survey_customers": ToolSpec("survey_customers", "Probe: estimate price sensitivity.", {}, survey_customers),
            "check_morale": ToolSpec("check_morale", "Probe: estimate employee motivation.", {}, check_morale),
            "spy_competitor": ToolSpec("spy_competitor", "Probe: estimate competitor state.", {}, spy_competitor),
            "market_research": ToolSpec("market_research", "Probe: estimate market trend.", {}, market_research),
            "set_price": ToolSpec("set_price", "Set unit price.", {"price": "float"}, set_price),
            "run_ad_spend": ToolSpec("run_ad_spend", "Spend cash on ads.", {"amount": "float"}, run_ad_spend),
            "hire": ToolSpec("hire", "Hire staff.", {"count": "int"}, hire),
            "order_inventory": ToolSpec("order_inventory", "Buy inventory.", {"units": "int"}, order_inventory),
            "end_day": ToolSpec("end_day", "Advance one day and compute sales.", {}, end_day),
            "submit_plan": ToolSpec("submit_plan", "Submit final plan.", {}, submit_plan),
        }

    def observation(self):
        return self.observation_template.format_map(self.state)

    def goal(self, state):
        return (
            state["submitted"]
            and state["revenue"] >= state["target_revenue"]
            and state["cash"] >= 0
            and state["satisfaction"] >= 0.6
        ), f"revenue={state['revenue']:.0f}, cash={state['cash']:.0f}, sat={state['satisfaction']:.2f}"
```

- **Latent state:** `price_sensitivity`, `employee_motivation`, `competitor_aggressiveness`, `market_trend`.  
- **Observable surface:** `day, cash, revenue, inventory, price, competitor_price, satisfaction, retention`.  
- **Probe tools:** `survey_customers`, `check_morale`, `spy_competitor`, `market_research` — each returns a noisy sample of one or more latent variables.  
- **Goal predicate:** `submitted and revenue ≥ target_revenue and cash ≥ 0 and satisfaction ≥ 0.6`.  
- **Experimentation:** the model must call probes, decide `price`, `ad_spend`, `hire`, `inventory`, then `end_day`. Because demand is nonlinear in `price_sensitivity`, it cannot optimize price without first probing.

---

## 4. Experimentation environments (#9)

`env_experimentation.py` extends the hidden-rule company env with an explicit belief and an information-gain bonus.

### 4.1 Belief representation

```python
from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class CategoricalBelief:
    bins: List[float]                      # e.g. [0.5, 1.0, 1.5]
    probs: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.probs:
            n = len(self.bins)
            self.probs = [1.0 / n] * n
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]

    def entropy(self):
        return -sum(p * math.log(p) for p in self.probs if p > 0)

    def normalize(self):
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]
```

```python
belief: Dict[str, CategoricalBelief] = {
    "price_sensitivity": CategoricalBelief(bins=[0.8, 1.2, 1.6]),
    "employee_motivation": CategoricalBelief(bins=[0.7, 1.0, 1.3]),
    "competitor_aggressiveness": CategoricalBelief(bins=[0.2, 0.5, 0.8]),
    "market_trend": CategoricalBelief(bins=[-0.05, 0.05, 0.15]),
}
```

### 4.2 Likelihood and update rule

Each probe tool has a sensor model `P(observation | latent_value)`. For continuous observations we discretize by assigning likelihood proportional to a Gaussian:

```python
def likelihood(observed: float, true: float, sigma: float = 0.15) -> float:
    return max(1e-6, math.exp(-0.5 * ((observed - true) / sigma) ** 2))
```

Update after a probe returns `obs`:

```python
def update_belief(belief: CategoricalBelief, observed: float, var: str):
    for i, bin_val in enumerate(belief.bins):
        belief.probs[i] *= likelihood(observed, bin_val, SENSOR_SIGMA[var])
    belief.normalize()
```

### 4.3 Per-step reward

```python
def information_gain(belief_before: Dict[str, CategoricalBelief],
                     belief_after: Dict[str, CategoricalBelief]) -> float:
    h_before = sum(b.entropy() for b in belief_before.values())
    h_after = sum(b.entropy() for b in belief_after.values())
    return max(0.0, h_before - h_after)


def step_reward(env_state, goal_achieved: bool, ig: float, lambda_ig=0.2):
    r_goal = 1.0 if goal_achieved else 0.0
    return r_goal + lambda_ig * ig
```

`ExperimentationEnv` wraps `HiddenRuleEnv` and maintains `self.belief`. On every tool call it:

1. snapshots `belief_before`,
2. applies the tool,
3. if the tool is a probe, `update_belief` for the relevant latent variable(s),
4. computes `ig = information_gain(before, after)`,
5. returns `step_reward(...)` (terminal reward also adds the final `R_goal`).

This forces the model to trade off probe cost (calls/day) against the value of information for later decisions.

---

## 5. Failure-first environments (#12)

`env_failure_first.py` injects faults into the tool-calling engine. The `ToolCallingTask` gains a `faults: List[Fault]` field and a custom `replay` applies them.

### 5.1 Fault dataclass

```python
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Fault:
    kind: str       # wrong_assumption, broken_tool, missing_dependency,
                    # misleading_info, resource_exhaustion
    trigger: str    # tool name that triggers the fault
    payload: dict   # fault-specific data
```

### 5.2 Five concrete fault scenarios

```python
FAULT_SCENARIOS = [
    # 1. wrong_assumption
    Fault(
        kind="wrong_assumption",
        trigger="read_inventory",
        payload={"stale": True, "true_count": 120, "stale_count": 40},
    ),
    # 2. broken_tool
    Fault(
        kind="broken_tool",
        trigger="submit_order",
        payload={"exception": "ToolConnectionError: submit_order unavailable"},
    ),
    # 3. missing_dependency
    Fault(
        kind="missing_dependency",
        trigger="ship_package",
        payload={"missing": "shipping_label", "need_first": "create_label"},
    ),
    # 4. misleading_info
    Fault(
        kind="misleading_info",
        trigger="demand_forecast",
        payload={"reported": 5000, "true": 1200},
    ),
    # 5. resource_exhaustion
    Fault(
        kind="resource_exhaustion",
        trigger="rate_limited_api",
        payload={"budget_cut": 0.5, "after_call": 3},
    ),
]
```

The replay logic injects the fault when the trigger tool is called:

```python
def _apply_fault(tool_name, state, fault: Fault):
    if tool_name != fault.trigger:
        return None
    if fault.kind == "wrong_assumption":
        return fault.payload["stale_count"]  # return stale data
    if fault.kind == "broken_tool":
        raise RuntimeError(fault.payload["exception"])
    if fault.kind == "missing_dependency":
        if not state.get(fault.payload["need_first"]):
            return None, f"ERROR: {fault.payload['missing']} not found (call {fault.payload['need_first']} first)"
    if fault.kind == "misleading_info":
        return fault.payload["reported"], f"{fault.payload['reported']}"  # plausible but wrong
    if fault.kind == "resource_exhaustion":
        if state.get("calls_made", 0) >= fault.payload["after_call"]:
            state["max_calls"] = int(state["max_calls"] * fault.payload["budget_cut"])
            return None, "ERROR: rate limit hit, budget halved"
    return None
```

### 5.3 Reward formula

```
R_success   = 1.0 if goal reached and final answer correct else 0.0
R_recovery  = 0.4 · detected + 0.4 · corrective + 0.2 · audit_honest
R_catastrophic = 1.0 if destructive_tool_called_on_bad_data
                = 0.5 if model gave up / exceeded budget / made no recovery attempt
                = 0.0 otherwise

R = 0.5 · R_success + 0.3 · R_recovery + 0.2 · (1 − R_catastrophic)
```

- **detected:** response mentions the fault or calls a probe tool (`audit_inventory`, `check_forecast`) that reveals the mismatch.  
- **corrective:** the response uses an alternative tool (`submit_order_backup`, `create_label`, `historical_demand`, `fallback_api`).  
- **audit_honest:** the `AUDIT:` section says PASS iff all tests pass.  
- **catastrophic:** a `write_file` / `delete_path` / `ship_package` is called on unverified data, or the model exceeds `max_calls` without a diagnosis.

---

## 6. Multi-agent evolutionary environments (#10, lightweight)

`population_evolution.py` implements a **population of trajectories**, not a full multi-agent simulation. It reuses existing components:

- `TrajectoryRecombinationEnv` (`trajectory_recombination.py L750`) for recombination.
- `DynamicsMutator` (`environment_mutation.py L373`) or a simple prompt mutator for mutation.

### 6.1 Generation loop

```python
from iloptimus.core.rl_factory.environments.trajectory_recombination import (
    TrajectoryRecombinationEnv, TrajectoryRecombinationVerifier
)
from iloptimus.core.rl_factory.environments.environment_mutation import DynamicsMutator


class PopulationEvolution:
    def __init__(self, base_problem, pop_size=32, elite_frac=0.25,
                 mutate_prob=0.2, recombine_prob=0.5, generations=10):
        self.base_problem = base_problem
        self.pop_size = pop_size
        self.elites = int(pop_size * elite_frac)
        self.mutate_prob = mutate_prob
        self.recombine_prob = recombine_prob
        self.generations = generations
        self.recombination_env = TrajectoryRecombinationEnv(problems=[base_problem], batch_size=pop_size)

    def run_generation(self, population: List[str]) -> dict:
        # 1. score all trajectories
        scores = []
        for resp in population:
            result = self.recombination_env._verifier.verify(resp)
            scores.append(result.score)

        # 2. selection
        indexed = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [p for _, p in indexed[:self.elites]]

        # 3. recombination + mutation
        offspring = []
        while len(offspring) < self.pop_size - self.elites:
            p1 = self._tournament_select(indexed)
            p2 = self._tournament_select(indexed)
            if self.rng.random() < self.recombine_prob:
                child = self._recombine(p1, p2)
            else:
                child = p1
            if self.rng.random() < self.mutate_prob:
                child = self._mutate(child)
            offspring.append(child)

        return elites + offspring, {"best": indexed[0][0], "mean": sum(scores)/len(scores)}

    def _recombine(self, a, b):
        # reuses TrajectoryRecombinationVerifier.recombine
        result = self.recombination_env._verifier.recombine([a, b])
        # extract best recombined code
        return result.metadata.get("best_code", a)

    def _mutate(self, trajectory):
        # lightweight: apply DynamicsMutator to the problem and re-prompt,
        # or perturb the trajectory with a paraphrase model
        return self.env_mutator.mutate(trajectory)

    def evolve(self, initial_population: List[str] = None):
        pop = initial_population or self._random_population()
        history = []
        for gen in range(self.generations):
            pop, metrics = self.run_generation(pop)
            history.append(metrics)
        return pop, history
```

### 6.2 Plug-in to RSI loops

Add to `iloptimus/core/rsi_loops.py`:

```python
LOOP_KINDS = ["coding", "reasoning", "math", "tool-calling", "agentic", "evolutionary"]

# in LOOP_TEMPLATES append:
{
    "id": "loop-evolutionary-trajectory",
    "kind": "evolutionary",
    "name": "Trajectory population evolution",
    "objective": "Run N rollouts, select elites, mutate with DynamicsMutator, recombine with TrajectoryRecombinationEnv, and keep the best trajectory per generation.",
    "default_minutes": 45,
    "default_iterations": 10,
    "needs_sandbox": False,
}
```

Each RSI iteration corresponds to one generation.

---

## 7. Integration plan

### 7.1 New files

| File | Responsibility |
|------|---------------|
| `iloptimus/core/env_skill_collision.py` | `SkillCollisionToolEnv` + 3 `ToolCallingTask` task builders; Pareto-aware batch scalarization. |
| `iloptimus/core/env_hidden_rule.py` | `HiddenRuleEnv` base + `SimulatedCompanyHiddenRuleEnv`; wraps `StateMachineRuntime`. |
| `iloptimus/core/env_experimentation.py` | `ExperimentationEnv` extends `HiddenRuleEnv`; `CategoricalBelief`, `information_gain`, `update_belief`. |
| `iloptimus/core/env_failure_first.py` | `Fault` dataclass, `FailureFirstToolEnv`, 5 fault scenarios, recovery/catastrophic reward. |
| `iloptimus/core/population_evolution.py` | `PopulationEvolution` loop using `TrajectoryRecombinationEnv` and `DynamicsMutator`. |

### 7.2 Registry edits

`iloptimus/core/tasksets.py` L23-L243 — append new taskset entries:

```python
{
    "id": "behavioral-skill-collision-v1",
    "name": "Behavioral Skill Collision v1",
    "package_name": "iloptimus.core.env_skill_collision",
    "path": "iloptimus/core/env_skill_collision",
    "domain": "behavioral-skill-collision",
    "description": "Tool-calling tasks where two skills interfere; rewards Pareto-aware tradeoff reasoning.",
    "num_tasks": 3,
    "needs_sandbox": False,
    "tags": ["behavioral", "tradeoff", "tools", "multi-objective"],
    "eval_config": {"num_examples": 3, "rollouts_per_example": 8},
},
{
    "id": "behavioral-hidden-rule-v1",
    "name": "Behavioral Hidden Rule v1",
    "package_name": "iloptimus.core.env_hidden_rule",
    "path": "iloptimus/core/env_hidden_rule",
    "domain": "behavioral-hidden-rule",
    "description": "Simulated company with latent dynamics; model must discover rules by probing.",
    "num_tasks": 4,
    "needs_sandbox": False,
    "tags": ["behavioral", "hidden-rule", "experimentation"],
    "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
},
{
    "id": "behavioral-experimentation-v1",
    "name": "Behavioral Experimentation v1",
    "package_name": "iloptimus.core.env_experimentation",
    "path": "iloptimus/core/env_experimentation",
    "domain": "behavioral-experimentation",
    "description": "Reward = goal + λ·information_gain over a belief about latent state.",
    "num_tasks": 4,
    "needs_sandbox": False,
    "tags": ["behavioral", "experimentation", "information-gain"],
    "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
},
{
    "id": "behavioral-failure-first-v1",
    "name": "Behavioral Failure First v1",
    "package_name": "iloptimus.core.env_failure_first",
    "path": "iloptimus/core/env_failure_first",
    "domain": "behavioral-failure-first",
    "description": "Tool-calling tasks with injected faults; score recovery and resilience.",
    "num_tasks": 5,
    "needs_sandbox": False,
    "tags": ["behavioral", "failure-first", "resilience", "tools"],
    "eval_config": {"num_examples": 5, "rollouts_per_example": 6},
},
```

`iloptimus/core/grader.py` — add domain-to-grader mapping around L499-L526:

```python
_BEHAVIORAL_GRADERS = {
    "behavioral-skill-collision": grade_skill_collision,
    "behavioral-hidden-rule": grade_hidden_rule,
    "behavioral-experimentation": grade_experimentation,
    "behavioral-failure-first": grade_failure_first,
}
_GRADERS.update(_BEHAVIORAL_GRADERS)
_INSTRUCTIONS.update({d: _TOOL_INSTRUCTION for d in _BEHAVIORAL_GRADERS})
```

Each `grade_*` function imports the corresponding `Env` class, runs `replay`/`score`, and returns a `GradedResult`.

`iloptimus/core/rsi_loops.py` L24-L117 — add `evolutionary` kind and template (see §6.2).

---

## 8. Validation plan

One minimal test per environment type using a stub “model” that emits fixed responses.

### 8.1 Skill collision

```python
def test_skill_collision():
    from iloptimus.core.env_skill_collision import SkillCollisionEnv
    env = SkillCollisionEnv(batch_size=4)
    obs, info = env.reset(seed=0)
    # stub: prefer verified lookup (robust) path
    responses = [
        "<tool>{\"name\":\"verified_lookup\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"submit_answer\",\"args\":{\"value\":\"42\"}}</tool>\n"
        "<answer>42</answer>",
    ] * 4
    obs, reward, terminated, truncated, info = env.step(responses)
    assert 0.0 <= reward <= 1.0
    assert info["aggregate"]["best_score"] >= 0.7
```

### 8.2 Hidden rule

```python
def test_hidden_rule():
    from iloptimus.core.env_hidden_rule import SimulatedCompanyHiddenRuleEnv
    env = SimulatedCompanyHiddenRuleEnv(seed=1)
    # stub: probe, set price, end days, submit
    response = (
        "<tool>{\"name\":\"survey_customers\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"set_price\",\"args\":{\"price\":\"9.5\"}}</tool>\n"
        "<tool>{\"name\":\"order_inventory\",\"args\":{\"units\":\"1000\"}}</tool>\n"
        + "<tool>{\"name\":\"end_day\",\"args\":{}}</tool>\n" * 7 +
        "<tool>{\"name\":\"submit_plan\",\"args\":{}}</tool>"
    )
    trace = env.replay(response)
    reached, _ = env.goal(trace.state)
    assert isinstance(reached, bool)
```

### 8.3 Experimentation

```python
def test_experimentation():
    from iloptimus.core.env_experimentation import ExperimentationEnv
    env = ExperimentationEnv(seed=0)
    obs, info = env.reset(seed=0)
    # stub: probe every latent variable
    response = (
        "<tool>{\"name\":\"survey_customers\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"check_morale\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"spy_competitor\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"market_research\",\"args\":{}}</tool>\n"
        "<tool>{\"name\":\"submit_plan\",\"args\":{}}</tool>"
    )
    obs, reward, terminated, truncated, info = env.step(response)
    assert "information_gain" in info
    assert info["belief_entropy_after"] < info["belief_entropy_before"]
```

### 8.4 Failure first

```python
def test_failure_first():
    from iloptimus.core.env_failure_first import FailureFirstToolEnv, FAULT_SCENARIOS
    env = FailureFirstToolEnv(FAULT_SCENARIOS[1])  # broken_tool
    response = (
        "DETECT: submit_order raised ToolConnectionError\n"
        "RECOVER: use submit_order_backup\n"
        "<tool>{\"name\":\"submit_order_backup\",\"args\":{}}</tool>\n"
        "AUDIT: PASS"
    )
    score, breakdown = env.score(response)
    assert 0.0 <= score <= 1.0
    assert breakdown["r_recovery"] > 0.0
```

### 8.5 Population evolution

```python
def test_population_evolution():
    from iloptimus.core.population_evolution import PopulationEvolution
    from iloptimus.core.rl_factory.environments.trajectory_recombination import trajectory_recombination_generator
    problem = trajectory_recombination_generator(seed=0)
    pop = PopulationEvolution(problem, pop_size=8, generations=2)
    final, history = pop.evolve()
    assert len(final) == 8
    assert len(history) == 2
    assert all(0.0 <= h["best"] <= 1.0 for h in history)
```

---

## 9. Recommended next actions

1. Land `env_skill_collision.py`, `env_hidden_rule.py`, `env_experimentation.py`, `env_failure_first.py`, and `population_evolution.py` in `iloptimus/core/`.
2. Wire the new domains into `tasksets.py`, `grader.py`, and `rsi_loops.py`.
3. Run the five stub-model tests in §8 to verify end-to-end scoring.
4. Once tests pass, run a small RSI loop (`loop-evolutionary-trajectory`) on `behavioral-skill-collision-v1` and inspect the Pareto front produced by `SkillCollisionEnv._aggregate_scores`.
```

---

### Key findings / references

- The tool-calling engine already supports distractor tools, destructive tools, and a `max_calls` budget, but the `faults` field is currently only documented and not enforced in `replay` (`engine.py L94-L131`). `env_failure_first.py` must implement fault injection inside the tool `fn`s or in a custom replay wrapper.
- `StateMachineRuntime` (`stateful_environments.py L180`) stores state and evaluates terminals but is not directly parameterized for tool calls. `env_hidden_rule.py` should keep a `StateMachineRuntime` instance for state/terminals and apply `ToolSpec` functions directly to `runtime.state`.
- `BatchEnvBase` (`batch_base.py L35`) already handles `N` parallel responses and requires only `_aggregate_scores` to be overridden, which is ideal for skill-collision Pareto bonuses and population evolution.
- Existing `SkillCollisionEnv` and `FailureFirstEnv` in `rl_factory/environments/` are code-oriented prototypes. The new core modules should reuse their reward ideas but be built on `ToolCallingTask` and `StateMachineRuntime` for tighter integration with the IL tool-calling and stateful environment stacks.
- `TrajectoryRecombinationEnv` (`trajectory_recombination.py L750`) and `DynamicsMutator` (`environment_mutation.py L373`) are the exact components to reuse for the lightweight evolutionary loop.
