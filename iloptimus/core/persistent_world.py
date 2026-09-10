"""Persistent World (#16) — a shared mutable company world across tasks.

The world is a single JSON document under ``~/.iloptimus/worlds/<world_id>/``
that advances simulated time after each episode and derives new tasks from
active triggers (degraded servers, churn risk, low runway, ...). Each episode
projects a slice of the world into a state-machine simulator (reusing
``stateful_environments.StateMachineRuntime``), runs the model, and writes the
final scalar state back into the world. Failure has persistent consequences.

See ``RESEARCH_meta_architecture.md`` Section 3.
"""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, worlds_dir


# ---------------------------------------------------------------------------
# World schema (initial state)
# ---------------------------------------------------------------------------


def _initial_world(world_id: str) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "tick": 0,
        "sim_time": "2026-01-01T00:00:00Z",
        "employees": [
            {
                "id": "e1", "role": "sre", "skill": "devops",
                "morale": 0.80, "load": 0.40, "salary": 120_000, "on_call": True,
            },
            {
                "id": "e2", "role": "eng", "skill": "coding",
                "morale": 0.70, "load": 0.55, "salary": 140_000, "on_call": False,
            },
        ],
        "customers": [
            {
                "id": "c1", "tier": "enterprise",
                "satisfaction": 0.75, "churn_risk": 0.20, "open_ticket": None,
            },
            {
                "id": "c2", "tier": "standard",
                "satisfaction": 0.60, "churn_risk": 0.45, "open_ticket": "billing-22",
            },
        ],
        "finances": {
            "cash": 1_000_000, "revenue": 100_000,
            "expenses": 80_000, "runway_months": 12.0,
        },
        "servers": [
            {
                "id": "srv-3", "health": 0.55, "cpu_load": 92,
                "disk_usage": 0.78, "last_restart_tick": 0, "role": "api",
            },
            {
                "id": "srv-1", "health": 0.92, "cpu_load": 40,
                "disk_usage": 0.45, "last_restart_tick": 0, "role": "db",
            },
        ],
        "code": {
            "repo": "shop", "bug_count": 12, "open_prs": 5,
            "test_pass_rate": 0.87, "tech_debt_score": 0.30,
        },
        "inventory": {"sku_count": 340, "low_stock_count": 7, "backlog": 23},
        "competitors": [
            {"id": "comp-1", "market_share": 0.25, "price_index": 1.00},
        ],
        "contracts": [
            {
                "id": "ct-7", "value": 50_000, "deadline_tick": 168,
                "status": "in_negotiation", "sla_breach": False,
            },
        ],
        "market": {"trend": 0.02, "demand_index": 1.0, "season": "post_holiday"},
        "policies": {"max_refund": 500, "sla_hours": 24},
        "unresolved_tickets": [],
    }


# ---------------------------------------------------------------------------
# Trigger rules — (skill, predicate over world) -> list of (key, obj)
# ---------------------------------------------------------------------------


TRIGGER_RULES: list[tuple[str, Any]] = [
    ("debugging", lambda w: [(s["id"], s) for s in w["servers"] if s["health"] < 0.60]),
    ("negotiation", lambda w: [(c["id"], c) for c in w["customers"] if c["churn_risk"] > 0.70]),
    ("finance", lambda w: [("runway", w["finances"])] if w["finances"]["runway_months"] < 3.0 else []),
    ("operations", lambda w: [("inventory", w["inventory"])] if w["inventory"]["low_stock_count"] > 5 else []),
    ("coding", lambda w: [("repo", w["code"])] if w["code"]["bug_count"] > 10 else []),
    ("strategy", lambda w: [("pricing", c) for c in w["competitors"] if c["price_index"] < 0.90]),
    ("planning", lambda w: [
        ("contract", c) for c in w["contracts"]
        if c["deadline_tick"] - w["tick"] < 48 and c["status"] != "signed"
    ]),
]


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    task_id: str
    prompt: str
    domain: str = "persistent_world"
    source: str = "persistent_world"
    difficulty: float = 0.5
    metadata: dict = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _add_hours(iso_ts: str, hours: int) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime(2026, 1, 1)
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _estimate_difficulty(world: dict[str, Any], skill: str, key: str) -> float:
    """Heuristic difficulty in [0,1] from the trigger's urgency."""
    if skill == "debugging":
        s = next((s for s in world["servers"] if s["id"] == key), None)
        if s:
            return _clamp(1.0 - s["health"])
    if skill == "negotiation":
        c = next((c for c in world["customers"] if c["id"] == key), None)
        if c:
            return _clamp(c["churn_risk"])
    if skill == "finance":
        return _clamp(1.0 - world["finances"]["runway_months"] / 12.0)
    if skill == "operations":
        return _clamp(world["inventory"]["low_stock_count"] / 20.0)
    if skill == "coding":
        return _clamp(world["code"]["bug_count"] / 30.0)
    if skill == "planning":
        c = next((c for c in world["contracts"] if c["id"] == key), None)
        if c:
            return _clamp(1.0 - (c["deadline_tick"] - world["tick"]) / 48.0)
    return 0.5


def _render_task_prompt(world: dict[str, Any], skill: str, key: str, obj: Any) -> str:
    if skill == "debugging":
        s = obj
        return (
            f"Server `{s['id']}` health is {s['health']:.2f} and CPU load is {s['cpu_load']}. "
            f"Diagnose the cause and return the minimal remediation actions as a JSON list of "
            f"action names from: check_logs, restart_server, scale_up. "
            f"The company finances and SLA are in the world state."
        )
    if skill == "negotiation":
        c = obj
        return (
            f"Customer `{c['id']}` (tier {c['tier']}) has churn risk {c['churn_risk']:.2f} "
            f"and an open ticket. Use the support tools to recover them without violating the "
            f"`max_refund` policy (max ${world['policies']['max_refund']})."
        )
    if skill == "finance":
        return (
            f"Runway is {world['finances']['runway_months']:.1f} months. Propose actions to "
            f"restore runway above 3 months while preserving customer satisfaction."
        )
    if skill == "operations":
        return (
            f"Low-stock count is {world['inventory']['low_stock_count']} with backlog "
            f"{world['inventory']['backlog']}. Reorder and allocate to clear the backlog."
        )
    if skill == "coding":
        return (
            f"Repo `{world['code']['repo']}` has {world['code']['bug_count']} bugs and "
            f"test pass rate {world['code']['test_pass_rate']:.2f}. Triage and fix the highest-"
            f"impact bugs; return the patch as a JSON list of file:fix pairs."
        )
    if skill == "strategy":
        return (
            f"Competitor `{obj['id']}` dropped price_index to {obj['price_index']:.2f}. "
            f"Update pricing and marketing while preserving runway > 3 months."
        )
    if skill == "planning":
        c = obj
        return (
            f"Contract `{c['id']}` (value ${c['value']}) has deadline_tick "
            f"{c['deadline_tick']} (current tick {world['tick']}). Plan delivery to meet the SLA."
        )
    return f"Resolve the active trigger for skill `{skill}` (key `{key}`)."


# ---------------------------------------------------------------------------
# Projected state-machine simulators (one per trigger skill)
# ---------------------------------------------------------------------------


def _project_simulator(world: dict[str, Any], skill: str, key: str, obj: Any) -> dict[str, Any]:
    """Build a state-machine simulator slice for one trigger."""
    if skill == "debugging":
        s = obj
        return {
            "template_id": f"persistent-world-server-debug-{key}",
            "observation": "server={server_id}, health={health}, cpu={cpu}, tickets={tickets}, reboots={reboots}",
            "scenarios": [{
                "name": key,
                "initial_state": {
                    "server_id": s["id"], "health": s["health"], "cpu": s["cpu_load"],
                    "tickets": len(world["unresolved_tickets"]) or 3, "reboots": 0, "fixed": False,
                },
            }],
            "actions": [
                {
                    "name": "check_logs",
                    "description": "Read the server log and reduce unknown ticket count",
                    "effects": [{"var": "tickets", "op": "add", "value": -1, "min": 0}],
                    "reward": 0.0,
                },
                {
                    "name": "restart_server",
                    "description": "Reboot the server to clear high CPU",
                    "requires": {"var": "cpu", "op": "gt", "value": 80},
                    "effects": [
                        {"var": "cpu", "op": "set", "value": 40},
                        {"var": "reboots", "op": "add", "value": 1},
                        {"var": "health", "op": "add", "value": 0.05, "max": 1.0},
                    ],
                    "reward": 0.0,
                },
                {
                    "name": "scale_up",
                    "description": "Provision a new instance after logs are checked",
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
            "rewards": {"step": -0.02, "invalid": -0.05, "timeout": -0.2},
            "max_steps": 12,
        }
    # Generic fallback simulator for non-debugging triggers: a single binary
    # "resolve" action whose success is graded by the outcome projection.
    return {
        "template_id": f"persistent-world-{skill}-{key}",
        "observation": f"trigger={skill}:{key}, resolve the active issue",
        "scenarios": [{
            "name": key,
            "initial_state": {"resolved": False, "attempted": False},
        }],
        "actions": [
            {
                "name": "resolve",
                "description": f"Resolve the {skill} trigger for {key}",
                "effects": [
                    {"var": "resolved", "op": "set", "value": True},
                    {"var": "attempted", "op": "set", "value": True},
                ],
                "reward": 0.0,
            },
        ],
        "terminals": [
            {"when": {"var": "resolved", "op": "eq", "value": True}, "outcome": "resolved", "success": True, "reward": 1.0},
        ],
        "rewards": {"step": -0.02, "invalid": -0.05, "timeout": -0.2},
        "max_steps": 6,
    }


# ---------------------------------------------------------------------------
# PersistentWorld
# ---------------------------------------------------------------------------


class PersistentWorld:
    """A shared mutable company world that persists across tasks."""

    def __init__(self, world_id: str | None = None, checkpoint_every: int = 10) -> None:
        self.world_id = world_id or f"world-{uuid.uuid4().hex[:10]}"
        self.root = worlds_dir() / self.world_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.checkpoint_every = checkpoint_every
        self.world = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return _initial_world(self.world_id)

    # -- persistence ------------------------------------------------------

    def save(self) -> None:
        atomic_write_json(self.state_path, self.world)
        if self.world["tick"] % self.checkpoint_every == 0:
            ckpt = self.root / "checkpoints"
            ckpt.mkdir(parents=True, exist_ok=True)
            atomic_write_json(ckpt / f"{self.world['tick']}.json", self.world)

    def log_event(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, world_id: str) -> "PersistentWorld | None":
        path = worlds_dir() / world_id / "state.json"
        if not path.exists():
            return None
        return cls(world_id)

    def public(self) -> dict[str, Any]:
        return copy.deepcopy(self.world)

    # -- time advancement -------------------------------------------------

    def advance(self, actions: list[dict[str, Any]] | None = None) -> None:
        """Advance the world by one tick, applying deterministic evolution."""
        actions = actions or []
        self.world["tick"] += 1
        self.world["sim_time"] = _add_hours(self.world["sim_time"], 1)

        for server in self.world["servers"]:
            if server["cpu_load"] > 80:
                server["health"] -= 0.01 * (server["cpu_load"] / 100.0)
            server["health"] = _clamp(server["health"])
            for action in actions:
                if action.get("target") == server["id"]:
                    self._apply_server_action(server, action)

        for customer in self.world["customers"]:
            if customer.get("open_ticket"):
                customer["churn_risk"] += 0.02
            else:
                customer["churn_risk"] -= 0.01
            customer["churn_risk"] = _clamp(customer["churn_risk"])

        for competitor in self.world["competitors"]:
            if random.random() < 0.02:
                competitor["price_index"] -= 0.05
                self.world["market"]["demand_index"] += 0.03

        if self.world["tick"] % 24 == 0:
            rev = self.world["finances"]["revenue"] * self.world["market"]["demand_index"]
            self.world["finances"]["cash"] += rev - self.world["finances"]["expenses"]
            exp = self.world["finances"]["expenses"]
            self.world["finances"]["runway_months"] = (
                self.world["finances"]["cash"] / exp if exp > 0 else 999.0
            )

    def _apply_server_action(self, server: dict[str, Any], action: dict[str, Any]) -> None:
        op = action.get("op")
        if op == "restart":
            server["cpu_load"] = 40
            server["health"] = min(1.0, server["health"] + 0.05)
            server["last_restart_tick"] = self.world["tick"]
        elif op == "scale_up":
            server["cpu_load"] = 30
        elif op == "patch":
            server["health"] = min(1.0, server["health"] + 0.10)

    # -- task derivation --------------------------------------------------

    def active_triggers(self) -> list[tuple[str, str, Any]]:
        triggers: list[tuple[str, str, Any]] = []
        for skill, rule in TRIGGER_RULES:
            for key, obj in rule(self.world):
                triggers.append((skill, key, obj))
        return triggers

    def derive_task(self, focus_skill: str | None = None) -> TaskSpec:
        triggers = self.active_triggers()
        if not triggers:
            # No active trigger: synthesize a low-stakes coding task so the
            # orchestrator always has something to do.
            skill, key, obj = ("coding", "repo", self.world["code"])
        elif focus_skill:
            matches = [t for t in triggers if t[0] == focus_skill]
            if matches:
                skill, key, obj = random.choice(matches)
            else:
                skill, key, obj = random.choice(triggers)
        else:
            skill, key, obj = random.choice(triggers)

        prompt = _render_task_prompt(self.world, skill, key, obj)
        sim = _project_simulator(self.world, skill, key, obj)
        return TaskSpec(
            task_id=f"world-{self.world_id}-{self.world['tick']}-{skill}-{key}",
            prompt=prompt,
            domain="persistent_world",
            source="persistent_world",
            difficulty=_estimate_difficulty(self.world, skill, key),
            metadata={
                "world_id": self.world_id,
                "tick": self.world["tick"],
                "trigger": (skill, key),
                "snapshot": copy.deepcopy(self.world),
                "simulator": sim,
            },
        )

    # -- outcome projection ----------------------------------------------

    def apply_outcome(self, task: TaskSpec, final_state: dict[str, Any], success: bool) -> None:
        """Write the episode's final scalar state back into the world JSON."""
        trigger = task.metadata["trigger"]  # (skill, key)
        skill, key = trigger

        if skill == "debugging":
            server = next((s for s in self.world["servers"] if s["id"] == key), None)
            if server:
                server["health"] = final_state.get("health", server["health"])
                server["cpu_load"] = final_state.get("cpu", server["cpu_load"])
                if not success:
                    server["health"] -= 0.10
                server["health"] = _clamp(server["health"])
        elif skill == "negotiation":
            customer = next((c for c in self.world["customers"] if c["id"] == key), None)
            if customer:
                customer["churn_risk"] = final_state.get("churn_risk", customer["churn_risk"])
                if success:
                    customer["satisfaction"] = min(1.0, customer["satisfaction"] + 0.15)
                    customer["open_ticket"] = None
                else:
                    customer["churn_risk"] = min(1.0, customer["churn_risk"] + 0.30)
        elif skill == "coding":
            if success:
                self.world["code"]["bug_count"] = max(0, self.world["code"]["bug_count"] - 3)
                self.world["code"]["test_pass_rate"] = min(1.0, self.world["code"]["test_pass_rate"] + 0.02)
            else:
                self.world["code"]["tech_debt_score"] = min(1.0, self.world["code"]["tech_debt_score"] + 0.05)
        elif skill == "operations":
            if success:
                self.world["inventory"]["low_stock_count"] = max(0, self.world["inventory"]["low_stock_count"] - 5)
                self.world["inventory"]["backlog"] = max(0, self.world["inventory"]["backlog"] - 10)
        elif skill == "finance":
            if success:
                self.world["finances"]["expenses"] = max(1000, int(self.world["finances"]["expenses"] * 0.9))
        elif skill == "strategy":
            if success:
                self.world["market"]["demand_index"] = min(2.0, self.world["market"]["demand_index"] + 0.05)
        elif skill == "planning":
            contract = next((c for c in self.world["contracts"] if c["id"] == key), None)
            if contract:
                contract["status"] = "signed" if success else "at_risk"
                if not success:
                    contract["sla_breach"] = True

        self.save()
        self.log_event({
            "tick": self.world["tick"],
            "task_id": task.task_id,
            "trigger": list(trigger),
            "success": success,
            "final_state": final_state,
        })


__all__ = [
    "PersistentWorld",
    "TaskSpec",
    "TRIGGER_RULES",
]
