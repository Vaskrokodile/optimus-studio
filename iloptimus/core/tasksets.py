"""Taskset discovery — scans the iloptimus repo for verifiers.v1 tasksets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TasksetInfo:
    id: str
    name: str
    package_name: str
    path: str
    domain: str  # "coding", "reasoning", "agentic-reasoning", "agentic-coding"
    description: str
    num_tasks: int
    needs_sandbox: bool
    tags: list[str] = field(default_factory=list)
    eval_config: dict = field(default_factory=dict)


# Static registry of the 4 IL tasksets (avoids importing verifiers at scan time)
TASKSET_REGISTRY: list[dict] = [
    {
        "id": "il-coding-v1",
        "name": "IL Coding v1",
        "package_name": "il_coding_v1",
        "path": "il_coding_v1",
        "domain": "coding",
        "description": "12 handcrafted coding tasks: algorithm implementation, debugging, refactoring, edge-case handling. Sandboxed test execution with anti-laziness and efficiency-aware reward shaping.",
        "num_tasks": 12,
        "needs_sandbox": True,
        "tags": ["code", "il", "single-turn", "execution"],
        "eval_config": {"num_examples": 12, "rollouts_per_example": 4},
    },
    {
        "id": "il-reasoning-v1",
        "name": "IL Reasoning v1",
        "package_name": "il_reasoning_v1",
        "path": "il_reasoning_v1",
        "domain": "reasoning",
        "description": "12 handcrafted pure-reasoning puzzles: knights & knaves, constraint scheduling, loop invariants, type inference, path counting, zebra logic, recursive traces, set operations, probability, graph cycles, combinatorial counting. Deterministic verification, no sandbox needed.",
        "num_tasks": 12,
        "needs_sandbox": False,
        "tags": ["reasoning", "il", "single-turn"],
        "eval_config": {"num_examples": 12, "rollouts_per_example": 4},
    },
    {
        "id": "il-agentic-reasoning-v1",
        "name": "IL Agentic Reasoning v1",
        "package_name": "il_agentic_reasoning_v1",
        "path": "il_agentic_reasoning_v1",
        "domain": "agentic-reasoning",
        "description": "10 handcrafted multi-step reasoning scenarios: cascading pipeline traces, cross-module data flow, invariant preservation, race conditions, API contract compliance, recursive repair, state machine simulation, differential analysis, error propagation, coverage gap analysis. Sustained deduction where each step depends on the previous.",
        "num_tasks": 10,
        "needs_sandbox": False,
        "tags": ["reasoning", "agentic", "il", "long-horizon"],
        "eval_config": {"num_examples": 10, "rollouts_per_example": 4},
    },
    {
        "id": "il-agentic-coding-v1",
        "name": "IL Agentic Coding v1",
        "package_name": "il_agentic_coding_v1",
        "path": "il_agentic_coding_v1",
        "domain": "agentic-coding",
        "description": "10 handcrafted multi-file codebase scenarios: cascading bug chains, codebase navigation, refactoring, error handling, API client impl, dead-code removal, type annotations, perf optimization, config fixes, test-writing for mutants. Sandboxed multi-file test harness with anti-laziness.",
        "num_tasks": 10,
        "needs_sandbox": True,
        "tags": ["code", "agentic", "il", "multi-file", "execution"],
        "eval_config": {"num_examples": 10, "rollouts_per_example": 4},
    },
    {
        "id": "humaneval-v1",
        "name": "HumanEval v1",
        "package_name": "humaneval_v1",
        "path": "humaneval_v1",
        "domain": "humaneval",
        "description": "25 curated HumanEval coding benchmark problems: string manipulation, math, algorithms, data structures, edge cases. Sandboxed test execution with anti-laziness and efficiency-aware reward shaping. Used by the self-improvement loop to benchmark and train on real coding tasks.",
        "num_tasks": 25,
        "needs_sandbox": True,
        "tags": ["code", "humaneval", "benchmark", "single-turn", "execution"],
        "eval_config": {"num_examples": 25, "rollouts_per_example": 4},
    },
    {
        "id": "gsm8k-v1",
        "name": "GSM8K v1",
        "package_name": "gsm8k_v1",
        "path": "gsm8k_v1",
        "domain": "gsm8k",
        "description": "25 curated GSM8K grade-school math word problems: arithmetic, multi-step reasoning, unit conversion, percentages, fractions, rates, geometry, logic. Deterministic numeric verification with efficiency-aware reward shaping. Used by the self-improvement loop to benchmark and train on math reasoning.",
        "num_tasks": 25,
        "needs_sandbox": False,
        "tags": ["math", "gsm8k", "benchmark", "reasoning", "single-turn"],
        "eval_config": {"num_examples": 25, "rollouts_per_example": 4},
    },
    {
        "id": "aime-2025",
        "name": "AIME 2025",
        "package_name": "aime_2025",
        "path": "aime_2025",
        "domain": "aime",
        "description": "30 AIME 2025 competition math problems from test-time-compute/aime_2025. Integer answers 0-999, deterministic verification via boxed answer extraction. The benchmark on which OmniCoder-9B + checkpoint-50 achieved 19/30 (pass@5). Used for continued RL training toward general intelligence.",
        "num_tasks": 30,
        "needs_sandbox": False,
        "tags": ["math", "aime", "competition", "benchmark", "reasoning", "single-turn"],
        "eval_config": {"num_examples": 30, "rollouts_per_example": 5},
    },
]


def get_all_tasksets() -> list[TasksetInfo]:
    builtins = [
        TasksetInfo(
            id=t["id"],
            name=t["name"],
            package_name=t["package_name"],
            path=t["path"],
            domain=t["domain"],
            description=t["description"],
            num_tasks=t["num_tasks"],
            needs_sandbox=t["needs_sandbox"],
            tags=t["tags"],
            eval_config=t["eval_config"],
        )
        for t in TASKSET_REGISTRY
    ]
    from .environments import list_environments

    custom = [
        TasksetInfo(
            id=environment["taskset_id"],
            name=environment["name"],
            package_name=f"user_environment_{environment['id'].replace('-', '_')}",
            path=str(environment["id"]),
            domain=f"custom:{environment['id']}",
            description=environment["description"],
            num_tasks=len(environment["tasks"]),
            needs_sandbox=False,
            tags=[environment["mode"].lower(), "custom", environment["domain"]],
            eval_config={"num_examples": len(environment["tasks"]), "rollouts_per_example": 4},
        )
        for environment in list_environments()
    ]
    return builtins + custom


def get_taskset(taskset_id: str) -> TasksetInfo | None:
    for t in get_all_tasksets():
        if t.id == taskset_id:
            return t
    return None
