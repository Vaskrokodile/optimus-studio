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
    # ------------------------------------------------------------------
    # IL tool-calling curriculum (10 environments, 4 tasks each)
    # ---------------------------------------------------------------------------
    {
        "id": "tool-fs-navigator-v1",
        "name": "Tool FS Navigator v1",
        "package_name": "il_tool_fs_navigator_v1",
        "path": "il_tool_fs_navigator_v1",
        "domain": "tool-fs",
        "description": "Simulated filesystem tool calls: list/read/grep/write with distractor destructive tools. Teaches explore-before-edit, minimal call count, precise path arguments.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "filesystem", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-sql-analyst-v1",
        "name": "Tool SQL Analyst v1",
        "package_name": "il_tool_sql_analyst_v1",
        "path": "il_tool_sql_analyst_v1",
        "domain": "tool-sql",
        "description": "Schema-first SQL analytics: introspect before querying, explicit projections (SELECT * rejected), query-count efficiency, destructive DDL blocked.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "sql", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-web-research-v1",
        "name": "Tool Web Research v1",
        "package_name": "il_tool_web_research_v1",
        "path": "il_tool_web_research_v1",
        "domain": "tool-web",
        "description": "Web research with distractor sources: search then read the authoritative docs page, avoid blogs/forums/outdated wikis and no-op tools.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "web", "research", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-booking-flow-v1",
        "name": "Tool Booking Flow v1",
        "package_name": "il_tool_booking_flow_v1",
        "path": "il_tool_booking_flow_v1",
        "domain": "tool-booking",
        "description": "tau-bench style booking flows: lookup-before-mutate policy enforced by the simulator, refund/cancel preconditions, rescheduling, confirmation answers.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "booking", "policy", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-pipeline-v1",
        "name": "Tool Pipeline v1",
        "package_name": "il_tool_pipeline_v1",
        "path": "il_tool_pipeline_v1",
        "domain": "tool-pipeline",
        "description": "Multi-tool pipelines with value carrying: authenticate -> list -> fetch -> transform -> submit. Teaches dependency ordering and threading outputs into arguments.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "pipeline", "multi-step"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-error-recovery-v1",
        "name": "Tool Error Recovery v1",
        "package_name": "il_tool_error_recovery_v1",
        "path": "il_tool_error_recovery_v1",
        "domain": "tool-recovery",
        "description": "Fault-injected APIs (429/500): read the error observation, back off, retry or switch to the fallback endpoint, never give up, stay in budget.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "error-recovery", "retry", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-distractor-v1",
        "name": "Tool Distractor Selection v1",
        "package_name": "il_tool_distractor_v1",
        "path": "il_tool_distractor_v1",
        "domain": "tool-distractor",
        "description": "BFCL-style irrelevance: 12-tool pool, only 2 relevant. Every distractor call is penalized; rewards precise tool selection and minimal calls.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "distractors", "selection", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-parallel-v1",
        "name": "Tool Parallel Calls v1",
        "package_name": "il_tool_parallel_v1",
        "path": "il_tool_parallel_v1",
        "domain": "tool-parallel",
        "description": "BFCL-style parallel batching: independent lookups issued together, data dependencies respected, submit only after the full gather.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "parallel", "bfcl", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-interpreter-v1",
        "name": "Tool Code Interpreter v1",
        "package_name": "il_tool_interpreter_v1",
        "path": "il_tool_interpreter_v1",
        "domain": "tool-interpreter",
        "description": "Code interpreter tool calls: execute-to-verify, read tracebacks (ZeroDivisionError), round economy, no unnecessary package installs.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "code-interpreter", "python", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-devops-triage-v1",
        "name": "Tool DevOps Triage v1",
        "package_name": "il_tool_devops_triage_v1",
        "path": "il_tool_devops_triage_v1",
        "domain": "tool-devops",
        "description": "Long-horizon stateful incident triage: health -> logs -> rollback (first attempt fails) -> verify -> escalate only if still degraded. tau-bench style state-diff reward.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "devops", "long-horizon", "stateful"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
    },
    {
        "id": "tool-api-reliability-v1",
        "name": "Tool API Reliability v1",
        "package_name": "il_tool_api_reliability_v1",
        "path": "il_tool_api_reliability_v1",
        "domain": "tool-api",
        "description": "Paged collection API with drifting page counts, duplicates and empty pages under a hard request budget. Teaches cursor pagination, dedup, totals-trap awareness.",
        "num_tasks": 4,
        "needs_sandbox": False,
        "tags": ["tools", "il", "pagination", "budget", "single-turn"],
        "eval_config": {"num_examples": 4, "rollouts_per_example": 4},
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
