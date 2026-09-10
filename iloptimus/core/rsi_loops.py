"""RSI Loops — configurable self-improvement loop sessions.

A loop bundles a model, an objective, a category (coding / reasoning / math /
tool-calling / agentic), a time budget and an iteration cap, then runs on top
of a persistent RSI panel worker. Loops are durable records under
`~/.iloptimus/rsi-loops/`.

Runnability scoring (the green/red bar) combines:
    model fit      — check_compatibility score (precision headroom, backend)
    time budget    — estimated tokens/sec vs iterations the budget allows
    category load  — sandbox-heavy categories (agentic/coding) need more headroom
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .storage import app_home, atomic_write_json

LOOP_KINDS = ["coding", "reasoning", "math", "tool-calling", "agentic", "adaptive", "adversarial"]

LOOP_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "loop-coding-fix",
        "kind": "coding",
        "name": "Coding fix loop",
        "objective": "Solve coding tasks from the IL Coding taskset, verify each fix against the test harness, and record which reasoning patterns led to passing tests.",
        "default_minutes": 30,
        "default_iterations": 5,
        "needs_sandbox": True,
    },
    {
        "id": "loop-refactor-sprint",
        "kind": "coding",
        "name": "Refactor sprint",
        "objective": "Refactor the assigned module for clarity and performance while keeping every test green. Summarize each refactor decision and its measured impact.",
        "default_minutes": 45,
        "default_iterations": 4,
        "needs_sandbox": True,
    },
    {
        "kind": "reasoning",
        "id": "loop-reasoning-deep-dive",
        "name": "Reasoning deep-dive",
        "objective": "Work through multi-step reasoning puzzles, verify each answer before moving on, and distill reusable deduction checklists from mistakes.",
        "default_minutes": 30,
        "default_iterations": 6,
        "needs_sandbox": False,
    },
    {
        "kind": "reasoning",
        "id": "loop-self-critique",
        "name": "Self-critique loop",
        "objective": "Answer a reasoning question, then critique your own answer and revise it. Keep only revisions that measurably improve rigor.",
        "default_minutes": 20,
        "default_iterations": 8,
        "needs_sandbox": False,
    },
    {
        "kind": "math",
        "id": "loop-math-drill",
        "name": "Math drill loop",
        "objective": "Drill competition math problems (GSM8K / AIME style). Verify every numeric answer, log failure modes, and re-attempt the hardest misses.",
        "default_minutes": 40,
        "default_iterations": 6,
        "needs_sandbox": False,
    },
    {
        "kind": "math",
        "id": "loop-proof-sketching",
        "name": "Proof sketching",
        "objective": "Sketch proofs for the given statements, check each step for gaps, and tighten the weakest step every iteration.",
        "default_minutes": 30,
        "default_iterations": 5,
        "needs_sandbox": False,
    },
    {
        "kind": "tool-calling",
        "id": "loop-toolcalling-curriculum",
        "name": "Tool-calling curriculum",
        "objective": "Run the IL tool-calling tasksets (filesystem, SQL, booking, pipelines). Replay the expert trajectory only after attempting it; log every wrong tool or wasted call.",
        "default_minutes": 30,
        "default_iterations": 6,
        "needs_sandbox": False,
    },
    {
        "kind": "tool-calling",
        "id": "loop-error-recovery-gym",
        "name": "Error-recovery gym",
        "objective": "Practice fault-injected API flows: read the error, back off, retry or fall back. Track how often recovery succeeds within budget.",
        "default_minutes": 25,
        "default_iterations": 6,
        "needs_sandbox": False,
    },
    {
        "kind": "agentic",
        "id": "loop-agentic-sprint",
        "name": "Agentic coding sprint",
        "objective": "Navigate a multi-file codebase, fix the reported bug chain, and keep the full test suite green. Minimize files touched and calls made.",
        "default_minutes": 60,
        "default_iterations": 4,
        "needs_sandbox": True,
    },
    {
        "kind": "agentic",
        "id": "loop-devops-triage",
        "name": "DevOps triage drill",
        "objective": "Triage the injected production incident: diagnose before acting, fix, verify, escalate only if needed. Record the triage order and its outcome.",
        "default_minutes": 25,
        "default_iterations": 4,
        "needs_sandbox": False,
    },
    {
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
    },
    {
        "id": "loop-adversarial-env",
        "kind": "adversarial",
        "name": "Adversarial environment gym",
        "objective": (
            "Given a base tool-calling or state-machine task, run an adversarial search to "
            "find a world-dynamics mutation the current model fails but the expert solves. "
            "Train on the resulting curriculum."
        ),
        "default_minutes": 30,
        "default_iterations": 6,
        "needs_sandbox": False,
    },
]


@dataclass
class RsiLoop:
    id: str
    name: str
    kind: str
    model_id: str
    objective: str
    time_budget_minutes: int
    max_iterations: int
    status: str = "draft"  # draft | running | completed | stopped | failed
    panel_id: str = ""
    iterations_done: int = 0
    best_score: float = 0.0
    score_history: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: str = ""
    # Adaptive-loop focus fields (used by kind == "adaptive"). The current
    # focus skill, how many rollouts have been collected on it, and the
    # ordered history of prior focuses are managed by plan_next_composition().
    current_focus: str = ""
    rollouts_on_focus: int = 0
    focus_history: list[str] = field(default_factory=list)
    frontier_profile_id: str = ""  # usually == model_id

    def public(self) -> dict[str, Any]:
        return asdict(self)


class RsiLoopStore:
    def __init__(self, root: Path | None = None):
        self.root = root or app_home() / "rsi-loops"
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, RsiLoop] = {}
        self._load()

    def _path(self, loop_id: str) -> Path:
        return self.root / f"{loop_id}.json"

    def _load(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                import json

                record = RsiLoop(**json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            self._store(record)

    def _records_safe(self, record: RsiLoop) -> None:
        if record.status == "running":
            record.status = "stopped"
        self._records[record.id] = record

    def list(self) -> list[dict[str, Any]]:
        return [r.public() for r in sorted(self._records.values(), key=lambda r: r.created_at)]

    def get(self, loop_id: str) -> RsiLoop | None:
        return self._records.get(loop_id)

    def save(self, loop: RsiLoop) -> RsiLoop:
        loop.updated_at = time.time()
        self._records[loop.id] = loop
        atomic_write_json(self._path(loop.id), loop.public())
        return loop

    def delete(self, loop_id: str) -> bool:
        if loop_id not in self._records:
            return False
        del self._records[loop_id]
        self._path(loop_id).unlink(missing_ok=True)
        return True

    def create(self, payload: dict[str, Any]) -> RsiLoop:
        loop = RsiLoop(
            id=f"loop-{uuid.uuid4().hex[:10]}",
            name=str(payload.get("name") or "Untitled loop")[:80],
            kind=str(payload.get("kind") or "coding"),
            model_id=str(payload.get("model_id") or ""),
            objective=str(payload.get("objective") or ""),
            time_budget_minutes=max(5, min(480, int(payload.get("time_budget_minutes") or 30))),
            max_iterations=max(1, min(50, int(payload.get("max_iterations") or 5))),
        )
        return self.save(loop)


def runnability(model_compat_score: float, model_mem_gb: float, available_gb: float,
                backend_ok: bool, minutes: int, iterations: int, needs_sandbox: bool,
                estimated_tps: float) -> dict[str, Any]:
    """Score how runnable a loop config is on this machine, 0..1 with factors."""
    fit = max(0.0, min(1.0, model_compat_score))

    # time budget: rough estimate — each iteration needs objective work + verify
    est_tokens_per_iter = 4000
    seconds_per_iter = est_tokens_per_iter / max(estimated_tps, 0.5)
    feasible_iters = (minutes * 60) / max(seconds_per_iter, 1)
    time_score = max(0.0, min(1.0, feasible_iters / max(max(feasible_iters, 1), 1)))
    time_score = 1.0 if feasible_iters >= 3 else max(0.15, feasible_iters / 3)

    headroom = max(0.0, available_gb - model_mem_gb)
    headroom_score = min(1.0, headroom / 4.0) if available_gb > 0 else 0.0

    sandbox_penalty = 0.85 if needs_sandbox else 1.0

    score = fit * 0.45 + time_score * 0.25 + headroom_score * 0.20 + (0.10 if model_compat_score >= 0.9 else 0.0)
    score *= sandbox_penalty
    score = max(0.0, min(1.0, score))

    factors = {
        "model_fit": round(fit, 3),
        "time_feasibility": round(time_score, 3),
        "memory_headroom": round(headroom_score, 3),
        "estimated_tps": round(estimated_tps, 2),
        "feasible_iterations": round(feasible_iters, 1),
        "needs_sandbox": needs_sandbox,
    }
    label = "excellent" if score >= 0.8 else "good" if score >= 0.6 else "tight" if score >= 0.35 else "not runnable"
    return {"score": round(score, 3), "label": label, "factors": factors}


# ---------------------------------------------------------------------------
# Adaptive-loop helpers (frontier sampler + composition planning)
# ---------------------------------------------------------------------------


def next_task(loop: RsiLoop, profile=None) -> tuple[Any, dict[str, Any]]:
    """Pick the next (domain, task_idx) for an adaptive loop.

    Returns ``(profile, selection)`` where ``selection`` is a dict with
    ``domain``, ``task_idx``, and ``taskset_id``. Loads the frontier profile
    for the loop's model if one is not supplied.
    """
    from .frontier_sampler import FrontierSampler, load_frontier_profile

    if profile is None:
        profile = load_frontier_profile(loop.frontier_profile_id or loop.model_id)
    sampler = FrontierSampler(profile)
    key = sampler.sample()
    return profile, {
        "domain": key.domain,
        "task_idx": key.task_idx,
        "taskset_id": key.taskset_id,
    }


def plan_next_composition(loop: RsiLoop, profiler, k: int = 1):
    """Select the next skill composition for an adaptive loop.

    Updates ``loop.current_focus`` to the chosen composition's first skill,
    appends the prior focus to ``loop.focus_history``, and resets
    ``loop.rollouts_on_focus``. Returns the chosen ``SkillComposition`` or
    ``None`` if the profiler has no suggestions yet.
    """
    profile = profiler.load(loop.model_id, getattr(loop, "model_fingerprint", "") if hasattr(loop, "model_fingerprint") else "")
    comps = profiler.suggest_compositions(profile, k=k)
    if not comps:
        return None
    chosen = comps[0]
    if loop.current_focus:
        loop.focus_history.append(loop.current_focus)
    loop.current_focus = sorted(chosen.node_ids)[0]
    loop.rollouts_on_focus = 0
    return chosen


def run_frontier_iteration(
    loop: RsiLoop,
    model_generate,
    estimate_params_b: float = 7.0,
) -> dict[str, Any]:
    """Run one frontier-sampler iteration: sample a taskset arm, generate,
    grade, observe, and update the loop's score history.

    ``model_generate`` is a callable ``(prompt: str) -> tuple[str, int]``
    returning ``(response, tokens)``.
    """
    from .frontier_sampler import FrontierSampler, RolloutKey
    from .grader import build_prompt, grade_response
    from .capability_metrics import tool_selection_entropy

    profile, selection = next_task(loop)
    domain = selection["domain"]
    task_idx = selection["task_idx"]

    prompt = build_prompt(domain, task_idx)
    response, tokens = model_generate(prompt)
    flops = tokens * 2 * estimate_params_b * 1e9

    graded = grade_response(domain, task_idx, response)

    tool_entropy = 0.0
    if domain.startswith("tool-"):
        try:
            from il_toolcalling_core.engine import parse_tool_calls
            tool_entropy = tool_selection_entropy(parse_tool_calls(response))
        except Exception:
            pass

    key = RolloutKey(domain=domain, task_idx=task_idx, taskset_id=selection.get("taskset_id", ""))
    FrontierSampler(profile).observe(
        key,
        {"score": graded.score, "correctness": graded.correctness},
        tokens=tokens, flops=flops, tool_entropy=tool_entropy,
    )

    loop.iterations_done += 1
    loop.score_history.append(graded.score)
    loop.best_score = max(loop.best_score, graded.score)
    loop.rollouts_on_focus = getattr(loop, "rollouts_on_focus", 0) + 1
    return {
        "selection": selection,
        "graded": {
            "score": graded.score,
            "correctness": graded.correctness,
            "reasoning_quality": graded.reasoning_quality,
        },
        "tokens": tokens,
        "tool_entropy": tool_entropy,
    }


__all__ = [
    "RsiLoop", "RsiLoopStore", "LOOP_KINDS", "LOOP_TEMPLATES", "runnability",
    "next_task", "plan_next_composition", "run_frontier_iteration",
]