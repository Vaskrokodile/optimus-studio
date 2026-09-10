"""Tool-calling environment engine for IL tasksets.

Deterministic, dependency-free simulation of tool worlds for teaching models
HOW to call tools: right tool, right arguments, right order, right efficiency.

A task defines:
    tools       — catalog of {name, description, params, fn(state, args)->obs}
                  including deliberate DISTRACTOR tools that must not be called
    init        — initial world state dict
    goal(state) -> (bool, str)   terminal success check
    expert      — the optimal trajectory [(tool, args), ...]
    max_calls   — call budget (efficiency shaping)
    faults      — optional injected failures the model must recover from

The model responds with tool calls inside <tool>...</tool> blocks:

    <tool>{"name": "read_file", "args": {"path": "/etc/app.conf"}}</tool>

The engine replays every call against the simulator, feeds observations back
conceptually (single-shot grading: the model must anticipate outputs), and
scores the trajectory:

    final = correctness * (0.55 + 0.25 * selection + 0.20 * efficiency)

    correctness  — goal reached AND final answer matches (hard gate)
    selection    — precision/recall of tool choice vs distractors & misuse
    efficiency   — call-count vs optimal + no redundant repeats
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]  # param -> type hint
    fn: Callable[[dict, dict], tuple[Any, str]]  # (state, args) -> (result, observation)
    distractor: bool = False  # true = must never be called
    destructive: bool = False  # true = requires prior read/lookup (tau-bench pattern)


@dataclass
class ToolCallTrace:
    calls: list[dict] = field(default_factory=list)          # parsed calls
    observations: list[str] = field(default_factory=list)    # simulator outputs
    errors: list[str] = field(default_factory=list)          # malformed / failed calls
    state: dict = field(default_factory=dict)
    goal_reached: bool = False
    goal_info: str = ""
    answer: str = ""
    parse_failures: int = 0


def parse_tool_calls(response: str) -> list[dict]:
    """Extract <tool_call> {...} </tool> calls; tolerate bare JSON lines as fallback."""
    calls: list[dict] = []
    for match in TOOL_RE.finditer(response):
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict) and "name" in payload:
                calls.append(payload)
        except json.JSONDecodeError:
            calls.append({"name": "__parse_error__", "args": {}})
    if not calls:
        # fallback: one JSON object per line with a "name" key
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict) and "name" in payload:
                        calls.append(payload)
                except json.JSONDecodeError:
                    pass
    return calls


def extract_answer(response: str) -> str:
    match = ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def replay(task, response: str) -> ToolCallTrace:
    """Replay parsed tool calls against the task's tool world."""
    trace = ToolCallTrace(state=dict(task.init))
    trace.calls = parse_tool_calls(response)
    trace.answer = extract_answer(response)
    read_done: set[str] = set()

    for call in trace.calls:
        name = str(call.get("name", ""))
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if name == "__parse_error__":
            trace.parse_failures += 1
            trace.errors.append("malformed JSON in <tool_call> block")
            continue
        tool = task.tools.get(name)
        if tool is None:
            trace.errors.append(f"unknown tool: {name}")
            trace.observations.append(f"ERROR: unknown tool '{name}'")
            continue
        missing = [p for p in tool.params if p not in args]
        if missing:
            trace.errors.append(f"{name}: missing args {missing}")
            trace.observations.append(f"ERROR: {name} missing required args: {missing}")
            continue
        try:
            result, observation = tool.fn(trace.state, args)
        except Exception as cause:  # deterministic simulators should not raise
            trace.errors.append(f"{name}: simulator error {cause}")
            continue
        read_done.add(name)
        trace.observations.append(str(observation))

    reached, info = task.goal(trace.state)
    trace.goal_reached = bool(reached)
    trace.goal_info = str(info)
    return trace


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trajectory(task, response: str) -> tuple[float, dict]:
    """Score a response against the task.

    final = correctness * (0.55 + 0.25 * tool_selection + 0.20 * efficiency)

    correctness gate: goal reached in the simulator AND the <tool_call> matches.
    """
    trace = replay(task, response)
    answer = trace.answer

    answer_ok, answer_info = task.verify_answer(answer, trace)
    goal_ok = trace.goal_reached
    correctness = 1.0 if (goal_ok and answer_ok) else 0.0

    # --- tool selection -----------------------------------------------------
    called = [c.get("name", "") for c in trace.calls if c.get("name") != "__parse_error__"]
    optimal = [t for t, _ in task.expert]
    optimal_set = set(optimal)
    called_set = set(called)
    distractor_hits = sum(1 for name in called if task.tools.get(name) and task.tools[name].distractor)
    unknown_hits = sum(1 for name in called if name not in task.tools)
    precision = len(called_set & optimal_set) / len(called_set) if called_set else 0.0
    recall = len(optimal_set & called_set) / len(optimal_set) if optimal_set else 1.0
    selection = max(0.0, 0.5 * precision + 0.5 * recall - 0.25 * distractor_hits - 0.25 * unknown_hits)
    selection = max(0.0, min(1.0, selection))

    # --- efficiency ---------------------------------------------------------
    n = len(called)
    optimal_n = len(optimal)
    repeats = n - len(called_set)
    if n == 0:
        efficiency = 0.0
    else:
        count_score = 1.0 if n <= optimal_n else max(0.0, 1.0 - (n - optimal_n) / max(optimal_n, 1))
        efficiency = max(0.0, count_score - 0.15 * repeats - 0.1 * trace.parse_failures)

    # argument validity: fraction of calls that produced no error
    if trace.calls:
        arg_validity = 1.0 - len(trace.errors) / len(trace.calls)
    else:
        arg_validity = 0.0
    efficiency = max(0.0, min(1.0, efficiency * (0.5 + 0.5 * arg_validity)))

    final = correctness * (0.55 + 0.25 * selection + 0.20 * efficiency)

    breakdown = {
        "correctness": correctness,
        "goal_reached": goal_ok,
        "goal_info": trace.goal_info,
        "answer_ok": answer_ok,
        "answer_info": answer_info,
        "tool_selection": round(selection, 3),
        "efficiency": round(efficiency, 3),
        "calls_made": len(trace.calls),
        "optimal_calls": optimal_n,
        "distractor_hits": distractor_hits,
        "errors": trace.errors[:8],
        "final_score": round(final, 4),
    }
    return final, breakdown


__all__ = [
    "ToolSpec",
    "ToolCallTrace",
    "parse_tool_calls",
    "extract_answer",
    "replay",
    "score_trajectory",
]
