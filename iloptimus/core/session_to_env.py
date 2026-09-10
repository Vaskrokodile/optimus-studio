"""Session -> environment specification (idea #24).

Extracts a ``FailureSpec`` from a coding/chat/tool-calling session trajectory:
the ordered skill path, the stuck skill, the missing skill, and the error.
The spec is written to ``~/.iloptimus/orchestrator/<loop_id>/failure_specs.jsonl``
and fed to the adversarial environment generator.

See ``RESEARCH_meta_architecture.md`` Section 5.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .grader import GradedResult
from .storage import atomic_write_json, orchestrator_dir

try:
    from il_toolcalling_core.engine import parse_tool_calls
except Exception:  # pragma: no cover - engine import is environment-dependent
    parse_tool_calls = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# FailureSpec
# ---------------------------------------------------------------------------


@dataclass
class FailureSpec:
    session_id: str
    domain: str
    skill_path: list[str]
    stuck_skill: str
    missing_skill: str | None
    error: str
    difficulty: str
    grade: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Missing-skill inference
# ---------------------------------------------------------------------------

# Maps (stuck_skill_substring, error_substring) -> a skill that would unblock it.
_MISSING_SKILL_HINTS: list[tuple[str, str, str]] = [
    ("restart_server", "503", "tool_call:check_load"),
    ("restart_server", "load", "tool_call:check_load"),
    ("write_file", "permission", "tool_call:check_permissions"),
    ("write_file", "locked", "tool_call:acquire_lock"),
    ("query", "syntax", "tool_call:validate_sql"),
    ("query", "no such table", "tool_call:list_tables"),
    ("deploy", "rollback", "tool_call:rollback"),
    ("deploy", "failed", "tool_call:preflight_check"),
    ("restart", "503", "tool_call:check_load"),
    ("", "timeout", "tool_call:retry_with_backoff"),
    ("", "rate limit", "tool_call:retry_with_backoff"),
]


def infer_missing_skill(error: str, stuck_skill: str) -> str | None:
    """Heuristic: map an error/stuck-skill pair to the skill that would unblock it."""
    err = (error or "").lower()
    stuck = (stuck_skill or "").lower()
    for stuck_sub, err_sub, missing in _MISSING_SKILL_HINTS:
        if stuck_sub and stuck_sub not in stuck:
            continue
        if err_sub and err_sub not in err:
            continue
        return missing
    # Generic fallback: if the stuck skill is a tool call, suggest "diagnose".
    if stuck.startswith("tool_call:"):
        base = stuck.split(":", 1)[1]
        return f"tool_call:diagnose_{base}"
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_skill_path(session: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (skill_path, calls) from a session.

    Tool-calling sessions are parsed via ``parse_tool_calls``; coding/chat
    sessions fall back to ``<action>`` tags or a single ``final_answer`` step.
    """
    response = session.get("response", "") or ""
    calls: list[dict[str, Any]] = []
    if parse_tool_calls is not None:
        try:
            calls = parse_tool_calls(response) or []
        except Exception:
            calls = []
    if calls:
        skill_path = [f"tool_call:{c.get('name', 'unknown')}" for c in calls]
        return skill_path, calls

    # Fallback: <action>...</action> tags (stateful environments).
    tagged = re.findall(r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE)
    if tagged:
        return [f"action:{a.strip()}" for a in tagged], [{"name": a.strip()} for a in tagged]

    # Coding/chat: a single final-answer step.
    return ["final_answer"], [{"name": "final_answer"}]


def _find_stuck_index(calls: list[dict[str, Any]], grade: GradedResult) -> int:
    """First explicit error, first repeated identical call, or the last call."""
    for i, call in enumerate(calls):
        if call.get("error"):
            return i
        if i > 0 and call == calls[i - 1]:
            return i
    if not bool(getattr(grade, "correctness", 0.0) >= 1.0):
        return max(0, len(calls) - 1)
    return 0


def extract_failure_spec(session: dict[str, Any], grade: GradedResult) -> FailureSpec:
    """Build a ``FailureSpec`` from a graded session trajectory."""
    skill_path, calls = _extract_skill_path(session)
    stuck_idx = _find_stuck_index(calls, grade)
    stuck_skill = skill_path[stuck_idx] if stuck_idx < len(skill_path) else "final_answer"

    errors = session.get("errors", []) or []
    error = errors[-1] if errors else (
        str(grade.info)[:200] if grade.info else "no_explicit_error"
    )
    missing_skill = infer_missing_skill(error, stuck_skill)

    # Difficulty bucket from the graded score.
    score = float(getattr(grade, "score", 0.0) or 0.0)
    if score < 0.15:
        difficulty = "hard"
    elif score < 0.5:
        difficulty = "medium"
    else:
        difficulty = "easy"

    return FailureSpec(
        session_id=str(session.get("id", session.get("session_id", "unknown"))),
        domain=str(session.get("domain", "unknown")),
        skill_path=skill_path[: stuck_idx + 1],
        stuck_skill=stuck_skill,
        missing_skill=missing_skill,
        error=error,
        difficulty=difficulty,
        grade={
            "score": score,
            "correctness": float(getattr(grade, "correctness", 0.0) or 0.0),
            "reasoning_quality": float(getattr(grade, "reasoning_quality", 0.0) or 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def failure_specs_path(loop_id: str) -> Path:
    return orchestrator_dir() / loop_id / "failure_specs.jsonl"


def append_failure_spec(loop_id: str, spec: FailureSpec) -> Path:
    path = failure_specs_path(loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(spec.public(), ensure_ascii=False) + "\n")
    return path


def load_failure_specs(loop_id: str, limit: int = 50) -> list[FailureSpec]:
    path = failure_specs_path(loop_id)
    if not path.exists():
        return []
    out: list[FailureSpec] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            out.append(FailureSpec(**data))
        except Exception:
            continue
    return out


__all__ = [
    "FailureSpec",
    "infer_missing_skill",
    "extract_failure_spec",
    "append_failure_spec",
    "load_failure_specs",
    "failure_specs_path",
]
