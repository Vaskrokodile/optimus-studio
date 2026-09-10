"""Shared tool-calling task dataclass + IL scoring glue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .engine import ToolSpec, score_trajectory


@dataclass
class ToolCallingTask:
    idx: int
    name: str
    spec: str
    tools: dict[str, ToolSpec]
    init: dict
    goal: Callable[[dict], tuple[bool, str]]
    expert: list[tuple[str, dict]]
    verify_answer: Callable[[str, object], tuple[bool, str]]
    expected_concepts: list[str] = field(default_factory=list)
    token_budget: int = 700
    difficulty: str = "hard"
    scenario: str = ""
    expert_answer: str = "done"

    def verify_answer_wrapper(self, answer: str, trace) -> tuple[bool, str]:
        return self.verify_answer(answer, trace)


def answer_contains(*expected: str):
    """Verifier factory: answer must contain all expected substrings (case-insensitive)."""

    def verify(answer: str, trace) -> tuple[bool, str]:
        low = answer.lower()
        missing = [e for e in expected if e.lower() not in low]
        if not expected:
            return True, "no answer constraint"
        if not missing:
            return True, "answer contains expected content"
        return False, f"answer missing: {missing}"

    return verify


def score_response(task: ToolCallingTask, response: str) -> tuple[float, dict]:
    return score_trajectory(task, response)


__all__ = ["ToolCallingTask", "score_response"]
