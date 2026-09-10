"""tool-fs-navigator-v1 â€” explore-before-edit filesystem tool calls (verifiers taskset)."""

from __future__ import annotations

import verifiers.v1 as vf

INSTRUCTION = (
    "You are a tool-calling agent operating a simulated environment. Make tool "
    "calls inside <tool>{\"name\": ..., \"args\": {...}}</tool> blocks. Choose the "
    "RIGHT tools with the RIGHT arguments, in the RIGHT order, using as few calls "
    "as possible â€” never call distractor tools. After the calls, give your final "
    "answer inside <answer>...</answer>.\n\n"
)


class ILToolFSTaskset(vf.Taskset):
    pass


__all__ = ["ILToolFSTaskset"]
