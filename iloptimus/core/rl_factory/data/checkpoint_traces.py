"""
Checkpoint extraction from real reasoning traces.

Takes the curated frontier model traces and splits them into logical
"checkpoints" — segments of reasoning that represent distinct sub-goals.
Each checkpoint has:
  - name: a short label
  - content: the reasoning text for that segment
  - success_criteria: what must be true for this checkpoint to pass
  - audit_keywords: keywords that should appear in a correct audit

This module is used by the checkpoint environments (21-25) to generate
problems from real trace data, with synthetic fallback.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Optional


_TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
_trace_cache: Optional[list[dict]] = None


def _load_traces() -> list[dict]:
    """Load curated traces (prefer curated_all.json)."""
    global _trace_cache
    if _trace_cache is not None:
        return _trace_cache

    traces = []
    traces_dir = os.path.normpath(_TRACES_DIR)
    if not os.path.isdir(traces_dir):
        _trace_cache = []
        return _trace_cache

    curated = os.path.join(traces_dir, "curated_all.json")
    if os.path.isfile(curated):
        try:
            with open(curated, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    _trace_cache = data
                    return _trace_cache
        except (json.JSONDecodeError, OSError):
            pass

    for fname in os.listdir(traces_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(traces_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    traces.extend(data)
        except (json.JSONDecodeError, OSError):
            continue

    _trace_cache = traces
    return traces


def _split_into_checkpoints(text: str, target_count: int = 3) -> list[dict]:
    """
    Split a reasoning trace into logical checkpoints.

    Uses sentence-boundary splitting, then groups sentences into
    approximately target_count segments of roughly equal length.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < target_count:
        # Not enough sentences — each sentence is a checkpoint
        target_count = max(1, len(sentences))

    # Group sentences into target_count segments
    per_group = max(1, len(sentences) // target_count)
    checkpoints = []
    for i in range(0, len(sentences), per_group):
        group = sentences[i:i + per_group]
        if group:
            checkpoints.append({
                "name": f"Checkpoint {len(checkpoints) + 1}",
                "content": " ".join(group),
                "index": len(checkpoints),
            })

    # Merge any tiny trailing group
    if len(checkpoints) > target_count and len(checkpoints[-1]["content"]) < 50:
        checkpoints[-2]["content"] += " " + checkpoints[-1]["content"]
        checkpoints.pop()

    return checkpoints


def _generate_success_criteria(checkpoint: dict, rng: random.Random) -> str:
    """Generate a success criteria string for a checkpoint."""
    content = checkpoint["content"]
    # Extract key terms
    words = re.findall(r"\b[a-z]{4,}\b", content.lower())
    common = {"that", "this", "with", "from", "have", "they", "will", "been",
              "were", "which", "their", "would", "should", "could", "about"}
    keywords = [w for w in words if w not in common][:3]

    if keywords:
        return f"Must address: {', '.join(keywords)}"
    return "Must produce a correct intermediate result"


def _generate_audit_keywords(checkpoint: dict, rng: random.Random) -> list[str]:
    """Generate keywords that a correct audit should mention."""
    content = checkpoint["content"].lower()
    words = re.findall(r"\b[a-z]{4,}\b", content)
    common = {"that", "this", "with", "from", "have", "they", "will", "been",
              "were", "which", "their", "would", "should", "could", "about"}
    keywords = [w for w in words if w not in common]
    return keywords[:5] if keywords else ["correct", "step", "result"]


def extract_checkpoint_problem(rng: random.Random) -> Optional[dict]:
    """
    Extract a checkpoint-annotated problem from a real trace.

    Returns a dict with:
      - prompt: the original task
      - checkpoints: list of checkpoint dicts
      - trace_type: "coding" / "math" / "reasoning"
      - source: dataset name
      - model: model name

    Returns None if no suitable traces are available.
    """
    traces = _load_traces()
    if not traces:
        return None

    # Filter for traces with enough reasoning content
    suitable = []
    for t in traces:
        text = t.get("reasoning", "") or t.get("response", "")
        if len(text) > 150:
            suitable.append(t)

    if not suitable:
        return None

    trace = rng.choice(suitable)
    text = trace.get("reasoning", "") or trace.get("response", "")
    prompt = trace.get("prompt", "")
    source = trace.get("source", "unknown")
    model = trace.get("model", "unknown")
    trace_type = trace.get("trace_type", "reasoning")

    # Split into 3-5 checkpoints
    num_checkpoints = rng.randint(3, 5)
    checkpoints = _split_into_checkpoints(text, target_count=num_checkpoints)

    if len(checkpoints) < 2:
        return None

    # Annotate each checkpoint
    for cp in checkpoints:
        cp["success_criteria"] = _generate_success_criteria(cp, rng)
        cp["audit_keywords"] = _generate_audit_keywords(cp, rng)
        # Mark some checkpoints as "passed" and some as "failed" for audit envs
        cp["status"] = "passed"  # default: all passed in the original trace

    return {
        "prompt": prompt,
        "checkpoints": checkpoints,
        "trace_type": trace_type,
        "source": source,
        "model": model,
        "original_reasoning": text,
    }


# ---------------------------------------------------------------------------
# Synthetic checkpoint problems (fallback)
# ---------------------------------------------------------------------------


_SYNTHETIC_CHECKPOINT_PROBLEMS = [
    {
        "prompt": "Write a function that reads a CSV file, filters rows where age > 30, and returns the filtered list.",
        "checkpoints": [
            {
                "name": "Read CSV",
                "content": "Open and read the CSV file into a list of rows.",
                "success_criteria": "Must read the file and parse rows",
                "audit_keywords": ["read", "csv", "file", "rows", "open"],
                "status": "passed",
            },
            {
                "name": "Filter rows",
                "content": "Filter rows where the age column value is greater than 30.",
                "success_criteria": "Must filter on age > 30",
                "audit_keywords": ["filter", "age", "30", "column", "condition"],
                "status": "passed",
            },
            {
                "name": "Return result",
                "content": "Return the filtered list of rows.",
                "success_criteria": "Must return the filtered list",
                "audit_keywords": ["return", "filtered", "list", "result"],
                "status": "passed",
            },
        ],
        "trace_type": "coding",
    },
    {
        "prompt": "Solve: A train travels 240 km in 4 hours. What is its speed? Then calculate how long it takes to travel 600 km.",
        "checkpoints": [
            {
                "name": "Calculate speed",
                "content": "Speed = distance / time = 240 / 4 = 60 km/h.",
                "success_criteria": "Must compute speed = 60 km/h",
                "audit_keywords": ["speed", "240", "4", "60", "distance", "time"],
                "status": "passed",
            },
            {
                "name": "Calculate time for 600 km",
                "content": "Time = distance / speed = 600 / 60 = 10 hours.",
                "success_criteria": "Must compute time = 10 hours",
                "audit_keywords": ["time", "600", "60", "10", "hours", "distance"],
                "status": "passed",
            },
        ],
        "trace_type": "math",
    },
    {
        "prompt": "Debug a function that crashes with TypeError when processing user input. Fix the bug and add error handling.",
        "checkpoints": [
            {
                "name": "Identify the bug",
                "content": "The TypeError occurs because the function tries to concatenate a string with an integer.",
                "success_criteria": "Must identify the type mismatch",
                "audit_keywords": ["typeerror", "string", "integer", "concatenate", "mismatch"],
                "status": "passed",
            },
            {
                "name": "Fix the bug",
                "content": "Convert the integer to a string before concatenation using str().",
                "success_criteria": "Must apply the str() conversion fix",
                "audit_keywords": ["str", "convert", "fix", "concatenation", "cast"],
                "status": "passed",
            },
            {
                "name": "Add error handling",
                "content": "Wrap the processing in a try-except block to handle future type errors gracefully.",
                "success_criteria": "Must add try-except error handling",
                "audit_keywords": ["try", "except", "error", "handle", "catch"],
                "status": "passed",
            },
        ],
        "trace_type": "coding",
    },
    {
        "prompt": "Prove that the sum of two even numbers is even.",
        "checkpoints": [
            {
                "name": "Define even numbers",
                "content": "An even number can be written as 2k for some integer k. Let a = 2m and b = 2n.",
                "success_criteria": "Must express both numbers as 2k form",
                "audit_keywords": ["even", "2k", "2m", "2n", "integer", "define"],
                "status": "passed",
            },
            {
                "name": "Compute the sum",
                "content": "a + b = 2m + 2n = 2(m + n).",
                "success_criteria": "Must factor out 2 from the sum",
                "audit_keywords": ["sum", "2m", "2n", "2(m+n)", "factor", "add"],
                "status": "passed",
            },
            {
                "name": "Conclude",
                "content": "Since m + n is an integer, 2(m + n) is even. Therefore a + b is even.",
                "success_criteria": "Must conclude the sum is even",
                "audit_keywords": ["even", "conclude", "integer", "therefore", "result"],
                "status": "passed",
            },
        ],
        "trace_type": "math",
    },
    {
        "prompt": "Implement a REST API endpoint for creating a new user with validation and error handling.",
        "checkpoints": [
            {
                "name": "Define the endpoint",
                "content": "Create a POST /users endpoint that accepts JSON body with name and email.",
                "success_criteria": "Must define POST /users with JSON body",
                "audit_keywords": ["post", "users", "endpoint", "json", "route"],
                "status": "passed",
            },
            {
                "name": "Add validation",
                "content": "Validate that name is non-empty and email matches a regex pattern.",
                "success_criteria": "Must validate name and email",
                "audit_keywords": ["validate", "name", "email", "regex", "empty"],
                "status": "passed",
            },
            {
                "name": "Add error handling",
                "content": "Return 400 for validation errors and 500 for database errors.",
                "success_criteria": "Must return appropriate HTTP error codes",
                "audit_keywords": ["400", "500", "error", "validation", "database"],
                "status": "passed",
            },
            {
                "name": "Save to database",
                "content": "Insert the validated user into the database and return 201 with the created user.",
                "success_criteria": "Must save to DB and return 201",
                "audit_keywords": ["database", "insert", "201", "save", "created"],
                "status": "passed",
            },
        ],
        "trace_type": "coding",
    },
    {
        "prompt": "Analyze the time complexity of merge sort and explain why it's O(n log n).",
        "checkpoints": [
            {
                "name": "Describe the algorithm",
                "content": "Merge sort recursively splits the array in half, sorts each half, then merges them.",
                "success_criteria": "Must describe split and merge steps",
                "audit_keywords": ["split", "merge", "recursive", "half", "sort"],
                "status": "passed",
            },
            {
                "name": "Analyze the recurrence",
                "content": "T(n) = 2T(n/2) + O(n). The recursion tree has log n levels, each doing O(n) work.",
                "success_criteria": "Must derive the recurrence T(n) = 2T(n/2) + O(n)",
                "audit_keywords": ["recurrence", "2t(n/2)", "log", "levels", "tree"],
                "status": "passed",
            },
            {
                "name": "Conclude complexity",
                "content": "Total work = O(n) * O(log n) = O(n log n).",
                "success_criteria": "Must conclude O(n log n)",
                "audit_keywords": ["n log n", "conclude", "total", "work", "complexity"],
                "status": "passed",
            },
        ],
        "trace_type": "reasoning",
    },
]


def get_synthetic_checkpoint_problem(rng: random.Random) -> dict:
    """Get a synthetic checkpoint problem (fallback when no real traces)."""
    template = rng.choice(_SYNTHETIC_CHECKPOINT_PROBLEMS)
    # Deep copy and add some randomization
    import copy
    problem = copy.deepcopy(template)
    problem["source"] = "synthetic"
    problem["model"] = "synthetic"
    problem["original_reasoning"] = "\n".join(cp["content"] for cp in problem["checkpoints"])
    return problem


def get_checkpoint_problem(rng: random.Random) -> dict:
    """
    Get a checkpoint problem — 70% from real traces, 30% synthetic.
    Falls back to 100% synthetic if no real traces available.
    """
    if rng.random() < 0.7:
        real = extract_checkpoint_problem(rng)
        if real is not None:
            return real
    return get_synthetic_checkpoint_problem(rng)
