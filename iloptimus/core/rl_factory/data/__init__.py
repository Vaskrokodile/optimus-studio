"""
Data collection pipeline for frontier model sessions.

This module provides utilities to:
  1. Download session datasets from HuggingFace
  2. Parse sessions into structured traces
  3. Extract anti-patterns and inefficiency patterns
  4. Convert sessions into TrajectoryDoctor problems
  5. Generate statistics on reasoning waste

Supported dataset formats:
  - HuggingFace datasets (via `datasets` library, optional)
  - Local JSONL files
  - Raw conversation logs

The pipeline is designed to be lazy and fault-tolerant: if a dataset
can't be loaded, it skips gracefully and continues with the next.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SessionTrace:
    """
    A single reasoning trace extracted from a frontier model session.

    Attributes:
        source: Dataset name or file path.
        model: Model name (e.g., "kimi-k3", "opus-5").
        prompt: The user's prompt / problem statement.
        response: The model's full response (including reasoning).
        reasoning: Just the reasoning/thinking portion (if separable).
        answer: Just the final answer (if extractable).
        token_count: Estimated token count of the response.
        metadata: Additional fields from the source dataset.
    """
    source: str = ""
    model: str = ""
    prompt: str = ""
    response: str = ""
    reasoning: str = ""
    answer: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WasteAnalysis:
    """
    Analysis of reasoning waste in a trace.
    """
    trace_id: str = ""
    total_tokens: int = 0
    anti_pattern_count: int = 0
    anti_pattern_types: dict[str, int] = field(default_factory=dict)
    backtracking_count: int = 0
    buzzword_count: int = 0
    filler_count: int = 0
    redundancy_score: float = 0.0  # 0 = no waste, 1 = highly wasteful
    estimated_compressible_tokens: int = 0


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Curated list of HuggingFace datasets with frontier model sessions.
# These are the datasets identified by our research as most useful for
# extracting reasoning inefficiency patterns.
DATASET_REGISTRY = {
    # Fable 5 traces
    "fable5_coding": {
        "hf_id": "greghavens/fable-5-coding-and-debugging-traces",
        "model": "fable-5",
        "format": "parquet",
        "description": "Real agentic coding traces from Claude Fable 5",
    },
    "fable5_sft": {
        "hf_id": "kelexine/fable-5-sft-traces",
        "model": "fable-5",
        "format": "parquet",
        "description": "Cleaned SFT traces from Fable 5",
    },
    # Kimi K3 traces
    "kimi_k3_coding": {
        "hf_id": "greghavens/kimi-k3-coding-and-debugging-traces",
        "model": "kimi-k3",
        "format": "parquet",
        "description": "Real Kimi K3 coding sessions",
    },
    "kimi_k3_aime": {
        "hf_id": "bevangelista/AIME_2026_Kimi_K3",
        "model": "kimi-k3",
        "format": "parquet",
        "description": "AIME 2026 math problems with Kimi K3 reasoning traces",
    },
    # Opus traces
    "opus_reasoning": {
        "hf_id": "Gryphe/Opus-4.6-Reasoning-24k",
        "model": "opus-5",
        "format": "parquet",
        "description": "Opus 4.6 reasoning traces with reasoning_content",
    },
    # Qwen traces
    "qwen3_distillation": {
        "hf_id": "bunker-core/qwen3.8-max-distillation-50k",
        "model": "qwen-3.8",
        "format": "parquet",
        "description": "Teacher traces from Qwen3.8-max",
    },
    # DeepSeek traces
    "deepseek_r1_math": {
        "hf_id": "open-r1/OpenR1-Math-220k",
        "model": "deepseek-r1",
        "format": "parquet",
        "description": "220k math problems with DeepSeek R1 reasoning traces",
    },
    # Aggregated
    "frontier_merged": {
        "hf_id": "SultanR/frontier-reasoning-traces-sft",
        "model": "mixed",
        "format": "parquet",
        "description": "Merged SFT dataset from 11 frontier-model trace datasets",
    },
    # Reasoning efficiency analysis
    "thinktank_labels": {
        "hf_id": "vanthienha199/thinktank-reasoning-labels",
        "model": "mixed",
        "format": "parquet",
        "description": "Human-labeled reasoning steps as useful/wasteful",
    },
    # Backtracking analysis
    "reasoningflow": {
        "hf_id": "jinulee-v/reasoningflow",
        "model": "mixed",
        "format": "parquet",
        "description": "DAG-annotated reasoning traces with backtracking info",
    },
    # Verification (incorrect CoT)
    "cot_verification": {
        "hf_id": "Zigeng/CoT-Verification-340k",
        "model": "mixed",
        "format": "parquet",
        "description": "160k correct + 190k incorrect CoT for verifier training",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(name: str, split: str = "train", streaming: bool = True) -> Iterator[dict]:
    """
    Load a dataset from HuggingFace by name (from the registry) or by HF ID.

    Args:
        name: Registry key or HuggingFace dataset ID.
        split: Dataset split to load.
        streaming: If True, stream rows one at a time (memory-efficient).

    Yields:
        Raw dataset rows as dicts.

    Raises:
        ImportError: If the `datasets` library is not installed.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required to load HuggingFace datasets. "
            "Install with: pip install datasets"
        )

    # Resolve name to HF ID
    if name in DATASET_REGISTRY:
        hf_id = DATASET_REGISTRY[name]["hf_id"]
    else:
        hf_id = name

    ds = hf_load_dataset(hf_id, split=split, streaming=streaming)
    yield from ds


def load_jsonl(path: str) -> Iterator[dict]:
    """Load rows from a local JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------


def extract_trace(row: dict, source: str = "", model: str = "") -> SessionTrace:
    """
    Extract a SessionTrace from a raw dataset row.

    Handles various common formats:
      - {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
      - {"prompt": ..., "response": ...}
      - {"question": ..., "answer": ...}
      - {"input": ..., "output": ...}
      - {"reasoning_content": ..., "content": ...}
    """
    trace = SessionTrace(source=source, model=model)

    # Try messages format
    if "messages" in row:
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "human"):
                trace.prompt = content
            elif role in ("assistant", "model", "gpt"):
                trace.response = content
                # Check for reasoning_content
                if "reasoning_content" in msg:
                    trace.reasoning = msg["reasoning_content"]

    # Try prompt/response format
    elif "prompt" in row and "response" in row:
        trace.prompt = row["prompt"]
        trace.response = row["response"]

    # Try question/answer format
    elif "question" in row and "answer" in row:
        trace.prompt = row["question"]
        trace.response = row["answer"]

    # Try input/output format
    elif "input" in row and "output" in row:
        trace.prompt = row["input"]
        trace.response = row["output"]

    # Try reasoning_content + content
    elif "reasoning_content" in row:
        trace.reasoning = row["reasoning_content"]
        trace.response = row.get("content", row.get("response", row.get("output", "")))
        trace.prompt = row.get("prompt", row.get("question", row.get("input", "")))

    # Extract answer from response (look for \boxed{} or final answer patterns)
    trace.answer = _extract_final_answer(trace.response)

    # If no reasoning was separated, try to extract it from the response
    if not trace.reasoning and trace.response:
        trace.reasoning = _extract_reasoning(trace.response)

    # Estimate token count
    from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
    trace.token_count = estimate_tokens(trace.response)

    # Copy metadata
    trace.metadata = {k: v for k, v in row.items()
                      if k not in ("messages", "prompt", "response", "question",
                                   "answer", "input", "output", "content",
                                   "reasoning_content")}

    return trace


def _extract_final_answer(response: str) -> str:
    """Try to extract the final answer from a response."""
    # Look for \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip()

    # Look for "The answer is ..." or "Answer: ..."
    match = re.search(r"(?:the\s+answer\s+is|answer:)\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for "CONCLUSION: ..."
    match = re.search(r"CONCLUSION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


def _extract_reasoning(response: str) -> str:
    """Try to extract the reasoning portion from a response."""
    # Look for <think>...</think> or <reasoning>...</reasoning>
    match = re.search(r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for "Let me think..." patterns
    match = re.search(r"(let\s+me\s+think.*?)(?:answer:|the\s+answer\s+is|CONCLUSION:|$)", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Waste analysis
# ---------------------------------------------------------------------------


def analyze_trace_waste(trace: SessionTrace) -> WasteAnalysis:
    """
    Analyze a trace for reasoning waste patterns.

    Uses the AntiPatternDetector to identify backtracking, buzzwords, filler,
    and other inefficiencies. Estimates how many tokens could be saved.
    """
    from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
    from iloptimus.core.rl_factory.core.token_budget import estimate_tokens

    detector = AntiPatternDetector()
    text = trace.reasoning if trace.reasoning else trace.response
    report = detector.analyze(text, prompt=trace.prompt)

    analysis = WasteAnalysis(
        trace_id=f"{trace.source}_{hash(trace.prompt) % 100000}",
        total_tokens=trace.token_count,
        anti_pattern_count=report.total_hits,
        anti_pattern_types=report.counts_by_type,
        backtracking_count=report.counts_by_type.get("backtrack", 0)
                          + report.counts_by_type.get("empty_correction", 0),
        buzzword_count=report.counts_by_type.get("buzzword", 0),
        filler_count=report.counts_by_type.get("filler", 0),
    )

    # Estimate compressible tokens
    # Each anti-pattern hit represents ~10-50 tokens of waste
    estimated_waste_tokens = sum(
        report.counts_by_type.get(t, 0) * avg_tokens
        for t, avg_tokens in [
            ("backtrack", 40), ("buzzword", 15), ("filler", 20),
            ("restate", 50), ("self_congratulate", 25),
            ("over_qualify", 30), ("repeat_step", 35),
            ("vague_reference", 10), ("empty_correction", 15),
        ]
    )
    analysis.estimated_compressible_tokens = min(estimated_waste_tokens, trace.token_count)

    # Redundancy score: fraction of tokens that are waste
    if trace.token_count > 0:
        analysis.redundancy_score = min(1.0, estimated_waste_tokens / trace.token_count)

    return analysis


# ---------------------------------------------------------------------------
# Problem conversion
# ---------------------------------------------------------------------------


def trace_to_trajectory_doctor_problem(trace: SessionTrace, analysis: WasteAnalysis) -> Optional[dict]:
    """
    Convert a wasteful trace into a TrajectoryDoctor problem.

    If the trace has significant waste, split it into steps and mark the
    wasteful steps as the "error" steps.

    Returns:
        A dict with problem data, or None if the trace is not suitable.
    """
    if analysis.anti_pattern_count < 2:
        return None  # not enough waste to be interesting

    text = trace.reasoning if trace.reasoning else trace.response
    if len(text) < 100:
        return None  # too short to analyze

    # Split into steps (rough sentence-based splitting)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 3:
        return None

    # Find the first sentence with anti-patterns
    from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
    detector = AntiPatternDetector()

    error_step = -1
    for i, sentence in enumerate(sentences):
        report = detector.analyze(sentence, prompt=trace.prompt)
        if report.total_hits > 0:
            error_step = i
            break

    if error_step == -1:
        return None

    # Build the problem
    steps = [f"Step {i+1}: {s}" for i, s in enumerate(sentences[:10])]  # cap at 10 steps

    return {
        "type": "trajectory_doctor",
        "trace_type": "waste",
        "steps": steps,
        "error_step": error_step,
        "source_model": trace.model,
        "source_dataset": trace.source,
        "waste_analysis": {
            "total_tokens": analysis.total_tokens,
            "compressible_tokens": analysis.estimated_compressible_tokens,
            "redundancy_score": analysis.redundancy_score,
        },
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_dataset(
    name: str,
    max_traces: int = 1000,
    min_waste: int = 2,
) -> list[tuple[SessionTrace, WasteAnalysis]]:
    """
    Process a dataset: load traces, analyze waste, return interesting ones.

    Args:
        name: Dataset registry key or HF ID.
        max_traces: Maximum number of traces to process.
        min_waste: Minimum anti-pattern count to keep a trace.

    Returns:
        List of (trace, analysis) pairs for traces with significant waste.
    """
    model = DATASET_REGISTRY.get(name, {}).get("model", "unknown")
    results = []

    try:
        for i, row in enumerate(load_dataset(name)):
            if i >= max_traces:
                break

            trace = extract_trace(row, source=name, model=model)
            if not trace.response:
                continue

            analysis = analyze_trace_waste(trace)

            if analysis.anti_pattern_count >= min_waste:
                results.append((trace, analysis))

    except Exception as e:
        print(f"Warning: Could not process dataset {name}: {e}")

    return results


def generate_problems_from_datasets(
    dataset_names: Optional[list[str]] = None,
    max_per_dataset: int = 500,
    min_waste: int = 2,
) -> list[dict]:
    """
    Generate TrajectoryDoctor problems from real frontier model sessions.

    Args:
        dataset_names: List of dataset registry keys. If None, uses all.
        max_per_dataset: Max traces to process per dataset.
        min_waste: Minimum waste to include a trace.

    Returns:
        List of problem dicts ready for TrajectoryDoctorEnv.
    """
    if dataset_names is None:
        dataset_names = list(DATASET_REGISTRY.keys())

    problems = []
    for name in dataset_names:
        traces = process_dataset(name, max_traces=max_per_dataset, min_waste=min_waste)
        for trace, analysis in traces:
            problem = trace_to_trajectory_doctor_problem(trace, analysis)
            if problem:
                problems.append(problem)

    return problems


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_waste_statistics(traces: list[tuple[SessionTrace, WasteAnalysis]]) -> dict:
    """Compute aggregate waste statistics across a set of traces."""
    if not traces:
        return {}

    total_tokens = sum(a.total_tokens for _, a in traces)
    total_waste = sum(a.estimated_compressible_tokens for _, a in traces)
    total_backtracking = sum(a.backtracking_count for _, a in traces)
    total_buzzwords = sum(a.buzzword_count for _, a in traces)
    total_filler = sum(a.filler_count for _, a in traces)

    # Per-model breakdown
    by_model: dict[str, dict] = {}
    for trace, analysis in traces:
        model = trace.model or "unknown"
        if model not in by_model:
            by_model[model] = {
                "count": 0, "total_tokens": 0, "waste_tokens": 0,
                "backtracking": 0, "buzzwords": 0, "filler": 0,
            }
        by_model[model]["count"] += 1
        by_model[model]["total_tokens"] += analysis.total_tokens
        by_model[model]["waste_tokens"] += analysis.estimated_compressible_tokens
        by_model[model]["backtracking"] += analysis.backtracking_count
        by_model[model]["buzzwords"] += analysis.buzzword_count
        by_model[model]["filler"] += analysis.filler_count

    # Compute averages
    for model_stats in by_model.values():
        n = model_stats["count"]
        if n > 0:
            model_stats["avg_tokens"] = model_stats["total_tokens"] / n
            model_stats["avg_waste"] = model_stats["waste_tokens"] / n
            model_stats["waste_fraction"] = model_stats["waste_tokens"] / max(1, model_stats["total_tokens"])

    return {
        "total_traces": len(traces),
        "total_tokens": total_tokens,
        "total_waste_tokens": total_waste,
        "waste_fraction": total_waste / max(1, total_tokens),
        "total_backtracking": total_backtracking,
        "total_buzzwords": total_buzzwords,
        "total_filler": total_filler,
        "by_model": by_model,
    }
