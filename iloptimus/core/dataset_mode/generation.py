"""Batched vLLM generation engine for dataset mode.

Generates training rows in parallel using vLLM's batch inference.  Key features:

- Diversity-aware batching: prompts in the same batch cover diverse semantic
  regions to prevent mode collapse.
- Temperature scheduling: start hot (0.9) for diversity, cool down (0.3) for
  correctness on re-rolls.
- Multi-turn refinement: for hard rows, allow a critique-revise cycle.
- Failure-driven: accepts seed problems from benchmark failures to guide
  generation toward weak areas.
- Curriculum-aware: generates across difficulty levels (easy/medium/hard).
"""

from __future__ import annotations

# Think tags
_TO = chr(60) + "think" + chr(62)
_TC = chr(60) + "/think" + chr(62)

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .prompts import dataset_system_prompt
from .skills import dataset_skills_prompt
from .templates import template_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_REASONING_TOKENS = 4096
DEFAULT_MAX_ANSWER_TOKENS = 512
DEFAULT_TEMP_HOT = 0.9  # initial generation
DEFAULT_TEMP_COOL = 0.3  # re-rolls for correctness
DEFAULT_TOP_P = 0.95
DEFAULT_REROLL_COUNT = 2  # re-rolls for failed verification
DEFAULT_MAX_ROWS = 200
DEFAULT_REFINEMENT_THRESHOLD = 0.2  # fraction of hardest rows to refine

# Difficulty distribution
_DIFFICULTY_SPLIT = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

# Topic seeds for diversity (domain-specific)
_TOPIC_SEEDS: dict[str, list[str]] = {
    "math": [
        "algebra: polynomial roots",
        "combinatorics: counting with restrictions",
        "number theory: modular arithmetic",
        "geometry: circle theorems",
        "probability: conditional probability",
        "inequalities: AM-GM applications",
        "sequences: recurrence relations",
        "trigonometry: identity proofs",
        "calculus: optimization",
        "discrete math: graph coloring",
    ],
    "coding": [
        "arrays: two-pointer technique",
        "graphs: shortest path",
        "dp: knapsack variants",
        "strings: pattern matching",
        "trees: traversal and balancing",
        "greedy: interval scheduling",
        "recursion: divide and conquer",
        "hashing: collision resolution",
        "bit manipulation: XOR properties",
        "data structures: LRU cache",
    ],
    "reasoning": [
        "logic: knights and knaves",
        "deduction: constraint satisfaction",
        "spatial: grid navigation",
        "temporal: scheduling puzzles",
        "causal: fault diagnosis",
        "analogical: pattern completion",
        "set theory: inclusion-exclusion",
        "game theory: nim variants",
        "cryptography: simple ciphers",
        "information: entropy puzzles",
    ],
    "agentic": [
        "debugging: cascading failures",
        "refactoring: API migration",
        "analysis: pipeline tracing",
        "design: system architecture",
        "planning: multi-step workflows",
        "tool use: API composition",
        "code review: finding bugs",
        "data analysis: anomaly detection",
        "optimization: performance tuning",
        "integration: service composition",
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenerationSpec:
    """Specification for a dataset generation run."""

    domain: str = "math"
    count: int = DEFAULT_MAX_ROWS
    batch_size: int = DEFAULT_BATCH_SIZE
    difficulty_split: dict[str, float] = field(default_factory=lambda: dict(_DIFFICULTY_SPLIT))
    seed_problems: list[dict[str, Any]] = field(default_factory=list)
    research_context: str = ""  # summary from continuous research
    corpus_summary: str = ""  # summary of existing corpus for anti-dup
    max_reasoning_tokens: int = DEFAULT_MAX_REASONING_TOKENS
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    reroll_count: int = DEFAULT_REROLL_COUNT
    enable_refinement: bool = True
    refinement_fraction: float = DEFAULT_REFINEMENT_THRESHOLD


@dataclass
class GeneratedRow:
    """A single generated training row (pre-quality-gate)."""

    input: str
    thinking_trace: str
    answer: str
    domain: str
    difficulty: str
    source: str = "synthetic"
    generation_time: float = 0.0
    raw_output: str = ""
    row_hash: str = ""

    def __post_init__(self) -> None:
        if not self.row_hash:
            content = f"{self.input}|{self.thinking_trace}|{self.answer}"
            self.row_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_generation_prompt(
    spec: GenerationSpec,
    topic: str,
    difficulty: str,
    rng: random.Random,
) -> str:
    """Build a single generation prompt for one batch element."""
    system = dataset_system_prompt(spec.domain)
    skills = dataset_skills_prompt()
    templates = template_prompt(spec.domain, count=2, rng=rng)

    parts = [system, "", skills, ""]
    if templates:
        parts.extend([templates, ""])
    if spec.research_context:
        parts.extend([f"Research context:\n{spec.research_context}", ""])
    if spec.corpus_summary:
        parts.extend([f"Existing corpus (do NOT duplicate these):\n{spec.corpus_summary}", ""])

    # Task instruction
    difficulty_hint = {
        "easy": "1-2 steps, straightforward",
        "medium": "3-5 steps, requires some insight",
        "hard": "6+ steps, may require creative insight or casework",
    }.get(difficulty, "3-5 steps")

    parts.append(
        f"Generate ONE {spec.domain} problem at {difficulty} difficulty ({difficulty_hint}).\n"
        f"Topic area: {topic}\n\n"
        f"Output format:\n"
        f"PROBLEM: <the problem statement>\n"
        f"\n{_TO}<your step-by-step reasoning>{_TC}\n"
        f"<the final answer>\n"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


_PROBLEM_RE = re.compile(r"PROBLEM:\s*(.*?)(?=" + _TO + ")", re.DOTALL)
_THINK_RE = re.compile(_TO + r"\s*(.*?)\n" + _TC, re.DOTALL)
_ANSWER_RE = re.compile(_TC + r"\n\s*(.*)", re.DOTALL)


def parse_generation(raw: str) -> GeneratedRow | None:
    """Parse a raw generation output into a GeneratedRow.

    Returns None if the output doesn't contain the required structure.
    """
    if not raw or not raw.strip():
        return None

    problem_match = _PROBLEM_RE.search(raw)
    think_match = _THINK_RE.search(raw)
    answer_match = _ANSWER_RE.search(raw)

    if not problem_match or not think_match:
        return None

    problem = problem_match.group(1).strip()
    thinking = think_match.group(1).strip()
    answer = answer_match.group(1).strip() if answer_match else ""

    if not problem or not thinking:
        return None
    if len(problem) < 20:  # too short to be a real problem
        return None
    if len(thinking) < 50:  # too short to be real reasoning
        return None

    return GeneratedRow(
        input=problem,
        thinking_trace=thinking,
        answer=answer,
        domain="",  # filled by caller
        difficulty="",  # filled by caller
        raw_output=raw,
    )


# ---------------------------------------------------------------------------
# Diversity-aware prompt expansion
# ---------------------------------------------------------------------------


def expand_generation_prompts(
    spec: GenerationSpec,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Expand a spec into a list of (prompt, topic, difficulty) tuples.

    Ensures diversity by:
    - Drawing from diverse topic seeds
    - Splitting across difficulty levels
    - Including seed problems from failures when available
    """
    topics = _TOPIC_SEEDS.get(spec.domain, _TOPIC_SEEDS["math"])
    # Add seed-problem-derived topics
    for seed in spec.seed_problems:
        problem_text = str(seed.get("problem", seed.get("input", "")))
        if problem_text:
            # Extract a short topic hint from the seed problem
            words = problem_text.split()[:6]
            topics.append(" ".join(words))

    prompts: list[tuple[str, str, str]] = []
    difficulties = list(spec.difficulty_split.keys())
    weights = [spec.difficulty_split[d] for d in difficulties]

    for _ in range(spec.count):
        topic = rng.choice(topics)
        difficulty = rng.choices(difficulties, weights=weights, k=1)[0]
        prompt = build_generation_prompt(spec, topic, difficulty, rng)
        prompts.append((prompt, topic, difficulty))

    return prompts


def diversity_order(
    prompts: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Reorder prompts so no two adjacent prompts share the same topic.

    This prevents mode collapse within a batch — diverse prompts produce
    diverse outputs.
    """
    if len(prompts) <= 1:
        return prompts
    # Group by topic
    by_topic: dict[str, list[tuple[str, str, str]]] = {}
    for p in prompts:
        topic = p[1]
        by_topic.setdefault(topic, []).append(p)
    # Round-robin across topics
    result: list[tuple[str, str, str]] = []
    max_len = max(len(v) for v in by_topic.values())
    topics = list(by_topic.keys())
    for i in range(max_len):
        for topic in topics:
            pool = by_topic[topic]
            if i < len(pool):
                result.append(pool[i])
    return result


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------


def generate_batch(
    handle: Any,
    spec: GenerationSpec,
    rng: random.Random | None = None,
) -> list[GeneratedRow]:
    """Generate a batch of training rows using vLLM batch inference.

    This is the main generation entry point. It:
    1. Expands the spec into diverse prompts
    2. Orders them for diversity within batches
    3. Runs batched vLLM inference
    4. Parses outputs into GeneratedRow objects
    5. Optionally refines hard rows with a critique-revise cycle
    """
    from ..inference import run_inference_batch

    rng = rng or random.Random()
    prompts = expand_generation_prompts(spec, rng)
    prompts = diversity_order(prompts)

    all_rows: list[GeneratedRow] = []
    batch_size = spec.batch_size

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_prompts = [p[0] for p in batch]
        start = time.monotonic()
        results = run_inference_batch(
            handle,
            batch_prompts,
            max_reasoning_tokens=spec.max_reasoning_tokens,
            max_answer_tokens=spec.max_answer_tokens,
            temperature=DEFAULT_TEMP_HOT,
            top_p=DEFAULT_TOP_P,
        )
        elapsed = time.monotonic() - start

        for (prompt, topic, difficulty), result in zip(batch, results):
            raw = result.text if hasattr(result, "text") else str(result)
            row = parse_generation(raw)
            if row:
                row.domain = spec.domain
                row.difficulty = difficulty
                row.generation_time = elapsed / len(batch)
                all_rows.append(row)

    # Multi-turn refinement for hardest rows
    if spec.enable_refinement and all_rows:
        refine_count = max(1, int(len(all_rows) * spec.refinement_fraction))
        hard_rows = [r for r in all_rows if r.difficulty == "hard"][:refine_count]
        if hard_rows:
            refined = _refine_rows(handle, hard_rows, spec)
            # Replace original rows with refined versions
            for orig, ref in zip(hard_rows, refined):
                if ref:
                    idx = all_rows.index(orig)
                    all_rows[idx] = ref

    return all_rows


def _refine_rows(
    handle: Any,
    rows: list[GeneratedRow],
    spec: GenerationSpec,
) -> list[GeneratedRow | None]:
    """Critique-revise cycle for hard rows.

    For each row, asks the model to critique the reasoning and produce
    an improved version.  Returns the refined rows (or None if refinement
    failed).
    """
    from ..inference import run_inference_batch

    critique_prompts: list[str] = []
    for row in rows:
        critique_prompts.append(
            f"Review this {spec.domain} problem and solution for correctness and completeness.\n"
            f"If the reasoning has gaps or errors, rewrite the full solution with corrections.\n"
            f"If it is already correct and complete, output it unchanged.\n\n"
            f"PROBLEM: {row.input}\n\n{_TO}\n{row.thinking_trace}\n{_TC}\n"
            f"{row.answer}\n\n"
            f"---\n\nProvide the reviewed version in the same format:\n"
            f"PROBLEM: <problem>\n\n{_TO}<reasoning>{_TC}\n<answer>"
        )

    results = run_inference_batch(
        handle,
        critique_prompts,
        max_reasoning_tokens=spec.max_reasoning_tokens,
        max_answer_tokens=spec.max_answer_tokens,
        temperature=DEFAULT_TEMP_COOL,  # cool for correctness
        top_p=DEFAULT_TOP_P,
    )

    refined: list[GeneratedRow | None] = []
    for orig_row, result in zip(rows, results):
        raw = result.text if hasattr(result, "text") else str(result)
        parsed = parse_generation(raw)
        if parsed:
            parsed.domain = orig_row.domain
            parsed.difficulty = orig_row.difficulty
            parsed.source = "synthetic-refined"
            refined.append(parsed)
        else:
            refined.append(None)
    return refined


def reroll_failed(
    handle: Any,
    failed_rows: list[GeneratedRow],
    spec: GenerationSpec,
) -> list[GeneratedRow | None]:
    """Re-roll rows that failed quality gates with lower temperature.

    Uses cool temperature for correctness.  Returns new rows (or None if
    the re-roll also fails to parse).
    """
    from ..inference import run_inference_batch

    if not failed_rows:
        return []

    # Reconstruct prompts from the original inputs
    rng = random.Random()
    prompts = []
    for row in failed_rows:
        prompt = build_generation_prompt(spec, row.input[:50], row.difficulty, rng)
        prompts.append(prompt)

    results = run_inference_batch(
        handle,
        prompts,
        max_reasoning_tokens=spec.max_reasoning_tokens,
        max_answer_tokens=spec.max_answer_tokens,
        temperature=DEFAULT_TEMP_COOL,
        top_p=DEFAULT_TOP_P,
    )

    rerolled: list[GeneratedRow | None] = []
    for orig_row, result in zip(failed_rows, results):
        raw = result.text if hasattr(result, "text") else str(result)
        parsed = parse_generation(raw)
        if parsed:
            parsed.domain = orig_row.domain
            parsed.difficulty = orig_row.difficulty
            parsed.source = "synthetic-reroll"
            rerolled.append(parsed)
        else:
            rerolled.append(None)
    return rerolled
