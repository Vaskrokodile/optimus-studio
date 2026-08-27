"""Template exemplars for dataset mode.

Loads, normalizes, and serves high-quality reasoning-trace rows from curated
datasets (Kimi K3, Fable 5, Qwen 3) as few-shot exemplars.  These show the
generation model what a good training row looks like — both the input and the
thinking trace.

Templates are shipped as JSONL in ``resources/dataset-templates/`` and loaded
from disk on first access.  Each row is normalized to a common format:

    {"input": str, "thinking_trace": str, "answer": str,
     "source": str, "domain": str}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "resources" / "dataset-templates"

_SOURCE_FILES = {
    "openthoughts": "openthoughts_sample.jsonl",
    "limo": "limo_sample.jsonl",
    "qwen3": "qwen3_sample.jsonl",
    "kimi_k3": "kimi_k3_sample.jsonl",
}

# Domain → preferred template sources (ordered by relevance)
_DOMAIN_SOURCES: dict[str, list[str]] = {
    "math": ["qwen3", "limo"],
    "reasoning": ["limo", "qwen3"],
    "coding": ["openthoughts", "kimi_k3"],
    "agentic": ["kimi_k3", "openthoughts"],
}


@dataclass(frozen=True)
class TemplateRow:
    """A normalized template exemplar."""

    input: str
    thinking_trace: str
    answer: str
    source: str
    domain: str

    def as_example(self) -> str:
        """Render as a few-shot example string for injection into prompts."""
        parts = [f"Input: {self.input}"]
        if self.thinking_trace:
            parts.append(f"<think>\n{self.thinking_trace}\n</think>")
        if self.answer:
            parts.append(self.answer)
        return "\n".join(parts)


def _load_source(name: str) -> list[TemplateRow]:
    filename = _SOURCE_FILES.get(name)
    if not filename:
        return []
    path = TEMPLATES_ROOT / filename
    if not path.exists():
        return []
    rows: list[TemplateRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(
            TemplateRow(
                input=data.get("input", ""),
                thinking_trace=data.get("thinking_trace", ""),
                answer=data.get("answer", ""),
                source=data.get("source", name),
                domain=data.get("domain", ""),
            )
        )
    return rows


def load_templates(domain: str = "math") -> list[TemplateRow]:
    """Load template exemplars for a domain, ordered by relevance.

    Returns templates from the most relevant source first, then less relevant
    sources.  All available templates are returned (typically 8-16 rows).
    """
    sources = _DOMAIN_SOURCES.get(domain, ["qwen3", "fable5"])
    # Also include sources not in the domain's preferred list for diversity
    all_sources = list(sources) + [s for s in _SOURCE_FILES if s not in sources]
    rows: list[TemplateRow] = []
    for source in all_sources:
        rows.extend(_load_source(source))
    return rows


def sample_templates(
    domain: str = "math",
    count: int = 3,
    rng: random.Random | None = None,
) -> list[TemplateRow]:
    """Sample a few diverse template exemplars for a domain.

    Prefers templates from the domain's primary sources.  Returns at most
    ``count`` rows, each from a different source when possible.
    """
    rng = rng or random.Random(0)
    rows = load_templates(domain)
    if not rows:
        return []
    # Group by source for diversity
    by_source: dict[str, list[TemplateRow]] = {}
    for row in rows:
        by_source.setdefault(row.source, []).append(row)
    # Round-robin sample across sources
    result: list[TemplateRow] = []
    sources = list(by_source.keys())
    rng.shuffle(sources)
    idx = 0
    while len(result) < count and any(by_source[s] for s in sources):
        source = sources[idx % len(sources)]
        pool = by_source[source]
        if pool:
            result.append(pool.pop(rng.randrange(len(pool))))
        idx += 1
    return result[:count]


def template_prompt(
    domain: str = "math",
    count: int = 3,
    max_chars: int = 4_000,
    rng: random.Random | None = None,
) -> str:
    """Build a few-shot template block for injection into generation prompts.

    Returns a string like:

        Here are examples of high-quality training rows:

        ### Example 1 (source: qwen3)
        Input: ...
        <think>...</think>
        ...

    Truncated to ``max_chars`` to fit the model's context budget. Each
    example is truncated to a per-example budget so that long coding traces
    don't crowd out other examples.
    """
    samples = sample_templates(domain, count, rng)
    if not samples:
        return ""
    header = "Here are examples of high-quality training rows. Match this quality:\n"
    parts = [header]
    remaining = max_chars - len(header)
    # Per-example budget: divide remaining evenly, with a floor of 500 chars
    per_example = max(500, remaining // max(1, len(samples)))
    for i, row in enumerate(samples, 1):
        example = row.as_example()
        # Truncate the example to its per-example budget
        if len(example) > per_example:
            example = example[: per_example - 20] + "\n[...truncated]\n"
        block = f"\n### Example {i} (source: {row.source})\n{example}\n"
        if len(block) > remaining:
            break
        parts.append(block)
        remaining -= len(block)
    return "".join(parts)


def list_template_sources() -> dict[str, dict]:
    """Return metadata about available template sources."""
    manifest_path = TEMPLATES_ROOT / "manifest.json"
    if manifest_path.exists():
        import json as _json

        return _json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})
    return {name: {"file": fname} for name, fname in _SOURCE_FILES.items()}
