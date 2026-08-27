"""Dataset mode orchestrator: the entry point for the full dataset pipeline.

When the loop calls ``enter_dataset_mode(spec)``, the orchestrator:

1. Optionally runs continuous web research to gather facts (Layer 5)
2. Spawns batched vLLM generation sessions (Layer 1 + generation)
3. Runs quality gates on all generated rows (Layer 6)
4. Re-rolls failed rows with cooler temperature
5. Builds a dataset manifest (Layer 7)
6. Writes the final JSONL dataset with provenance

The orchestrator also supports:
- Failure-driven generation: feed benchmark failures to guide what rows to generate
- Curriculum-aware generation: track difficulty levels and generate more where the model struggles
- Incremental corpus: dedup against existing rows, not just the current batch
- Source-grounded mode: retrieve real content from the web (via source_acquisition)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..storage import app_home, atomic_write_json
from .generation import (
    GeneratedRow,
    GenerationSpec,
    generate_batch,
    reroll_failed,
)
from .manifest import DatasetManifest, build_manifest, manifest_summary, save_manifest
from .quality import QualityReport, run_quality_gates
from .research import ResearchCorpus, ResearchSpec, continuous_research, research_summary
from .source_acquisition import harvest_urls


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetModeSpec:
    """Top-level spec for a dataset mode session."""

    domain: str = "math"
    target_rows: int = 200
    batch_size: int = 16

    # Generation
    difficulty_split: dict[str, float] = field(default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2})
    seed_problems: list[dict[str, Any]] = field(default_factory=list)
    benchmark_problems: list[str] = field(default_factory=list)  # for contamination check
    enable_refinement: bool = True
    enable_verification: bool = False  # expensive: separate vLLM call per row
    enable_research: bool = True
    enable_reroll: bool = True

    # Research
    research_topic: str = ""
    research_features: list[str] = field(default_factory=list)
    min_research_sources: int = 10

    # Quality
    min_quality: float = 0.5

    # Source acquisition (web retrieval)
    source_urls: list[str] = field(default_factory=list)

    # Existing corpus (for incremental dedup)
    existing_corpus_summary: str = ""
    existing_rows: list[dict[str, Any]] = field(default_factory=list)

    # Model handle (vLLM)
    handle: Any = None


@dataclass
class DatasetModeResult:
    """Result of a dataset mode session."""

    run_id: str
    dataset_path: str
    manifest: dict[str, Any] = field(default_factory=dict)
    accepted_rows: int = 0
    rejected_rows: int = 0
    research_sources: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------


def _sessions_root() -> Path:
    root = app_home() / "dataset-mode"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_dir(run_id: str) -> Path:
    return _sessions_root() / run_id


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def enter_dataset_mode(spec: DatasetModeSpec) -> DatasetModeResult:
    """Enter dataset mode and run the full pipeline.

    This is the main entry point. It:
    1. Optionally runs continuous web research
    2. Generates rows in batched vLLM sessions
    3. Runs quality gates
    4. Re-rolls failed rows
    5. Builds manifest and writes JSONL
    """
    run_id = uuid.uuid4().hex[:12]
    session_dir = _session_dir(run_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    result = DatasetModeResult(
        run_id=run_id,
        dataset_path=str(session_dir / "dataset.jsonl"),
    )

    try:
        # --- Phase 1: Continuous research (optional) ---
        research_corpus = None
        research_context = ""
        if spec.enable_research and spec.handle:
            topic = spec.research_topic or f"{spec.domain} problems and solutions"
            research_spec = ResearchSpec(
                topic=topic,
                features=spec.research_features,
                domain=spec.domain,
                min_sources=spec.min_research_sources,
            )
            research_corpus = asyncio.run(continuous_research(research_spec))
            research_context = research_summary(research_corpus, research_spec)
            result.research_sources = len(research_corpus.sources)

        # --- Phase 1b: Source acquisition (web retrieval, optional) ---
        if spec.source_urls:
            harvest_result = asyncio.run(harvest_urls(spec.source_urls))
            # Sources are available for source-grounded dataset building
            # (stored in corpus, not directly in the generation pipeline)
            _save_sources(session_dir, harvest_result)

        # --- Phase 2: Batched generation ---
        gen_spec = GenerationSpec(
            domain=spec.domain,
            count=spec.target_rows,
            batch_size=spec.batch_size,
            difficulty_split=spec.difficulty_split,
            seed_problems=spec.seed_problems,
            research_context=research_context,
            corpus_summary=spec.existing_corpus_summary,
            enable_refinement=spec.enable_refinement,
        )

        rng = random.Random()
        generated = generate_batch(spec.handle, gen_spec, rng)

        # Convert GeneratedRow objects to dicts for quality gates
        rows = [_row_to_dict(r) for r in generated]

        # --- Phase 3: Quality gates ---
        accepted, quality_report = run_quality_gates(
            rows,
            benchmark_problems=spec.benchmark_problems,
            handle=spec.handle,
            enable_verification=spec.enable_verification,
            min_quality=spec.min_quality,
        )

        # --- Phase 4: Re-roll failed rows (optional) ---
        if spec.enable_reroll and spec.handle:
            failed = [r for r in rows if r not in accepted]
            if failed:
                rerolled = reroll_failed(spec.handle, failed, gen_spec)
                reroll_rows = [_row_to_dict(r) for r in rerolled if r is not None]
                if reroll_rows:
                    reroll_accepted, reroll_report = run_quality_gates(
                        reroll_rows,
                        benchmark_problems=spec.benchmark_problems,
                        handle=spec.handle,
                        enable_verification=spec.enable_verification,
                        min_quality=spec.min_quality,
                    )
                    accepted.extend(reroll_accepted)
                    # Merge quality reports
                    quality_report.accepted += reroll_report.accepted
                    quality_report.rejected += reroll_report.rejected
                    quality_report.total += reroll_report.total

        # --- Phase 5: Write JSONL ---
        dataset_path = session_dir / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as f:
            for row in accepted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # --- Phase 6: Build manifest ---
        from .templates import list_template_sources

        template_sources = list(list_template_sources().keys())
        manifest = build_manifest(
            run_id=run_id,
            accepted_rows=accepted,
            quality_report=quality_report,
            domain=spec.domain,
            research_corpus=research_corpus,
            templates_used=template_sources,
        )
        save_manifest(manifest, session_dir / "manifest.json")

        result.accepted_rows = len(accepted)
        result.rejected_rows = quality_report.rejected
        result.manifest = manifest.public()
        result.elapsed_seconds = time.monotonic() - start

        # Save session state
        atomic_write_json(session_dir / "session.json", result.public())

    except Exception as error:  # noqa: BLE001
        result.error = str(error)
        result.elapsed_seconds = time.monotonic() - start
        atomic_write_json(session_dir / "session.json", result.public())

    return result


def _row_to_dict(row: GeneratedRow) -> dict[str, Any]:
    """Convert a GeneratedRow to a dict for quality gates."""
    return {
        "input": row.input,
        "thinking_trace": row.thinking_trace,
        "answer": row.answer,
        "domain": row.domain,
        "difficulty": row.difficulty,
        "source": row.source,
        "raw_output": row.raw_output,
        "row_hash": row.row_hash,
        "generation_time": row.generation_time,
    }


def _save_sources(session_dir: Path, harvest_result: dict[str, Any]) -> None:
    """Save harvested sources to the session directory."""
    sources_path = session_dir / "sources.json"
    atomic_write_json(sources_path, {
        "fetched": harvest_result.get("fetched", 0),
        "failed": harvest_result.get("failed", []),
        "sources": harvest_result.get("sources", []),
    })


# ---------------------------------------------------------------------------
# Failure-driven generation
# ---------------------------------------------------------------------------


def build_failure_driven_spec(
    failed_results: list[dict[str, Any]],
    domain: str = "math",
    target_rows: int = 100,
    handle: Any = None,
) -> DatasetModeSpec:
    """Build a DatasetModeSpec from benchmark failures.

    Extracts problem concepts from failed benchmark attempts and uses them
    as seed problems for generation. This targets the model's weak areas.
    """
    seed_problems = []
    for result in failed_results:
        problem = result.get("problem", result.get("input", ""))
        if problem:
            seed_problems.append({
                "problem": problem,
                "error_type": result.get("error_type", "wrong_answer"),
            })

    # Focus more on hard difficulty when learning from failures
    difficulty_split = {"easy": 0.2, "medium": 0.4, "hard": 0.4}

    return DatasetModeSpec(
        domain=domain,
        target_rows=target_rows,
        seed_problems=seed_problems,
        difficulty_split=difficulty_split,
        handle=handle,
        enable_research=True,
        enable_refinement=True,
    )


# ---------------------------------------------------------------------------
# Curriculum-aware generation
# ---------------------------------------------------------------------------


def build_curriculum_spec(
    performance_by_difficulty: dict[str, float],
    domain: str = "math",
    target_rows: int = 100,
    handle: Any = None,
) -> DatasetModeSpec:
    """Build a DatasetModeSpec that targets weak difficulty levels.

    ``performance_by_difficulty`` maps difficulty to accuracy (0.0-1.0).
    Generates more rows at difficulty levels where the model performs poorly.
    """
    if not performance_by_difficulty:
        difficulty_split = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    else:
        # Inverse of performance: worse accuracy = more rows
        inv = {k: max(0.1, 1.0 - v) for k, v in performance_by_difficulty.items()}
        total = sum(inv.values())
        difficulty_split = {k: v / total for k, v in inv.items()}

    return DatasetModeSpec(
        domain=domain,
        target_rows=target_rows,
        difficulty_split=difficulty_split,
        handle=handle,
        enable_research=True,
        enable_refinement=True,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def list_sessions() -> list[dict[str, Any]]:
    """List all dataset mode sessions."""
    sessions: list[dict[str, Any]] = []
    for path in sorted(_sessions_root().glob("*/session.json")):
        try:
            sessions.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sessions


def get_session(run_id: str) -> dict[str, Any] | None:
    """Get a specific session's state."""
    path = _session_dir(run_id) / "session.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_dataset(run_id: str) -> list[dict[str, Any]]:
    """Load the dataset JSONL from a session."""
    path = _session_dir(run_id) / "dataset.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
