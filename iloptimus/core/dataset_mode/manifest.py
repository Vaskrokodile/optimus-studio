"""Dataset manifest: auto-generated dataset card.

After each generation run, a manifest is produced with:
- Row count, domain distribution, difficulty distribution
- Quality score histogram and mean
- Dedup and contamination stats
- Coverage audit (which capabilities have enough demonstrations)
- Provenance summary (which templates/skills/research sources contributed)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class DatasetManifest:
    run_id: str
    created_at: str = ""
    domain: str = "math"
    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    domain_distribution: dict[str, int] = field(default_factory=dict)
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    mean_quality: float = 0.0
    quality_histogram: dict[str, int] = field(default_factory=dict)
    rejections_by_type: dict[str, int] = field(default_factory=dict)
    contamination_flags: int = 0
    semantic_duplicates: int = 0
    research_sources: int = 0
    research_origins: int = 0
    research_iterations: int = 0
    templates_used: list[str] = field(default_factory=list)
    feature_coverage: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(
    run_id: str,
    accepted_rows: list[dict[str, Any]],
    quality_report: Any,
    *,
    domain: str = "math",
    research_corpus: Any = None,
    templates_used: list[str] | None = None,
    feature_coverage: dict[str, Any] | None = None,
) -> DatasetManifest:
    manifest = DatasetManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        domain=domain,
        total_rows=quality_report.total if quality_report else len(accepted_rows),
        accepted_rows=len(accepted_rows),
        rejected_rows=quality_report.rejected if quality_report else 0,
        mean_quality=quality_report.mean_quality if quality_report else 0.0,
        quality_histogram=dict(quality_report.quality_histogram) if quality_report else {},
        rejections_by_type=dict(quality_report.rejections_by_type) if quality_report else {},
        contamination_flags=quality_report.contamination_flags if quality_report else 0,
        semantic_duplicates=quality_report.semantic_duplicates if quality_report else 0,
        templates_used=templates_used or [],
        feature_coverage=feature_coverage or {},
    )

    domain_dist: Counter[str] = Counter()
    difficulty_dist: Counter[str] = Counter()
    source_dist: Counter[str] = Counter()
    for row in accepted_rows:
        domain_dist[str(row.get("domain", domain))] += 1
        difficulty_dist[str(row.get("difficulty", "unknown"))] += 1
        source_dist[str(row.get("source", "unknown"))] += 1

    manifest.domain_distribution = dict(domain_dist)
    manifest.difficulty_distribution = dict(difficulty_dist)
    manifest.source_distribution = dict(source_dist)

    if research_corpus:
        manifest.research_sources = len(research_corpus.sources)
        manifest.research_origins = len(research_corpus.origins)
        manifest.research_iterations = research_corpus.iterations

    return manifest


def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.public(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def manifest_summary(manifest: DatasetManifest) -> str:
    lines = [
        f"Dataset Manifest: {manifest.run_id}",
        f"  Domain: {manifest.domain}",
        f"  Rows: {manifest.accepted_rows} accepted / {manifest.rejected_rows} rejected / {manifest.total_rows} total",
        f"  Mean quality: {manifest.mean_quality:.3f}",
    ]
    if manifest.difficulty_distribution:
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(manifest.difficulty_distribution.items()))
        lines.append(f"  Difficulty: {dist}")
    if manifest.rejections_by_type:
        top_rej = sorted(manifest.rejections_by_type.items(), key=lambda x: -x[1])[:5]
        rej_str = ", ".join(f"{k}: {v}" for k, v in top_rej)
        lines.append(f"  Top rejections: {rej_str}")
    if manifest.research_sources:
        lines.append(f"  Research: {manifest.research_sources} sources from {manifest.research_origins} origins ({manifest.research_iterations} iterations)")
    if manifest.contamination_flags:
        lines.append(f"  Contamination flags: {manifest.contamination_flags}")
    if manifest.semantic_duplicates:
        lines.append(f"  Semantic duplicates: {manifest.semantic_duplicates}")
    return "\n".join(lines)
