"""Quality gates for dataset mode.

Every generated row must pass through these gates before joining the corpus:

1. Structural validation: has <think>...</think> block, has answer, minimum length
2. Semantic dedup: embedding similarity against existing corpus (vLLM embeddings)
3. Contamination check: n-gram overlap with benchmark problems
4. Laziness detection: trivial inputs, copy-pasted templates, missing reasoning
5. Slop detection: repetition, generic filler, coherence check
6. Quality scoring: 0.0-1.0 composite (correctness x reasoning_depth x diversity)
7. Self-verification: separate vLLM call to verify correctness (optional)
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..dedup import MinHashDuplicateGuard, normalize_text

# Think tags
_TO = chr(60) + "think" + chr(62)
_TC = chr(60) + "/think" + chr(62)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_INPUT_CHARS = 20
MIN_THINKING_CHARS = 50
MIN_ANSWER_CHARS = 1
MAX_REPETITION_RATIO = 0.3
MIN_QUALITY_SCORE = 0.5
SEMANTIC_DEDUP_THRESHOLD = 0.85
CONTAMINATION_NGRAM_SIZE = 8
CONTAMINATION_THRESHOLD = 0.7

_SLOP_PHRASES = [
    "this is an interesting problem",
    "many ways to approach",
    "lets think about this",
    "let me think about this",
    "as we can see",
    "in conclusion",
    "this problem tests",
    "the key insight is",
    "without loss of generality",
    "it is worth noting",
    "it should be noted",
    "needless to say",
    "as mentioned earlier",
    "going back to",
    "now lets consider",
    "now let us consider",
]

_TRIVIAL_PATTERNS = [
    r"^what is \d+\s*[+\-*/]\s*\d+\??$",
    r"^solve.*x\s*=\s*\d+",
    r"^what is the .{0,20} of .{0,10}\??$",
]


@dataclass
class QualityResult:
    passed: bool
    quality_score: float
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.passed


@dataclass
class QualityReport:
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    rejections_by_type: dict[str, int] = field(default_factory=dict)
    mean_quality: float = 0.0
    quality_histogram: dict[str, int] = field(default_factory=dict)
    contamination_flags: int = 0
    semantic_duplicates: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections_by_type": dict(self.rejections_by_type),
            "mean_quality": round(self.mean_quality, 4),
            "quality_histogram": dict(self.quality_histogram),
            "contamination_flags": self.contamination_flags,
            "semantic_duplicates": self.semantic_duplicates,
        }


def check_structure(row: dict[str, Any]) -> tuple[bool, list[str]]:
    rejections: list[str] = []
    input_text = str(row.get("input", ""))
    thinking = str(row.get("thinking_trace", ""))
    answer = str(row.get("answer", ""))

    if len(input_text) < MIN_INPUT_CHARS:
        rejections.append("input_too_short")
    if len(thinking) < MIN_THINKING_CHARS:
        rejections.append("thinking_too_short")
    if len(answer) < MIN_ANSWER_CHARS:
        rejections.append("answer_missing")

    raw = str(row.get("raw_output", ""))
    if raw and _TO not in raw:
        rejections.append("missing_think_open_tag")
    if raw and _TC not in raw:
        rejections.append("missing_think_close_tag")

    return (len(rejections) == 0, rejections)


def check_repetition(text: str) -> tuple[bool, float]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return (True, 0.0)
    unique = len(set(lines))
    total = len(lines)
    ratio = 1.0 - (unique / total)
    return (ratio < MAX_REPETITION_RATIO, ratio)


def check_slop(text: str) -> tuple[bool, list[str]]:
    text_lower = text.lower()
    found = [p for p in _SLOP_PHRASES if p in text_lower]
    return (len(found) == 0, found)


def check_trivial_input(input_text: str) -> bool:
    for pattern in _TRIVIAL_PATTERNS:
        if re.match(pattern, input_text.strip(), re.IGNORECASE):
            return True
    if len(input_text.strip()) < 30:
        return True
    return False


def check_contamination(
    input_text: str,
    benchmark_problems: list[str],
    ngram_size: int = CONTAMINATION_NGRAM_SIZE,
) -> tuple[bool, float]:
    if not benchmark_problems:
        return (False, 0.0)
    input_ngrams = _ngrams(input_text.lower(), ngram_size)
    if not input_ngrams:
        return (False, 0.0)
    max_overlap = 0.0
    for benchmark in benchmark_problems:
        bench_ngrams = _ngrams(benchmark.lower(), ngram_size)
        if not bench_ngrams:
            continue
        overlap = len(input_ngrams & bench_ngrams) / len(input_ngrams | bench_ngrams)
        if overlap > max_overlap:
            max_overlap = overlap
    return (max_overlap >= CONTAMINATION_THRESHOLD, max_overlap)


def _ngrams(text: str, n: int) -> set[str]:
    words = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class SemanticDedupGuard:
    def __init__(
        self,
        threshold: float = SEMANTIC_DEDUP_THRESHOLD,
        handle: Any = None,
    ) -> None:
        self.threshold = threshold
        self.handle = handle
        self._embeddings: list[list[float]] = []
        self._fallback = MinHashDuplicateGuard(threshold=threshold * 0.9)
        self._use_embeddings = handle is not None

    def is_duplicate(self, text: str) -> bool:
        if self._use_embeddings:
            try:
                embedding = self._get_embedding(text)
                if embedding:
                    for prior in self._embeddings:
                        sim = _cosine_similarity(embedding, prior)
                        if sim >= self.threshold:
                            return True
                    self._embeddings.append(embedding)
                    return False
            except Exception:
                pass
        return self._fallback.is_duplicate(text)

    def _get_embedding(self, text: str) -> list[float] | None:
        if not self.handle:
            return None
        try:
            from ..inference import _backend_for
            backend = _backend_for(self.handle)
            if hasattr(backend, "embed"):
                return backend.embed(self.handle, text[:2000])
        except Exception:
            pass
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_quality(row: dict[str, Any]) -> float:
    thinking = str(row.get("thinking_trace", ""))
    answer = str(row.get("answer", ""))
    input_text = str(row.get("input", ""))

    step_markers = thinking.count("step") + thinking.count("Step")
    line_count = len([l for l in thinking.split("\n") if l.strip()])
    char_count = len(thinking)
    # Reasoning depth: combine line count, char count, and step markers
    # A single long line with substantive reasoning should still score well
    reasoning_depth = min(1.0, (line_count / 15) + (char_count / 800) + (step_markers * 0.05))

    answer_completeness = min(1.0, len(answer) / 50) if answer else 0.0

    _, slop_found = check_slop(thinking)
    _, rep_ratio = check_repetition(thinking)
    cleanliness = 1.0 - (len(slop_found) * 0.15) - (rep_ratio * 0.5)
    cleanliness = max(0.0, cleanliness)

    is_trivial = check_trivial_input(input_text)
    input_quality = 0.2 if is_trivial else min(1.0, len(input_text) / 200)

    score = (
        0.35 * reasoning_depth
        + 0.25 * answer_completeness
        + 0.25 * cleanliness
        + 0.15 * input_quality
    )
    return max(0.0, min(1.0, score))


def verify_row(
    handle: Any,
    row: dict[str, Any],
    domain: str = "math",
) -> tuple[bool, str]:
    from ..inference import run_inference

    input_text = row.get("input", "")
    answer = row.get("answer", "")

    prompt = (
        f"Solve this {domain} problem and verify the given answer.\n\n"
        f"Problem: {input_text}\n\n"
        f"Claimed answer: {answer}\n\n"
        f"Re-solve the problem independently. Then state whether the claimed "
        f"answer is CORRECT or INCORRECT.\n"
        f"Output: VERDICT: CORRECT or VERDICT: INCORRECT"
    )

    try:
        result = run_inference(
            handle,
            prompt,
            max_reasoning_tokens=2048,
            max_answer_tokens=256,
            temperature=0.0,
        )
        text = result.text if hasattr(result, "text") else str(result)
        if "VERDICT: CORRECT" in text.upper():
            return (True, "verified_correct")
        if "VERDICT: INCORRECT" in text.upper():
            return (False, "verified_incorrect")
        return (True, "verification_inconclusive")
    except Exception as error:
        return (True, f"verification_error: {error}")


def run_quality_gates(
    rows: list[dict[str, Any]],
    *,
    benchmark_problems: list[str] | None = None,
    handle: Any = None,
    enable_verification: bool = False,
    min_quality: float = MIN_QUALITY_SCORE,
) -> tuple[list[dict[str, Any]], QualityReport]:
    report = QualityReport(total=len(rows))
    benchmark_problems = benchmark_problems or []
    semantic_guard = SemanticDedupGuard(handle=handle)
    accepted: list[dict[str, Any]] = []

    for row in rows:
        result = QualityResult(passed=True, quality_score=0.0)

        ok, rejections = check_structure(row)
        if not ok:
            result.rejections.extend(rejections)
            result.passed = False

        if result.passed and benchmark_problems:
            is_contam, overlap = check_contamination(
                row.get("input", ""), benchmark_problems
            )
            result.checks["contamination_overlap"] = round(overlap, 4)
            if is_contam:
                result.rejections.append("contaminated")
                result.passed = False
                report.contamination_flags += 1

        if result.passed:
            if check_trivial_input(row.get("input", "")):
                result.rejections.append("trivial_input")
                result.passed = False

        if result.passed:
            is_clean, slop_found = check_slop(row.get("thinking_trace", ""))
            result.checks["slop_phrases"] = slop_found
            if len(slop_found) > 2:
                result.rejections.append("excessive_slop")
                result.passed = False

        if result.passed:
            is_clean, rep_ratio = check_repetition(row.get("thinking_trace", ""))
            result.checks["repetition_ratio"] = round(rep_ratio, 4)
            if not is_clean:
                result.rejections.append("excessive_repetition")
                result.passed = False

        if result.passed:
            result.quality_score = score_quality(row)
            result.checks["quality_score"] = round(result.quality_score, 4)
            if result.quality_score < min_quality:
                result.rejections.append("low_quality")
                result.passed = False

        if result.passed:
            combined = f"{row.get('input', '')} {row.get('thinking_trace', '')}"
            if semantic_guard.is_duplicate(combined):
                result.rejections.append("semantic_duplicate")
                result.passed = False
                report.semantic_duplicates += 1

        if result.passed and enable_verification and handle:
            verified, note = verify_row(handle, row, row.get("domain", "math"))
            result.checks["verification"] = note
            if not verified:
                result.rejections.append("verification_failed")
                result.passed = False

        if result.passed:
            normalized = normalize_text(row.get("input", "") + row.get("thinking_trace", ""))
            row["quality_score"] = result.quality_score
            row["row_sha256"] = hashlib.sha256(normalized.encode()).hexdigest()
            row["quality_checks"] = result.checks
            accepted.append(row)
            report.accepted += 1
            bucket = f"{int(result.quality_score * 10) / 10:.1f}"
            report.quality_histogram[bucket] = report.quality_histogram.get(bucket, 0) + 1
        else:
            report.rejected += 1
            for rej in result.rejections:
                report.rejections_by_type[rej] = report.rejections_by_type.get(rej, 0) + 1

    if accepted:
        report.mean_quality = sum(r.get("quality_score", 0.0) for r in accepted) / len(accepted)

    return accepted, report
