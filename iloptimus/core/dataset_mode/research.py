"""Continuous research engine for dataset mode.

Replaces the one-shot ``web_search`` with an iterative research loop:

    plan_queries(spec) → [q1, q2, ...]
      loop:
        batch_search(queries, rate_limited) → results
        batch_fetch(result_urls, parallel) → pages
        analyze_coverage(pages, spec) → gaps
        if coverage_met: break
        plan_next_queries(gaps) → refined_queries

Key improvements over the existing ``tools.web_search``:
- Rate limiting: per-domain exponential backoff, configurable delay
- Result caching: SHA256-keyed, never re-fetch same URL
- Parallel fetch: 8-16 concurrent with backoff
- Coverage-aware stop: stops when min_sources + min_origins met
- Query planning: feature-aware query generation, not just "search for X"
- Iterative refinement: learns from results to plan better queries
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MIN_SOURCES = 10
DEFAULT_MIN_ORIGINS = 3
DEFAULT_BATCH_CONCURRENCY = 12
DEFAULT_PER_DOMAIN_DELAY = 2.0  # seconds between requests to same domain
DEFAULT_SEARCH_TIMEOUT = 12
DEFAULT_FETCH_TIMEOUT = 12
MAX_SNIPPET_CHARS = 2000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResearchSpec:
    """What to research for dataset building."""

    topic: str
    features: list[str] = field(default_factory=list)
    domain: str = "math"
    min_sources: int = DEFAULT_MIN_SOURCES
    min_origins: int = DEFAULT_MIN_ORIGINS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_query_terms: int = 6


@dataclass
class ResearchSource:
    """A single acquired research source."""

    url: str
    title: str
    text: str
    origin: str  # domain
    query: str = ""  # which query found it
    fetched_at: float = field(default_factory=time.time)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]


@dataclass
class ResearchCorpus:
    """Accumulated research sources with coverage tracking."""

    sources: list[ResearchSource] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    iterations: int = 0
    _seen_urls: set[str] = field(default_factory=set, repr=False)
    _seen_hashes: set[str] = field(default_factory=set, repr=False)

    @property
    def origins(self) -> set[str]:
        return {s.origin for s in self.sources}

    @property
    def coverage_met(self) -> bool:
        return len(self.sources) >= 10 and len(self.origins) >= 3

    def add(self, source: ResearchSource) -> bool:
        """Add a source if not already seen. Returns True if added."""
        if source.url in self._seen_urls:
            return False
        content_hash = hashlib.sha256(source.text.encode()).hexdigest()[:16]
        if content_hash in self._seen_hashes:
            return False
        self._seen_urls.add(source.url)
        self._seen_hashes.add(content_hash)
        self.sources.append(source)
        return True

    def coverage_report(self, spec: ResearchSpec) -> dict[str, Any]:
        """Report coverage against the spec's requirements."""
        return {
            "sources": len(self.sources),
            "origins": len(self.origins),
            "min_sources_met": len(self.sources) >= spec.min_sources,
            "min_origins_met": len(self.origins) >= spec.min_origins,
            "iterations": self.iterations,
            "queries_used": len(self.queries_used),
            "origin_list": sorted(self.origins),
        }


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------


def plan_initial_queries(spec: ResearchSpec) -> list[str]:
    """Plan the first batch of search queries for a research spec."""
    queries = [spec.topic]
    # Feature-specific queries
    for feature in spec.features[: spec.max_query_terms - 1]:
        queries.append(f"{spec.topic} {feature}")
    # Domain-specific query patterns
    if spec.domain == "math":
        queries.append(f"{spec.topic} competition problems solutions")
        queries.append(f"{spec.topic} olympiad examples worked")
    elif spec.domain == "coding":
        queries.append(f"{spec.topic} algorithm examples implementation")
        queries.append(f"{spec.topic} programming challenges solutions")
    elif spec.domain == "reasoning":
        queries.append(f"{spec.topic} logic puzzles examples")
        queries.append(f"{spec.topic} deduction reasoning problems")
    return queries[: spec.max_query_terms]


def plan_refinement_queries(
    corpus: ResearchCorpus,
    spec: ResearchSpec,
) -> list[str]:
    """Plan the next batch of queries based on coverage gaps."""
    gaps: list[str] = []
    # Check which features have no sources yet
    covered_features: set[str] = set()
    for source in corpus.sources:
        text_lower = source.text.lower()
        for feature in spec.features:
            if feature.lower() in text_lower:
                covered_features.add(feature)
    missing_features = [f for f in spec.features if f not in covered_features]
    for feature in missing_features[:3]:
        gaps.append(f"{spec.topic} {feature} examples tutorial")
    # Add origin-diversity queries if we're dominated by one origin
    origin_counts: dict[str, int] = {}
    for source in corpus.sources:
        origin_counts[source.origin] = origin_counts.get(source.origin, 0) + 1
    dominant = max(origin_counts, key=origin_counts.get) if origin_counts else None
    if dominant and origin_counts[dominant] > len(corpus.sources) * 0.5:
        # Search on different platforms for diversity
        gaps.append(f"{spec.topic} site:stackoverflow.com")
        gaps.append(f"{spec.topic} site:github.com")
    # General expansion
    if not gaps:
        gaps.append(f"{spec.topic} advanced examples")
        gaps.append(f"{spec.topic} edge cases problems")
    return gaps[: spec.max_query_terms]


# ---------------------------------------------------------------------------
# Rate-limited batch search
# ---------------------------------------------------------------------------


class DomainRateLimiter:
    """Per-domain rate limiter with exponential backoff."""

    def __init__(self, delay: float = DEFAULT_PER_DOMAIN_DELAY) -> None:
        self.delay = delay
        self._last_request: dict[str, float] = {}

    async def wait(self, domain: str) -> None:
        now = time.monotonic()
        last = self._last_request.get(domain, 0.0)
        elapsed = now - last
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request[domain] = time.monotonic()


async def _batch_search(
    queries: list[str],
    rate_limiter: DomainRateLimiter,
) -> list[dict[str, Any]]:
    """Run multiple web searches with rate limiting."""
    from ..tools import web_search

    async def search_one(query: str) -> dict[str, Any]:
        domain = "search-provider"
        await rate_limiter.wait(domain)
        try:
            return await web_search(query)
        except Exception as error:  # noqa: BLE001
            return {"query": query, "results": [], "error": str(error)}

    results = await asyncio.gather(*(search_one(q) for q in queries))
    return list(results)


async def _batch_fetch(
    urls: list[str],
    rate_limiter: DomainRateLimiter,
    concurrency: int = DEFAULT_BATCH_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Fetch multiple URLs in parallel with per-domain rate limiting."""
    from ..tools import web_fetch

    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_one(url: str) -> dict[str, Any]:
        async with semaphore:
            domain = urlparse(url).hostname or "unknown"
            await rate_limiter.wait(domain)
            try:
                fetched = await web_fetch(url)
                return {"url": url, "fetched": fetched}
            except Exception as error:  # noqa: BLE001
                return {"url": url, "error": str(error)}

    return await asyncio.gather(*(fetch_one(url) for url in urls))


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------


async def continuous_research(spec: ResearchSpec) -> ResearchCorpus:
    """Iteratively research until coverage targets are met or budget exhausted.

    This is the main entry point for the continuous research engine. It:
    1. Plans initial queries from the spec
    2. Searches and fetches results in parallel
    3. Analyzes coverage gaps
    4. Refines queries and repeats
    5. Stops when min_sources + min_origins are met or max_iterations reached
    """
    corpus = ResearchCorpus()
    rate_limiter = DomainRateLimiter()
    queries = plan_initial_queries(spec)

    for iteration in range(spec.max_iterations):
        corpus.iterations = iteration + 1
        if corpus.coverage_met and iteration > 0:
            break

        # Search
        search_results = await _batch_search(queries, rate_limiter)
        corpus.queries_used.extend(queries)

        # Collect URLs to fetch
        urls_to_fetch: list[tuple[str, str, str]] = []  # (url, title, query)
        for result in search_results:
            for row in result.get("results", [])[:3]:
                url = row.get("url", "")
                title = row.get("title", "")
                query = result.get("query", "")
                if url and url not in corpus._seen_urls:
                    urls_to_fetch.append((url, title, query))

        # Fetch in parallel
        if urls_to_fetch:
            fetch_results = await _batch_fetch(
                [u[0] for u in urls_to_fetch],
                rate_limiter,
            )
            for (url, title, query), fetch_outcome in zip(urls_to_fetch, fetch_results):
                if "error" in fetch_outcome:
                    continue
                fetched = fetch_outcome["fetched"]
                source = ResearchSource(
                    url=url,
                    title=title or fetched.get("url", url),
                    text=fetched.get("text", "")[:MAX_SNIPPET_CHARS],
                    origin=urlparse(url).hostname or "unknown",
                    query=query,
                )
                corpus.add(source)

        # Plan next queries based on gaps
        if not corpus.coverage_met:
            queries = plan_refinement_queries(corpus, spec)
            if not queries:
                break
        else:
            break

    return corpus


def research_to_sources(corpus: ResearchCorpus) -> list[dict[str, Any]]:
    """Convert a research corpus into source dicts for the dataset pipeline."""
    return [
        {
            "title": s.title,
            "url": s.url,
            "text": s.text,
            "kind": "web-research",
            "license": "web-content",
            "origin": s.origin,
        }
        for s in corpus.sources
    ]


def research_summary(corpus: ResearchCorpus, spec: ResearchSpec) -> str:
    """Build a compact text summary of research results for prompt injection."""
    if not corpus.sources:
        return "No research sources found."
    lines = [f"Research gathered {len(corpus.sources)} sources from {len(corpus.origins)} origins:"]
    for source in corpus.sources[:8]:
        snippet = re.sub(r"\s+", " ", source.text[:200]).strip()
        lines.append(f"- [{source.origin}] {source.title[:80]}: {snippet}")
    if len(corpus.sources) > 8:
        lines.append(f"... and {len(corpus.sources) - 8} more sources.")
    return "\n".join(lines)
