"""Source acquisition: web retrieval for dataset building.

Migrated from ``dataset_factory.py`` — preserves the URL harvesting capability
that allows the model to retrieve real content from the web to build datasets.
This is the "source-grounded" path of dataset mode; the model can also generate
synthetic rows via the generation engine.

The model in dataset mode can choose to:
  (a) generate synthetic rows (generation.py)
  (b) retrieve real content from the web (this module)
  (c) both — retrieve facts, then generate rows grounded in those facts
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..dataset_store import CorpusStore
from ..storage import app_home

HARVEST_CONCURRENCY = 8


async def harvest_urls(urls: list[str], *, kind: str = "web-documentation") -> dict[str, Any]:
    """Fetch many URL candidates in one bounded-concurrency pass.

    Failures on individual URLs are recorded, not raised — a bulk harvest
    keeps whatever succeeded. Returns source dicts ready for the corpus.
    """
    from ..tools import web_fetch

    semaphore = asyncio.Semaphore(HARVEST_CONCURRENCY)

    async def fetch_one(url: str) -> dict[str, Any]:
        async with semaphore:
            try:
                fetched = await web_fetch(url)
            except Exception as error:  # noqa: BLE001
                return {"url": url, "error": str(error)}
            return {"url": url, "fetched": fetched}

    outcomes = await asyncio.gather(*(fetch_one(url) for url in urls))
    sources: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for outcome in outcomes:
        if "error" in outcome:
            failed.append({"url": outcome["url"], "error": outcome["error"]})
            continue
        fetched = outcome["fetched"]
        sources.append(
            {
                "title": fetched.get("url", outcome["url"]),
                "url": fetched.get("url", outcome["url"]),
                "text": fetched.get("text", ""),
                "kind": kind,
                "license": "web-content",
            }
        )
    return {"sources": sources, "failed": failed, "fetched": len(sources)}


def add_sources_to_corpus(
    sources: list[dict[str, Any]],
    store: CorpusStore | None = None,
) -> dict[str, Any]:
    """Add acquired sources to the persistent content-addressed corpus."""
    store = store or CorpusStore()
    return store.add_sources(sources)


def corpus_root() -> Path:
    """Return the corpus storage root."""
    return app_home() / "corpus"


# Re-export for backward compatibility
from pathlib import Path  # noqa: E402
