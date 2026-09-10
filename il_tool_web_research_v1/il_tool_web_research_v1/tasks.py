"""tool-web-research-v1 â€” search, read the right page, cite evidence.

World: a tiny web corpus with distractor pages (SEO spam, outdated mirrors).
Teaches: query formulation, choosing the authoritative source over
distractors, no re-searching loops, evidence citation in the answer.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

PAGES = {
    "docs.optimus.dev/rate-limits": "Official docs: the API rate limit is 120 requests per minute per key. Bursts up to 240 are tolerated for 5 seconds.",
    "blog.random.dev/rate-limits-rumors": "Some blog claiming the rate limit might be 60 rpm, unverified.",
    "forum.thread/8812": "User claims rate limits are 100 rpm 'I think'.",
    "docs.optimus.dev/v2/migration": "Migration guide: v1 endpoints were removed in 2025. Use /v2 endpoints with bearer tokens.",
    "wiki.outdated.com/api": "Outdated wiki: rate limit used to be 60 rpm in 2023.",
}


def _make_tools(corpus: dict[str, str]):
    def search(state, args):
        query = str(args.get("query", "")).lower()
        state["searches"] = state.get("searches", 0) + 1
        hits = [url for url, text in corpus.items() if any(w in text.lower() for w in query.split())]
        return hits, "results: " + (", ".join(hits) if hits else "(none)")

    def read_page(state, args):
        url = args.get("url", "")
        state.setdefault("read", set()).add(url)
        if url not in corpus:
            return None, f"ERROR: 404 {url}"
        return corpus[url], f"[{url}] {corpus[url]}"

    def post_comment(state, args):
        return None, "ERROR: posting is not available in read-only research mode"

    return {
        "web_search": ToolSpec("web_search", "Search the corpus; returns matching URLs.", {"query": "str"}, search),
        "read_page": ToolSpec("read_page", "Fetch the full text of one URL.", {"url": "str"}, read_page),
        "post_comment": ToolSpec("post_comment", "Post a comment on a page.", {"url": "str", "text": "str"}, post_comment, distractor=True),
        "follow_links": ToolSpec("follow_links", "Alias that does nothing useful.", {"url": "str"}, post_comment, distractor=True),
    }


def _web_task(idx, name, corpus, question, expected_tokens, expert):
    tools = _make_tools(corpus)

    def goal(state):
        return any("docs.optimus.dev" in u for u in state.get("read", [])) or any(
            "docs.optimus.dev" in u for u in state.get("read", [])
        ), f"read={state.get('read', [])}"

    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=(
            "You research a simulated web through tools:\n"
            "- web_search(query) -> list of URLs\n"
            "- read_page(url) -> page text\n\n"
            f"{question}\n\n"
            "Prefer the authoritative source (docs.optimus.dev) over blogs, forums and "
            "outdated wikis. Make your tool calls in <tool>...</tool> blocks and give the "
            "final answer in <answer>...</answer>."
        ),
        tools=tools,
        init={},
        goal=goal,
        expert=expert,
        verify_answer=answer_contains(*expected_tokens),
        expert_answer=" ".join(expected_tokens),
        expected_concepts=["web_search", "read_page", "docs.optimus.dev"],
        scenario="web-research",
    )


CORPUS_A = {
    "docs.optimus.dev/rate-limits": "Official docs: the rate limit is 120 requests per minute with 5 second burst.",
    "blog.random.com/rumors": "Rumor: the limit might be 60 rpm.",
    "forum.thread/881": "Forum guess: maybe 100 rpm?",
    "wiki.outdated.org/api": "Outdated wiki from 2023: limit was 30 rpm.",
}
CORPUS_B = {
    "docs.optimus.dev/v2/auth": "Official: authentication uses bearer tokens issued by /v2/token with 3600s expiry.",
    "blog.tutorials.io/auth-explained": "Tutorial: 'auth is complicated, use any token'.",
    "pastebin.snippet/9x": "Random snippet mentioning api keys.",
}

TASKS = [
    _web_task(0, "find_official_rate_limit", CORPUS_A,
              "What is the official API rate limit? Find it and report the number.",
              ["120"], [("web_search", {"query": "optimus api rate limit"}), ("read_page", {"url": "docs.optimus.dev/rate-limits"})]),
    _web_task(1, "auth_token_expiry", CORPUS_B,
              "How long do auth bearer tokens last? Use the official documentation.",
              ["3600"], [("web_search", {"query": "optimus auth bearer token expiry"}), ("read_page", {"url": "docs.optimus.dev/v2/auth"})]),
    _web_task(2, "avoid_outdated_wiki", CORPUS_A,
              "Report the current rate limit. Beware: an outdated wiki claims 30 rpm.",
              ["120"], [("web_search", {"query": "rate limit"}), ("read_page", {"url": "docs.optimus.dev/rate-limits"})]),
    _web_task(3, "search_then_read", CORPUS_B,
              "Find how authentication works and name the token type in your answer.",
              ["bearer"], [("web_search", {"query": "authentication"}), ("read_page", {"url": "docs.optimus.dev/v2/auth"})]),
]
