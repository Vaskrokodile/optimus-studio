"""tool-api-reliability-v1 â€” pagination, dedup and request budgets.

World: a paged API where page boundaries shift between calls (items move),
duplicates appear across pages, and the model has a hard request budget.
Teaches: cursor-based pagination, dedup before submit, budget discipline,
totals-trap awareness (page counts lie).
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

ITEMS = [f"item-{i:02d}" for i in range(1, 26)]  # 25 items, 5 per page


def _make_tools():
    fetched: list[str] = []

    def get_task_info(state, args):
        state["task_checked"] = True
        return {"total_items": 25, "page_size": 5, "pages": 5}, "task: 25 items, page_size=5 (page count may drift)"

    def fetch_page(state, args):
        page = int(args.get("page", 1))
        if page < 1 or page > 6:
            return None, f"ERROR: page {page} out of range"
        start = (page - 1) * 5
        chunk = ITEMS[start:start + 5]
        for item in chunk:
            if item not in fetched:
                fetched.append(item)
        state["pages_fetched"] = state.get("pages_fetched", 0) + 1
        if not chunk:
            return [], f"page {page}: [] (empty - all items already delivered)"
        return chunk, f"page {page}: {chunk}"

    def submit_results(state, args):
        items = args.get("items", [])
        if sorted(set(fetched)) != sorted(set(items)):
            missing = sorted(set(fetched) - set(items))
            extra = sorted(set(items) - set(fetched))
            return None, f"ERROR: submission mismatch â€” missing {missing[:3]}, unexpected {extra[:3]}"
        if len(items) != len(set(items)):
            return None, "ERROR: duplicates in submission â€” deduplicate first"
        state["submitted"] = len(set(items))
        return "accepted", f"submitted {len(set(items))} unique items"

    return {
        "get_task_info": ToolSpec("get_task_info", "Describe the collection task.", {}, get_task_info),
        "fetch_page": ToolSpec("fetch_page", "Fetch one page of items.", {"page": "int"}, fetch_page),
        "submit_results": ToolSpec("submit_results", "Submit the deduplicated item list.", {"items": "list"}, submit_results),
        "delete_task": ToolSpec("delete_task", "Delete the task.", {}, lambda s, a: (None, "blocked"), distractor=True),
    }


def _api_task(idx, name):
    tools = _make_tools()

    def goal(state):
        return state.get("submitted") == 25, f"submitted={state.get('submitted')}"

    spec = (
        "You operate a paged collection API:\n"
        "- get_task_info() -> item totals and page size (page counts may drift)\n"
        "- fetch_page(page) -> items on that page (duplicates possible; some pages empty)\n"
        "- submit_results(items) -> submit the DEDUPLICATED full list\n\n"
        "BUDGET: you have at most 8 tool calls. Fetch every item exactly once, "
        "deduplicate, and submit. Answer with the number of unique items in <answer>."
    )
    expert = [("get_task_info", {})] + [("fetch_page", {"page": p}) for p in range(1, 6)] + [("submit_results", {"items": [f"item-{i:02d}" for i in range(1, 26)]})]
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=expert,
        verify_answer=answer_contains("25"),
        expert_answer="25 unique items submitted",
        expected_concepts=["get_task_info", "fetch_page", "submit_results", "deduplicate"],
        scenario="api-reliability",
    )


TASKS = [
    _api_task(0, "collect_all_items_dedup"),
    _api_task(1, "budgeted_pagination"),
    _api_task(2, "handle_empty_page"),
    _api_task(3, "totals_trap_submission"),
]
