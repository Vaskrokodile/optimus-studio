"""tool-parallel-batching-v1 â€” parallel vs sequential call discipline.

BFCL-inspired: independent lookups should be issued together (no artificial
serialization), while dependent calls MUST wait for their inputs. The engine
detects dependency violations: calling a consumer before its producer.
Teaches: dependency analysis, batching independent reads, ordering writes.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

USERS = {"u-1": {"name": "ada", "tier": "pro"}, "u-2": {"name": "grace", "tier": "free"}}
ORDERS = {"u-1": ["o-11", "o-12"], "u-2": ["o-21"]}
INVENTORY = {"sku-a": 5, "sku-b": 0}


def _make_tools():
    def get_user(state, args):
        uid = str(args.get("user_id", ""))
        if uid not in USERS:
            return None, f"ERROR: unknown user {uid}"
        state["user"] = uid
        return USERS[uid], f"user {uid}: {USERS[uid]}"

    def get_orders(state, args):
        uid = str(args.get("user_id", ""))
        if uid not in ORDERS:
            return None, f"ERROR: unknown user {uid}"
        state["orders"] = ORDERS[uid]
        return ORDERS[uid], f"orders for {uid}: {ORDERS[uid]}"

    def get_stock(state, args):
        sku = str(args.get("sku", ""))
        if sku not in INVENTORY:
            return None, f"ERROR: unknown sku {sku}"
        state["stock"] = INVENTORY[sku]
        return INVENTORY[sku], f"{sku} stock: {INVENTORY[sku]}"

    def submit_report(state, args):
        needed = ("user", "orders", "stock")
        if not all(k in state for k in needed):
            missing = [k for k in needed if k not in state]
            return None, f"ERROR: report incomplete â€” missing {missing}"
        state["submitted"] = True
        return "ok", "report submitted"

    return {
        "get_user": ToolSpec("get_user", "Fetch a user profile.", {"user_id": "str"}, get_user),
        "get_orders": ToolSpec("get_orders", "List orders for a user.", {"user_id": "str"}, get_orders),
        "get_inventory": ToolSpec("get_inventory", "Check stock for a sku.", {"sku": "str"}, get_stock),
        "submit_report": ToolSpec("submit_report", "Submit the consolidated report.", {"summary": "str"}, submit_report),
    }


def _parallel_task(idx, name, user_id, sku):
    tools = _make_tools()

    def goal(state):
        return all(k in state for k in ("user", "orders", "stock")) and state.get("submitted"), f"state keys={sorted(state)}"

    spec = (
        "You operate a reporting API through tools:\n"
        "- get_user(user_id) -> profile\n"
        "- get_orders(user_id) -> order ids\n"
        "- get_inventory(sku) -> stock count\n"
        "- submit_report(summary) -> submission id (only after all data gathered)\n\n"
        "EFFICIENCY POLICY: get_user, get_orders and get_inventory are INDEPENDENT â€” "
        "issue them together before submitting. submit_report fails if any data is "
        "missing. Answer with the user tier and stock count in <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=[
            ("get_user", {"user_id": user_id}),
            ("get_orders", {"user_id": user_id}),
            ("get_inventory", {"sku": sku}),
            ("submit_report", {"summary": "done"}),
        ],
        verify_answer=answer_contains(user_id, sku),
        expert_answer=f"user {user_id} tier reported, {sku} stock checked",
        expected_concepts=["get_user", "get_orders", "get_inventory", "submit_report"],
        scenario="parallel-batching",
    )


TASKS = [
    _parallel_task(0, "batch_independent_reads", "u-1", "sku-a"),
    _parallel_task(1, "batch_then_submit", "u-2", "sku-b"),
    _parallel_task(2, "report_pro_user", "u-1", "sku-a"),
    _parallel_task(3, "report_free_user", "u-2", "sku-a"),
]
