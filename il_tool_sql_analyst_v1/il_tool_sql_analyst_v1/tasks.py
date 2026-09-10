"""tool-sql-analyst-v1 â€” schema-first SQL tool calls.

World: two small tables. The model must call get_schema before querying
(wildcard SELECT * is penalized as a distractor habit), run a minimal number
of precise queries, and report the aggregate. Teaches: schema awareness,
targeted projections, query-count efficiency.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

ORDERS = [
    {"id": 1, "region": "eu", "amount": 120, "status": "paid"},
    {"id": 2, "region": "us", "amount": 200, "status": "paid"},
    {"id": 3, "region": "eu", "amount": 80, "status": "refunded"},
    {"id": 4, "region": "us", "amount": 150, "status": "paid"},
    {"id": 5, "region": "apac", "amount": 300, "status": "paid"},
    {"id": 6, "region": "eu", "amount": 150, "status": "paid"},
]

CUSTOMERS = [
    {"id": 1, "name": "ada", "region": "eu", "vip": True},
    {"id": 2, "name": "grace", "region": "us", "vip": False},
    {"id": 3, "name": "linus", "region": "eu", "vip": False},
]


def _make_tools():
    def get_schema(state, args):
        state["schema_seen"] = True
        return "orders(id, region, amount, status); customers(id, name, region, vip)", (
            "orders(id, region, amount, status)\ncustomers(id, name, region, vip)"
        )

    def execute(state, args):
        sql = str(args.get("query", "")).lower()
        state.setdefault("queries", []).append(sql)
        if "select *" in sql:
            return None, "ERROR: SELECT * rejected by query policy â€” project explicit columns"
        if "region = 'eu'" in sql or 'region = "eu"' in sql or "region='eu'" in sql:
            paid = [r["amount"] for r in ORDERS if r["region"] == "eu" and r["status"] == "paid"]
            state["eu_paid_total"] = sum(paid)
            return sum(paid), f"rows: {paid} -> total {sum(paid)}"
        if "count" in sql and "vip" in sql:
            n = sum(1 for c in CUSTOMERS if c["vip"])
            state["vip_count"] = n
            return n, f"count: {n}"
        return None, "empty result (check your WHERE clause)"

    def drop(state, args):
        return None, "ERROR: destructive DDL blocked"

    return {
        "get_schema": ToolSpec("get_schema", "Introspect table schemas.", {}, get_schema),
        "execute_query": ToolSpec("execute_query", "Run a read-only SQL query.", {"query": "str"}, execute),
        "drop_table": ToolSpec("drop_table", "Drop a table.", {"table": "str"}, drop, distractor=True),
        "insert_row": ToolSpec("insert_row", "Insert rows.", {"table": "str", "row": "dict"}, drop, distractor=True),
    }


def _sql_task(idx, name, question, expected_tokens, expert):
    tools = _make_tools()

    def goal(state):
        return state.get("eu_paid_total") is not None or state.get("vip_count") is not None, "query executed"

    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=(
            "You operate a SQL analytics database through tools:\n"
            "- get_schema() -> table/column listing\n"
            "- execute_query(query) -> result rows (SELECT * is rejected by policy)\n\n"
            f"{question}\n\n"
            "Make your tool calls in <tool>{\"name\": ..., \"args\": {...}}</tool> blocks, "
            "then answer in <answer>...</answer>."
        ),
        tools=tools,
        init={},
        goal=goal,
        expert=expert,
        verify_answer=answer_contains(*expected_tokens),
        expert_answer=" ".join(expected_tokens),
        expected_concepts=["get_schema", "execute_query", "region"],
        scenario="sql",
    )


TASKS = [
    _sql_task(
        0, "eu_revenue_total",
        "What is the total paid amount for the 'eu' region? Query the orders table.",
        ["270"], [("get_schema", {}), ("execute_query", {"query": "SELECT region, SUM(amount) FROM orders WHERE region = 'eu' AND status = 'paid'"})],
    ),
    _sql_task(1, "vip_customer_count",
        "How many customers are VIP? Use an explicit COUNT projection.",
        ["1"], [("get_schema", {}), ("execute_query", {"query": "SELECT COUNT(*) FROM customers WHERE vip = true"})],
    ),
    _sql_task(2, "regional_breakdown",
        "Report the eu paid total AND the number of VIP customers in one session.",
        ["270", "1"],
        [("get_schema", {}), ("execute_query", {"query": "SELECT region, SUM(amount) FROM orders WHERE region = 'eu' AND status = 'paid'"}), ("execute_query", {"query": "SELECT COUNT(*) FROM customers WHERE vip = true"})],
    ),
    _sql_task(3, "schema_first_discipline",
        "Before querying, always introspect the schema. Then report the eu paid total.",
        ["270"], [("get_schema", {}), ("execute_query", {"query": "SELECT SUM(amount) FROM orders WHERE region = 'eu' AND status = 'paid'"})],
    ),
]

