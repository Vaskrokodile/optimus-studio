"""tool-pipeline-v1 â€” multi-tool sequential pipelines with value carrying.

World: an API chain where each call's output feeds the next call's arguments
(auth token -> dataset id -> record ids -> transformed payload -> submission).
Teaches: threading values between calls, dependency ordering, submitting the
derived value.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

RECORDS = {"ds-42": {"records": [3, 9, 12], "owner": "team-a"}}
PIPELINES = {
    "daily-sales": {"dataset": "ds-42", "transform": "sum", "submit_to": "ledger"},
    "user-count": {"dataset": "ds-42", "transform": "count", "submit_to": "metrics"},
}


def _make_tools():
    def authenticate(state, args):
        state["token"] = "tok-7788"
        return "tok-7788", "token issued: tok-7788"

    def list_datasets(state, args):
        if state.get("token") != "tok-7788":
            return None, "ERROR: 401 unauthorized â€” call authenticate first"
        state["datasets_seen"] = True
        return ["ds-42"], "datasets: ['ds-42']"

    def fetch_records(state, args):
        if state.get("token") != "tok-7788":
            return None, "ERROR: 401 unauthorized"
        ds = str(args.get("dataset_id", ""))
        if ds not in RECORDS:
            return None, f"ERROR: unknown dataset {ds}"
        state["records"] = RECORDS[ds]["records"]
        return RECORDS[ds]["records"], f"records: {RECORDS[ds]['records']}"

    def transform(state, args):
        records = state.get("records")
        if records is None:
            return None, "ERROR: no records loaded â€” fetch_records first"
        op = str(args.get("op", "sum"))
        value = sum(records) if op == "sum" else len(records)
        state["value"] = value
        return value, f"transform({op}) -> {value}"

    def submit(state, args):
        if "value" not in state:
            return None, "ERROR: nothing transformed to submit"
        if str(args.get("value")) != str(state["value"]):
            return None, f"ERROR: submitted value {args.get('value')} != computed {state['value']}"
        state["submitted"] = True
        return "SUB-881", "submission accepted: SUB-881"

    return {
        "authenticate": ToolSpec("authenticate", "Obtain an API token.", {}, authenticate),
        "list_datasets": ToolSpec("list_datasets", "List available datasets.", {}, list_datasets),
        "fetch_records": ToolSpec("fetch_records", "Fetch records for a dataset id.", {"dataset_id": "str"}, fetch_records),
        "transform": ToolSpec("transform", "Apply an operation to loaded records.", {"op": "str"}, transform),
        "submit": ToolSpec("submit", "Submit the computed value.", {"value": "int"}, submit),
        "delete_dataset": ToolSpec("delete_dataset", "Delete a dataset.", {"dataset_id": "str"}, lambda s, a: (None, "blocked"), distractor=True),
    }


def _pipeline_task(idx, name, pipeline, op, expected_value):
    tools = _make_tools()

    def goal(state):
        return state.get("value") == expected_value and state.get("submitted") is not None, f"value={state.get('value')}"

    spec = (
        "You operate a data API through tools:\n"
        "- authenticate() -> token\n"
        "- list_datasets() -> dataset ids (requires auth)\n"
        "- fetch_records(dataset_id) -> records (requires auth)\n"
        "- transform(op) -> computed value (requires loaded records)\n"
        "- submit(value) -> submission id (value must match the transform output)\n\n"
        f"Pipeline '{pipeline}' uses transform '{op}'. Thread each call's output into the "
        "next call's arguments. Make your tool calls in <tool>{...}</tool> blocks and give "
        "the submission confirmation in <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=[
            ("authenticate", {}),
            ("list_datasets", {}),
            ("fetch_records", {"dataset_id": "ds-42"}),
            ("transform", {"op": op}),
            ("submit", {"value": expected_value}),
        ],
        verify_answer=answer_contains(str(expected_value)),
        expert_answer=f"submission accepted, value {expected_value}",
        expected_concepts=["authenticate", "list_datasets", "fetch_records", "transform", "submit"],
        scenario="pipeline",
    )


TASKS = [
    _pipeline_task(0, "daily_sales_sum", "daily-sales", "sum", 24),
    _pipeline_task(1, "user_count_report", "user-count", "count", 3),
    _pipeline_task(2, "sales_pipeline_strict_auth", "daily-sales", "sum", 24),
    _pipeline_task(3, "count_pipeline_minimal_calls", "user-count", "count", 3),
]
