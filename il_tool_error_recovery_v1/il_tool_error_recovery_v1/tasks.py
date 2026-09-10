"""tool-error-recovery-v1 â€” fault-injected APIs with retry budgets.

World: an HTTP API that fails the first N calls with 429/500. The model must
interpret the error observation, back off / retry or switch to the fallback
endpoint, and still deliver the result within a tight call budget. Teaches:
reading error payloads, correct retry semantics, budget discipline.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains


def _make_tools(fail_first: int, fail_code: str):
    counters = {"primary": 0, "fallback": 0}

    def fetch_primary(state, args):
        counters["primary"] += 1
        if counters["primary"] <= fail_first:
            state["retries"] = state.get("retries", 0) + 1
            return None, f"HTTP {fail_code} â€” {'rate limited, retry after backoff' if fail_code == '429' else 'internal server error, retry later'}"
        state["primary_ok"] = True
        return {"status": "ok", "payload": "report-v9"}, "payload: report-2026-09 (from primary)"

    def fetch_fallback(state, args):
        counters["fallback"] += 1
        state["used_fallback"] = True
        return {"payload": "report-2026-09"}, "payload: report-2026-09 (from fallback mirror)"

    def backoff(state, args):
        state["backed_off"] = True
        return "waited", "backoff window elapsed"

    def give_up(state, args):
        state["gave_up"] = True
        return None, "session aborted"

    return {
        "fetch_report": ToolSpec("fetch_report", "Fetch the report from the primary API.", {"report_id": "str"}, fetch_primary),
        "fetch_report_fallback": ToolSpec("fetch_report_fallback", "Fetch from the fallback endpoint (slower but resilient).", {"report_id": "str"}, fetch_fallback),
        "backoff": ToolSpec("backoff", "Wait out a retry window.", {"seconds": "int"}, backoff),
        "give_up": ToolSpec("give_up", "Abandon the task.", {}, give_up, distractor=True),
    }


def _recovery_task(idx, name, fail_first, fail_code, expected_source, extra_retry=False):
    tools = _make_tools(fail_first, fail_code)

    def goal(state):
        return state.get("primary_ok") is True or state.get("used_fallback") is True, f"state={ {k: v for k, v in state.items() if k != 'retries'} }"

    spec = (
        "You operate a flaky HTTP API through tools:\n"
        "- fetch_report(report_id) -> primary endpoint (may return HTTP 429/500)\n"
        "- fetch_report_fallback(report_id) -> resilient fallback endpoint\n"
        "- backoff(seconds) -> wait out a retry window\n\n"
        "ERROR POLICY: on 429, back off then retry. On 500, either retry once or use "
        "the fallback. Never give up. Keep total calls minimal.\n\n"
        "Task: retrieve report 'rep-2026-09' despite failures. Make tool calls in "
        "<tool>{...}</tool> blocks and answer with the payload name in <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=(
            [("fetch_report", {"report_id": "rep-2026-09"}), ("backoff", {"seconds": 5})]
            + ([("fetch_report", {"report_id": "rep-2026-09"}), ("backoff", {"seconds": 5})] if fail_first > 1 else [])
            + ([("fetch_report_fallback", {"report_id": "rep-2026-09"})] if expected_source == "fallback" else [("fetch_report", {"report_id": "rep-2026-09"})])
        ),
        verify_answer=answer_contains("report-2026-09"),
        expert_answer="payload: report-2026-09",
        expected_concepts=["fetch_report", "backoff", fail_code],
        scenario="error-recovery",
    )


TASKS = [
    _recovery_task(0, "retry_after_429", 1, "429", "primary"),
    _recovery_task(1, "retry_after_500", 1, "500", "primary"),
    _recovery_task(2, "double_failure_retry", 2, "429", "primary", extra_retry=True),
    _recovery_task(3, "fallback_after_500", 2, "500", "fallback"),
]
