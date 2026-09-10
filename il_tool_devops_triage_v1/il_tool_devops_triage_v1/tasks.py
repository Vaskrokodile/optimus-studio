"""tool-devops-incidents-v1 â€” long-horizon stateful incident triage.

World: a production service mid-incident. The model must triage in the right
order: check health -> read logs -> identify bad deploy -> roll back (first
attempt fails, must retry) -> verify recovery -> page on-call only if still
unhealthy. DB-state-diff reward in the spirit of tau-bench / E-Bench.
Teaches: diagnosis before action, verification after mutation, escalation
discipline.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains

LOGS = [
    "12:00:01 INFO  deploy v2.4.1 started",
    "12:00:04 ERROR /api/checkout: 500 spike after deploy v2.4.1",
    "12:00:05 ERROR error rate 42% (baseline 0.1%)",
]


def _make_tools():
    def check_health(state, args):
        healthy = state.get("rolled_back") == "verified"
        state["health_checked"] = True
        return ("healthy" if healthy else "degraded"), f"service status: {'healthy' if healthy else 'degraded (error rate 41%)'}"

    def read_logs(state, args):
        state["logs_read"] = True
        return LOGS, "\n".join(LOGS)

    def rollback(state, args):
        if not state.get("logs_read"):
            return None, "ERROR: inspect logs before rolling back (triage policy)"
        if state.get("rollback_attempts", 0) == 0:
            state["rollback_attempts"] = 1
            return None, "ERROR: rollback job timeout â€” retry the rollback"
        state["rolled_back"] = True
        return "ok", "deploy v2.4.0 restored"

    def verify_deploy(state, args):
        if not state.get("rolled_back"):
            return None, "ERROR: nothing to verify â€” rollback not applied"
        if state.get("hard_mode"):
            return "degraded", "post-rollback check: error rate still 38% â€” NOT healthy"
        state["verified"] = True
        return "healthy", "post-rollback check: error rate 0.1% â€” healthy"

    def page_oncall(state, args):
        state["paged"] = True
        return "paged", "on-call engineer paged"

    def restart_cluster(state, args):
        state["restarted"] = True
        return None, "ERROR: restart refused â€” would lose in-flight requests; use rollback"

    return {
        "check_health": ToolSpec("check_health", "Check service health.", {}, check_health),
        "read_logs": ToolSpec("read_logs", "Read recent service logs.", {"lines": "int"}, read_logs),
        "rollback_deploy": ToolSpec("rollback_deploy", "Roll back to the previous deploy.", {"service": "str"}, rollback, destructive=True),
        "verify_deploy": ToolSpec("verify_deploy", "Verify service health after a rollback.", {}, verify_deploy),
        "page_oncall": ToolSpec("page_oncall", "Page the on-call engineer.", {"reason": "str"}, page_oncall),
        "restart_cluster": ToolSpec("restart_cluster", "Restart the whole cluster.", {}, lambda s, a: (None, "blocked: too destructive"), distractor=True),
        "delete_logs": ToolSpec("delete_logs", "Purge logs.", {}, lambda s, a: (None, "blocked"), distractor=True),
    }


def _incident_task(idx, name, expect_page):
    tools = _make_tools()

    def goal(state):
        if expect_page:
            return state.get("paged") is True, f"paged={state.get('paged')}"
        return state.get("verified") is True, f"verified={state.get('verified')}"

    spec = (
        "You are the on-call agent for a production incident. Tools:\n"
        "- check_health() -> service status\n"
        "- read_logs() -> recent log lines\n"
        "- rollback_deploy() -> revert the last deploy (may fail; retry once)\n"
        "- verify_deploy() -> post-fix health check\n"
        "- page_oncall(reason) -> escalate to a human\n\n"
        "TRIAGE POLICY: diagnose before acting (health + logs), fix, then VERIFY the "
        "fix. Page on-call only if the service is still unhealthy after the fix.\n\n"
        "Task: the checkout service is throwing 500s. Triage and resolve the incident. "
        "Answer with the final service status in <answer>."
    )
    expert = [
        ("check_health", {}),
        ("read_logs", {"lines": 50}),
        ("rollback_deploy", {"service": "checkout"}),
        ("rollback_deploy", {"service": "checkout"}),
        ("verify_deploy", {}),
    ]
    if expect_page:
        expert.append(("page_oncall", {"reason": "still degraded after rollback"}))
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={"hard_mode": expect_page},
        goal=goal,
        expert=expert,
        verify_answer=answer_contains("healthy" if not expect_page else "paged"),
        expert_answer="service healthy after rollback" if not expect_page else "paged on-call, still degraded",
        expected_concepts=["check_health", "read_logs", "rollback_deploy", "verify_deploy"],
        scenario="devops-incident",
    )


TASKS = [
    _incident_task(0, "rollback_recovers_service", False),
    _incident_task(1, "rollback_then_page_oncall", True),
    _incident_task(2, "diagnose_before_rollback", False),
    _incident_task(3, "verify_after_fix", False),
]
