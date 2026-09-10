"""Smoke test: every tool-calling taskset loads, expert trajectories pass, distractors fail."""

import sys

sys.path.insert(0, r"E:\optimusstudio\iloptimus")

from iloptimus.core.grader import _load_module, _taskset_path, grade_response, build_prompt, get_num_tasks

TOOL_DOMAINS = {
    "tool-fs": "il_tool_fs_navigator_v1",
    "tool-sql": "il_tool_sql_analyst_v1",
    "tool-web": "il_tool_web_research_v1",
    "tool-booking": "il_tool_booking_flow_v1",
    "tool-pipeline": "il_tool_pipeline_v1",
    "tool-recovery": "il_tool_error_recovery_v1",
    "tool-distractor": "il_tool_distractor_v1",
    "tool-parallel": "il_tool_parallel_v1",
    "tool-interpreter": "il_tool_interpreter_v1",
    "tool-devops": "il_tool_devops_triage_v1",
    "tool-api": "il_tool_api_reliability_v1",
}


def expert_response(task):
    import json

    calls = "".join(
        f"<tool>{json.dumps({'name': tool, 'args': args})}</tool>\n" for tool, args in task.expert
    )
    return f"<reasoning>plan then verify</reasoning>\n{calls}\n<answer>{task.expert_answer}</answer>"


failures = []
for domain, pkg in TOOL_DOMAINS.items():
    tasks_mod = _load_module(f"{pkg}_tasks", str(_taskset_path(pkg, "tasks.py")))
    n = len(tasks_mod.TASKS)
    assert get_num_tasks(domain) == n, f"{domain}: num_tasks mismatch"
    for idx in range(n):
        task = tasks_mod.TASKS[idx]
        prompt = build_prompt(domain, idx)
        assert task.name in prompt, f"{domain}/{idx}: prompt missing task name"

        # 1. expert trajectory must score full correctness
        result = grade_response(domain, idx, expert_response(task))
        if result.correctness != 1.0:
            failures.append(f"{domain}[{idx}] {task.name}: expert not correct: {result.info}")

        # 2. distractor spam must not be rewarded
        bad = (
            expert_response(task)
            + "<tool>{\"name\": \"nonexistent_tool\", \"args\": {}}</tool>\n"
        )
        bad_result = grade_response(domain, idx, bad)
        if bad_result.score >= result.score:
            failures.append(f"{domain}[{idx}] {task.name}: errors not penalized")

print(f"domains: {len(TOOL_DOMAINS)}, tasks: {sum(len(_load_module(f'{p}_tasks', str(_taskset_path(p, 'tasks.py'))).TASKS) for p in TOOL_DOMAINS.values())}")
if failures:
    print("FAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL TOOL-CALLING TASKSETS PASS")
