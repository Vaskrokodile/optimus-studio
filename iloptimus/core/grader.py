"""Real grader — bypasses verifiers.v1 framework, calls scoring functions directly.

For each taskset type, this module:
1. Loads the task definitions (TASKS list) from the taskset package
2. Builds the prompt for each task (same format as the taskset.py INSTRUCTION)
3. Grades a model response using the same scoring logic (correctness + reasoning quality)
4. Runs sandbox verify.py scripts directly via subprocess (no runtime abstraction)

This is the bridge between the web pipeline runner and the IL taskset scoring logic.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Module loading (bypasses __init__.py which imports verifiers)
# ---------------------------------------------------------------------------

_LOADED: dict[str, object] = {}


def _load_module(name: str, path: str):
    """Load a Python module from a file path, bypassing package __init__.py."""
    cache_key = f"{name}:{path}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _LOADED[cache_key] = mod
    return mod


# Base paths
_TASKSETS_DIR = Path(__file__).parent.parent.parent / "il_coding_v1"
# Actually, the tasksets are siblings of the iloptimus package
_REPO_ROOT = Path(__file__).parent.parent.parent  # primeILtasks/


def _taskset_path(taskset_id: str, filename: str) -> Path:
    """Get path to a file within a taskset package."""
    package_name = taskset_id.replace("-", "_")
    package_spec = importlib.util.find_spec(package_name)
    if package_spec and package_spec.submodule_search_locations:
        packaged = Path(next(iter(package_spec.submodule_search_locations))) / filename
        if packaged.exists():
            return packaged
    repository_path = _REPO_ROOT / package_name / package_name / filename
    if repository_path.exists():
        return repository_path
    raise FileNotFoundError(f"Bundled taskset file not found: {package_name}/{filename}")


# ---------------------------------------------------------------------------
# Prompt builders (mirror the INSTRUCTION strings from each taskset.py)
# ---------------------------------------------------------------------------

_CODING_INSTRUCTION = (
    "Solve the coding task below. First, reason through the problem inside "
    "<reasoning>...</reasoning> tags — explain your approach, trace edge cases, "
    "and verify your solution mentally. Then provide your code inside "
    "<answer>```python\n...\n```</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)

_REASONING_INSTRUCTION = (
    "Solve the reasoning puzzle below. First, work through it inside "
    "<reasoning>...</reasoning> tags — show your deduction step by step, "
    "check your answer, and avoid generic filler. Then give your final answer "
    "inside <answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work.\n\n"
)

_AGENTIC_REASONING_INSTRUCTION = (
    "Solve the multi-step reasoning task below. This requires SUSTAINED "
    "reasoning — trace through each stage carefully, as later steps depend on "
    "earlier ones. Work inside <reasoning>...</reasoning> tags, showing each "
    "step and verifying your answer. Then give your final answer inside "
    "<answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: cover the key concepts, stay "
    "within budget, verify your work, and avoid generic filler.\n\n"
)

_AGENTIC_CODING_INSTRUCTION = (
    "You are given a multi-file codebase with a bug, missing feature, or "
    "refactoring task. First, reason through the code inside "
    "<reasoning>...</reasoning> tags — trace the call chain, identify the "
    "root cause, and verify your fix mentally. Then provide your fixed file(s) "
    "as code blocks tagged with the filename, like:\n"
    "```python:filename.py\n...your fixed code...\n```\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)

_HUMANEVAL_INSTRUCTION = (
    "Solve the coding task below. First, reason through the problem inside "
    "<reasoning>...</reasoning> tags — explain your approach, trace edge cases, "
    "and verify your solution mentally. Then provide your code inside "
    "<answer>```python\n...\n```</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)

_GSM8K_INSTRUCTION = (
    "Solve the math word problem below. First, work through it inside "
    "<reasoning>...</reasoning> tags — show your calculation step by step, "
    "check your answer, and avoid generic filler. Then give your final answer "
    "inside <answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, show "
    "the key calculations, and verify your work.\n\n"
)

CODE_BLOCK_RE = re.compile(r"```python:([^\n]+)\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Graded result
# ---------------------------------------------------------------------------


@dataclass
class GradedResult:
    score: float  # final IL score (0.0 to 1.0)
    correctness: float  # raw correctness (0.0 to 1.0)
    reasoning_quality: float  # reasoning quality (0.0 to 1.0)
    coverage: float = 0.0
    verification: float = 0.0
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sandbox runner (replaces verifiers Runtime)
# ---------------------------------------------------------------------------


def _run_sandbox(script_path: str, payload: dict, timeout: float) -> dict:
    """Run a verify.py script in a subprocess with a JSON payload.

    Writes payload to a temp file, runs the script, reads JSON output.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp") as f:
        json.dump(payload, f)
        payload_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, payload_path, str(timeout)],
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # extra margin for script overhead
        )
        out = (result.stdout or "").strip()
        lines = out.splitlines()
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        return {"error": (result.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    finally:
        # verify.py deletes the payload itself, but clean up just in case
        if os.path.exists(payload_path):
            try:
                os.unlink(payload_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Taskset graders
# ---------------------------------------------------------------------------


def grade_coding(response: str, task_idx: int) -> GradedResult:
    """Grade a coding task response."""
    tasks_mod = _load_module(
        "il_coding_tasks",
        str(_taskset_path("il_coding_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_coding_scoring",
        str(_taskset_path("il_coding_v1", "scoring.py")),
    )
    verify_script = str(_taskset_path("il_coding_v1", "verify.py"))

    task = tasks_mod.TASKS[task_idx]
    code = scoring_mod.extract_code(response)
    if not code.strip():
        return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0)

    # Run hidden tests in sandbox
    test_result = _run_sandbox(
        verify_script,
        {"code": code, "tests": task.tests},
        timeout=5.0,
    )
    correctness = test_result.get("pass_rate", 0.0)

    # Anti-laziness penalty
    laziness = scoring_mod.detect_laziness(code, task.required_params, task.required_constructs)
    if laziness.score > 0:
        correctness *= max(0.2, 1.0 - laziness.score * 0.8)

    # Reasoning quality shaping
    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={
            "laziness_reasons": laziness.reasons,
            "test_result": test_result,
        },
    )


def grade_reasoning(response: str, task_idx: int) -> GradedResult:
    """Grade a reasoning task response (no sandbox needed)."""
    tasks_mod = _load_module(
        "il_reasoning_tasks",
        str(_taskset_path("il_reasoning_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_reasoning_scoring",
        str(_taskset_path("il_reasoning_v1", "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    answer = tasks_mod._extract_answer_text(response)
    correct, info = task.verify(answer)
    correctness = 1.0 if correct else 0.0

    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={"verify_info": info},
    )


def grade_agentic_reasoning(response: str, task_idx: int) -> GradedResult:
    """Grade an agentic reasoning task response (no sandbox needed)."""
    tasks_mod = _load_module(
        "il_agentic_reasoning_tasks",
        str(_taskset_path("il_agentic_reasoning_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_agentic_reasoning_scoring",
        str(_taskset_path("il_agentic_reasoning_v1", "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    answer = tasks_mod._extract_answer_text(response)
    correct, info = task.verify(answer)
    correctness = 1.0 if correct else 0.0

    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={"verify_info": info},
    )


def grade_agentic_coding(response: str, task_idx: int) -> GradedResult:
    """Grade an agentic coding task response."""
    tasks_mod = _load_module(
        "il_agentic_coding_tasks",
        str(_taskset_path("il_agentic_coding_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_agentic_coding_scoring",
        str(_taskset_path("il_agentic_coding_v1", "scoring.py")),
    )
    verify_script = str(_taskset_path("il_agentic_coding_v1", "verify.py"))

    task = tasks_mod.TASKS[task_idx]

    # Extract code blocks with filenames
    fixes: dict[str, str] = {}
    for match in CODE_BLOCK_RE.finditer(response):
        fname = match.group(1).strip()
        code = match.group(2)
        fixes[fname] = code

    if not fixes:
        return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0)

    # Merge fixes into codebase
    merged = dict(task.codebase)
    merged.update(fixes)

    # Anti-laziness: check target files were actually changed
    laziness_reasons: list[str] = []
    for fname, code in fixes.items():
        if fname in task.target_files:
            if code.strip() == task.codebase.get(fname, "").strip():
                laziness_reasons.append(f"{fname}: unchanged (no-op fix)")
            elif not code.strip():
                laziness_reasons.append(f"{fname}: empty fix")

    # Run test harness in sandbox
    test_result = _run_sandbox(
        verify_script,
        {"files": merged, "harness": task.test_harness},
        timeout=10.0,
    )
    correctness = 1.0 if test_result.get("passed") else 0.0

    # Anti-laziness penalty
    if laziness_reasons:
        correctness *= 0.5

    # Structural laziness on changed files
    for fname, code in fixes.items():
        laz = scoring_mod.detect_laziness(code, [], [])
        if laz.score > 0:
            correctness *= max(0.2, 1.0 - laz.score * 0.5)
            laziness_reasons.extend(laz.reasons)

    # Reasoning quality shaping
    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={
            "laziness_reasons": laziness_reasons,
            "test_output": test_result.get("output", "")[-500:],
        },
    )


def grade_humaneval(response: str, task_idx: int) -> GradedResult:
    """Grade a HumanEval coding task response."""
    tasks_mod = _load_module(
        "humaneval_v1_tasks",
        str(_taskset_path("humaneval_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "humaneval_v1_scoring",
        str(_taskset_path("humaneval_v1", "scoring.py")),
    )
    verify_script = str(_taskset_path("humaneval_v1", "verify.py"))

    task = tasks_mod.TASKS[task_idx]
    code = scoring_mod.extract_code(response)
    if not code.strip():
        return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0)

    # Run hidden tests in sandbox
    test_result = _run_sandbox(
        verify_script,
        {"code": code, "tests": task.tests, "entry_point": task.entry_point},
        timeout=5.0,
    )
    correctness = test_result.get("pass_rate", 0.0)

    # Anti-laziness penalty
    laziness = scoring_mod.detect_laziness(code, task.required_params, task.required_constructs)
    if laziness.score > 0:
        correctness *= max(0.2, 1.0 - laziness.score * 0.8)

    # Reasoning quality shaping
    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={
            "laziness_reasons": laziness.reasons,
            "test_result": test_result,
        },
    )


_TOOL_DOMAINS = {
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


def grade_tool_calling(response: str, task_idx: int, domain: str) -> GradedResult:
    """Grade a tool-calling task: replay <tool_call> calls through the simulator."""
    pkg_id = _TOOL_DOMAINS[domain]
    tasks_mod = _load_module(
        f"{pkg_id}_tasks",
        str(_taskset_path(pkg_id, "tasks.py")),
    )
    scoring_mod = _load_module(
        f"{pkg_id}_scoring",
        str(_taskset_path(pkg_id, "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    score, breakdown = scoring_mod.score(task, response)

    return GradedResult(
        score=score,
        correctness=breakdown["correctness"],
        reasoning_quality=breakdown["tool_selection"],
        coverage=breakdown["efficiency"],
        verification=1.0 if not breakdown["errors"] else 0.0,
        info=breakdown,
    )


def grade_gsm8k(response: str, task_idx: int) -> GradedResult:
    """Grade a GSM8K math task response (no sandbox needed)."""
    tasks_mod = _load_module(
        "gsm8k_v1_tasks",
        str(_taskset_path("gsm8k_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "gsm8k_v1_scoring",
        str(_taskset_path("gsm8k_v1", "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    answer = tasks_mod._extract_answer_text(response)
    correct, info = task.verify(answer)
    correctness = 1.0 if correct else 0.0

    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={"verify_info": info},
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

# Map taskset domain -> (grader function, prompt builder, instruction string)
_GRADERS = {
    "coding": grade_coding,
    "reasoning": grade_reasoning,
    "agentic-reasoning": grade_agentic_reasoning,
    "agentic-coding": grade_agentic_coding,
    "humaneval": grade_humaneval,
    "gsm8k": grade_gsm8k,
}

_TOOL_INSTRUCTION = (
    "You are a tool-calling agent. Make tool calls inside "
    "<tool>{\"name\": \"tool_name\", \"args\": {...}}</tool> blocks, one block per "
    "call, in execution order. Choose the RIGHT tools with the RIGHT arguments in "
    "the RIGHT order — use as few calls as possible, never call distractor tools, "
    "and respect policy preconditions (e.g. look up before you modify). Simulated "
    "observations are deterministic; anticipate them. After your calls, give the "
    "final answer inside <answer>...</answer>.\n\n"
)

_INSTRUCTIONS = {
    "coding": _CODING_INSTRUCTION,
    "reasoning": _REASONING_INSTRUCTION,
    "agentic-reasoning": _AGENTIC_REASONING_INSTRUCTION,
    "agentic-coding": _AGENTIC_CODING_INSTRUCTION,
    "humaneval": _HUMANEVAL_INSTRUCTION,
    "gsm8k": _GSM8K_INSTRUCTION,
    **{domain: _TOOL_INSTRUCTION for domain in _TOOL_DOMAINS},
}


def grade_response(domain: str, task_idx: int, response: str) -> GradedResult:
    """Grade a model response for a given taskset domain and task index."""
    if domain.startswith("custom:"):
        from .environment_framework import score_task
        from .environments import get_environment

        environment = get_environment(domain.split(":", 1)[1])
        if not environment:
            raise ValueError(f"Unknown custom environment: {domain}")
        if environment.get("kind") == "state-machine":
            from .stateful_environments import simulate_response

            simulation = simulate_response(environment, task_idx, response)
            score = float(simulation["score"])
            return GradedResult(
                score=score,
                correctness=float(simulation["success"]),
                reasoning_quality=score,
                coverage=score,
                verification=float(simulation["success"]),
                info=simulation,
            )
        task = environment["tasks"][task_idx]
        metrics = score_task(task, response)
        reward = environment["reward"]
        score = metrics["correctness"] * reward["correctness"]
        score += metrics["reasoning_quality"] * reward["reasoning"]
        if metrics["correctness"]:
            score += reward["efficiency"]
        return GradedResult(
            score=min(1.0, score),
            correctness=metrics["correctness"],
            reasoning_quality=metrics["reasoning_quality"],
            coverage=metrics["coverage"],
            verification=metrics["verification"],
        )

    if domain.startswith("tool-"):
        return grade_tool_calling(response, task_idx, domain)

    grader = _GRADERS.get(domain)
    if not grader:
        raise ValueError(f"Unknown domain: {domain}")
    return grader(response, task_idx)


def build_prompt(domain: str, task_idx: int) -> str:
    """Build the prompt for a given taskset domain and task index."""
    if domain.startswith("custom:"):
        from .environments import get_environment

        environment = get_environment(domain.split(":", 1)[1])
        if not environment:
            raise ValueError(f"Unknown custom environment: {domain}")
        if environment.get("kind") == "state-machine":
            from .stateful_environments import build_stateful_prompt

            return build_stateful_prompt(environment, task_idx)
        task = environment["tasks"][task_idx]
        criteria = ", ".join(task.get("criteria", [])) or "correctness and clarity"
        return (
            f"Environment goal: {environment['goal']}\n\n"
            f"Complete the task below. Think inside <reasoning> tags and put the final response inside <answer> tags. "
            f"Success criteria: {criteria}.\n\n## Task: {task['name']}\n\n{task['prompt']}"
        )

    instruction = _INSTRUCTIONS.get(domain, "")

    if domain.startswith("tool-"):
        pkg_id = _TOOL_DOMAINS[domain]
        tasks_mod = _load_module(
            f"{pkg_id}_tasks",
            str(_taskset_path(pkg_id, "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}"

    if domain == "coding":
        tasks_mod = _load_module(
            "il_coding_tasks",
            str(_taskset_path("il_coding_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nSignature: `{task.signature}`"

    elif domain == "reasoning":
        tasks_mod = _load_module(
            "il_reasoning_tasks",
            str(_taskset_path("il_reasoning_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nAnswer format: {task.answer_format}."

    elif domain == "agentic-reasoning":
        tasks_mod = _load_module(
            "il_agentic_reasoning_tasks",
            str(_taskset_path("il_agentic_reasoning_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nAnswer format: {task.answer_format}."

    elif domain == "agentic-coding":
        tasks_mod = _load_module(
            "il_agentic_coding_tasks",
            str(_taskset_path("il_agentic_coding_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        codebase_str = "\n\n".join(f"### {fname}\n```python\n{content}```" for fname, content in task.codebase.items())
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\n## Codebase\n\n{codebase_str}"

    elif domain == "humaneval":
        tasks_mod = _load_module(
            "humaneval_v1_tasks",
            str(_taskset_path("humaneval_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nSignature: `{task.signature}`"

    elif domain == "gsm8k":
        tasks_mod = _load_module(
            "gsm8k_v1_tasks",
            str(_taskset_path("gsm8k_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nAnswer format: {task.answer_format}."

    raise ValueError(f"Unknown domain: {domain}")


def get_num_tasks(domain: str) -> int:
    """Get the number of tasks for a given domain."""
    if domain.startswith("custom:"):
        from .environments import get_environment

        environment = get_environment(domain.split(":", 1)[1])
        if not environment:
            raise ValueError(f"Unknown custom environment: {domain}")
        return len(environment["tasks"])

    pkg_map = {
        "coding": ("il_coding_tasks", "il_coding_v1", "tasks.py"),
        "reasoning": ("il_reasoning_tasks", "il_reasoning_v1", "tasks.py"),
        "agentic-reasoning": ("il_agentic_reasoning_tasks", "il_agentic_reasoning_v1", "tasks.py"),
        "agentic-coding": ("il_agentic_coding_tasks", "il_agentic_coding_v1", "tasks.py"),
        "humaneval": ("humaneval_v1_tasks", "humaneval_v1", "tasks.py"),
        "gsm8k": ("gsm8k_v1_tasks", "gsm8k_v1", "tasks.py"),
        **{domain: (f"{pkg}_tasks", pkg, "tasks.py") for domain, pkg in _TOOL_DOMAINS.items()},
    }
    mod_name, pkg_id, filename = pkg_map[domain]
    tasks_mod = _load_module(mod_name, str(_taskset_path(pkg_id, filename)))
    return len(tasks_mod.TASKS)
