"""
ToolGolf: Minimum tool calls to solve an agentic coding task.

Environment concept:
  The model is given a coding task and a set of available tools (with schemas).
  It must produce a sequence of tool calls that solves the task. The reward
  heavily favors FEWER tool calls — like golf, fewer strokes wins.

  This directly attacks the #1 frontier-model agentic waste pattern:
  "uncapped continuation loops" (GitHub found a single run with 244 turns,
  12.3M tokens) and "redundant tool calls" (40 unused MCP tools loaded
  every call).

  The model must:
    1. Choose the RIGHT tool (not just any tool that might work)
    2. Provide CORRECT arguments (wrong args = zero reward)
    3. Minimize the NUMBER of calls (each unnecessary call costs reward)
    4. Not retry failed calls (one shot per tool)

Why this environment is worth using for 10T-100T param models:
  Tool-call efficiency is the single biggest cost driver in agentic AI.
  A model that solves a task in 3 tool calls vs 15 tool calls saves 5x
  inference cost. At 100T parameters, this is the difference between
  $1M and $5M per training run. The environment is also compact: the
  tool schemas and tasks are small enough that a 1B model can attempt them,
  but the optimization landscape is rich enough that a 100T model can
  still find better solutions.

Problem types:
  - File operations (read, write, search, list)
  - Code execution (run, test, debug)
  - Git operations (status, diff, commit)
  - Shell commands (grep, find, sed)
  - Multi-step workflows (search → read → edit → test)

Verification:
  Each problem has:
    - A task description
    - Available tools with schemas
    - A "golden path" (the minimum set of tool calls that solves it)
    - A simulator that executes tool calls and returns results
  The verifier:
    1. Simulates each tool call in sequence
    2. Checks if the task is solved after the calls
    3. Counts the number of calls (fewer = better)

Reward design:
  - Task solved: gate (if not solved, reward = 0)
  - If solved: reward = 1.0 * (golden_calls / actual_calls)
    e.g., if golden path is 3 calls and model used 3, reward = 1.0
    if model used 6, reward = 0.5
  - Wrong tool args: that call fails (no penalty beyond wasted call)
  - Anti-pattern penalties for verbose reasoning between calls
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Tool simulator
# ---------------------------------------------------------------------------


class ToolSimulator:
    """
    Simulates a set of tools for agentic coding tasks.

    Each tool has a name, a schema (parameter names + types), and a handler
    that processes the call and returns a result.
    """

    def __init__(self, tools: dict[str, dict], initial_state: dict[str, Any]):
        """
        Args:
            tools: Dict of tool_name -> {"params": [...], "handler": callable}
            initial_state: Initial filesystem/state for the simulation.
        """
        self._tools = tools
        self._state = dict(initial_state)
        self._call_log: list[dict] = []

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @property
    def call_log(self) -> list[dict]:
        return self._call_log

    def call(self, tool_name: str, args: dict) -> dict:
        """Execute a tool call and return the result."""
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool = self._tools[tool_name]
        required_params = tool.get("params", [])

        # Check required params
        for param in required_params:
            if param not in args:
                return {"error": f"Missing required parameter: {param}"}

        # Execute the handler
        try:
            result = tool["handler"](self._state, args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        self._call_log.append({
            "tool": tool_name,
            "args": args,
            "result": result,
        })
        return result

    def reset(self, initial_state: dict[str, Any]):
        self._state = dict(initial_state)
        self._call_log.clear()


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------


def _make_file_search_problem(rng: random.Random) -> Problem:
    """
    Task: Find which file contains a specific function definition.
    Golden path: 1 call (grep/search)
    """
    files = {
        "src/auth.py": "def login(user, password):\n    ...\n\ndef logout(user):\n    ...",
        "src/models.py": "class User:\n    ...\n\nclass Post:\n    ...",
        "src/utils.py": "def format_date(dt):\n    ...\n\ndef parse_json(s):\n    ...",
        "src/api.py": "def get_users():\n    ...\n\ndef create_user(data):\n    ...",
    }

    target_function = rng.choice(["login", "format_date", "get_users", "logout"])
    target_file = None
    for fname, content in files.items():
        if f"def {target_function}" in content:
            target_file = fname
            break

    # Define tools
    def grep_handler(state, args):
        pattern = args.get("pattern", "")
        results = []
        for fname, content in state.get("files", {}).items():
            if pattern in content:
                results.append(fname)
        return {"matches": results}

    def read_file_handler(state, args):
        path = args.get("path", "")
        return {"content": state.get("files", {}).get(path, "File not found")}

    def list_files_handler(state, args):
        return {"files": list(state.get("files", {}).keys())}

    tools = {
        "grep": {"params": ["pattern"], "handler": grep_handler},
        "read_file": {"params": ["path"], "handler": read_file_handler},
        "list_files": {"params": [], "handler": list_files_handler},
    }

    tool_schemas = [
        {"name": "grep", "params": ["pattern"], "description": "Search for a pattern in all files"},
        {"name": "read_file", "params": ["path"], "description": "Read a file's contents"},
        {"name": "list_files", "params": [], "description": "List all files"},
    ]

    return Problem(
        id=f"toolgolf_search_{rng.randint(0, 99999)}",
        prompt=_format_prompt(
            task=f"Find which file contains the function `{target_function}`. Return the filename.",
            tools=tool_schemas,
        ),
        difficulty=0.2,
        metadata={
            "type": "toolgolf",
            "tools": tools,
            "initial_state": {"files": files},
            "golden_calls": [{"tool": "grep", "args": {"pattern": f"def {target_function}"}}],
            "golden_count": 1,
            "target_answer": target_file,
            "check_fn": "find_file",
            "target_function": target_function,
        },
        token_budget=500,
        source="generated",
    )


def _make_read_edit_test_problem(rng: random.Random) -> Problem:
    """
    Task: Read a file, identify a bug, fix it, run tests.
    Golden path: 3 calls (read_file → edit_file → run_tests)
    """
    buggy_code = "def add(a, b):\n    return a - b\n"  # bug: should be +
    fixed_code = "def add(a, b):\n    return a + b\n"

    files = {"src/calc.py": buggy_code}
    tests = [
        ({"a": 1, "b": 2}, 3),
        ({"a": 0, "b": 0}, 0),
        ({"a": -1, "b": 1}, 0),
    ]

    def read_file_handler(state, args):
        path = args.get("path", "")
        return {"content": state.get("files", {}).get(path, "File not found")}

    def edit_file_handler(state, args):
        path = args.get("path", "")
        content = args.get("content", "")
        if path in state.get("files", {}):
            state["files"][path] = content
            return {"status": "ok"}
        return {"error": "File not found"}

    def run_tests_handler(state, args):
        # Execute the code and run tests
        code = state.get("files", {}).get("src/calc.py", "")
        namespace = {}
        try:
            exec(code, namespace)
            func = namespace.get("add")
            if not func:
                return {"passed": 0, "total": len(tests), "error": "Function 'add' not found"}
            passed = 0
            for test_args, expected in tests:
                try:
                    result = func(**test_args)
                    if result == expected:
                        passed += 1
                except Exception:
                    pass
            return {"passed": passed, "total": len(tests)}
        except Exception as e:
            return {"passed": 0, "total": len(tests), "error": str(e)}

    tools = {
        "read_file": {"params": ["path"], "handler": read_file_handler},
        "edit_file": {"params": ["path", "content"], "handler": edit_file_handler},
        "run_tests": {"params": [], "handler": run_tests_handler},
    }

    tool_schemas = [
        {"name": "read_file", "params": ["path"], "description": "Read a file's contents"},
        {"name": "edit_file", "params": ["path", "content"], "description": "Write new content to a file"},
        {"name": "run_tests", "params": [], "description": "Run the test suite"},
    ]

    return Problem(
        id=f"toolgolf_edit_{rng.randint(0, 99999)}",
        prompt=_format_prompt(
            task="The function `add` in src/calc.py has a bug. Read the file, fix the bug, and run tests to verify. All tests must pass.",
            tools=tool_schemas,
        ),
        difficulty=0.4,
        metadata={
            "type": "toolgolf",
            "tools": tools,
            "initial_state": {"files": dict(files)},
            "golden_calls": [
                {"tool": "read_file", "args": {"path": "src/calc.py"}},
                {"tool": "edit_file", "args": {"path": "src/calc.py", "content": fixed_code}},
                {"tool": "run_tests", "args": {}},
            ],
            "golden_count": 3,
            "target_answer": "all_tests_pass",
            "check_fn": "tests_pass",
        },
        token_budget=800,
        source="generated",
    )


def _make_multi_file_search_problem(rng: random.Random) -> Problem:
    """
    Task: Find all files that import a specific module.
    Golden path: 1 call (grep)
    """
    files = {
        "src/main.py": "import os\nimport sys\nfrom utils import helper",
        "src/api.py": "import os\nfrom models import User",
        "src/test.py": "import sys\nimport os\nfrom api import get_users",
        "src/config.py": "import json\nimport os",
    }

    target_import = rng.choice(["os", "sys", "json"])
    expected_files = [f for f, c in files.items() if f"import {target_import}" in c]

    def grep_handler(state, args):
        pattern = args.get("pattern", "")
        results = []
        for fname, content in state.get("files", {}).items():
            if pattern in content:
                results.append(fname)
        return {"matches": results}

    def read_file_handler(state, args):
        path = args.get("path", "")
        return {"content": state.get("files", {}).get(path, "File not found")}

    def list_files_handler(state, args):
        return {"files": list(state.get("files", {}).keys())}

    tools = {
        "grep": {"params": ["pattern"], "handler": grep_handler},
        "read_file": {"params": ["path"], "handler": read_file_handler},
        "list_files": {"params": [], "handler": list_files_handler},
    }

    tool_schemas = [
        {"name": "grep", "params": ["pattern"], "description": "Search for a pattern in all files"},
        {"name": "read_file", "params": ["path"], "description": "Read a file's contents"},
        {"name": "list_files", "params": [], "description": "List all files"},
    ]

    return Problem(
        id=f"toolgolf_multi_{rng.randint(0, 99999)}",
        prompt=_format_prompt(
            task=f"Find ALL files that contain `import {target_import}`. Return the list of filenames.",
            tools=tool_schemas,
        ),
        difficulty=0.25,
        metadata={
            "type": "toolgolf",
            "tools": tools,
            "initial_state": {"files": files},
            "golden_calls": [{"tool": "grep", "args": {"pattern": f"import {target_import}"}}],
            "golden_count": 1,
            "target_answer": expected_files,
            "check_fn": "find_all_files",
        },
        token_budget=500,
        source="generated",
    )


def _make_git_workflow_problem(rng: random.Random) -> Problem:
    """
    Task: Check git status, see what changed, and commit with a message.
    Golden path: 2 calls (git_status → git_commit)
    """
    changed_files = ["src/main.py", "src/utils.py"]
    new_content = "def main():\n    pass\n"

    def git_status_handler(state, args):
        return {"modified": state.get("modified_files", []), "staged": state.get("staged_files", [])}

    def git_diff_handler(state, args):
        diffs = {}
        for f in state.get("modified_files", []):
            diffs[f] = f"+ {new_content}"
        return {"diffs": diffs}

    def git_commit_handler(state, args):
        message = args.get("message", "")
        if not message:
            return {"error": "Commit message required"}
        if not state.get("staged_files"):
            # Auto-stage modified files
            state["staged_files"] = state.get("modified_files", [])
        state["committed"] = True
        state["commit_message"] = message
        return {"status": "committed", "message": message}

    def git_add_handler(state, args):
        files_to_add = args.get("files", [])
        if not state.get("staged_files"):
            state["staged_files"] = []
        state["staged_files"].extend(files_to_add)
        return {"status": "staged", "files": files_to_add}

    tools = {
        "git_status": {"params": [], "handler": git_status_handler},
        "git_diff": {"params": [], "handler": git_diff_handler},
        "git_commit": {"params": ["message"], "handler": git_commit_handler},
        "git_add": {"params": ["files"], "handler": git_add_handler},
    }

    tool_schemas = [
        {"name": "git_status", "params": [], "description": "Show working tree status"},
        {"name": "git_diff", "params": [], "description": "Show unstaged changes"},
        {"name": "git_add", "params": ["files"], "description": "Stage files for commit"},
        {"name": "git_commit", "params": ["message"], "description": "Commit staged changes"},
    ]

    return Problem(
        id=f"toolgolf_git_{rng.randint(0, 99999)}",
        prompt=_format_prompt(
            task="Check what files have changed and commit them with a descriptive message.",
            tools=tool_schemas,
        ),
        difficulty=0.35,
        metadata={
            "type": "toolgolf",
            "tools": tools,
            "initial_state": {
                "modified_files": list(changed_files),
                "staged_files": [],
                "committed": False,
            },
            "golden_calls": [
                {"tool": "git_status", "args": {}},
                {"tool": "git_commit", "args": {"message": "Update main and utils"}},
            ],
            "golden_count": 2,
            "target_answer": "committed",
            "check_fn": "git_committed",
        },
        token_budget=600,
        source="generated",
    )


def _make_debug_workflow_problem(rng: random.Random) -> Problem:
    """
    Task: Read a file with a bug, run it to see the error, fix it, verify.
    Golden path: 3 calls (read_file → run_code → edit_file) or 4 with verify
    """
    buggy_code = "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\nprint(result)"
    fixed_code = "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\n\nresult = divide(10, 0)\nprint(result)"

    files = {"src/app.py": buggy_code}

    def read_file_handler(state, args):
        path = args.get("path", "")
        return {"content": state.get("files", {}).get(path, "File not found")}

    def run_code_handler(state, args):
        path = args.get("path", "src/app.py")
        code = state.get("files", {}).get(path, "")
        try:
            exec(code, {})
            return {"status": "ok", "output": "0"}
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    def edit_file_handler(state, args):
        path = args.get("path", "")
        content = args.get("content", "")
        if path in state.get("files", {}):
            state["files"][path] = content
            return {"status": "ok"}
        return {"error": "File not found"}

    tools = {
        "read_file": {"params": ["path"], "handler": read_file_handler},
        "run_code": {"params": [], "handler": run_code_handler},
        "edit_file": {"params": ["path", "content"], "handler": edit_file_handler},
    }

    tool_schemas = [
        {"name": "read_file", "params": ["path"], "description": "Read a file's contents"},
        {"name": "run_code", "params": [], "description": "Execute the Python file and return output/errors"},
        {"name": "edit_file", "params": ["path", "content"], "description": "Write new content to a file"},
    ]

    return Problem(
        id=f"toolgolf_debug_{rng.randint(0, 99999)}",
        prompt=_format_prompt(
            task="The file src/app.py has a runtime error. Read it, run it to see the error, fix it, and verify the fix works.",
            tools=tool_schemas,
        ),
        difficulty=0.5,
        metadata={
            "type": "toolgolf",
            "tools": tools,
            "initial_state": {"files": dict(files)},
            "golden_calls": [
                {"tool": "read_file", "args": {"path": "src/app.py"}},
                {"tool": "run_code", "args": {}},
                {"tool": "edit_file", "args": {"path": "src/app.py", "content": fixed_code}},
                {"tool": "run_code", "args": {}},
            ],
            "golden_count": 4,
            "target_answer": "code_runs",
            "check_fn": "code_runs_ok",
        },
        token_budget=900,
        source="generated",
    )


def _format_prompt(task: str, tools: list[dict]) -> str:
    """Format the problem prompt with task and tool schemas."""
    tool_descriptions = "\n".join(
        f"  - {t['name']}({', '.join(t['params'])}): {t['description']}"
        for t in tools
    )
    return (
        f"Task: {task}\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"Output your tool calls as a JSON array, one per line:\n"
        f'[{{"tool": "<name>", "args": {{<params>}}}}]\n\n'
        f"Rules:\n"
        f"  - Use the MINIMUM number of tool calls\n"
        f"  - Each call must have correct arguments\n"
        f"  - Do NOT include reasoning between calls — just the JSON array\n"
        f"  - Fewer calls = higher reward"
    )


def toolgolf_generator(seed: int) -> Problem:
    """Master generator for ToolGolf problems."""
    rng = random.Random(seed)
    generators = [
        _make_file_search_problem,
        _make_read_edit_test_problem,
        _make_multi_file_search_problem,
        _make_git_workflow_problem,
        _make_debug_workflow_problem,
    ]
    gen = rng.choice(generators)
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ToolGolfVerifier(Verifier):
    """
    Verifies a ToolGolf response by:
      1. Parsing the JSON array of tool calls
      2. Simulating each call against the tool simulator
      3. Checking if the task is solved
      4. Computing efficiency = golden_count / actual_count
    """

    def __init__(self, tools: dict, initial_state: dict, golden_count: int,
                 check_fn: str, target_answer: Any):
        super().__init__()
        self._tools = tools
        self._initial_state = initial_state
        self._golden_count = golden_count
        self._check_fn = check_fn
        self._target_answer = target_answer

    def verify(self, response: str) -> VerifierResult:
        # Parse the JSON array of tool calls
        calls = self._parse_tool_calls(response)
        if not calls:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No valid tool calls found in response",
            )

        # Simulate the calls
        sim = ToolSimulator(self._tools, self._initial_state)
        for call in calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            result = sim.call(tool_name, args)
            # If a call errors, continue (the model might work around it)

        # Check if the task is solved
        solved = self._check_solved(sim)

        if not solved:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Task not solved after {len(calls)} calls. Last state: {sim.state}",
            )

        # Compute efficiency
        actual_count = len(calls)
        efficiency = self._golden_count / max(actual_count, 1)
        efficiency = min(1.0, efficiency)  # cap at 1.0

        return VerifierResult(
            correct=True,
            score=efficiency,
            partial_credit={"calls_used": actual_count, "golden_count": self._golden_count, "efficiency": efficiency},
            diagnostics=f"Task solved in {actual_count} calls (golden: {self._golden_count}). Efficiency: {efficiency:.2f}",
            metadata={"calls": calls, "call_log": sim.call_log},
        )

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from the response."""
        # Try to find a JSON array
        json_match = re.search(r"\[[\s\S]*?\]", response)
        if json_match:
            try:
                calls = json.loads(json_match.group())
                if isinstance(calls, list):
                    return calls
            except json.JSONDecodeError:
                pass

        # Try line-by-line parsing
        calls = []
        for line in response.strip().split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    call = json.loads(line)
                    calls.append(call)
                except json.JSONDecodeError:
                    pass

        return calls

    def _check_solved(self, sim: ToolSimulator) -> bool:
        """Check if the task is solved based on the simulator state."""
        state = sim.state

        if self._check_fn == "find_file":
            # Check if any grep call found the target file
            for entry in sim.call_log:
                if entry["tool"] == "grep":
                    matches = entry["result"].get("matches", [])
                    if self._target_answer in matches:
                        return True
            return False

        elif self._check_fn == "find_all_files":
            # Check if grep found all expected files
            for entry in sim.call_log:
                if entry["tool"] == "grep":
                    matches = set(entry["result"].get("matches", []))
                    expected = set(self._target_answer)
                    return matches == expected
            return False

        elif self._check_fn == "tests_pass":
            # Check if run_tests was called and all tests passed
            for entry in sim.call_log:
                if entry["tool"] == "run_tests":
                    result = entry["result"]
                    return result.get("passed", 0) == result.get("total", 0)
            return False

        elif self._check_fn == "git_committed":
            return state.get("committed", False)

        elif self._check_fn == "code_runs_ok":
            # Check if run_code was called after the fix and succeeded
            found_success = False
            for entry in sim.call_log:
                if entry["tool"] == "run_code":
                    if entry["result"].get("status") == "ok":
                        found_success = True
            return found_success

        return False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ToolGolfEnv(BaseReasoningEnv):
    """
    ToolGolf environment: minimum tool calls to solve an agentic coding task.

    The model receives a task + tool schemas and must produce a JSON array
    of tool calls. Reward = task_solved * (golden_calls / actual_calls).
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
    ):
        if problems is None and problem_generator is None:
            problem_generator = toolgolf_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ToolGolfVerifier(
            tools=problem.metadata["tools"],
            initial_state=problem.metadata["initial_state"],
            golden_count=problem.metadata["golden_count"],
            check_fn=problem.metadata["check_fn"],
            target_answer=problem.metadata["target_answer"],
        )

    def _check_format(self, response: str) -> float:
        """Check for JSON array format."""
        if re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
