"""tool-interpreter-v1 â€” code interpreter with stdout/stderr feedback.

World: a deterministic mini-interpreter. The model runs Python snippets, must
read errors (e.g. ZeroDivisionError) and fix course, keep executions minimal,
and avoid distractor tools (install_package, shell). Teaches: execute-to-verify,
error reading, round economy.
"""

from __future__ import annotations

from il_toolcalling_core import ToolCallingTask, ToolSpec, answer_contains


def _make_tools():
    def python_exec(state, args):
        code = str(args.get("code", ""))
        state["runs"] = state.get("runs", 0) + 1
        if "/ 0" in code or "1/0" in code or "/ count" in code:
            state["ok"] = True
            return None, "Traceback: ZeroDivisionError â€” division by zero"
        if "fib" in code:
            state["ok"] = True
            return "[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]", "stdout: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]"
        if "sum(" in code:
            state["ok"] = True
            state["value"] = 385
            return 385, "385"
        if "*" in code:
            try:
                left, right = code.split("*")
                value = int("".join(ch for ch in left if ch.isdigit()) or 0) * int("".join(c for c in right if c.isdigit()))
            except Exception:
                return None, "SyntaxError"
            state["ok"] = True
            state["value"] = value
            return value, str(value)
        return "", "(no output)"

    def install_package(state, args):
        state["installed"] = True
        return "ok", "package installed (unnecessary for this task)"

    def open_shell(state, args):
        return None, "ERROR: shell access is disabled in the interpreter sandbox"

    return {
        "python_exec": ToolSpec("python_exec", "Run Python code; returns stdout/stderr.", {"code": "str"}, python_exec),
        "install_package": ToolSpec("install_package", "pip install a package.", {"package": "str"}, lambda s, a: (None, "installed (unnecessary)"), distractor=True),
        "open_shell": ToolSpec("open_shell", "Open a raw shell.", {"cmd": "str"}, lambda s, a: (None, "ERROR: shell disabled"), distractor=True),
    }


def _interpreter_task(idx, name, prompt_body, expected_tokens, expert_code):
    tools = _make_tools()

    def goal(state):
        return state.get("ok") is True, f"runs={state.get('runs', 0)}"

    spec = (
        "You have a code interpreter:\n"
        "- python_exec(code) -> stdout / stderr\n\n"
        "Use it to verify computations before answering. Read errors carefully and "
        "name the exception type when asked. Do NOT install packages or open shells. "
        "Keep the number of executions minimal.\n\n"
        f"Task: {prompt_body} Put the final result in <answer>."
    )
    return ToolCallingTask(
        idx=idx,
        name=name,
        spec=spec,
        tools=tools,
        init={},
        goal=goal,
        expert=[("python_exec", {"code": expert_code})],
        verify_answer=answer_contains(*expected_tokens),
        expert_answer=" ".join(expected_tokens),
        expected_concepts=["python_exec", "verify"],
        scenario="code-interpreter",
    )


TASKS = [
    _interpreter_task(
        0, "sum_of_squares_1_to_10",
        "Compute 1Â² + 2Â² + ... + 10Â² with the interpreter and report the value.",
        ["385"], "print(sum(i * i for i in range(1, 11)))",
    ),
    _interpreter_task(
        1, "verify_fibonacci_sequence",
        "Print the first 10 Fibonacci numbers with the interpreter and report the 10th value.",
        ["34"], "fib = [1, 1]\nfor _ in range(8): fib.append(fib[-1] + fib[-2])\nprint(fib)",
    ),
    _interpreter_task(
        2, "diagnose_zero_division",
        "The snippet `total = 100; count = 0; avg = total / count` fails. Execute it, read the traceback, and name the exception type in your answer.",
        ["zerodivisionerror"], "print(100 / 0)",
    ),
    _interpreter_task(
        3, "minimal_execution_product",
        "Compute 17 * 23 with a single interpreter call and report the product.",
        ["391"], "print(17 * 23)",
    ),
]
