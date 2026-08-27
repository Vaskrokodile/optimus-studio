"""
ErrorRecovery: One-shot minimal fix for a runtime error.

Environment concept:
  The model is given a piece of code and a runtime error (traceback or
  error message). It must produce the MINIMAL fix that resolves the error
  without changing anything else.

  This trains the model to:
    1. Diagnose the root cause from the error message (not the symptom)
    2. Fix only what's broken (not refactor surrounding code)
    3. Not introduce new errors
    4. Not add unnecessary error handling (over-engineering)

  This directly attacks the frontier-model pattern of "fixing" errors by
  wrapping everything in try/except or adding defensive checks everywhere,
  when a single line change would suffice. It also trains against
  "fix cascade" — where each fix introduces a new error, leading to
  10+ round trips.

Why this environment is worth using for 10T-100T param models:
  Error recovery is the most common agentic coding loop. A 100T model
  that fixes errors in one shot (vs 5 round trips) saves 5x inference
  cost per debugging cycle. At scale, this is the difference between
  profitable and unprofitable agentic AI. The environment is compact
  (small code snippets, small fixes) but the optimization landscape is
  deep (the model must understand the code, the error, and the minimal
  fix surface).

Problem types:
  - NameError: undefined variable
  - TypeError: wrong argument type
  - IndexError: out of bounds
  - KeyError: missing dict key
  - AttributeError: wrong method/attribute
  - ZeroDivisionError: division by zero
  - SyntaxError: syntax issues
  - ValueError: invalid value

Verification:
  Each problem has:
    - The buggy code
    - The error message/traceback
    - The expected fixed code (or a test that must pass)
    - The "minimal fix" (the exact lines that should change)
  The verifier:
    1. Executes the fixed code
    2. Checks if the error is resolved (code runs without the original error)
    3. Checks if the fix is minimal (diff size vs the golden fix)
    4. Runs any test cases

Reward design:
  - Error resolved: gate (if the error persists, reward = 0)
  - If resolved:
    - Minimality: reward = golden_diff_size / actual_diff_size
      (capped at 1.0 — if the model's fix is smaller than golden, full reward)
    - No new errors: if the fix introduces a new error, penalty
  - Anti-pattern penalties for verbose explanations
"""

from __future__ import annotations

import difflib
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


def _make_name_error_problem(rng: random.Random) -> Problem:
    """Generate a NameError problem."""
    templates = [
        {
            "buggy": "def greet(name):\n    return f'Hello, {nme}'\n",
            "error": "NameError: name 'nme' is not defined",
            "fixed": "def greet(name):\n    return f'Hello, {name}'\n",
            "tests": [("Alice", "Hello, Alice"), ("Bob", "Hello, Bob")],
        },
        {
            "buggy": "def calculate(x):\n    result = x * 2\n    return resul\n",
            "error": "NameError: name 'resul' is not defined",
            "fixed": "def calculate(x):\n    result = x * 2\n    return result\n",
            "tests": [(5, 10), (0, 0), (-3, -6)],
        },
        {
            "buggy": "def get_first(lst):\n    if lst:\n        return frst\n    return None\n",
            "error": "NameError: name 'frst' is not defined",
            "fixed": "def get_first(lst):\n    if lst:\n        return lst[0]\n    return None\n",
            "tests": [([1, 2, 3], 1), ([], None), (["a"], "a")],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "name_error")


def _make_type_error_problem(rng: random.Random) -> Problem:
    """Generate a TypeError problem."""
    templates = [
        {
            "buggy": "def add(a, b):\n    return a + b\n\nresult = add('hello', 5)\n",
            "error": "TypeError: can only concatenate str (not \"int\") to str",
            "fixed": "def add(a, b):\n    return a + b\n\nresult = add('hello', '5')\n",
            "tests": [],
        },
        {
            "buggy": "def repeat(s, n):\n    return s * n\n\nprint(repeat('ab', 3.5))\n",
            "error": "TypeError: can't multiply sequence by non-int of type 'float'",
            "fixed": "def repeat(s, n):\n    return s * int(n)\n\nprint(repeat('ab', 3))\n",
            "tests": [],
        },
        {
            "buggy": "def count_items(items):\n    return len(items.split(','))\n\ncount_items(123)\n",
            "error": "TypeError: 'int' object has no attribute 'split'",
            "fixed": "def count_items(items):\n    return len(str(items).split(','))\n\ncount_items('1,2,3')\n",
            "tests": [],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "type_error")


def _make_index_error_problem(rng: random.Random) -> Problem:
    """Generate an IndexError problem."""
    templates = [
        {
            "buggy": "def get_third(lst):\n    return lst[3]\n",
            "error": "IndexError: list index out of range",
            "fixed": "def get_third(lst):\n    return lst[2] if len(lst) > 2 else None\n",
            "tests": [([1, 2, 3], 3), ([1], None), ([], None)],
        },
        {
            "buggy": "def last_element(data):\n    return data[len(data)]\n",
            "error": "IndexError: list index out of range",
            "fixed": "def last_element(data):\n    return data[len(data) - 1] if data else None\n",
            "tests": [([1, 2, 3], 3), ([5], 5), ([], None)],
        },
        {
            "buggy": "def get_pair(coords):\n    x = coords[0]\n    y = coords[2]\n    return (x, y)\n",
            "error": "IndexError: list index out of range",
            "fixed": "def get_pair(coords):\n    x = coords[0]\n    y = coords[1]\n    return (x, y)\n",
            "tests": [([1, 2], (1, 2)), ([0, 0], (0, 0))],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "index_error")


def _make_key_error_problem(rng: random.Random) -> Problem:
    """Generate a KeyError problem."""
    templates = [
        {
            "buggy": "def get_age(person):\n    return person['age']\n",
            "error": "KeyError: 'age'",
            "fixed": "def get_age(person):\n    return person.get('age', None)\n",
            "tests": [({"age": 25}, 25), ({}, None), ({"name": "Bob"}, None)],
        },
        {
            "buggy": "def get_config(key):\n    return config[key]\n\nconfig = {'host': 'localhost', 'port': 8080}\nprint(get_config('debug'))\n",
            "error": "KeyError: 'debug'",
            "fixed": "def get_config(key):\n    return config.get(key, False)\n\nconfig = {'host': 'localhost', 'port': 8080}\nprint(get_config('debug'))\n",
            "tests": [],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "key_error")


def _make_attribute_error_problem(rng: random.Random) -> Problem:
    """Generate an AttributeError problem."""
    templates = [
        {
            "buggy": "def get_length(s):\n    return s.lenght()\n",
            "error": "AttributeError: 'str' object has no attribute 'lenght'",
            "fixed": "def get_length(s):\n    return len(s)\n",
            "tests": [("hello", 5), ("", 0), ("a", 1)],
        },
        {
            "buggy": "def upper_case(s):\n    return s.toUpper()\n",
            "error": "AttributeError: 'str' object has no attribute 'toUpper'",
            "fixed": "def upper_case(s):\n    return s.upper()\n",
            "tests": [("hello", "HELLO"), ("", ""), ("AbC", "ABC")],
        },
        {
            "buggy": "def append_item(lst, item):\n    lst.add(item)\n    return lst\n",
            "error": "AttributeError: 'list' object has no attribute 'add'",
            "fixed": "def append_item(lst, item):\n    lst.append(item)\n    return lst\n",
            "tests": [([1, 2], 3, [1, 2, 3]), ([], 0, [0])],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "attribute_error")


def _make_zero_division_problem(rng: random.Random) -> Problem:
    """Generate a ZeroDivisionError problem."""
    templates = [
        {
            "buggy": "def safe_divide(a, b):\n    return a / b\n",
            "error": "ZeroDivisionError: division by zero",
            "fixed": "def safe_divide(a, b):\n    return a / b if b != 0 else 0\n",
            "tests": [(10, 2, 5), (10, 0, 0), (0, 5, 0)],
        },
        {
            "buggy": "def percentage(part, total):\n    return (part / total) * 100\n",
            "error": "ZeroDivisionError: division by zero",
            "fixed": "def percentage(part, total):\n    return (part / total) * 100 if total != 0 else 0\n",
            "tests": [(50, 100, 50.0), (0, 0, 0), (25, 50, 50.0)],
        },
        {
            "buggy": "def average(numbers):\n    return sum(numbers) / len(numbers)\n",
            "error": "ZeroDivisionError: division by zero",
            "fixed": "def average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0\n",
            "tests": [([1, 2, 3], 2.0), ([], 0), ([5], 5.0)],
        },
    ]
    t = rng.choice(templates)
    return _build_problem(t, rng, "zero_division")


def _build_problem(template: dict, rng: random.Random, error_type: str) -> Problem:
    """Build a Problem from a template dict."""
    # Compute the minimal diff size
    buggy_lines = template["buggy"].split("\n")
    fixed_lines = template["fixed"].split("\n")
    diff = list(difflib.unified_diff(buggy_lines, fixed_lines, lineterm=""))
    changed_lines = sum(1 for l in diff if l.startswith("+") or l.startswith("-"))
    golden_diff_size = max(1, changed_lines // 2)  # approximate

    difficulty_map = {
        "name_error": 0.15, "type_error": 0.25, "index_error": 0.2,
        "key_error": 0.2, "attribute_error": 0.2, "zero_division": 0.25,
    }
    difficulty = difficulty_map.get(error_type, 0.2) + 0.1 * rng.random()

    return Problem(
        id=f"error_recovery_{error_type}_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["buggy"], template["error"]),
        difficulty=difficulty,
        metadata={
            "type": "error_recovery",
            "error_type": error_type,
            "buggy_code": template["buggy"],
            "fixed_code": template["fixed"],
            "error_message": template["error"],
            "tests": template["tests"],
            "golden_diff_size": golden_diff_size,
        },
        token_budget=500,
        source="generated",
    )


def _format_prompt(buggy_code: str, error_message: str) -> str:
    """Format the problem prompt."""
    return (
        f"The following code produces a runtime error:\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"Error: {error_message}\n\n"
        f"Provide the MINIMAL fix. Output only the fixed code in a ```python block.\n"
        f"Rules:\n"
        f"  - Fix ONLY the error — do not refactor or add features\n"
        f"  - Do not add unnecessary try/except or defensive checks\n"
        f"  - Do not explain — just output the fixed code\n"
        f"  - Smaller diffs = higher reward"
    )


def error_recovery_generator(seed: int) -> Problem:
    """Master generator for ErrorRecovery problems."""
    rng = random.Random(seed)
    generators = [
        _make_name_error_problem,
        _make_type_error_problem,
        _make_index_error_problem,
        _make_key_error_problem,
        _make_attribute_error_problem,
        _make_zero_division_problem,
    ]
    gen = rng.choice(generators)
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ErrorRecoveryVerifier(Verifier):
    """
    Verifies an ErrorRecovery response by:
      1. Executing the fixed code
      2. Checking if the original error is resolved
      3. Computing minimality (diff size vs golden)
      4. Running test cases
    """

    def __init__(self, buggy_code: str, fixed_code: str, error_message: str,
                 tests: list, golden_diff_size: int):
        super().__init__()
        self._buggy_code = buggy_code
        self._fixed_code = fixed_code
        self._error_message = error_message
        self._tests = tests
        self._golden_diff_size = golden_diff_size

    def verify(self, response: str) -> VerifierResult:
        # Extract code from ```python block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)

        if not code_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No code block found in response",
            )

        submitted_code = code_match.group(1).strip()

        # Execute the submitted code
        namespace: dict[str, Any] = {}
        try:
            exec(submitted_code, namespace)
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            # Check if it's the SAME error as the original
            if type(e).__name__ in self._error_message:
                return VerifierResult(
                    correct=False,
                    score=0.0,
                    diagnostics=f"Original error persists: {error_str}",
                )
            else:
                # A NEW error was introduced
                return VerifierResult(
                    correct=False,
                    score=0.0,
                    diagnostics=f"New error introduced: {error_str}",
                )

        # Code runs without error — check if it's actually fixed
        # (not just commented out or empty)
        if len(submitted_code.strip()) < 10:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="Fixed code is too short — likely just removed the code",
            )

        # Run test cases if any
        test_pass_rate = 1.0
        if self._tests:
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_") and name != "__builtins__":
                    if hasattr(obj, "__module__") and obj.__module__ in ("builtins",):
                        continue
                    func = obj
                    break

            if func:
                passed = 0
                total = len(self._tests)
                for test in self._tests:
                    try:
                        if len(test) == 2:
                            args, expected = test
                            result = func(args) if not isinstance(args, tuple) else func(*args)
                        elif len(test) == 3:
                            a, b, expected = test
                            result = func(a, b)
                        else:
                            args = test[:-1]
                            expected = test[-1]
                            result = func(*args)
                        if result == expected:
                            passed += 1
                    except Exception:
                        pass
                test_pass_rate = passed / total

        if test_pass_rate < 1.0:
            return VerifierResult(
                correct=False,
                score=test_pass_rate * 0.3,
                partial_credit={"test_pass_rate": test_pass_rate},
                diagnostics=f"Error resolved but tests fail: {test_pass_rate:.0%}",
            )

        # Compute minimality
        submitted_lines = submitted_code.split("\n")
        fixed_lines = self._fixed_code.split("\n")
        diff = list(difflib.unified_diff(submitted_lines, fixed_lines, lineterm=""))
        actual_diff_size = max(1, sum(1 for l in diff if l.startswith("+") or l.startswith("-")) // 2)

        # Also compute diff from buggy to submitted (how much did the model change?)
        buggy_lines = self._buggy_code.split("\n")
        model_diff = list(difflib.unified_diff(buggy_lines, submitted_lines, lineterm=""))
        model_diff_size = max(1, sum(1 for l in model_diff if l.startswith("+") or l.startswith("-")) // 2)

        # Minimality: how close is the model's diff to the golden diff size?
        if model_diff_size <= self._golden_diff_size:
            minimality = 1.0
        else:
            minimality = self._golden_diff_size / model_diff_size

        correct = True
        score = min(1.0, minimality)

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "test_pass_rate": 1.0,
                "minimality": minimality,
                "model_diff_size": model_diff_size,
                "golden_diff_size": self._golden_diff_size,
            },
            diagnostics=f"Error resolved. Tests pass. Minimality: {minimality:.2f} (model diff: {model_diff_size}, golden: {self._golden_diff_size})",
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ErrorRecoveryEnv(BaseReasoningEnv):
    """
    ErrorRecovery environment: one-shot minimal fix for runtime errors.

    The model receives buggy code + error message and must produce the
    minimal fix. Reward = error_resolved * test_pass * minimality.
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
            problem_generator = error_recovery_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ErrorRecoveryVerifier(
            buggy_code=problem.metadata["buggy_code"],
            fixed_code=problem.metadata["fixed_code"],
            error_message=problem.metadata["error_message"],
            tests=problem.metadata["tests"],
            golden_diff_size=problem.metadata["golden_diff_size"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if re.search(r"```\n.*?```", response, re.DOTALL):
            return 0.5
        return 0.0

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response
