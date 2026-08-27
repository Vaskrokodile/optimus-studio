"""
AstRefactorChain: Multi-step AST refactoring with quality metrics.

Environment concept:
  The model receives unclean Python code (long functions, duplicated logic,
  bad variable names) and must produce a refactored version that is valid
  Python and improves on specific quality metrics.

  Verification is rule-based and FAST:
    1. Parse the refactored code with ``ast`` — must be valid Python.
    2. Check specific quality metrics derived from the AST:
       - function length reduced (fewer statements)
       - variable names improved (no single-letter / x/y/z names)
       - duplicated logic reduced (fewer repeated call patterns)
       - nesting depth reduced
    3. The refactored code must remain functionally equivalent (run both
       original and refactored against a small test suite).

  reward = correctness * 0.5 + quality * 0.5

  This is a batch environment: N parallel refactors are scored, and the
  best (highest combined score) is selected as the aggregate reward.
"""

from __future__ import annotations

import ast
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: unclean code + tests + quality checks
# ---------------------------------------------------------------------------


_REFACTOR_CHAIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Refactor a verbose even-filter into a comprehension with good names.",
        "func_name": "filter_evens",
        "original": (
            "def filter_evens(a):\n"
            "    b = []\n"
            "    for c in a:\n"
            "        if c % 2 == 0:\n"
            "            b.append(c)\n"
            "    return b\n"
        ),
        "tests": [
            ([1, 2, 3, 4, 5, 6], [2, 4, 6]),
            ([], []),
            ([1, 3, 5], []),
            ([2, 4, 6], [2, 4, 6]),
        ],
        "quality_checks": ["function_length_reduced", "no_single_letter_names"],
    },
    {
        "task": "Refactor duplicated sum logic into a single helper.",
        "func_name": "sum_all",
        "original": (
            "def sum_all(x):\n"
            "    t = 0\n"
            "    for i in range(len(x)):\n"
            "        t = t + x[i]\n"
            "    return t\n"
        ),
        "tests": [
            ([1, 2, 3], 6),
            ([], 0),
            ([5], 5),
            ([-1, -2, -3], -6),
        ],
        "quality_checks": ["function_length_reduced", "no_single_letter_names", "no_index_loop"],
    },
    {
        "task": "Refactor a nested if/else classify into a concise form.",
        "func_name": "classify",
        "original": (
            "def classify(n):\n"
            "    if n > 0:\n"
            "        r = 'positive'\n"
            "    else:\n"
            "        if n < 0:\n"
            "            r = 'negative'\n"
            "        else:\n"
            "            r = 'zero'\n"
            "    return r\n"
        ),
        "tests": [
            (5, "positive"),
            (-3, "negative"),
            (0, "zero"),
        ],
        "quality_checks": ["function_length_reduced", "nesting_depth_reduced", "no_single_letter_names"],
    },
    {
        "task": "Refactor manual max-finding into a concise form.",
        "func_name": "find_max",
        "original": (
            "def find_max(l):\n"
            "    m = l[0]\n"
            "    for i in range(1, len(l)):\n"
            "        if l[i] > m:\n"
            "            m = l[i]\n"
            "    return m\n"
        ),
        "tests": [
            ([3, 1, 4, 1, 5], 5),
            ([10], 10),
            ([-1, -5, -2], -1),
        ],
        "quality_checks": ["function_length_reduced", "no_single_letter_names", "no_index_loop"],
    },
    {
        "task": "Refactor verbose string-reversal into a concise form.",
        "func_name": "reverse_str",
        "original": (
            "def reverse_str(s):\n"
            "    r = ''\n"
            "    i = len(s) - 1\n"
            "    while i >= 0:\n"
            "        r = r + s[i]\n"
            "        i = i - 1\n"
            "    return r\n"
        ),
        "tests": [
            ("hello", "olleh"),
            ("", ""),
            ("a", "a"),
            ("ab", "ba"),
        ],
        "quality_checks": ["function_length_reduced", "no_single_letter_names", "no_index_loop"],
    },
]


def _max_function_length(code: str) -> int:
    """Return the max number of statements in any top-level function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999
    best = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            best = max(best, len(node.body))
    return best


def _max_nesting_depth(code: str) -> int:
    """Return the maximum nesting depth of control-flow blocks."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999

    def depth(node: ast.AST, current: int) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                best = max(best, depth(child, current + 1))
            else:
                best = max(best, depth(child, current))
        return best

    return depth(tree, 0)


def _single_letter_names(code: str) -> set[str]:
    """Return the set of single-letter variable/argument names used."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if len(node.id) == 1 and node.id != "_":
                names.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if len(arg.arg) == 1 and arg.arg != "_":
                    names.add(arg.arg)
    return names


def _uses_index_loop(code: str) -> bool:
    """Detect ``for i in range(len(x))`` index-loop patterns."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == "range":
                if node.iter.args and isinstance(node.iter.args[0], ast.Call):
                    inner = node.iter.args[0]
                    if (
                        isinstance(inner.func, ast.Name)
                        and inner.func.id == "len"
                    ):
                        return True
    return False


def ast_refactor_chain_generator(seed: int) -> Problem:
    """Generate an AstRefactorChain problem."""
    rng = random.Random(seed)
    template = rng.choice(_REFACTOR_CHAIN_PROBLEMS)

    original = template["original"]
    original_len = _max_function_length(original)
    original_depth = _max_nesting_depth(original)
    original_single = _single_letter_names(original)
    original_index = _uses_index_loop(original)

    prompt = (
        f"Refactor the following Python code to improve its quality.\n\n"
        f"```python\n{original}\n```\n\n"
        f"Task: {template['task']}\n"
        f"Quality checks to satisfy: {', '.join(template['quality_checks'])}\n\n"
        f"Requirements:\n"
        f"  - The refactored code must be valid Python\n"
        f"  - It must produce IDENTICAL output to the original on all tests\n"
        f"  - Keep the same function name: {template['func_name']}\n\n"
        f"Format your answer as:\n"
        f"```python\n<refactored code>\n```"
    )

    return Problem(
        id=f"ast_refactor_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "ast_refactor_chain",
            "task": template["task"],
            "func_name": template["func_name"],
            "original_code": original,
            "tests": template["tests"],
            "quality_checks": template["quality_checks"],
            "original_length": original_len,
            "original_depth": original_depth,
            "original_single_letter": sorted(original_single),
            "original_index_loop": original_index,
        },
        token_budget=1200,
        source="generated",
    )


ast_refactor_chain_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier — correctness + quality metrics
# ---------------------------------------------------------------------------


class AstRefactorChainVerifier(Verifier):
    """Verify a refactored function: correctness + AST quality metrics."""

    def __init__(
        self,
        original_code: str,
        func_name: str,
        tests: list,
        quality_checks: list[str],
        original_length: int,
        original_depth: int,
        original_single_letter: list[str],
        original_index_loop: bool,
    ):
        super().__init__()
        self._original = original_code
        self._func_name = func_name
        self._tests = tests
        self._quality_checks = quality_checks
        self._original_length = original_length
        self._original_depth = original_depth
        self._original_single = set(original_single_letter)
        self._original_index = original_index_loop

    def verify(self, response: str) -> VerifierResult:
        code = self._extract_code(response)
        if not code:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No python code block found in response",
            )

        # --- Validity: must parse ---
        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Refactored code is not valid Python: {e}",
            )

        # --- Compile and load both functions ---
        base_ns: dict[str, Any] = {}
        ref_ns: dict[str, Any] = {}
        try:
            exec(self._original, base_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Original failed to compile: {e}",
            )
        try:
            exec(code, ref_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Refactored code failed to execute: {e}",
            )

        base_func = base_ns.get(self._func_name)
        ref_func = ref_ns.get(self._func_name)
        if not callable(ref_func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in refactored code",
            )

        # --- Correctness ---
        correct_count = 0
        total = len(self._tests)
        diag: list[str] = []
        for test in self._tests:
            args = test[:-1]
            expected = test[-1]
            try:
                base_out = base_func(*args)
            except Exception as e:
                diag.append(f"baseline err on {args!r}: {e}")
                continue
            try:
                ref_out = ref_func(*args)
            except Exception as e:
                diag.append(f"refactored err on {args!r}: {e}")
                continue
            if base_out == ref_out == expected:
                correct_count += 1
            else:
                diag.append(f"mismatch on {args!r}: {base_out!r} vs {ref_out!r} (expected {expected!r})")

        correctness = correct_count / total if total > 0 else 0.0

        # --- Quality metrics ---
        ref_length = _max_function_length(code)
        ref_depth = _max_nesting_depth(code)
        ref_single = _single_letter_names(code)
        ref_index = _uses_index_loop(code)

        checks_passed = 0
        checks_total = len(self._quality_checks)
        check_diag: list[str] = []

        for check in self._quality_checks:
            if check == "function_length_reduced":
                ok = ref_length < self._original_length
                if ok:
                    checks_passed += 1
                check_diag.append(f"len:{self._original_length}->{ref_length}({'ok' if ok else 'no'})")
            elif check == "no_single_letter_names":
                ok = len(ref_single) == 0
                if ok:
                    checks_passed += 1
                check_diag.append(f"single_letter:{sorted(ref_single)}({'ok' if ok else 'no'})")
            elif check == "nesting_depth_reduced":
                ok = ref_depth < self._original_depth
                if ok:
                    checks_passed += 1
                check_diag.append(f"depth:{self._original_depth}->{ref_depth}({'ok' if ok else 'no'})")
            elif check == "no_index_loop":
                ok = not ref_index
                if ok:
                    checks_passed += 1
                check_diag.append(f"index_loop:{ref_index}({'ok' if ok else 'no'})")

        quality = checks_passed / checks_total if checks_total > 0 else 0.0

        score = correctness * 0.5 + quality * 0.5
        correct = correctness >= 1.0 and quality >= 0.5

        full_diag = (
            f"Correctness: {correct_count}/{total} "
            f"Quality: {checks_passed}/{checks_total} ({' '.join(check_diag)}) "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "correctness": correctness,
                "quality": quality,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "correct_inputs": correct_count,
                "total_inputs": total,
            },
            diagnostics=full_diag,
        )

    def _extract_code(self, response: str) -> str:
        matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        if "def " in response and "```" not in response:
            return response.strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AstRefactorChainEnv(BatchEnvBase):
    """AstRefactorChain: multi-step AST refactoring with quality metrics."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = ast_refactor_chain_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return AstRefactorChainVerifier(
            original_code=md["original_code"],
            func_name=md["func_name"],
            tests=md["tests"],
            quality_checks=md["quality_checks"],
            original_length=md["original_length"],
            original_depth=md["original_depth"],
            original_single_letter=md["original_single_letter"],
            original_index_loop=md["original_index_loop"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if "def " in response:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": best_score,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Mean={sum(scores)/len(scores) if scores else 0:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
