"""
DependencyChainBuilder: Build code in dependency order with audits.

Environment concept:
  The model receives a coding task with multiple interdependent functions.
  Each function depends on earlier functions. The model must:
    1. Identify the dependency order
    2. Build functions in the correct order (dependencies first)
    3. After each function, AUDIT by testing it (real execution)
    4. Only proceed to the next function if the audit passes

  The checkpoints are the functions themselves, ordered by dependency.
  For example:
    - Checkpoint 1: build `helper()` (no dependencies)
    - Checkpoint 2: build `process()` (depends on `helper()`)
    - Checkpoint 3: build `main()` (depends on `process()`)

  The audit at each checkpoint runs the function and checks its output.
  If a function fails, the model must fix it before building the next one.

  This is a batch environment: N parallel builds are scored.
  The reward measures:
    - Functions correct: how many functions pass their tests
    - Dependency order: were functions built in the right order?
    - Audit honesty: did the model's self-audit match reality?
    - Integration: do all functions work together?

  reward = functions_correct * 0.4 + order_correct * 0.2 + audit_honesty * 0.2 + integration * 0.2

  This trains the model to:
    1. Think about DEPENDENCIES before coding (topological sort thinking)
    2. Build bottom-up (foundations first)
    3. Test each component independently before integration
    4. Verify integration works (not just individual components)

  The key insight: dependency-aware coding is what separates junior from
  senior engineers. A model that builds in the right order, tests each
  component, and verifies integration is worth 10x more than one that
  writes everything at once and hopes it works.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: functions with dependencies
# ---------------------------------------------------------------------------


_DEP_PROBLEMS = [
    {
        "task": "Build a temperature converter: celsius_to_fahrenheit, fahrenheit_to_celsius, and convert (which takes a value, from_unit, to_unit).",
        "functions": [
            {
                "name": "celsius_to_fahrenheit",
                "signature": "def celsius_to_fahrenheit(c):",
                "depends_on": [],
                "test": "assert celsius_to_fahrenheit(0) == 32; assert celsius_to_fahrenheit(100) == 212; assert celsius_to_fahrenheit(-40) == -40",
                "order": 1,
            },
            {
                "name": "fahrenheit_to_celsius",
                "signature": "def fahrenheit_to_celsius(f):",
                "depends_on": [],
                "test": "assert fahrenheit_to_celsius(32) == 0; assert fahrenheit_to_celsius(212) == 100; assert fahrenheit_to_celsius(-40) == -40",
                "order": 2,
            },
            {
                "name": "convert",
                "signature": "def convert(value, from_unit, to_unit):",
                "depends_on": ["celsius_to_fahrenheit", "fahrenheit_to_celsius"],
                "test": "assert convert(100, 'C', 'F') == 212; assert convert(32, 'F', 'C') == 0; assert convert(0, 'C', 'C') == 0",
                "order": 3,
            },
        ],
    },
    {
        "task": "Build a string utility library: reverse_string, is_palindrome, and count_palindromes (which counts palindromes in a list).",
        "functions": [
            {
                "name": "reverse_string",
                "signature": "def reverse_string(s):",
                "depends_on": [],
                "test": "assert reverse_string('hello') == 'olleh'; assert reverse_string('') == ''; assert reverse_string('a') == 'a'",
                "order": 1,
            },
            {
                "name": "is_palindrome",
                "signature": "def is_palindrome(s):",
                "depends_on": ["reverse_string"],
                "test": "assert is_palindrome('racecar') == True; assert is_palindrome('hello') == False; assert is_palindrome('') == True",
                "order": 2,
            },
            {
                "name": "count_palindromes",
                "signature": "def count_palindromes(words):",
                "depends_on": ["is_palindrome"],
                "test": "assert count_palindromes(['racecar', 'hello', 'madam']) == 2; assert count_palindromes([]) == 0; assert count_palindromes(['a', 'b', 'c']) == 3",
                "order": 3,
            },
        ],
    },
    {
        "task": "Build a math utility: square, sum_of_squares (sum of squares of a list), and variance (population variance).",
        "functions": [
            {
                "name": "square",
                "signature": "def square(n):",
                "depends_on": [],
                "test": "assert square(3) == 9; assert square(0) == 0; assert square(-2) == 4",
                "order": 1,
            },
            {
                "name": "sum_of_squares",
                "signature": "def sum_of_squares(lst):",
                "depends_on": ["square"],
                "test": "assert sum_of_squares([1,2,3]) == 14; assert sum_of_squares([]) == 0; assert sum_of_squares([5]) == 25",
                "order": 2,
            },
            {
                "name": "variance",
                "signature": "def variance(lst):",
                "depends_on": ["sum_of_squares"],
                "test": "assert variance([1,2,3,4,5]) == 2.0; assert variance([5,5,5]) == 0.0; assert abs(variance([1,3]) - 1.0) < 0.001",
                "order": 3,
            },
        ],
    },
    {
        "task": "Build a list utility: max_element, second_max (second largest), and top_two (returns (max, second_max)).",
        "functions": [
            {
                "name": "max_element",
                "signature": "def max_element(lst):",
                "depends_on": [],
                "test": "assert max_element([1,3,2]) == 3; assert max_element([5]) == 5; assert max_element([-1,-2,-3]) == -1",
                "order": 1,
            },
            {
                "name": "second_max",
                "signature": "def second_max(lst):",
                "depends_on": ["max_element"],
                "test": "assert second_max([1,3,2]) == 2; assert second_max([5,1,4,2]) == 4; assert second_max([3,3,1]) == 3",
                "order": 2,
            },
            {
                "name": "top_two",
                "signature": "def top_two(lst):",
                "depends_on": ["max_element", "second_max"],
                "test": "assert top_two([1,3,2]) == (3,2); assert top_two([5,1,4,2]) == (5,4); assert top_two([10,20]) == (20,10)",
                "order": 3,
            },
        ],
    },
    {
        "task": "Build a text processor: tokenize (split into words), word_count, and avg_word_length.",
        "functions": [
            {
                "name": "tokenize",
                "signature": "def tokenize(text):",
                "depends_on": [],
                "test": "assert tokenize('hello world') == ['hello', 'world']; assert tokenize('') == []; assert tokenize('one') == ['one']",
                "order": 1,
            },
            {
                "name": "word_count",
                "signature": "def word_count(text):",
                "depends_on": ["tokenize"],
                "test": "assert word_count('hello world') == 2; assert word_count('') == 0; assert word_count('a b c d') == 4",
                "order": 2,
            },
            {
                "name": "avg_word_length",
                "signature": "def avg_word_length(text):",
                "depends_on": ["tokenize"],
                "test": "assert avg_word_length('hi there') == 4.0; assert avg_word_length('a') == 1.0; assert avg_word_length('ab cd') == 2.0",
                "order": 3,
            },
        ],
    },
    {
        "task": "Build a grade calculator: letter_grade (score to letter), pass_fail (letter to P/F), and class_stats (list of scores to pass/fail counts).",
        "functions": [
            {
                "name": "letter_grade",
                "signature": "def letter_grade(score):",
                "depends_on": [],
                "test": "assert letter_grade(95) == 'A'; assert letter_grade(85) == 'B'; assert letter_grade(75) == 'C'; assert letter_grade(50) == 'F'",
                "order": 1,
            },
            {
                "name": "pass_fail",
                "signature": "def pass_fail(letter):",
                "depends_on": ["letter_grade"],
                "test": "assert pass_fail('A') == 'P'; assert pass_fail('B') == 'P'; assert pass_fail('F') == 'F'",
                "order": 2,
            },
            {
                "name": "class_stats",
                "signature": "def class_stats(scores):",
                "depends_on": ["pass_fail", "letter_grade"],
                "test": "assert class_stats([95, 85, 50]) == {'P': 2, 'F': 1}; assert class_stats([]) == {'P': 0, 'F': 0}; assert class_stats([90, 80, 70, 60]) == {'P': 4, 'F': 0}",
                "order": 3,
            },
        ],
    },
]


def dependency_chain_builder_generator(seed: int) -> Problem:
    """Generate a DependencyChainBuilder problem."""
    rng = random.Random(seed)
    template = rng.choice(_DEP_PROBLEMS)

    functions = sorted(template["functions"], key=lambda f: f["order"])
    checkpoints = []
    for i, func in enumerate(functions):
        deps = func["depends_on"]
        deps_str = ", ".join(deps) if deps else "none"
        checkpoints.append({
            "name": f"Build {func['name']}()",
            "func_name": func["name"],
            "signature": func["signature"],
            "depends_on": deps,
            "test_code": func["test"],
            "order": func["order"],
            "success_criteria": f"Function {func['name']}() must pass its test (depends on: {deps_str})",
        })

    cp_text = "\n".join(
        f"CHECKPOINT {i+1}: {cp['name']}\n"
        f"  Signature: {cp['signature']}\n"
        f"  Depends on: {', '.join(cp['depends_on']) if cp['depends_on'] else 'none'}\n"
        f"  Test: {cp['test_code']}"
        for i, cp in enumerate(checkpoints)
    )

    return Problem(
        id=f"dep_chain_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Build the functions in dependency order. Each checkpoint = one function.\n"
            f"After each, AUDIT by running the test.\n\n"
            f"Checkpoints:\n{cp_text}\n\n"
            f"Format:\n"
            f"CHECKPOINT <n>: <function name>\n"
            f"CODE: ```python\n<all code so far>\n```\n"
            f"AUDIT: PASS or FAIL - <test result>\n"
            f"---\n\n"
            f"Rules:\n"
            f"  - Build in dependency order (dependencies first)\n"
            f"  - Each checkpoint includes ALL previous code\n"
            f"  - AUDIT honestly — we run the tests\n"
            f"  - Don't proceed if a dependency fails"
        ),
        difficulty=0.35 + 0.1 * rng.random(),
        metadata={
            "type": "dependency_chain_builder",
            "task": template["task"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
            "functions": functions,
        },
        token_budget=200 + len(checkpoints) * 250,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — runs actual tests for each function
# ---------------------------------------------------------------------------


class DependencyChainVerifier(Verifier):
    """Verifies a dependency chain build by running tests."""

    def __init__(self, checkpoints: list[dict]):
        super().__init__()
        self._checkpoints = checkpoints

    def verify(self, response: str) -> VerifierResult:
        executed = self._parse_checkpoints(response)

        if not executed:
            return VerifierResult(correct=False, score=0.0, diagnostics="No checkpoints found")

        total = len(self._checkpoints)
        functions_correct = 0
        order_correct = 0
        audit_honest = 0
        audit_total = 0
        integration_ok = False

        # Check each checkpoint
        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            code = exe_cp.get("code", "")
            model_audit = exe_cp.get("audit_result", "unknown")

            # Check order: was this function built after its dependencies?
            func_name = exp_cp["func_name"]
            deps = exp_cp["depends_on"]

            # Check if dependencies were built earlier
            order_ok = True
            for dep in deps:
                dep_found = False
                for j in range(i):
                    if j < len(executed) and dep in executed[j].get("name", "").lower():
                        dep_found = True
                        break
                if not dep_found:
                    order_ok = False
                    break

            if order_ok:
                order_correct += 1

            # Run the test
            actual_pass = self._run_test(code, exp_cp["test_code"])

            audit_total += 1
            if (actual_pass and "pass" in model_audit.lower()) or \
               (not actual_pass and "fail" in model_audit.lower()):
                audit_honest += 1

            if actual_pass:
                functions_correct += 1

        # Integration test: run ALL tests with the final code
        if executed:
            final_code = executed[-1].get("code", "")
            all_tests_pass = True
            for cp in self._checkpoints:
                if not self._run_test(final_code, cp["test_code"]):
                    all_tests_pass = False
                    break
            integration_ok = all_tests_pass

        func_score = functions_correct / total if total > 0 else 0.0
        order_score = order_correct / total if total > 0 else 0.0
        audit_score = audit_honest / audit_total if audit_total > 0 else 0.0
        integration_score = 1.0 if integration_ok else 0.0

        score = func_score * 0.4 + order_score * 0.2 + audit_score * 0.2 + integration_score * 0.2
        correct = func_score >= 0.6 and integration_ok

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "functions_correct": functions_correct,
                "total_functions": total,
                "order_correct": order_correct,
                "audit_honesty": audit_score,
                "integration": integration_ok,
            },
            diagnostics=f"Funcs: {functions_correct}/{total} Order: {order_correct}/{total} Audit: {audit_score:.0%} Integration: {'OK' if integration_ok else 'FAIL'}",
        )

    def _run_test(self, code: str, test_code: str) -> bool:
        code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
            exec(test_code, namespace)
            return True
        except Exception:
            return False

    def _parse_checkpoints(self, response: str) -> list[dict]:
        executed = []
        blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE)

        for block in blocks[1:]:
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""
            code = ""
            audit_result = "unknown"
            in_code = False
            code_lines = []

            for line in lines[1:]:
                if "```python" in line or (line.strip() == "```" and in_code):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)
                elif line.strip().upper().startswith("AUDIT:"):
                    audit = line.split(":", 1)[1].strip()
                    if "pass" in audit.lower():
                        audit_result = "pass"
                    elif "fail" in audit.lower():
                        audit_result = "fail"

            code = "\n".join(code_lines) if code_lines else ""
            if name:
                executed.append({"name": name, "code": code, "audit_result": audit_result})

        return executed


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DependencyChainBuilderEnv(BatchEnvBase):
    """DependencyChainBuilder: build code in dependency order with real test audits."""

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = dependency_chain_builder_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return DependencyChainVerifier(checkpoints=problem.metadata["checkpoints"])

    def _check_format(self, response: str) -> float:
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE) and "AUDIT:" in response.upper():
            return 1.0
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
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
            "reward": best_score if best_score > 0 else 0.0,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
