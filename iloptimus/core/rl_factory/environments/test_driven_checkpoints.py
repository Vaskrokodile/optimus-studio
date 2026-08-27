"""
TestDrivenCheckpoints: Each checkpoint = make a specific test pass.

Environment concept:
  The model receives a coding task with a test suite. The tests are ordered
  from easiest to hardest. The model must:
    1. Write code that passes test 1 (checkpoint 1)
    2. AUDIT by running the test — does it pass?
    3. Extend the code to pass test 2 (checkpoint 2) WITHOUT breaking test 1
    4. AUDIT by running ALL tests so far
    5. Continue until all tests pass

  This is fundamentally different from the generic CheckpointExecutor
  because the AUDIT IS REAL — we actually execute the code against the
  tests. No keyword matching, no heuristics. The audit is ground truth.

  This is a batch environment: N parallel implementations are scored.
  The reward measures:
    - Tests passed: how many of the N tests are passing in the final code
    - Audit honesty: did the model's self-reported audit match reality?
    - Non-regression: did later checkpoints break earlier tests?
    - Code quality: is the code concise (token efficiency)?

  reward = tests_passed * 0.5 + audit_honesty * 0.3 + non_regression * 0.2

  This trains the model to:
    1. Build incrementally (test by test, not all at once)
    2. Self-audit by running tests (the most reliable audit method)
    3. Avoid regressions (each new feature must not break existing tests)
    4. Be honest about test results (not claim PASS when tests fail)

  The key insight: TDD checkpoints are how real software engineers work.
  A model that can build code test-by-test with honest self-auditing is
  production-ready. This is the most practical coding environment in the
  entire suite.
"""

from __future__ import annotations

import random
import re
import traceback
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: each has ordered tests (easy → hard)
# ---------------------------------------------------------------------------


_TDD_PROBLEMS = [
    {
        "task": "Write a Stack class with push, pop, and peek methods.",
        "func_name": "Stack",
        "tests": [
            {"name": "push and pop", "code": "s = Stack(); s.push(1); s.push(2); assert s.pop() == 2; assert s.pop() == 1"},
            {"name": "peek without removing", "code": "s = Stack(); s.push(10); assert s.peek() == 10; assert s.pop() == 10"},
            {"name": "pop from empty returns None", "code": "s = Stack(); assert s.pop() is None"},
            {"name": "push multiple and check order", "code": "s = Stack(); [s.push(i) for i in range(5)]; assert [s.pop() for _ in range(5)] == [4,3,2,1,0]"},
        ],
    },
    {
        "task": "Write a Queue class with enqueue and dequeue methods.",
        "func_name": "Queue",
        "tests": [
            {"name": "enqueue and dequeue", "code": "q = Queue(); q.enqueue(1); q.enqueue(2); assert q.dequeue() == 1; assert q.dequeue() == 2"},
            {"name": "dequeue from empty returns None", "code": "q = Queue(); assert q.dequeue() is None"},
            {"name": "FIFO order maintained", "code": "q = Queue(); [q.enqueue(i) for i in range(5)]; assert [q.dequeue() for _ in range(5)] == [0,1,2,3,4]"},
            {"name": "interleaved operations", "code": "q = Queue(); q.enqueue(1); q.enqueue(2); assert q.dequeue() == 1; q.enqueue(3); assert q.dequeue() == 2; assert q.dequeue() == 3"},
        ],
    },
    {
        "task": "Write a function `is_palindrome(s)` that checks if a string is a palindrome.",
        "func_name": "is_palindrome",
        "tests": [
            {"name": "simple palindrome", "code": "assert is_palindrome('racecar') == True"},
            {"name": "non-palindrome", "code": "assert is_palindrome('hello') == False"},
            {"name": "empty string", "code": "assert is_palindrome('') == True"},
            {"name": "single character", "code": "assert is_palindrome('a') == True"},
            {"name": "case insensitive", "code": "assert is_palindrome('RaceCar') == True"},
        ],
    },
    {
        "task": "Write a function `fibonacci(n)` returning the nth Fibonacci number (F(0)=0, F(1)=1).",
        "func_name": "fibonacci",
        "tests": [
            {"name": "base cases", "code": "assert fibonacci(0) == 0; assert fibonacci(1) == 1"},
            {"name": "small values", "code": "assert fibonacci(2) == 1; assert fibonacci(5) == 5"},
            {"name": "larger value", "code": "assert fibonacci(10) == 55"},
            {"name": "fibonacci(20)", "code": "assert fibonacci(20) == 6765"},
        ],
    },
    {
        "task": "Write a function `factorial(n)` that returns n!.",
        "func_name": "factorial",
        "tests": [
            {"name": "base case", "code": "assert factorial(0) == 1; assert factorial(1) == 1"},
            {"name": "small values", "code": "assert factorial(3) == 6; assert factorial(5) == 120"},
            {"name": "larger value", "code": "assert factorial(10) == 3628800"},
        ],
    },
    {
        "task": "Write a function `count_words(s)` that counts words in a string.",
        "func_name": "count_words",
        "tests": [
            {"name": "simple sentence", "code": "assert count_words('hello world') == 2"},
            {"name": "empty string", "code": "assert count_words('') == 0"},
            {"name": "multiple spaces", "code": "assert count_words('  hello   world  ') == 2"},
            {"name": "single word", "code": "assert count_words('hello') == 1"},
        ],
    },
    {
        "task": "Write a function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
        "func_name": "merge_sorted",
        "tests": [
            {"name": "basic merge", "code": "assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]"},
            {"name": "one empty", "code": "assert merge_sorted([], [1,2,3]) == [1,2,3]; assert merge_sorted([1,2], []) == [1,2]"},
            {"name": "both empty", "code": "assert merge_sorted([], []) == []"},
            {"name": "duplicates", "code": "assert merge_sorted([1,2,2], [2,3]) == [1,2,2,2,3]"},
        ],
    },
    {
        "task": "Write a function `is_prime(n)` that returns True if n is a prime number.",
        "func_name": "is_prime",
        "tests": [
            {"name": "small primes", "code": "assert is_prime(2) == True; assert is_prime(3) == True; assert is_prime(5) == True"},
            {"name": "non-primes", "code": "assert is_prime(4) == False; assert is_prime(6) == False; assert is_prime(1) == False"},
            {"name": "edge cases", "code": "assert is_prime(0) == False; assert is_prime(-1) == False"},
            {"name": "larger prime", "code": "assert is_prime(97) == True; assert is_prime(100) == False"},
        ],
    },
    {
        "task": "Write a function `reverse_words(s)` that reverses the order of words in a string.",
        "func_name": "reverse_words",
        "tests": [
            {"name": "two words", "code": "assert reverse_words('hello world') == 'world hello'"},
            {"name": "three words", "code": "assert reverse_words('a b c') == 'c b a'"},
            {"name": "single word", "code": "assert reverse_words('hello') == 'hello'"},
            {"name": "empty string", "code": "assert reverse_words('') == ''"},
        ],
    },
    {
        "task": "Write a function `sum_digits(n)` that returns the sum of digits of a non-negative integer.",
        "func_name": "sum_digits",
        "tests": [
            {"name": "simple", "code": "assert sum_digits(123) == 6"},
            {"name": "zero", "code": "assert sum_digits(0) == 0"},
            {"name": "single digit", "code": "assert sum_digits(5) == 5"},
            {"name": "large number", "code": "assert sum_digits(99999) == 45"},
        ],
    },
]


def test_driven_checkpoints_generator(seed: int) -> Problem:
    """Generate a TestDrivenCheckpoints problem."""
    rng = random.Random(seed)
    template = rng.choice(_TDD_PROBLEMS)

    tests = template["tests"]
    checkpoints = []
    for i, test in enumerate(tests):
        checkpoints.append({
            "name": f"Pass test: {test['name']}",
            "test_code": test["code"],
            "test_index": i,
            "success_criteria": f"Test '{test['name']}' must pass",
        })

    cp_text = "\n".join(
        f"CHECKPOINT {i+1}: {cp['name']}\n  Test: {cp['test_code']}"
        for i, cp in enumerate(checkpoints)
    )

    return Problem(
        id=f"tdd_checkpoints_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Build the solution checkpoint by checkpoint. Each checkpoint = make a test pass.\n\n"
            f"Checkpoints:\n{cp_text}\n\n"
            f"Format:\n"
            f"CHECKPOINT <n>: <test name>\n"
            f"CODE: ```python\n<your code so far>\n```\n"
            f"AUDIT: PASS or FAIL - <did the test pass?>\n"
            f"---\n\n"
            f"Rules:\n"
            f"  - Each checkpoint's code must include ALL previous checkpoints' code\n"
            f"  - AUDIT honestly — we WILL run the tests to verify\n"
            f"  - Don't break earlier tests when adding new features\n"
            f"  - Be concise"
        ),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "test_driven_checkpoints",
            "task": template["task"],
            "func_name": template["func_name"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
        },
        token_budget=200 + len(checkpoints) * 200,
        source="generated",
    )


# Prevent pytest from collecting this generator function as a test
test_driven_checkpoints_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier — runs actual tests
# ---------------------------------------------------------------------------


class TestDrivenCheckpointsVerifier(Verifier):
    """Verifies a TDD checkpoint response by RUNNING the tests."""

    def __init__(self, checkpoints: list[dict], func_name: str):
        super().__init__()
        self._checkpoints = checkpoints
        self._func_name = func_name

    def verify(self, response: str) -> VerifierResult:
        executed = self._parse_checkpoints(response)

        if not executed:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No checkpoint blocks found",
            )

        total_cps = len(self._checkpoints)
        tests_passed = 0
        audit_honest = 0
        audit_total = 0
        regressions = 0

        # Track which tests have passed so far
        passed_so_far = set()

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            code = exe_cp.get("code", "")
            model_audit = exe_cp.get("audit_result", "unknown")

            # Actually run the test
            actual_pass = self._run_test(code, exp_cp["test_code"])

            audit_total += 1
            # Check audit honesty
            if (actual_pass and "pass" in model_audit.lower()) or \
               (not actual_pass and "fail" in model_audit.lower()):
                audit_honest += 1

            if actual_pass:
                tests_passed += 1
                # Check for regression: did earlier tests break?
                for prev_idx in passed_so_far:
                    if not self._run_test(code, self._checkpoints[prev_idx]["test_code"]):
                        regressions += 1
                passed_so_far.add(i)
            else:
                # Check if earlier tests still pass (non-regression)
                for prev_idx in passed_so_far:
                    if not self._run_test(code, self._checkpoints[prev_idx]["test_code"]):
                        regressions += 1

        test_score = tests_passed / total_cps if total_cps > 0 else 0.0
        audit_accuracy = audit_honest / audit_total if audit_total > 0 else 0.0
        non_regression = 1.0 - min(1.0, regressions / max(total_cps, 1))

        score = test_score * 0.5 + audit_accuracy * 0.3 + non_regression * 0.2
        correct = test_score >= 0.6 and audit_accuracy >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "tests_passed": tests_passed,
                "total_tests": total_cps,
                "test_score": test_score,
                "audit_accuracy": audit_accuracy,
                "non_regression": non_regression,
                "regressions": regressions,
                "executed_count": len(executed),
            },
            diagnostics=f"Tests: {tests_passed}/{total_cps} Audit: {audit_accuracy:.0%} Regressions: {regressions} NonReg: {non_regression:.0%}",
        )

    def _run_test(self, code: str, test_code: str) -> bool:
        """Execute code + test and return True if test passes."""
        # Extract code from markdown block
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
        """Parse CHECKPOINT n / CODE / AUDIT blocks."""
        executed = []
        blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE)

        for block in blocks[1:]:
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""
            code = ""
            audit = ""
            audit_result = "unknown"

            # Collect code block and audit
            in_code = False
            code_lines = []
            for line in lines[1:]:
                if "```python" in line or "```" in line:
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
            # Also check for inline CODE: format
            if not code:
                for line in lines[1:]:
                    if line.strip().upper().startswith("CODE:"):
                        code = line.split(":", 1)[1].strip()

            if name:
                executed.append({
                    "name": name,
                    "code": code,
                    "audit": audit,
                    "audit_result": audit_result,
                })

        return executed


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestDrivenCheckpointsEnv(BatchEnvBase):
    """TestDrivenCheckpoints: build code test-by-test with real execution audits.

    Batch-aware: N parallel implementations, reward = best implementation.
    The audit is REAL — we run the tests to verify.
    """
    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = test_driven_checkpoints_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return TestDrivenCheckpointsVerifier(
            checkpoints=problem.metadata["checkpoints"],
            func_name=problem.metadata["func_name"],
        )

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
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Mean={sum(scores)/len(scores) if scores else 0:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
