"""
TrajectoryRecombination: Train models to produce composable code components.

Environment concept:
  The model receives a multi-checkpoint coding task. Each checkpoint adds a
  new function or method. The model builds the solution checkpoint by
  checkpoint, with each checkpoint including ALL previous code.

  This is a BATCH environment with a critical innovation: the verifier
  doesn't just score each response independently — it actively RECOMBINES
  components from multiple trajectories and tests the result.

  The recombination process:
    1. Parse N trajectories into checkpoint components (code per checkpoint)
    2. Score each checkpoint individually (run tests per checkpoint)
    3. Extract function/class definitions from each agent's final code
    4. Try multiple recombination strategies:
       a. Sequential merge: exec agent_A's code, then agent_B's code
       b. Function-level patching: replace failing functions with better
          versions from other agents (using AST manipulation)
    5. Run the full test suite on each recombined variant
    6. Keep the best recombined result

  Final score = best_individual_score * 0.4 + best_recombined_score * 0.6

  This trains the model to produce components that COMPOSE WELL with others.
  A model that writes self-contained, well-isolated functions will score
  higher because its components can be spliced into other trajectories.
  A model that writes tightly coupled, monolithic code will score lower
  because its components don't recombine cleanly.

  The key insight: in real software engineering, code is written by many
  people and must interoperate. Trajectory recombination is a training-time
  analog of this — it rewards modularity and composability.
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
# Problem templates: multi-function tasks with 3-4 checkpoints each
# ---------------------------------------------------------------------------


_RECOMBINATION_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Write a Stack class with push, pop, peek, and size methods.",
        "checkpoints": [
            {
                "name": "push and pop",
                "test_code": "s = Stack(); s.push(1); s.push(2); assert s.pop() == 2; assert s.pop() == 1",
                "target_name": "Stack",
            },
            {
                "name": "peek without removing",
                "test_code": "s = Stack(); s.push(10); assert s.peek() == 10; assert s.pop() == 10",
                "target_name": "Stack",
            },
            {
                "name": "pop from empty returns None",
                "test_code": "s = Stack(); assert s.pop() is None",
                "target_name": "Stack",
            },
            {
                "name": "size method",
                "test_code": "s = Stack(); s.push(1); s.push(2); assert s.size() == 2; s.pop(); assert s.size() == 1",
                "target_name": "Stack",
            },
        ],
    },
    {
        "task": "Write calculator functions: add(a, b), subtract(a, b), multiply(a, b), and divide(a, b).",
        "checkpoints": [
            {
                "name": "add function",
                "test_code": "assert add(2, 3) == 5; assert add(-1, 1) == 0; assert add(0, 0) == 0",
                "target_name": "add",
            },
            {
                "name": "subtract function",
                "test_code": "assert subtract(5, 3) == 2; assert subtract(0, 5) == -5; assert subtract(3, 3) == 0",
                "target_name": "subtract",
            },
            {
                "name": "multiply function",
                "test_code": "assert multiply(3, 4) == 12; assert multiply(0, 5) == 0; assert multiply(-2, 3) == -6",
                "target_name": "multiply",
            },
            {
                "name": "divide function (returns None on divide by zero)",
                "test_code": "assert divide(10, 2) == 5; assert divide(7, 2) == 3.5; assert divide(5, 0) is None",
                "target_name": "divide",
            },
        ],
    },
    {
        "task": "Write a BankAccount class with deposit, withdraw, get_balance, and transfer methods.",
        "checkpoints": [
            {
                "name": "deposit and get_balance",
                "test_code": "a = BankAccount(100); a.deposit(50); assert a.get_balance() == 150",
                "target_name": "BankAccount",
            },
            {
                "name": "withdraw (cannot overdraw)",
                "test_code": "a = BankAccount(100); assert a.withdraw(30) == True; assert a.get_balance() == 70; assert a.withdraw(100) == False",
                "target_name": "BankAccount",
            },
            {
                "name": "get_balance after multiple operations",
                "test_code": "a = BankAccount(50); a.deposit(25); a.withdraw(10); a.deposit(5); assert a.get_balance() == 70",
                "target_name": "BankAccount",
            },
            {
                "name": "transfer between accounts",
                "test_code": "a = BankAccount(100); b = BankAccount(50); assert a.transfer(b, 30) == True; assert a.get_balance() == 70; assert b.get_balance() == 80; assert a.transfer(b, 1000) == False",
                "target_name": "BankAccount",
            },
        ],
    },
    {
        "task": "Write string utility functions: count_vowels(s), reverse_string(s), and is_palindrome(s).",
        "checkpoints": [
            {
                "name": "count_vowels",
                "test_code": "assert count_vowels('hello') == 2; assert count_vowels('AEIOU') == 5; assert count_vowels('xyz') == 0; assert count_vowels('') == 0",
                "target_name": "count_vowels",
            },
            {
                "name": "reverse_string",
                "test_code": "assert reverse_string('hello') == 'olleh'; assert reverse_string('') == ''; assert reverse_string('a') == 'a'",
                "target_name": "reverse_string",
            },
            {
                "name": "is_palindrome (case insensitive)",
                "test_code": "assert is_palindrome('racecar') == True; assert is_palindrome('RaceCar') == True; assert is_palindrome('hello') == False; assert is_palindrome('') == True",
                "target_name": "is_palindrome",
            },
        ],
    },
    {
        "task": "Write number utility functions: is_even(n), factorial(n), fibonacci(n), and is_prime(n).",
        "checkpoints": [
            {
                "name": "is_even",
                "test_code": "assert is_even(2) == True; assert is_even(3) == False; assert is_even(0) == True; assert is_even(-4) == True",
                "target_name": "is_even",
            },
            {
                "name": "factorial",
                "test_code": "assert factorial(0) == 1; assert factorial(1) == 1; assert factorial(5) == 120; assert factorial(3) == 6",
                "target_name": "factorial",
            },
            {
                "name": "fibonacci (F(0)=0, F(1)=1)",
                "test_code": "assert fibonacci(0) == 0; assert fibonacci(1) == 1; assert fibonacci(10) == 55; assert fibonacci(7) == 13",
                "target_name": "fibonacci",
            },
            {
                "name": "is_prime",
                "test_code": "assert is_prime(2) == True; assert is_prime(7) == True; assert is_prime(4) == False; assert is_prime(1) == False; assert is_prime(97) == True",
                "target_name": "is_prime",
            },
        ],
    },
]


def trajectory_recombination_generator(seed: int) -> Problem:
    """Generate a TrajectoryRecombination problem.

    Each problem has 3-4 checkpoints, each adding a new function or method.
    The problems are designed with multiple independent functions so that
    recombination across trajectories is meaningful — one agent may excel
    at early checkpoints while another excels at later ones, and splicing
    their best components yields a better overall solution.

    Args:
        seed: Random seed for reproducible problem selection.

    Returns:
        A Problem with checkpoint metadata including test code and the
        target function/class name for each checkpoint.
    """
    rng = random.Random(seed)
    template = rng.choice(_RECOMBINATION_PROBLEMS)

    tests = template["checkpoints"]
    checkpoints = []
    for i, test in enumerate(tests):
        checkpoints.append({
            "name": test["name"],
            "test_code": test["test_code"],
            "target_name": test["target_name"],
            "test_index": i,
            "success_criteria": f"Test '{test['name']}' must pass",
        })

    cp_text = "\n".join(
        f"CHECKPOINT {i + 1}: {cp['name']}\n"
        f"  Test: {cp['test_code']}"
        for i, cp in enumerate(checkpoints)
    )

    return Problem(
        id=f"traj_recombo_{rng.randint(0, 99999)}",
        prompt=(
            f"TASK: {template['task']}\n"
            f"CHECKPOINTS:\n"
            f"{cp_text}\n\n"
            f"Build the solution checkpoint by checkpoint. Each checkpoint's code must\n"
            f"include ALL previous checkpoints' code. Your checkpoints will be RECOMBINED\n"
            f"with other agents' checkpoints — make them compatible.\n\n"
            f"Format:\n"
            f"CHECKPOINT <n>: <name>\n"
            f"CODE: ```python\n<all code so far>\n```\n"
            f"AUDIT: PASS or FAIL\n"
        ),
        difficulty=0.35 + 0.1 * rng.random(),
        metadata={
            "type": "trajectory_recombination",
            "task": template["task"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
        },
        token_budget=200 + len(checkpoints) * 250,
        source="generated",
    )


# Prevent pytest from collecting this generator function as a test
trajectory_recombination_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier — scores individual responses AND recombines across a batch
# ---------------------------------------------------------------------------


class TrajectoryRecombinationVerifier(Verifier):
    """Verifies trajectory recombination responses.

    This verifier has two modes:
      1. ``verify(response)`` — scores a SINGLE response by parsing its
         checkpoints and running each checkpoint's test. Returns a
         VerifierResult with per-checkpoint pass/fail in metadata.
      2. ``recombine(responses)`` — takes ALL N responses, extracts
         function/class definitions from each, and tries multiple
         recombination strategies to produce a better combined solution.

    The recombination strategies are:
      - **Sequential merge**: exec agent A's code, then agent B's code
        (B's definitions override A's where they conflict).
      - **Function-level patching**: start with the best agent's code,
        then for each failing test, try replacing the target function/class
        with another agent's version. Uses AST manipulation for precise
        replacement. Greedy: keep a patch only if it improves the score.

    The final recombined score is the best result across all strategies.
    """

    def __init__(self, checkpoints: list[dict]):
        """Initialize the verifier.

        Args:
            checkpoints: List of checkpoint dicts, each containing:
                - name: Human-readable checkpoint name.
                - test_code: Python test code to execute.
                - target_name: The function or class name this checkpoint
                  tests (used for targeted patching during recombination).
        """
        super().__init__()
        self._checkpoints = checkpoints
        self._all_tests = [cp["test_code"] for cp in checkpoints]

    # ------------------------------------------------------------------
    # Single-response verification
    # ------------------------------------------------------------------

    def verify(self, response: str) -> VerifierResult:
        """Score a single response by running each checkpoint's test.

        Parses the response into CHECKPOINT blocks, extracts the code at
        each checkpoint position, and runs the corresponding test. The
        score is the fraction of tests that pass.

        Args:
            response: The model's response containing checkpoint blocks.

        Returns:
            VerifierResult with score = tests_passed / total_tests.
            Metadata includes per-checkpoint pass/fail details.
        """
        executed = self._parse_checkpoints(response)

        if not executed:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No checkpoint blocks found",
                metadata={"test_results": {}, "tests_passed": 0, "total_tests": len(self._checkpoints)},
            )

        total = len(self._checkpoints)
        tests_passed = 0
        test_results: dict[int, bool] = {}

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                test_results[i] = False
                continue

            code = executed[i].get("code", "")
            passed = self._run_test(code, exp_cp["test_code"])
            test_results[i] = passed
            if passed:
                tests_passed += 1

        score = tests_passed / total if total > 0 else 0.0
        correct = tests_passed == total

        return VerifierResult(
            correct=correct,
            score=score,
            metadata={
                "test_results": test_results,
                "tests_passed": tests_passed,
                "total_tests": total,
            },
            diagnostics=f"Tests: {tests_passed}/{total}",
        )

    # ------------------------------------------------------------------
    # Batch recombination
    # ------------------------------------------------------------------

    def recombine(self, responses: list[str]) -> VerifierResult:
        """Recombine components from multiple trajectories.

        This is the core innovation of the environment. Instead of just
        scoring each response independently, we actively try to build a
        better solution by splicing together the best components from
        different agents.

        Strategies tried:
          1. Best individual final code (baseline).
          2. Sequential merge: for each pair (A, B), exec A then B.
          3. Function-level patching: start with best agent, replace
             failing functions with better versions from other agents.

        The best score across all strategies is the recombined score.

        Args:
            responses: List of N model responses.

        Returns:
            VerifierResult with:
              - score = best_individual * 0.4 + best_recombined * 0.6
              - metadata with per-agent pass counts, best agent index,
                recombination attempts, and per-strategy scores.
        """
        if not responses:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No responses provided",
                metadata={},
            )

        # 1. Parse all responses and extract final code from each
        all_final_codes: list[str] = []
        for resp in responses:
            checkpoints = self._parse_checkpoints(resp)
            final_code = checkpoints[-1].get("code", "") if checkpoints else ""
            all_final_codes.append(final_code)

        # 2. Run all tests on each agent's final code
        per_agent_passes: list[list[bool]] = []
        for code in all_final_codes:
            passes = self._run_all_tests_detailed(code, self._all_tests)
            per_agent_passes.append(passes)

        pass_counts = [sum(p) for p in per_agent_passes]
        total_tests = len(self._all_tests)

        # 3. Identify best individual
        if pass_counts:
            best_agent_idx = max(range(len(pass_counts)), key=lambda i: pass_counts[i])
            best_individual_score = pass_counts[best_agent_idx] / total_tests if total_tests > 0 else 0.0
        else:
            best_agent_idx = 0
            best_individual_score = 0.0

        # 4. Recombination strategies
        recombined_scores: list[float] = []
        recombination_details: list[str] = []

        # Strategy A: Sequential merge — exec A's code, then B's code
        n = len(all_final_codes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                code_i = self._extract_code(all_final_codes[i])
                code_j = self._extract_code(all_final_codes[j])
                if not code_i or not code_j:
                    continue
                merged = code_i + "\n\n" + code_j
                score = self._score_all_tests(merged, self._all_tests)
                recombined_scores.append(score)
                recombination_details.append(f"merge({i},{j})={score:.2f}")

        # Strategy B: Function-level patching using AST
        all_defs: list[dict[str, str]] = [
            self._extract_definitions(code) for code in all_final_codes
        ]

        if all_final_codes and best_agent_idx < len(all_final_codes):
            patched_code = self._extract_code(all_final_codes[best_agent_idx])
            current_score = best_individual_score

            # For each test that the best agent fails, try patching
            best_passes = per_agent_passes[best_agent_idx]
            for test_idx in range(min(len(best_passes), len(self._checkpoints))):
                if best_passes[test_idx]:
                    continue  # Already passing

                target_name = self._checkpoints[test_idx].get("target_name", "")
                if not target_name:
                    continue

                # Try replacing target with other agents' versions
                for agent_idx, defs in enumerate(all_defs):
                    if agent_idx == best_agent_idx or not defs:
                        continue
                    if target_name in defs:
                        trial_code = self._replace_definition(
                            patched_code, target_name, defs[target_name]
                        )
                        trial_score = self._score_all_tests(trial_code, self._all_tests)
                        if trial_score > current_score:
                            patched_code = trial_code
                            current_score = trial_score
                            recombination_details.append(
                                f"patch({target_name} from agent {agent_idx})={trial_score:.2f}"
                            )
                            break  # Move to next failing test

            recombined_scores.append(current_score)

        # Strategy C: Best-per-checkpoint assembly
        # For each checkpoint position, find the agent whose code at that
        # position passes the most tests cumulatively, and use that code.
        # Since each checkpoint includes all previous code, the code at
        # position i is a complete solution for tests 0..i.
        for agent_idx in range(n):
            checkpoints = self._parse_checkpoints(responses[agent_idx])
            for cp_idx in range(len(checkpoints)):
                code = checkpoints[cp_idx].get("code", "")
                score = self._score_all_tests(code, self._all_tests)
                recombined_scores.append(score)
                recombination_details.append(
                    f"checkpoint({agent_idx},{cp_idx})={score:.2f}"
                )

        # 5. Best recombined score
        best_recombined = max(recombined_scores) if recombined_scores else 0.0

        # 6. Final weighted score
        final_score = best_individual_score * 0.4 + best_recombined * 0.6
        correct = final_score >= 0.8

        return VerifierResult(
            correct=correct,
            score=final_score,
            metadata={
                "best_individual_score": best_individual_score,
                "recombined_score": best_recombined,
                "best_agent_idx": best_agent_idx,
                "per_agent_pass_counts": pass_counts,
                "recombination_attempts": len(recombined_scores),
                "recombination_details": recombination_details[:10],  # top 10 for logging
            },
            diagnostics=(
                f"Best individual={best_individual_score:.2f} "
                f"Recombined={best_recombined:.2f} "
                f"Final={final_score:.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # Code execution helpers
    # ------------------------------------------------------------------

    def _run_test(self, code: str, test_code: str) -> bool:
        """Execute code + test and return True if the test passes.

        Args:
            code: The solution code (may contain markdown fences).
            test_code: The test code to run after the solution.

        Returns:
            True if both code execution and test pass without exception.
        """
        code = self._extract_code(code)
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
            exec(test_code, namespace)
            return True
        except Exception:
            return False

    def _run_all_tests_detailed(self, code: str, tests: list[str]) -> list[bool]:
        """Run all tests against code, returning per-test pass/fail.

        Each test gets a fresh copy of the namespace (with the solution
        code pre-loaded) to prevent tests from interfering with each other.

        Args:
            code: The solution code (may contain markdown fences).
            tests: List of test code strings.

        Returns:
            List of booleans, one per test. True = test passed.
        """
        code = self._extract_code(code)
        if not code.strip():
            return [False] * len(tests)

        # Load the solution code once
        try:
            base_namespace: dict[str, Any] = {}
            exec(code, base_namespace)
        except Exception:
            return [False] * len(tests)

        results: list[bool] = []
        for test_code in tests:
            # Fresh copy of namespace for each test
            test_ns = dict(base_namespace)
            try:
                exec(test_code, test_ns)
                results.append(True)
            except Exception:
                results.append(False)
        return results

    def _score_all_tests(self, code: str, tests: list[str]) -> float:
        """Run all tests and return the fraction that pass.

        Args:
            code: The solution code (raw, no markdown fences).
            tests: List of test code strings.

        Returns:
            Fraction of tests passed, in [0, 1].
        """
        if not code.strip():
            return 0.0
        results = self._run_all_tests_detailed(code, tests)
        passed = sum(results)
        return passed / len(tests) if tests else 0.0

    def _extract_code(self, code: str) -> str:
        """Extract Python code from a string, handling markdown fences.

        If the code is wrapped in ```python ... ``` or ``` ... ```,
        extracts the inner content. Otherwise returns the code as-is.

        Args:
            code: Potentially markdown-wrapped code string.

        Returns:
            The raw Python code without markdown fences.
        """
        code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return code.strip()

    # ------------------------------------------------------------------
    # AST-based function extraction and replacement
    # ------------------------------------------------------------------

    def _extract_definitions(self, code: str) -> dict[str, str]:
        """Extract top-level function and class definitions from code.

        Uses Python's ast module to parse the code and extract the source
        of each top-level FunctionDef, AsyncFunctionDef, and ClassDef.

        Args:
            code: Python code (may contain markdown fences).

        Returns:
            Dict mapping definition name to source code string.
        """
        code = self._extract_code(code)
        if not code.strip():
            return {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        definitions: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                try:
                    source = ast.get_source_segment(code, node)
                    if source:
                        definitions[node.name] = source
                except Exception:
                    pass
        return definitions

    def _replace_definition(self, code: str, name: str, new_source: str) -> str:
        """Replace a function/class definition in code with a new version.

        Uses AST manipulation to find the definition with the given name
        and replace it with new_source. If the name is not found, the new
        definition is appended to the code.

        Args:
            code: The original Python code.
            name: The function or class name to replace.
            new_source: The new definition source code.

        Returns:
            The modified code with the definition replaced.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Can't parse original code; just append the new definition
            return code + "\n\n" + new_source

        new_body: list[ast.stmt] = []
        replaced = False
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ):
                # Replace with the new definition
                try:
                    new_tree = ast.parse(new_source)
                    new_body.extend(new_tree.body)
                    replaced = True
                except SyntaxError:
                    # Can't parse new source; keep original
                    new_body.append(node)
            else:
                new_body.append(node)

        if not replaced:
            # Name not found in original code; append the new definition
            try:
                new_tree = ast.parse(new_source)
                new_body.extend(new_tree.body)
            except SyntaxError:
                pass

        new_tree = ast.Module(body=new_body, type_ignores=[])
        try:
            return ast.unparse(new_tree)
        except Exception:
            # Fallback: append new definition to original code
            return code + "\n\n" + new_source

    # ------------------------------------------------------------------
    # Checkpoint parsing
    # ------------------------------------------------------------------

    def _parse_checkpoints(self, response: str) -> list[dict[str, str]]:
        """Parse CHECKPOINT n / CODE / AUDIT blocks from a response.

        Each checkpoint block has the format:
            CHECKPOINT <n>: <name>
            CODE: ```python
            <code>
            ```
            AUDIT: PASS or FAIL

        Args:
            response: The model's full response text.

        Returns:
            List of dicts with keys: name, code, audit_result.
        """
        executed: list[dict[str, str]] = []
        blocks = re.split(
            r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE
        )

        for block in blocks[1:]:
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""
            code = ""
            audit_result = "unknown"
            in_code = False
            code_lines: list[str] = []

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
            # Also check for inline CODE: format
            if not code:
                for line in lines[1:]:
                    if line.strip().upper().startswith("CODE:"):
                        code = line.split(":", 1)[1].strip()

            if name:
                executed.append({
                    "name": name,
                    "code": code,
                    "audit_result": audit_result,
                })

        return executed


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TrajectoryRecombinationEnv(BatchEnvBase):
    """TrajectoryRecombination: train composable code generation.

    A batch-aware environment where N parallel implementations are scored
    both individually AND as recombined components. The reward emphasizes
    recombination success (60%) over individual success (40%), training
    the model to produce modular, composable code.

    The environment overrides ``step()`` to store the batch responses so
    that ``_aggregate_scores()`` can invoke the verifier's recombination
    logic across all N responses.

    Args:
        problems: Fixed list of problems to sample from.
        problem_generator: Callable(seed) -> Problem. Defaults to
            trajectory_recombination_generator.
        reward_config: Configuration for the reward function.
        anti_pattern_detector: Detector for wasteful reasoning patterns.
        render_mode: "text" or None.
        batch_size: Number of parallel responses (default 16).
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = trajectory_recombination_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )
        self._current_batch_responses: list[str] = []

    def step(
        self, action: str | list[str]
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one batch step with recombination scoring.

        Stores the batch responses before delegating to the parent's
        ``step()`` so that ``_aggregate_scores()`` can access all N
        responses for recombination.

        Args:
            action: A list of N response strings (the batch), or a
                    single string (treated as N=1).

        Returns:
            (observation, reward, terminated, truncated, info)
            The info dict contains per-sample scores plus aggregate
            metrics including recombination results.
        """
        # Store responses for _aggregate_scores
        if isinstance(action, str):
            self._current_batch_responses = [action]
        else:
            self._current_batch_responses = list(action)

        return super().step(action)

    def _make_verifier(self, problem: Problem) -> Verifier:
        """Create a TrajectoryRecombinationVerifier for the given problem.

        Args:
            problem: The current Problem instance.

        Returns:
            A TrajectoryRecombinationVerifier configured with the
            problem's checkpoints.
        """
        return TrajectoryRecombinationVerifier(
            checkpoints=problem.metadata["checkpoints"],
        )

    def _check_format(self, response: str) -> float:
        """Check if the response follows the checkpoint format.

        Full bonus (1.0) if both CHECKPOINT and AUDIT markers are present.
        Partial bonus (0.5) if only CHECKPOINT markers are present.

        Args:
            response: The model's response text.

        Returns:
            Format bonus in [0, 1].
        """
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE) and "AUDIT:" in response.upper():
            return 1.0
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """Aggregate per-sample scores with recombination.

        This is the core of the TrajectoryRecombination environment.
        It computes:
          - best_individual: the best per-agent score
          - recombined_score: the best score from recombination strategies
          - recombined_correct: whether the recombined solution is correct
          - diversity: how diverse the batch responses are

        The final reward = best_individual * 0.4 + recombined_score * 0.6.

        Args:
            per_sample: List of per-sample score dicts from BatchEnvBase.step().

        Returns:
            Aggregate dict with all metrics for reward computation and logging.
        """
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_individual = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Run recombination across all batch responses
        recombined_score = 0.0
        recombined_correct = False
        recombination_metadata: dict[str, Any] = {}

        if (
            self._current_batch_responses
            and isinstance(self._verifier, TrajectoryRecombinationVerifier)
        ):
            recombined_result = self._verifier.recombine(self._current_batch_responses)
            recombined_score = recombined_result.score
            recombined_correct = recombined_result.correct
            recombination_metadata = recombined_result.metadata

        # Compute diversity: fraction of unique (score, correct) patterns
        diversity = self._compute_diversity(per_sample)

        # Final weighted reward
        reward = best_individual * 0.4 + recombined_score * 0.6

        # The overall "best score" is the max of individual and recombined
        best_score = max(best_individual, recombined_score)

        return {
            "best_score": best_score,
            "best_individual": best_individual,
            "recombined_score": recombined_score,
            "recombined_correct": recombined_correct,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": recombined_correct or any_correct,
            "recombination_metadata": recombination_metadata,
            "diagnostics": (
                f"Best={best_individual:.2f} "
                f"Recombined={recombined_score:.2f} "
                f"Diversity={diversity:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _compute_diversity(self, per_sample: list[dict]) -> float:
        """Compute the diversity of the batch responses.

        Diversity is measured as the fraction of unique (score, correct)
        pairs across all samples. High diversity means agents are producing
        qualitatively different solutions, which is valuable for
        recombination — diverse solutions offer more components to splice.

        Args:
            per_sample: List of per-sample score dicts.

        Returns:
            Diversity score in [0, 1]. 0 = all identical, 1 = all unique.
        """
        if len(per_sample) <= 1:
            return 0.0
        patterns: set[tuple[float, bool]] = set()
        for s in per_sample:
            patterns.add((round(s["verifier_score"], 2), s["correct"]))
        return len(patterns) / len(per_sample)

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response.

        For this environment, the full response IS the answer (it contains
        all checkpoint blocks). No extraction needed.

        Args:
            response: The model's response text.

        Returns:
            The response unchanged.
        """
        return response
