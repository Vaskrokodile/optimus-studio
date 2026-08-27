"""
AdversarialCodeTransformEnv: Train robustness to semantics-preserving code transformations.

Environment concept:
  An adversary transforms code using AST-based transformations that preserve
  semantics but hurt readability (rename variables, add dead code, convert for
  loops to while, inline constants, expand ternaries). The model must still
  understand the transformed code and complete the task (fix a bug or implement
  a missing piece).

  This trains ROBUSTNESS — the model learns to see through obfuscation and
  focus on semantics rather than surface syntax. A model that breaks when
  variable names change or loops are rewritten is fragile; this environment
  builds resilience.

Transformations (all semantics-preserving):
  1. rename_variables  — Rename all local variables to v1, v2, v3, ...
  2. add_dead_code     — Insert `if False: <something>` blocks (noise)
  3. for_to_while      — Convert `for i in range(...)` to while loops
  4. inline_constant   — Replace named constants with their literal values
  5. expand_ternary    — Convert `x if cond else y` to if/else blocks

Generator:
  1. Pick a base coding problem (simple function with a bug to fix)
  2. Apply 1-3 random transformations to create the adversarial version
  3. The model must understand the transformed code and fix the bug
  4. Tests verify correctness

Verifier:
  - Extract the solution code from the response
  - Run ALL tests against it (REAL EXECUTION)
  - Score: tests_passed / total_tests + identification_bonus
  - correct = all tests pass

Batch reward: best_score * 0.8 + diversity * 0.2
"""

from __future__ import annotations

import ast
import builtins
import copy
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Base problems: simple functions with bugs to fix
# ---------------------------------------------------------------------------

_BASE_PROBLEMS = [
    {
        "name": "is_palindrome",
        "description": "Check if a string is a palindrome (case-insensitive).",
        "task": (
            "Fix the is_palindrome function so it handles case correctly. "
            "A palindrome reads the same forwards and backwards, ignoring case."
        ),
        "code": (
            "def is_palindrome(s):\n"
            "    cleaned = s\n"
            "    return True if len(cleaned) <= 1 else cleaned == cleaned[::-1]\n"
        ),
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('Racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('a') == True",
            "assert is_palindrome('') == True",
            "assert is_palindrome('AbA') == True",
        ],
    },
    {
        "name": "fibonacci",
        "description": "Compute the nth Fibonacci number.",
        "task": (
            "Fix the fibonacci function. It currently returns incorrect values "
            "for n >= 2 due to an off-by-one error in the loop range."
        ),
        "code": (
            "def fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    if n == 1:\n"
            "        return 1\n"
            "    a = 0\n"
            "    b = 1\n"
            "    for i in range(2, n):\n"
            "        temp = a + b\n"
            "        a = b\n"
            "        b = temp\n"
            "    return b\n"
        ),
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(2) == 1",
            "assert fibonacci(5) == 5",
            "assert fibonacci(10) == 55",
            "assert fibonacci(7) == 13",
        ],
    },
    {
        "name": "factorial",
        "description": "Compute the factorial of a non-negative integer.",
        "task": (
            "Fix the factorial function. It has an off-by-one error in the loop "
            "range that causes it to return incorrect values for n >= 2."
        ),
        "code": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n):\n"
            "        result = result * i\n"
            "    return result\n"
        ),
        "tests": [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(2) == 2",
            "assert factorial(5) == 120",
            "assert factorial(3) == 6",
            "assert factorial(7) == 5040",
        ],
    },
    {
        "name": "merge_sorted",
        "description": "Merge two sorted lists into one sorted list.",
        "task": (
            "Fix the merge_sorted function. The comparison operator is wrong, "
            "causing elements to be appended in the wrong order."
        ),
        "code": (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = 0\n"
            "    j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] > b[j]:\n"
            "            result.append(a[i])\n"
            "            i = i + 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j = j + 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result\n"
        ),
        "tests": [
            "assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]",
            "assert merge_sorted([1,2], [3,4]) == [1,2,3,4]",
            "assert merge_sorted([], [1,2]) == [1,2]",
            "assert merge_sorted([1,2], []) == [1,2]",
            "assert merge_sorted([1,1,2], [1,2,3]) == [1,1,1,2,2,3]",
        ],
    },
    {
        "name": "is_prime",
        "description": "Check if a number is prime.",
        "task": (
            "Fix the is_prime function. The boundary check is wrong, causing it "
            "to incorrectly classify 1 as prime."
        ),
        "code": (
            "def is_prime(n):\n"
            "    if n < 1:\n"
            "        return False\n"
            "    for i in range(2, n):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        "tests": [
            "assert is_prime(2) == True",
            "assert is_prime(3) == True",
            "assert is_prime(4) == False",
            "assert is_prime(1) == False",
            "assert is_prime(17) == True",
            "assert is_prime(25) == False",
        ],
    },
    {
        "name": "count_vowels",
        "description": "Count the number of vowels in a string.",
        "task": (
            "Fix the count_vowels function. The vowels string is missing a "
            "vowel, causing it to undercount."
        ),
        "code": (
            "def count_vowels(s):\n"
            "    vowels = 'aeio'\n"
            "    count = 0\n"
            "    for c in s:\n"
            "        if c in vowels:\n"
            "            count = count + 1\n"
            "    return count\n"
        ),
        "tests": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('education') == 5",
            "assert count_vowels('aeiou') == 5",
            "assert count_vowels('') == 0",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('queue') == 4",
        ],
    },
]


# ---------------------------------------------------------------------------
# AdversarialCodeTransformer: AST-based semantics-preserving transformations
# ---------------------------------------------------------------------------


class AdversarialCodeTransformer:
    """
    Applies AST-based transformations that preserve semantics but hurt readability.

    Each transformation uses Python's `ast` module to parse, transform, and
    unparse the code. All transformations are semantics-preserving: the
    transformed code behaves identically to the original.

    Transformations:
        rename_variables  — Rename local variables to v1, v2, v3, ...
        add_dead_code     — Insert `if False: <something>` blocks
        for_to_while      — Convert `for i in range(...)` to while loops
        inline_constant   — Replace named constants with literal values
        expand_ternary    — Convert ternary expressions to if/else blocks
    """

    TRANSFORM_NAMES = [
        "rename_variables",
        "add_dead_code",
        "for_to_while",
        "inline_constant",
        "expand_ternary",
    ]

    # Keywords the verifier looks for when scoring transform identification
    TRANSFORM_KEYWORDS: dict[str, list[str]] = {
        "rename_variables": ["rename", "renamed", "variable name", "obfuscated variable"],
        "add_dead_code": ["dead code", "dead", "unreachable", "if false"],
        "for_to_while": ["while loop", "for to while", "converted to while", "while equivalent"],
        "inline_constant": ["inline", "inlined", "constant", "literal"],
        "expand_ternary": ["ternary", "if/else", "if-else", "expanded"],
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def transform(self, code: str, transforms: list[str]) -> tuple[str, list[str]]:
        """
        Apply the given transforms to the code in canonical order.

        Returns (transformed_code, applied_transforms) where applied_transforms
        contains only the transforms that actually changed the code.
        """
        result = code
        applied: list[str] = []

        # Apply in a fixed canonical order for consistency
        ordered = [t for t in self.TRANSFORM_NAMES if t in transforms]

        for tname in ordered:
            method = getattr(self, tname)
            try:
                new_code = method(result)
                # Verify the result is valid Python
                ast.parse(new_code)
                if new_code.strip() != result.strip():
                    result = new_code
                    applied.append(tname)
            except Exception:
                # If a transform fails, skip it and keep the current code
                continue

        return result, applied

    # -- rename_variables -----------------------------------------------

    def rename_variables(self, code: str) -> str:
        """
        Rename all local variables to v1, v2, v3, etc.

        Collects variable names from Store-context Name nodes, function
        arguments, and for-loop targets. Does not rename builtins or the
        function name itself (FunctionDef names are not Name nodes).
        """
        tree = ast.parse(code)
        builtins_set = set(dir(builtins))

        # Collect variable names in order of appearance
        names: list[str] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in seen and node.id not in builtins_set:
                    names.append(node.id)
                    seen.add(node.id)
            elif isinstance(node, ast.arg):
                if node.arg not in seen and node.arg not in builtins_set:
                    names.append(node.arg)
                    seen.add(node.arg)

        if not names:
            return code

        mapping = {name: f"v{i + 1}" for i, name in enumerate(names)}

        class _Renamer(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in mapping:
                    n.id = mapping[n.id]
                return n

            def visit_arg(self, n: ast.arg) -> ast.AST:
                if n.arg in mapping:
                    n.arg = mapping[n.arg]
                return n

        _Renamer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    # -- add_dead_code --------------------------------------------------

    def add_dead_code(self, code: str) -> str:
        """
        Insert `if False: <something>` blocks into function bodies.

        The dead code is never executed (condition is always False), so
        semantics are preserved. Inserts 1-2 blocks per function at random
        positions (after the docstring if present).
        """
        tree = ast.parse(code)
        rng = self._rng

        dead_bodies = [
            [ast.Pass()],
            [ast.Assign(
                targets=[ast.Name(id="_unused", ctx=ast.Store())],
                value=ast.Constant(value=0),
            )],
            [ast.Assign(
                targets=[ast.Name(id="_noop", ctx=ast.Store())],
                value=ast.Constant(value=None),
            )],
        ]

        class _DeadCodeInserter(ast.NodeTransformer):
            def visit_FunctionDef(self, n: ast.FunctionDef) -> ast.AST:
                self.generic_visit(n)

                # Skip docstring if present
                insert_idx = 0
                if n.body and isinstance(n.body[0], ast.Expr) and \
                   isinstance(n.body[0].value, ast.Constant) and \
                   isinstance(n.body[0].value.value, str):
                    insert_idx = 1

                # Insert dead code at the beginning (after docstring)
                dead1 = ast.If(
                    test=ast.Constant(value=False),
                    body=copy.deepcopy(rng.choice(dead_bodies)),
                    orelse=[],
                )
                n.body.insert(insert_idx, dead1)

                # Optionally insert a second dead code block in the middle
                if len(n.body) > 3 and rng.random() < 0.5:
                    dead2 = ast.If(
                        test=ast.Constant(value=False),
                        body=copy.deepcopy(rng.choice(dead_bodies)),
                        orelse=[],
                    )
                    mid = len(n.body) // 2
                    n.body.insert(mid, dead2)

                return n

        _DeadCodeInserter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    # -- for_to_while ---------------------------------------------------

    def for_to_while(self, code: str) -> str:
        """
        Convert `for i in range(...)` loops to equivalent while loops.

        Handles range with 1, 2, or 3 arguments. The while loop uses an
        explicit counter variable with the same name as the for-loop target.
        For negative step values, uses `>` comparison; otherwise uses `<`.
        """
        tree = ast.parse(code)

        class _ForToWhileTransformer(ast.NodeTransformer):
            def visit_For(self, n: ast.For) -> ast.AST:
                self.generic_visit(n)

                # Only convert for x in range(...) loops
                if not (isinstance(n.iter, ast.Call) and
                        isinstance(n.iter.func, ast.Name) and
                        n.iter.func.id == "range"):
                    return n

                # Target must be a simple Name
                if not isinstance(n.target, ast.Name):
                    return n

                args = n.iter.args
                if len(args) == 1:
                    start = ast.Constant(value=0)
                    stop = copy.deepcopy(args[0])
                    step = ast.Constant(value=1)
                elif len(args) == 2:
                    start = copy.deepcopy(args[0])
                    stop = copy.deepcopy(args[1])
                    step = ast.Constant(value=1)
                elif len(args) == 3:
                    start = copy.deepcopy(args[0])
                    stop = copy.deepcopy(args[1])
                    step = copy.deepcopy(args[2])
                else:
                    return n

                loop_var = n.target.id

                # Determine comparison operator based on step sign
                if isinstance(step, ast.Constant) and \
                   isinstance(step.value, (int, float)) and step.value < 0:
                    compare_op = ast.Gt()
                else:
                    compare_op = ast.Lt()

                # init: loop_var = start
                init = ast.Assign(
                    targets=[ast.Name(id=loop_var, ctx=ast.Store())],
                    value=start,
                )

                # condition: while loop_var < stop (or > for negative step)
                condition = ast.Compare(
                    left=ast.Name(id=loop_var, ctx=ast.Load()),
                    ops=[compare_op],
                    comparators=[stop],
                )

                # increment: loop_var += step
                increment = ast.AugAssign(
                    target=ast.Name(id=loop_var, ctx=ast.Store()),
                    op=ast.Add(),
                    value=step,
                )

                new_body = list(n.body) + [increment]

                while_loop = ast.While(
                    test=condition,
                    body=new_body,
                    orelse=list(n.orelse),
                )

                # Return a list to splice init + while into the parent body
                return [init, while_loop]

        _ForToWhileTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    # -- inline_constant ------------------------------------------------

    def inline_constant(self, code: str) -> str:
        """
        Replace named constants with their literal values.

        Finds variables assigned a constant value (e.g., `LIMIT = 10`) that
        are never reassigned, and replaces all subsequent uses with the
        literal value. The original assignment remains as dead code.
        """
        tree = ast.parse(code)

        # Find variables assigned to constants
        assignments: dict[str, ast.Constant] = {}
        reassigned: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant):
                            if target.id in assignments:
                                # Assigned to a constant, then assigned again
                                reassigned.add(target.id)
                            else:
                                assignments[target.id] = node.value
                        else:
                            # Assigned a non-constant value
                            reassigned.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    reassigned.add(node.target.id)
            elif isinstance(node, ast.For):
                # For-loop targets are reassigned each iteration
                if isinstance(node.target, ast.Name):
                    reassigned.add(node.target.id)

        # Only keep variables that are never reassigned
        constants = {
            k: v for k, v in assignments.items()
            if k not in reassigned
        }

        if not constants:
            return code

        class _ConstantInliner(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                if isinstance(n.ctx, ast.Load) and n.id in constants:
                    return copy.deepcopy(constants[n.id])
                return n

        _ConstantInliner().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    # -- expand_ternary -------------------------------------------------

    def expand_ternary(self, code: str) -> str:
        """
        Convert ternary expressions `x if cond else y` to if/else blocks.

        Handles two cases:
          - `target = x if cond else y` -> if/else with assignment in each branch
          - `return x if cond else y`   -> if/else with return in each branch

        Only transforms top-level ternaries in assignments and returns;
        nested ternaries inside other expressions are left unchanged.
        """
        tree = ast.parse(code)

        class _TernaryExpander(ast.NodeTransformer):
            def visit_Assign(self, n: ast.Assign) -> ast.AST:
                self.generic_visit(n)
                if isinstance(n.value, ast.IfExp):
                    if_exp = n.value
                    if_body = [ast.Assign(
                        targets=copy.deepcopy(n.targets),
                        value=if_exp.body,
                    )]
                    else_body = [ast.Assign(
                        targets=copy.deepcopy(n.targets),
                        value=if_exp.orelse,
                    )]
                    return ast.If(
                        test=if_exp.test,
                        body=if_body,
                        orelse=else_body,
                    )
                return n

            def visit_Return(self, n: ast.Return) -> ast.AST:
                self.generic_visit(n)
                if isinstance(n.value, ast.IfExp):
                    if_exp = n.value
                    if_body = [ast.Return(value=if_exp.body)]
                    else_body = [ast.Return(value=if_exp.orelse)]
                    return ast.If(
                        test=if_exp.test,
                        body=if_body,
                        orelse=else_body,
                    )
                return n

        _TernaryExpander().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def adversarial_code_transform_generator(seed: int) -> Problem:
    """
    Generate an adversarial code transform problem.

    Picks a random base problem, applies 1-3 random AST transformations,
    and creates a prompt asking the model to fix the bug in the transformed
    code.
    """
    rng = random.Random(seed)
    template = rng.choice(_BASE_PROBLEMS)

    # Choose 1-3 random transformations, retrying until at least one applies
    transformer = AdversarialCodeTransformer(seed=seed)
    transformed_code = template["code"]
    applied_transforms: list[str] = []

    for attempt in range(5):
        num_transforms = rng.randint(1, 3)
        chosen = rng.sample(
            AdversarialCodeTransformer.TRANSFORM_NAMES,
            min(num_transforms, len(AdversarialCodeTransformer.TRANSFORM_NAMES)),
        )
        transformed_code, applied_transforms = transformer.transform(
            template["code"], chosen
        )
        if applied_transforms:
            break

    # If no transforms applied after retries, use original code
    if not applied_transforms:
        transformed_code = template["code"]
        applied_transforms = ["none"]

    # Build the transform list string
    transform_list = ", ".join(applied_transforms) if applied_transforms else "none"

    # Build test text
    tests_text = "\n".join(f"  {t}" for t in template["tests"])

    # Build the prompt
    prompt = (
        f"ADVERSARIAL TRANSFORM: {transform_list}\n"
        f"TASK: {template['task']}\n"
        f"CODE (adversarially transformed):\n"
        f"```python\n{transformed_code}\n```\n"
        f"TESTS:\n{tests_text}\n\n"
        f"Your solution must pass ALL tests. The code has been adversarially "
        f"transformed (rename variables, add dead code, etc.) but is "
        f"semantically equivalent to the original. Understand what it does, "
        f"then complete the task.\n\n"
        f"SOLUTION: ```python\n<your complete code>\n```"
    )

    # Difficulty scales with number of transforms applied
    difficulty = 0.3 + 0.1 * len(applied_transforms) + 0.05 * rng.random()

    return Problem(
        id=f"adv_code_transform_{template['name']}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=min(0.9, difficulty),
        metadata={
            "type": "adversarial_code_transform",
            "function_name": template["name"],
            "description": template["description"],
            "task": template["task"],
            "original_code": template["code"],
            "transformed_code": transformed_code,
            "tests": template["tests"],
            "applied_transforms": applied_transforms,
        },
        token_budget=2048,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — runs real tests against the extracted solution
# ---------------------------------------------------------------------------


class AdversarialCodeVerifier(Verifier):
    """
    Verifies a solution by executing it and running all tests.

    Extracts the solution code from the response (looking for SOLUTION:
    ```python ... ``` or the last python code block), executes it, and runs
    each test case. Score = tests_passed / total_tests + identification_bonus.

    The identification bonus rewards the model for correctly identifying
    which adversarial transforms were applied (by mentioning keywords like
    "rename", "dead code", "while loop", "inline", "ternary").
    """

    def __init__(
        self,
        tests: list[str],
        applied_transforms: list[str],
        timeout: float = 5.0,
    ):
        super().__init__()
        self._tests = tests
        self._applied_transforms = applied_transforms
        self._timeout = timeout

    def verify(self, response: str) -> VerifierResult:
        # Extract solution code
        code = self._extract_solution(response)
        if not code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No solution code found in response",
            )

        # Execute the solution and run tests
        passed, total, errors = self._run_tests(code)

        base_score = passed / total if total > 0 else 0.0

        # Identification bonus
        identified = self._count_identified_transforms(response)
        identification_bonus = min(0.15, 0.05 * identified)
        score = min(1.0, base_score + identification_bonus)

        correct = (passed == total) and (total > 0)

        diagnostics_parts = [
            f"Tests: {passed}/{total}",
        ]
        if errors:
            diagnostics_parts.append(f"Errors: {errors[:2]}")
        diagnostics_parts.append(
            f"Transforms identified: {identified}/{len(self._applied_transforms)}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "tests_passed": passed,
                "total_tests": total,
                "test_score": base_score,
                "transforms_identified": identified,
                "identification_bonus": identification_bonus,
            },
            diagnostics=" | ".join(diagnostics_parts),
        )

    def _extract_solution(self, response: str) -> str:
        """Extract the solution code from the model's response."""
        # Try SOLUTION: ```python ... ```
        match = re.search(
            r"SOLUTION:\s*```python\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try SOLUTION: ``` ... ```
        match = re.search(
            r"SOLUTION:\s*```\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try the last ```python ... ``` block
        matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Try the last ``` ... ``` block
        matches = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Fall back to the entire response (might be just code)
        stripped = response.strip()
        if stripped:
            return stripped
        return ""

    def _run_tests(self, code: str) -> tuple[int, int, list[str]]:
        """Execute the solution code and run all tests. Returns (passed, total, errors)."""
        namespace: dict[str, Any] = {}

        # Execute the solution code
        try:
            exec(code, namespace)
        except SyntaxError as e:
            return 0, len(self._tests), [f"SyntaxError: {e}"]
        except Exception as e:
            return 0, len(self._tests), [f"Execution error: {type(e).__name__}: {e}"]

        # Run each test
        passed = 0
        errors: list[str] = []

        for test in self._tests:
            try:
                exec(test, namespace)
                passed += 1
            except AssertionError:
                errors.append(f"AssertionError: {test}")
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

        return passed, len(self._tests), errors

    def _count_identified_transforms(self, response: str) -> int:
        """Count how many applied transforms the model correctly identified."""
        response_lower = response.lower()
        count = 0

        for tname in self._applied_transforms:
            if tname == "none":
                continue
            keywords = AdversarialCodeTransformer.TRANSFORM_KEYWORDS.get(tname, [])
            if any(kw in response_lower for kw in keywords):
                count += 1

        return count


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AdversarialCodeTransformEnv(BatchEnvBase):
    """
    AdversarialCodeTransformEnv: robustness to semantics-preserving transforms.

    The model receives adversarially transformed code (renamed variables,
    dead code, while loops instead of for, inlined constants, expanded
    ternaries) and must understand it well enough to fix a bug.

    Batch reward: best_score * 0.8 + diversity * 0.2.

    Diversity is measured as the fraction of distinct normalized code
    solutions among the N batch samples.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = adversarial_code_transform_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return AdversarialCodeVerifier(
            tests=meta["tests"],
            applied_transforms=meta["applied_transforms"],
        )

    def _check_format(self, response: str) -> float:
        """Check if the response follows the expected SOLUTION format."""
        if re.search(
            r"SOLUTION:\s*```python\n.*?```",
            response,
            re.DOTALL | re.IGNORECASE,
        ):
            return 1.0
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """
        Aggregate per-sample scores: best_score * 0.8 + diversity * 0.2.

        If all samples are wrong (best_score = 0), reward is 0 regardless
        of diversity.
        """
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Compute diversity as fraction of distinct code solutions
        diversity = self._compute_diversity(per_sample)

        # Reward: best_score * 0.8 + diversity * 0.2
        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.8 + diversity * 0.2
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Diversity={diversity:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _compute_diversity(self, per_sample: list[dict]) -> float:
        """
        Compute diversity as the fraction of distinct normalized code solutions.

        Extracts the solution code from each response, normalizes whitespace,
        and counts distinct solutions.
        """
        if not per_sample:
            return 0.0

        solutions: set[str] = set()

        for s in per_sample:
            resp = s.get("response", "")
            # Try to extract solution code
            match = re.search(
                r"```python\n(.*?)```",
                resp,
                re.DOTALL,
            )
            if match:
                code = match.group(1).strip()
            else:
                code = resp.strip()

            # Normalize: remove whitespace differences
            normalized = re.sub(r"\s+", " ", code).strip()
            solutions.add(normalized)

        if not solutions:
            return 0.0

        n = len(per_sample)
        distinct = len(solutions)
        diversity = min(1.0, distinct / max(n, 1))
        return diversity

    def _extract_answer(self, response: str) -> str:
        return response
