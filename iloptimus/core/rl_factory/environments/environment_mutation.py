"""
EnvironmentMutationEnv: Train robustness to environment mutations.

Environment concept:
  The same underlying problem is presented with different world dynamics:
  hidden constraints, misleading information, partial observability,
  requirement changes, and resource constraints. The model must:
    1. Recognize that the environment may be adversarial or incomplete
    2. Produce a solution that satisfies the ACTUAL constraints (not just
       the stated ones)
    3. Discover and articulate what mutation was applied

  This trains meta-reasoning about problem statements — the model learns
  to NOT trust prompts blindly, to infer missing information from tests,
  and to adapt when requirements shift.

  Five mutation types:
    - hidden_constraint:  A constraint exists but is not stated. Tests enforce it.
    - misleading_info:    The prompt contains incorrect hints. The model must
                          recognize and ignore them.
    - partial_observability: Part of the problem is redacted. The model must
                          infer the full specification from tests.
    - requirement_change: A requirement is modified from the base problem.
                          The model must notice and adapt.
    - resource_constraint: A tight constraint is imposed (no loops, no builtins,
                          etc.). The model must solve within it.

  This is a batch environment: N parallel solutions are scored.
  reward = best_score * 0.7 + diversity * 0.3

  Per-sample score = tests_passed * 0.6 + mutation_handled * 0.4
  correct = all tests pass AND mutation handled
"""

from __future__ import annotations

import random
import re
import traceback
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Mutation type constants
# ---------------------------------------------------------------------------

MUTATION_TYPES: list[str] = [
    "hidden_constraint",
    "misleading_info",
    "partial_observability",
    "requirement_change",
    "resource_constraint",
]

# ---------------------------------------------------------------------------
# Base problems — each is a self-contained coding task with a reference
# solution, base tests, and per-mutation configuration.
# ---------------------------------------------------------------------------

_BASE_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "sort_list",
        "task": "Write a function `sort_list(lst)` that sorts a list of numbers.",
        "func_name": "sort_list",
        "base_code": "def sort_list(lst):\n    return sorted(lst)",
        "base_tests": [
            "assert sort_list([3, 1, 2]) == [1, 2, 3]",
            "assert sort_list([]) == []",
            "assert sort_list([1]) == [1]",
            "assert sort_list([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]",
        ],
        "mutations": {
            "hidden_constraint": {
                "description": (
                    "The sort must be stable — equal elements preserve their "
                    "original relative order."
                ),
                "prompt_task": (
                    "Write a function `sort_list(lst)` that sorts a list. "
                    "The function should work for any list of comparable elements."
                ),
                "extra_tests": [
                    "assert sort_list([(3, 'a'), (1, 'b'), (3, 'c'), (1, 'd')]) == "
                    "[(1, 'b'), (1, 'd'), (3, 'a'), (3, 'c')]",
                    "assert sort_list([(2, 'x'), (1, 'y'), (2, 'z')]) == "
                    "[(1, 'y'), (2, 'x'), (2, 'z')]",
                ],
                "forbidden_patterns": [],
            },
            "misleading_info": {
                "description": (
                    "The hint claims bubble sort is the most efficient algorithm, "
                    "but bubble sort is O(n^2). The model should use a better "
                    "algorithm."
                ),
                "misleading_hint": (
                    "Hint: For maximum efficiency, use bubble sort — it runs in "
                    "O(n) time and is the fastest known sorting algorithm."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"bubble",
                    r"for\s+\w+\s+in\s+range\s*\(",
                    r"while\s+.*swap",
                ],
            },
            "partial_observability": {
                "description": (
                    "The prompt only shows 2 of 5 data fields. The model must "
                    "infer the full record structure from the tests."
                ),
                "prompt_task": (
                    "Write a function `sort_list(records)` that sorts a list of "
                    "record dictionaries. Each record has the fields: 'name' "
                    "(string) and 'age' (int). Sort by age.\n\n"
                    "Note: The full record structure is partially shown. Only 2 "
                    "of 5 fields are visible above. Infer the remaining fields "
                    "from the tests."
                ),
                "extra_tests": [
                    "assert sort_list([{'name': 'Alice', 'age': 30, 'email': 'a@x.com', "
                    "'phone': '123', 'address': 'St 1'}]) == "
                    "[{'name': 'Alice', 'age': 30, 'email': 'a@x.com', "
                    "'phone': '123', 'address': 'St 1'}]",
                    "assert sort_list([{'name': 'Bob', 'age': 25, 'email': 'b@x.com', "
                    "'phone': '456', 'address': 'St 2'}, "
                    "{'name': 'Alice', 'age': 30, 'email': 'a@x.com', "
                    "'phone': '123', 'address': 'St 1'}]) == "
                    "[{'name': 'Bob', 'age': 25, 'email': 'b@x.com', "
                    "'phone': '456', 'address': 'St 2'}, "
                    "{'name': 'Alice', 'age': 30, 'email': 'a@x.com', "
                    "'phone': '123', 'address': 'St 1'}]",
                ],
                "forbidden_patterns": [],
            },
            "requirement_change": {
                "description": (
                    "The base problem asks for ascending sort, but the mutation "
                    "requires DESCENDING order."
                ),
                "prompt_task": (
                    "Write a function `sort_list(lst)` that sorts a list of "
                    "numbers in DESCENDING order (largest first)."
                ),
                "extra_tests": [
                    "assert sort_list([3, 1, 2]) == [3, 2, 1]",
                    "assert sort_list([5, 4, 3, 2, 1]) == [5, 4, 3, 2, 1]",
                    "assert sort_list([1, 2, 3]) == [3, 2, 1]",
                ],
                "forbidden_patterns": [],
            },
            "resource_constraint": {
                "description": (
                    "No loops (for/while) are allowed. The solution must use "
                    "recursion."
                ),
                "constraint_text": (
                    "CONSTRAINT: No loops (for, while) are allowed. You must use "
                    "recursion to implement the sort."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"\bfor\b\s+\w+\s+in\b",
                    r"\bwhile\b\s*",
                ],
            },
        },
    },
    {
        "id": "find_max",
        "task": "Write a function `find_max(lst)` that returns the maximum element in a list.",
        "func_name": "find_max",
        "base_code": "def find_max(lst):\n    return max(lst) if lst else None",
        "base_tests": [
            "assert find_max([1, 3, 2]) == 3",
            "assert find_max([5]) == 5",
            "assert find_max([-1, -2, -3]) == -1",
        ],
        "mutations": {
            "hidden_constraint": {
                "description": (
                    "The function must return None for an empty list, but this "
                    "is not stated in the prompt."
                ),
                "prompt_task": (
                    "Write a function `find_max(lst)` that returns the maximum "
                    "element in a list of numbers."
                ),
                "extra_tests": [
                    "assert find_max([]) is None",
                    "assert find_max([0]) == 0",
                ],
                "forbidden_patterns": [],
            },
            "misleading_info": {
                "description": (
                    "The hint suggests sorting the list first for O(1) access, "
                    "but sorting is O(n log n) — it is NOT O(1). The model "
                    "should find the max directly in O(n)."
                ),
                "misleading_hint": (
                    "Hint: For O(1) time complexity, sort the list first and "
                    "then return the first element. This is the most efficient "
                    "approach."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"\.sort\s*\(\s*\)",
                    r"sorted\s*\(",
                ],
            },
            "partial_observability": {
                "description": (
                    "The prompt says 'sequence of numbers' but doesn't specify "
                    "that inputs can be floats, negative, or mixed types. The "
                    "tests reveal the full input space."
                ),
                "prompt_task": (
                    "Write a function `find_max(lst)` that returns the maximum "
                    "element in a sequence.\n\n"
                    "Note: The input specification is partially hidden. Infer "
                    "the full input types from the tests."
                ),
                "extra_tests": [
                    "assert find_max([3.14, 2.71, 1.41]) == 3.14",
                    "assert find_max([-5, -10, -3]) == -3",
                    "assert find_max([0, -1, 1]) == 1",
                ],
                "forbidden_patterns": [],
            },
            "requirement_change": {
                "description": (
                    "The base problem asks for the maximum, but the mutation "
                    "requires the MINIMUM element."
                ),
                "prompt_task": (
                    "Write a function `find_max(lst)` that returns the MINIMUM "
                    "element in a list of numbers (smallest value)."
                ),
                "extra_tests": [
                    "assert find_max([1, 3, 2]) == 1",
                    "assert find_max([5, 4, 3, 2, 1]) == 1",
                    "assert find_max([-1, -2, -3]) == -3",
                ],
                "forbidden_patterns": [],
            },
            "resource_constraint": {
                "description": (
                    "No built-in max() or min() functions are allowed. The "
                    "solution must implement the logic manually."
                ),
                "constraint_text": (
                    "CONSTRAINT: You may NOT use the built-in max() or min() "
                    "functions. Implement the comparison logic yourself."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"\bmax\s*\(",
                    r"\bmin\s*\(",
                ],
            },
        },
    },
    {
        "id": "is_palindrome",
        "task": "Write a function `is_palindrome(s)` that checks if a string is a palindrome.",
        "func_name": "is_palindrome",
        "base_code": "def is_palindrome(s):\n    return s == s[::-1]",
        "base_tests": [
            'assert is_palindrome("racecar") == True',
            'assert is_palindrome("hello") == False',
            'assert is_palindrome("") == True',
        ],
        "mutations": {
            "hidden_constraint": {
                "description": (
                    "The check must be case-insensitive, but the prompt does "
                    "not mention this."
                ),
                "prompt_task": (
                    "Write a function `is_palindrome(s)` that checks if a string "
                    "is a palindrome (reads the same forwards and backwards)."
                ),
                "extra_tests": [
                    'assert is_palindrome("RaceCar") == True',
                    'assert is_palindrome("AbA") == True',
                    'assert is_palindrome("Hello") == False',
                ],
                "forbidden_patterns": [],
            },
            "misleading_info": {
                "description": (
                    "The hint claims converting to a list and using recursion "
                    "gives O(1) space, but list conversion uses O(n) space. "
                    "The model should use a direct comparison."
                ),
                "misleading_hint": (
                    "Hint: For O(1) space complexity, convert the string to a "
                    "list of characters and use recursion to compare elements. "
                    "This avoids creating any copies."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"list\s*\(\s*s\s*\)",
                    r"def\s+\w+\s*\([^)]*\):\s*.*\n.*\1\s*\(",
                ],
            },
            "partial_observability": {
                "description": (
                    "The prompt says 'string' but the function must also handle "
                    "lists and tuples. The tests reveal this."
                ),
                "prompt_task": (
                    "Write a function `is_palindrome(s)` that checks if a "
                    "sequence is a palindrome.\n\n"
                    "Note: The input type is partially hidden. Infer the full "
                    "set of supported input types from the tests."
                ),
                "extra_tests": [
                    "assert is_palindrome([1, 2, 1]) == True",
                    "assert is_palindrome([1, 2, 3]) == False",
                    "assert is_palindrome((1, 2, 1)) == True",
                ],
                "forbidden_patterns": [],
            },
            "requirement_change": {
                "description": (
                    "The base problem checks if a string IS a palindrome, but "
                    "the mutation requires checking if it is NOT a palindrome."
                ),
                "prompt_task": (
                    "Write a function `is_palindrome(s)` that returns True if "
                    "the string is NOT a palindrome, and False if it IS a "
                    "palindrome."
                ),
                "extra_tests": [
                    'assert is_palindrome("racecar") == False',
                    'assert is_palindrome("hello") == True',
                    'assert is_palindrome("") == False',
                ],
                "forbidden_patterns": [],
            },
            "resource_constraint": {
                "description": (
                    "No string slicing (s[::-1], s[1:], etc.) is allowed. The "
                    "solution must use a different approach."
                ),
                "constraint_text": (
                    "CONSTRAINT: You may NOT use string slicing (e.g. s[::-1], "
                    "s[1:], s[:n]). Use indexing and comparison instead."
                ),
                "extra_tests": [],
                "forbidden_patterns": [
                    r"\[::-1\]",
                    r"\[\s*:\s*[^]]*\]",
                    r"\[\s*-?\d+\s*:\s*\]",
                ],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# DynamicsMutator — applies mutations to base problems
# ---------------------------------------------------------------------------


class DynamicsMutator:
    """
    Applies a mutation to a base problem, producing a new problem with
    modified prompt, tests, and metadata.

    The mutator is the core of the environment mutation concept: it takes
    a well-defined base problem and introduces a "world dynamics change"
    that the model must detect and adapt to.

    Attributes:
        base: The base problem dict (id, task, func_name, base_code, base_tests, mutations).
    """

    def __init__(self, base_problem: dict[str, Any]) -> None:
        self._base = base_problem

    def apply_mutation(self, mutation_type: str) -> dict[str, Any]:
        """
        Apply a mutation to the base problem.

        Args:
            mutation_type: One of MUTATION_TYPES.

        Returns:
            A dict with keys: prompt, tests, func_name, mutation_type,
            mutation_description, forbidden_patterns, extra_tests.
        """
        if mutation_type not in MUTATION_TYPES:
            raise ValueError(
                f"Unknown mutation type: {mutation_type}. "
                f"Expected one of {MUTATION_TYPES}"
            )

        mutations = self._base.get("mutations", {})
        if mutation_type not in mutations:
            raise ValueError(
                f"Base problem '{self._base['id']}' has no mutation config "
                f"for '{mutation_type}'"
            )

        config = mutations[mutation_type]
        handler = {
            "hidden_constraint": self._apply_hidden_constraint,
            "misleading_info": self._apply_misleading_info,
            "partial_observability": self._apply_partial_observability,
            "requirement_change": self._apply_requirement_change,
            "resource_constraint": self._apply_resource_constraint,
        }[mutation_type]

        return handler(config)

    def _build_prompt(
        self,
        mutation_type: str,
        task_text: str,
        config: dict[str, Any],
        extra_sections: str = "",
    ) -> str:
        """Build the full problem prompt with mutation header and format instructions.

        Args:
            mutation_type: The mutation type (used for test collection).
            task_text: The task description for the prompt.
            config: The mutation configuration dict.
            extra_sections: Additional text to insert before the tests section.
        """
        sections: list[str] = [
            f"MUTATION: {mutation_type}",
            f"TASK: {task_text}",
        ]

        # Add misleading hint if present
        if "misleading_hint" in config:
            sections.append(f"HINT: {config['misleading_hint']}")

        # Add constraint text if present
        if "constraint_text" in config:
            sections.append(f"CONSTRAINT: {config['constraint_text']}")

        # Add any extra sections (e.g. redacted info notes)
        if extra_sections:
            sections.append(extra_sections)

        # Add tests
        all_tests = self._collect_tests(config, mutation_type)
        tests_text = "\n".join(f"  - {t}" for t in all_tests)
        sections.append(f"TESTS:\n{tests_text}")

        # Add format instructions
        sections.append(
            "Solve the task. Be aware that the environment may contain hidden "
            "constraints,\nmisleading information, or modified requirements. "
            "Verify your solution against\nthe tests.\n\n"
            "SOLUTION: ```python\n<your code>\n```\n"
            "DISCOVERY: <did you notice any hidden constraints or misleading info?>"
        )

        return "\n\n".join(sections)

    def _collect_tests(self, config: dict[str, Any], mutation_type: str) -> list[str]:
        """
        Collect all tests for the mutation.

        For hidden_constraint: base tests + extra tests (extra tests check
        the hidden constraint on top of the base requirements).
        For requirement_change and partial_observability: ONLY extra tests
        (they replace the base tests, since the requirements/specification
        have changed).
        For misleading_info and resource_constraint: base tests only (the
        task itself is unchanged; only the prompt or constraint differs).
        """
        extra = config.get("extra_tests", [])
        if mutation_type in ("requirement_change", "partial_observability"):
            # Extra tests fully replace base tests — the requirements changed
            return list(extra) if extra else list(self._base["base_tests"])
        # hidden_constraint, misleading_info, resource_constraint:
        # base tests + any extra tests
        tests = list(self._base["base_tests"])
        tests.extend(extra)
        return tests

    def _apply_hidden_constraint(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Add a constraint not explicitly stated in the prompt.
        The prompt is generic; the tests enforce the hidden constraint.
        """
        task = config.get("prompt_task", self._base["task"])
        prompt = self._build_prompt("hidden_constraint", task, config)

        return {
            "prompt": prompt,
            "tests": self._collect_tests(config, "hidden_constraint"),
            "func_name": self._base["func_name"],
            "mutation_type": "hidden_constraint",
            "mutation_description": config["description"],
            "forbidden_patterns": config.get("forbidden_patterns", []),
            "extra_tests": config.get("extra_tests", []),
        }

    def _apply_misleading_info(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Add incorrect information to the prompt. The model must recognize
        the misleading hint and use a correct approach instead.
        """
        task = self._base["task"]
        prompt = self._build_prompt("misleading_info", task, config)

        return {
            "prompt": prompt,
            "tests": self._collect_tests(config, "misleading_info"),
            "func_name": self._base["func_name"],
            "mutation_type": "misleading_info",
            "mutation_description": config["description"],
            "forbidden_patterns": config.get("forbidden_patterns", []),
            "extra_tests": config.get("extra_tests", []),
        }

    def _apply_partial_observability(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Redact part of the problem. The model must infer the full
        specification from the tests.
        """
        task = config.get("prompt_task", self._base["task"])
        prompt = self._build_prompt("partial_observability", task, config)

        return {
            "prompt": prompt,
            "tests": self._collect_tests(config, "partial_observability"),
            "func_name": self._base["func_name"],
            "mutation_type": "partial_observability",
            "mutation_description": config["description"],
            "forbidden_patterns": config.get("forbidden_patterns", []),
            "extra_tests": config.get("extra_tests", []),
        }

    def _apply_requirement_change(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Modify a requirement from the base problem. The model must notice
        the change and adapt.
        """
        task = config.get("prompt_task", self._base["task"])
        prompt = self._build_prompt("requirement_change", task, config)

        return {
            "prompt": prompt,
            "tests": self._collect_tests(config, "requirement_change"),
            "func_name": self._base["func_name"],
            "mutation_type": "requirement_change",
            "mutation_description": config["description"],
            "forbidden_patterns": config.get("forbidden_patterns", []),
            "extra_tests": config.get("extra_tests", []),
        }

    def _apply_resource_constraint(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Add a tight resource constraint. The model must solve within it.
        """
        task = self._base["task"]
        prompt = self._build_prompt("resource_constraint", task, config)

        return {
            "prompt": prompt,
            "tests": self._collect_tests(config, "resource_constraint"),
            "func_name": self._base["func_name"],
            "mutation_type": "resource_constraint",
            "mutation_description": config["description"],
            "forbidden_patterns": config.get("forbidden_patterns", []),
            "extra_tests": config.get("extra_tests", []),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def environment_mutation_generator(seed: int) -> Problem:
    """
    Generate a problem with a random mutation applied to a random base problem.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A Problem with mutation metadata for the verifier.
    """
    rng = random.Random(seed)
    base = rng.choice(_BASE_PROBLEMS)
    mutation_type = rng.choice(MUTATION_TYPES)

    # If the base problem doesn't have this mutation configured, fall back
    # to a mutation it does have.
    available = list(base.get("mutations", {}).keys())
    if mutation_type not in available:
        mutation_type = rng.choice(available)

    mutator = DynamicsMutator(base)
    mutated = mutator.apply_mutation(mutation_type)

    # Difficulty scales with mutation type complexity
    difficulty_map = {
        "hidden_constraint": 0.55,
        "misleading_info": 0.50,
        "partial_observability": 0.65,
        "requirement_change": 0.45,
        "resource_constraint": 0.60,
    }
    difficulty = difficulty_map.get(mutation_type, 0.5) + rng.uniform(-0.05, 0.05)
    difficulty = max(0.1, min(0.9, difficulty))

    problem_id = f"env_mutation_{base['id']}_{mutation_type}_{rng.randint(0, 99999)}"

    return Problem(
        id=problem_id,
        prompt=mutated["prompt"],
        difficulty=difficulty,
        metadata={
            "type": "environment_mutation",
            "base_problem_id": base["id"],
            "func_name": mutated["func_name"],
            "tests": mutated["tests"],
            "mutation_type": mutated["mutation_type"],
            "mutation_description": mutated["mutation_description"],
            "forbidden_patterns": mutated["forbidden_patterns"],
            "extra_tests": mutated["extra_tests"],
        },
        token_budget=2048,
        source="generated",
    )


# Prevent pytest from collecting this generator function as a test
environment_mutation_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier — runs real tests AND checks mutation handling
# ---------------------------------------------------------------------------


class MutationAwareVerifier(Verifier):
    """
    Verifies a mutation-environment response by:
      1. Extracting the SOLUTION code block
      2. Running the tests via real execution
      3. Checking whether the mutation was handled (type-specific)
      4. Checking the DISCOVERY field for mutation awareness

    Score = tests_passed_ratio * 0.6 + mutation_handled * 0.4
    correct = all tests pass AND mutation_handled >= 0.5
    """

    def __init__(
        self,
        tests: list[str],
        func_name: str,
        mutation_type: str,
        mutation_description: str,
        forbidden_patterns: list[str],
        extra_tests: list[str],
        timeout: float = 5.0,
    ) -> None:
        super().__init__()
        self._tests = tests
        self._func_name = func_name
        self._mutation_type = mutation_type
        self._mutation_description = mutation_description
        self._forbidden_patterns = forbidden_patterns
        self._extra_tests = extra_tests
        self._timeout = timeout

    def verify(self, response: str) -> VerifierResult:
        # Extract solution code
        code = self._extract_solution(response)
        if not code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No SOLUTION code block found in response",
                partial_credit={"tests_passed": 0, "mutation_handled": 0.0},
            )

        # Extract discovery text
        discovery = self._extract_discovery(response)

        # Run all tests via real execution
        tests_passed, tests_total, test_details = self._run_tests(code)
        tests_passed_ratio = tests_passed / tests_total if tests_total > 0 else 0.0

        # Check mutation handling (type-specific)
        mutation_handled = self._check_mutation_handled(code, tests_passed_ratio)

        # Check discovery awareness
        discovery_awareness = self._check_discovery(discovery)

        # Incorporate discovery awareness into mutation_handled
        # mutation_handled = 0.7 * solution_quality + 0.3 * discovery
        mutation_score = mutation_handled * 0.7 + discovery_awareness * 0.3

        # Final score
        score = tests_passed_ratio * 0.6 + mutation_score * 0.4

        # Correct only if all tests pass AND mutation is handled
        correct = (tests_passed == tests_total) and (mutation_handled >= 0.5)

        diagnostics_parts = [
            f"Tests: {tests_passed}/{tests_total}",
            f"Mutation ({self._mutation_type}): handled={mutation_handled:.2f}",
            f"Discovery: {discovery_awareness:.2f}",
        ]
        if test_details:
            # Include first few failure details
            failures = [d for d in test_details if not d["passed"]]
            if failures:
                diagnostics_parts.append(
                    f"Failures: {'; '.join(d['error'][:80] for d in failures[:3])}"
                )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "tests_passed_ratio": tests_passed_ratio,
                "mutation_handled": mutation_handled,
                "discovery_awareness": discovery_awareness,
                "mutation_score": mutation_score,
            },
            diagnostics=" | ".join(diagnostics_parts),
            metadata={
                "mutation_type": self._mutation_type,
                "forbidden_patterns_found": self._check_forbidden_patterns(code),
            },
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_solution(self, response: str) -> str:
        """Extract the Python code from the SOLUTION block."""
        # Try: SOLUTION: ```python\n...\n```
        match = re.search(
            r"SOLUTION\s*:\s*```python\s*\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try: SOLUTION: ```\n...\n```
        match = re.search(
            r"SOLUTION\s*:\s*```\s*\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try: just a ```python block after SOLUTION:
        match = re.search(
            r"SOLUTION\s*:\s*\n(.*?)```python\s*\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(2).strip()

        # Fallback: any ```python block in the response
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: any ``` block
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_discovery(self, response: str) -> str:
        """Extract the DISCOVERY text from the response."""
        match = re.search(
            r"DISCOVERY\s*:\s*(.*?)(?:\n```|\nSOLUTION|\Z)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def _run_tests(self, code: str) -> tuple[int, int, list[dict[str, Any]]]:
        """
        Execute the solution code and run all tests.

        Returns:
            (tests_passed, tests_total, details)
            details is a list of {"test": str, "passed": bool, "error": str}
        """
        # Strip markdown fences if present in the code
        code = self._strip_fences(code)

        # First, try to exec the solution code
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            return 0, len(self._tests), [
                {"test": t, "passed": False, "error": error_msg}
                for t in self._tests
            ]

        # Run each test
        passed = 0
        details: list[dict[str, Any]] = []
        for test_code in self._tests:
            try:
                # Use a copy of the namespace so tests don't pollute each other
                test_ns = dict(namespace)
                exec(test_code, test_ns)
                passed += 1
                details.append({"test": test_code, "passed": True, "error": ""})
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                details.append({"test": test_code, "passed": False, "error": error_msg})

        return passed, len(self._tests), details

    def _strip_fences(self, code: str) -> str:
        """Remove markdown code fences if present."""
        code = code.strip()
        if code.startswith("```"):
            # Remove opening fence
            lines = code.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code

    # ------------------------------------------------------------------
    # Mutation-specific checks
    # ------------------------------------------------------------------

    def _check_mutation_handled(self, code: str, tests_passed_ratio: float) -> float:
        """
        Check whether the solution handles the mutation.

        Each mutation type has a different check:
          - hidden_constraint: extra tests (the hidden constraint tests) pass
          - misleading_info: forbidden patterns absent AND tests pass
          - partial_observability: all tests pass (tests reveal hidden info)
          - requirement_change: all tests pass (tests check new requirement)
          - resource_constraint: forbidden patterns absent AND tests pass

        Returns:
            A score in [0, 1] for how well the mutation was handled.
        """
        if self._mutation_type == "hidden_constraint":
            return self._check_hidden_constraint(code)
        elif self._mutation_type == "misleading_info":
            return self._check_misleading_info(code, tests_passed_ratio)
        elif self._mutation_type == "partial_observability":
            return self._check_partial_observability(code, tests_passed_ratio)
        elif self._mutation_type == "requirement_change":
            return self._check_requirement_change(code, tests_passed_ratio)
        elif self._mutation_type == "resource_constraint":
            return self._check_resource_constraint(code, tests_passed_ratio)
        else:
            return 0.0

    def _check_hidden_constraint(self, code: str) -> float:
        """Check that the hidden constraint tests pass."""
        if not self._extra_tests:
            return 1.0
        return self._run_specific_tests(code, self._extra_tests)

    def _check_misleading_info(
        self, code: str, tests_passed_ratio: float
    ) -> float:
        """
        Check that the model ignored the misleading info.
        The solution should NOT contain the forbidden patterns (which
        correspond to following the misleading advice) AND tests should pass.
        """
        forbidden_found = self._check_forbidden_patterns(code)
        if forbidden_found:
            # Model followed the misleading advice
            return 0.0
        # Model avoided the misleading approach; reward proportional to test success
        return tests_passed_ratio

    def _check_partial_observability(
        self, code: str, tests_passed_ratio: float
    ) -> float:
        """
        Check that the model inferred the hidden information.
        If all tests pass (including the ones that use the hidden fields),
        the model successfully inferred the full specification.
        """
        return tests_passed_ratio

    def _check_requirement_change(
        self, code: str, tests_passed_ratio: float
    ) -> float:
        """
        Check that the model noticed and adapted to the requirement change.
        If the tests (which check the NEW requirement) pass, the model adapted.
        """
        return tests_passed_ratio

    def _check_resource_constraint(
        self, code: str, tests_passed_ratio: float
    ) -> float:
        """
        Check that the solution respects the resource constraint.
        Forbidden patterns (e.g. loops, builtins) should be absent AND
        tests should pass.
        """
        forbidden_found = self._check_forbidden_patterns(code)
        if forbidden_found:
            return 0.0
        return tests_passed_ratio

    # ------------------------------------------------------------------
    # Utility checks
    # ------------------------------------------------------------------

    def _check_forbidden_patterns(self, code: str) -> list[str]:
        """
        Check if the code contains any forbidden patterns.

        Returns:
            List of matched pattern strings (empty if none found).
        """
        found: list[str] = []
        for pattern in self._forbidden_patterns:
            try:
                if re.search(pattern, code, re.IGNORECASE):
                    found.append(pattern)
            except re.error:
                # Skip invalid patterns
                continue
        return found

    def _run_specific_tests(self, code: str, tests: list[str]) -> float:
        """Run a subset of tests and return the pass ratio."""
        code = self._strip_fences(code)
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception:
            return 0.0

        passed = 0
        for test_code in tests:
            try:
                test_ns = dict(namespace)
                exec(test_code, test_ns)
                passed += 1
            except Exception:
                pass

        return passed / len(tests) if tests else 1.0

    def _check_discovery(self, discovery: str) -> float:
        """
        Check the DISCOVERY field for awareness of the mutation.

        Looks for keywords related to the mutation type and the specific
        mutation description.

        Returns:
            1.0 if the discovery clearly identifies the mutation,
            0.5 if it mentions something relevant,
            0.0 if empty or irrelevant.
        """
        if not discovery or len(discovery.strip()) < 10:
            return 0.0

        discovery_lower = discovery.lower()

        # Keywords associated with each mutation type
        type_keywords: dict[str, list[str]] = {
            "hidden_constraint": [
                "hidden", "unstated", "not mentioned", "not specified",
                "implicit", "undocumented", "constraint",
            ],
            "misleading_info": [
                "misleading", "incorrect", "wrong hint", "bad hint",
                "not efficient", "not optimal", "false", "inaccurate",
                "hint is wrong", "bad advice",
            ],
            "partial_observability": [
                "partial", "redacted", "hidden", "incomplete", "missing",
                "not fully specified", "inferred", "infer",
            ],
            "requirement_change": [
                "change", "modified", "different", "not the same",
                "reversed", "descending", "minimum instead",
                "not a palindrome", "requirement",
            ],
            "resource_constraint": [
                "constraint", "no loops", "recursion", "no slicing",
                "no built-in", "no max", "no min", "restriction",
                "limit", "forbidden",
            ],
        }

        keywords = type_keywords.get(self._mutation_type, [])
        matches = sum(1 for kw in keywords if kw in discovery_lower)

        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.5
        else:
            # Check if the discovery mentions the mutation description keywords
            desc_words = [
                w.lower()
                for w in re.findall(r"\b\w{4,}\b", self._mutation_description)
            ]
            desc_matches = sum(1 for w in desc_words if w in discovery_lower)
            if desc_matches >= 2:
                return 0.5
            return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EnvironmentMutationEnv(BatchEnvBase):
    """
    EnvironmentMutation: train robustness to environment mutations.

    The model receives a coding task with a mutation applied (hidden
    constraint, misleading info, partial observability, requirement change,
    or resource constraint). It must produce a correct solution AND
    discover what mutation was applied.

    Batch-aware: N parallel solutions are scored, and the reward combines
    the best score with diversity across the batch.

    reward = best_score * 0.7 + diversity * 0.3
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
    ) -> None:
        if problems is None and problem_generator is None:
            problem_generator = environment_mutation_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        """Create a MutationAwareVerifier for the given problem."""
        meta = problem.metadata
        return MutationAwareVerifier(
            tests=meta["tests"],
            func_name=meta["func_name"],
            mutation_type=meta["mutation_type"],
            mutation_description=meta["mutation_description"],
            forbidden_patterns=meta["forbidden_patterns"],
            extra_tests=meta["extra_tests"],
        )

    def _check_format(self, response: str) -> float:
        """
        Check if the response follows the expected format.
        Full marks require both SOLUTION and DISCOVERY blocks.
        """
        has_solution = bool(re.search(r"SOLUTION\s*:", response, re.IGNORECASE))
        has_discovery = bool(re.search(r"DISCOVERY\s*:", response, re.IGNORECASE))
        has_code_block = bool(re.search(r"```python", response, re.IGNORECASE))

        score = 0.0
        if has_solution:
            score += 0.4
        if has_code_block:
            score += 0.3
        if has_discovery:
            score += 0.3
        return min(1.0, score)

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """
        Aggregate per-sample scores into a batch reward.

        reward = best_score * 0.7 + diversity * 0.3

        Diversity is measured as the fraction of distinct scores (rounded
        to 2 decimal places) across the batch. A diverse batch that explores
        different approaches is rewarded, even if only one succeeds.
        """
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]

        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Diversity: fraction of unique scores (rounded to avoid near-duplicates)
        if scores:
            rounded = [round(s, 2) for s in scores]
            unique = len(set(rounded))
            diversity = unique / len(rounded)
        else:
            diversity = 0.0

        # Final reward: best_score * 0.7 + diversity * 0.3
        reward = best_score * 0.7 + diversity * 0.3

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
                f"Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores) / len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        """Extract the solution code from the response."""
        match = re.search(
            r"SOLUTION\s*:\s*```python\s*\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response
