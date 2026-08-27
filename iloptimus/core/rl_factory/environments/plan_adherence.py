"""
PlanAdherence: Plan then code, penalize deviations.

Environment concept:
  The model must produce a plan (numbered steps) and then code that
  implements EXACTLY that plan. The verifier checks if the code follows
  the plan step-by-step. Deviations are penalized.

  This trains the model to:
    1. Plan accurately (not over-plan, not under-plan)
    2. Follow its own plan (not drift mid-execution)
    3. Write concise plans (verbose plans waste tokens)
    4. Not add unplanned steps ("while I'm here, let me also...")

  This directly attacks "plan drift" — identified by cLens as a key
  frontier-model anti-pattern where the agent's actual execution diverges
  from its stated plan, leading to wasted work and incorrect results.

Why this environment is worth using for 10T-100T param models:
  Plan adherence is the foundation of reliable agentic behavior. A 100T
  model that plans well and follows its plan can solve complex multi-step
  tasks in one shot. A model that drifts from its plan wastes tokens on
  dead ends and produces incorrect results. This environment trains the
  plan→execute loop as a single coherent skill.

Problem types:
  - Algorithm implementation (plan the steps, then code them)
  - Data processing pipeline (plan the transformations, then code)
  - Bug fix workflow (plan the diagnosis, then execute)
  - Refactoring task (plan the changes, then apply them)

Verification:
  Each problem has:
    - A task description
    - Test cases for the final code
    - A "plan checker" that maps plan steps to code sections
  The verifier:
    1. Parses the plan (numbered steps)
    2. Parses the code
    3. Checks if each plan step has corresponding code
    4. Checks if the code has sections NOT in the plan (extra work)
    5. Runs the test cases

Reward design:
  - Tests pass: gate (if tests fail, reward = 0)
  - If tests pass:
    - Plan coverage: fraction of plan steps that have corresponding code
    - Code excess: fraction of code sections NOT in the plan (penalized)
    - Plan conciseness: shorter plans get a bonus
  - Final reward = test_score * plan_coverage * (1 - code_excess_penalty)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


def _make_algorithm_problem(rng: random.Random) -> Problem:
    """Generate an algorithm implementation problem."""
    templates = [
        {
            "task": "Write a function `is_palindrome(s)` that checks if a string is a palindrome (reads the same forwards and backwards). Ignore case and spaces.",
            "expected_plan_steps": [
                "normalize the string (lowercase, remove spaces)",
                "compare the string to its reverse",
                "return the comparison result",
            ],
            "tests": [
                ("racecar", True),
                ("hello", False),
                ("A man a plan a canal Panama", True),
                ("", True),
                ("a", True),
            ],
            "difficulty": 0.2,
        },
        {
            "task": "Write a function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed). F(0)=0, F(1)=1.",
            "expected_plan_steps": [
                "handle base cases (n=0 returns 0, n=1 returns 1)",
                "use iteration or recursion to compute F(n)",
                "return the result",
            ],
            "tests": [
                (0, 0), (1, 1), (2, 1), (5, 5), (10, 55),
            ],
            "difficulty": 0.25,
        },
        {
            "task": "Write a function `count_vowels(s)` that counts the number of vowels (a, e, i, o, u) in a string. Case insensitive.",
            "expected_plan_steps": [
                "define the set of vowels",
                "iterate through the string and count matching characters",
                "return the count",
            ],
            "tests": [
                ("hello", 2), ("", 0), ("AEIOU", 5), ("xyz", 0), ("aAeEiIoOuU", 10),
            ],
            "difficulty": 0.15,
        },
        {
            "task": "Write a function `reverse_words(s)` that reverses the order of words in a string. Words are separated by spaces.",
            "expected_plan_steps": [
                "split the string into words",
                "reverse the list of words",
                "join the reversed list back into a string",
                "return the result",
            ],
            "tests": [
                ("hello world", "world hello"),
                ("one", "one"),
                ("", ""),
                ("a b c", "c b a"),
            ],
            "difficulty": 0.2,
        },
        {
            "task": "Write a function `find_max(numbers)` that finds the maximum value in a list. Return None for an empty list.",
            "expected_plan_steps": [
                "handle the empty list case",
                "iterate through the list tracking the maximum",
                "return the maximum",
            ],
            "tests": [
                ([1, 2, 3], 3),
                ([], None),
                ([-1, -2, -3], -1),
                ([5], 5),
            ],
            "difficulty": 0.15,
        },
        {
            "task": "Write a function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
            "expected_plan_steps": [
                "initialize an empty result list and two pointers",
                "compare elements at both pointers, append the smaller one",
                "advance the pointer of the appended element",
                "append remaining elements from either list",
                "return the merged list",
            ],
            "tests": [
                ([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]),
                ([], [], []),
                ([1], [], [1]),
                ([], [2], [2]),
                ([1, 2], [3, 4], [1, 2, 3, 4]),
            ],
            "difficulty": 0.35,
        },
    ]

    template = rng.choice(templates)
    return Problem(
        id=f"plan_adhere_algo_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["task"], len(template["expected_plan_steps"])),
        difficulty=template["difficulty"],
        metadata={
            "type": "plan_adherence",
            "task": template["task"],
            "expected_plan_steps": template["expected_plan_steps"],
            "tests": template["tests"],
        },
        token_budget=1000,
        source="generated",
    )


def _make_data_pipeline_problem(rng: random.Random) -> Problem:
    """Generate a data processing pipeline problem."""
    templates = [
        {
            "task": "Write a function `process_data(records)` that takes a list of dicts with 'name' and 'score' keys, filters out records with score < 50, sorts by score descending, and returns a list of names.",
            "expected_plan_steps": [
                "filter records to keep only score >= 50",
                "sort the filtered records by score in descending order",
                "extract the 'name' from each record",
                "return the list of names",
            ],
            "tests": [
                ([{"name": "Alice", "score": 80}, {"name": "Bob", "score": 30}], ["Alice"]),
                ([{"name": "A", "score": 90}, {"name": "B", "score": 60}, {"name": "C", "score": 50}], ["A", "B", "C"]),
                ([], []),
                ([{"name": "X", "score": 49}], []),
            ],
            "difficulty": 0.3,
        },
        {
            "task": "Write a function `deduplicate(items)` that removes duplicates from a list while preserving order.",
            "expected_plan_steps": [
                "initialize an empty set to track seen items and an empty result list",
                "iterate through the input list",
                "for each item, if not seen, add to result and mark as seen",
                "return the result list",
            ],
            "tests": [
                ([1, 2, 2, 3, 1], [1, 2, 3]),
                ([], []),
                (["a", "b", "a"], ["a", "b"]),
                ([1], [1]),
            ],
            "difficulty": 0.25,
        },
    ]

    template = rng.choice(templates)
    return Problem(
        id=f"plan_adhere_pipe_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["task"], len(template["expected_plan_steps"])),
        difficulty=template["difficulty"],
        metadata={
            "type": "plan_adherence",
            "task": template["task"],
            "expected_plan_steps": template["expected_plan_steps"],
            "tests": template["tests"],
        },
        token_budget=1000,
        source="generated",
    )


def _format_prompt(task: str, suggested_steps: int) -> str:
    """Format the problem prompt."""
    return (
        f"Task: {task}\n\n"
        f"First, write a plan as numbered steps (aim for {suggested_steps}-5 steps).\n"
        f"Then, write the Python code that implements EXACTLY your plan.\n\n"
        f"Format:\n"
        f"PLAN:\n"
        f"1. <step>\n"
        f"2. <step>\n"
        f"...\n\n"
        f"CODE:\n"
        f"```python\n<your code>\n```\n\n"
        f"Rules:\n"
        f"  - The code must follow the plan step-by-step\n"
        f"  - Do NOT add code that isn't in the plan\n"
        f"  - Do NOT add steps to the plan that you don't implement\n"
        f"  - Keep the plan concise — each step should be one sentence"
    )


def plan_adherence_generator(seed: int) -> Problem:
    """Master generator for PlanAdherence problems."""
    rng = random.Random(seed)
    generators = [_make_algorithm_problem, _make_data_pipeline_problem]
    weights = [0.7, 0.3]
    gen = rng.choices(generators, weights=weights, k=1)[0]
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PlanAdherenceVerifier(Verifier):
    """
    Verifies a PlanAdherence response by:
      1. Parsing the plan (numbered steps)
      2. Parsing the code
      3. Running the code against test cases
      4. Checking if the code follows the plan
    """

    def __init__(self, expected_plan_steps: list[str], tests: list):
        super().__init__()
        self._expected_steps = expected_plan_steps
        self._tests = tests

    def verify(self, response: str) -> VerifierResult:
        # Parse the plan
        plan_steps = self._parse_plan(response)
        if not plan_steps:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No PLAN: section found",
            )

        # Parse the code
        code = self._parse_code(response)
        if not code:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No CODE: section with python block found",
            )

        # Execute the code
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Code execution error: {type(e).__name__}: {e}",
            )

        # Find the function
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_") and name != "__builtins__":
                if hasattr(obj, "__module__") and obj.__module__ in ("builtins",):
                    continue
                func = obj
                break

        if func is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No function found in code",
            )

        # Run tests
        passed = 0
        total = len(self._tests)
        failures = []
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
                else:
                    failures.append(f"Input {test[:-1]}: expected {expected}, got {result}")
            except Exception as e:
                failures.append(f"Input {test[:-1]}: raised {type(e).__name__}: {e}")

        test_score = passed / total if total > 0 else 0.0

        if passed < total:
            return VerifierResult(
                correct=False,
                score=test_score * 0.3,  # partial credit for tests
                partial_credit={"test_pass_rate": test_score},
                diagnostics=f"Tests: {passed}/{total} passed. Failures: {'; '.join(failures[:2])}",
            )

        # All tests pass — check plan adherence
        plan_coverage = self._check_plan_coverage(plan_steps, code)
        code_excess = self._check_code_excess(plan_steps, code)

        # Plan conciseness: fewer steps = bonus
        plan_conciseness = min(1.0, len(self._expected_steps) / max(len(plan_steps), 1))

        # Final score
        adherence_score = plan_coverage * (1.0 - code_excess * 0.3)
        final_score = min(1.0, adherence_score * 0.7 + plan_conciseness * 0.3)

        correct = final_score >= 0.6

        return VerifierResult(
            correct=correct,
            score=final_score,
            partial_credit={
                "test_pass_rate": 1.0,
                "plan_coverage": plan_coverage,
                "code_excess": code_excess,
                "plan_conciseness": plan_conciseness,
                "plan_steps": len(plan_steps),
            },
            diagnostics=f"All {total} tests passed. Plan coverage: {plan_coverage:.2f}, Code excess: {code_excess:.2f}, Steps: {len(plan_steps)}",
        )

    def _parse_plan(self, response: str) -> list[str]:
        """Parse numbered plan steps from the response."""
        plan_match = re.search(r"PLAN:\s*\n(.*?)(?:\n\s*CODE:|$)", response, re.DOTALL | re.IGNORECASE)
        if not plan_match:
            return []

        plan_text = plan_match.group(1)
        steps = re.findall(r"\d+\.\s*(.+?)(?:\n|$)", plan_text)
        return [s.strip() for s in steps if s.strip()]

    def _parse_code(self, response: str) -> str:
        """Parse the code from the CODE: section."""
        code_match = re.search(r"CODE:\s*\n\s*```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"CODE:\s*\n\s*```\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return ""

    def _check_plan_coverage(self, plan_steps: list[str], code: str) -> float:
        """
        Check what fraction of plan steps have corresponding code.

        Uses keyword matching: for each plan step, check if key verbs/nouns
        from the step appear in the code.
        """
        if not plan_steps:
            return 0.0

        covered = 0
        code_lower = code.lower()

        for step in plan_steps:
            # Extract keywords from the step
            words = re.findall(r"\b[a-z]{3,}\b", step.lower())
            # Filter common words
            common = {"the", "and", "for", "with", "that", "this", "from", "into", "then", "step"}
            keywords = [w for w in words if w not in common]

            if not keywords:
                covered += 1
                continue

            # Check if at least 30% of keywords appear in the code
            matches = sum(1 for kw in keywords if kw in code_lower)
            if matches / len(keywords) >= 0.3:
                covered += 1

        return covered / len(plan_steps)

    def _check_code_excess(self, plan_steps: list[str], code: str) -> float:
        """
        Estimate code excess: fraction of code lines not accounted for by the plan.

        This is a heuristic: we compare the number of code lines to the number
        of plan steps. If there are significantly more code lines than expected,
        we infer excess.
        """
        code_lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
        expected_lines = len(plan_steps) * 3  # ~3 lines per plan step

        if len(code_lines) <= expected_lines:
            return 0.0

        excess_lines = len(code_lines) - expected_lines
        return min(1.0, excess_lines / max(len(code_lines), 1))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PlanAdherenceEnv(BaseReasoningEnv):
    """
    PlanAdherence environment: plan then code, penalize deviations.

    The model produces a plan + code. Reward = test_score * plan_coverage *
    (1 - code_excess_penalty) + plan_conciseness_bonus.
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
            problem_generator = plan_adherence_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return PlanAdherenceVerifier(
            expected_plan_steps=problem.metadata["expected_plan_steps"],
            tests=problem.metadata["tests"],
        )

    def _check_format(self, response: str) -> float:
        """Check for PLAN: and CODE: sections."""
        has_plan = bool(re.search(r"PLAN:\s*\n\s*\d+\.", response, re.IGNORECASE))
        has_code = bool(re.search(r"CODE:\s*\n\s*```", response, re.IGNORECASE))
        if has_plan and has_code:
            return 1.0
        elif has_plan or has_code:
            return 0.3
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
