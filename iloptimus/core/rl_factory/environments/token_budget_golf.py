"""
TokenBudgetGolf: Solve a task within a HARD token budget.

Environment concept:
  The model is given a coding or reasoning task and a STRICT token budget
  that is set just below what a typical verbose solution would require.
  The model must produce a correct solution within the budget — no exceptions.

  This is the most direct environment for forcing extreme conciseness.
  Unlike other environments where conciseness is rewarded, here it is
  REQUIRED. If the solution exceeds the budget, reward = 0 regardless of
  correctness.

  The budget is calibrated so that:
    - A verbose solution (with anti-patterns, filler, hedging) will FAIL
    - A standard correct solution will BARELY pass
    - A concise correct solution will pass comfortably
    - An over-compressed (obfuscated) solution will pass but may fail tests

  This trains the model to find the "sweet spot" of conciseness — not
  verbose, not obfuscated, just efficient.

Why this environment is worth using for 10T-100T param models:
  At 100T parameters, every token costs money. A model that can solve
  tasks within a tight budget is directly more profitable. This environment
  trains the model to think of tokens as a scarce resource — the most
  fundamental skill for cost-efficient AI.

Problem types:
  - Code generation with token budget
  - Math proofs with token budget
  - Explanations with token budget
  - Bug fixes with token budget

Verification:
  Each problem has:
    - A task description
    - A token budget (set to ~80% of a typical correct solution)
    - Test cases or an answer verifier
  The verifier:
    1. Counts the tokens in the response
    2. If over budget → reward = 0 (regardless of correctness)
    3. If under budget → check correctness
    4. If correct and under budget → reward = 1.0 * (remaining_budget / total_budget)
      (more remaining = higher reward, incentivizing conciseness beyond minimum)

Reward design:
  - Over budget: reward = 0.0 (hard gate)
  - Under budget + incorrect: reward = 0.0
  - Under budget + correct: reward = 0.5 + 0.5 * (remaining_fraction)
    where remaining_fraction = (budget - used) / budget
  - Anti-pattern penalties still apply (double penalty for waste under budget)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult, ExactMatchVerifier, NumericVerifier


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_MATH_PROBLEMS = [
    {"question": "What is 15 * 12?", "answer": "180", "budget": 30},
    {"question": "What is 7^3?", "answer": "343", "budget": 30},
    {"question": "What is 144 / 12?", "answer": "12", "budget": 25},
    {"question": "What is 25 + 37?", "answer": "62", "budget": 20},
    {"question": "What is 100 - 47?", "answer": "53", "budget": 20},
    {"question": "What is 9 * 13?", "answer": "117", "budget": 30},
    {"question": "What is 2^10?", "answer": "1024", "budget": 30},
    {"question": "What is 50 * 50?", "answer": "2500", "budget": 30},
    {"question": "What is 17 * 23?", "answer": "391", "budget": 35},
    {"question": "What is 81 / 9?", "answer": "9", "budget": 20},
    {"question": "What is 11^2?", "answer": "121", "budget": 25},
    {"question": "What is 3^5?", "answer": "243", "budget": 30},
    {"question": "What is 1000 / 8?", "answer": "125", "budget": 30},
    {"question": "What is 45 + 55?", "answer": "100", "budget": 20},
    {"question": "What is 13 * 13?", "answer": "169", "budget": 30},
]

_CODE_PROBLEMS = [
    {
        "task": "Write a function `is_even(n)` that returns True if n is even, False otherwise.",
        "tests": [(2, True), (3, False), (0, True), (-1, False), (100, True)],
        "budget": 60,
    },
    {
        "task": "Write a function `square(n)` that returns n squared.",
        "tests": [(2, 4), (0, 0), (-3, 9), (5, 25), (10, 100)],
        "budget": 50,
    },
    {
        "task": "Write a function `max_of_two(a, b)` that returns the larger of two numbers.",
        "tests": [(1, 2, 2), (5, 3, 5), (0, 0, 0), (-1, -2, -1)],
        "budget": 60,
    },
    {
        "task": "Write a function `is_positive(n)` that returns True if n > 0.",
        "tests": [(1, True), (0, False), (-1, False), (100, True)],
        "budget": 50,
    },
    {
        "task": "Write a function `double(n)` that returns n * 2.",
        "tests": [(1, 2), (0, 0), (-5, -10), (50, 100)],
        "budget": 45,
    },
    {
        "task": "Write a function `count_items(lst)` that returns the length of a list.",
        "tests": [([1, 2, 3], 3), ([], 0), (["a"], 1), ([1, 2, 3, 4, 5], 5)],
        "budget": 55,
    },
    {
        "task": "Write a function `first_item(lst)` that returns the first item or None if empty.",
        "tests": [([1, 2], 1), ([], None), (["a", "b"], "a")],
        "budget": 60,
    },
    {
        "task": "Write a function `last_item(lst)` that returns the last item or None if empty.",
        "tests": [([1, 2, 3], 3), ([], None), (["a"], "a")],
        "budget": 60,
    },
    {
        "task": "Write a function `sum_list(lst)` that returns the sum of all items.",
        "tests": [([1, 2, 3], 6), ([], 0), ([5], 5), ([-1, -2, 3], 0)],
        "budget": 55,
    },
    {
        "task": "Write a function `is_empty(lst)` that returns True if the list is empty.",
        "tests": [([], True), ([1], False), ([1, 2], False)],
        "budget": 45,
    },
]


def _make_math_budget_problem(rng: random.Random) -> Problem:
    template = rng.choice(_MATH_PROBLEMS)
    # Tighten the budget by a random factor
    budget = int(template["budget"] * rng.uniform(0.7, 1.0))

    return Problem(
        id=f"token_budget_math_{rng.randint(0, 99999)}",
        prompt=_format_math_prompt(template["question"], budget),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "token_budget_golf",
            "subtype": "math",
            "answer": template["answer"],
            "budget": budget,
        },
        token_budget=budget,
        source="generated",
    )


def _make_code_budget_problem(rng: random.Random) -> Problem:
    template = rng.choice(_CODE_PROBLEMS)
    budget = int(template["budget"] * rng.uniform(0.7, 1.0))

    return Problem(
        id=f"token_budget_code_{rng.randint(0, 99999)}",
        prompt=_format_code_prompt(template["task"], budget),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "token_budget_golf",
            "subtype": "code",
            "task": template["task"],
            "tests": template["tests"],
            "budget": budget,
        },
        token_budget=budget,
        source="generated",
    )


def _format_math_prompt(question: str, budget: int) -> str:
    return (
        f"Question: {question}\n\n"
        f"TOKEN BUDGET: {budget} tokens. Responses exceeding this budget get ZERO reward.\n"
        f"End with: ANSWER: <value>\n"
        f"Be as concise as possible. No explanations needed."
    )


def _format_code_prompt(task: str, budget: int) -> str:
    return (
        f"Task: {task}\n\n"
        f"TOKEN BUDGET: {budget} tokens. Responses exceeding this budget get ZERO reward.\n"
        f"Output ONLY the code in a ```python block. No explanations.\n"
        f"Be as concise as possible."
    )


def token_budget_golf_generator(seed: int) -> Problem:
    """Master generator for TokenBudgetGolf problems."""
    rng = random.Random(seed)
    if rng.random() < 0.4:
        return _make_math_budget_problem(rng)
    else:
        return _make_code_budget_problem(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TokenBudgetGolfVerifier(Verifier):
    """
    Verifies a TokenBudgetGolf response:
      1. Check if response is within the token budget (hard gate)
      2. Check correctness (math or code)
      3. Reward = 0.5 + 0.5 * remaining_fraction if correct and under budget
    """

    def __init__(self, subtype: str, budget: int, answer: str = "",
                 tests: Optional[list] = None):
        super().__init__()
        self._subtype = subtype
        self._budget = budget
        self._answer = answer
        self._tests = tests or []

    def verify(self, response: str) -> VerifierResult:
        # Count tokens
        used = estimate_tokens(response)

        if used > self._budget:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Over budget: {used} > {self._budget} tokens",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        # Under budget — check correctness
        if self._subtype == "math":
            return self._verify_math(response, used)
        else:
            return self._verify_code(response, used)

    def _verify_math(self, response: str, used: int) -> VerifierResult:
        # Extract answer
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if not match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No ANSWER: found",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        submitted = match.group(1).strip()
        if submitted == self._answer:
            remaining_frac = (self._budget - used) / self._budget
            score = 0.5 + 0.5 * remaining_frac
            return VerifierResult(
                correct=True,
                score=score,
                partial_credit={"tokens_used": used, "budget": self._budget, "remaining_fraction": remaining_frac},
                diagnostics=f"Correct. {used}/{self._budget} tokens. Score: {score:.2f}",
                metadata={"tokens_used": used, "budget": self._budget},
            )
        else:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Wrong answer: {submitted} != {self._answer}",
                metadata={"tokens_used": used, "budget": self._budget},
            )

    def _verify_code(self, response: str, used: int) -> VerifierResult:
        # Extract code
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if not code_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No code block found",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        code = code_match.group(1).strip()

        # Execute
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Execution error: {type(e).__name__}: {e}",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        # Find function
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
                diagnostics="No function found",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        # Run tests
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

        if passed < total:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Tests: {passed}/{total} passed",
                metadata={"tokens_used": used, "budget": self._budget},
            )

        # All tests pass and under budget
        remaining_frac = (self._budget - used) / self._budget
        score = 0.5 + 0.5 * remaining_frac
        return VerifierResult(
            correct=True,
            score=score,
            partial_credit={"tokens_used": used, "budget": self._budget, "remaining_fraction": remaining_frac},
            diagnostics=f"Correct. {used}/{self._budget} tokens. Score: {score:.2f}",
            metadata={"tokens_used": used, "budget": self._budget},
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TokenBudgetGolfEnv(BaseReasoningEnv):
    """
    TokenBudgetGolf environment: solve a task within a hard token budget.

    Over budget = zero reward. Under budget + correct = high reward.
    Remaining budget fraction adds bonus.
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
            problem_generator = token_budget_golf_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return TokenBudgetGolfVerifier(
            subtype=meta["subtype"],
            budget=meta["budget"],
            answer=meta.get("answer", ""),
            tests=meta.get("tests", []),
        )

    def _check_format(self, response: str) -> float:
        if "ANSWER:" in response:
            return 1.0
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
