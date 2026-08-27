"""
SkillCollision: Train tradeoff reasoning — tasks where two skills interfere.

Environment concept:
  The model receives tasks where two objectives CONFLICT. It must choose
  one approach over another and justify the tradeoff. This trains a
  fundamentally different skill than pure correctness: navigating Pareto
  frontiers.

  5 collision types:
    1. speed_vs_correctness: Fast heuristic (may be wrong) vs exhaustive (more tokens)
    2. local_vs_global: Greedy local optimum vs planned global optimum
    3. short_vs_long_term: Quick fix (works now, breaks later) vs proper fix
    4. correctness_vs_resources: Must be correct BUT within resource limits
    5. robustness_vs_conciseness: Defensive code (handles edge cases) vs minimal code

  reward = dimension_a_score * w_a + dimension_b_score * w_b

  This is a batch environment: N parallel solutions, reward = best + diversity
  (diversity = different tradeoff choices across the batch).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: each has two conflicting dimensions
# ---------------------------------------------------------------------------


_COLLISION_PROBLEMS = [
    # --- speed_vs_correctness ---
    {
        "type": "speed_vs_correctness",
        "task": "Determine if 104729 is prime. You have a tight token budget.",
        "option_a": "Fast heuristic: check divisibility by 2, 3, 5, 7 only (may miss some primes)",
        "option_b": "Exhaustive: check divisibility up to sqrt(104729) ~ 323 (guaranteed correct but more tokens)",
        "constraint": "Token budget: 80 tokens. The exhaustive approach may exceed this.",
        "test": "assert is_prime(104729) == True; assert is_prime(104730) == False; assert is_prime(2) == True",
        "fast_code": "def is_prime(n):\n    if n < 2: return False\n    for d in [2,3,5,7]:\n        if n == d: return True\n        if n % d == 0: return False\n    return True",
        "correct_code": "def is_prime(n):\n    if n < 2: return False\n    if n < 4: return True\n    if n % 2 == 0: return False\n    d = 3\n    while d * d <= n:\n        if n % d == 0: return False\n        d += 2\n    return True",
        "w_a": 0.4, "w_b": 0.6,
    },
    {
        "type": "speed_vs_correctness",
        "task": "Find the maximum element in a list. You have a tight token budget.",
        "option_a": "Fast: return lst[0] (O(1) but wrong for most lists)",
        "option_b": "Correct: iterate through all elements (O(n) but always right)",
        "constraint": "Token budget: 40 tokens.",
        "test": "assert find_max([3,1,4,1,5,9,2,6]) == 9; assert find_max([1]) == 1; assert find_max([-1,-2,-3]) == -1",
        "fast_code": "def find_max(lst): return lst[0]",
        "correct_code": "def find_max(lst): return max(lst)",
        "w_a": 0.3, "w_b": 0.7,
    },
    {
        "type": "speed_vs_correctness",
        "task": "Check if a string is a palindrome. You have a tight token budget.",
        "option_a": "Fast: check first and last char only (may be wrong)",
        "option_b": "Correct: check all characters (guaranteed correct)",
        "constraint": "Token budget: 60 tokens.",
        "test": "assert is_palindrome('racecar') == True; assert is_palindrome('hello') == False; assert is_palindrome('a') == True; assert is_palindrome('ab') == False",
        "fast_code": "def is_palindrome(s): return s[0] == s[-1] if s else True",
        "correct_code": "def is_palindrome(s): return s == s[::-1]",
        "w_a": 0.3, "w_b": 0.7,
    },
    # --- local_vs_global ---
    {
        "type": "local_vs_global",
        "task": "Make change for 67 cents using coins [25, 10, 5, 1]. Minimize total coins.",
        "option_a": "Greedy: always take the largest coin that fits (fast, optimal for US coins but not all coin systems)",
        "option_b": "Dynamic programming: try all combinations (guaranteed optimal but more code)",
        "constraint": "The greedy approach works for US coins. But can you guarantee it's optimal?",
        "test": "assert make_change(67, [25,10,5,1]) == 6; assert make_change(30, [25,10,5,1]) == 2; assert make_change(7, [25,10,5,1]) == 3",
        "fast_code": "def make_change(amount, coins):\n    count = 0\n    for c in sorted(coins, reverse=True):\n        count += amount // c\n        amount %= c\n    return count",
        "correct_code": "def make_change(amount, coins):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for i in range(1, amount + 1):\n        for c in coins:\n            if c <= i:\n                dp[i] = min(dp[i], dp[i - c] + 1)\n    return dp[amount]",
        "w_a": 0.6, "w_b": 0.4,
    },
    {
        "type": "local_vs_global",
        "task": "Find the shortest path in a weighted graph from A to D. Edges: A-B:1, A-C:4, B-C:1, B-D:5, C-D:1.",
        "option_a": "Greedy: always go to nearest unvisited node (A->B->C->D, cost=3)",
        "option_b": "Exhaustive: try all paths and pick shortest (guaranteed optimal)",
        "constraint": "The greedy approach may not find the shortest path in general graphs.",
        "test": "assert shortest_path('A', 'D') == 3; assert shortest_path('A', 'C') == 2; assert shortest_path('B', 'D') == 2",
        "fast_code": "def shortest_path(start, end):\n    # Greedy: A->B->C->D\n    paths = {('A','B'):1,('A','C'):4,('B','C'):1,('B','D'):5,('C','D'):1}\n    # Simplified greedy for this specific graph\n    if start=='A' and end=='D': return 3\n    if start=='A' and end=='C': return 2\n    if start=='B' and end=='D': return 2\n    return float('inf')",
        "correct_code": "def shortest_path(start, end):\n    import heapq\n    graph = {'A':[('B',1),('C',4)], 'B':[('C',1),('D',5)], 'C':[('D',1)], 'D':[]}\n    dist = {n: float('inf') for n in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, node = heapq.heappop(pq)\n        if node == end: return d\n        for nb, w in graph.get(node, []):\n            if d + w < dist[nb]:\n                dist[nb] = d + w\n                heapq.heappush(pq, (d + w, nb))\n    return dist[end]",
        "w_a": 0.5, "w_b": 0.5,
    },
    # --- short_vs_long_term ---
    {
        "type": "short_vs_long_term",
        "task": "Fix a function that crashes on empty input. def average(lst): return sum(lst) / len(lst)",
        "option_a": "Quick fix: add try/except (works now but hides other potential errors)",
        "option_b": "Proper fix: check for empty list and return None (more robust, handles edge cases explicitly)",
        "constraint": "The quick fix is faster to write. The proper fix is more maintainable.",
        "test": "assert average([1,2,3]) == 2.0; assert average([]) == None; assert average([5]) == 5.0",
        "fast_code": "def average(lst):\n    try:\n        return sum(lst) / len(lst)\n    except:\n        return None",
        "correct_code": "def average(lst):\n    if not lst:\n        return None\n    return sum(lst) / len(lst)",
        "w_a": 0.4, "w_b": 0.6,
    },
    {
        "type": "short_vs_long_term",
        "task": "Fix a function that has an off-by-one error. def countdown(n): return list(range(n, 0, -1)) — should include 0.",
        "option_a": "Quick fix: append 0 to the result (works but is hacky)",
        "option_b": "Proper fix: change range to range(n, -1, -1) (correct and clean)",
        "constraint": "Both fixes work. The proper fix is cleaner and more maintainable.",
        "test": "assert countdown(5) == [5,4,3,2,1,0]; assert countdown(0) == [0]; assert countdown(3) == [3,2,1,0]",
        "fast_code": "def countdown(n):\n    result = list(range(n, 0, -1))\n    result.append(0)\n    return result",
        "correct_code": "def countdown(n):\n    return list(range(n, -1, -1))",
        "w_a": 0.4, "w_b": 0.6,
    },
    # --- correctness_vs_resources ---
    {
        "type": "correctness_vs_resources",
        "task": "Write a function to compute fibonacci(20). You must use NO loops and NO recursion.",
        "option_a": "Recursive (correct but uses recursion — violates constraint)",
        "option_b": "Closed-form Binet's formula (correct, no loops/recursion, but uses floating point)",
        "constraint": "No loops, no recursion. Must be correct for fibonacci(20).",
        "test": "assert fibonacci(20) == 6765; assert fibonacci(10) == 55; assert fibonacci(0) == 0",
        "fast_code": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "correct_code": "def fibonacci(n):\n    phi = (1 + 5 ** 0.5) / 2\n    return round(phi ** n / 5 ** 0.5)",
        "w_a": 0.2, "w_b": 0.8,
    },
    {
        "type": "correctness_vs_resources",
        "task": "Sort a list of 1000 elements. You must use NO built-in sort functions.",
        "option_a": "Use sorted() (correct but violates constraint)",
        "option_b": "Implement merge sort (correct, no built-ins, but more code)",
        "constraint": "No sorted(), no .sort(). Must be correct and efficient.",
        "test": "assert sort_list([3,1,4,1,5,9,2,6]) == [1,1,2,3,4,5,6,9]; assert sort_list([]) == []; assert sort_list([1]) == [1]",
        "fast_code": "def sort_list(lst): return sorted(lst)",
        "correct_code": "def sort_list(lst):\n    if len(lst) <= 1: return lst\n    mid = len(lst) // 2\n    left = sort_list(lst[:mid])\n    right = sort_list(lst[mid:])\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
        "w_a": 0.1, "w_b": 0.9,
    },
    # --- robustness_vs_conciseness ---
    {
        "type": "robustness_vs_conciseness",
        "task": "Write a function to divide two numbers. Handle all edge cases.",
        "option_a": "Concise: def divide(a, b): return a / b (clean but crashes on zero division)",
        "option_b": "Robust: check for zero, check for None, check for non-numeric (verbose but safe)",
        "constraint": "Must handle: division by zero, None inputs, non-numeric inputs.",
        "test": "assert divide(10, 2) == 5; assert divide(7, 0) == None; assert divide(None, 2) == None; assert divide(10, None) == None",
        "fast_code": "def divide(a, b): return a / b",
        "correct_code": "def divide(a, b):\n    if a is None or b is None: return None\n    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)): return None\n    if b == 0: return None\n    return a / b",
        "w_a": 0.3, "w_b": 0.7,
    },
    {
        "type": "robustness_vs_conciseness",
        "task": "Write a function to get the first element of a list. Handle edge cases.",
        "option_a": "Concise: def first(lst): return lst[0] (clean but crashes on empty list)",
        "option_b": "Robust: check for None, check for empty, check for non-list (verbose but safe)",
        "constraint": "Must handle: empty list, None input, non-list input.",
        "test": "assert first([1,2,3]) == 1; assert first([]) == None; assert first(None) == None; assert first('hello') == None",
        "fast_code": "def first(lst): return lst[0]",
        "correct_code": "def first(lst):\n    if lst is None: return None\n    if not isinstance(lst, list): return None\n    if len(lst) == 0: return None\n    return lst[0]",
        "w_a": 0.3, "w_b": 0.7,
    },
]


def skill_collision_generator(seed: int) -> Problem:
    """Generate a SkillCollision problem."""
    rng = random.Random(seed)
    template = rng.choice(_COLLISION_PROBLEMS)

    collision_type = template["type"]
    w_a = template["w_a"]
    w_b = template["w_b"]

    return Problem(
        id=f"skill_collision_{rng.randint(0, 99999)}",
        prompt=(
            f"TRADEOFF: {collision_type}\n"
            f"TASK: {template['task']}\n"
            f"OPTION A: {template['option_a']}\n"
            f"OPTION B: {template['option_b']}\n"
            f"CONSTRAINT: {template['constraint']}\n\n"
            f"TESTS:\n{template['test']}\n\n"
            f"Choose ONE approach and implement it.\n"
            f"CHOICE: A or B\n"
            f"JUSTIFICATION: <why this tradeoff is right here>\n"
            f"SOLUTION: ```python\n<your code>\n```"
        ),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "skill_collision",
            "collision_type": collision_type,
            "task": template["task"],
            "test_code": template["test"],
            "fast_code": template["fast_code"],
            "correct_code": template["correct_code"],
            "w_a": w_a,
            "w_b": w_b,
            "option_a": template["option_a"],
            "option_b": template["option_b"],
        },
        token_budget=200,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — scores both dimensions of the tradeoff
# ---------------------------------------------------------------------------


class SkillCollisionVerifier(Verifier):
    """Verifies a skill collision response by scoring both tradeoff dimensions."""

    def __init__(self, test_code: str, fast_code: str, correct_code: str,
                 w_a: float, w_b: float, collision_type: str):
        super().__init__()
        self._test_code = test_code
        self._fast_code = fast_code
        self._correct_code = correct_code
        self._w_a = w_a
        self._w_b = w_b
        self._collision_type = collision_type

    def verify(self, response: str) -> VerifierResult:
        choice = self._parse_choice(response)
        solution_code = self._parse_solution(response)

        if not solution_code:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No solution code found",
            )

        # Run the tests on the solution
        tests_pass = self._run_tests(solution_code, self._test_code)

        # Score dimension A (speed/local/short-term/concise)
        # and dimension B (correctness/global/long-term/robust)
        if self._collision_type == "speed_vs_correctness":
            dim_a = self._score_speed(solution_code)
            dim_b = 1.0 if tests_pass else 0.0
        elif self._collision_type == "local_vs_global":
            dim_a = self._score_conciseness(solution_code, self._fast_code)
            dim_b = 1.0 if tests_pass else 0.0
        elif self._collision_type == "short_vs_long_term":
            dim_a = self._score_conciseness(solution_code, self._fast_code)
            dim_b = 1.0 if tests_pass else 0.0
        elif self._collision_type == "correctness_vs_resources":
            constraint_met = self._check_constraint(solution_code)
            dim_a = 1.0 if constraint_met else 0.0
            dim_b = 1.0 if tests_pass else 0.0
        elif self._collision_type == "robustness_vs_conciseness":
            edge_cases_pass = tests_pass  # Tests include edge cases
            dim_a = self._score_conciseness(solution_code, self._correct_code)
            dim_b = 1.0 if edge_cases_pass else 0.0
        else:
            dim_a = 1.0 if tests_pass else 0.0
            dim_b = 1.0 if tests_pass else 0.0

        # Weighted combination
        score = dim_a * self._w_a + dim_b * self._w_b
        correct = tests_pass and score >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "collision_type": self._collision_type,
                "choice": choice,
                "dim_a_score": dim_a,
                "dim_b_score": dim_b,
                "w_a": self._w_a,
                "w_b": self._w_b,
                "tests_pass": tests_pass,
            },
            diagnostics=f"Type={self._collision_type} Choice={choice} DimA={dim_a:.2f} DimB={dim_b:.2f} Score={score:.2f}",
        )

    def _parse_choice(self, response: str) -> str:
        match = re.search(r"CHOICE:\s*([AB])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "unknown"

    def _parse_solution(self, response: str) -> str:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try SOLUTION: prefix
        match = re.search(r"SOLUTION:\s*(.+?)(?:\n[A-Z]|\Z)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _run_tests(self, code: str, test_code: str) -> bool:
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
            exec(test_code, namespace)
            return True
        except Exception:
            return False

    def _score_speed(self, code: str) -> float:
        """Score speed dimension: fewer tokens = higher score."""
        tokens = estimate_tokens(code)
        # Normalize: 20 tokens = 1.0, 200 tokens = 0.0
        return max(0.0, min(1.0, 1.0 - (tokens - 20) / 180))

    def _score_conciseness(self, code: str, reference: str) -> float:
        """Score conciseness: shorter than reference = higher score."""
        code_tokens = estimate_tokens(code)
        ref_tokens = estimate_tokens(reference)
        if ref_tokens == 0:
            return 1.0
        return max(0.0, min(1.0, ref_tokens / max(code_tokens, 1)))

    def _check_constraint(self, code: str) -> bool:
        """Check if the solution respects the resource constraint (no loops/recursion)."""
        # Check for loop keywords
        if re.search(r"\bfor\b|\bwhile\b", code):
            return False
        # Check for recursive calls (simplified: function calling itself)
        # This is a heuristic — can't fully detect recursion statically
        func_match = re.search(r"def (\w+)\(", code)
        if func_match:
            func_name = func_match.group(1)
            if re.search(rf"\b{func_name}\s*\(", code.split("def", 1)[-1]):
                return False
        return True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SkillCollisionEnv(BatchEnvBase):
    """
    SkillCollision: train tradeoff reasoning with conflicting objectives.

    Batch-aware: N parallel solutions, reward = best_score * 0.7 + diversity * 0.3
    (diversity = different tradeoff choices across the batch).
    """

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = skill_collision_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SkillCollisionVerifier(
            test_code=problem.metadata["test_code"],
            fast_code=problem.metadata["fast_code"],
            correct_code=problem.metadata["correct_code"],
            w_a=problem.metadata["w_a"],
            w_b=problem.metadata["w_b"],
            collision_type=problem.metadata["collision_type"],
        )

    def _check_format(self, response: str) -> float:
        has_choice = bool(re.search(r"CHOICE:\s*[AB]", response, re.IGNORECASE))
        has_code = bool(re.search(r"```python", response, re.IGNORECASE))
        has_just = bool(re.search(r"JUSTIFICATION:", response, re.IGNORECASE))
        if has_choice and has_code and has_just:
            return 1.0
        if has_choice and has_code:
            return 0.7
        if has_code:
            return 0.4
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        # Diversity: different tradeoff choices across the batch
        choices = set()
        for s in per_sample:
            resp = s.get("response", "")
            match = re.search(r"CHOICE:\s*([AB])", resp, re.IGNORECASE)
            if match:
                choices.add(match.group(1).upper())
        diversity = len(choices) / 2.0 if per_sample else 0.0

        if best_score <= 0:
            reward = 0.0
        else:
            reward = best_score * 0.7 + diversity * 0.3
            reward = min(1.0, reward)

        return {
            "best_score": best_score,
            "diversity": diversity,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Diversity={diversity:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
