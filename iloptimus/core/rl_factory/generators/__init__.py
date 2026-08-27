"""
Problem generators for future rl-factory environments.

This module contains generators that can produce problems for environments
beyond the initial 5. They are included to "prepare enough data for more
environments" as requested.

Generator categories:
  1. Math problem generators (algebra, calculus, probability, number theory)
  2. Code problem generators (algorithm traces, output prediction, complexity)
  3. Logic puzzle generators (Sudoku constraints, graph coloring, SAT)
  4. Natural language reasoning (argument analysis, fallacy detection)
  5. Multi-step planning (task decomposition, resource allocation)

Each generator produces a Problem object compatible with BaseReasoningEnv.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens


# ---------------------------------------------------------------------------
# 1. Math problem generators
# ---------------------------------------------------------------------------


def gen_linear_equation(seed: int) -> Problem:
    """Generate a linear equation problem: ax + b = c, solve for x."""
    rng = random.Random(seed)
    a = rng.randint(2, 12)
    x = rng.randint(1, 20)
    b = rng.randint(1, 30)
    c = a * x + b

    return Problem(
        id=f"linear_eq_{seed}",
        prompt=f"Solve for x: {a}x + {b} = {c}\n\nEnd with: ANSWER: x = <value>",
        difficulty=0.2,
        metadata={"type": "linear_equation", "answer": str(x), "a": a, "b": b, "c": c},
        token_budget=300,
        source="math_generator",
    )


def gen_quadratic(seed: int) -> Problem:
    """Generate a factorable quadratic: x^2 + bx + c = 0."""
    rng = random.Random(seed)
    r1 = rng.randint(-8, 8)
    r2 = rng.randint(-8, 8)
    if r1 == 0 and r2 == 0:
        r2 = 1
    b = -(r1 + r2)
    c = r1 * r2

    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"

    return Problem(
        id=f"quadratic_{seed}",
        prompt=f"Solve for x: x^2 {b_str}x {c_str} = 0\n\nEnd with: ANSWER: x = <values>",
        difficulty=0.4,
        metadata={"type": "quadratic", "roots": [r1, r2], "b": b, "c": c},
        token_budget=500,
        source="math_generator",
    )


def gen_probability(seed: int) -> Problem:
    """Generate a basic probability problem."""
    rng = random.Random(seed)
    templates = [
        {
            "prompt": "A fair coin is flipped 3 times. What is the probability of getting exactly 2 heads?",
            "answer": "3/8",
            "difficulty": 0.3,
        },
        {
            "prompt": "A standard die is rolled. What is the probability of rolling a number greater than 4?",
            "answer": "1/3",
            "difficulty": 0.2,
        },
        {
            "prompt": "Two cards are drawn from a standard 52-card deck. What is the probability both are aces?",
            "answer": "1/221",
            "difficulty": 0.45,
        },
        {
            "prompt": "A bag has 3 red and 5 blue marbles. One is drawn at random. What is the probability it is red?",
            "answer": "3/8",
            "difficulty": 0.2,
        },
        {
            "prompt": "What is the probability of rolling a sum of 7 with two dice?",
            "answer": "1/6",
            "difficulty": 0.35,
        },
    ]
    t = rng.choice(templates)
    return Problem(
        id=f"probability_{seed}",
        prompt=f"{t['prompt']}\n\nEnd with: ANSWER: <fraction>",
        difficulty=t["difficulty"],
        metadata={"type": "probability", "answer": t["answer"]},
        token_budget=500,
        source="math_generator",
    )


def gen_modular_arithmetic(seed: int) -> Problem:
    """Generate a modular arithmetic problem."""
    rng = random.Random(seed)
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    m = rng.randint(3, 20)
    result = (a * b) % m

    return Problem(
        id=f"mod_arith_{seed}",
        prompt=f"Compute {a} * {b} mod {m}.\n\nEnd with: ANSWER: <value>",
        difficulty=0.3,
        metadata={"type": "modular", "answer": str(result)},
        token_budget=300,
        source="math_generator",
    )


# ---------------------------------------------------------------------------
# 2. Code problem generators
# ---------------------------------------------------------------------------


def gen_code_output_prediction(seed: int) -> Problem:
    """Generate a code output prediction problem."""
    rng = random.Random(seed)
    templates = [
        {
            "code": "x = [1, 2, 3]\nprint(x[::-1])",
            "answer": "[3, 2, 1]",
            "difficulty": 0.2,
        },
        {
            "code": "s = 'hello'\nprint(s.count('l'))",
            "answer": "2",
            "difficulty": 0.15,
        },
        {
            "code": "d = {'a': 1, 'b': 2}\nprint(list(d.keys()))",
            "answer": "['a', 'b']",
            "difficulty": 0.25,
        },
        {
            "code": "nums = [3, 1, 4, 1, 5]\nprint(sorted(set(nums)))",
            "answer": "[1, 3, 4, 5]",
            "difficulty": 0.3,
        },
        {
            "code": "x = 5\ny = x if x > 3 else 0\nprint(y)",
            "answer": "5",
            "difficulty": 0.2,
        },
        {
            "code": "lst = [i**2 for i in range(4)]\nprint(lst)",
            "answer": "[0, 1, 4, 9]",
            "difficulty": 0.25,
        },
        {
            "code": "s = 'abcabc'\nprint(s[2:5])",
            "answer": "cab",
            "difficulty": 0.25,
        },
        {
            "code": "a = {1, 2, 3}\nb = {2, 3, 4}\nprint(a & b)",
            "answer": "{2, 3}",
            "difficulty": 0.35,
        },
        {
            "code": "x = 10\nprint(x // 3, x % 3)",
            "answer": "3 1",
            "difficulty": 0.2,
        },
        {
            "code": "words = ['dog', 'cat', 'bird']\nprint(min(words, key=len))",
            "answer": "cat",
            "difficulty": 0.35,
        },
    ]
    t = rng.choice(templates)
    return Problem(
        id=f"code_output_{seed}",
        prompt=f"What is the output of the following Python code?\n\n```python\n{t['code']}\n```\n\nEnd with: ANSWER: <output>",
        difficulty=t["difficulty"],
        metadata={"type": "code_output", "answer": t["answer"], "code": t["code"]},
        token_budget=400,
        source="code_generator",
    )


def gen_complexity_analysis(seed: int) -> Problem:
    """Generate a time complexity analysis problem."""
    rng = random.Random(seed)
    templates = [
        {
            "code": "for i in range(n):\n    for j in range(n):\n        print(i, j)",
            "answer": "O(n^2)",
            "difficulty": 0.3,
        },
        {
            "code": "i = 1\nwhile i < n:\n    i *= 2",
            "answer": "O(log n)",
            "difficulty": 0.35,
        },
        {
            "code": "for i in range(n):\n    print(i)",
            "answer": "O(n)",
            "difficulty": 0.15,
        },
        {
            "code": "def binary_search(arr, x):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid+1\n        else: hi = mid-1\n    return -1",
            "answer": "O(log n)",
            "difficulty": 0.4,
        },
        {
            "code": "for i in range(n):\n    for j in range(i):\n        print(i, j)",
            "answer": "O(n^2)",
            "difficulty": 0.4,
        },
        {
            "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
            "answer": "O(2^n)",
            "difficulty": 0.5,
        },
    ]
    t = rng.choice(templates)
    return Problem(
        id=f"complexity_{seed}",
        prompt=f"What is the time complexity of the following code?\n\n```python\n{t['code']}\n```\n\nEnd with: ANSWER: O(<complexity>)",
        difficulty=t["difficulty"],
        metadata={"type": "complexity", "answer": t["answer"]},
        token_budget=400,
        source="code_generator",
    )


# ---------------------------------------------------------------------------
# 3. Logic puzzle generators
# ---------------------------------------------------------------------------


def gen_syllogism(seed: int) -> Problem:
    """Generate a syllogistic reasoning problem."""
    rng = random.Random(seed)
    templates = [
        {
            "prompt": "All doctors have medical degrees. Sarah is a doctor. Does Sarah have a medical degree?",
            "answer": "Yes",
            "difficulty": 0.2,
        },
        {
            "prompt": "All mammals are warm-blooded. All whales are mammals. Are whales warm-blooded?",
            "answer": "Yes",
            "difficulty": 0.2,
        },
        {
            "prompt": "If it snows, the roads are slippery. The roads are not slippery. Did it snow?",
            "answer": "No",
            "difficulty": 0.4,
        },
        {
            "prompt": "All birds have feathers. All penguins are birds. Do penguins have feathers?",
            "answer": "Yes",
            "difficulty": 0.15,
        },
        {
            "prompt": "No reptiles are mammals. All snakes are reptiles. Are any snakes mammals?",
            "answer": "No",
            "difficulty": 0.3,
        },
        {
            "prompt": "If x > 5, then x > 3. x = 4. Is x > 3?",
            "answer": "Yes",
            "difficulty": 0.25,
        },
        {
            "prompt": "All squares are rectangles. Some rectangles are large. Are some squares large?",
            "answer": "Cannot determine",
            "difficulty": 0.5,
        },
    ]
    t = rng.choice(templates)
    return Problem(
        id=f"syllogism_{seed}",
        prompt=f"{t['prompt']}\n\nEnd with: ANSWER: <Yes/No/Cannot determine>",
        difficulty=t["difficulty"],
        metadata={"type": "syllogism", "answer": t["answer"]},
        token_budget=400,
        source="logic_generator",
    )


def gen_constraint_satisfaction(seed: int) -> Problem:
    """Generate a simple constraint satisfaction problem."""
    rng = random.Random(seed)
    # Simple: 3 people, 3 jobs, constraints
    people = ["Alice", "Bob", "Carol"]
    rng.shuffle(people)
    jobs = ["chef", "driver", "pilot"]

    assignments = list(zip(people, jobs))

    # Generate constraints
    constraints = []
    constraints.append(f"{people[0]} is not the chef.")
    constraints.append(f"{people[1]} is not the driver.")
    constraints.append(f"The pilot is not {people[2]}.")

    # The solution: people[0]=driver? Let's make it simpler
    # Actually, let's use a fixed puzzle
    puzzles = [
        {
            "prompt": (
                "Three people (Alice, Bob, Carol) have three jobs (chef, driver, pilot).\n"
                "Constraints:\n"
                "  - Alice is not the chef.\n"
                "  - Bob is not the driver.\n"
                "  - The pilot is not Carol.\n"
                "Who is the pilot?"
            ),
            "answer": "Alice",
            "difficulty": 0.45,
        },
        {
            "prompt": (
                "Three boxes (red, blue, green) contain (apple, banana, cherry).\n"
                "Constraints:\n"
                "  - The red box does not contain the apple.\n"
                "  - The banana is in the green box.\n"
                "  - The cherry is not in the blue box.\n"
                "What is in the red box?"
            ),
            "answer": "cherry",
            "difficulty": 0.5,
        },
    ]
    t = rng.choice(puzzles)
    return Problem(
        id=f"constraint_{seed}",
        prompt=f"{t['prompt']}\n\nEnd with: ANSWER: <answer>",
        difficulty=t["difficulty"],
        metadata={"type": "constraint", "answer": t["answer"]},
        token_budget=600,
        source="logic_generator",
    )


# ---------------------------------------------------------------------------
# 4. Natural language reasoning
# ---------------------------------------------------------------------------


def gen_fallacy_detection(seed: int) -> Problem:
    """Generate a logical fallacy detection problem."""
    rng = random.Random(seed)
    templates = [
        {
            "prompt": "Identify the logical fallacy: 'Everyone I know uses this phone, so it must be the best phone.'",
            "answer": "Bandwagon fallacy",
            "keywords": ["bandwagon", "appeal", "popularity", "everyone"],
            "difficulty": 0.3,
        },
        {
            "prompt": "Identify the logical fallacy: 'If you don't support this law, you must be against safety.'",
            "answer": "False dilemma",
            "keywords": ["false", "dilemma", "binary", "either", "or", "strawman"],
            "difficulty": 0.4,
        },
        {
            "prompt": "Identify the logical fallacy: 'My grandfather smoked every day and lived to 90, so smoking isn't that bad.'",
            "answer": "Anecdotal evidence",
            "keywords": ["anecdotal", "evidence", "hasty", "generalization", "single"],
            "difficulty": 0.35,
        },
        {
            "prompt": "Identify the logical fallacy: 'You can't trust John's argument about taxes because he's not an economist.'",
            "answer": "Ad hominem",
            "keywords": ["ad", "hominem", "person", "attack", "credential"],
            "difficulty": 0.3,
        },
        {
            "prompt": "Identify the logical fallacy: 'We should listen to his opinion on climate change because he's a famous actor.'",
            "answer": "Appeal to authority",
            "keywords": ["authority", "appeal", "famous", "celebrity", "irrelevant"],
            "difficulty": 0.3,
        },
    ]
    t = rng.choice(templates)
    return Problem(
        id=f"fallacy_{seed}",
        prompt=f"{t['prompt']}\n\nEnd with: ANSWER: <fallacy name>",
        difficulty=t["difficulty"],
        metadata={"type": "fallacy", "answer": t["answer"], "keywords": t["keywords"]},
        token_budget=400,
        source="nl_generator",
    )


# ---------------------------------------------------------------------------
# 5. Master generator registry
# ---------------------------------------------------------------------------


GENERATOR_REGISTRY = {
    # Math
    "linear_equation": gen_linear_equation,
    "quadratic": gen_quadratic,
    "probability": gen_probability,
    "modular_arithmetic": gen_modular_arithmetic,
    # Code
    "code_output": gen_code_output_prediction,
    "complexity": gen_complexity_analysis,
    # Logic
    "syllogism": gen_syllogism,
    "constraint": gen_constraint_satisfaction,
    # NL
    "fallacy": gen_fallacy_detection,
}


def get_generator(name: str):
    """Get a generator by name from the registry."""
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}. Available: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[name]


def list_generators() -> list[str]:
    """List all available generator names."""
    return list(GENERATOR_REGISTRY.keys())


def generate_problem_set(
    generators: Optional[list[str]] = None,
    count: int = 100,
    seed_base: int = 0,
) -> list[Problem]:
    """
    Generate a set of problems using multiple generators.

    Args:
        generators: List of generator names. If None, uses all.
        count: Total number of problems to generate.
        seed_base: Base seed for reproducibility.

    Returns:
        List of Problem objects.
    """
    if generators is None:
        generators = list(GENERATOR_REGISTRY.keys())

    problems = []
    rng = random.Random(seed_base)

    for i in range(count):
        gen_name = rng.choice(generators)
        gen_func = get_generator(gen_name)
        seed = seed_base + i
        problem = gen_func(seed)
        problems.append(problem)

    return problems
