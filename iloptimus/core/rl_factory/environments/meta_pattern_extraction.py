"""
MetaPatternExtraction: Extract abstract pattern from solved problems.

Environment concept:
  The model is given several solved problems and must identify the common
  abstract pattern (e.g. "all use divide-and-conquer", "all use two-pointer").

  Why: pattern abstraction is the core of transfer learning. A model that
  can identify "these all use the same technique" can apply that technique
  to new problems. This trains meta-level reasoning about algorithms.

Verification:
  - Check the extracted pattern matches the ground truth pattern keywords.
  - Partial credit for each keyword found.

Reward design:
  fraction of pattern_keywords found in the extracted pattern.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem bank — groups of solved problems sharing a common pattern
# ---------------------------------------------------------------------------

_PATTERN_GROUPS = [
    {
        "problems": [
            "Merge sort: split array in half, sort each half, merge.",
            "Quick sort: pick pivot, partition, recursively sort partitions.",
            "Binary search: split search space in half each step.",
        ],
        "pattern_description": "Divide and conquer: break the problem into smaller subproblems, solve recursively, combine results.",
        "pattern_keywords": ["divide", "conquer", "recursive", "subproblem", "split", "combine"],
    },
    {
        "problems": [
            "Two-sum in sorted array: one pointer at start, one at end, move inward.",
            "Palindrome check: pointers from both ends moving to center.",
            "Container with most water: two pointers moving toward each other.",
        ],
        "pattern_description": "Two-pointer technique: use two indices moving through the data structure, typically from opposite ends.",
        "pattern_keywords": ["two", "pointer", "index", "move", "opposite", "inward"],
    },
    {
        "problems": [
            "Fibonacci: F(n) = F(n-1) + F(n-2), store results in a table.",
            "Knapsack: build table of max value for each capacity.",
            "Edit distance: fill table of distances between substrings.",
        ],
        "pattern_description": "Dynamic programming: solve overlapping subproblems by building a table of solutions bottom-up.",
        "pattern_keywords": ["dynamic", "programming", "table", "subproblem", "bottom", "overlap"],
    },
    {
        "problems": [
            "BFS: explore nodes level by level using a queue.",
            "Dijkstra: use a priority queue to find shortest paths.",
            "Topological sort: process nodes in dependency order using a queue.",
        ],
        "pattern_description": "Graph traversal using a queue: explore nodes in breadth-first order, level by level.",
        "pattern_keywords": ["queue", "breadth", "level", "traversal", "graph", "fifo"],
    },
    {
        "problems": [
            "DFS: explore as deep as possible using a stack/recursion before backtracking.",
            "Maze solving: follow a path until dead end, backtrack, try another.",
            "Tree traversal: go deep to leaves before visiting siblings.",
        ],
        "pattern_description": "Depth-first search: explore deeply using a stack or recursion, backtrack when stuck.",
        "pattern_keywords": ["depth", "stack", "recursive", "backtrack", "deep", "lifo"],
    },
    {
        "problems": [
            "Hash map for counting: store frequency of each element.",
            "Hash set for membership: O(1) lookup for presence.",
            "Hash map for grouping: group items by key.",
        ],
        "pattern_description": "Hashing: use hash maps or sets for O(1) average lookup, insertion, and grouping.",
        "pattern_keywords": ["hash", "map", "set", "lookup", "o(1)", "key"],
    },
    {
        "problems": [
            "Greedy scheduling: always pick the earliest deadline first.",
            "Greedy coin change: always use the largest coin possible.",
            "Greedy Huffman coding: merge two least frequent nodes.",
        ],
        "pattern_description": "Greedy algorithm: make the locally optimal choice at each step, hoping for a global optimum.",
        "pattern_keywords": ["greedy", "local", "optimal", "choice", "step", "maximum"],
    },
    {
        "problems": [
            "Sliding window max: maintain a window of size k, slide across array.",
            "Longest substring without repeats: expand/contract window.",
            "Fixed window sum: compute sum of each consecutive k elements.",
        ],
        "pattern_description": "Sliding window: maintain a window of elements that slides across the data, updating incrementally.",
        "pattern_keywords": ["window", "slide", "subarray", "consecutive", "range", "incremental"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def meta_pattern_extraction_generator(seed: int) -> Problem:
    """Generate a MetaPatternExtraction problem: find the common pattern."""
    rng = random.Random(seed)
    group = rng.choice(_PATTERN_GROUPS)
    problems = group["problems"]
    pattern_description = group["pattern_description"]
    pattern_keywords = group["pattern_keywords"]

    problems_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(problems))

    prompt = (
        f"Several solved problems are listed below. They all share a common "
        f"algorithmic pattern. Identify the abstract pattern.\n\n"
        f"Solved problems:\n{problems_text}\n\n"
        f"What common pattern/technique do all these problems use?\n"
        f"Format your answer as: PATTERN: <description>"
    )

    difficulty = 0.5 + len(pattern_keywords) * 0.03
    difficulty = min(0.8, difficulty)

    return Problem(
        id=f"meta_pattern_extraction_{seed}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "problems": problems,
            "pattern_description": pattern_description,
            "pattern_keywords": pattern_keywords,
        },
        token_budget=512,
        source="meta_pattern_extraction_generator",
    )


meta_pattern_extraction_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class MetaPatternExtractionVerifier(Verifier):
    """Verify a meta-pattern extraction: check for pattern keywords.

    reward = fraction of pattern_keywords found in the description.
    """

    def __init__(self, pattern_keywords: list[str], pattern_description: str):
        super().__init__()
        self._keywords = [k.lower() for k in pattern_keywords]
        self._description = pattern_description.lower()

    def verify(self, response: str) -> VerifierResult:
        pattern_match = re.search(r"PATTERN\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if pattern_match:
            text = pattern_match.group(1).lower()
        else:
            text = response.lower()

        found = 0
        for kw in self._keywords:
            if kw in text:
                found += 1

        fraction = found / len(self._keywords) if self._keywords else 0.0
        correct = fraction >= 0.5

        return VerifierResult(
            correct=correct,
            score=fraction,
            partial_credit={
                "keywords_found": float(found),
                "keywords_total": float(len(self._keywords)),
                "keyword_fraction": fraction,
            },
            diagnostics=(
                f"keywords={found}/{len(self._keywords)} "
                f"fraction={fraction:.2f}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MetaPatternExtractionEnv(BatchEnvBase):
    """MetaPatternExtraction: extract abstract pattern from solved problems.

    Batch-aware: N parallel attempts; reward = best pattern extraction.
    """

    __test__ = False

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
            problem_generator = meta_pattern_extraction_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return MetaPatternExtractionVerifier(
            pattern_keywords=problem.metadata["pattern_keywords"],
            pattern_description=problem.metadata["pattern_description"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"PATTERN\s*:", response, re.IGNORECASE):
            return 1.0
        if len(response.strip()) > 20:
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
            "diagnostics": (
                f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores)/len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"PATTERN\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response
