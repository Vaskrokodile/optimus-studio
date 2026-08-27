"""
InstructionDecomposition: Break a complex instruction into minimal sub-tasks.

Environment concept:
  The model is given a complex, multi-part instruction and must decompose it
  into a set of sub-tasks. The reward penalizes:
    - Over-decomposition: splitting simple steps into unnecessary sub-steps
    - Under-decomposition: missing necessary sub-tasks
    - Redundant sub-tasks: sub-tasks that repeat each other

  This trains the model to:
    1. Identify the minimal set of distinct operations needed
    2. Not over-plan (a 3-step task should not become 10 steps)
    3. Not under-plan (missing steps leads to execution failure)
    4. Write concise, actionable sub-task descriptions

  This directly attacks the frontier-model pattern of over-decomposing
  tasks into trivial micro-steps ("Step 1: Open the file. Step 2: Read the
  first line. Step 3: Read the second line...") when a single step would
  suffice.

Why this environment is worth using for 10T-100T param models:
  Task decomposition is the foundation of agentic planning. A 100T model
  that decomposes a 5-step task into 5 steps (not 20) saves 75% of planning
  tokens and 75% of execution overhead. This is the difference between a
  profitable and unprofitable agent.

Problem types:
  - Code tasks (implement a feature, fix a bug, refactor)
  - Data tasks (process, transform, analyze)
  - DevOps tasks (deploy, configure, monitor)
  - Mixed tasks (code + data + deployment)

Verification:
  Each problem has:
    - The complex instruction
    - The expected sub-tasks (ground truth decomposition)
    - The minimum and maximum acceptable number of sub-tasks
  The verifier:
    1. Parses the model's sub-tasks
    2. Checks count (too many = over-decomposition penalty)
    3. Matches each sub-task to expected sub-tasks (keyword overlap)
    4. Computes coverage and excess

Reward design:
  - Coverage: fraction of expected sub-tasks that have a match
  - Excess: fraction of model's sub-tasks that don't match any expected
  - Count penalty: if model produces >1.5x the expected count, penalty
  - Final = coverage * (1 - excess * 0.5) * count_factor
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


_DECOMPOSITION_PROBLEMS = [
    {
        "instruction": "Read a CSV file, filter rows where age > 30, sort by name, and write the result to a new file.",
        "expected_tasks": [
            "Read the CSV file",
            "Filter rows where age > 30",
            "Sort the filtered rows by name",
            "Write the result to a new file",
        ],
        "min_tasks": 3,
        "max_tasks": 5,
        "difficulty": 0.25,
    },
    {
        "instruction": "Fix a bug in the login function where passwords are compared in plaintext, then add unit tests for the fix.",
        "expected_tasks": [
            "Fix the password comparison to use hashed values",
            "Add unit tests for the login function",
        ],
        "min_tasks": 2,
        "max_tasks": 3,
        "difficulty": 0.3,
    },
    {
        "instruction": "Add a new API endpoint for deleting a user, including input validation, database operation, and error handling.",
        "expected_tasks": [
            "Add the API endpoint route",
            "Add input validation for the user ID",
            "Add the database delete operation",
            "Add error handling for not-found and database errors",
        ],
        "min_tasks": 3,
        "max_tasks": 5,
        "difficulty": 0.35,
    },
    {
        "instruction": "Refactor the calculate_total function to use a list comprehension and add type hints.",
        "expected_tasks": [
            "Refactor calculate_total to use list comprehension",
            "Add type hints to the function",
        ],
        "min_tasks": 1,
        "max_tasks": 3,
        "difficulty": 0.2,
    },
    {
        "instruction": "Set up a CI pipeline that runs linting, unit tests, and builds a Docker image on every push to main.",
        "expected_tasks": [
            "Configure the CI pipeline to trigger on push to main",
            "Add a linting step",
            "Add a unit test step",
            "Add a Docker image build step",
        ],
        "min_tasks": 3,
        "max_tasks": 5,
        "difficulty": 0.3,
    },
    {
        "instruction": "Write a function that fetches data from an API, parses the JSON response, extracts the 'results' field, and returns it as a list.",
        "expected_tasks": [
            "Fetch data from the API",
            "Parse the JSON response",
            "Extract the 'results' field and return as a list",
        ],
        "min_tasks": 2,
        "max_tasks": 4,
        "difficulty": 0.25,
    },
    {
        "instruction": "Debug why the application crashes on startup: check the logs, identify the error, fix the code, and verify the fix.",
        "expected_tasks": [
            "Check the application logs for errors",
            "Identify the root cause of the crash",
            "Fix the code that causes the crash",
            "Verify the fix by restarting the application",
        ],
        "min_tasks": 3,
        "max_tasks": 5,
        "difficulty": 0.3,
    },
    {
        "instruction": "Add pagination to the user list endpoint: accept page and limit parameters, query the database with OFFSET and LIMIT, and return paginated results.",
        "expected_tasks": [
            "Add page and limit parameters to the endpoint",
            "Query the database with OFFSET and LIMIT",
            "Return the paginated results",
        ],
        "min_tasks": 2,
        "max_tasks": 4,
        "difficulty": 0.3,
    },
    {
        "instruction": "Write a script that reads all Python files in a directory, counts the lines of code in each, and outputs a summary report.",
        "expected_tasks": [
            "Read all Python files in the directory",
            "Count lines of code in each file",
            "Output a summary report",
        ],
        "min_tasks": 2,
        "max_tasks": 4,
        "difficulty": 0.25,
    },
    {
        "instruction": "Migrate the user database from MySQL to PostgreSQL: export the data, transform the schema, import to PostgreSQL, and verify data integrity.",
        "expected_tasks": [
            "Export data from MySQL",
            "Transform the schema for PostgreSQL",
            "Import the data to PostgreSQL",
            "Verify data integrity after migration",
        ],
        "min_tasks": 3,
        "max_tasks": 5,
        "difficulty": 0.35,
    },
]


def instruction_decomposition_generator(seed: int) -> Problem:
    """Generate an InstructionDecomposition problem."""
    rng = random.Random(seed)
    template = rng.choice(_DECOMPOSITION_PROBLEMS)

    return Problem(
        id=f"instr_decomp_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["instruction"], template["max_tasks"]),
        difficulty=template["difficulty"],
        metadata={
            "type": "instruction_decomposition",
            "instruction": template["instruction"],
            "expected_tasks": template["expected_tasks"],
            "min_tasks": template["min_tasks"],
            "max_tasks": template["max_tasks"],
        },
        token_budget=400,
        source="generated",
    )


def _format_prompt(instruction: str, max_tasks: int) -> str:
    return (
        f"Decompose the following instruction into minimal sub-tasks.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Output format:\n"
        f"1. <sub-task description>\n"
        f"2. <sub-task description>\n"
        f"...\n\n"
        f"Rules:\n"
        f"  - Use the MINIMUM number of sub-tasks (max {max_tasks})\n"
        f"  - Each sub-task should be one clear, actionable step\n"
        f"  - Do NOT over-decompose (e.g., 'read file' should not become 'open file, read line 1, read line 2...')\n"
        f"  - Do NOT include redundant or repeated sub-tasks\n"
        f"  - Do NOT include explanation — just the numbered list"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class InstructionDecompositionVerifier(Verifier):
    """
    Verifies an InstructionDecomposition response by:
      1. Parsing the numbered sub-tasks
      2. Matching each to expected sub-tasks (keyword overlap)
      3. Computing coverage, excess, and count penalty
    """

    def __init__(self, expected_tasks: list[str], min_tasks: int, max_tasks: int):
        super().__init__()
        self._expected_tasks = expected_tasks
        self._min_tasks = min_tasks
        self._max_tasks = max_tasks

    def verify(self, response: str) -> VerifierResult:
        # Parse numbered sub-tasks
        tasks = self._parse_tasks(response)

        if not tasks:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No numbered sub-tasks found",
            )

        # Match each model task to expected tasks
        matched_expected = set()
        matched_model = set()

        for i, model_task in enumerate(tasks):
            best_match = -1
            best_score = 0.0

            for j, expected_task in enumerate(self._expected_tasks):
                if j in matched_expected:
                    continue
                score = self._keyword_overlap(model_task, expected_task)
                if score > best_score:
                    best_score = score
                    best_match = j

            if best_score >= 0.3 and best_match >= 0:
                matched_expected.add(best_match)
                matched_model.add(i)

        # Coverage: fraction of expected tasks matched
        coverage = len(matched_expected) / len(self._expected_tasks) if self._expected_tasks else 1.0

        # Excess: fraction of model tasks that don't match any expected
        excess = (len(tasks) - len(matched_model)) / len(tasks) if tasks else 0.0

        # Count penalty: if model produces too many tasks
        expected_count = len(self._expected_tasks)
        actual_count = len(tasks)
        if actual_count <= expected_count * 1.5:
            count_factor = 1.0
        elif actual_count <= self._max_tasks:
            count_factor = 0.8
        else:
            count_factor = max(0.3, 1.0 - (actual_count - self._max_tasks) * 0.15)

        # Final score
        score = coverage * (1.0 - excess * 0.5) * count_factor
        score = min(1.0, max(0.0, score))

        correct = score >= 0.7

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "coverage": coverage,
                "excess": excess,
                "count_factor": count_factor,
                "actual_count": actual_count,
                "expected_count": expected_count,
            },
            diagnostics=f"Coverage={coverage:.2f} Excess={excess:.2f} Count={actual_count}/{expected_count} (factor={count_factor:.2f})",
        )

    def _parse_tasks(self, response: str) -> list[str]:
        """Parse numbered sub-tasks from the response."""
        # Pattern: 1. <task> or 1) <task> — match until next numbered line or end
        matches = re.findall(r"(?:^|\n)\d+[\.\)]\s*(.+?)(?=\n\d+[\.\)]|\Z)", response)
        return [m.strip() for m in matches if m.strip()]

    def _keyword_overlap(self, text1: str, text2: str) -> float:
        """Compute keyword overlap between two strings."""
        # Extract meaningful words
        def extract_keywords(s: str) -> set[str]:
            s = s.lower()
            words = re.findall(r"\b[a-z]{3,}\b", s)
            common = {"the", "and", "for", "with", "that", "this", "from", "into", "then", "step", "add", "new"}
            return set(w for w in words if w not in common)

        kw1 = extract_keywords(text1)
        kw2 = extract_keywords(text2)

        if not kw1 or not kw2:
            return 0.0

        overlap = len(kw1 & kw2)
        union = len(kw1 | kw2)
        return overlap / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class InstructionDecompositionEnv(BaseReasoningEnv):
    """
    InstructionDecomposition environment: minimal task decomposition.

    The model decomposes a complex instruction into sub-tasks.
    Reward = coverage * (1 - excess * 0.5) * count_factor.
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
            problem_generator = instruction_decomposition_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return InstructionDecompositionVerifier(
            expected_tasks=problem.metadata["expected_tasks"],
            min_tasks=problem.metadata["min_tasks"],
            max_tasks=problem.metadata["max_tasks"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"(?:^|\n)\d+[\.\)]\s+", response):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
