"""
ContextHunter: Extract only the relevant code from a codebase.

Environment concept:
  The model is given a multi-file codebase and a task (find a bug, add a
  feature, understand a behavior). It must output a set of file:line ranges
  that contain ONLY the relevant code — no more, no less.

  This trains the model to:
    1. Navigate codebases efficiently (not read everything)
    2. Identify the root cause location precisely
    3. Avoid pulling in irrelevant context (which wastes tokens)
    4. Not miss critical context (which leads to wrong fixes)

  This directly attacks the frontier-model pattern of reading entire files
  or entire codebases when only a few lines matter. At 100T parameters,
  context window efficiency is the difference between $10 and $100 per
  inference call.

Why this environment is worth using for 10T-100T param models:
  Context distillation is the most under-trained skill in frontier models.
  Models are trained to GENERATE code but not to SELECT what to read.
  ContextHunter trains the "attention" skill — knowing what matters —
  which compounds with every other agentic capability.

Problem types:
  - Bug location: given a bug report, find the exact lines
  - Feature impact: given a feature request, find all affected lines
  - Dependency tracing: given a function call, find the definition + all callers
  - Dead code: find code that is never called
  - Security: find all input validation points

Verification:
  Each problem has:
    - A codebase (dict of filename -> content)
    - A task description
    - "Relevant lines": the ground-truth set of file:line ranges
  The verifier computes:
    - Precision: fraction of model's selections that are relevant
    - Recall: fraction of relevant lines the model found
    - F1 score = 2 * P * R / (P + R)

Reward design:
  - F1 score is the primary reward (must find all relevant + no irrelevant)
  - Conciseness bonus: if the model outputs fewer lines than the total
    relevant lines, it gets a bonus (it found the signal in the noise)
  - Anti-pattern penalties for verbose reasoning
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


def _make_bug_location_problem(rng: random.Random) -> Problem:
    """
    Task: Find the exact line with a bug in a multi-file codebase.
    """
    templates = [
        {
            "files": {
                "src/calc.py": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b\n",
                "src/main.py": "from calc import add, subtract\n\nresult = add(1, 2)\nprint(result)\n",
                "src/utils.py": "def format_result(value):\n    return str(value)\n\ndef log_message(msg):\n    print(msg)\n",
            },
            "task": "The function `divide` in src/calc.py will crash when b=0. Find the exact line that needs fixing.",
            "relevant": {"src/calc.py": [8]},  # line 8: return a / b
        },
        {
            "files": {
                "src/auth.py": "def check_password(user_pw, stored_pw):\n    return user_pw == stored_pw\n\ndef hash_password(pw):\n    return hash(pw)\n",
                "src/models.py": "class User:\n    def __init__(self, name, pw):\n        self.name = name\n        self.password = pw\n",
                "src/api.py": "def login(username, password):\n    user = get_user(username)\n    if check_password(password, user.password):\n        return True\n    return False\n",
            },
            "task": "The password check in src/auth.py compares passwords in plaintext. Find the exact line that should use hashed comparison instead.",
            "relevant": {"src/auth.py": [2]},  # line 2: return user_pw == stored_pw
        },
        {
            "files": {
                "src/loop.py": "def process_items(items):\n    result = []\n    for i in range(len(items)):\n        result.append(items[i] * 2)\n    return result\n\ndef filter_positive(numbers):\n    return [n for n in numbers if n > 0]\n",
                "src/main.py": "from loop import process_items\ndata = [1, -2, 3, -4, 5]\nprocessed = process_items(data)\nprint(processed)\n",
            },
            "task": "The function `process_items` works but `filter_positive` has an off-by-one: it should include 0. Find the exact line.",
            "relevant": {"src/loop.py": [7]},  # line 7: if n > 0
        },
        {
            "files": {
                "src/db.py": "def get_user(user_id):\n    query = f'SELECT * FROM users WHERE id = {user_id}'\n    return execute(query)\n\ndef get_all_users():\n    query = 'SELECT * FROM users'\n    return execute(query)\n",
                "src/helpers.py": "def execute(query):\n    # Simulated DB execution\n    return []\n\ndef validate_input(value):\n    if not isinstance(value, str):\n        raise ValueError('Input must be string')\n",
            },
            "task": "The function `get_user` in src/db.py has a SQL injection vulnerability. Find the exact line.",
            "relevant": {"src/db.py": [2]},  # line 2: query = f'SELECT...
        },
    ]

    template = rng.choice(templates)
    difficulty = 0.3 + 0.2 * rng.random()

    return Problem(
        id=f"context_hunt_bug_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["files"], template["task"]),
        difficulty=difficulty,
        metadata={
            "type": "context_hunter",
            "files": template["files"],
            "relevant": template["relevant"],
            "task": template["task"],
        },
        token_budget=600,
        source="generated",
    )


def _make_dependency_trace_problem(rng: random.Random) -> Problem:
    """
    Task: Find the definition of a function and all its callers.
    """
    templates = [
        {
            "files": {
                "src/api.py": "from utils import format_date\n\ndef get_users():\n    users = fetch_all()\n    for u in users:\n        u['created'] = format_date(u['created'])\n    return users\n",
                "src/utils.py": "def format_date(dt):\n    return dt.strftime('%Y-%m-%d')\n\ndef parse_date(s):\n    from datetime import datetime\n    return datetime.strptime(s, '%Y-%m-%d')\n",
                "src/report.py": "from utils import format_date\n\ndef generate_report(data):\n    lines = []\n    for item in data:\n        lines.append(f'{format_date(item[\"date\"])}: {item[\"value\"]}')\n    return '\\n'.join(lines)\n",
                "src/main.py": "from api import get_users\nfrom report import generate_report\n\nusers = get_users()\nprint(generate_report(users))\n",
            },
            "task": "Find the definition of `format_date` and ALL files that call it.",
            "relevant": {
                "src/utils.py": [1],  # definition
                "src/api.py": [1, 5],  # import + call
                "src/report.py": [1, 5],  # import + call
            },
        },
        {
            "files": {
                "src/core.py": "def validate_email(email):\n    return '@' in email and '.' in email\n\ndef validate_phone(phone):\n    return phone.isdigit() and len(phone) == 10\n",
                "src/forms.py": "from core import validate_email, validate_phone\n\ndef process_form(data):\n    if not validate_email(data.get('email', '')):\n        return 'Invalid email'\n    if not validate_phone(data.get('phone', '')):\n        return 'Invalid phone'\n    return 'OK'\n",
                "src/api.py": "from core import validate_email\n\ndef register_user(email):\n    if not validate_email(email):\n        return False\n    return True\n",
            },
            "task": "Find the definition of `validate_email` and ALL files that call it.",
            "relevant": {
                "src/core.py": [1],  # definition
                "src/forms.py": [1, 3],  # import + call
                "src/api.py": [1, 3],  # import + call
            },
        },
    ]

    template = rng.choice(templates)
    difficulty = 0.4 + 0.2 * rng.random()

    return Problem(
        id=f"context_hunt_dep_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["files"], template["task"]),
        difficulty=difficulty,
        metadata={
            "type": "context_hunter",
            "files": template["files"],
            "relevant": template["relevant"],
            "task": template["task"],
        },
        token_budget=700,
        source="generated",
    )


def _make_feature_impact_problem(rng: random.Random) -> Problem:
    """
    Task: Given a feature change request, find all lines that need modification.
    """
    templates = [
        {
            "files": {
                "src/config.py": "MAX_RETRIES = 3\nTIMEOUT = 30\nDEBUG = False\n",
                "src/client.py": "import config\n\ndef make_request(url):\n    for i in range(config.MAX_RETRIES):\n        try:\n            return fetch(url, timeout=config.TIMEOUT)\n        except TimeoutError:\n            continue\n    return None\n",
                "src/logger.py": "def log_request(url, status):\n    if config.DEBUG:\n        print(f'{url} -> {status}')\n",
            },
            "task": "We need to make MAX_RETRIES configurable per-request instead of global. Find ALL lines that need to change.",
            "relevant": {
                "src/config.py": [1],  # MAX_RETRIES definition
                "src/client.py": [3, 4],  # using MAX_RETRIES
            },
        },
    ]

    template = rng.choice(templates)
    difficulty = 0.5 + 0.2 * rng.random()

    return Problem(
        id=f"context_hunt_feature_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["files"], template["task"]),
        difficulty=difficulty,
        metadata={
            "type": "context_hunter",
            "files": template["files"],
            "relevant": template["relevant"],
            "task": template["task"],
        },
        token_budget=700,
        source="generated",
    )


def _format_prompt(files: dict[str, str], task: str) -> str:
    """Format the problem prompt with the codebase and task."""
    codebase_str = ""
    for fname, content in files.items():
        lines = content.split("\n")
        numbered = "\n".join(f"  {i+1}: {line}" for i, line in enumerate(lines))
        codebase_str += f"\n--- {fname} ---\n{numbered}\n"

    return (
        f"Codebase:\n{codebase_str}\n\n"
        f"Task: {task}\n\n"
        f"Output the relevant code locations in this format:\n"
        f"FILE: <filename>\nLINES: <start>-<end>\n\n"
        f"Repeat for each relevant section. Be precise — include ONLY the "
        f"lines that are directly relevant to the task. Including irrelevant "
        f"lines reduces your score. Missing relevant lines also reduces your score."
    )


def context_hunter_generator(seed: int) -> Problem:
    """Master generator for ContextHunter problems."""
    rng = random.Random(seed)
    generators = [
        _make_bug_location_problem,
        _make_dependency_trace_problem,
        _make_feature_impact_problem,
    ]
    weights = [0.4, 0.4, 0.2]
    gen = rng.choices(generators, weights=weights, k=1)[0]
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ContextHunterVerifier(Verifier):
    """
    Verifies a ContextHunter response by computing precision, recall, and F1
    of the model's selected line ranges vs the ground truth.
    """

    def __init__(self, relevant: dict[str, list[int]]):
        super().__init__()
        self._relevant = relevant
        # Flatten to a set of (filename, line) tuples
        self._relevant_set = set()
        for fname, lines in relevant.items():
            for line in lines:
                self._relevant_set.add((fname, line))

    def verify(self, response: str) -> VerifierResult:
        # Parse the model's selections
        selected = self._parse_selections(response)

        if not selected:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No valid FILE/LINES selections found",
            )

        # Compute precision and recall
        selected_set = set(selected)
        true_positives = len(selected_set & self._relevant_set)
        precision = true_positives / len(selected_set) if selected_set else 0.0
        recall = true_positives / len(self._relevant_set) if self._relevant_set else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Correct only if F1 is high enough
        correct = f1 >= 0.8

        # Conciseness bonus: if the model selected fewer lines than total
        # relevant lines (found signal in noise), give a bonus
        total_relevant = len(self._relevant_set)
        total_selected = len(selected_set)
        if total_selected <= total_relevant and recall == 1.0:
            conciseness_bonus = 0.1
        else:
            conciseness_bonus = 0.0

        score = min(1.0, f1 + conciseness_bonus)

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "selected_count": total_selected,
                "relevant_count": total_relevant,
            },
            diagnostics=f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} (TP={true_positives}, selected={total_selected}, relevant={total_relevant})",
        )

    def _parse_selections(self, response: str) -> list[tuple[str, int]]:
        """Parse FILE/LINES selections from the response."""
        selections = []

        # Pattern: FILE: <name>\nLINES: <start>-<end>
        pattern = r"FILE:\s*(\S+)\s*\n\s*LINES:\s*(\d+)(?:-(\d+))?"
        matches = re.finditer(pattern, response, re.IGNORECASE)

        for match in matches:
            fname = match.group(1).strip()
            start = int(match.group(2))
            end = int(match.group(3)) if match.group(3) else start

            for line in range(start, end + 1):
                selections.append((fname, line))

        return selections


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ContextHunterEnv(BaseReasoningEnv):
    """
    ContextHunter environment: extract only relevant code from a codebase.

    The model receives a codebase + task and must output file:line ranges.
    Reward = F1 score of selections vs ground truth + conciseness bonus.
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
            problem_generator = context_hunter_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ContextHunterVerifier(relevant=problem.metadata["relevant"])

    def _check_format(self, response: str) -> float:
        """Check for FILE:/LINES: format."""
        if re.search(r"FILE:\s*\S+\s*\n\s*LINES:\s*\d+", response, re.IGNORECASE):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
