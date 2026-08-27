"""
DiffJudge: Review agent-produced diffs for necessity and correctness.

Environment concept:
  The model is given a code diff (like what an AI agent would produce)
  and must judge each change:
    1. Is the change NECESSARY? (or is it unnecessary refactoring?)
    2. Is the change CORRECT? (does it introduce bugs?)
    3. Is anything MISSING? (should there be a change that isn't there?)

  This trains the model to be a CRITIC of agent-produced code — the
  "code review" skill. This is crucial because research shows only 44%
  of agent code survives to commits, meaning 56% is unnecessary or wrong.

  A model that can judge diffs well can:
    - Self-correct before producing final output
    - Act as a verifier for other agents
    - Reduce the review burden on human developers

Why this environment is worth using for 10T-100T param models:
  At 100T parameters, the model will be used as an autonomous agent.
  Without self-criticism, it will produce 56% waste code. With
  DiffJudge-trained self-criticism, it can catch its own unnecessary
  changes before committing them. This is the difference between a
  $10M and $22M inference bill (the 56% waste fraction).

Problem types:
  - Bug fix diffs (some necessary, some over-engineered)
  - Refactoring diffs (some helpful, some unnecessary)
  - Feature diffs (some complete, some missing edge cases)
  - Mixed diffs (a combination of necessary and unnecessary changes)

Verification:
  Each problem has:
    - The original code
    - The diff (unified diff format)
    - Labels for each changed line: "necessary", "unnecessary", "wrong"
    - Any missing changes that should have been made
  The verifier checks:
    1. Did the model identify all unnecessary changes?
    2. Did the model identify all wrong changes?
    3. Did the model identify all missing changes?

Reward design:
  - Precision: fraction of model's flagged issues that are real issues
  - Recall: fraction of real issues the model flagged
  - F1 = 2 * P * R / (P + R)
  - Conciseness bonus for short, precise reviews
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


def _make_bug_fix_diff_problem(rng: random.Random) -> Problem:
    """Generate a diff that fixes a bug but may include unnecessary changes."""
    templates = [
        {
            "original": "def calculate_total(items):\n    total = 0\n    for item in items:\n        total += item['price']\n    return total\n",
            "diff": """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,4 +1,6 @@
 def calculate_total(items):
-    total = 0
+    # Initialize the total accumulator
+    total = 0
     for item in items:
-        total += item['price']
+        total += item.get('price', 0)
     return total
""",
            "labels": {
                "added_comment": "unnecessary",  # the comment is unnecessary
                "changed_to_get": "necessary",   # .get() prevents KeyError
            },
            "missing": [],
            "task": "Review this diff. Flag any unnecessary changes and identify if anything is missing.",
        },
        {
            "original": "def get_user_name(user):\n    return user['name']\n",
            "diff": """--- a/src/api.py
+++ b/src/api.py
@@ -1,2 +1,5 @@
 def get_user_name(user):
-    return user['name']
+    if user is None:
+        return 'Unknown'
+    return user.get('name', 'Unknown')
""",
            "labels": {
                "none_check": "necessary",       # None check is good
                "get_with_default": "necessary",  # .get() prevents KeyError
            },
            "missing": [],
            "task": "Review this diff. Flag any unnecessary changes and identify if anything is missing.",
        },
        {
            "original": "def process(data):\n    result = []\n    for d in data:\n        result.append(d * 2)\n    return result\n",
            "diff": """--- a/src/proc.py
+++ b/src/proc.py
@@ -1,4 +1,7 @@
-def process(data):
+def process_data(data):
     result = []
     for d in data:
-        result.append(d * 2)
+        # Double each element
+        result.append(d * 2)
+    # Return the processed result
     return result
""",
            "labels": {
                "renamed_function": "unnecessary",  # renaming is unnecessary
                "added_comment1": "unnecessary",     # comment is unnecessary
                "added_comment2": "unnecessary",     # comment is unnecessary
            },
            "missing": [],
            "task": "Review this diff. Flag any unnecessary changes and identify if anything is missing.",
        },
        {
            "original": "def validate_age(age):\n    if age < 0:\n        return False\n    return True\n",
            "diff": """--- a/src/validate.py
+++ b/src/validate.py
@@ -1,3 +1,3 @@
 def validate_age(age):
-    if age < 0:
+    if age <= 0:
         return False
     return True
""",
            "labels": {
                "changed_lt_to_le": "wrong",  # age=0 should be valid, not invalid
            },
            "missing": [],
            "task": "Review this diff. Flag any unnecessary changes and identify if anything is missing.",
        },
        {
            "original": "def send_email(to, subject, body):\n    # TODO: implement email sending\n    pass\n",
            "diff": """--- a/src/email.py
+++ b/src/email.py
@@ -1,3 +1,3 @@
 def send_email(to, subject, body):
-    # TODO: implement email sending
-    pass
+    pass
""",
            "labels": {
                "removed_todo_comment": "wrong",  # removing the TODO without implementing is wrong
            },
            "missing": ["Email sending logic should be implemented, not just the TODO removed."],
            "task": "Review this diff. Flag any unnecessary changes and identify if anything is missing.",
        },
    ]

    template = rng.choice(templates)
    difficulty = 0.3 + 0.2 * rng.random()

    # Count total issues
    total_issues = len(template["labels"]) + len(template["missing"])

    return Problem(
        id=f"diff_judge_bug_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["original"], template["diff"], template["task"]),
        difficulty=difficulty,
        metadata={
            "type": "diff_judge",
            "labels": template["labels"],
            "missing": template["missing"],
            "total_issues": total_issues,
        },
        token_budget=600,
        source="generated",
    )


def _make_refactor_diff_problem(rng: random.Random) -> Problem:
    """Generate a refactoring diff — some changes are good, some unnecessary."""
    templates = [
        {
            "original": "def get_value(d, key, default=None):\n    if key in d:\n        return d[key]\n    else:\n        return default\n",
            "diff": """--- a/src/utils.py
+++ b/src/utils.py
@@ -1,4 +1,1 @@
 def get_value(d, key, default=None):
-    if key in d:
-        return d[key]
-    else:
-        return default
+    return d.get(key, default)
""",
            "labels": {
                "simplified_to_get": "necessary",  # simplification is good
            },
            "missing": [],
            "task": "Review this refactoring diff. Is the change necessary and correct?",
        },
        {
            "original": "def add(a, b):\n    return a + b\n",
            "diff": """--- a/src/math.py
+++ b/src/math.py
@@ -1,2 +1,8 @@
 def add(a, b):
+    # Type checking for robustness
+    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
+        raise TypeError('Arguments must be numbers')
+    # Log the operation for debugging
+    print(f'Adding {a} + {b}')
     return a + b
""",
            "labels": {
                "type_checking": "unnecessary",  # over-engineering for a simple add
                "logging": "unnecessary",        # unnecessary print statement
            },
            "missing": [],
            "task": "Review this refactoring diff. Is the change necessary and correct?",
        },
    ]

    template = rng.choice(templates)
    difficulty = 0.4 + 0.2 * rng.random()

    total_issues = len(template["labels"]) + len(template["missing"])

    return Problem(
        id=f"diff_judge_refactor_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["original"], template["diff"], template["task"]),
        difficulty=difficulty,
        metadata={
            "type": "diff_judge",
            "labels": template["labels"],
            "missing": template["missing"],
            "total_issues": total_issues,
        },
        token_budget=600,
        source="generated",
    )


def _format_prompt(original: str, diff: str, task: str) -> str:
    """Format the problem prompt."""
    return (
        f"Task: {task}\n\n"
        f"Original code:\n```python\n{original}\n```\n\n"
        f"Diff:\n```\n{diff}\n```\n\n"
        f"Output your review in this format:\n"
        f"ISSUES:\n"
        f"- <issue description>\n"
        f"- <issue description>\n\n"
        f"MISSING:\n"
        f"- <missing change description> (or 'NONE' if nothing missing)\n\n"
        f"Rules:\n"
        f"  - Flag unnecessary changes (comments, renaming, over-engineering)\n"
        f"  - Flag wrong changes (logic errors, incorrect fixes)\n"
        f"  - Flag missing changes (things that should have been changed but weren't)\n"
        f"  - Be concise — one line per issue"
    )


def diff_judge_generator(seed: int) -> Problem:
    """Master generator for DiffJudge problems."""
    rng = random.Random(seed)
    generators = [_make_bug_fix_diff_problem, _make_refactor_diff_problem]
    weights = [0.6, 0.4]
    gen = rng.choices(generators, weights=weights, k=1)[0]
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class DiffJudgeVerifier(Verifier):
    """
    Verifies a DiffJudge response by checking if the model identified
    the right issues (unnecessary changes, wrong changes, missing changes).
    """

    # Keywords that map to issue types
    ISSUE_KEYWORDS = {
        "unnecessary": ["unnecessary", "not needed", "redundant", "over-engineer", "no need", "superfluous"],
        "wrong": ["wrong", "incorrect", "bug", "error", "mistake", "should not", "breaks"],
        "missing": ["missing", "should have", "not included", "absent", "omitted", "forgot"],
    }

    def __init__(self, labels: dict[str, str], missing: list[str], total_issues: int):
        super().__init__()
        self._labels = labels
        self._missing = missing
        self._total_issues = total_issues

    def verify(self, response: str) -> VerifierResult:
        # Parse the model's issues
        flagged_issues = self._parse_issues(response)
        flagged_missing = self._parse_missing(response)

        # Count real issues by type
        real_unnecessary = sum(1 for v in self._labels.values() if v == "unnecessary")
        real_wrong = sum(1 for v in self._labels.values() if v == "wrong")
        real_missing = len(self._missing)

        total_real = real_unnecessary + real_wrong + real_missing

        # Count what the model flagged
        model_unnecessary = flagged_issues.count("unnecessary")
        model_wrong = flagged_issues.count("wrong")
        model_missing = len(flagged_missing)

        total_flagged = len(flagged_issues) + len(flagged_missing)

        # Compute precision and recall
        if total_flagged == 0:
            precision = 0.0
        else:
            # Approximate: if the model flagged issues and there are real issues,
            # estimate true positives as min(flagged, real) per category
            tp = min(model_unnecessary, real_unnecessary) + min(model_wrong, real_wrong) + min(model_missing, real_missing)
            precision = tp / total_flagged

        if total_real == 0:
            recall = 1.0 if total_flagged == 0 else 0.5  # no issues, model should say NONE
        else:
            tp = min(model_unnecessary, real_unnecessary) + min(model_wrong, real_wrong) + min(model_missing, real_missing)
            recall = tp / total_real

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Special case: if there are no issues and the model says "NONE"
        if total_real == 0 and total_flagged == 0:
            f1 = 1.0
            precision = 1.0
            recall = 1.0

        correct = f1 >= 0.6

        return VerifierResult(
            correct=correct,
            score=f1,
            partial_credit={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "real_issues": total_real,
                "flagged_issues": total_flagged,
            },
            diagnostics=f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} (real={total_real}, flagged={total_flagged})",
        )

    def _parse_issues(self, response: str) -> list[str]:
        """Parse the ISSUES section and classify each issue."""
        issues_match = re.search(r"ISSUES:\s*\n(.*?)(?:\n\s*MISSING:|$)", response, re.DOTALL | re.IGNORECASE)
        if not issues_match:
            return []

        issues_text = issues_match.group(1)
        lines = re.findall(r"-\s*(.+?)(?:\n|$)", issues_text)

        classified = []
        for line in lines:
            line_lower = line.lower()
            if line_lower.strip() == "none" or line_lower.strip() == "n/a":
                continue
            # Classify by keywords
            for issue_type, keywords in self.ISSUE_KEYWORDS.items():
                if any(kw in line_lower for kw in keywords):
                    classified.append(issue_type)
                    break
            else:
                # Default to "unnecessary" if unclassifiable
                classified.append("unnecessary")

        return classified

    def _parse_missing(self, response: str) -> list[str]:
        """Parse the MISSING section."""
        missing_match = re.search(r"MISSING:\s*\n(.*?)(?:\n\s*\n|$)", response, re.DOTALL | re.IGNORECASE)
        if not missing_match:
            return []

        missing_text = missing_match.group(1)
        lines = re.findall(r"-\s*(.+?)(?:\n|$)", missing_text)

        result = []
        for line in lines:
            if line.lower().strip() in ("none", "n/a", "nothing"):
                continue
            result.append(line.strip())

        return result


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DiffJudgeEnv(BaseReasoningEnv):
    """
    DiffJudge environment: review agent-produced diffs.

    The model receives a diff and must flag unnecessary/wrong changes and
    identify missing changes. Reward = F1 of issue detection.
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
            problem_generator = diff_judge_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return DiffJudgeVerifier(
            labels=problem.metadata["labels"],
            missing=problem.metadata["missing"],
            total_issues=problem.metadata["total_issues"],
        )

    def _check_format(self, response: str) -> float:
        """Check for ISSUES: and MISSING: sections."""
        has_issues = bool(re.search(r"ISSUES:\s*\n", response, re.IGNORECASE))
        has_missing = bool(re.search(r"MISSING:\s*\n", response, re.IGNORECASE))
        if has_issues and has_missing:
            return 1.0
        elif has_issues:
            return 0.5
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
