"""
CodeReviewCheckpoints: Review a PR by dimension (style/correctness/security/perf).

Environment concept:
  The model receives a code change (a "pull request") and must review it
  checkpoint by checkpoint, where each checkpoint is a REVIEW DIMENSION:
    1. Style checkpoint: check naming, formatting, readability
    2. Correctness checkpoint: check logic, edge cases, test coverage
    3. Security checkpoint: check for injection, auth, data exposure
    4. Performance checkpoint: check for O(n²), unnecessary allocations, etc.

  For each checkpoint, the model must:
    1. Review the code through that dimension's lens
    2. Identify issues (or confirm there are none)
    3. AUDIT: report PASS (no issues) or FAIL (issues found) with specifics

  This is a batch environment: N parallel reviews are scored.
  The reward measures:
    - Issue detection: did the model find the real issues in each dimension?
    - Audit honesty: did PASS/FAIL match reality?
    - Explanation quality: for FAILs, were the explanations specific?
    - Coverage: did the model review all dimensions?

  reward = issue_detection * 0.4 + audit_honesty * 0.3 + explanation_quality * 0.2 + coverage * 0.1

  This trains the model to:
    1. Review code SYSTEMATICALLY (dimension by dimension, not randomly)
    2. Be HONEST (not rubber-stamp with "looks good")
    3. Be SPECIFIC (not vague "might have issues")
    4. Cover ALL dimensions (not just correctness)

  The key insight: code review is where senior engineers add the most value.
  A model that can review code across dimensions, find real issues, and
  provide specific feedback is a force multiplier for any engineering team.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: code with known issues per dimension
# ---------------------------------------------------------------------------


_REVIEW_PROBLEMS = [
    {
        "task": "Review this user authentication function.",
        "code": (
            "def authenticate(username, password):\n"
            "    # check the database\n"
            "    query = \"SELECT * FROM users WHERE name='\" + username + \"' AND pass='\" + password + \"'\"\n"
            "    result = db.execute(query)\n"
            "    if result:\n"
            "        return True\n"
            "    return False"
        ),
        "dimensions": [
            {
                "name": "Style",
                "description": "Check naming, formatting, readability",
                "expected": "FAIL",
                "issues": ["Comment is vague ('check the database')", "Variable 'result' is not descriptive"],
                "audit_keywords": ["comment", "naming", "variable", "descriptive", "readable"],
            },
            {
                "name": "Correctness",
                "description": "Check logic, edge cases",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
            {
                "name": "Security",
                "description": "Check for injection, auth, data exposure",
                "expected": "FAIL",
                "issues": ["SQL injection vulnerability — string concatenation in query", "Passwords in plain text", "No password hashing"],
                "audit_keywords": ["sql", "injection", "password", "plaintext", "hash", "concatenation"],
            },
            {
                "name": "Performance",
                "description": "Check for unnecessary work, O(n²), etc.",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
        ],
    },
    {
        "task": "Review this data processing function.",
        "code": (
            "def process_data(data_list):\n"
            "    results = []\n"
            "    for item in data_list:\n"
            "        for other in data_list:\n"
            "            if item == other:\n"
            "                results.append(item)\n"
            "    return results"
        ),
        "dimensions": [
            {
                "name": "Style",
                "description": "Check naming, formatting, readability",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
            {
                "name": "Correctness",
                "description": "Check logic, edge cases",
                "expected": "FAIL",
                "issues": ["Duplicates every item (compares item to itself)", "Returns duplicates of matching items"],
                "audit_keywords": ["duplicate", "self", "compare", "itself", "logic"],
            },
            {
                "name": "Security",
                "description": "Check for injection, auth, data exposure",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
            {
                "name": "Performance",
                "description": "Check for unnecessary work, O(n²), etc.",
                "expected": "FAIL",
                "issues": ["O(n²) nested loop — could be O(n) with a set", "Unnecessary inner loop"],
                "audit_keywords": ["o(n", "nested", "loop", "quadratic", "set", "performance"],
            },
        ],
    },
    {
        "task": "Review this file reading function.",
        "code": (
            "def read_file(filename):\n"
            "    f = open(filename)\n"
            "    data = f.read()\n"
            "    return data"
        ),
        "dimensions": [
            {
                "name": "Style",
                "description": "Check naming, formatting, readability",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
            {
                "name": "Correctness",
                "description": "Check logic, edge cases",
                "expected": "FAIL",
                "issues": ["File is never closed — resource leak", "No error handling for missing file"],
                "audit_keywords": ["close", "leak", "resource", "error", "missing", "file"],
            },
            {
                "name": "Security",
                "description": "Check for injection, auth, data exposure",
                "expected": "FAIL",
                "issues": ["Path traversal — no validation of filename", "Could read arbitrary files"],
                "audit_keywords": ["path", "traversal", "filename", "validation", "arbitrary"],
            },
            {
                "name": "Performance",
                "description": "Check for unnecessary work, O(n²), etc.",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
        ],
    },
    {
        "task": "Review this sorting function.",
        "code": (
            "def sort_list(lst):\n"
            "    sorted_list = lst\n"
            "    for i in range(len(sorted_list)):\n"
            "        for j in range(len(sorted_list) - 1):\n"
            "            if sorted_list[j] > sorted_list[j+1]:\n"
            "                temp = sorted_list[j]\n"
            "                sorted_list[j] = sorted_list[j+1]\n"
            "                sorted_list[j+1] = temp\n"
            "    return sorted_list"
        ),
        "dimensions": [
            {
                "name": "Style",
                "description": "Check naming, formatting, readability",
                "expected": "FAIL",
                "issues": ["Uses temp variable instead of Python tuple swap", "Variable names are not descriptive enough"],
                "audit_keywords": ["temp", "swap", "tuple", "naming", "pythonic"],
            },
            {
                "name": "Correctness",
                "description": "Check logic, edge cases",
                "expected": "FAIL",
                "issues": ["Modifies the input list in place (sorted_list = lst is not a copy)"],
                "audit_keywords": ["modify", "input", "in-place", "copy", "mutation"],
            },
            {
                "name": "Security",
                "description": "Check for injection, auth, data exposure",
                "expected": "PASS",
                "issues": [],
                "audit_keywords": [],
            },
            {
                "name": "Performance",
                "description": "Check for unnecessary work, O(n²), etc.",
                "expected": "FAIL",
                "issues": ["O(n²) bubble sort — should use built-in sorted()", "Unnecessary nested loop"],
                "audit_keywords": ["o(n", "bubble", "sorted", "builtin", "performance"],
            },
        ],
    },
]


def code_review_checkpoints_generator(seed: int) -> Problem:
    """Generate a CodeReviewCheckpoints problem."""
    rng = random.Random(seed)
    template = rng.choice(_REVIEW_PROBLEMS)

    dimensions = template["dimensions"]
    checkpoints = []
    for i, dim in enumerate(dimensions):
        checkpoints.append({
            "name": f"Review: {dim['name']}",
            "dimension": dim["name"],
            "description": dim["description"],
            "expected": dim["expected"],
            "issues": dim["issues"],
            "audit_keywords": dim["audit_keywords"],
            "success_criteria": f"Correctly identify {dim['expected']} for {dim['name']}",
        })

    cp_text = "\n".join(
        f"CHECKPOINT {i+1}: {cp['name']}\n  Focus: {cp['description']}"
        for i, cp in enumerate(checkpoints)
    )

    return Problem(
        id=f"code_review_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Code to review:\n```python\n{template['code']}\n```\n\n"
            f"Review dimensions:\n{cp_text}\n\n"
            f"Format:\n"
            f"REVIEW <n>: <dimension name>\n"
            f"VERDICT: PASS or FAIL\n"
            f"DETAILS: <specific issues found, or 'no issues'>\n"
            f"---\n\n"
            f"Rules:\n"
            f"  - Review ALL {len(checkpoints)} dimensions\n"
            f"  - Be honest — don't rubber-stamp\n"
            f"  - For FAIL, list specific issues\n"
            f"  - For PASS, confirm what's correct\n"
            f"  - Be specific, not vague"
        ),
        difficulty=0.3 + 0.1 * rng.random(),
        metadata={
            "type": "code_review_checkpoints",
            "task": template["task"],
            "code": template["code"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
        },
        token_budget=400 + len(checkpoints) * 150,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CodeReviewVerifier(Verifier):
    """Verifies a code review response against expected dimensions."""

    def __init__(self, checkpoints: list[dict]):
        super().__init__()
        self._checkpoints = checkpoints

    def verify(self, response: str) -> VerifierResult:
        parsed = self._parse_reviews(response)

        if not parsed:
            return VerifierResult(correct=False, score=0.0, diagnostics="No review blocks found")

        total = len(self._checkpoints)
        correct_verdicts = 0
        explanation_quality_scores = []
        coverage = 0

        for i, exp_cp in enumerate(self._checkpoints):
            dim_name = exp_cp["dimension"]
            expected = exp_cp["expected"]
            keywords = exp_cp.get("audit_keywords", [])

            # Find the model's review for this dimension
            model_review = parsed.get(dim_name.lower(), parsed.get(str(i + 1), {}))
            model_verdict = model_review.get("verdict", "unknown").lower()
            model_details = model_review.get("details", "")

            if model_verdict == expected.lower():
                correct_verdicts += 1
                coverage += 1

            # Explanation quality: for FAILs, check if keywords are mentioned
            if expected == "FAIL" and model_verdict == "fail":
                kw_matches = sum(1 for kw in keywords if kw.lower() in model_details.lower())
                explanation_quality_scores.append(kw_matches / max(len(keywords), 1))
            elif expected == "PASS" and model_verdict == "pass":
                explanation_quality_scores.append(1.0)
            else:
                explanation_quality_scores.append(0.0)

        issue_detection = correct_verdicts / total if total > 0 else 0.0
        coverage_score = coverage / total if total > 0 else 0.0
        explanation_quality = sum(explanation_quality_scores) / len(explanation_quality_scores) if explanation_quality_scores else 0.0

        # Audit honesty: avoid false positives (saying FAIL when should be PASS)
        false_positives = 0
        expected_passes = 0
        for i, exp_cp in enumerate(self._checkpoints):
            if exp_cp["expected"] == "PASS":
                expected_passes += 1
                dim_name = exp_cp["dimension"]
                model_review = parsed.get(dim_name.lower(), parsed.get(str(i + 1), {}))
                if model_review.get("verdict", "").lower() == "fail":
                    false_positives += 1

        audit_honesty = 1.0 - (false_positives / max(expected_passes, 1)) if expected_passes > 0 else 1.0

        score = issue_detection * 0.4 + audit_honesty * 0.3 + explanation_quality * 0.2 + coverage_score * 0.1
        correct = issue_detection >= 0.6 and coverage_score >= 0.75

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "issue_detection": issue_detection,
                "audit_honesty": audit_honesty,
                "explanation_quality": explanation_quality,
                "coverage": coverage_score,
                "correct_verdicts": correct_verdicts,
                "total": total,
                "false_positives": false_positives,
            },
            diagnostics=f"Detection: {correct_verdicts}/{total} Honesty: {audit_honesty:.0%} Explanation: {explanation_quality:.0%} Coverage: {coverage_score:.0%}",
        )

    def _parse_reviews(self, response: str) -> dict[str, dict]:
        """Parse REVIEW n: dimension / VERDICT: PASS|FAIL / DETAILS: ..."""
        results = {}
        # Try REVIEW <n>: <dimension> format
        pattern = r"REVIEW\s+(\d+)\s*:\s*(.+?)\n\s*VERDICT:\s*(PASS|FAIL)\s*[-:]?\s*(.*?)(?=\n\s*REVIEW|\Z)"
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)

        for match in matches:
            num = match.group(1)
            dim_name = match.group(2).strip()
            verdict = match.group(3).lower()
            details = match.group(4).strip()

            # Also extract DETAILS: line if present
            details_match = re.search(r"DETAILS:\s*(.+?)(?:\n\s*REVIEW|\Z)", details, re.IGNORECASE | re.DOTALL)
            if details_match:
                details = details_match.group(1).strip()

            results[num] = {"dimension": dim_name, "verdict": verdict, "details": details}
            results[dim_name.lower()] = {"dimension": dim_name, "verdict": verdict, "details": details}

        return results


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CodeReviewCheckpointsEnv(BatchEnvBase):
    """CodeReviewCheckpoints: review code by dimension with honest verdicts."""

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = code_review_checkpoints_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CodeReviewVerifier(checkpoints=problem.metadata["checkpoints"])

    def _check_format(self, response: str) -> float:
        if re.search(r"REVIEW\s+\d+", response, re.IGNORECASE) and "VERDICT:" in response.upper():
            return 1.0
        if re.search(r"REVIEW\s+\d+", response, re.IGNORECASE):
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
            "reward": best_score if best_score > 0 else 0.0,
            "correct": any_correct,
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
