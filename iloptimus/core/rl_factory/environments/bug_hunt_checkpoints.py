"""
BugHuntCheckpoints: Find+fix bugs checkpoint by checkpoint (real execution).

Environment concept:
  The model receives a codebase with MULTIPLE bugs. Each bug is a checkpoint.
  The model must:
    1. For each checkpoint (bug): identify the bug, fix it, audit by running tests
    2. Fix bugs in order (each fix must not break previous fixes)
    3. The audit is REAL — we run the test suite to verify

  This is a batch environment: N parallel bug hunters compete.
  The reward measures:
    - Bugs fixed: how many bugs were correctly fixed
    - No regressions: did fixes not break other tests
    - Audit honesty: did the model's self-audit match reality
    - Minimality: were fixes small and targeted

  reward = bugs_fixed * 0.4 + no_regressions * 0.3 + audit_honesty * 0.2 + minimality * 0.1

  This trains the model to:
    1. Find bugs systematically (not randomly)
    2. Fix bugs surgically (not rewrite everything)
    3. Verify fixes with tests (the ground truth)
    4. Avoid regressions (each fix must be isolated)

  The key insight: bug hunting is the #1 time sink in real software engineering.
  A model that can find and fix bugs checkpoint by checkpoint, with real test
  verification, is a production-ready debugging agent.
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
# Problem templates: code with multiple bugs
# ---------------------------------------------------------------------------


_BUG_HUNT_PROBLEMS = [
    {
        "task": "Fix the bugs in this calculator module.",
        "buggy_code": (
            "def add(a, b):\n"
            "    return a - b  # BUG: should be +\n\n"
            "def multiply(a, b):\n"
            "    result = 0\n"
            "    for i in range(b):\n"
            "        result = result + a\n"
            "    return result  # BUG: doesn't handle negative b\n\n"
            "def divide(a, b):\n"
            "    return a / b  # BUG: no zero division check\n\n"
            "def power(base, exp):\n"
            "    result = 0  # BUG: should be 1\n"
            "    for i in range(exp):\n"
            "        result = result * base\n"
            "    return result"
        ),
        "bugs": [
            {
                "name": "Fix add() — wrong operator",
                "description": "add() uses subtraction instead of addition",
                "test": "assert add(2, 3) == 5; assert add(-1, 1) == 0; assert add(0, 0) == 0",
                "fix_hint": "Change a - b to a + b",
            },
            {
                "name": "Fix multiply() — negative numbers",
                "description": "multiply() doesn't handle negative second argument",
                "test": "assert multiply(3, 4) == 12; assert multiply(3, -1) == -3; assert multiply(0, 5) == 0",
                "fix_hint": "Handle negative b by negating the result",
            },
            {
                "name": "Fix divide() — zero division",
                "description": "divide() crashes on division by zero",
                "test": "assert divide(10, 2) == 5; assert divide(7, 0) == None; assert divide(0, 5) == 0",
                "fix_hint": "Add a check for b == 0 and return None",
            },
            {
                "name": "Fix power() — wrong initial value",
                "description": "power() initializes result to 0 instead of 1",
                "test": "assert power(2, 3) == 8; assert power(5, 0) == 1; assert power(3, 2) == 9",
                "fix_hint": "Change result = 0 to result = 1",
            },
        ],
    },
    {
        "task": "Fix the bugs in this string utilities module.",
        "buggy_code": (
            "def reverse(s):\n"
            "    result = ''\n"
            "    for c in s:\n"
            "        result = result + c  # BUG: should prepend\n"
            "    return result\n\n"
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for c in s:\n"
            "        if c in 'aeiou':  # BUG: missing uppercase\n"
            "            count += 1\n"
            "    return count\n\n"
            "def is_palindrome(s):\n"
            "    return s == reverse(s)  # BUG: case sensitive\n\n"
            "def capitalize_words(s):\n"
            "    words = s.split()\n"
            "    result = ''\n"
            "    for w in words:\n"
            "        result = result + w.capitalize()  # BUG: missing space\n"
            "    return result"
        ),
        "bugs": [
            {
                "name": "Fix reverse() — wrong concatenation order",
                "description": "reverse() appends instead of prepending",
                "test": "assert reverse('hello') == 'olleh'; assert reverse('') == ''; assert reverse('a') == 'a'",
                "fix_hint": "Change result + c to c + result",
            },
            {
                "name": "Fix count_vowels() — missing uppercase",
                "description": "count_vowels() doesn't count uppercase vowels",
                "test": "assert count_vowels('Hello') == 2; assert count_vowels('AEIOU') == 5; assert count_vowels('xyz') == 0",
                "fix_hint": "Lowercase the string before checking",
            },
            {
                "name": "Fix is_palindrome() — case sensitivity",
                "description": "is_palindrome() is case-sensitive",
                "test": "assert is_palindrome('RaceCar') == True; assert is_palindrome('hello') == False; assert is_palindrome('A') == True",
                "fix_hint": "Lowercase s before comparing",
            },
            {
                "name": "Fix capitalize_words() — missing spaces",
                "description": "capitalize_words() doesn't add spaces between words",
                "test": "assert capitalize_words('hello world') == 'Hello World'; assert capitalize_words('one') == 'One'; assert capitalize_words('') == ''",
                "fix_hint": "Add spaces between capitalized words",
            },
        ],
    },
    {
        "task": "Fix the bugs in this list utilities module.",
        "buggy_code": (
            "def find_max(lst):\n"
            "    max_val = 0  # BUG: should be lst[0]\n"
            "    for item in lst:\n"
            "        if item > max_val:\n"
            "            max_val = item\n"
            "    return max_val\n\n"
            "def find_min(lst):\n"
            "    min_val = lst[0]\n"
            "    for item in lst:\n"
            "        if item < min_val:\n"
            "            min_val = item\n"
            "    return min_val  # BUG: doesn't handle empty list\n\n"
            "def remove_duplicates(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "    return result  # BUG: preserves order but inefficient — not a bug per se\n\n"
            "def merge(a, b):\n"
            "    result = a + b  # BUG: modifies original lists if they're mutated\n"
            "    return result"
        ),
        "bugs": [
            {
                "name": "Fix find_max() — wrong initial value",
                "description": "find_max() initializes to 0, fails for all-negative lists",
                "test": "assert find_max([1,3,2]) == 3; assert find_max([-1,-2,-3]) == -1; assert find_max([5]) == 5",
                "fix_hint": "Initialize max_val to lst[0]",
            },
            {
                "name": "Fix find_min() — empty list crash",
                "description": "find_min() crashes on empty list",
                "test": "assert find_min([3,1,2]) == 1; assert find_min([]) == None; assert find_min([5]) == 5",
                "fix_hint": "Check if list is empty and return None",
            },
            {
                "name": "Fix merge() — should return new list",
                "description": "merge() should return a new list, not reference inputs",
                "test": "a = [1,2]; b = [3,4]; m = merge(a, b); assert m == [1,2,3,4]; a.append(99); assert m == [1,2,3,4]",
                "fix_hint": "Use list(a) + list(b) to create copies",
            },
        ],
    },
]


def bug_hunt_checkpoints_generator(seed: int) -> Problem:
    """Generate a BugHuntCheckpoints problem."""
    rng = random.Random(seed)
    template = rng.choice(_BUG_HUNT_PROBLEMS)

    bugs = template["bugs"]
    checkpoints = []
    for i, bug in enumerate(bugs):
        checkpoints.append({
            "name": bug["name"],
            "description": bug["description"],
            "test_code": bug["test"],
            "fix_hint": bug["fix_hint"],
            "bug_index": i,
            "success_criteria": f"Test must pass after fix: {bug['name']}",
        })

    cp_text = "\n".join(
        f"CHECKPOINT {i+1}: {cp['name']}\n"
        f"  Bug: {cp['description']}\n"
        f"  Test: {cp['test_code']}\n"
        f"  Hint: {cp['fix_hint']}"
        for i, cp in enumerate(checkpoints)
    )

    return Problem(
        id=f"bug_hunt_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Buggy code:\n```python\n{template['buggy_code']}\n```\n\n"
            f"Bugs to fix (in order):\n{cp_text}\n\n"
            f"Format:\n"
            f"CHECKPOINT <n>: <bug name>\n"
            f"IDENTIFY: <what's wrong>\n"
            f"FIX: ```python\n<all code with this fix applied>\n```\n"
            f"AUDIT: PASS or FAIL - <did the test pass?>\n"
            f"---\n\n"
            f"Rules:\n"
            f"  - Fix bugs in order, one at a time\n"
            f"  - Each checkpoint includes ALL previous fixes\n"
            f"  - AUDIT honestly — we run the tests\n"
            f"  - Don't break previous fixes\n"
            f"  - Be surgical — fix only the specific bug"
        ),
        difficulty=0.35 + 0.1 * rng.random(),
        metadata={
            "type": "bug_hunt_checkpoints",
            "task": template["task"],
            "buggy_code": template["buggy_code"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
        },
        token_budget=300 + len(checkpoints) * 300,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — runs tests for each bug fix
# ---------------------------------------------------------------------------


class BugHuntVerifier(Verifier):
    """Verifies a bug hunt response by running tests for each fix."""

    def __init__(self, checkpoints: list[dict]):
        super().__init__()
        self._checkpoints = checkpoints

    def verify(self, response: str) -> VerifierResult:
        executed = self._parse_checkpoints(response)

        if not executed:
            return VerifierResult(correct=False, score=0.0, diagnostics="No checkpoints found")

        total = len(self._checkpoints)
        bugs_fixed = 0
        audit_honest = 0
        audit_total = 0
        regressions = 0

        # Track which bug tests have passed
        passed_so_far = set()

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            code = exe_cp.get("code", "")
            model_audit = exe_cp.get("audit_result", "unknown")

            # Run the test for this bug
            actual_pass = self._run_test(code, exp_cp["test_code"])

            audit_total += 1
            if (actual_pass and "pass" in model_audit.lower()) or \
               (not actual_pass and "fail" in model_audit.lower()):
                audit_honest += 1

            if actual_pass:
                bugs_fixed += 1
                # Check for regressions: do previous tests still pass?
                for prev_idx in passed_so_far:
                    if not self._run_test(code, self._checkpoints[prev_idx]["test_code"]):
                        regressions += 1
                passed_so_far.add(i)
            else:
                # Check if previous fixes still work
                for prev_idx in passed_so_far:
                    if not self._run_test(code, self._checkpoints[prev_idx]["test_code"]):
                        regressions += 1

        bug_score = bugs_fixed / total if total > 0 else 0.0
        audit_score = audit_honest / audit_total if audit_total > 0 else 0.0
        no_regression = 1.0 - min(1.0, regressions / max(total, 1))

        # Minimality: check if the model identified the bug (not just rewrote)
        # We check if the IDENTIFY section is non-empty
        identify_count = sum(1 for e in executed if e.get("identify", "").strip())

        minimality = identify_count / len(executed) if executed else 0.0

        score = bug_score * 0.4 + no_regression * 0.3 + audit_score * 0.2 + minimality * 0.1
        correct = bug_score >= 0.6 and no_regression >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "bugs_fixed": bugs_fixed,
                "total_bugs": total,
                "bug_score": bug_score,
                "audit_honesty": audit_score,
                "no_regression": no_regression,
                "regressions": regressions,
                "minimality": minimality,
            },
            diagnostics=f"Bugs: {bugs_fixed}/{total} Regressions: {regressions} Audit: {audit_score:.0%} Minimality: {minimality:.0%}",
        )

    def _run_test(self, code: str, test_code: str) -> bool:
        code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
            exec(test_code, namespace)
            return True
        except Exception:
            return False

    def _parse_checkpoints(self, response: str) -> list[dict]:
        executed = []
        blocks = re.split(r"(?:^|\n)CHECKPOINT\s+\d+\s*:", response, flags=re.IGNORECASE)

        for block in blocks[1:]:
            lines = block.strip().split("\n")
            name = lines[0].strip() if lines else ""
            code = ""
            identify = ""
            audit_result = "unknown"
            in_code = False
            code_lines = []

            for line in lines[1:]:
                if "```python" in line or (line.strip() == "```" and in_code):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)
                elif line.strip().upper().startswith("IDENTIFY:"):
                    identify = line.split(":", 1)[1].strip()
                elif line.strip().upper().startswith("AUDIT:"):
                    audit = line.split(":", 1)[1].strip()
                    if "pass" in audit.lower():
                        audit_result = "pass"
                    elif "fail" in audit.lower():
                        audit_result = "fail"

            code = "\n".join(code_lines) if code_lines else ""
            if name:
                executed.append({
                    "name": name,
                    "code": code,
                    "identify": identify,
                    "audit_result": audit_result,
                })

        return executed


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BugHuntCheckpointsEnv(BatchEnvBase):
    """BugHuntCheckpoints: find+fix bugs checkpoint by checkpoint with real test audits."""

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = bug_hunt_checkpoints_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return BugHuntVerifier(checkpoints=problem.metadata["checkpoints"])

    def _check_format(self, response: str) -> float:
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE) and "AUDIT:" in response.upper():
            return 1.0
        if re.search(r"CHECKPOINT\s+\d+", response, re.IGNORECASE):
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
