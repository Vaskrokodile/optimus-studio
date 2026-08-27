"""
RefactorCheckpointSequence: Refactor code through checkpoints, tests must pass.

Environment concept:
  The model receives working but messy code. It must refactor it through a
  sequence of checkpoints, where each checkpoint is a specific refactoring
  step (extract function, rename, simplify, remove duplication, etc.).
  After each step, the AUDIT runs the test suite to verify behavior is
  preserved.

  Checkpoints (refactoring steps):
    1. Extract a helper function from inline code
    2. Rename variables for clarity
    3. Simplify a complex conditional
    4. Remove code duplication
    5. Final: all tests still pass + code is shorter

  The key constraint: tests must pass after EVERY checkpoint. If a
  refactoring step breaks tests, the model must fix it before proceeding.

  This is a batch environment: N parallel refactoring sequences are scored.
  The reward measures:
    - Tests preserved: did all tests pass after each step?
    - Refactoring quality: did the model actually improve the code?
    - Code reduction: is the final code shorter than the original?
    - Audit honesty: did the model's self-audit match reality?

  reward = tests_preserved * 0.4 + refactor_quality * 0.3 + code_reduction * 0.2 + audit_honesty * 0.1

  This trains the model to:
    1. Refactor SAFELY (tests are the safety net)
    2. Make small, verifiable changes (not big-bang rewrites)
    3. Improve code quality while preserving behavior
    4. Self-audit by running tests after each change

  The key insight: safe refactoring is the most valuable senior engineer
  skill. A model that can refactor without breaking things is worth 100x
  more than one that can only write new code. This environment trains
  that skill directly.
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
# Problem templates: messy code + refactoring checkpoints + tests
# ---------------------------------------------------------------------------


_REFACTOR_PROBLEMS = [
    {
        "task": "Refactor a messy temperature converter.",
        "original_code": (
            "def convert_temp(val, unit_from, unit_to):\n"
            "    if unit_from == 'C' and unit_to == 'F':\n"
            "        result = val * 9 / 5 + 32\n"
            "        return result\n"
            "    elif unit_from == 'F' and unit_to == 'C':\n"
            "        result = (val - 32) * 5 / 9\n"
            "        return result\n"
            "    elif unit_from == 'C' and unit_to == 'K':\n"
            "        result = val + 273.15\n"
            "        return result\n"
            "    elif unit_from == 'K' and unit_to == 'C':\n"
            "        result = val - 273.15\n"
            "        return result\n"
            "    else:\n"
            "        return val"
        ),
        "tests": [
            "assert convert_temp(100, 'C', 'F') == 212",
            "assert convert_temp(32, 'F', 'C') == 0",
            "assert convert_temp(0, 'C', 'K') == 273.15",
            "assert convert_temp(273.15, 'K', 'C') == 0",
            "assert convert_temp(50, 'C', 'C') == 50",
        ],
        "refactor_steps": [
            {"name": "Extract c_to_f helper", "description": "Extract the C→F conversion into a helper function"},
            {"name": "Extract f_to_c helper", "description": "Extract the F→C conversion into a helper function"},
            {"name": "Simplify return statements", "description": "Remove redundant 'result = ...; return result' patterns"},
            {"name": "Use a conversion dict", "description": "Replace if-elif chain with a dictionary of conversion functions"},
        ],
        "golden_refactored": (
            "def c_to_f(c): return c * 9/5 + 32\n"
            "def f_to_c(f): return (f - 32) * 5/9\n"
            "def convert_temp(val, unit_from, unit_to):\n"
            "    conversions = {\n"
            "        ('C','F'): c_to_f, ('F','C'): f_to_c,\n"
            "        ('C','K'): lambda c: c + 273.15,\n"
            "        ('K','C'): lambda k: k - 273.15,\n"
            "    }\n"
            "    return conversions.get((unit_from, unit_to), lambda x: x)(val)"
        ),
    },
    {
        "task": "Refactor a messy string processor.",
        "original_code": (
            "def process_string(s, mode):\n"
            "    if mode == 'upper':\n"
            "        result = ''\n"
            "        for c in s:\n"
            "            result = result + c.upper()\n"
            "        return result\n"
            "    elif mode == 'lower':\n"
            "        result = ''\n"
            "        for c in s:\n"
            "            result = result + c.lower()\n"
            "        return result\n"
            "    elif mode == 'reverse':\n"
            "        result = ''\n"
            "        for c in s:\n"
            "            result = c + result\n"
            "        return result\n"
            "    else:\n"
            "        return s"
        ),
        "tests": [
            "assert process_string('hello', 'upper') == 'HELLO'",
            "assert process_string('WORLD', 'lower') == 'world'",
            "assert process_string('abc', 'reverse') == 'cba'",
            "assert process_string('test', 'unknown') == 'test'",
        ],
        "refactor_steps": [
            {"name": "Replace loop with built-in", "description": "Replace manual upper/lower loops with .upper()/.lower()"},
            {"name": "Simplify reverse", "description": "Replace manual reverse loop with slicing s[::-1]"},
            {"name": "Use dispatch dict", "description": "Replace if-elif chain with a dictionary of operations"},
        ],
        "golden_refactored": (
            "def process_string(s, mode):\n"
            "    ops = {'upper': str.upper, 'lower': str.lower, 'reverse': lambda x: x[::-1]}\n"
            "    return ops.get(mode, lambda x: x)(s)"
        ),
    },
    {
        "task": "Refactor a messy list utility.",
        "original_code": (
            "def list_op(lst, op):\n"
            "    if op == 'sum':\n"
            "        total = 0\n"
            "        for item in lst:\n"
            "            total = total + item\n"
            "        return total\n"
            "    elif op == 'product':\n"
            "        total = 1\n"
            "        for item in lst:\n"
            "            total = total * item\n"
            "        return total\n"
            "    elif op == 'max':\n"
            "        current_max = lst[0]\n"
            "        for item in lst:\n"
            "            if item > current_max:\n"
            "                current_max = item\n"
            "        return current_max\n"
            "    elif op == 'min':\n"
            "        current_min = lst[0]\n"
            "        for item in lst:\n"
            "            if item < current_min:\n"
            "                current_min = item\n"
            "        return current_min\n"
            "    else:\n"
            "        return None"
        ),
        "tests": [
            "assert list_op([1,2,3], 'sum') == 6",
            "assert list_op([2,3,4], 'product') == 24",
            "assert list_op([3,1,2], 'max') == 3",
            "assert list_op([3,1,2], 'min') == 1",
            "assert list_op([1], 'unknown') == None",
        ],
        "refactor_steps": [
            {"name": "Use built-in sum", "description": "Replace manual sum loop with sum()"},
            {"name": "Use built-in max/min", "description": "Replace manual max/min loops with max()/min()"},
            {"name": "Use dispatch dict", "description": "Replace if-elif chain with a dictionary of built-in operations"},
        ],
        "golden_refactored": (
            "def list_op(lst, op):\n"
            "    ops = {'sum': sum, 'max': max, 'min': min,\n"
            "           'product': lambda l: __import__('functools').reduce(lambda a,b: a*b, l, 1)}\n"
            "    return ops.get(op, lambda l: None)(lst)"
        ),
    },
    {
        "task": "Refactor a messy grade calculator.",
        "original_code": (
            "def get_grade(score):\n"
            "    if score >= 90:\n"
            "        grade = 'A'\n"
            "        return grade\n"
            "    elif score >= 80:\n"
            "        grade = 'B'\n"
            "        return grade\n"
            "    elif score >= 70:\n"
            "        grade = 'C'\n"
            "        return grade\n"
            "    elif score >= 60:\n"
            "        grade = 'D'\n"
            "        return grade\n"
            "    else:\n"
            "        grade = 'F'\n"
            "        return grade"
        ),
        "tests": [
            "assert get_grade(95) == 'A'",
            "assert get_grade(85) == 'B'",
            "assert get_grade(75) == 'C'",
            "assert get_grade(65) == 'D'",
            "assert get_grade(50) == 'F'",
        ],
        "refactor_steps": [
            {"name": "Remove redundant variable", "description": "Remove 'grade = ...; return grade' — return directly"},
            {"name": "Use bisect or dict lookup", "description": "Replace if-elif chain with a data-driven approach"},
        ],
        "golden_refactored": (
            "def get_grade(score):\n"
            "    for threshold, grade in [(90,'A'),(80,'B'),(70,'C'),(60,'D')]:\n"
            "        if score >= threshold: return grade\n"
            "    return 'F'"
        ),
    },
]


def refactor_checkpoint_sequence_generator(seed: int) -> Problem:
    """Generate a RefactorCheckpointSequence problem."""
    rng = random.Random(seed)
    template = rng.choice(_REFACTOR_PROBLEMS)

    steps = template["refactor_steps"]
    checkpoints = []
    for i, step in enumerate(steps):
        checkpoints.append({
            "name": step["name"],
            "description": step["description"],
            "step_index": i,
            "success_criteria": f"Tests must still pass after: {step['description']}",
        })

    # Final checkpoint: all tests pass + code is shorter
    checkpoints.append({
        "name": "Final verification",
        "description": "All tests pass and code is shorter than original",
        "step_index": len(steps),
        "success_criteria": "All tests pass + code reduction",
    })

    cp_text = "\n".join(
        f"CHECKPOINT {i+1}: {cp['name']}\n  {cp['description']}"
        for i, cp in enumerate(checkpoints)
    )

    original_tokens = estimate_tokens(template["original_code"])

    return Problem(
        id=f"refactor_seq_{rng.randint(0, 99999)}",
        prompt=(
            f"Task: {template['task']}\n\n"
            f"Original code:\n```python\n{template['original_code']}\n```\n\n"
            f"Tests (must pass after EVERY step):\n" +
            "\n".join(f"  {t}" for t in template["tests"]) + "\n\n"
            f"Refactor through these checkpoints:\n{cp_text}\n\n"
            f"Format:\n"
            f"CHECKPOINT <n>: <step name>\n"
            f"CODE: ```python\n<refactored code so far>\n```\n"
            f"AUDIT: PASS or FAIL - <did tests pass?>\n"
            f"---\n\n"
            f"Rules:\n"
            f"  - Tests MUST pass after every step\n"
            f"  - Each step should improve the code\n"
            f"  - Final code should be shorter than original\n"
            f"  - AUDIT honestly — we run the tests"
        ),
        difficulty=0.35 + 0.1 * rng.random(),
        metadata={
            "type": "refactor_checkpoint_sequence",
            "task": template["task"],
            "original_code": template["original_code"],
            "tests": template["tests"],
            "checkpoints": checkpoints,
            "num_checkpoints": len(checkpoints),
            "golden_refactored": template.get("golden_refactored", ""),
            "original_tokens": original_tokens,
        },
        token_budget=300 + len(checkpoints) * 300,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — runs tests after each refactoring step
# ---------------------------------------------------------------------------


class RefactorCheckpointVerifier(Verifier):
    """Verifies a refactoring sequence by running tests after each step."""

    def __init__(self, checkpoints: list[dict], tests: list[str], original_code: str, original_tokens: int):
        super().__init__()
        self._checkpoints = checkpoints
        self._tests = tests
        self._original_code = original_code
        self._original_tokens = original_tokens

    def verify(self, response: str) -> VerifierResult:
        executed = self._parse_checkpoints(response)

        if not executed:
            return VerifierResult(correct=False, score=0.0, diagnostics="No checkpoints found")

        total = len(self._checkpoints)
        steps_with_passing_tests = 0
        audit_honest = 0
        audit_total = 0

        for i, exp_cp in enumerate(self._checkpoints):
            if i >= len(executed):
                break

            exe_cp = executed[i]
            code = exe_cp.get("code", "")
            model_audit = exe_cp.get("audit_result", "unknown")

            # Run ALL tests against this code
            actual_pass = self._run_all_tests(code)

            audit_total += 1
            if (actual_pass and "pass" in model_audit.lower()) or \
               (not actual_pass and "fail" in model_audit.lower()):
                audit_honest += 1

            if actual_pass:
                steps_with_passing_tests += 1

        # Check final code reduction
        final_code = executed[-1].get("code", "") if executed else ""
        final_tokens = estimate_tokens(final_code)
        code_reduction = max(0.0, 1.0 - (final_tokens / max(self._original_tokens, 1)))

        # Check refactor quality: does the final code differ from original?
        # (it should be different — a refactor happened)
        refactor_happened = final_code.strip() != self._original_code.strip()

        test_preservation = steps_with_passing_tests / total if total > 0 else 0.0
        audit_score = audit_honest / audit_total if audit_total > 0 else 0.0

        # Refactor quality: tests pass + code is shorter + code changed
        refactor_quality = 0.0
        if steps_with_passing_tests == total and refactor_happened:
            refactor_quality = 0.5 + 0.5 * code_reduction

        score = test_preservation * 0.4 + refactor_quality * 0.3 + code_reduction * 0.2 + audit_score * 0.1
        correct = test_preservation >= 0.6 and refactor_happened

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "steps_with_tests": steps_with_passing_tests,
                "total_steps": total,
                "test_preservation": test_preservation,
                "audit_honesty": audit_score,
                "code_reduction": code_reduction,
                "refactor_happened": refactor_happened,
                "original_tokens": self._original_tokens,
                "final_tokens": final_tokens,
            },
            diagnostics=f"Tests preserved: {steps_with_passing_tests}/{total} Reduction: {code_reduction:.0%} Refactored: {refactor_happened} Audit: {audit_score:.0%}",
        )

    def _run_all_tests(self, code: str) -> bool:
        code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace)
            for test in self._tests:
                exec(test, namespace)
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
            audit_result = "unknown"
            in_code = False
            code_lines = []

            for line in lines[1:]:
                if "```python" in line or (line.strip() == "```" and in_code):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)
                elif line.strip().upper().startswith("AUDIT:"):
                    audit = line.split(":", 1)[1].strip()
                    if "pass" in audit.lower():
                        audit_result = "pass"
                    elif "fail" in audit.lower():
                        audit_result = "fail"

            code = "\n".join(code_lines) if code_lines else ""
            if name:
                executed.append({"name": name, "code": code, "audit_result": audit_result})

        return executed


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RefactorCheckpointSequenceEnv(BatchEnvBase):
    """RefactorCheckpointSequence: refactor through checkpoints with real test audits."""

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = refactor_checkpoint_sequence_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return RefactorCheckpointVerifier(
            checkpoints=problem.metadata["checkpoints"],
            tests=problem.metadata["tests"],
            original_code=problem.metadata["original_code"],
            original_tokens=problem.metadata["original_tokens"],
        )

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
