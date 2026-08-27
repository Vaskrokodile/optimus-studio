"""
FailureFirstEnv: Recover from deliberately injected failures.

Environment concept:
  The model receives a task with a DELIBERATELY INJECTED FAILURE — a false
  premise, a broken utility, a missing dependency, a misleading hint, or
  code that fails on edge cases. The model must:

    1. DETECT the failure (what's wrong)
    2. RECOVER from it (the fix)
    3. Produce FIXED CODE that passes the tests
    4. AUDIT honestly (did all tests pass?)

  This trains RESILIENCE — the ability to recognize when something is
  wrong in the problem itself, not just in the solution, and to recover
  without making things worse.

  R = R_success + R_recovery - R_catastrophic

  R_success:     did the fixed code pass all tests? (real execution)
  R_recovery:    detection_score * 0.4 + fix_correctness * 0.4 + audit_honesty * 0.2
  R_catastrophic: 1.0 if fixed code is worse than original,
                  0.5 if gave up,
                  0.0 otherwise

  Final: R = R_success * 0.5 + R_recovery * 0.3 + (1 - R_catastrophic) * 0.2

Five failure types:
  - wrong_assumption:  task description contains a false premise
  - broken_tool:       a provided utility function has a bug
  - missing_dependency: the solution requires a function that doesn't exist
  - misleading_info:   the problem description contains incorrect hints
  - partial_failure:   provided code works for most cases but fails on edge cases

This is a batch environment: N parallel recovery agents compete, and the
aggregate reward uses best-of-N selection.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: tasks with injected failures
# ---------------------------------------------------------------------------

_FAILURE_FIRST_PROBLEMS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # wrong_assumption — the task description contains a false premise
    # ------------------------------------------------------------------
    {
        "failure_type": "wrong_assumption",
        "task": (
            "Sort the following list in ascending order and return the result. "
            "The list is [5, 4, 3, 2, 1]. The expected output is [1, 2, 3, 4, 5]."
        ),
        "provided_code": (
            "def sort_list(lst):\n"
            "    return sorted(lst)\n"
        ),
        "tests": (
            "assert sort_list([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]\n"
            "assert sort_list([3, 1, 2]) == [1, 2, 3]\n"
            "assert sort_list([]) == []\n"
            "assert sort_list([1]) == [1]\n"
        ),
        "false_premise": (
            "The description says the list is already sorted descending, "
            "implying sorting is trivial. In reality the provided code works "
            "correctly — the false premise is that there is a problem with "
            "the 'expected output'. The expected output IS correct. The "
            "false premise is the claim that the list is 'already sorted "
            "descending' which is true, but the description implies this "
            "makes sorting unnecessary. The model must recognize the code "
            "is actually correct and state that no fix is needed."
        ),
        "detection_hint": "already sorted descending",
        "fix_description": "No code fix needed — the provided code is correct.",
    },
    {
        "failure_type": "wrong_assumption",
        "task": (
            "Find the maximum value in the list [3, 1, 4, 1, 5, 9, 2, 6]. "
            "The expected answer is 6 because 6 is the largest even number."
        ),
        "provided_code": (
            "def find_max(lst):\n"
            "    return max(lst)\n"
        ),
        "tests": (
            "assert find_max([3, 1, 4, 1, 5, 9, 2, 6]) == 9\n"
            "assert find_max([1, 2, 3]) == 3\n"
            "assert find_max([-1, -2, -3]) == -1\n"
            "assert find_max([42]) == 42\n"
        ),
        "false_premise": (
            "The description claims the answer is 6 because '6 is the largest "
            "even number.' This is a false premise — the task says 'find the "
            "maximum value', not 'find the maximum even value.' The actual "
            "maximum is 9. The model must detect that the expected answer "
            "in the description is wrong and that the provided code (max()) "
            "is correct."
        ),
        "detection_hint": "largest even number",
        "fix_description": "No code fix needed — the provided code is correct; the description's expected answer is wrong.",
    },
    {
        "failure_type": "wrong_assumption",
        "task": (
            "Count the number of vowels in 'HELLO WORLD'. The expected "
            "answer is 2 because uppercase letters are not vowels."
        ),
        "provided_code": (
            "def count_vowels(s):\n"
            "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
        ),
        "tests": (
            "assert count_vowels('HELLO WORLD') == 3\n"
            "assert count_vowels('aeiou') == 5\n"
            "assert count_vowels('XYZ') == 0\n"
            "assert count_vowels('') == 0\n"
        ),
        "false_premise": (
            "The description claims uppercase letters are not vowels and "
            "the expected answer is 2. This is a false premise — 'E', 'O', "
            "'O' are all vowels regardless of case. The correct answer is 3. "
            "The provided code correctly lowercases before counting, so it "
            "is already correct. The model must detect the false premise "
            "and confirm the code is right."
        ),
        "detection_hint": "uppercase letters are not vowels",
        "fix_description": "No code fix needed — the provided code correctly handles case; the description's claim is false.",
    },

    # ------------------------------------------------------------------
    # broken_tool — a provided utility function has a bug
    # ------------------------------------------------------------------
    {
        "failure_type": "broken_tool",
        "task": (
            "Use the provided max_of_list() function to find the maximum "
            "value in [3, 7, 2, 9, 4] and return it."
        ),
        "provided_code": (
            "def max_of_list(lst):\n"
            "    return lst[0]\n"
        ),
        "tests": (
            "assert max_of_list([3, 7, 2, 9, 4]) == 9\n"
            "assert max_of_list([1, 2, 3]) == 3\n"
            "assert max_of_list([-1, -2, -3]) == -1\n"
            "assert max_of_list([5]) == 5\n"
            "assert max_of_list([0, 0, 0]) == 0\n"
        ),
        "false_premise": (
            "The provided max_of_list() function simply returns lst[0] — "
            "it does not actually find the maximum. The model must identify "
            "this bug, fix it, and use the fixed version."
        ),
        "detection_hint": "returns lst[0]",
        "fix_description": "Fix max_of_list to iterate and find the true maximum.",
    },
    {
        "failure_type": "broken_tool",
        "task": (
            "Use the provided is_even() function to filter even numbers "
            "from [1, 2, 3, 4, 5, 6] and return them as a list."
        ),
        "provided_code": (
            "def is_even(n):\n"
            "    return n % 2 == 1\n"
        ),
        "tests": (
            "assert is_even(2) == True\n"
            "assert is_even(3) == False\n"
            "assert is_even(0) == True\n"
            "assert is_even(-4) == True\n"
            "assert is_even(-3) == False\n"
        ),
        "false_premise": (
            "The provided is_even() function returns n % 2 == 1, which "
            "actually checks for ODD numbers, not even. The model must "
            "detect this, fix it to n % 2 == 0, and use the corrected version."
        ),
        "detection_hint": "n % 2 == 1",
        "fix_description": "Fix is_even to check n % 2 == 0 instead of n % 2 == 1.",
    },
    {
        "failure_type": "broken_tool",
        "task": (
            "Use the provided reverse_string() function to reverse "
            "'hello' and return the result."
        ),
        "provided_code": (
            "def reverse_string(s):\n"
            "    result = ''\n"
            "    for c in s:\n"
            "        result = result + c\n"
            "    return result\n"
        ),
        "tests": (
            "assert reverse_string('hello') == 'olleh'\n"
            "assert reverse_string('abc') == 'cba'\n"
            "assert reverse_string('') == ''\n"
            "assert reverse_string('a') == 'a'\n"
            "assert reverse_string('ab') == 'ba'\n"
        ),
        "false_premise": (
            "The provided reverse_string() appends each character instead "
            "of prepending, so it returns the original string unchanged. "
            "The model must detect this and fix the concatenation order."
        ),
        "detection_hint": "result + c",
        "fix_description": "Fix reverse_string to prepend: result = c + result.",
    },

    # ------------------------------------------------------------------
    # missing_dependency — the solution requires a function that doesn't exist
    # ------------------------------------------------------------------
    {
        "failure_type": "missing_dependency",
        "task": (
            "Use the flatten() function to flatten the nested list "
            "[1, [2, 3], [4, [5, 6]]] into a single flat list."
        ),
        "provided_code": (
            "def process_data(nested):\n"
            "    return flatten(nested)\n"
        ),
        "tests": (
            "assert process_data([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\n"
            "assert process_data([1, 2, 3]) == [1, 2, 3]\n"
            "assert process_data([]) == []\n"
            "assert process_data([[1], [2], [3]]) == [1, 2, 3]\n"
            "assert process_data([[[1]]]) == [1]\n"
        ),
        "false_premise": (
            "The task instructs the model to use flatten(), but flatten() "
            "is not defined anywhere. The model must implement flatten() "
            "first, then ensure process_data works correctly."
        ),
        "detection_hint": "flatten is not defined",
        "fix_description": "Implement the missing flatten() function that recursively flattens nested lists.",
    },
    {
        "failure_type": "missing_dependency",
        "task": (
            "Use the word_count() function to count words in the sentence "
            "'the quick brown fox' and return the result as a dictionary."
        ),
        "provided_code": (
            "def analyze_sentence(sentence):\n"
            "    return word_count(sentence)\n"
        ),
        "tests": (
            "assert analyze_sentence('the quick brown fox') == {'the': 1, 'quick': 1, 'brown': 1, 'fox': 1}\n"
            "assert analyze_sentence('hello hello world') == {'hello': 2, 'world': 1}\n"
            "assert analyze_sentence('') == {}\n"
            "assert analyze_sentence('one') == {'one': 1}\n"
        ),
        "false_premise": (
            "The task instructs the model to use word_count(), but "
            "word_count() is not defined. The model must implement "
            "word_count() to split the sentence and count occurrences."
        ),
        "detection_hint": "word_count is not defined",
        "fix_description": "Implement the missing word_count() function that splits and counts word frequencies.",
    },
    {
        "failure_type": "missing_dependency",
        "task": (
            "Use the is_prime() function to filter prime numbers from "
            "[2, 3, 4, 5, 6, 7, 8, 9, 10] and return them as a list."
        ),
        "provided_code": (
            "def filter_primes(numbers):\n"
            "    return [n for n in numbers if is_prime(n)]\n"
        ),
        "tests": (
            "assert filter_primes([2, 3, 4, 5, 6, 7, 8, 9, 10]) == [2, 3, 5, 7]\n"
            "assert filter_primes([11, 13, 17]) == [11, 13, 17]\n"
            "assert filter_primes([1, 4, 6, 8]) == []\n"
            "assert filter_primes([2]) == [2]\n"
            "assert filter_primes([]) == []\n"
        ),
        "false_premise": (
            "The task instructs the model to use is_prime(), but is_prime() "
            "is not defined. The model must implement is_prime() correctly "
            "(handling edge cases like n < 2) and then ensure filter_primes "
            "works."
        ),
        "detection_hint": "is_prime is not defined",
        "fix_description": "Implement the missing is_prime() function that correctly checks primality (n < 2 is not prime).",
    },

    # ------------------------------------------------------------------
    # misleading_info — the problem description contains incorrect hints
    # ------------------------------------------------------------------
    {
        "failure_type": "misleading_info",
        "task": (
            "Compute the factorial of 5. Hint: use recursion — a recursive "
            "factorial is the most efficient approach for this problem."
        ),
        "provided_code": (
            "def factorial(n):\n"
            "    pass\n"
        ),
        "tests": (
            "assert factorial(5) == 120\n"
            "assert factorial(0) == 1\n"
            "assert factorial(1) == 1\n"
            "assert factorial(3) == 6\n"
            "assert factorial(10) == 3628800\n"
        ),
        "false_premise": (
            "The hint claims recursion is 'the most efficient approach.' "
            "This is misleading — iteration is more efficient (no call stack "
            "overhead) and avoids stack overflow for large n. The model must "
            "recognize the hint is wrong and use iteration instead. The "
            "provided code is a stub (pass) that must be implemented."
        ),
        "detection_hint": "most efficient",
        "fix_description": "Ignore the misleading recursion hint; implement factorial iteratively for efficiency and safety.",
    },
    {
        "failure_type": "misleading_info",
        "task": (
            "Check if a string is a palindrome. Hint: convert the string "
            "to a list first, then compare element by element — this is "
            "the fastest method."
        ),
        "provided_code": (
            "def is_palindrome(s):\n"
            "    pass\n"
        ),
        "tests": (
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('a') == True\n"
            "assert is_palindrome('abba') == True\n"
            "assert is_palindrome('AbA') == True\n"
        ),
        "false_premise": (
            "The hint claims converting to a list and comparing element by "
            "element is 'the fastest method.' This is misleading — direct "
            "string comparison (s == s[::-1]) is simpler and faster. Also, "
            "the tests require case-insensitive comparison (AbA is a "
            "palindrome), which the hint doesn't mention. The model must "
            "recognize the hint is suboptimal and implement a correct, "
            "case-insensitive solution."
        ),
        "detection_hint": "fastest method",
        "fix_description": "Ignore the misleading hint; use s.lower() == s.lower()[::-1] for a correct, case-insensitive palindrome check.",
    },
    {
        "failure_type": "misleading_info",
        "task": (
            "Find the sum of all even numbers in [1, 2, 3, 4, 5, 6, 7, 8]. "
            "Hint: use a while loop with a manual counter — this avoids "
            "off-by-one errors that for loops are prone to."
        ),
        "provided_code": (
            "def sum_evens(numbers):\n"
            "    pass\n"
        ),
        "tests": (
            "assert sum_evens([1, 2, 3, 4, 5, 6, 7, 8]) == 20\n"
            "assert sum_evens([2, 4, 6]) == 12\n"
            "assert sum_evens([1, 3, 5]) == 0\n"
            "assert sum_evens([]) == 0\n"
            "assert sum_evens([0]) == 0\n"
        ),
        "false_premise": (
            "The hint claims while loops avoid off-by-one errors that for "
            "loops are 'prone to.' This is misleading — for loops are "
            "actually safer (no manual counter management) and more Pythonic. "
            "The model must recognize the hint is wrong and use a for loop "
            "or a comprehension."
        ),
        "detection_hint": "prone to",
        "fix_description": "Ignore the misleading hint; use a for loop or comprehension (sum(n for n in numbers if n % 2 == 0)).",
    },

    # ------------------------------------------------------------------
    # partial_failure — provided code works for most cases but fails on edge cases
    # ------------------------------------------------------------------
    {
        "failure_type": "partial_failure",
        "task": (
            "The provided divide() function should safely divide two numbers. "
            "Fix any edge cases where it fails."
        ),
        "provided_code": (
            "def divide(a, b):\n"
            "    return a / b\n"
        ),
        "tests": (
            "assert divide(10, 2) == 5\n"
            "assert divide(7, 0) == None\n"
            "assert divide(0, 5) == 0\n"
            "assert divide(-6, 3) == -2\n"
            "assert divide(1, 1) == 1\n"
        ),
        "false_premise": (
            "The provided divide() works for normal cases but crashes with "
            "ZeroDivisionError when b=0. The model must identify this edge "
            "case, fix it (return None for division by zero), and verify "
            "all tests pass."
        ),
        "detection_hint": "b=0",
        "fix_description": "Add a guard: return None when b == 0 to handle division by zero.",
    },
    {
        "failure_type": "partial_failure",
        "task": (
            "The provided get_last() function should return the last element "
            "of a list. Fix any edge cases where it fails."
        ),
        "provided_code": (
            "def get_last(lst):\n"
            "    return lst[-1]\n"
        ),
        "tests": (
            "assert get_last([1, 2, 3]) == 3\n"
            "assert get_last(['a', 'b']) == 'b'\n"
            "assert get_last([42]) == 42\n"
            "assert get_last([]) == None\n"
            "assert get_last([0]) == 0\n"
        ),
        "false_premise": (
            "The provided get_last() works for non-empty lists but crashes "
            "with IndexError on an empty list. The model must identify this "
            "edge case, fix it (return None for empty lists), and verify."
        ),
        "detection_hint": "empty list",
        "fix_description": "Add a guard: return None when the list is empty before accessing lst[-1].",
    },
    {
        "failure_type": "partial_failure",
        "task": (
            "The provided average() function should compute the average "
            "of a list of numbers. Fix any edge cases where it fails."
        ),
        "provided_code": (
            "def average(numbers):\n"
            "    return sum(numbers) / len(numbers)\n"
        ),
        "tests": (
            "assert average([1, 2, 3]) == 2.0\n"
            "assert average([5]) == 5.0\n"
            "assert average([0, 0, 0]) == 0.0\n"
            "assert average([]) == 0\n"
            "assert average([10, 20, 30, 40]) == 25.0\n"
        ),
        "false_premise": (
            "The provided average() works for non-empty lists but crashes "
            "with ZeroDivisionError on an empty list. The model must "
            "identify this edge case, fix it (return 0 for empty lists), "
            "and verify all tests pass."
        ),
        "detection_hint": "empty list",
        "fix_description": "Add a guard: return 0 when the list is empty to avoid ZeroDivisionError.",
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def failure_first_generator(seed: int) -> Problem:
    """
    Generate a FailureFirst problem.

    Picks a random failure template and constructs a Problem with the
    injected failure, provided code, and tests that must pass after recovery.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A Problem with metadata describing the failure, tests, and expected
        detection/fix.
    """
    rng = random.Random(seed)
    template = rng.choice(_FAILURE_FIRST_PROBLEMS)

    failure_type = template["failure_type"]
    task = template["task"]
    provided_code = template["provided_code"]
    tests = template["tests"]
    detection_hint = template["detection_hint"]
    fix_description = template["fix_description"]

    difficulty_map = {
        "wrong_assumption": 0.35,
        "broken_tool": 0.30,
        "missing_dependency": 0.40,
        "misleading_info": 0.35,
        "partial_failure": 0.30,
    }
    base_difficulty = difficulty_map.get(failure_type, 0.30)
    difficulty = base_difficulty + 0.1 * rng.random()

    prompt = (
        f"FAILURE TYPE: {failure_type}\n"
        f"TASK: {task}\n"
        f"PROVIDED CODE: ```python\n{provided_code}\n```\n"
        f"TESTS: {tests}\n\n"
        f"DETECT: <what's wrong>\n"
        f"RECOVER: <your fix>\n"
        f"FIXED CODE: ```python\n<fixed code>\n```\n"
        f"AUDIT: PASS or FAIL - <did all tests pass?>\n\n"
        f"Rules:\n"
        f"  - The provided code or task description may contain a deliberate failure.\n"
        f"  - You must DETECT the failure, RECOVER from it, and produce FIXED CODE.\n"
        f"  - The FIXED CODE must pass ALL tests (we run them for real).\n"
        f"  - AUDIT honestly — we verify by executing the tests.\n"
        f"  - If the provided code is already correct, state that in DETECT and "
        f"provide it unchanged in FIXED CODE.\n"
        f"  - Do not make things worse — your fix must not break any test."
    )

    return Problem(
        id=f"failure_first_{failure_type}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": "failure_first",
            "failure_type": failure_type,
            "task": task,
            "provided_code": provided_code,
            "tests": tests,
            "detection_hint": detection_hint,
            "fix_description": fix_description,
        },
        token_budget=1200,
        source="generated",
    )


# ---------------------------------------------------------------------------
# Verifier — runs real code to check recovery
# ---------------------------------------------------------------------------


class FailureFirstVerifier(Verifier):
    """
    Verifies a FailureFirst response by:

      1. Parsing DETECT, RECOVER, FIXED CODE, and AUDIT sections.
      2. Running the tests on the FIXED CODE (REAL EXECUTION).
      3. Scoring:
         R_success:     did the fixed code pass all tests?
         R_recovery:    detection_score * 0.4 + fix_correctness * 0.4 + audit_honesty * 0.2
         R_catastrophic: 1.0 if fixed code is worse than original,
                         0.5 if gave up,
                         0.0 otherwise
      4. Final: R = R_success * 0.5 + R_recovery * 0.3 + (1 - R_catastrophic) * 0.2
    """

    def __init__(
        self,
        provided_code: str,
        tests: str,
        detection_hint: str,
        fix_description: str,
        failure_type: str,
    ):
        super().__init__()
        self._provided_code = provided_code
        self._tests = tests
        self._detection_hint = detection_hint
        self._fix_description = fix_description
        self._failure_type = failure_type

    def verify(self, response: str) -> VerifierResult:
        # ---------------------------------------------------------------
        # 1. Parse the response sections
        # ---------------------------------------------------------------
        detect_text = self._extract_section(response, "DETECT")
        recover_text = self._extract_section(response, "RECOVER")
        fixed_code = self._extract_code_block(response, "FIXED CODE")
        audit_text = self._extract_section(response, "AUDIT")

        # If no FIXED CODE block was found, try any ```python block
        if not fixed_code:
            fixed_code = self._extract_any_code_block(response)

        # ---------------------------------------------------------------
        # 2. Run the tests on the FIXED CODE (REAL EXECUTION)
        # ---------------------------------------------------------------
        r_success = 0.0
        tests_passed = 0
        tests_total = 0
        test_details: list[str] = []

        if fixed_code:
            r_success, tests_passed, tests_total, test_details = self._run_tests(
                fixed_code, self._tests
            )
        else:
            test_details.append("No FIXED CODE block found in response.")

        # ---------------------------------------------------------------
        # 3. Run the tests on the ORIGINAL provided code (for comparison)
        # ---------------------------------------------------------------
        original_success, orig_passed, orig_total, _ = self._run_tests(
            self._provided_code, self._tests
        )

        # ---------------------------------------------------------------
        # 4. Score R_recovery components
        # ---------------------------------------------------------------
        detection_score = self._score_detection(detect_text)
        fix_correctness = self._score_fix_correctness(
            fixed_code, r_success, detect_text
        )
        audit_honesty = self._score_audit_honesty(audit_text, r_success)

        r_recovery = (
            detection_score * 0.4
            + fix_correctness * 0.4
            + audit_honesty * 0.2
        )

        # ---------------------------------------------------------------
        # 5. Score R_catastrophic
        # ---------------------------------------------------------------
        r_catastrophic = self._score_catastrophic(
            fixed_code, r_success, original_success, detect_text, recover_text
        )

        # ---------------------------------------------------------------
        # 6. Final reward
        # ---------------------------------------------------------------
        final_score = (
            r_success * 0.5
            + r_recovery * 0.3
            + (1.0 - r_catastrophic) * 0.2
        )
        final_score = max(0.0, min(1.0, final_score))

        correct = r_success >= 1.0 and r_catastrophic == 0.0

        diagnostics_parts: list[str] = []
        diagnostics_parts.append(
            f"R_success={r_success:.2f} ({tests_passed}/{tests_total} tests)"
        )
        diagnostics_parts.append(
            f"R_recovery={r_recovery:.2f} "
            f"(detect={detection_score:.2f}, fix={fix_correctness:.2f}, "
            f"audit={audit_honesty:.2f})"
        )
        diagnostics_parts.append(f"R_catastrophic={r_catastrophic:.2f}")
        diagnostics_parts.append(f"Final={final_score:.2f}")
        if test_details:
            diagnostics_parts.append("; ".join(test_details[:3]))

        return VerifierResult(
            correct=correct,
            score=final_score,
            partial_credit={
                "r_success": r_success,
                "r_recovery": r_recovery,
                "r_catastrophic": r_catastrophic,
                "detection_score": detection_score,
                "fix_correctness": fix_correctness,
                "audit_honesty": audit_honesty,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "original_tests_passed": orig_passed,
                "original_tests_total": orig_total,
            },
            diagnostics=" | ".join(diagnostics_parts),
            metadata={
                "failure_type": self._failure_type,
                "detect_text": detect_text,
                "recover_text": recover_text,
                "audit_text": audit_text,
                "has_fixed_code": bool(fixed_code),
            },
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_section(self, response: str, section_name: str) -> str:
        """
        Extract the text following a section header like 'DETECT:'.

        Captures everything from the header until the next known section
        header or end of response. Code blocks within a section are
        excluded (they belong to FIXED CODE).
        """
        # Known section headers that terminate a section
        known_headers = [
            "DETECT:", "RECOVER:", "FIXED CODE:", "AUDIT:",
            "DETECT", "RECOVER", "FIXED CODE", "AUDIT",
        ]

        # Build a regex to find the section header (case-insensitive)
        pattern = re.compile(
            rf"(?im)^\s*{re.escape(section_name)}\s*:\s*(.*)$"
        )
        match = pattern.search(response)
        if not match:
            return ""

        # Start capturing from the matched line
        start_pos = match.start()
        rest = response[start_pos:]

        # Get the content on the same line as the header
        first_line_content = match.group(1).strip()

        # Collect subsequent lines until we hit another known header
        lines = rest.split("\n")
        content_lines: list[str] = []
        if first_line_content:
            content_lines.append(first_line_content)

        for line in lines[1:]:
            stripped = line.strip()
            # Check if this line is a known section header
            is_header = False
            for header in known_headers:
                if stripped.upper().startswith(header.upper()):
                    is_header = True
                    break
            if is_header:
                break
            content_lines.append(line)

        return "\n".join(content_lines).strip()

    def _extract_code_block(self, response: str, section_name: str) -> str:
        """
        Extract the ```python code block that follows a section header.

        Looks for the section header, then finds the next fenced code block.
        """
        pattern = re.compile(
            rf"(?im)^\s*{re.escape(section_name)}\s*:?\s*$"
        )
        match = pattern.search(response)
        if match:
            rest = response[match.end():]
            code = self._find_first_code_block(rest)
            if code:
                return code

        # Fallback: look for FIXED CODE: followed by code on the same flow
        # Try a more lenient pattern: FIXED CODE: ```python ... ```
        lenient = re.compile(
            rf"(?is){re.escape(section_name)}\s*:?\s*```(?:python)?\n?(.*?)```",
        )
        match = lenient.search(response)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_any_code_block(self, response: str) -> str:
        """Extract the last ```python code block in the response."""
        blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        blocks = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        return ""

    def _find_first_code_block(self, text: str) -> str:
        """Find the first fenced code block in text."""
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _run_tests(
        self, code: str, tests: str
    ) -> tuple[float, int, int, list[str]]:
        """
        Execute the code and run the tests.

        Returns:
            (pass_rate, num_passed, num_total, detail_messages)
        """
        namespace: dict[str, Any] = {}
        details: list[str] = []

        # Execute the fixed code
        try:
            exec(code, namespace)
        except Exception as e:
            details.append(
                f"Code execution error: {type(e).__name__}: {e}"
            )
            return 0.0, 0, 0, details

        # Parse and run individual test assertions
        test_lines = [
            line.strip()
            for line in tests.strip().split("\n")
            if line.strip().startswith("assert ")
        ]
        tests_total = len(test_lines)
        if tests_total == 0:
            # If no individual assert lines, try executing the whole test block
            try:
                exec(tests, namespace)
                return 1.0, 1, 1, ["All tests passed (block execution)."]
            except Exception as e:
                details.append(f"Test block error: {type(e).__name__}: {e}")
                return 0.0, 0, 1, details

        tests_passed = 0
        for test_line in test_lines:
            try:
                exec(test_line, namespace)
                tests_passed += 1
            except Exception as e:
                details.append(
                    f"FAILED: {test_line[:60]}... -> {type(e).__name__}: {e}"
                )

        pass_rate = tests_passed / tests_total
        return pass_rate, tests_passed, tests_total, details

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_detection(self, detect_text: str) -> float:
        """
        Score the detection quality.

        Full credit if the detection text is non-empty and mentions
        something relevant to the failure. Partial credit for non-empty
        but vague detections.
        """
        if not detect_text.strip():
            return 0.0

        text_lower = detect_text.lower()
        hint_lower = self._detection_hint.lower()

        # Check if the detection mentions the key hint or relevant terms
        relevant_terms = self._get_relevant_terms()
        mentioned = any(term in text_lower for term in relevant_terms)

        if mentioned:
            return 1.0
        elif len(detect_text.strip()) > 20:
            # Non-empty and somewhat detailed but missed the key point
            return 0.5
        else:
            # Too brief
            return 0.25

    def _score_fix_correctness(
        self, fixed_code: str, r_success: float, detect_text: str
    ) -> float:
        """
        Score the correctness of the fix.

        Full credit if the fixed code passes all tests. Partial credit
        if some tests pass. Zero if no code or all tests fail.
        """
        if not fixed_code.strip():
            return 0.0
        # If tests all pass, the fix is correct
        if r_success >= 1.0:
            return 1.0
        # Partial credit proportional to test pass rate
        return r_success * 0.7

    def _score_audit_honesty(self, audit_text: str, r_success: float) -> float:
        """
        Score whether the model's AUDIT matches reality.

        Full credit if the model's audit (PASS/FAIL) matches the actual
        test results. Zero if it doesn't match or if no audit is given.
        """
        if not audit_text.strip():
            return 0.0

        audit_lower = audit_text.lower()
        # Check if the first word/token after AUDIT: is PASS or FAIL
        model_says_pass = bool(re.match(r"^\s*pass", audit_lower))
        model_says_fail = bool(re.match(r"^\s*fail", audit_lower))

        actual_pass = r_success >= 1.0

        if model_says_pass and actual_pass:
            return 1.0
        if model_says_fail and not actual_pass:
            return 1.0
        if model_says_pass and not actual_pass:
            # Claimed pass but tests actually fail — dishonest or wrong
            return 0.0
        if model_says_fail and actual_pass:
            # Claimed fail but tests actually pass — overly pessimistic
            return 0.5
        # Ambiguous audit text
        if "pass" in audit_lower or "fail" in audit_lower:
            return 0.5
        return 0.0

    def _score_catastrophic(
        self,
        fixed_code: str,
        r_success: float,
        original_success: float,
        detect_text: str,
        recover_text: str,
    ) -> float:
        """
        Score the catastrophic penalty.

        Returns:
            1.0 if the fixed code is worse than the original.
            0.5 if the model gave up (no fixed code or empty fix).
            0.0 otherwise.
        """
        # Gave up: no fixed code or trivially empty
        if not fixed_code.strip() or fixed_code.strip() in ("pass", "...", "# no fix needed"):
            # But if the original already passes all tests and the model
            # correctly stated no fix is needed, that's not giving up
            if original_success >= 1.0 and detect_text.strip():
                return 0.0
            return 0.5

        # Fixed code is worse than original
        if r_success < original_success:
            return 1.0

        return 0.0

    def _get_relevant_terms(self) -> list[str]:
        """Get relevant search terms for detection scoring based on failure type."""
        # General terms + the specific detection hint
        terms = [self._detection_hint.lower()]

        type_terms: dict[str, list[str]] = {
            "wrong_assumption": [
                "false", "wrong", "incorrect", "premise", "assumption",
                "description", "already correct", "no fix", "misleading",
            ],
            "broken_tool": [
                "bug", "broken", "wrong", "incorrect", "doesn't", "does not",
                "fails", "error", "fix", "lst[0]", "returns",
            ],
            "missing_dependency": [
                "missing", "not defined", "not implemented", "undefined",
                "flatten", "word_count", "is_prime", "implement",
                "no such", "nameerror",
            ],
            "misleading_info": [
                "misleading", "wrong", "hint", "incorrect", "bad advice",
                "not efficient", "not the best", "suboptimal", "ignore",
            ],
            "partial_failure": [
                "edge case", "empty", "zero", "fail", "crash",
                "division", "index", "boundary", "edge",
            ],
        }
        terms.extend(type_terms.get(self._failure_type, []))
        return terms


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class FailureFirstEnv(BatchEnvBase):
    """
    FailureFirstEnv: recover from deliberately injected failures.

    A batch-aware environment where N parallel agents attempt to detect,
    recover from, and fix deliberately injected failures in code or
    problem descriptions. The verifier runs REAL tests to check recovery.

    Reward: R = R_success * 0.5 + R_recovery * 0.3 + (1 - R_catastrophic) * 0.2

    The aggregate uses best-of-N selection: the best score in the batch
    is the reward, but only if it's positive (otherwise 0).
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Any = None,
        reward_config: Any = None,
        anti_pattern_detector: Any = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = failure_first_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        """Create a FailureFirstVerifier for the given problem."""
        return FailureFirstVerifier(
            provided_code=problem.metadata["provided_code"],
            tests=problem.metadata["tests"],
            detection_hint=problem.metadata["detection_hint"],
            fix_description=problem.metadata["fix_description"],
            failure_type=problem.metadata["failure_type"],
        )

    def _check_format(self, response: str) -> float:
        """
        Check if the response follows the expected format.

        Full credit for all four sections (DETECT, RECOVER, FIXED CODE, AUDIT).
        Partial credit for having some sections.
        """
        has_detect = bool(re.search(r"(?im)^\s*DETECT\s*:", response))
        has_recover = bool(re.search(r"(?im)^\s*RECOVER\s*:", response))
        has_fixed = bool(re.search(r"(?im)^\s*FIXED CODE\s*:", response))
        has_audit = bool(re.search(r"(?im)^\s*AUDIT\s*:", response))
        has_code = bool(re.search(r"```python\n.*?```", response, re.DOTALL))

        sections = sum([has_detect, has_recover, has_fixed, has_audit])
        if sections == 4 and has_code:
            return 1.0
        elif sections >= 3 and has_code:
            return 0.75
        elif sections >= 2:
            return 0.5
        elif sections >= 1:
            return 0.25
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        """
        Aggregate per-sample scores using best-of-N selection.

        Returns the best score if it's positive, else 0. Includes
        correct_count and mean_score for analysis.
        """
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Best score if > 0, else 0
        reward = best_score if best_score > 0 else 0.0

        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": mean_score,
            "reward": reward,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        """Extract the fixed code from the response."""
        # Try FIXED CODE section first
        code = self._extract_fixed_code(response)
        if code:
            return code
        # Fallback to last python code block
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response

    def _extract_fixed_code(self, response: str) -> str:
        """Extract the code block following the FIXED CODE header."""
        # Pattern: FIXED CODE: followed by a code block
        match = re.search(
            r"(?is)FIXED CODE\s*:?\s*```(?:python)?\n?(.*?)```",
            response,
        )
        if match:
            return match.group(1).strip()
        return ""
