"""
PerformanceProfiling: Optimize code with profiling feedback.

Environment concept:
  The model is given slow code plus a fake profiling report that identifies
  the bottleneck. The model must optimize the code based on the profile.

  Verification is rule-based and FAST:
    - Correctness: run both original and optimized on test inputs, compare.
    - Performance: time both on a large input, compute speedup ratio.

  reward = correctness * 0.5 + speedup * 0.3 + code_quality * 0.2
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem templates: slow code + profile report
# ---------------------------------------------------------------------------


def _gen_list(n: int, rng: random.Random) -> list[int]:
    return [rng.randint(-1000, 1000) for _ in range(n)]


_PROFILING_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Optimize a slow duplicate-removal function.",
        "func_name": "remove_dupes",
        "slow": (
            "def remove_dupes(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "profile_report": (
            "Profiling report:\n"
            "  remove_dupes() — 85% of runtime\n"
            "  Bottleneck: 'item not in result' performs O(n) list scan\n"
            "  Called 1 time on input of size 3000\n"
            "  Suggestion: use a set for O(1) membership checks\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 3000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Optimize a slow count-occurrences function.",
        "func_name": "count_all",
        "slow": (
            "def count_all(lst):\n"
            "    result = {}\n"
            "    for item in lst:\n"
            "        count = 0\n"
            "        for other in lst:\n"
            "            if item == other:\n"
            "                count += 1\n"
            "        result[item] = count\n"
            "    return result\n"
        ),
        "profile_report": (
            "Profiling report:\n"
            "  count_all() — 92% of runtime\n"
            "  Bottleneck: inner loop 'for other in lst' is O(n) per item\n"
            "  Called 1 time on input of size 2000\n"
            "  Suggestion: use collections.Counter or a dict accumulator\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 2000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Optimize a slow prefix-sum function.",
        "func_name": "prefix_sum",
        "slow": (
            "def prefix_sum(lst):\n"
            "    result = []\n"
            "    for i in range(len(lst)):\n"
            "        result.append(sum(lst[:i + 1]))\n"
            "    return result\n"
        ),
        "profile_report": (
            "Profiling report:\n"
            "  prefix_sum() — 88% of runtime\n"
            "  Bottleneck: 'sum(lst[:i+1])' recomputes the sum each iteration\n"
            "  Called 1 time on input of size 5000\n"
            "  Suggestion: maintain a running total\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 5000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Optimize a slow string-unique check.",
        "func_name": "all_unique",
        "slow": (
            "def all_unique(s):\n"
            "    for i in range(len(s)):\n"
            "        for j in range(i + 1, len(s)):\n"
            "            if s[i] == s[j]:\n"
            "                return False\n"
            "    return True\n"
        ),
        "profile_report": (
            "Profiling report:\n"
            "  all_unique() — 90% of runtime\n"
            "  Bottleneck: nested loop compares all character pairs\n"
            "  Called 1 time on input of size 5000\n"
            "  Suggestion: use a set to track seen characters\n"
        ),
        "input_gen": lambda n, rng: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)),
        "small_sizes": [0, 1, 5, 10],
        "large_size": 5000,
        "target_complexity": "O(n)",
    },
]


def performance_profiling_generator(seed: int) -> Problem:
    """Generate a PerformanceProfiling problem."""
    rng = random.Random(seed)
    template = rng.choice(_PROFILING_PROBLEMS)

    test_inputs: list[Any] = []
    for size in template["small_sizes"]:
        test_inputs.append(template["input_gen"](size, rng))

    large_input = template["input_gen"](template["large_size"], rng)

    prompt = (
        f"Optimize this function based on the profiling report:\n\n"
        f"```python\n{template['slow']}```\n\n"
        f"{template['profile_report']}\n"
        f"Task: {template['task']}\n"
        f"Target complexity: {template['target_complexity']}\n\n"
        f"Requirements:\n"
        f"  - The optimized function must produce IDENTICAL output to the original\n"
        f"  - It must be faster (target: {template['target_complexity']})\n"
        f"  - Keep the same function name: {template['func_name']}\n\n"
        f"Format your answer as:\n"
        f"```python\n<optimized function>\n```"
    )

    return Problem(
        id=f"perf_profile_{seed}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "performance_profiling",
            "task": template["task"],
            "func_name": template["func_name"],
            "original_code": template["slow"],
            "profile_report": template["profile_report"],
            "test_inputs": test_inputs,
            "large_input": large_input,
            "large_size": template["large_size"],
            "target_complexity": template["target_complexity"],
        },
        token_budget=1500,
        source="generated",
    )


performance_profiling_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PerformanceProfilingVerifier(Verifier):
    """Verify an optimized function: correctness + speedup + code quality."""

    def __init__(
        self,
        original_code: str,
        func_name: str,
        test_inputs: list[Any],
        large_input: Any,
        timeout_seconds: float = 3.0,
    ):
        super().__init__()
        self._original = original_code
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._large_input = large_input
        self._timeout = timeout_seconds

    def verify(self, response: str) -> VerifierResult:
        opt_code = self._extract_code(response)
        if not opt_code:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No python code block found in response",
            )

        base_ns: dict[str, Any] = {}
        opt_ns: dict[str, Any] = {}
        try:
            exec(self._original, base_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Original failed to compile: {e}",
            )
        try:
            exec(opt_code, opt_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Optimized code failed to compile: {e}",
            )

        base_func = base_ns.get(self._func_name)
        opt_func = opt_ns.get(self._func_name)
        if not callable(opt_func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in optimized code",
            )
        if not callable(base_func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in original",
            )

        # --- Correctness ---
        correct_count = 0
        total = len(self._test_inputs)
        diag: list[str] = []
        for inp in self._test_inputs:
            try:
                base_out = base_func(inp)
            except Exception as e:
                diag.append(f"baseline err on {inp!r}: {e}")
                continue
            try:
                opt_out = opt_func(inp)
            except Exception as e:
                diag.append(f"opt err on {inp!r}: {e}")
                continue
            if base_out == opt_out:
                correct_count += 1
            else:
                diag.append(f"mismatch on {inp!r}: {base_out!r} != {opt_out!r}")

        correctness = correct_count / total if total > 0 else 0.0

        # --- Performance ---
        speedup = 0.0
        perf_diag = ""
        if correctness >= 0.5:
            base_time = self._time_function(base_func)
            opt_time = self._time_function(opt_func)
            if base_time is not None and opt_time is not None:
                if opt_time <= 0:
                    speedup = 10.0
                else:
                    speedup = base_time / opt_time
                perf_diag = f"base={base_time:.4f}s opt={opt_time:.4f}s speedup={speedup:.2f}x"
            elif base_time is None:
                perf_diag = "baseline timed out"
            elif opt_time is None:
                perf_diag = "optimized timed out"
                speedup = 0.0

        speedup_score = min(1.0, max(0.0, (speedup - 1.0) / 9.0)) if speedup > 0 else 0.0

        # --- Code quality ---
        base_stripped = re.sub(r"\s+", "", self._original)
        opt_stripped = re.sub(r"\s+", "", opt_code)
        differs = base_stripped != opt_stripped
        opt_lines = opt_code.strip().count("\n") + 1
        length_ok = opt_lines <= 40
        code_quality = 0.0
        if differs:
            code_quality += 0.5
        if length_ok:
            code_quality += 0.5

        score = correctness * 0.5 + speedup_score * 0.3 + code_quality * 0.2
        correct = correctness >= 0.9 and speedup >= 2.0

        full_diag = (
            f"Correctness: {correct_count}/{total} "
            f"Speedup: {speedup:.2f}x ({perf_diag}) "
            f"CodeQuality: {code_quality:.1f} "
            f"{'| '.join(diag[:3])}"
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "correctness": correctness,
                "speedup": speedup,
                "speedup_score": speedup_score,
                "code_quality": code_quality,
                "correct_inputs": correct_count,
                "total_inputs": total,
            },
            diagnostics=full_diag,
        )

    def _time_function(self, func: Any) -> Optional[float]:
        try:
            start = time.time()
            func(self._large_input)
            elapsed = time.time() - start
            if elapsed > self._timeout * 5:
                return None
            return elapsed
        except Exception:
            return None

    def _extract_code(self, response: str) -> str:
        matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        if "def " in response and "```" not in response:
            return response.strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PerformanceProfilingEnv(BatchEnvBase):
    """PerformanceProfiling: optimize code with profiling feedback."""

    __test__ = False

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = performance_profiling_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return PerformanceProfilingVerifier(
            original_code=md["original_code"],
            func_name=md["func_name"],
            test_inputs=md["test_inputs"],
            large_input=md["large_input"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"```python\n.*?```", response, re.DOTALL):
            return 1.0
        if "def " in response:
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
            "diagnostics": f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} Mean={sum(scores)/len(scores) if scores else 0:.2f}",
        }

    def _extract_answer(self, response: str) -> str:
        return response
