"""
KernelOptimizer: Optimize Python functions for performance while maintaining correctness.

Environment concept:
  The model receives a baseline Python function (typically O(n^2) or worse)
  and must produce an optimized version that is asymptotically faster
  (O(n log n) or O(n)) while returning identical results on all test inputs.

  Verification is rule-based and FAST:
    1. Correctness: run both baseline and optimized on a suite of test inputs
       (small + medium sizes) and compare outputs exactly.
    2. Performance: time both implementations on a large input and compute
       the speedup ratio. A timeout protects against infinite loops.
    3. Code quality: check that the optimized code is not identical to the
       baseline (real optimization happened) and is reasonably concise.

  reward = correctness * 0.5 + speedup * 0.3 + code_quality * 0.2

  This is a batch environment: N parallel optimizations are scored, and the
  best (highest combined score) is selected as the aggregate reward.

  The key insight: kernel optimization is the core of AGI-scale systems
  engineering. A model that can take a slow function and make it fast —
  while proving correctness via test equivalence — is directly useful for
  compiler backends, database engines, and ML infrastructure.
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
# Problem templates: baseline (slow) functions + test input generators
# ---------------------------------------------------------------------------


def _gen_list(n: int, rng: random.Random) -> list[int]:
    return [rng.randint(-1000, 1000) for _ in range(n)]


def _gen_sorted_list(n: int, rng: random.Random) -> list[int]:
    return sorted(_gen_list(n, rng))


def _gen_string(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(n))


_KERNEL_PROBLEMS: list[dict[str, Any]] = [
    {
        "task": "Sort a list of integers (baseline uses bubble sort).",
        "func_name": "sort_list",
        "baseline": (
            "def sort_list(lst):\n"
            "    result = list(lst)\n"
            "    for i in range(len(result)):\n"
            "        for j in range(len(result) - 1 - i):\n"
            "            if result[j] > result[j + 1]:\n"
            "                result[j], result[j + 1] = result[j + 1], result[j]\n"
            "    return result\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 3000,
        "target_complexity": "O(n log n)",
    },
    {
        "task": "Count occurrences of each element (baseline uses nested loops).",
        "func_name": "count_occurrences",
        "baseline": (
            "def count_occurrences(lst):\n"
            "    result = {}\n"
            "    for item in lst:\n"
            "        count = 0\n"
            "        for other in lst:\n"
            "            if item == other:\n"
            "                count += 1\n"
            "        result[item] = count\n"
            "    return result\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 2000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Remove duplicates from a list preserving order (baseline uses 'in' on list).",
        "func_name": "remove_duplicates",
        "baseline": (
            "def remove_duplicates(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if item not in result:\n"
            "            result.append(item)\n"
            "        # O(n) membership check makes this O(n^2)\n"
            "    return result\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 3000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Find the maximum subarray sum (baseline checks all subarrays).",
        "func_name": "max_subarray_sum",
        "baseline": (
            "def max_subarray_sum(lst):\n"
            "    best = 0\n"
            "    for i in range(len(lst)):\n"
            "        total = 0\n"
            "        for j in range(i, len(lst)):\n"
            "            total += lst[j]\n"
            "            if total > best:\n"
            "                best = total\n"
            "    return best\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 5000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Check if a string has all unique characters (baseline uses nested loops).",
        "func_name": "all_unique",
        "baseline": (
            "def all_unique(s):\n"
            "    for i in range(len(s)):\n"
            "        for j in range(i + 1, len(s)):\n"
            "            if s[i] == s[j]:\n"
            "                return False\n"
            "    return True\n"
        ),
        "input_gen": _gen_string,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 5000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Find intersection of two lists (baseline uses nested loops).",
        "func_name": "intersect",
        "baseline": (
            "def intersect(a, b):\n"
            "    result = []\n"
            "    for item in a:\n"
            "        for other in b:\n"
            "            if item == other and item not in result:\n"
            "                result.append(item)\n"
            "    return result\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 1500,
        "target_complexity": "O(n + m)",
        "two_inputs": True,
    },
    {
        "task": "Compute prefix sums (baseline uses sum() in a loop = O(n^2)).",
        "func_name": "prefix_sums",
        "baseline": (
            "def prefix_sums(lst):\n"
            "    result = []\n"
            "    for i in range(len(lst)):\n"
            "        result.append(sum(lst[:i + 1]))\n"
            "    return result\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 5000,
        "target_complexity": "O(n)",
    },
    {
        "task": "Two-sum: find indices of two elements that sum to target (baseline nested loop).",
        "func_name": "two_sum",
        "baseline": (
            "def two_sum(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return [i, j]\n"
            "    return []\n"
        ),
        "input_gen": _gen_list,
        "small_sizes": [0, 1, 5, 10],
        "large_size": 3000,
        "target_complexity": "O(n)",
        "extra_arg": "target",
    },
]


def kernel_optimizer_generator(seed: int) -> Problem:
    """Generate a KernelOptimizer problem.

    Picks a template, generates random test inputs of various sizes, and
    builds a prompt asking the model to optimize the baseline function.
    """
    rng = random.Random(seed)
    template = rng.choice(_KERNEL_PROBLEMS)

    # Generate test inputs (deterministic from seed)
    test_inputs: list[Any] = []
    for size in template["small_sizes"]:
        if template.get("two_inputs"):
            a = template["input_gen"](size, rng)
            b = template["input_gen"](size, rng)
            test_inputs.append((a, b))
        else:
            test_inputs.append(template["input_gen"](size, rng))

    # Large input for timing
    large_n = template["large_size"]
    if template.get("two_inputs"):
        large_input = (
            template["input_gen"](large_n, rng),
            template["input_gen"](large_n, rng),
        )
    else:
        large_input = template["input_gen"](large_n, rng)

    # Extra argument (e.g. target for two_sum)
    extra_arg = None
    if template.get("extra_arg") == "target":
        # Pick a target that exists in the input (guarantees a solution)
        if isinstance(large_input, list) and large_input:
            extra_arg = large_input[0] + large_input[-1]
        else:
            extra_arg = rng.randint(-1000, 1000)

    prompt = (
        f"Optimize this function:\n"
        f"```python\n{template['baseline']}```\n\n"
        f"Task: {template['task']}\n"
        f"Target complexity: {template['target_complexity']}\n\n"
        f"Requirements:\n"
        f"  - The optimized function must produce IDENTICAL output to the baseline\n"
        f"  - It must be asymptotically faster (target: {template['target_complexity']})\n"
        f"  - Keep the same function name: {template['func_name']}\n\n"
        f"Format your answer as:\n"
        f"```python\n<optimized function>\n```"
    )

    return Problem(
        id=f"kernel_opt_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=0.4 + 0.15 * rng.random(),
        metadata={
            "type": "kernel_optimizer",
            "task": template["task"],
            "func_name": template["func_name"],
            "baseline": template["baseline"],
            "test_inputs": test_inputs,
            "large_input": large_input,
            "extra_arg": extra_arg,
            "two_inputs": template.get("two_inputs", False),
            "large_size": large_n,
            "target_complexity": template["target_complexity"],
        },
        token_budget=1500,
        source="generated",
    )


# Prevent pytest from collecting this generator as a test
kernel_optimizer_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier — correctness + performance + code quality
# ---------------------------------------------------------------------------


class KernelOptimizerVerifier(Verifier):
    """Verify an optimized function: correctness, speedup, code quality.

    All checks are rule-based and FAST:
      - Correctness: run baseline and optimized on small/medium inputs, compare.
      - Performance: time both on a large input with a hard timeout.
      - Code quality: check the optimized code differs from baseline and is concise.
    """

    def __init__(
        self,
        baseline: str,
        func_name: str,
        test_inputs: list[Any],
        large_input: Any,
        extra_arg: Any,
        two_inputs: bool,
        large_size: int,
        timeout_seconds: float = 3.0,
    ):
        super().__init__()
        self._baseline = baseline
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._large_input = large_input
        self._extra_arg = extra_arg
        self._two_inputs = two_inputs
        self._large_size = large_size
        self._timeout = timeout_seconds

    def verify(self, response: str) -> VerifierResult:
        optimized_code = self._extract_code(response)
        if not optimized_code:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics="No python code block found in response",
            )

        # --- Compile and load both functions ---
        baseline_ns: dict[str, Any] = {}
        optimized_ns: dict[str, Any] = {}
        try:
            exec(self._baseline, baseline_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Baseline failed to compile: {type(e).__name__}: {e}",
            )
        try:
            exec(optimized_code, optimized_ns)
        except Exception as e:
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Optimized code failed to compile: {type(e).__name__}: {e}",
            )

        base_func = baseline_ns.get(self._func_name)
        opt_func = optimized_ns.get(self._func_name)
        if not callable(opt_func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in optimized code",
            )
        if not callable(base_func):
            return VerifierResult(
                correct=False, score=0.0,
                diagnostics=f"Function '{self._func_name}' not found in baseline",
            )

        # --- Correctness: compare outputs on all test inputs ---
        correct_count = 0
        total_inputs = len(self._test_inputs)
        correctness_diag: list[str] = []
        for inp in self._test_inputs:
            try:
                if self._two_inputs:
                    base_out = base_func(inp[0], inp[1])
                elif self._extra_arg is not None:
                    base_out = base_func(inp, self._extra_arg)
                else:
                    base_out = base_func(inp)
            except Exception as e:
                correctness_diag.append(f"baseline err on {inp!r}: {e}")
                continue
            try:
                if self._two_inputs:
                    opt_out = opt_func(inp[0], inp[1])
                elif self._extra_arg is not None:
                    opt_out = opt_func(inp, self._extra_arg)
                else:
                    opt_out = opt_func(inp)
            except Exception as e:
                correctness_diag.append(f"opt err on {inp!r}: {e}")
                continue
            if base_out == opt_out:
                correct_count += 1
            else:
                correctness_diag.append(f"mismatch on {inp!r}: {base_out!r} != {opt_out!r}")

        correctness = correct_count / total_inputs if total_inputs > 0 else 0.0

        # --- Performance: time both on large input ---
        speedup = 0.0
        perf_diag = ""
        if correctness >= 0.5:  # only time if at least somewhat correct
            base_time = self._time_function(base_func)
            opt_time = self._time_function(opt_func)
            if base_time is not None and opt_time is not None:
                if opt_time <= 0:
                    speedup = 10.0  # extremely fast (capped)
                else:
                    speedup = base_time / opt_time
                perf_diag = f"base={base_time:.4f}s opt={opt_time:.4f}s speedup={speedup:.2f}x"
            elif base_time is None:
                perf_diag = "baseline timed out"
            elif opt_time is None:
                perf_diag = "optimized timed out"
                speedup = 0.0

        # Normalize speedup to [0, 1]: 1x = no improvement, 10x+ = perfect
        speedup_score = min(1.0, max(0.0, (speedup - 1.0) / 9.0)) if speedup > 0 else 0.0

        # --- Code quality: differs from baseline + reasonable length ---
        base_stripped = re.sub(r"\s+", "", self._baseline)
        opt_stripped = re.sub(r"\s+", "", optimized_code)
        differs = base_stripped != opt_stripped
        # Conciseness: prefer code under ~30 lines
        opt_lines = optimized_code.strip().count("\n") + 1
        length_ok = opt_lines <= 40
        code_quality = 0.0
        if differs:
            code_quality += 0.5
        if length_ok:
            code_quality += 0.5

        # --- Final score ---
        score = correctness * 0.5 + speedup_score * 0.3 + code_quality * 0.2
        correct = correctness >= 0.9 and speedup >= 2.0

        diag = (
            f"Correctness: {correct_count}/{total_inputs} "
            f"Speedup: {speedup:.2f}x ({perf_diag}) "
            f"CodeQuality: {code_quality:.1f} "
            f"{'| '.join(correctness_diag[:3])}"
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
                "total_inputs": total_inputs,
            },
            diagnostics=diag,
        )

    def _time_function(self, func: Any) -> Optional[float]:
        """Time a function on the large input with a timeout. Returns seconds or None on timeout."""
        try:
            start = time.time()
            if self._two_inputs:
                func(self._large_input[0], self._large_input[1])
            elif self._extra_arg is not None:
                func(self._large_input, self._extra_arg)
            else:
                func(self._large_input)
            elapsed = time.time() - start
            if elapsed > self._timeout * 5:  # hard cap
                return None
            return elapsed
        except Exception:
            return None

    def _extract_code(self, response: str) -> str:
        """Extract the last python code block from the response."""
        matches = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"```\n(.*?)```", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
        # Fallback: if the whole response looks like code
        if "def " in response and "```" not in response:
            return response.strip()
        return ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class KernelOptimizerEnv(BatchEnvBase):
    """KernelOptimizer: optimize Python functions for performance + correctness.

    Batch-aware: N parallel optimizations are scored; reward = best score.
    Verification is rule-based (test equivalence + timing) and FAST.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = kernel_optimizer_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        md = problem.metadata
        return KernelOptimizerVerifier(
            baseline=md["baseline"],
            func_name=md["func_name"],
            test_inputs=md["test_inputs"],
            large_input=md["large_input"],
            extra_arg=md.get("extra_arg"),
            two_inputs=md.get("two_inputs", False),
            large_size=md["large_size"],
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
