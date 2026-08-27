"""
TrajectoryDoctor: Diagnose the first wrong step in a flawed reasoning trace.

Environment concept:
  The model is given a reasoning trace from a frontier model (or a synthetic
  one modeled on real frontier-model patterns) that contains an error. The
  model must:
    1. Identify the FIRST step that goes wrong
    2. Explain WHY it's wrong (in one sentence)
    3. Provide the correct step that should replace it

  This trains the model to:
    - Read reasoning traces critically, not passively
    - Identify errors precisely (not "somewhere around step 3")
    - Explain errors concisely (not write a paragraph)
    - Fix errors minimally (not rewrite the whole trace)

  The key insight: a model that can diagnose WHERE reasoning goes wrong in
  OTHERS' traces will develop the metacognitive skill to catch errors in
  ITS OWN traces before they propagate. This is the "debugger" skill applied
  to reasoning itself.

Why this environment is unique:
  Most RL environments train the model to PRODUCE correct reasoning.
  TrajectoryDoctor trains the model to EVALUATE reasoning — a fundamentally
  different skill that creates a self-monitoring capability. It's like the
  difference between writing code and doing code review.

Data sources:
  - Synthetic traces modeled on real frontier-model anti-patterns
    (backtracking, wrong steps, circular reasoning, etc.)
  - Can be extended with real traces from HuggingFace datasets:
    * Zigeng/CoT-Verification-340k (190k incorrect CoT)
    * Salesforce/Hard2Verify (first-error annotations)
    * jinulee-v/reasoningflow (DAG-annotated traces)
    * vanthienha199/thinktank-reasoning-labels (useful/wasteful labels)

Problem types:
  - Math traces with an arithmetic error at a specific step
  - Logic traces with a faulty inference
  - Code traces with a wrong algorithm step
  - Traces with anti-pattern waste (the "wrong step" is a wasted step)

Verification:
  Each problem includes:
    - The flawed trace (broken into numbered steps)
    - The index of the first wrong step
    - The explanation of why it's wrong
    - The correct replacement step
  The verifier checks:
    1. Did the model identify the correct step number?
    2. Is the explanation semantically correct (keyword overlap)?
    3. Is the replacement step correct (produces the right result)?

Reward design:
  - Step identification: 40% of reward (binary: correct/incorrect step number)
  - Explanation quality: 30% (keyword overlap with expected explanation)
  - Fix correctness: 30% (does the replacement step fix the trace?)
  - Conciseness bonus: up to 20% extra for short, precise diagnoses
  - Anti-pattern penalties apply to the model's own reasoning
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Real trace loading
# ---------------------------------------------------------------------------

_TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
_real_trace_cache: Optional[list[dict]] = None


def _load_real_traces() -> list[dict]:
    """Load real frontier model traces from the downloaded JSON files.

    Prefers the curated file (curated_all.json) if available, which contains
    traces filtered by anti-pattern count and annotated with waste_score
    and trace_type. Falls back to loading individual trace files.
    """
    global _real_trace_cache
    if _real_trace_cache is not None:
        return _real_trace_cache

    traces = []
    traces_dir = os.path.normpath(_TRACES_DIR)
    if not os.path.isdir(traces_dir):
        _real_trace_cache = []
        return _real_trace_cache

    # Prefer curated file
    curated_path = os.path.join(traces_dir, "curated_all.json")
    if os.path.isfile(curated_path):
        try:
            with open(curated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    _real_trace_cache = data
                    return _real_trace_cache
        except (json.JSONDecodeError, OSError):
            pass  # fall through to individual files

    for fname in os.listdir(traces_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(traces_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    traces.extend(data)
        except (json.JSONDecodeError, OSError):
            continue

    _real_trace_cache = traces
    return traces


def _split_trace_into_steps(text: str) -> list[str]:
    """Split a reasoning trace into steps (sentence-based)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _find_first_wasteful_step(steps: list[str], prompt: str) -> int:
    """Find the first step with anti-patterns using the AntiPatternDetector."""
    from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
    detector = AntiPatternDetector()

    for i, step in enumerate(steps):
        report = detector.analyze(step, prompt=prompt)
        if report.total_hits > 0:
            return i
    return -1


def _generate_real_trace_problem(rng: random.Random) -> Optional[Problem]:
    """
    Generate a TrajectoryDoctor problem from a real frontier model trace.

    Returns None if no suitable traces are available.
    """
    traces = _load_real_traces()
    if not traces:
        return None

    # Filter for traces with enough content
    suitable = []
    for t in traces:
        text_candidate = t.get("reasoning", "") or t.get("response", "")
        if len(text_candidate) > 100:
            suitable.append(t)
    if not suitable:
        return None

    trace = rng.choice(suitable)
    text = trace.get("reasoning", "") or trace.get("response", "")
    prompt = trace.get("prompt", "")
    source = trace.get("source", "unknown")
    model = trace.get("model", "unknown")

    # Split into steps
    steps = _split_trace_into_steps(text)
    if len(steps) < 3:
        return None

    # Cap at 10 steps for the problem
    steps = steps[:10]

    # Find the first wasteful step
    error_step = _find_first_wasteful_step(steps, prompt)
    if error_step == -1:
        # No anti-pattern found — use a random step as "potentially improvable"
        # This still trains the evaluation skill
        error_step = min(2, len(steps) - 1)

    # Build keywords from the wasteful step
    from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
    detector = AntiPatternDetector()
    report = detector.analyze(steps[error_step], prompt=prompt)

    keywords = []
    for hit in report.hits[:5]:
        keywords.extend(hit.text.lower().split()[:3])
    if not keywords:
        # Extract key nouns from the step
        words = re.findall(r"\b[a-z]{4,}\b", steps[error_step].lower())
        common = {"that", "this", "with", "from", "have", "they", "will", "been", "were", "which", "their", "would"}
        keywords = [w for w in words if w not in common][:5]

    # Determine trace type — prefer curated metadata if available
    trace_meta = trace.get("metadata", {})
    curated_type = trace.get("trace_type") or trace_meta.get("trace_type")
    if curated_type in ("coding", "math", "reasoning"):
        trace_type = "code" if curated_type == "coding" else curated_type
    else:
        trace_type = "waste" if report.total_hits > 0 else "math"
        if "cot" in source.lower() or "math" in source.lower() or "openr1" in source.lower():
            trace_type = "math"
        elif "coding" in source.lower() or "kimi" in source.lower() or "agent" in source.lower() or "code" in source.lower():
            trace_type = "code"

    # Format the trace
    trace_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))

    # Build explanation
    if report.total_hits > 0:
        ap_types = list(report.counts_by_type.keys())
        explanation = f"This step contains {', '.join(ap_types)} — it adds no information toward solving the problem."
    else:
        explanation = "This step may contain an error or unnecessary content."

    # Correct step: for waste, suggest removal; for errors, suggest the step is wrong
    if trace_type == "waste":
        correct_step_text = "REMOVE"
    else:
        correct_step_text = f"(Step {error_step+1} needs correction)"

    difficulty = 0.4 + 0.2 * rng.random()

    return Problem(
        id=f"traj_real_{source}_{rng.randint(0, 99999)}",
        prompt=_format_prompt(trace_text, len(steps), is_waste=(trace_type == "waste")),
        difficulty=difficulty,
        metadata={
            "type": "trajectory_doctor",
            "trace_type": trace_type,
            "flawed_steps": steps,
            "correct_steps": steps,
            "error_step": error_step,
            "explanation": explanation,
            "keywords": keywords if keywords else ["error", "wrong", "step"],
            "correct_step_text": correct_step_text,
            "source_model": model,
            "source_dataset": source,
            "is_real_trace": True,
            "waste_score": trace.get("waste_score", trace_meta.get("waste_score", report.total_penalty)),
            "is_clean_trace": trace_meta.get("clean", False),
        },
        token_budget=600,
        source="real_trace",
    )



# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------


def _generate_math_trace_with_error(rng: random.Random) -> Problem:
    """Generate a math reasoning trace with an arithmetic error at a specific step."""

    templates = [
        {
            "steps": [
                "We need to compute 15 * 12.",
                "15 * 12 = 15 * (10 + 2) = 150 + 30 = 180.",
                "So 15 * 12 = 180.",
            ],
            "error_step": 1,
            "error_text": "15 * 12 = 15 * (10 + 2) = 150 + 30 = 120",
            "correct_step": "15 * 12 = 15 * (10 + 2) = 150 + 30 = 180",
            "explanation": "The addition 150 + 30 = 180, not 120. The arithmetic is wrong.",
            "keywords": ["180", "addition", "arithmetic", "wrong", "150", "30"],
        },
        {
            "steps": [
                "We need to find x where 2x + 6 = 14.",
                "Subtract 6 from both sides: 2x = 14 - 6 = 8.",
                "Divide by 2: x = 8 / 2 = 4.",
                "So x = 4.",
            ],
            "error_step": 1,
            "error_text": "Subtract 6 from both sides: 2x = 14 - 6 = 6",
            "correct_step": "Subtract 6 from both sides: 2x = 14 - 6 = 8",
            "explanation": "14 - 6 = 8, not 6. The subtraction is wrong.",
            "keywords": ["8", "subtraction", "wrong", "14", "6"],
        },
        {
            "steps": [
                "Compute the sum: 3 + 7 + 12 + 8.",
                "3 + 7 = 10.",
                "10 + 12 = 22.",
                "22 + 8 = 30.",
                "The sum is 30.",
            ],
            "error_step": 2,
            "error_text": "10 + 12 = 20",
            "correct_step": "10 + 12 = 22",
            "explanation": "10 + 12 = 22, not 20. The addition is incorrect.",
            "keywords": ["22", "addition", "wrong", "10", "12"],
        },
        {
            "steps": [
                "Simplify: 48 / 6 * 2.",
                "48 / 6 = 8.",
                "8 * 2 = 16.",
                "The answer is 16.",
            ],
            "error_step": 1,
            "error_text": "48 / 6 = 6",
            "correct_step": "48 / 6 = 8",
            "explanation": "48 / 6 = 8, not 6. The division is wrong.",
            "keywords": ["8", "division", "wrong", "48", "6"],
        },
        {
            "steps": [
                "Evaluate: (3 + 4)^2.",
                "3 + 4 = 7.",
                "7^2 = 49.",
                "The answer is 49.",
            ],
            "error_step": 1,
            "error_text": "3 + 4 = 8",
            "correct_step": "3 + 4 = 7",
            "explanation": "3 + 4 = 7, not 8. The addition is wrong.",
            "keywords": ["7", "addition", "wrong", "3", "4"],
        },
        {
            "steps": [
                "Compute 25% of 80.",
                "25% = 0.25.",
                "0.25 * 80 = 20.",
                "So 25% of 80 is 20.",
            ],
            "error_step": 2,
            "error_text": "0.25 * 80 = 200",
            "correct_step": "0.25 * 80 = 20",
            "explanation": "0.25 * 80 = 20, not 200. The multiplication is wrong — likely decimal point error.",
            "keywords": ["20", "multiplication", "wrong", "decimal", "0.25", "80"],
        },
        {
            "steps": [
                "Solve: x^2 = 49.",
                "x = sqrt(49) = 7.",
                "But x could also be -7.",
                "So x = 7 or x = -7.",
            ],
            "error_step": 1,
            "error_text": "x = sqrt(49) = 7. This is the only solution.",
            "correct_step": "x = sqrt(49) = 7. But x could also be -7.",
            "explanation": "The square root has two solutions: +7 and -7. The trace incorrectly claims only one solution.",
            "keywords": ["-7", "two", "solutions", "square", "root", "both"],
        },
    ]

    template = rng.choice(templates)

    # Build the flawed trace by replacing the correct step with the error
    flawed_steps = list(template["steps"])
    flawed_steps[template["error_step"]] = template["error_text"]

    # Format the trace with step numbers
    trace_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(flawed_steps)
    )

    difficulty = 0.3 + 0.3 * rng.random()

    return Problem(
        id=f"traj_math_{rng.randint(0, 99999)}",
        prompt=_format_prompt(trace_text, len(flawed_steps)),
        difficulty=difficulty,
        metadata={
            "type": "trajectory_doctor",
            "trace_type": "math",
            "flawed_steps": flawed_steps,
            "correct_steps": template["steps"],
            "error_step": template["error_step"],
            "explanation": template["explanation"],
            "keywords": template["keywords"],
            "correct_step_text": template["correct_step"],
        },
        token_budget=600,
        source="generated",
    )


def _generate_logic_trace_with_error(rng: random.Random) -> Problem:
    """Generate a logic reasoning trace with a faulty inference."""

    templates = [
        {
            "steps": [
                "Given: All cats are mammals. Whiskers is a cat.",
                "By the definition of 'all', every cat is a mammal.",
                "Since Whiskers is a cat, Whiskers is a mammal.",
                "Conclusion: Whiskers is a mammal.",
            ],
            "error_step": 1,
            "error_text": "By the definition of 'all', every mammal is a cat.",
            "correct_step": "By the definition of 'all', every cat is a mammal.",
            "explanation": "The statement reverses the implication. 'All cats are mammals' means cat → mammal, not mammal → cat.",
            "keywords": ["reverse", "implication", "converse", "cat", "mammal", "direction"],
        },
        {
            "steps": [
                "Given: If it rains, the ground is wet. The ground is wet.",
                "The ground being wet is evidence that it rained.",
                "But the ground could be wet for other reasons (sprinkler, hose).",
                "So we cannot conclude it rained. This is the fallacy of affirming the consequent.",
            ],
            "error_step": 1,
            "error_text": "The ground being wet proves that it rained.",
            "correct_step": "The ground being wet is evidence that it rained, but does not prove it.",
            "explanation": "This is affirming the consequent: rain → wet does not mean wet → rain. Other causes are possible.",
            "keywords": ["affirming", "consequent", "fallacy", "prove", "other", "causes"],
        },
        {
            "steps": [
                "Given: All birds can fly. Penguins are birds.",
                "Therefore penguins can fly.",
                "But this contradicts known facts — penguins cannot fly.",
                "The premise 'all birds can fly' is false.",
            ],
            "error_step": 1,
            "error_text": "Therefore penguins can fly. This is correct.",
            "correct_step": "Therefore penguins can fly. But this contradicts known facts.",
            "explanation": "The step ignores the contradiction with reality. The logic is valid but the premise is false — the trace should recognize this.",
            "keywords": ["contradiction", "premise", "false", "penguins", "fly", "reality"],
        },
        {
            "steps": [
                "We need to prove: if n is even, then n^2 is even.",
                "Let n = 2k for some integer k.",
                "n^2 = (2k)^2 = 4k^2.",
                "4k^2 = 2(2k^2), which is even.",
                "Therefore n^2 is even. QED.",
            ],
            "error_step": 2,
            "error_text": "n^2 = (2k)^2 = 4k^2 = 2k^2",
            "correct_step": "n^2 = (2k)^2 = 4k^2",
            "explanation": "(2k)^2 = 4k^2, not 2k^2. The squaring was done incorrectly — the factor of 2 was lost.",
            "keywords": ["4k", "squaring", "wrong", "factor", "2k", "lost"],
        },
    ]

    template = rng.choice(templates)

    flawed_steps = list(template["steps"])
    flawed_steps[template["error_step"]] = template["error_text"]

    trace_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(flawed_steps)
    )

    difficulty = 0.5 + 0.2 * rng.random()

    return Problem(
        id=f"traj_logic_{rng.randint(0, 99999)}",
        prompt=_format_prompt(trace_text, len(flawed_steps)),
        difficulty=difficulty,
        metadata={
            "type": "trajectory_doctor",
            "trace_type": "logic",
            "flawed_steps": flawed_steps,
            "correct_steps": template["steps"],
            "error_step": template["error_step"],
            "explanation": template["explanation"],
            "keywords": template["keywords"],
            "correct_step_text": template["correct_step"],
        },
        token_budget=700,
        source="generated",
    )


def _generate_waste_trace(rng: random.Random) -> Problem:
    """
    Generate a trace where the 'error' is a wasteful step — the model must
    identify the step that adds no value (backtracking, restating, filler).
    """

    templates = [
        {
            "steps": [
                "We need to compute 7 * 8.",
                "7 * 8 = 56.",
                "Let me double-check: 7 * 8. Yes, 7 * 8 = 56.",
                "Let me verify once more: 7 * 8 = 56. Confirmed.",
                "The answer is 56.",
            ],
            "error_step": 2,  # The "let me double-check" step is waste
            "error_text": "Let me double-check: 7 * 8. Yes, 7 * 8 = 56.",
            "correct_step": "(This step is redundant — the answer was already computed in step 2.)",
            "explanation": "This step is unnecessary verification. The answer was already correctly computed in step 2. Re-checking wastes tokens without adding information.",
            "keywords": ["redundant", "verification", "unnecessary", "waste", "already", "computed"],
        },
        {
            "steps": [
                "Problem: What is 15 + 27?",
                "Let me think about this. This is an interesting problem.",
                "15 + 27 = 42.",
                "The answer is 42.",
            ],
            "error_step": 1,  # The filler step
            "error_text": "Let me think about this. This is an interesting problem.",
            "correct_step": "(This step is filler — it adds no information toward solving the problem.)",
            "explanation": "This step is pure filler. 'Let me think about this' and 'this is an interesting problem' add no information and waste tokens.",
            "keywords": ["filler", "waste", "no", "information", "unnecessary", "interesting"],
        },
        {
            "steps": [
                "We need to find the maximum of [3, 7, 2, 9, 1].",
                "The maximum is 9.",
                "Wait, let me reconsider. Actually, let me check each element.",
                "3 < 9, 7 < 9, 2 < 9, 9 is the max, 1 < 9.",
                "So the maximum is 9.",
            ],
            "error_step": 2,  # The backtrack step
            "error_text": "Wait, let me reconsider. Actually, let me check each element.",
            "correct_step": "(This step is unnecessary backtracking — the correct answer was already given in step 2.)",
            "explanation": "This step backtracks on a correct answer. The model said 9, then unnecessarily doubted itself and re-did the work. This is wasteful backtracking.",
            "keywords": ["backtrack", "unnecessary", "doubt", "reconsider", "waste", "correct"],
        },
    ]

    template = rng.choice(templates)

    flawed_steps = list(template["steps"])
    # For waste traces, the "error" step is the wasteful step itself
    # We don't replace it — we want the model to identify it as waste

    trace_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(flawed_steps)
    )

    difficulty = 0.4 + 0.2 * rng.random()

    return Problem(
        id=f"traj_waste_{rng.randint(0, 99999)}",
        prompt=_format_prompt(trace_text, len(flawed_steps), is_waste=True),
        difficulty=difficulty,
        metadata={
            "type": "trajectory_doctor",
            "trace_type": "waste",
            "flawed_steps": flawed_steps,
            "correct_steps": flawed_steps,  # same, but the error step is waste
            "error_step": template["error_step"],
            "explanation": template["explanation"],
            "keywords": template["keywords"],
            "correct_step_text": template["correct_step"],
        },
        token_budget=500,
        source="generated",
    )


def _format_prompt(trace_text: str, num_steps: int, is_waste: bool = False) -> str:
    """Format the problem prompt."""
    task = "wasteful or unnecessary step" if is_waste else "first incorrect step"
    return (
        f"The following reasoning trace contains a flawed step.\n\n"
        f"Your task:\n"
        f"  1. Identify the {task} (give the step number)\n"
        f"  2. Explain WHY it's wrong or wasteful (one sentence)\n"
        f"  3. Provide the correct replacement (or state it should be removed)\n\n"
        f"Format your response as:\n"
        f"  WRONG STEP: <step number>\n"
        f"  REASON: <one sentence explanation>\n"
        f"  FIX: <correct step or 'REMOVE'>\n\n"
        f"Trace:\n{trace_text}\n\n"
        f"Be concise — do not over-explain."
    )


def trajectory_doctor_generator(seed: int) -> Problem:
    """
    Master generator for TrajectoryDoctor problems.

    70% of problems come from real frontier model traces (downloaded from
    HuggingFace). 30% come from synthetic templates (which have precise
    error annotations). If no real traces are available, falls back to
    100% synthetic.
    """
    rng = random.Random(seed)

    # Try real traces first (70% chance)
    if rng.random() < 0.7:
        real_problem = _generate_real_trace_problem(rng)
        if real_problem is not None:
            return real_problem

    # Fallback to synthetic
    generators = [
        _generate_math_trace_with_error,
        _generate_logic_trace_with_error,
        _generate_waste_trace,
    ]
    weights = [0.4, 0.3, 0.3]
    gen = rng.choices(generators, weights=weights, k=1)[0]
    return gen(rng)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TrajectoryDoctorVerifier(Verifier):
    """
    Verifies a trajectory diagnosis by checking:
      1. Step identification (40%)
      2. Explanation quality (30%) — keyword overlap
      3. Fix correctness (30%) — semantic check
    """

    def __init__(
        self,
        error_step: int,
        explanation: str,
        keywords: list[str],
        correct_step_text: str,
        trace_type: str,
    ):
        super().__init__()
        self._error_step = error_step
        self._explanation = explanation
        self._keywords = keywords
        self._correct_step_text = correct_step_text
        self._trace_type = trace_type

    def verify(self, response: str) -> VerifierResult:
        # Parse the response
        step_match = re.search(r"WRONG\s+STEP:\s*(\d+)", response, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+?)(?:\n\s*FIX:|\n|$)", response, re.IGNORECASE | re.DOTALL)
        fix_match = re.search(r"FIX:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)

        partial = {}

        # 1. Step identification (40%)
        if not step_match:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics="No WRONG STEP: <number> found in response",
            )

        try:
            identified_step = int(step_match.group(1))
        except ValueError:
            return VerifierResult(
                correct=False,
                score=0.0,
                diagnostics=f"Invalid step number: {step_match.group(1)}",
            )

        # Steps are 1-indexed in the prompt, 0-indexed internally
        step_correct = (identified_step - 1) == self._error_step
        step_score = 1.0 if step_correct else 0.0
        partial["step_identification"] = step_score

        # 2. Explanation quality (30%) — keyword overlap
        reason_text = reason_match.group(1).strip().lower() if reason_match else ""
        keyword_hits = sum(1 for kw in self._keywords if kw.lower() in reason_text)
        keyword_score = min(1.0, keyword_hits / max(1, len(self._keywords) * 0.4))
        partial["explanation_quality"] = keyword_score

        # 3. Fix correctness (30%)
        fix_text = fix_match.group(1).strip() if fix_match else ""
        fix_score = self._check_fix(fix_text)
        partial["fix_correctness"] = fix_score

        # Combined score
        total_score = (
            step_score * 0.4 +
            keyword_score * 0.3 +
            fix_score * 0.3
        )

        # Correct only if step is identified AND at least some explanation
        correct = step_correct and keyword_score > 0.3 and fix_score > 0.3

        diagnostics_parts = []
        diagnostics_parts.append(f"Step: expected {self._error_step + 1}, got {identified_step} ({'correct' if step_correct else 'wrong'})")
        diagnostics_parts.append(f"Keywords: {keyword_hits}/{len(self._keywords)} matched")
        diagnostics_parts.append(f"Fix: {fix_score:.1f}")

        return VerifierResult(
            correct=correct,
            score=total_score,
            partial_credit=partial,
            diagnostics=" | ".join(diagnostics_parts),
        )

    def _check_fix(self, fix_text: str) -> float:
        """Check if the proposed fix is correct."""
        if not fix_text:
            return 0.0

        fix_lower = fix_text.lower().strip()

        # Check for "REMOVE" — valid for waste traces
        if self._trace_type == "waste":
            if "remove" in fix_lower or "delete" in fix_lower or "skip" in fix_lower:
                return 1.0
            # Also accept if they say the step is unnecessary
            if "unnecessary" in fix_lower or "redundant" in fix_lower or "waste" in fix_lower:
                return 0.8
            return 0.3  # partial credit for attempting

        # For math/logic traces, check if the fix contains the correct values
        correct_lower = self._correct_step_text.lower()

        # Extract numbers from the correct step
        correct_numbers = set(re.findall(r"-?\d+\.?\d*", correct_lower))
        fix_numbers = set(re.findall(r"-?\d+\.?\d*", fix_lower))

        if correct_numbers and fix_numbers:
            overlap = len(correct_numbers & fix_numbers) / len(correct_numbers)
            return min(1.0, overlap)

        # Fallback: check for keyword overlap
        correct_words = set(correct_lower.split())
        fix_words = set(fix_lower.split())
        if correct_words:
            overlap = len(correct_words & fix_words) / len(correct_words)
            return min(1.0, overlap * 1.5)  # scale up since word overlap is stricter

        return 0.3  # partial credit for attempting


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TrajectoryDoctorEnv(BaseReasoningEnv):
    """
    TrajectoryDoctor environment: diagnose the first wrong step in a trace.

    The model receives a flawed reasoning trace and must identify the wrong
    step, explain why, and provide a fix. Reward = weighted combination of
    step identification, explanation quality, and fix correctness.
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
            problem_generator = trajectory_doctor_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return TrajectoryDoctorVerifier(
            error_step=problem.metadata["error_step"],
            explanation=problem.metadata["explanation"],
            keywords=problem.metadata["keywords"],
            correct_step_text=problem.metadata["correct_step_text"],
            trace_type=problem.metadata["trace_type"],
        )

    def _check_format(self, response: str) -> float:
        """Check for the WRONG STEP / REASON / FIX format."""
        has_step = bool(re.search(r"WRONG\s+STEP:\s*\d+", response, re.IGNORECASE))
        has_reason = bool(re.search(r"REASON:\s*.+", response, re.IGNORECASE))
        has_fix = bool(re.search(r"FIX:\s*.+", response, re.IGNORECASE))

        if has_step and has_reason and has_fix:
            return 1.0
        elif has_step and has_reason:
            return 0.5
        elif has_step:
            return 0.2
        return 0.0

    def _extract_answer(self, response: str) -> str:
        """Return the full response — the verifier parses it."""
        return response
