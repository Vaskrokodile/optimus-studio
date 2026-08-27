"""
Anti-pattern detection for reasoning traces.

Identifies specific wasteful reasoning patterns that frontier models exhibit:
  - Hedging / backtracking ("oh wait", "but actually", "hmm, let me reconsider")
  - Buzzword padding ("leverage", "utilize", "delve into", "tapestry")
  - Redundant restating of the problem
  - Empty filler ("let's think about this", "this is an interesting problem")
  - Self-congratulation ("great, so we've established that")
  - Excessive qualification ("it might be the case that", "one could argue")

Each anti-pattern is detected via regex + heuristics and contributes a penalty
to the reward signal. The detector is designed to be fast (O(n) in text length)
and to produce structured findings that can be used for analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class AntiPatternType(Enum):
    """Categories of wasteful reasoning patterns."""
    BACKTRACK = "backtrack"             # "oh wait", "but actually", "hmm, let me reconsider"
    BUZZWORD = "buzzword"               # "leverage", "utilize", "delve", "tapestry"
    RESTATE = "restate"                 # repeating the problem statement
    FILLER = "filler"                   # "let's think about this", "interesting problem"
    SELF_CONGRATULATE = "self_congratulate"  # "great, so we've established"
    OVER_QUALIFY = "over_qualify"       # "it might be the case that", "one could argue"
    REPEAT_STEP = "repeat_step"         # repeating a previous step verbatim/near-verbatim
    VAGUE_REFERENCE = "vague_reference" # "as mentioned above", "as we discussed"
    EMPTY_CORRECTION = "empty_correction"  # "wait, no" without explaining what was wrong
    # New patterns for agentic coding waste
    OVER_VERIFY = "over_verify"         # "let me double-check", "let me verify once more"
    NEEDLESS_CONTEXT = "needless_context"  # "for context, ...", "to give some background"
    HEDGE_COMMIT = "hedge_commit"       # "I think this might work", "this should probably be correct"
    PRE_EXPLANATION = "pre_explanation"  # "before we start, let me explain", "first, some context"
    TOOL_OVERJUSTIFY = "tool_overjustify"  # "I'm going to use X because Y Z..." (long justification for simple tool call)


@dataclass
class AntiPatternHit:
    """A single detected anti-pattern instance."""
    pattern_type: AntiPatternType
    text: str           # the matched text
    start: int          # character offset in the full trace
    end: int
    penalty: float      # reward penalty for this hit


@dataclass
class AntiPatternReport:
    """Full analysis of a reasoning trace."""
    hits: list[AntiPatternHit] = field(default_factory=list)
    total_penalty: float = 0.0
    counts_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def total_hits(self) -> int:
        return len(self.hits)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each pattern: (compiled_regex, AntiPatternType, per-hit penalty)
# Penalties are calibrated so that a trace with many hits can significantly
# reduce the reward, but a single accidental hit won't zero it out.

_PATTERNS: list[tuple[re.Pattern, AntiPatternType, float]] = [
    # --- Backtracking ---
    # "oh wait", "wait, no", "actually, let me reconsider", "hmm, but"
    (re.compile(r"\b(?:oh\s+wait|wait[,\s]+no|actually[,\s]+let\s+me\s+reconsider|hmm[,\s]+but|hold\s+on[,\s]+let\s+me|no[,\s]+wait[,\s]+actually)\b", re.IGNORECASE),
     AntiPatternType.BACKTRACK, 0.8),
    (re.compile(r"\b(?:but\s+actually|on\s+second\s+thought|let\s+me\s+reconsider|scratch\s+that)\b", re.IGNORECASE),
     AntiPatternType.BACKTRACK, 0.6),
    (re.compile(r"\b(?:I\s+was\s+wrong|that's\s+not\s+right|that's\s+incorrect|my\s+mistake)\b", re.IGNORECASE),
     AntiPatternType.BACKTRACK, 0.4),

    # --- Empty corrections (backtracking without explaining what was wrong) ---
    (re.compile(r"\b(?:wait[,\s]+no[,\s]+that's?\s+(?:not\s+)?(?:right|correct|it)\s*[.!])\b", re.IGNORECASE),
     AntiPatternType.EMPTY_CORRECTION, 0.5),

    # --- Buzzword padding ---
    (re.compile(r"\b(?:leverage[ds]?|utilizing?|delve\s+into|tapestry|navigating\s+the\s+(?:complex|intricate)|seamlessly|holistically|paradigm\s+shift|game[\s-]changer|synergistic|robust\s+solution|elegant\s+solution)\b", re.IGNORECASE),
     AntiPatternType.BUZZWORD, 0.3),
    (re.compile(r"\b(?:it's\s+worth\s+noting\s+that|it\s+should\s+be\s+noted|needless\s+to\s+say|at\s+the\s+end\s+of\s+the\s+day|the\s+bottom\s+line\s+is)\b", re.IGNORECASE),
     AntiPatternType.BUZZWORD, 0.2),

    # --- Filler / empty throat-clearing ---
    (re.compile(r"\b(?:let'?s\s+think\s+about\s+this|this\s+is\s+an\s+interesting\s+(?:problem|question)|let\s+me\s+start\s+by\s+saying|first[,\s]+let\s+me\s+understand)\b", re.IGNORECASE),
     AntiPatternType.FILLER, 0.2),
    (re.compile(r"\b(?:alright[,\s]+so|okay[,\s]+so[,\s]+let'?s|so[,\s]+um[,\s]+|well[,\s]+let'?s\s+see)\b", re.IGNORECASE),
     AntiPatternType.FILLER, 0.1),

    # --- Self-congratulation ---
    (re.compile(r"\b(?:great[,\s]+so\s+we'?ve\s+(?:established|shown|demonstrated)|perfect[,\s]+now|excellent[,\s]+so|so\s+we\s+successfully)\b", re.IGNORECASE),
     AntiPatternType.SELF_CONGRATULATE, 0.2),

    # --- Over-qualification ---
    (re.compile(r"\b(?:it\s+might\s+be\s+the\s+case\s+that|one\s+could\s+argue\s+that|it\s+is\s+worth\s+considering\s+that|there\s+exists\s+a\s+possibility\s+that)\b", re.IGNORECASE),
     AntiPatternType.OVER_QUALIFY, 0.2),

    # --- Vague references (instead of being specific) ---
    (re.compile(r"\b(?:as\s+mentioned\s+(?:above|earlier|before)|as\s+(?:we|I)\s+(?:discussed|noted|saw)\s+(?:above|earlier)|as\s+previously\s+stated)\b", re.IGNORECASE),
     AntiPatternType.VAGUE_REFERENCE, 0.15),

    # --- Over-verification (redundant checking of already-correct results) ---
    (re.compile(r"\b(?:let\s+me\s+(?:double[\s-]check|verify\s+(?:once\s+more|again|this))|let'?s\s+verify\s+(?:that|this)|just\s+to\s+be\s+safe|to\s+make\s+sure)\b", re.IGNORECASE),
     AntiPatternType.OVER_VERIFY, 0.4),

    # --- Needless context (providing background nobody asked for) ---
    (re.compile(r"\b(?:for\s+(?:some\s+)?context[,.]|to\s+give\s+(?:some\s+)?background|before\s+we\s+(?:dive|start|begin)[,\s]+(?:let\s+me\s+)?(?:explain|provide)|a\s+bit\s+of\s+background)\b", re.IGNORECASE),
     AntiPatternType.NEEDLESS_CONTEXT, 0.3),

    # --- Hedging on commits (uncertainty when submitting) ---
    (re.compile(r"\b(?:I\s+think\s+this\s+(?:might|should)\s+work|this\s+should\s+probably\s+be\s+correct|I\s+believe\s+this\s+is\s+(?:right|correct)|hopefully\s+this\s+(?:fixes|works))\b", re.IGNORECASE),
     AntiPatternType.HEDGE_COMMIT, 0.25),

    # --- Pre-explanation (explaining before doing) ---
    (re.compile(r"\b(?:before\s+we\s+start[,\s]+let\s+me\s+(?:explain|clarify)|first[,\s]+some\s+(?:context|background)|let\s+me\s+first\s+explain\s+(?:the\s+)?(?:approach|strategy))\b", re.IGNORECASE),
     AntiPatternType.PRE_EXPLANATION, 0.3),

    # --- Tool over-justification (long-winded reasoning for simple tool calls) ---
    (re.compile(r"\b(?:going\s+to\s+(?:use|call|run)\s+\w+\s+because\s+(?:I\s+(?:need|want)|it\s+will\s+(?:help|allow)))\b", re.IGNORECASE),
     AntiPatternType.TOOL_OVERJUSTIFY, 0.2),

    # --- Restating the problem (heuristic: if the first 200 chars repeat >60% of the prompt) ---
    # This is handled separately in the analyze() method, not via regex.
]


class AntiPatternDetector:
    """
    Detects wasteful reasoning patterns in text traces.

    Usage:
        detector = AntiPatternDetector()
        report = detector.analyze(trace_text)
        penalty = report.total_penalty
    """

    def __init__(self, max_total_penalty: float = 5.0):
        """
        Args:
            max_total_penalty: Cap on the total anti-pattern penalty for a single
                trace, so that a very long trace with many small hits doesn't
                completely zero out the reward.
        """
        self._patterns = _PATTERNS
        self._max_penalty = max_total_penalty

    def analyze(self, text: str, prompt: str = "") -> AntiPatternReport:
        """
        Analyze *text* for anti-patterns.

        Args:
            text: The reasoning trace to analyze.
            prompt: The original problem prompt, used for restatement detection.

        Returns:
            An AntiPatternReport with all hits and the total penalty.
        """
        report = AntiPatternReport()

        # Regex-based patterns
        for regex, ptype, per_hit_penalty in self._patterns:
            for match in regex.finditer(text):
                hit = AntiPatternHit(
                    pattern_type=ptype,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    penalty=per_hit_penalty,
                )
                report.hits.append(hit)
                report.total_penalty += per_hit_penalty
                key = ptype.value
                report.counts_by_type[key] = report.counts_by_type.get(key, 0) + 1

        # Restatement detection: check if the trace repeats >40% of the prompt
        if prompt and len(prompt) > 20:
            restatement_penalty = self._detect_restatement(text, prompt)
            if restatement_penalty > 0:
                hit = AntiPatternHit(
                    pattern_type=AntiPatternType.RESTATE,
                    text="(restates problem)",
                    start=0,
                    end=min(len(text), 200),
                    penalty=restatement_penalty,
                )
                report.hits.append(hit)
                report.total_penalty += restatement_penalty
                report.counts_by_type["restate"] = 1

        # Repeat-step detection: look for near-duplicate sentences
        repeat_penalty = self._detect_repeated_steps(text)
        if repeat_penalty > 0:
            hit = AntiPatternHit(
                pattern_type=AntiPatternType.REPEAT_STEP,
                text="(repeated steps)",
                start=0,
                end=0,
                penalty=repeat_penalty,
            )
            report.hits.append(hit)
            report.total_penalty += repeat_penalty
            report.counts_by_type["repeat_step"] = int(repeat_penalty / 0.3)

        # Cap the total penalty
        report.total_penalty = min(report.total_penalty, self._max_penalty)

        return report

    def _detect_restatement(self, text: str, prompt: str) -> float:
        """
        Detect if the trace restates the problem.

        Uses a simple n-gram overlap heuristic: extract 4-grams from the prompt
        and check how many appear in the first 500 chars of the trace.
        """
        # Normalize both texts
        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s.lower().strip())

        norm_prompt = normalize(prompt)
        norm_text = normalize(text[:500])

        if len(norm_prompt) < 20:
            return 0.0

        # Extract 4-grams from prompt
        prompt_words = norm_prompt.split()
        if len(prompt_words) < 4:
            return 0.0

        prompt_4grams = set()
        for i in range(len(prompt_words) - 3):
            prompt_4grams.add(" ".join(prompt_words[i:i + 4]))

        # Extract 4-grams from the beginning of the trace
        text_words = norm_text.split()
        if len(text_words) < 4:
            return 0.0

        text_4grams = set()
        for i in range(min(len(text_words), 100) - 3):
            text_4grams.add(" ".join(text_words[i:i + 4]))

        if not prompt_4grams:
            return 0.0

        overlap = len(prompt_4grams & text_4grams) / len(prompt_4grams)

        # Penalize if >40% of prompt 4-grams appear in the trace opening
        if overlap > 0.4:
            return min(1.0, overlap * 1.5)
        return 0.0

    def _detect_repeated_steps(self, text: str) -> float:
        """
        Detect near-duplicate sentences in the trace.

        Splits into sentences, hashes normalized versions, and penalizes
        when the same normalized sentence appears multiple times.
        """
        # Split into sentences (rough)
        sentences = re.split(r"[\.!\n]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if len(sentences) < 3:
            return 0.0

        # Normalize: lowercase, remove articles and common words, collapse whitespace
        def normalize_sentence(s: str) -> str:
            s = s.lower()
            s = re.sub(r"\b(?:the|a|an|is|are|was|were|this|that|we|I|so|then)\b", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        seen: dict[str, int] = {}
        duplicates = 0
        for sent in sentences:
            norm = normalize_sentence(sent)
            if len(norm) < 5:
                continue
            if norm in seen:
                duplicates += 1
            else:
                seen[norm] = 1

        if duplicates > 0:
            return min(2.0, duplicates * 0.3)
        return 0.0
