"""Dataset-mode skills: anti-slop, anti-laziness, and curation guidance.

These are compact, read-only prompt injections — not executable skills.  They
are adapted from the best external skill definitions (Leonxlnx/unlazy,
adewale/anti-slop-writing, wshobson/agents dataset-curation, datacrux
contamination) into a unified block that fits within a small model's context
budget.

The skills are always active in dataset mode.  They are injected after the
system prompt and before the template exemplars.
"""

from __future__ import annotations

# Think tags
_TO = chr(60) + "think" + chr(62)
_TC = chr(60) + "/think" + chr(62)

# ---------------------------------------------------------------------------
# Anti-laziness skill (adapted from Leonxlnx/unlazy — Depth Tree + gates)
# ---------------------------------------------------------------------------

_ANTI_LAZINESS = """\
## Anti-laziness skill

Do not take shortcuts. Every row must be complete and correct.

- DEPTH: Before writing a row, decompose the problem into its actual steps.
  If a problem needs 5 steps to solve, write 5 steps — not 2 steps and
  "the rest follows similarly."
- GATES: Each row must pass these self-checks before you output it:
  1. Is the answer actually correct? (re-derive it)
  2. Is the reasoning trace complete? (no skipped steps)
  3. Is the input non-trivial? (would a model need to reason to solve it?)
  4. Is this row different from the others? (not a paraphrase)
- EVIDENCE: Do not claim a result without showing the work that produces it.
  "By symmetry, the answer is X" is lazy. Show the symmetry argument.
"""

# ---------------------------------------------------------------------------
# Anti-slop skill (adapted from adewale/anti-slop-writing + ch040602/anti-ai-slop)
# ---------------------------------------------------------------------------

_ANTI_SLOP = """\
## Anti-slop skill

Generic filler degrades training data quality. Eliminate it.

- A punchy line without a named mechanism still reads as slop. "This is a
  classic problem" is slop. "This is a classic problem because it requires
  simultaneous satisfaction of two conflicting constraints" is not.
- Forbidden phrases: "interesting problem", "many ways to approach",
  "let's think about this", "as we can see", "in conclusion",
  "this problem tests", "the key insight is" (without stating the insight).
- Every sentence in the <think> block must carry information: a calculation,
  a logical deduction, a constraint identification, or a verification.
- Restating the problem is slop. Engaging with it is not.
"""

# ---------------------------------------------------------------------------
# Dataset curation skill (adapted from wshobson/agents dataset-curation)
# ---------------------------------------------------------------------------

_CURATION = """\
## Dataset curation skill

- FORMAT: Every row must have <think>…</think> then the answer. No exceptions.
  The think block and answer are separated by </think>. The answer comes after.
- DIVERSITY: Vary problem structure, not just surface features. Changing
  numbers in the same template is not diversity. Use different problem
  setups, different mathematical tools, different code patterns.
- COMPLETENESS: A row with a truncated think block is worse than no row.
  If you cannot finish the reasoning within the token budget, output a
  shorter problem that you CAN finish.
- PROVENANCE: Each row is independent. Do not reference "the previous
  problem" or "as mentioned above."
"""

# ---------------------------------------------------------------------------
# Contamination defense skill (adapted from stef41/datacrux)
# ---------------------------------------------------------------------------

_CONTAMINATION = """\
## Contamination defense skill

- Never reproduce a known benchmark problem verbatim or with minor edits.
  If you recognize a problem as being from AIME, AMC, GSM8K, HumanEval,
  MMLU, or any other standard benchmark, do not output it.
- N-gram overlap with benchmark problems contaminates the training set and
  inflates evaluation scores. If your problem shares a distinctive phrase
  with a known benchmark problem, rephrase it entirely.
- When in doubt, generate a novel problem on the same concept rather than
  risking contamination.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SKILL_BLOCKS = [
    ("anti-laziness", _ANTI_LAZINESS),
    ("anti-slop", _ANTI_SLOP),
    ("curation", _CURATION),
    ("contamination", _CONTAMINATION),
]


def dataset_skills_prompt(max_chars: int = 2_500) -> str:
    """Build the combined dataset skills prompt block.

    Returns all four skill blocks concatenated, truncated to ``max_chars``.
    The total is ~800 tokens — compact enough to fit alongside the system
    prompt and template exemplars in a small model's context.
    """
    parts: list[str] = []
    remaining = max_chars
    for _name, block in _SKILL_BLOCKS:
        if len(block) > remaining:
            parts.append(block[:remaining])
            break
        parts.append(block)
        remaining -= len(block)
    return "\n".join(parts).strip()


def list_dataset_skills() -> list[dict[str, str]]:
    """Return metadata about available dataset skills."""
    return [
        {"id": name, "chars": len(block)}
        for name, block in _SKILL_BLOCKS
    ]
