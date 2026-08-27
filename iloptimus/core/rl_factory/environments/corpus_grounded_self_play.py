"""
CorpusGroundedSelfPlay: Corpus-grounded self-play to prevent reward hacking.

Environment concept:
  The model is given a small document corpus (3-5 sentences) and a
  question answerable from it. The model must answer using ONLY the
  corpus — no hallucination. This implements the *corpus-grounded
  self-play* paradigm: by requiring a citation to a specific corpus
  sentence, we prevent the model from reward-hacking (inventing answers
  that happen to match the ground truth but aren't grounded).

  The generator builds a corpus, picks a supporting sentence, and forms
  a question whose answer is derivable from that sentence. Verification
  checks BOTH (a) the answer matches the ground truth AND (b) the cited
  sentence actually supports the answer (string overlap check).

Verification:
  1. Answer correctness: does the answer match the ground truth?
  2. Citation grounding: does the cited sentence exist in the corpus
     AND does it have sufficient string overlap with the answer?

Reward:
  reward = answer_match * 0.5 + citation_grounded * 0.5
  correct = answer_match AND citation_grounded

Format:
  ANSWER: <answer>
  CITATION: <corpus sentence>
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Corpus templates: each has a corpus, a question, an answer, and the
# index of the supporting sentence.
# ---------------------------------------------------------------------------


_CG_CORPORA: list[dict[str, Any]] = [
    {
        "corpus": [
            "The Eiffel Tower was completed in 1889 and stands in Paris.",
            "It was designed by Gustave Eiffel for the 1889 World's Fair.",
            "The tower is 330 meters tall and made of wrought iron.",
            "Millions of tourists visit the Eiffel Tower every year.",
        ],
        "question": "In what year was the Eiffel Tower completed?",
        "answer": "1889",
        "supporting_sentence_idx": 0,
    },
    {
        "corpus": [
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "It primarily occurs in the chloroplasts of plant cells.",
            "The process uses carbon dioxide and water to produce glucose and oxygen.",
            "Chlorophyll is the green pigment responsible for absorbing light.",
        ],
        "question": "What gas is produced as a byproduct of photosynthesis?",
        "answer": "oxygen",
        "supporting_sentence_idx": 2,
    },
    {
        "corpus": [
            "The Great Wall of China stretches over 13,000 miles across northern China.",
            "Construction began in the 7th century BCE during the Zhou dynasty.",
            "The wall was built primarily for defense against nomadic invasions.",
            "It is one of the most recognizable landmarks in the world.",
        ],
        "question": "Approximately how many miles long is the Great Wall of China?",
        "answer": "13000",
        "supporting_sentence_idx": 0,
    },
    {
        "corpus": [
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "The boiling point decreases at higher altitudes due to lower pressure.",
            "Freezing occurs at 0 degrees Celsius under the same conditions.",
            "Water is composed of two hydrogen atoms and one oxygen atom.",
        ],
        "question": "At what temperature does water freeze at standard pressure?",
        "answer": "0",
        "supporting_sentence_idx": 2,
    },
    {
        "corpus": [
            "Jupiter is the largest planet in our solar system.",
            "It is a gas giant with a prominent feature called the Great Red Spot.",
            "Jupiter has at least 79 known moons orbiting it.",
            "The planet completes one orbit around the Sun every 12 years.",
        ],
        "question": "How many known moons does Jupiter have?",
        "answer": "79",
        "supporting_sentence_idx": 2,
    },
    {
        "corpus": [
            "The Amazon River is the second longest river in the world.",
            "It flows through South America, primarily through Brazil.",
            "The Amazon rainforest produces about 20% of the world's oxygen.",
            "The river basin is home to millions of species of plants and animals.",
        ],
        "question": "What percentage of the world's oxygen does the Amazon rainforest produce?",
        "answer": "20",
        "supporting_sentence_idx": 2,
    },
    {
        "corpus": [
            "The speed of light in a vacuum is approximately 300000 kilometers per second.",
            "Light travels slower when passing through media like water or glass.",
            "The speed of light is a fundamental constant in physics, denoted by c.",
            "Albert Einstein's theory of relativity is based on the constancy of c.",
        ],
        "question": "What is the approximate speed of light in a vacuum (in km/s)?",
        "answer": "300000",
        "supporting_sentence_idx": 0,
    },
    {
        "corpus": [
            "Honey never spoils due to its low water content and acidic pH.",
            "Archaeologists have found edible honey in ancient Egyptian tombs.",
            "Bees produce honey by regurgitating nectar and evaporating water from it.",
            "A single bee produces about one-twelfth of a teaspoon of honey in its lifetime.",
        ],
        "question": "Why does honey never spoil?",
        "answer": "low water content and acidic pH",
        "supporting_sentence_idx": 0,
    },
]


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _overlap(a: str, b: str) -> float:
    """Word-overlap score between two strings (Jaccard-like)."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def corpus_grounded_self_play_generator(seed: int) -> Problem:
    """Generate a CorpusGroundedSelfPlay problem.

    Selects a corpus template and builds a prompt presenting the corpus
    and a question. The model must answer AND cite the supporting
    sentence.

    Args:
        seed: Random seed for reproducible problem generation.

    Returns:
        A Problem with corpus, question, answer, and supporting index.
    """
    rng = random.Random(seed)
    template = rng.choice(_CG_CORPORA)

    corpus = template["corpus"]
    question = template["question"]
    answer = template["answer"]
    support_idx = template["supporting_sentence_idx"]

    corpus_text = "\n".join(
        f"[{i}] {s}" for i, s in enumerate(corpus)
    )

    prompt = (
        f"Corpus:\n{corpus_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer using ONLY the corpus. You must cite the supporting "
        f"corpus sentence.\n\n"
        f"Format:\n"
        f"ANSWER: <answer>\n"
        f"CITATION: <the exact corpus sentence that supports your answer>"
    )

    return Problem(
        id=f"corpus_grounded_{seed}_{rng.randint(0, 9999)}",
        prompt=prompt,
        difficulty=0.35,
        metadata={
            "type": "corpus_grounded_self_play",
            "corpus": corpus,
            "question": question,
            "answer": answer,
            "supporting_sentence_idx": support_idx,
        },
        token_budget=400,
        source="corpus_grounded_self_play_generator",
    )


corpus_grounded_self_play_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CorpusGroundedVerifier(Verifier):
    """Verify a corpus-grounded answer with citation.

    Args:
        corpus: List of corpus sentences.
        answer: Ground-truth answer.
        supporting_sentence_idx: Index of the supporting sentence.
    """

    def __init__(
        self,
        corpus: list[str],
        answer: str,
        supporting_sentence_idx: int,
    ):
        super().__init__()
        self._corpus = corpus
        self._answer = answer
        self._support_idx = supporting_sentence_idx

    def verify(self, response: str) -> VerifierResult:
        given_answer = self._parse_answer(response)
        given_citation = self._parse_citation(response)

        # 1. Answer correctness
        answer_match = self._answer_matches(given_answer)

        # 2. Citation grounding: does the citation exist in the corpus
        #    AND overlap with the answer?
        citation_grounded = self._citation_is_grounded(given_citation, given_answer)

        score = answer_match * 0.5 + citation_grounded * 0.5
        correct = answer_match >= 0.5 and citation_grounded >= 0.5

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "answer_match": answer_match,
                "citation_grounded": citation_grounded,
            },
            diagnostics=(
                f"Answer={answer_match:.2f} Citation={citation_grounded:.2f}"
            ),
            metadata={
                "given_answer": given_answer,
                "given_citation": given_citation,
            },
        )

    def _parse_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_citation(self, response: str) -> str:
        match = re.search(
            r"CITATION\s*:\s*(.+?)(?:\n[A-Z]+:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _answer_matches(self, given: str) -> float:
        if not given:
            return 0.0
        g = _normalize(given)
        a = _normalize(self._answer)
        if g == a:
            return 1.0
        # Partial: answer appears as a substring or number match
        if a in g:
            return 0.8
        # Number match
        a_nums = re.findall(r"\d+", a)
        g_nums = re.findall(r"\d+", g)
        if a_nums and g_nums and a_nums == g_nums:
            return 0.7
        if a_nums and g_nums and a_nums[-1] in g_nums:
            return 0.5
        return 0.0

    def _citation_is_grounded(self, citation: str, answer: str) -> float:
        if not citation:
            return 0.0
        # Check if the citation matches any corpus sentence
        best_corpus_overlap = 0.0
        for sent in self._corpus:
            ov = _overlap(citation, sent)
            if ov > best_corpus_overlap:
                best_corpus_overlap = ov
        # Check if the citation overlaps with the answer
        answer_overlap = _overlap(citation, answer) if answer else 0.0
        # Grounding requires the citation to be in the corpus
        if best_corpus_overlap < 0.3:
            return 0.0
        # Combine: citation is in corpus AND relates to answer
        return min(1.0, best_corpus_overlap * 0.7 + answer_overlap * 0.3)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CorpusGroundedSelfPlayEnv(BatchEnvBase):
    """CorpusGroundedSelfPlay environment: grounded answers with citations.

    Batch-aware: N parallel attempts; reward = best attempt.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator: Optional[Any] = None,
        reward_config: Optional[Any] = None,
        anti_pattern_detector: Optional[Any] = None,
        render_mode: Optional[str] = None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = corpus_grounded_self_play_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        meta = problem.metadata
        return CorpusGroundedVerifier(
            corpus=meta["corpus"],
            answer=meta["answer"],
            supporting_sentence_idx=meta["supporting_sentence_idx"],
        )

    def _check_format(self, response: str) -> float:
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        has_citation = bool(re.search(r"CITATION\s*:", response, re.IGNORECASE))
        if has_answer and has_citation:
            return 1.0
        if has_answer or has_citation:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "mean_score": mean_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "reward": best_score,
            "correct": any_correct and best_score >= 0.5,
            "diagnostics": (
                f"Best={best_score:.2f} Mean={mean_score:.2f} "
                f"Correct={sum(corrects)}/{len(per_sample)}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
