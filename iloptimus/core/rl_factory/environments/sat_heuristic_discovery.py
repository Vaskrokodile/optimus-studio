"""
SatHeuristicDiscovery: Discover SAT solving heuristics.

Environment concept:
  The model is given a small 3-SAT problem and must propose a heuristic for
  which variable to assign first. A good heuristic picks a variable that
  appears in many clauses (maximizing constraint propagation).

  This trains the model to:
    1. Understand SAT problem structure
    2. Identify high-impact variables
    3. Articulate a reasonable heuristic strategy

Verification:
  - The HEURISTIC line must contain a description.
  - The VARIABLE line must name a valid variable from the problem.
  - The chosen variable is scored by how many clauses it appears in,
    relative to the best variable (the one in the most clauses).

Reward design:
  format_score * 0.3 + variable_validity * 0.3 + quality * 0.4
  where quality = clauses_with_var / clauses_with_best_var
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _generate_3sat(rng: random.Random, n_vars: int, n_clauses: int) -> tuple[list[list[str]], list[str]]:
    """Generate a random 3-SAT problem.

    Returns (clauses, variables) where each clause is a list of 3 literals
    like ["x1", "-x2", "x3"] (negation indicated by a leading '-').
    """
    variables = [f"x{i+1}" for i in range(n_vars)]
    clauses: list[list[str]] = []
    for _ in range(n_clauses):
        chosen = rng.sample(variables, 3)
        literals = []
        for v in chosen:
            if rng.random() < 0.5:
                literals.append(f"-{v}")
            else:
                literals.append(v)
        clauses.append(literals)
    return clauses, variables


def _count_clause_appearances(clauses: list[list[str]], var: str) -> int:
    """Count how many clauses contain the given variable (in either polarity)."""
    count = 0
    for clause in clauses:
        for lit in clause:
            v = lit.lstrip("-")
            if v == var:
                count += 1
                break
    return count


def _best_variable(clauses: list[list[str]], variables: list[str]) -> str:
    """Return the variable that appears in the most clauses."""
    best = variables[0]
    best_count = -1
    for v in variables:
        c = _count_clause_appearances(clauses, v)
        if c > best_count:
            best_count = c
            best = v
    return best


def sat_heuristic_discovery_generator(seed: int) -> Problem:
    """Generate a small 3-SAT problem for heuristic discovery."""
    rng = random.Random(seed)
    n_vars = rng.randint(3, 5)
    n_clauses = rng.randint(5, 10)
    clauses, variables = _generate_3sat(rng, n_vars, n_clauses)
    best_var = _best_variable(clauses, variables)

    # Format clauses for display
    clause_strs = [" OR ".join(c) for c in clauses]
    clauses_text = "\n".join(f"  ({c})" for c in clause_strs)

    prompt = (
        f"You are solving a 3-SAT problem with {n_vars} variables and "
        f"{n_clauses} clauses.\n\n"
        f"Variables: {', '.join(variables)}\n\n"
        f"Clauses:\n{clauses_text}\n\n"
        f"Propose a heuristic for which variable to assign first to maximize\n"
        f"constraint propagation.\n"
        f"Format:\n"
        f"HEURISTIC: <brief description of your strategy>\n"
        f"VARIABLE: <variable name, e.g. x1>"
    )

    return Problem(
        id=f"sat_heuristic_discovery_{seed}",
        prompt=prompt,
        difficulty=min(1.0, 0.3 + n_clauses * 0.05),
        metadata={
            "clauses": clauses,
            "variables": variables,
            "best_variable": best_var,
            "n_vars": n_vars,
            "n_clauses": n_clauses,
        },
        token_budget=384,
        source="sat_heuristic_discovery_generator",
    )


sat_heuristic_discovery_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class SatHeuristicDiscoveryVerifier(Verifier):
    """Verify a SAT heuristic proposal.

    reward = format_score * 0.3 + variable_validity * 0.3 + quality * 0.4
    """

    def __init__(self, variables: list[str], clauses: list[list[str]], best_variable: str):
        super().__init__()
        self._variables = set(variables)
        self._clauses = clauses
        self._best_variable = best_variable
        self._best_count = _count_clause_appearances(clauses, best_variable)

    def verify(self, response: str) -> VerifierResult:
        # Parse HEURISTIC line
        heur_match = re.search(
            r"HEURISTIC\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        heuristic_text = heur_match.group(1).strip() if heur_match else ""

        # Parse VARIABLE line
        var_match = re.search(
            r"VARIABLE\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        given_var = var_match.group(1).strip() if var_match else ""

        # Format score
        format_score = 0.0
        if heuristic_text:
            format_score += 0.5
        if given_var:
            format_score += 0.5

        # Variable validity
        var_valid = 0.0
        chosen_var = None
        if given_var:
            # Normalize: strip whitespace, leading '-', etc.
            norm = given_var.strip().lstrip("-").strip()
            if norm in self._variables:
                var_valid = 1.0
                chosen_var = norm
            else:
                # Try to find a variable name within the text
                for v in self._variables:
                    if v in given_var:
                        var_valid = 1.0
                        chosen_var = v
                        break

        # Quality: how good is the chosen variable?
        quality = 0.0
        if chosen_var is not None:
            count = _count_clause_appearances(self._clauses, chosen_var)
            if self._best_count > 0:
                quality = count / self._best_count

        score = format_score * 0.3 + var_valid * 0.3 + quality * 0.4
        correct = (
            format_score >= 1.0
            and var_valid >= 1.0
            and chosen_var == self._best_variable
        )

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "format_score": format_score,
                "variable_validity": var_valid,
                "quality": quality,
                "chosen_variable": chosen_var or "",
                "best_variable": self._best_variable,
            },
            diagnostics=(
                f"format={format_score:.2f} "
                f"var_valid={var_valid:.2f} "
                f"quality={quality:.2f} "
                f"chosen={chosen_var} best={self._best_variable}"
            ),
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SatHeuristicDiscoveryEnv(BatchEnvBase):
    """SatHeuristicDiscovery: propose a variable-assignment heuristic for SAT.

    Batch-aware: N parallel proposals; reward = best score across the batch.
    """

    __test__ = False

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = sat_heuristic_discovery_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return SatHeuristicDiscoveryVerifier(
            variables=problem.metadata["variables"],
            clauses=problem.metadata["clauses"],
            best_variable=problem.metadata["best_variable"],
        )

    def _check_format(self, response: str) -> float:
        has_heur = bool(re.search(r"HEURISTIC\s*:", response, re.IGNORECASE))
        has_var = bool(re.search(r"VARIABLE\s*:", response, re.IGNORECASE))
        if has_heur and has_var:
            return 1.0
        if has_heur or has_var:
            return 0.5
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        return {
            "best_score": best_score,
            "any_correct": any(corrects),
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": best_score,
            "correct": any(corrects),
            "diagnostics": (
                f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores)/len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r"VARIABLE\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
