"""
LogicPuzzleSuite: Diverse logic puzzles.

Environment concept:
  The model is given a small logic puzzle (4x4 Sudoku, Einstein-style ordering
  puzzle, or knight/knave puzzle) and must provide the solution. The verifier
  checks that the solution satisfies all puzzle constraints.

  This trains the model to:
    1. Understand diverse constraint-satisfaction structures
    2. Apply deductive reasoning
    3. Produce solutions in the expected format

Verification:
  - Sudoku: check the grid satisfies row, column, and box constraints.
  - Einstein: check the assignment satisfies all ordering constraints.
  - Knight/Knave: check the answer is consistent with the puzzle logic.

Reward design:
  format_score * 0.3 + correctness * 0.7
  where correctness is 1.0 if all constraints are satisfied, else partial.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Sudoku 4x4 generator
# ---------------------------------------------------------------------------


def _generate_sudoku4(rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
    """Generate a valid 4x4 Sudoku solution and a puzzle (with some blanks).

    Returns (puzzle, solution) where 0 indicates a blank cell.
    """
    # Base valid 4x4 solution pattern
    base = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]
    # Apply random permutations of symbols and row/column swaps within bands
    perm = list(range(1, 5))
    rng.shuffle(perm)
    mapping = {i + 1: perm[i] for i in range(4)}
    solution = [[mapping[base[r][c]] for c in range(4)] for r in range(4)]

    # Swap rows within bands (rows 0-1 and rows 2-3)
    if rng.random() < 0.5:
        solution[0], solution[1] = solution[1], solution[0]
    if rng.random() < 0.5:
        solution[2], solution[3] = solution[3], solution[2]
    # Swap columns within stacks
    if rng.random() < 0.5:
        for r in range(4):
            solution[r][0], solution[r][1] = solution[r][1], solution[r][0]
    if rng.random() < 0.5:
        for r in range(4):
            solution[r][2], solution[r][3] = solution[r][3], solution[r][2]

    # Create puzzle by blanking some cells
    puzzle = [row[:] for row in solution]
    n_blanks = rng.randint(4, 8)
    cells = [(r, c) for r in range(4) for c in range(4)]
    rng.shuffle(cells)
    for r, c in cells[:n_blanks]:
        puzzle[r][c] = 0
    return puzzle, solution


def _format_sudoku(puzzle: list[list[int]]) -> str:
    lines = []
    for row in puzzle:
        lines.append(" ".join(str(v) if v != 0 else "_" for v in row))
    return "\n".join(lines)


def _parse_sudoku_grid(text: str, size: int = 4) -> Optional[list[list[int]]]:
    """Parse a Sudoku grid from text. Returns a size x size grid or None."""
    # Extract numbers and underscores/blanks
    rows = []
    for line in text.strip().split("\n"):
        # Remove formatting characters
        cleaned = re.sub(r"[|\-+]", "", line).strip()
        if not cleaned:
            continue
        tokens = re.split(r"[\s,]+", cleaned)
        nums = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if t in ("_", ".", "0", "x", "X", "*"):
                nums.append(0)
            else:
                try:
                    nums.append(int(t))
                except ValueError:
                    continue
        if nums:
            rows.append(nums)
    # Filter to rows that have the right number of entries
    rows = [r for r in rows if len(r) == size]
    if len(rows) == size:
        return rows
    # Try to extract exactly size*size numbers
    all_nums = []
    for r in rows:
        all_nums.extend(r)
    if len(all_nums) >= size * size:
        grid = []
        for i in range(size):
            grid.append(all_nums[i * size:(i + 1) * size])
        return grid
    return None


def _check_sudoku(grid: list[list[int]], size: int = 4) -> bool:
    """Check if a Sudoku grid is valid (no duplicates in rows/cols/boxes)."""
    valid_vals = set(range(1, size + 1))
    box_size = int(size ** 0.5)
    if box_size * box_size != size:
        box_size = 2  # default for 4x4
    # Check rows
    for row in grid:
        vals = [v for v in row if v != 0]
        if len(vals) != len(set(vals)):
            return False
        if any(v not in valid_vals for v in vals):
            return False
        if len(vals) < size:
            return False  # must be complete
    # Check columns
    for c in range(size):
        col = [grid[r][c] for r in range(size)]
        vals = [v for v in col if v != 0]
        if len(vals) != len(set(vals)):
            return False
        if len(vals) < size:
            return False
    # Check boxes
    for br in range(0, size, box_size):
        for bc in range(0, size, box_size):
            box = []
            for r in range(br, br + box_size):
                for c in range(bc, bc + box_size):
                    box.append(grid[r][c])
            vals = [v for v in box if v != 0]
            if len(vals) != len(set(vals)):
                return False
            if len(vals) < size:
                return False
    return True


# ---------------------------------------------------------------------------
# Einstein-style ordering puzzle generator
# ---------------------------------------------------------------------------


def _generate_einstein(rng: random.Random) -> tuple[str, list[str], list[str]]:
    """Generate a simple Einstein-style ordering puzzle.

    Returns (prompt, constraints, solution) where solution is the ordered list.
    """
    n = 4
    # Use house positions 1..n with attributes
    names = ["Alice", "Bob", "Carol", "Dave"]
    rng.shuffle(names)
    solution = names[:]  # solution[i] = person at position i+1

    # Generate constraints from the solution
    constraints = []
    # Position constraints: "X lives in position P"
    # Adjacency: "X lives next to Y"
    # Relative: "X lives to the left of Y"
    constraint_types = ["position", "adjacent", "left_of"]
    for _ in range(n + 1):
        ctype = rng.choice(constraint_types)
        if ctype == "position":
            i = rng.randint(0, n - 1)
            constraints.append(f"{solution[i]} lives in position {i + 1}")
        elif ctype == "adjacent" and n > 1:
            i = rng.randint(0, n - 2)
            constraints.append(f"{solution[i]} lives next to {solution[i + 1]}")
        else:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            constraints.append(f"{solution[i]} lives to the left of {solution[j]}")

    # Deduplicate while keeping order
    seen = set()
    unique_constraints = []
    for c in constraints:
        if c not in seen:
            seen.add(c)
            unique_constraints.append(c)

    prompt = (
        f"Four people (Alice, Bob, Carol, Dave) live in positions 1 through 4.\n\n"
        f"Constraints:\n"
        + "\n".join(f"  - {c}" for c in unique_constraints)
        + "\n\nDetermine who lives in each position.\n"
        f"Format: ANSWER: <name1>, <name2>, <name3>, <name4> "
        f"(position 1 through 4 in order)"
    )
    return prompt, unique_constraints, solution


def _check_einstein(given: list[str], solution: list[str]) -> bool:
    """Check if the given ordering matches the solution."""
    if len(given) != len(solution):
        return False
    g = [x.strip().lower() for x in given]
    s = [x.strip().lower() for x in solution]
    return g == s


# ---------------------------------------------------------------------------
# Knight/Knave puzzle generator
# ---------------------------------------------------------------------------


_KK_PUZZLES = [
    {
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie.\n"
            "You meet two people, A and B.\n"
            "A says: 'B is a knave.'\n"
            "B says: 'A and I are both knights.'\n"
            "Determine who is a knight and who is a knave.\n"
            "Format: ANSWER: A is <knight/knave>, B is <knight/knave>"
        ),
        "solution": "A is knight, B is knave",
        "constraints": ["A is knight", "B is knave"],
    },
    {
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie.\n"
            "You meet two people, A and B.\n"
            "A says: 'We are both knaves.'\n"
            "Determine who is a knight and who is a knave.\n"
            "Format: ANSWER: A is <knight/knave>, B is <knight/knave>"
        ),
        "solution": "A is knave, B is knight",
        "constraints": ["A is knave", "B is knight"],
    },
    {
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie.\n"
            "You meet two people, A and B.\n"
            "A says: 'At least one of us is a knave.'\n"
            "Determine who is a knight and who is a knave.\n"
            "Format: ANSWER: A is <knight/knave>, B is <knight/knave>"
        ),
        "solution": "A is knight, B is knave",
        "constraints": ["A is knight", "B is knave"],
    },
    {
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie.\n"
            "You meet two people, A and B.\n"
            "B says: 'A is a knight.'\n"
            "A says nothing.\n"
            "Determine who is a knight and who is a knave (if determinable).\n"
            "Format: ANSWER: A is <knight/knave>, B is <knight/knave>"
        ),
        "solution": "A is knight, B is knight",
        "constraints": ["A is knight", "B is knight"],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def logic_puzzle_suite_generator(seed: int) -> Problem:
    """Generate a logic puzzle (Sudoku, Einstein, or Knight/Knave)."""
    rng = random.Random(seed)
    puzzle_type = rng.choice(["sudoku", "einstein", "knight_knave"])

    if puzzle_type == "sudoku":
        puzzle, solution = _generate_sudoku4(rng)
        prompt = (
            f"Solve the following 4x4 Sudoku puzzle.\n"
            f"Each row, column, and 2x2 box must contain 1, 2, 3, 4 exactly once.\n\n"
            f"Puzzle:\n{_format_sudoku(puzzle)}\n\n"
            f"Provide the completed grid.\n"
            f"Format: ANSWER: <4 rows of 4 numbers, space or comma separated>"
        )
        return Problem(
            id=f"logic_puzzle_sudoku_{seed}",
            prompt=prompt,
            difficulty=0.4 + rng.random() * 0.2,
            metadata={
                "puzzle_type": "sudoku",
                "puzzle_data": puzzle,
                "solution": solution,
                "constraints": ["rows_unique", "cols_unique", "boxes_unique"],
            },
            token_budget=384,
            source="logic_puzzle_suite_generator",
        )

    elif puzzle_type == "einstein":
        prompt, constraints, solution = _generate_einstein(rng)
        return Problem(
            id=f"logic_puzzle_einstein_{seed}",
            prompt=prompt,
            difficulty=0.4 + rng.random() * 0.2,
            metadata={
                "puzzle_type": "einstein",
                "puzzle_data": constraints,
                "solution": solution,
                "constraints": constraints,
            },
            token_budget=384,
            source="logic_puzzle_suite_generator",
        )

    else:  # knight_knave
        data = rng.choice(_KK_PUZZLES)
        return Problem(
            id=f"logic_puzzle_kk_{seed}",
            prompt=data["prompt"],
            difficulty=0.3 + rng.random() * 0.2,
            metadata={
                "puzzle_type": "knight_knave",
                "puzzle_data": data["prompt"],
                "solution": data["solution"],
                "constraints": data["constraints"],
            },
            token_budget=256,
            source="logic_puzzle_suite_generator",
        )


logic_puzzle_suite_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class LogicPuzzleSuiteVerifier(Verifier):
    """Verify a logic puzzle solution.

    reward = format_score * 0.3 + correctness * 0.7
    """

    def __init__(self, puzzle_type: str, solution: Any, constraints: list[str], puzzle_data: Any):
        super().__init__()
        self._puzzle_type = puzzle_type
        self._solution = solution
        self._constraints = constraints
        self._puzzle_data = puzzle_data

    def verify(self, response: str) -> VerifierResult:
        # Parse ANSWER line
        ans_match = re.search(
            r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )
        format_score = 1.0 if ans_match else 0.0
        given_answer = ans_match.group(1).strip() if ans_match else ""

        if self._puzzle_type == "sudoku":
            correctness = self._check_sudoku_answer(given_answer, response)
        elif self._puzzle_type == "einstein":
            correctness = self._check_einstein_answer(given_answer)
        else:  # knight_knave
            correctness = self._check_kk_answer(given_answer)

        score = format_score * 0.3 + correctness * 0.7
        correct = format_score >= 1.0 and correctness >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "format_score": format_score,
                "correctness": correctness,
                "puzzle_type": self._puzzle_type,
            },
            diagnostics=(
                f"type={self._puzzle_type} "
                f"format={format_score:.2f} "
                f"correctness={correctness:.2f} "
                f"given={given_answer[:60]!r}"
            ),
        )

    def _check_sudoku_answer(self, given: str, full_response: str) -> float:
        """Check a Sudoku solution. Try the ANSWER line first, then the full response."""
        # Try parsing from the ANSWER line
        grid = _parse_sudoku_grid(given) if given else None
        # If that fails, try the full response (minus the ANSWER: prefix)
        if grid is None:
            grid = _parse_sudoku_grid(full_response)
        if grid is None:
            return 0.0
        # Check validity
        if _check_sudoku(grid, size=4):
            return 1.0
        # Partial credit: count correct cells vs solution
        solution = self._solution
        if len(grid) == 4 and all(len(r) == 4 for r in grid):
            correct_cells = 0
            total = 16
            for r in range(4):
                for c in range(4):
                    if grid[r][c] == solution[r][c]:
                        correct_cells += 1
            return correct_cells / total
        return 0.0

    def _check_einstein_answer(self, given: str) -> float:
        """Check an Einstein puzzle ordering solution."""
        if not given:
            return 0.0
        # Parse names from the answer
        parts = re.split(r"[,\s]+", given.strip())
        names = [p for p in parts if p.strip()]
        # Filter to known names (case-insensitive)
        known = ["alice", "bob", "carol", "dave"]
        given_names = [p.lower() for p in names if p.lower() in known]
        if len(given_names) != 4:
            return 0.0
        if _check_einstein(given_names, self._solution):
            return 1.0
        # Partial: count correct positions
        sol_lower = [s.lower() for s in self._solution]
        correct_pos = sum(
            1 for i in range(4) if i < len(given_names) and given_names[i] == sol_lower[i]
        )
        return correct_pos / 4

    def _check_kk_answer(self, given: str) -> float:
        """Check a knight/knave puzzle solution."""
        if not given:
            return 0.0
        given_lower = given.lower().replace(" ", "")
        expected_lower = self._solution.lower().replace(" ", "")
        if given_lower == expected_lower:
            return 1.0
        # Partial: check each person's classification
        # Parse "A is knight, B is knave" style
        given_pairs = self._parse_kk_pairs(given)
        expected_pairs = self._parse_kk_pairs(self._solution)
        if given_pairs and expected_pairs:
            matched = 0
            for person, role in expected_pairs.items():
                if given_pairs.get(person) == role:
                    matched += 1
            return matched / len(expected_pairs) if expected_pairs else 0.0
        return 0.0

    @staticmethod
    def _parse_kk_pairs(text: str) -> dict[str, str]:
        """Parse 'A is knight, B is knave' into {'a': 'knight', 'b': 'knave'}."""
        pairs = {}
        for m in re.finditer(r"([AB])\s+is\s+(knight|knave)", text, re.IGNORECASE):
            pairs[m.group(1).lower()] = m.group(2).lower()
        return pairs


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LogicPuzzleSuiteEnv(BatchEnvBase):
    """LogicPuzzleSuite: diverse logic puzzles (Sudoku, Einstein, Knight/Knave).

    Batch-aware: N parallel solutions; reward = best score across the batch.
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
            problem_generator = logic_puzzle_suite_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return LogicPuzzleSuiteVerifier(
            puzzle_type=problem.metadata["puzzle_type"],
            solution=problem.metadata["solution"],
            constraints=problem.metadata["constraints"],
            puzzle_data=problem.metadata["puzzle_data"],
        )

    def _check_format(self, response: str) -> float:
        has_answer = bool(re.search(r"ANSWER\s*:", response, re.IGNORECASE))
        return 1.0 if has_answer else 0.0

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
        match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return match.group(1).strip() if match else response
