"""IL scoring for tool-sql-analyst-v1 (shared engine)."""

from il_toolcalling_core.engine import score_trajectory


def score(task, response: str) -> tuple[float, dict]:
    return score_trajectory(task, response)


__all__ = ["score"]
