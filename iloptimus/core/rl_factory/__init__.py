"""
rl-factory: High-efficiency reasoning RL environments for frontier model training.

Five environments designed to train models toward concise, correct, non-redundant reasoning:

  1. ProofGolf       — shortest valid proof wins (mathematical reasoning compression)
  2. SurgicalPatch   — minimal bug fix under a context budget (surgical debugging)
  3. CalibratedQA    — buy hints vs commit (calibrated confidence under uncertainty)
  4. RefactorArena   — produce equivalent but more concise code (code compression)
  5. TrajectoryDoctor— diagnose the first wrong step in a flawed reasoning trace

All environments implement the Gymnasium API and share a common reward framework
that penalizes token waste, backtracking anti-patterns, and buzzword padding.
"""

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv
from iloptimus.core.rl_factory.core.reward import RewardComponents, compute_combined_reward
from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector
from iloptimus.core.rl_factory.core.token_budget import TokenBudget
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult

__version__ = "1.0.0"

__all__ = [
    "BaseReasoningEnv",
    "RewardComponents",
    "compute_combined_reward",
    "AntiPatternDetector",
    "TokenBudget",
    "Verifier",
    "VerifierResult",
]
