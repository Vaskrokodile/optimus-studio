"""
Reward computation for rl-factory environments.

Implements a multi-component reward function that combines:
  1. Task correctness (from the verifier)
  2. Token efficiency (penalize verbosity, reward conciseness)
  3. Anti-pattern penalties (backtracking, buzzwords, filler, etc.)
  4. Difficulty-aware scaling (harder problems → weaker length penalty)

Design informed by the research:
  - L1 (arXiv 2503.04697): length-controlled policy optimization
  - T2T (arXiv 2602.04265): thickening-to-thinning dual-phase
  - LASER-D (ICLR 2026): difficulty-aware length penalties
  - ExpThink (arXiv 2605.07501): three-tier reward (concise-correct, verbose-correct, wrong)
  - "How Much Thinking is Enough?": length-agnostic rewards cause structural overthinking
  - DRPO (arXiv 2510.04474): decouple length penalty from incorrect rollouts
  - CoT-Pass@K (arXiv 2506.14245): require reasoning path AND answer correctness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from iloptimus.core.rl_factory.core.anti_patterns import AntiPatternDetector, AntiPatternReport
from iloptimus.core.rl_factory.core.token_budget import TokenBudget, estimate_tokens
from iloptimus.core.rl_factory.core.verifier import VerifierResult


@dataclass
class RewardComponents:
    """
    Individual components of the reward signal.

    All components are in [0, 1] except penalties which are in [-max_penalty, 0].
    The final reward is a weighted combination of these.
    """
    correctness: float = 0.0          # from verifier, [0, 1]
    efficiency: float = 0.0           # token efficiency bonus, [0, 1]
    anti_pattern_penalty: float = 0.0  # negative, [-max, 0]
    format_bonus: float = 0.0         # bonus for correct output format, [0, 1]
    difficulty_bonus: float = 0.0     # bonus for solving harder problems, [0, 1]

    # Metadata for analysis
    token_count: int = 0
    budget_used_fraction: float = 0.0
    anti_pattern_report: Optional[AntiPatternReport] = None
    verifier_result: Optional[VerifierResult] = None


@dataclass
class RewardConfig:
    """
    Configuration for the reward function.

    Weights determine how much each component contributes to the final reward.
    The defaults are calibrated based on the research literature.
    """
    # Weight of correctness in the final reward (dominant signal)
    w_correct: float = 1.0

    # Weight of token efficiency (conciseness bonus)
    w_efficiency: float = 0.3

    # Weight of anti-pattern penalty (applied as negative)
    w_anti_pattern: float = 0.5

    # Weight of format bonus
    w_format: float = 0.1

    # Weight of difficulty bonus
    w_difficulty: float = 0.2

    # Efficiency curve parameters
    # The efficiency bonus is a function of how few tokens were used relative
    # to the budget. We use a sigmoid-like curve centered at 50% budget usage.
    efficiency_midpoint: float = 0.5   # fraction of budget at which efficiency bonus = 0.5
    efficiency_steepness: float = 4.0  # how sharply the bonus drops off

    # Difficulty-aware length penalty scaling
    # For easy problems (difficulty=0), length penalty is full strength.
    # For hard problems (difficulty=1), length penalty is reduced.
    # This follows LASER-D's finding that uniform penalties over-compress hard problems.
    difficulty_penalty_scale: float = 0.5  # at max difficulty, penalty is reduced by this factor

    # Three-tier reward (ExpThink-style):
    #   concise + correct → full credit
    #   verbose + correct → discounted credit
    #   wrong → zero
    verbose_threshold: float = 0.7   # fraction of budget above which "verbose" kicks in
    verbose_discount: float = 0.7    # multiplier for verbose-but-correct responses

    # Minimum reward floor (so that a correct answer always gets > 0)
    correct_floor: float = 0.3

    # Whether to apply T2T-style dual-phase logic
    # On incorrect: no length penalty (encourage exploration)
    # On correct: apply length penalty (encourage conciseness)
    dual_phase: bool = True


def compute_efficiency_bonus(
    budget: TokenBudget,
    config: RewardConfig,
    difficulty: float = 0.0,
) -> float:
    """
    Compute the token efficiency bonus.

    Returns a value in [0, 1] where 1 = very concise, 0 = used entire budget.
    The bonus is difficulty-aware: harder problems get a gentler penalty for
    using more tokens.

    Args:
        budget: The token budget tracker for this episode.
        config: Reward configuration.
        difficulty: Problem difficulty in [0, 1].
    """
    frac_used = budget.fraction_used

    # Sigmoid: bonus = 1 / (1 + exp(steepness * (frac_used - midpoint)))
    # At frac_used=0 → bonus ≈ 1, at frac_used=midpoint → bonus = 0.5,
    # at frac_used=1 → bonus ≈ 0
    import math
    raw_bonus = 1.0 / (1.0 + math.exp(
        config.efficiency_steepness * (frac_used - config.efficiency_midpoint)
    ))

    # Difficulty-aware scaling: reduce the penalty for using more tokens on harder problems
    # The "penalty" portion is (1 - raw_bonus). Scale it by difficulty.
    penalty = 1.0 - raw_bonus
    difficulty_factor = 1.0 - difficulty * config.difficulty_penalty_scale
    scaled_penalty = penalty * difficulty_factor

    return 1.0 - scaled_penalty


def compute_combined_reward(
    components: RewardComponents,
    config: RewardConfig,
    difficulty: float = 0.0,
) -> tuple[float, dict]:
    """
    Compute the final scalar reward from all components.

    Args:
        components: The individual reward components.
        config: Reward configuration.
        difficulty: Problem difficulty in [0, 1].

    Returns:
        (reward, info_dict) where reward is a scalar and info_dict contains
        the breakdown for logging/analysis.
    """
    correct = components.correctness > 0
    is_verbose = components.budget_used_fraction > config.verbose_threshold

    # --- T2T Dual-phase logic ---
    if config.dual_phase:
        if not correct:
            # Thickening phase: incorrect → no efficiency penalty, no anti-pattern penalty
            # Reward is purely based on correctness (which is 0 or partial)
            reward = components.correctness * config.w_correct
            # Small format bonus even if wrong (encourages structured attempts)
            reward += components.format_bonus * config.w_format * 0.5
            reward = max(0.0, reward)
        else:
            # Thinning phase: correct → apply efficiency and anti-pattern penalties
            base = components.correctness * config.w_correct

            # Apply correct floor
            base = max(base, config.correct_floor)

            # Three-tier: discount verbose-but-correct
            if is_verbose:
                base *= config.verbose_discount

            # Efficiency bonus
            eff = components.efficiency * config.w_efficiency

            # Anti-pattern penalty (negative)
            ap_penalty = components.anti_pattern_penalty * config.w_anti_pattern

            # Format and difficulty bonuses
            fmt = components.format_bonus * config.w_format
            diff = components.difficulty_bonus * config.w_difficulty

            reward = base + eff + fmt + diff + ap_penalty
            reward = max(0.0, reward)
    else:
        # Non-dual-phase: simple weighted sum
        base = components.correctness * config.w_correct
        if correct and is_verbose:
            base *= config.verbose_discount
        if correct:
            base = max(base, config.correct_floor)

        eff = components.efficiency * config.w_efficiency
        ap_penalty = components.anti_pattern_penalty * config.w_anti_pattern
        fmt = components.format_bonus * config.w_format
        diff = components.difficulty_bonus * config.w_difficulty

        reward = base + eff + fmt + diff + ap_penalty
        reward = max(0.0, reward)

    # Clamp to [0, 2] — allow >1 for exceptional concise+correct+hard solutions
    reward = max(0.0, min(2.0, reward))

    info = {
        "reward": reward,
        "correct": correct,
        "correctness": components.correctness,
        "efficiency": components.efficiency,
        "anti_pattern_penalty": components.anti_pattern_penalty,
        "format_bonus": components.format_bonus,
        "difficulty_bonus": components.difficulty_bonus,
        "token_count": components.token_count,
        "budget_used_fraction": components.budget_used_fraction,
        "is_verbose": is_verbose,
        "difficulty": difficulty,
        "anti_pattern_hits": components.anti_pattern_report.total_hits if components.anti_pattern_report else 0,
        "anti_pattern_counts": components.anti_pattern_report.counts_by_type if components.anti_pattern_report else {},
    }

    return reward, info
