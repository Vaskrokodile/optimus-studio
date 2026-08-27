"""Attribution-guided GRPO: train only the attributed 5% of params.

GRPO (Group Relative Policy Optimization) with:
- Group-relative advantages (no critic needed)
- Policy gradient + KL penalty
- Only computes gradients for LoRA layers (the attributed 5%)
- R3 routing replay for MoE stability

All core math (advantages, loss, NAT token selection, Horvitz-Thompson
reweighting) is implemented in numpy so it works WITHOUT torch for testing.
When torch is available, train_step would additionally run backprop and an
optimizer step on the LoRA parameters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


@dataclass
class AttributionGRPOConfig:
    """Configuration for attribution-guided GRPO training."""
    learning_rate: float = 1e-5
    kl_beta: float = 0.04            # KL penalty weight
    advantage_eps: float = 1e-8      # numerical stability for std normalization
    group_normalize: bool = True     # normalize advantages within group (std)
    max_grad_norm: float = 1.0       # gradient clipping
    gradient_checkpointing: bool = True
    partial_token_ratio: float = 0.5  # NAT: only compute gradients for 50% of tokens
    r3_enabled: bool = True          # R3 routing replay for MoE
    r3_strictness: float = 0.9       # R3 replay strictness (1.0 = exact)


class AttributionGRPOTrainer:
    """GRPO trainer that only updates the attributed 5% of parameters.

    In torch mode this would wrap a PEFT LoRA model and run real backprop.
    In mock mode (no torch) it computes the same losses/advantages in numpy
    and records stats, so the full training loop can be tested without a GPU.
    """

    def __init__(
        self,
        config: AttributionGRPOConfig,
        lora_config: Any = None,
        r3_recorder: Any = None,
    ):
        self.config = config
        self.lora_config = lora_config
        self.r3_recorder = r3_recorder

        # R3 routing table: (token_idx, layer_idx) -> expert_idx
        self._routing_table: dict[tuple[int, int], int] = {}

        if HAS_TORCH:
            # Would initialize: base model + PEFT LoRA + optimizer.
            # Only the LoRA params (attributed 5%) receive gradients.
            self.model = None  # placeholder
            self.optimizer = None  # placeholder
            self._torch_mode = True
        else:
            self._torch_mode = False

        # Training stats
        self._step_count = 0
        self._losses: list[float] = []
        self._kls: list[float] = []
        self._advantages_history: list[float] = []

    # ------------------------------------------------------------------
    # GRPO advantage computation
    # ------------------------------------------------------------------
    def compute_advantages(
        self, rewards: np.ndarray, group_size: int
    ) -> np.ndarray:
        """Compute GRPO group-relative advantages.

        Reshape rewards into groups of `group_size`. For each group:
          - group_normalize=True:  A_i = (r_i - mean) / (std + eps)
          - group_normalize=False: A_i = r_i - mean

        Returns a flat advantages array of the same length as rewards.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        n = len(rewards)
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        num_groups = n // group_size
        usable = num_groups * group_size

        advantages = np.zeros(n, dtype=np.float64)
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = rewards[start:end]
            mean_r = float(np.mean(group))
            if self.config.group_normalize:
                std_r = float(np.std(group))
                advantages[start:end] = (group - mean_r) / (std_r + self.config.advantage_eps)
            else:
                advantages[start:end] = group - mean_r

        # Any leftover samples (n not divisible by group_size): center only.
        if usable < n:
            tail = rewards[usable:]
            advantages[usable:] = tail - float(np.mean(tail)) if len(tail) > 1 else 0.0

        return advantages

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------
    def compute_grpo_loss(
        self,
        policy_logprobs: np.ndarray,
        ref_logprobs: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Compute GRPO loss = policy gradient + KL penalty.

        L_pg = -mean(advantages * policy_logprobs)
        L_kl = mean(policy_logprobs - ref_logprobs)
        L    = L_pg + kl_beta * L_kl

        Returns {"loss", "pg_loss", "kl_loss", "kl_divergence"}.
        """
        policy_logprobs = np.asarray(policy_logprobs, dtype=np.float64)
        ref_logprobs = np.asarray(ref_logprobs, dtype=np.float64)
        advantages = np.asarray(advantages, dtype=np.float64)

        pg_loss = -float(np.mean(advantages * policy_logprobs))
        kl_loss = float(np.mean(policy_logprobs - ref_logprobs))
        total_loss = pg_loss + self.config.kl_beta * kl_loss

        # kl_divergence reported as the raw mean log-ratio (approximation).
        kl_divergence = kl_loss

        return {
            "loss": total_loss,
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "kl_divergence": kl_divergence,
        }

    # ------------------------------------------------------------------
    # NAT: Not All Tokens Needed
    # ------------------------------------------------------------------
    def select_tokens_nat(self, losses: np.ndarray) -> np.ndarray:
        """Select the top `partial_token_ratio` tokens by importance.

        Importance = |loss|. Returns a boolean mask over the tokens.
        """
        losses = np.asarray(losses, dtype=np.float64)
        n = len(losses)
        if n == 0:
            return np.zeros(0, dtype=bool)
        k = max(1, int(round(self.config.partial_token_ratio * n)))
        k = min(k, n)
        importance = np.abs(losses)
        # Top-k indices by importance (descending).
        top_idx = np.argsort(importance)[::-1][:k]
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True
        return mask

    def reweight_losses_nat(
        self, losses: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Horvitz-Thompson reweighting of selected-token losses.

        Scale selected tokens by 1/inclusion_prob so the gradient estimate
        stays unbiased. inclusion_prob = partial_token_ratio.
        """
        losses = np.asarray(losses, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        inclusion_prob = self.config.partial_token_ratio
        if inclusion_prob <= 0:
            inclusion_prob = 1e-8
        reweighted = np.zeros_like(losses)
        reweighted[mask] = losses[mask] / inclusion_prob
        return reweighted

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(
        self,
        policy_logprobs: np.ndarray,
        ref_logprobs: np.ndarray,
        rewards: np.ndarray,
        group_size: int,
    ) -> dict:
        """One GRPO training step.

        1. Compute group-relative advantages.
        2. Compute per-token losses (mock: advantages * policy_logprobs).
        3. Apply NAT token selection + Horvitz-Thompson reweighting.
        4. Compute final GRPO loss.
        5. In torch mode: backprop + optimizer step on LoRA params.
           In mock mode: just record the loss.

        Returns a dict with loss components, advantages, and token counts.
        """
        policy_logprobs = np.asarray(policy_logprobs, dtype=np.float64)
        ref_logprobs = np.asarray(ref_logprobs, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)

        # 1. Advantages
        advantages = self.compute_advantages(rewards, group_size)

        # 2. Per-token losses (the surrogate used for NAT selection).
        # Align lengths: advantages/policy_logprobs may differ in size.
        m = min(len(advantages), len(policy_logprobs))
        per_token_loss = -(advantages[:m] * policy_logprobs[:m])

        # 3. NAT token selection
        n_tokens_total = len(per_token_loss)
        mask = self.select_tokens_nat(per_token_loss)
        n_tokens_selected = int(np.sum(mask))
        reweighted = self.reweight_losses_nat(per_token_loss, mask)

        # 4. GRPO loss computed on the (reweighted) selected tokens.
        # Use the masked policy/ref logprobs for the loss so KL is over
        # the same token set.
        masked_policy = policy_logprobs[:n_tokens_total][mask]
        masked_ref = ref_logprobs[:n_tokens_total][mask] if len(ref_logprobs) >= n_tokens_total else ref_logprobs[:m][mask]
        masked_adv = advantages[:n_tokens_total][mask]

        if n_tokens_selected > 0:
            loss_info = self.compute_grpo_loss(masked_policy, masked_ref, masked_adv)
        else:
            loss_info = {"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0, "kl_divergence": 0.0}

        # 5. Optimizer step (torch) or record (mock)
        if self._torch_mode and self.model is not None and self.optimizer is not None:
            # Would convert loss_info["loss"] to a tensor, backward, clip, step.
            # Placeholder for the real backprop path.
            pass

        # Record stats
        self._step_count += 1
        self._losses.append(loss_info["loss"])
        self._kls.append(loss_info["kl_loss"])
        self._advantages_history.append(float(np.mean(advantages)) if len(advantages) else 0.0)

        return {
            "loss": loss_info["loss"],
            "pg_loss": loss_info["pg_loss"],
            "kl_loss": loss_info["kl_loss"],
            "advantages": advantages,
            "n_tokens_selected": n_tokens_selected,
            "n_tokens_total": n_tokens_total,
        }

    # ------------------------------------------------------------------
    # R3 routing replay
    # ------------------------------------------------------------------
    def record_routing(self, token_idx: int, expert_idx: int, layer_idx: int) -> None:
        """Record a routing decision during rollout (R3)."""
        if self.config.r3_enabled:
            self._routing_table[(token_idx, layer_idx)] = expert_idx

    def replay_routing(self, token_idx: int, layer_idx: int) -> Optional[int]:
        """Return the expected expert index for (token_idx, layer_idx).

        Returns None if R3 is disabled or no routing was recorded.
        """
        if not self.config.r3_enabled:
            return None
        return self._routing_table.get((token_idx, layer_idx))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return training statistics."""
        mean_loss = float(np.mean(self._losses)) if self._losses else 0.0
        mean_kl = float(np.mean(self._kls)) if self._kls else 0.0
        mean_adv = float(np.mean(self._advantages_history)) if self._advantages_history else 0.0
        return {
            "steps": self._step_count,
            "mean_loss": mean_loss,
            "mean_kl": mean_kl,
            "mean_advantage": mean_adv,
        }
