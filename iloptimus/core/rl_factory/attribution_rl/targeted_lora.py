"""Targeted LoRA: use attribution to select which layers to train.

Only the layers that attribution says handle the target capability get LoRA adapters.
This is the '5% of params' — not random layers, but the RIGHT layers.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from iloptimus.core.rl_factory.attribution_rl.attribution import AttributionResult


@dataclass
class TargetedLoRAConfig:
    """Configuration for attribution-targeted LoRA adapters."""

    target_domain: str
    param_budget_fraction: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    selected_layers: list = field(default_factory=list)
    selected_modules: list = field(default_factory=list)
    total_trainable_params: int = 0
    total_model_params: int = 0


class TargetedLoRASelector:
    """Select which layers/modules get LoRA adapters based on attribution."""

    def __init__(self, attribution_result: AttributionResult):
        self.attribution_result = attribution_result

    def select_layers(
        self,
        domain: str,
        budget_fraction: float = 0.05,
        params_per_layer: Optional[int] = None,
    ) -> TargetedLoRAConfig:
        """Select layers to train for a target domain within a param budget."""
        ar = self.attribution_result
        if params_per_layer is None:
            params_per_layer = ar.params_per_layer
        total_params = ar.total_params

        # 1. layer importance scores for the target domain
        importance = ar.layer_importance.get(domain)
        if importance is None:
            # fall back to average across all domains
            all_imp = list(ar.layer_importance.values())
            if all_imp:
                importance = [float(np.mean([a[i] for a in all_imp]))
                              for i in range(ar.n_layers)]
            else:
                importance = [0.0] * ar.n_layers
        importance = [float(x) for x in importance]

        # 2. sort layers by importance (descending)
        order = sorted(range(ar.n_layers), key=lambda i: importance[i], reverse=True)

        # 3. how many layers fit in budget
        if params_per_layer > 0:
            n_keep = max(1, int(budget_fraction * total_params / params_per_layer))
        else:
            n_keep = max(1, int(budget_fraction * ar.n_layers))
        n_keep = min(n_keep, ar.n_layers)

        # 4. decide prefix vs suffix: compare mass of late vs early halves
        half = ar.n_layers // 2
        late_mass = sum(importance[half:])
        early_mass = sum(importance[:half])
        prefer_suffix = late_mass >= early_mass

        # 5. select top-n_keep layers, prefer contiguous block for residual coherence
        top_set = set(order[:n_keep])
        # find the contiguous block covering the most top layers
        selected = self._best_contiguous_block(top_set, ar.n_layers, n_keep, prefer_suffix)

        # 6. determine module types from param_knowledge_map
        # union of modules that are relevant across selected layers
        module_set = set()
        for L in selected:
            module_set.update(self.select_modules_for_layer(L, domain))
        selected_modules = [m for m in ar.weight_matrix_sensitivity.keys() if m in module_set]
        if not selected_modules:
            selected_modules = list(ar.weight_matrix_sensitivity.keys())

        config = TargetedLoRAConfig(
            target_domain=domain,
            param_budget_fraction=budget_fraction,
            selected_layers=sorted(selected),
            selected_modules=selected_modules,
            total_model_params=total_params,
        )
        return config

    @staticmethod
    def _best_contiguous_block(top_set, n_layers, n_keep, prefer_suffix):
        """Pick a contiguous block of n_keep layers maximizing overlap with top_set."""
        if n_keep >= n_layers:
            return list(range(n_layers))
        best_start = 0
        best_overlap = -1
        for start in range(0, n_layers - n_keep + 1):
            block = set(range(start, start + n_keep))
            overlap = len(block & top_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = start
            elif overlap == best_overlap:
                # tie-break: prefer suffix or prefix
                if prefer_suffix and start > best_start:
                    best_start = start
                elif not prefer_suffix and start < best_start:
                    best_start = start
        return list(range(best_start, best_start + n_keep))

    def select_modules_for_layer(self, layer_idx: int, domain: str) -> list:
        """Determine which matrix types in a layer are relevant for a domain."""
        ar = self.attribution_result
        layer_map = ar.param_knowledge_map.get(layer_idx, {})
        modules = []
        for mt, classification in layer_map.items():
            if classification == "dead":
                continue
            if classification.startswith("domain-specific:"):
                spec_dom = classification.split(":", 1)[1]
                if spec_dom != domain:
                    continue
                modules.append(mt)
            elif classification == "shared":
                modules.append(mt)
        return modules

    def estimate_trainable_params(
        self, config: TargetedLoRAConfig, params_per_module: dict
    ) -> int:
        """Estimate total trainable params for the LoRA config.

        LoRA adds two low-rank matrices per module: A (in x rank) and B (rank x out).
        Trainable params per module = 2 * rank * module_dim (using in dim as proxy).
        """
        total = 0
        for L in config.selected_layers:
            for mt in config.selected_modules:
                base = params_per_module.get(mt, 0)
                if base <= 0:
                    continue
                # module_dim ~ sqrt(base) as a rough in-dim proxy
                in_dim = int(np.sqrt(base))
                total += 2 * config.lora_rank * in_dim
        config.total_trainable_params = total
        return total

    def get_coverage_report(self, config: TargetedLoRAConfig) -> dict:
        """Return stats about the LoRA coverage."""
        ar = self.attribution_result
        modules_per_layer = {}
        for L in config.selected_layers:
            modules_per_layer[L] = self.select_modules_for_layer(L, config.target_domain)
        total_params = ar.total_params if config.total_model_params == 0 else config.total_model_params
        fraction = 0.0
        if total_params > 0:
            fraction = config.total_trainable_params / total_params
        return {
            "fraction_of_params": fraction,
            "n_layers_selected": len(config.selected_layers),
            "n_layers_total": ar.n_layers,
            "modules_per_layer": modules_per_layer,
            "domain": config.target_domain,
        }
