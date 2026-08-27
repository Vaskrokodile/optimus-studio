"""Attribution engine: determine which weights/layers handle which capabilities.

Adapted from slm-microscope's attribution_v2.py. Works with mock model traces
for testing (no torch required) and real model traces when available.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Callable


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (numpy)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    # Use float64 but clamp to avoid overflow with large real-model activations
    a = np.clip(a, -1e6, 1e6)
    b = np.clip(b, -1e6, 1e6)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy (natural log) of an attention distribution."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TraceData:
    """Captured model internals for one prompt."""

    residuals: np.ndarray  # (n_layers+1, seq_len, d_model)
    attention: np.ndarray  # (n_layers, n_heads, seq_len, seq_len)
    mlp_acts: np.ndarray   # (n_layers, seq_len, d_mlp)
    logits: np.ndarray     # (seq_len, vocab_size)
    domain: str


@dataclass
class AttributionResult:
    """Output of attribution analysis."""

    domains: list
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    layer_importance: dict  # domain -> list[float] per layer
    weight_matrix_sensitivity: dict  # matrix_type -> (n_layers, n_domains)
    attention_head_spec: dict  # domain -> (n_layers, n_heads)
    param_knowledge_map: dict  # layer -> {matrix_type: classification}
    total_params: int
    params_per_layer: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AttributionEngine:
    """Determine which layers/weights handle which capabilities."""

    DEFAULT_MATRIX_TYPES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    DEFAULT_DOMAINS = ["english", "reasoning", "coding"]

    # matrix types grouped by which proxy trace signal to use
    MLP_MATRIX_TYPES = {"gate_proj", "up_proj", "down_proj"}
    ATTN_MATRIX_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj"}

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_mlp: int,
        matrix_types: Optional[list] = None,
        domains: Optional[list] = None,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.matrix_types = list(matrix_types) if matrix_types else list(self.DEFAULT_MATRIX_TYPES)
        self.domains = list(domains) if domains else list(self.DEFAULT_DOMAINS)

    # -- attention head specialization -------------------------------------

    @staticmethod
    def _attention_features(attn: np.ndarray) -> np.ndarray:
        """Compute 5-D feature vector for one attention pattern (seq, seq).

        Features: [mean entropy, first-token attn, prev-token attn,
                   self-attn, max attn]
        """
        # attn: (seq, seq) probabilities (rows sum to 1)
        seq = attn.shape[0]
        feats = np.zeros(5, dtype=np.float64)
        entropies = np.zeros(seq, dtype=np.float64)
        first_token = np.zeros(seq, dtype=np.float64)
        prev_token = np.zeros(seq, dtype=np.float64)
        self_attn = np.zeros(seq, dtype=np.float64)
        max_attn = np.zeros(seq, dtype=np.float64)
        for i in range(seq):
            row = attn[i]
            entropies[i] = entropy(row)
            first_token[i] = row[0]
            prev_token[i] = row[i - 1] if i > 0 else 0.0
            self_attn[i] = row[i]
            max_attn[i] = row.max()
        feats[0] = entropies.mean()
        feats[1] = first_token.mean()
        feats[2] = prev_token.mean()
        feats[3] = self_attn.mean()
        feats[4] = max_attn.mean()
        return feats

    def compute_attention_head_specialization(
        self, traces_by_domain: dict
    ) -> dict:
        """Per-domain (n_layers, n_heads) specialization scores."""
        n_layers = self.n_layers
        n_heads = self.n_heads
        domains = self.domains

        # mean feature vector per (domain, layer, head) averaged over prompts
        feat_shape = (len(domains), n_layers, n_heads, 5)
        domain_feats = np.zeros(feat_shape, dtype=np.float64)
        domain_counts = np.zeros(len(domains), dtype=np.int64)

        for di, dom in enumerate(domains):
            traces = traces_by_domain.get(dom, [])
            domain_counts[di] = len(traces)
            for tr in traces:
                # attention: (n_layers, n_heads, seq, seq)
                for L in range(n_layers):
                    for h in range(n_heads):
                        domain_feats[di, L, h] += self._attention_features(
                            tr.attention[L, h]
                        )
            if domain_counts[di] > 0:
                domain_feats[di] /= domain_counts[di]

        # z-score normalize per feature dimension (across all domain/layer/head)
        flat = domain_feats.reshape(-1, 5)
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (domain_feats - mu) / sd  # (D, L, H, 5)

        # specialization: distance of domain k from mean of other domains
        result = {}
        for di, dom in enumerate(domains):
            others = np.delete(np.arange(len(domains)), di, axis=0)
            other_mean = z[others].mean(axis=0)  # (L, H, 5)
            dist = np.linalg.norm(z[di] - other_mean, axis=-1)  # (L, H)
            result[dom] = dist
        return result

    # -- weight matrix sensitivity ----------------------------------------

    def compute_weight_matrix_sensitivity(
        self, traces_by_domain: dict
    ) -> dict:
        """Per matrix type (n_layers, n_domains) sensitivity."""
        n_layers = self.n_layers
        domains = self.domains

        # centroids per domain per layer for the two proxy signals
        # mlp proxy: mean mlp_acts over seq & prompts -> (D, L, d_mlp)
        # resid proxy: mean residuals[1:] over seq & prompts -> (D, L, d_model)
        mlp_cent = np.zeros((len(domains), n_layers, self.d_mlp), dtype=np.float64)
        res_cent = np.zeros((len(domains), n_layers, self.d_model), dtype=np.float64)
        counts = np.zeros(len(domains), dtype=np.int64)

        for di, dom in enumerate(domains):
            traces = traces_by_domain.get(dom, [])
            counts[di] = len(traces)
            for tr in traces:
                # mlp_acts: (n_layers, seq, d_mlp) -> mean over seq
                mlp_cent[di] += tr.mlp_acts.mean(axis=1)
                # residuals: (n_layers+1, seq, d_model); use layers 1..n_layers
                res_cent[di] += tr.residuals[1:].mean(axis=1)
            if counts[di] > 0:
                mlp_cent[di] /= counts[di]
                res_cent[di] /= counts[di]

        result = {}
        for mt in self.matrix_types:
            sens = np.zeros((n_layers, len(domains)), dtype=np.float64)
            for di, dom in enumerate(domains):
                others = np.delete(np.arange(len(domains)), di, axis=0)
                for L in range(n_layers):
                    if mt in self.MLP_MATRIX_TYPES:
                        a = mlp_cent[di, L]
                        b = mlp_cent[others, L].mean(axis=0)
                    else:
                        a = res_cent[di, L]
                        b = res_cent[others, L].mean(axis=0)
                    sens[L, di] = 1.0 - cosine_similarity(a, b)
            result[mt] = sens
        return result

    # -- causal attribution ------------------------------------------------

    def compute_causal_attribution(
        self, traces_by_domain: dict
    ) -> np.ndarray:
        """(n_layers, n_domains) causal attribution via residual patching proxy."""
        n_layers = self.n_layers
        domains = self.domains
        result = np.zeros((n_layers, len(domains)), dtype=np.float64)

        for di, dom in enumerate(domains):
            others = [d for j, d in enumerate(domains) if j != di]
            for L in range(n_layers):
                layer_shifts = []
                for tr_a in traces_by_domain.get(dom, []):
                    for odom in others:
                        for tr_b in traces_by_domain.get(odom, []):
                            res_a = tr_a.residuals[L]  # (seq, d_model)
                            res_b = tr_b.residuals[L]
                            # align seq lengths
                            s = min(res_a.shape[0], res_b.shape[0])
                            # Use float32 to avoid overflow, and mean-pool over seq
                            # to reduce magnitude before computing norms
                            ra = np.asarray(res_a[:s].mean(axis=0), dtype=np.float32)
                            rb = np.asarray(res_b[:s].mean(axis=0), dtype=np.float32)
                            norm_a = float(np.linalg.norm(ra))
                            if norm_a < 1e-12:
                                continue
                            shift = float(np.linalg.norm(ra - rb)) / norm_a
                            if np.isfinite(shift):
                                layer_shifts.append(shift)
                if layer_shifts:
                    result[L, di] = float(np.mean(layer_shifts))
                else:
                    result[L, di] = 0.0
        return result

    # -- mlp selectivity ---------------------------------------------------

    def compute_mlp_selectivity(
        self, traces_by_domain: dict, threshold: float = 0.5, eps: float = 1e-8
    ) -> dict:
        """Per-domain list of specialist neuron counts per layer."""
        n_layers = self.n_layers
        d_mlp = self.d_mlp
        domains = self.domains

        # mean activation per (domain, layer, neuron) averaged over seq & prompts
        mean_acts = np.zeros((len(domains), n_layers, d_mlp), dtype=np.float64)
        counts = np.zeros(len(domains), dtype=np.int64)
        for di, dom in enumerate(domains):
            traces = traces_by_domain.get(dom, [])
            counts[di] = len(traces)
            for tr in traces:
                # mlp_acts: (n_layers, seq, d_mlp) -> mean over seq
                mean_acts[di] += tr.mlp_acts.mean(axis=1)
            if counts[di] > 0:
                mean_acts[di] /= counts[di]

        result = {}
        for di, dom in enumerate(domains):
            specialist_counts = []
            others = np.delete(np.arange(len(domains)), di, axis=0)
            for L in range(n_layers):
                a_k = mean_acts[di, L]  # (d_mlp,)
                a_others = mean_acts[others, L]  # (n_other, d_mlp)
                max_other = a_others.max(axis=0)
                std_across = mean_acts[:, L].std(axis=0)
                selectivity = (a_k - max_other) / (std_across + eps)
                specialist_counts.append(int(np.sum(selectivity > threshold)))
            result[dom] = specialist_counts
        return result

    # -- param classification ---------------------------------------------

    def classify_params(
        self,
        sensitivity: np.ndarray,
        threshold: float = 0.3,
        dead_percentile: float = 10.0,
    ) -> str:
        """Classify one weight matrix given per-domain sensitivities."""
        sens = np.asarray(sensitivity, dtype=np.float64).ravel()
        if sens.size == 0:
            return "dead"
        above = [self.domains[i] for i in range(len(sens)) if sens[i] > threshold]
        if len(above) == 1:
            return f"domain-specific:{above[0]}"
        max_val = float(sens.max())
        dead_thresh = np.percentile(sens, dead_percentile)
        # if all values are below the dead percentile of the max, treat as dead
        if max_val > 0 and np.all(sens <= dead_thresh) and np.all(sens < 1e-6):
            return "dead"
        if max_val < 1e-6:
            return "dead"
        return "shared"

    # -- run ---------------------------------------------------------------

    def run(self, traces_by_domain: dict) -> AttributionResult:
        """Compute all signals and combine into AttributionResult."""
        domains = self.domains
        n_layers = self.n_layers

        attn_spec = self.compute_attention_head_specialization(traces_by_domain)
        weight_sens = self.compute_weight_matrix_sensitivity(traces_by_domain)
        causal = self.compute_causal_attribution(traces_by_domain)
        mlp_sel = self.compute_mlp_selectivity(traces_by_domain)

        # Normalize signals to [0,1]
        # causal: (n_layers, n_domains)
        causal_norm = normalize_to_01(causal)
        # attn: per domain (n_layers, n_heads) -> mean over heads -> (n_layers,)
        attn_per_layer = {}
        for dom in domains:
            arr = attn_spec.get(dom, np.zeros((n_layers, self.n_heads)))
            per_layer = arr.mean(axis=1)
            attn_per_layer[dom] = normalize_to_01(per_layer)
        # mlp: per domain list of specialist counts -> normalize
        mlp_per_layer = {}
        for dom in domains:
            counts = np.asarray(mlp_sel.get(dom, [0] * n_layers), dtype=np.float64)
            mlp_per_layer[dom] = normalize_to_01(counts)

        # combined layer importance
        layer_importance = {}
        for di, dom in enumerate(domains):
            combined = (
                0.3 * mlp_per_layer[dom]
                + 0.2 * attn_per_layer[dom]
                + 0.5 * causal_norm[:, di]
            )
            layer_importance[dom] = combined.tolist()

        # param knowledge map: per layer, per matrix type
        param_knowledge_map = {}
        for L in range(n_layers):
            param_knowledge_map[L] = {}
            for mt in self.matrix_types:
                sens = weight_sens[mt][L]  # (n_domains,)
                param_knowledge_map[L][mt] = self.classify_params(sens)

        # total params estimate: 7 matrix types per layer
        # attn: 4 * d_model^2 ; mlp: 3 * d_model * d_mlp
        attn_params = 4 * (self.d_model * self.d_model)
        mlp_params = 3 * (self.d_model * self.d_mlp)
        params_per_layer = attn_params + mlp_params
        total_params = params_per_layer * n_layers

        return AttributionResult(
            domains=list(domains),
            n_layers=n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            layer_importance=layer_importance,
            weight_matrix_sensitivity=weight_sens,
            attention_head_spec=attn_spec,
            param_knowledge_map=param_knowledge_map,
            total_params=total_params,
            params_per_layer=params_per_layer,
        )
