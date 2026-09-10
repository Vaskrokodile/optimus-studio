"""Capability Profiler — per-model skill mastery persistence.

Wraps the skill graph with a per-model ``SkillGraphProfile`` stored under
``~/.iloptimus/profiles/<model_id>/skill_graph.json``. Each graded rollout is
folded into the Beta posteriors of the active skill nodes and the composition
that was exercised. On model-fingerprint drift the old profile is archived and
a new one is seeded from the old with halved Beta concentration (transfer
prior) to reflect uncertainty about the new checkpoint.

See ``RESEARCH_skill_graph.md`` section 4/6.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .skill_graph import (
    DOMAIN_TO_SKILLS,
    SkillComposition,
    SkillGraph,
    SkillGraphProfile,
    SkillNodeState,
    CompositionState,
    domain_to_active_skills,
    find_unmastered_compositions,
)
from .storage import atomic_write_json, profiles_dir

# Score at/above which a rollout counts as a strict pass for binary evidence.
PASS_THRESHOLD = 0.7
# Fraction of leaf evidence propagated to each ancestor (optional, per the doc).
ANCESTOR_PROPAGATION = 0.25


def _profile_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace("\\", "_")
    return profiles_dir() / safe / "skill_graph.json"


def _serialize(profile: SkillGraphProfile) -> dict[str, Any]:
    """JSON-safe view: compositions keyed by their sorted-id string."""
    return {
        "model_id": profile.model_id,
        "model_fingerprint": profile.model_fingerprint,
        "version": profile.version,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "rollout_count": profile.rollout_count,
        "nodes": {k: asdict(v) for k, v in profile.nodes.items()},
        "compositions": {k: asdict(v) | {"node_ids": sorted(v.node_ids)} for k, v in profile.compositions.items()},
    }


def _deserialize(data: dict[str, Any]) -> SkillGraphProfile:
    nodes = {
        k: SkillNodeState(**{kk: vv for kk, vv in v.items() if kk in SkillNodeState.__dataclass_fields__})
        for k, v in data.get("nodes", {}).items()
    }
    comps: dict[str, CompositionState] = {}
    for key, v in data.get("compositions", {}).items():
        fields = {kk: vv for kk, vv in v.items() if kk in CompositionState.__dataclass_fields__}
        nids = v.get("node_ids") or key.split("+")
        fields["node_ids"] = frozenset(nids)
        comp = CompositionState(**fields)
        comps["+".join(sorted(comp.node_ids))] = comp
    return SkillGraphProfile(
        model_id=data["model_id"],
        model_fingerprint=data.get("model_fingerprint", ""),
        version=data.get("version", 1),
        created_at=data.get("created_at", time.time()),
        updated_at=data.get("updated_at", time.time()),
        rollout_count=data.get("rollout_count", 0),
        nodes=nodes,
        compositions=comps,
    )


class CapabilityProfiler:
    """Per-model Beta-posterior skill profiler with on-disk persistence."""

    def __init__(self, graph: SkillGraph | None = None) -> None:
        self.graph = graph or SkillGraph()

    # -- persistence -------------------------------------------------------

    def load(self, model_id: str, fingerprint: str = "") -> SkillGraphProfile:
        path = _profile_path(model_id)
        if path.exists():
            try:
                profile = _deserialize(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                profile = SkillGraphProfile(model_id=model_id, model_fingerprint=fingerprint)
        else:
            profile = SkillGraphProfile(model_id=model_id, model_fingerprint=fingerprint)

        # Fingerprint drift: archive + transfer prior.
        if fingerprint and profile.model_fingerprint and fingerprint != profile.model_fingerprint:
            self._archive(profile)
            profile = self._transfer_prior(profile, fingerprint)
        elif fingerprint and not profile.model_fingerprint:
            profile.model_fingerprint = fingerprint

        # Ensure every ontology node has a state entry (uniform prior).
        for node in self.graph.all_nodes():
            profile.nodes.setdefault(node, SkillNodeState(node_id=node))
        return profile

    def save(self, profile: SkillGraphProfile) -> None:
        profile.updated_at = time.time()
        atomic_write_json(_profile_path(profile.model_id), _serialize(profile))

    def _archive(self, profile: SkillGraphProfile) -> None:
        path = _profile_path(profile.model_id)
        if not path.exists():
            return
        archive = path.with_name(f"skill_graph.json.v{int(time.time())}")
        try:
            archive.write_bytes(path.read_bytes())
        except OSError:
            pass

    def _transfer_prior(self, old: SkillGraphProfile, new_fingerprint: str) -> SkillGraphProfile:
        """Seed a new profile from the old with halved Beta concentration."""
        new = SkillGraphProfile(
            model_id=old.model_id,
            model_fingerprint=new_fingerprint,
            version=old.version + 1,
            created_at=time.time(),
            updated_at=time.time(),
            rollout_count=0,
        )
        for nid, st in old.nodes.items():
            new.nodes[nid] = SkillNodeState(
                node_id=nid,
                alpha=max(1.0, st.alpha * 0.5),
                beta=max(1.0, st.beta * 0.5),
                attempts=0,
                successes=0.0,
                last_seen=st.last_seen,
            )
        for key, comp in old.compositions.items():
            new.compositions[key] = CompositionState(
                node_ids=comp.node_ids,
                alpha=max(1.0, comp.alpha * 0.5),
                beta=max(1.0, comp.beta * 0.5),
                attempts=0,
                successes=0.0,
            )
        return new

    # -- recording --------------------------------------------------------

    def record_rollout(
        self,
        model_id: str,
        domain: str,
        task_idx: int,
        graded: Any,
        fingerprint: str = "",
        profile: SkillGraphProfile | None = None,
    ) -> SkillGraphProfile:
        """Fold one graded rollout into the profile's Beta posteriors."""
        if profile is None:
            profile = self.load(model_id, fingerprint)

        active = domain_to_active_skills(domain, graded)
        if not active:
            return profile

        score = float(getattr(graded, "score", 0.0) or 0.0)
        # Continuous evidence r in [0,1]; for binary tasks callers may pre-binarize.
        r = score

        node_ids: list[str] = []
        for node_id, weight in active:
            node_ids.append(node_id)
            self._update_node(profile, node_id, weight)
            # Optional ancestor propagation.
            for anc in self.graph.ancestors(node_id):
                self._update_node(profile, anc, weight * ANCESTOR_PROPAGATION)

        # Record the composition exercised by this task.
        comp_nodes = frozenset(DOMAIN_TO_SKILLS.get(domain, []))
        if comp_nodes:
            self._update_composition(profile, comp_nodes, r)
            self.graph.record_cooccurrence(list(comp_nodes))

        profile.rollout_count += 1
        return profile

    def _update_node(self, profile: SkillGraphProfile, node_id: str, r: float) -> None:
        st = profile.nodes.get(node_id) or SkillNodeState(node_id=node_id)
        st.alpha += max(0.0, min(1.0, r))
        st.beta += max(0.0, min(1.0, 1.0 - r))
        st.attempts += 1
        st.successes += max(0.0, min(1.0, r))
        st.last_seen = time.time()
        profile.nodes[node_id] = st

    def _update_composition(self, profile: SkillGraphProfile, node_ids: frozenset[str], r: float) -> None:
        key = "+".join(sorted(node_ids))
        comp = profile.compositions.get(key) or CompositionState(node_ids=node_ids)
        comp.alpha += max(0.0, min(1.0, r))
        comp.beta += max(0.0, min(1.0, 1.0 - r))
        comp.attempts += 1
        comp.successes += max(0.0, min(1.0, r))
        profile.compositions[key] = comp

    # -- queries ----------------------------------------------------------

    def mastery(self, profile: SkillGraphProfile, node_id: str) -> float:
        st = profile.nodes.get(node_id)
        return st.mastery if st else 0.5

    def composition_mastery(self, profile: SkillGraphProfile, node_ids: frozenset[str]) -> float:
        key = "+".join(sorted(node_ids))
        comp = profile.compositions.get(key)
        if comp is not None and comp.attempts > 0:
            return comp.mastery
        product = 1.0
        for n in node_ids:
            st = profile.nodes.get(n)
            product *= st.mastery if st else 0.5
        return product

    def suggest_compositions(
        self,
        profile: SkillGraphProfile,
        loop_kind: str = "",
        k: int = 5,
        threshold: float = 0.85,
        max_size: int = 4,
        frontier_distance: int = 2,
    ) -> list[SkillComposition]:
        return find_unmastered_compositions(
            self.graph,
            profile,
            k=k,
            threshold=threshold,
            max_size=max_size,
            frontier_distance=frontier_distance,
        )


__all__ = ["CapabilityProfiler", "PASS_THRESHOLD"]
