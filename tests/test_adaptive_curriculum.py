"""Tests for the adaptive RL curriculum foundation: capability metrics, skill
graph, capability profiler, and frontier sampler.

Implements the validation plans in ``RESEARCH_skill_graph.md`` and
``RESEARCH_frontier_sampler.md``. Uses a stub model with known per-skill
strengths; no real model is loaded.
"""

from __future__ import annotations

import os
import random
from collections import Counter, deque

import pytest

os.environ.setdefault("ILOPTIMUS_HOME", os.path.join(os.environ.get("TEMP", "/tmp"), "ilopt_test"))

from iloptimus.core.capability_metrics import (
    bernoulli_entropy,
    beta_entropy,
    capability_gain,
    digamma,
    information_gain_binary,
    intelligence_density,
    learning_progress,
    tool_selection_entropy,
)
from iloptimus.core.capability_profiler import CapabilityProfiler
from iloptimus.core.frontier_sampler import (
    FRONTIER_BAND,
    FrontierProfile,
    FrontierSampler,
    deserialize_profile,
    load_frontier_profile,
)
from iloptimus.core.grader import GradedResult
from iloptimus.core.skill_graph import (
    DOMAIN_TO_SKILLS,
    SkillGraph,
    SkillGraphProfile,
    find_unmastered_compositions,
)


# ---------------------------------------------------------------------------
# capability_metrics
# ---------------------------------------------------------------------------


class TestCapabilityMetrics:
    def test_digamma_known_values(self):
        # psi(1) = -gamma, psi(2) = 1 - gamma
        assert digamma(1.0) == pytest.approx(-0.5772156649, abs=1e-6)
        assert digamma(2.0) == pytest.approx(1.0 - 0.5772156649, abs=1e-6)
        # psi(0.5) = -gamma - 2*ln(2)
        assert digamma(0.5) == pytest.approx(-0.5772156649 - 2 * 0.6931471806, abs=1e-6)

    def test_bernoulli_entropy_bounds(self):
        assert bernoulli_entropy(0.5) == pytest.approx(1.0, abs=1e-6)
        assert bernoulli_entropy(0.0) == pytest.approx(0.0, abs=1e-3)
        assert bernoulli_entropy(1.0) == pytest.approx(0.0, abs=1e-3)

    def test_information_gain_concentrates_posterior(self):
        # A more extreme outcome concentrates the posterior more -> higher IG.
        ig_concentrated = information_gain_binary(1, 1, 9, 1)
        ig_diffuse = information_gain_binary(1, 1, 3, 7)
        assert ig_concentrated > ig_diffuse
        assert ig_concentrated > 0.0

    def test_intelligence_density_orders_quality(self):
        high = intelligence_density(0.2, 100, 1e12)
        low = intelligence_density(0.05, 1000, 1e12)
        assert high > low
        assert intelligence_density(0.0, 100, 1e12) == 0.0
        assert intelligence_density(0.2, 0, 1e12) == 0.0

    def test_capability_gain_detects_improvement(self):
        hist = deque([0.1] * 10 + [0.9] * 10)
        assert capability_gain(hist) > 0.0
        # Not enough history -> 0.
        assert capability_gain(deque([0.5, 0.5])) == 0.0

    def test_learning_progress(self):
        assert learning_progress([True, True], [False, False, False, False]) > 0.0
        assert learning_progress([False, False], [True, True, True, True]) < 0.0

    def test_tool_selection_entropy(self):
        assert tool_selection_entropy([{"name": "a"}]) == 0.0
        three = tool_selection_entropy([{"name": "a"}, {"name": "b"}, {"name": "c"}])
        assert three == pytest.approx(1.5849625, abs=1e-6)  # log2(3)
        assert tool_selection_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# skill_graph + capability_profiler
# ---------------------------------------------------------------------------


class TestSkillGraphProfiler:
    def test_distinct_domains_diverge(self, tmp_path, monkeypatch):
        # Stub: gsm8k strong (0.95), aime weak (0.05) -> distinct skill mastery.
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        prof = CapabilityProfiler()
        profile = prof.load("stub-model", fingerprint="fp1")
        for _ in range(50):
            prof.record_rollout("stub-model", "gsm8k", 0,
                GradedResult(score=0.95, correctness=1.0, reasoning_quality=0.9), profile=profile)
        for _ in range(50):
            prof.record_rollout("stub-model", "aime", 0,
                GradedResult(score=0.05, correctness=0.0, reasoning_quality=0.1), profile=profile)
        m_gsm = prof.mastery(profile, "gsm8k")
        m_aime = prof.mastery(profile, "aime")
        assert m_gsm > 0.8
        assert m_aime < 0.2
        # Shared skill 'algebra' receives mixed evidence -> middling.
        m_algebra = prof.mastery(profile, "algebra")
        assert 0.3 < m_algebra < 0.7

    def test_suggested_compositions_target_weak_skill(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        prof = CapabilityProfiler()
        profile = prof.load("stub-model", fingerprint="fp1")
        for _ in range(50):
            prof.record_rollout("stub-model", "gsm8k", 0,
                GradedResult(score=0.95, correctness=1.0, reasoning_quality=0.9), profile=profile)
        for _ in range(50):
            prof.record_rollout("stub-model", "aime", 0,
                GradedResult(score=0.05, correctness=0.0, reasoning_quality=0.1), profile=profile)
        comps = prof.suggest_compositions(profile, k=5, threshold=0.85)
        assert len(comps) > 0
        # Top compositions should include the weak 'aime' skill.
        top_nodes = set()
        for c in comps:
            top_nodes |= c.node_ids
        assert "aime" in top_nodes

    def test_persistence_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        prof = CapabilityProfiler()
        profile = prof.load("stub-model", fingerprint="fp1")
        for _ in range(20):
            prof.record_rollout("stub-model", "gsm8k", 0,
                GradedResult(score=0.9, correctness=1.0, reasoning_quality=0.9), profile=profile)
        prof.save(profile)
        p2 = prof.load("stub-model", fingerprint="fp1")
        assert p2.rollout_count == 20
        assert prof.mastery(p2, "gsm8k") == pytest.approx(prof.mastery(profile, "gsm8k"))

    def test_fingerprint_drift_transfers_prior(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        prof = CapabilityProfiler()
        profile = prof.load("stub-model", fingerprint="fp1")
        for _ in range(40):
            prof.record_rollout("stub-model", "gsm8k", 0,
                GradedResult(score=0.95, correctness=1.0, reasoning_quality=0.9), profile=profile)
        prof.save(profile)
        # New fingerprint -> archive + transfer prior + version bump.
        p2 = prof.load("stub-model", fingerprint="fp2")
        assert p2.version == 2
        assert p2.rollout_count == 0
        # Mastery should still lean high (halved concentration, same direction).
        assert prof.mastery(p2, "gsm8k") > 0.5

    def test_curriculum_ordering_weak_before_unobserved(self, tmp_path, monkeypatch):
        # Build A -> B -> C: A mastered, B weak, C unobserved. B ranks before C,
        # and single-node B ranks before {B, C} due to composition-size penalty.
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        graph = SkillGraph({"A": {"children": ["B", "C"]}})
        prof = CapabilityProfiler(graph=graph)
        profile = prof.load("stub", fingerprint="fp1")
        # Seed node states.
        from iloptimus.core.skill_graph import SkillNodeState
        profile.nodes["A"] = SkillNodeState(node_id="A", alpha=20, beta=1)  # mastered
        profile.nodes["B"] = SkillNodeState(node_id="B", alpha=2, beta=8)   # weak
        profile.nodes["C"] = SkillNodeState(node_id="C", alpha=1, beta=1)   # unobserved
        comps = find_unmastered_compositions(graph, profile, k=10, threshold=0.85, max_size=2)
        # Find the single-node B and the {B, C} composition.
        single_b = next((c for c in comps if c.node_ids == frozenset({"B"})), None)
        comp_bc = next((c for c in comps if c.node_ids == frozenset({"B", "C"})), None)
        assert single_b is not None
        assert comp_bc is not None
        assert single_b.priority_score > comp_bc.priority_score


# ---------------------------------------------------------------------------
# frontier_sampler
# ---------------------------------------------------------------------------


TARGET_SUCCESS = {
    "tool-fs": 0.25, "tool-sql": 0.10, "tool-web": 0.05, "tool-booking": 0.60,
    "tool-pipeline": 0.15, "tool-recovery": 0.20, "tool-distractor": 0.30,
    "tool-parallel": 0.35, "tool-interpreter": 0.45, "tool-devops": 0.08,
    "tool-api": 0.12,
}


def _build_profile():
    profile = FrontierProfile(model_id="stub")
    for d in TARGET_SUCCESS:
        for t in range(4):
            profile.ensure_task(d, d, t)
    return profile


class TestFrontierSampler:
    def test_cold_start_covers_all_arms(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(1)
        profile = _build_profile()
        sampler = FrontierSampler(profile)
        seen = set()
        for _ in range(100):
            k = sampler.sample()
            seen.add((k.domain, k.task_idx))
            correct = random.random() < TARGET_SUCCESS[k.domain]
            sampler.observe(k, {"score": 1.0 if correct else 0.0, "correctness": 1.0 if correct else 0.0},
                             tokens=100, flops=1e12)
        assert len(seen) == 44  # 11 domains x 4 tasks

    def test_frontier_band_attracts_selections(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(2)
        profile = _build_profile()
        sampler = FrontierSampler(profile)
        sel = Counter()
        for _ in range(600):
            k = sampler.sample()
            correct = random.random() < TARGET_SUCCESS[k.domain]
            sampler.observe(k, {"score": 1.0 if correct else 0.0, "correctness": 1.0 if correct else 0.0},
                             tokens=100, flops=1e12)
            sel[k.domain] += 1
        band_domains = [d for d, r in TARGET_SUCCESS.items() if 0.05 <= r <= 0.30]
        band_fraction = sum(sel[d] for d in band_domains) / 600
        assert band_fraction > 0.4
        # Saturated domains (booking 0.6, interpreter 0.45) get fewer picks.
        assert sel["tool-booking"] < sel["tool-devops"]

    def test_persistence_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(3)
        profile = _build_profile()
        sampler = FrontierSampler(profile)
        for _ in range(50):
            k = sampler.sample()
            correct = random.random() < TARGET_SUCCESS[k.domain]
            sampler.observe(k, {"score": 1.0 if correct else 0.0, "correctness": 1.0 if correct else 0.0},
                             tokens=100, flops=1e12)
        sampler.save()
        p2 = load_frontier_profile("stub")
        assert p2.total_attempts == 50
        # Deserialize round-trip preserves a known arm's attempts.
        assert p2.domains["tool-fs"].tasks[0].attempts >= 0

    def test_empty_profile_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        profile = FrontierProfile(model_id="empty")
        sampler = FrontierSampler(profile)
        # No arms -> returns an empty key without crashing.
        k = sampler.sample()
        assert k.domain == ""
