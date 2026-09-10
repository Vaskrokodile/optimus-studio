"""Tests for the adaptive orchestrator and its subsystems: persistent world,
session-to-env failure extraction, env generator agent, and the end-to-end
orchestrator smoke test.

Implements the validation plans in ``RESEARCH_meta_architecture.md`` Sections
5 and 8.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

os.environ.setdefault("ILOPTIMUS_HOME", os.path.join(os.environ.get("TEMP", "/tmp"), "ilopt_test"))

from iloptimus.core.adaptive_orchestrator import (
    AdaptiveConfig,
    AdaptiveOrchestrator,
    OrchestratorStore,
    TrainingSignal,
)
from iloptimus.core.env_generator_agent import (
    EnvGeneratorAgent,
    GeneratorAction,
    Rollout,
    generator_reward,
    run_generated_verifier,
)
from iloptimus.core.grader import GradedResult
from iloptimus.core.persistent_world import PersistentWorld, TaskSpec
from iloptimus.core.session_to_env import (
    extract_failure_spec,
    infer_missing_skill,
    append_failure_spec,
    load_failure_specs,
)


# ---------------------------------------------------------------------------
# Persistent world
# ---------------------------------------------------------------------------


class TestPersistentWorld:
    def test_initial_triggers_fire(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        w = PersistentWorld("test-world")
        triggers = w.active_triggers()
        # srv-3 health 0.55 < 0.60 -> debugging; inventory low_stock 7 > 5; bugs 12 > 10.
        skills = {t[0] for t in triggers}
        assert "debugging" in skills
        assert "operations" in skills
        assert "coding" in skills

    def test_derive_task_uses_current_world_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        w = PersistentWorld("test-world")
        task = w.derive_task(focus_skill="debugging")
        assert task.source == "persistent_world"
        assert task.metadata["world_id"] == "test-world"
        assert task.metadata["tick"] == w.world["tick"]
        assert "simulator" in task.metadata

    def test_advance_and_apply_outcome_persists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        w = PersistentWorld("test-world")
        task = w.derive_task(focus_skill="debugging")
        w.advance()
        w.apply_outcome(task, {"health": 0.9, "cpu": 30}, success=True)
        # State persisted to disk.
        w2 = PersistentWorld("test-world")
        srv = next(s for s in w2.world["servers"] if s["id"] == "srv-3")
        assert srv["health"] == pytest.approx(0.9, abs=0.05)
        assert w2.world["tick"] == 1
        # Event log appended.
        assert (w2.root / "events.jsonl").exists()

    def test_failure_has_consequences(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        w = PersistentWorld("test-world")
        task = w.derive_task(focus_skill="debugging")
        before = next(s for s in w.world["servers"] if s["id"] == "srv-3")["health"]
        w.apply_outcome(task, {"health": 0.55, "cpu": 92}, success=False)
        after = next(s for s in w.world["servers"] if s["id"] == "srv-3")["health"]
        assert after < before  # failure degrades health further


# ---------------------------------------------------------------------------
# session_to_env
# ---------------------------------------------------------------------------


class TestSessionToEnv:
    def test_extract_failure_spec_identifies_stuck_and_missing(self):
        session = {
            "id": "s1", "domain": "tool-devops",
            "response": '<tool>{"name":"read_file","args":{}}</tool>'
                        '<tool>{"name":"restart_server","args":{}}</tool>'
                        '<tool>{"name":"restart_server","args":{}}</tool>',
            "errors": ["restart_server returned 503"],
        }
        grade = GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0, info={})
        spec = extract_failure_spec(session, grade)
        assert spec.stuck_skill == "tool_call:restart_server"
        assert spec.missing_skill == "tool_call:check_load"
        assert spec.skill_path == ["tool_call:read_file", "tool_call:restart_server", "tool_call:restart_server"]
        assert spec.error == "restart_server returned 503"

    def test_infer_missing_skill_fallbacks(self):
        assert infer_missing_skill("restart_server returned 503", "tool_call:restart_server") == "tool_call:check_load"
        # Generic fallback for an unknown tool_call stuck skill.
        assert infer_missing_skill("some error", "tool_call:write_file") == "tool_call:diagnose_write_file"
        assert infer_missing_skill("", "not_a_tool") is None

    def test_append_and_load_failure_specs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        spec = extract_failure_spec(
            {"id": "s1", "domain": "tool-devops",
             "response": '<tool>{"name":"x","args":{}}</tool>', "errors": ["e"]},
            GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0, info={}),
        )
        append_failure_spec("loop-1", spec)
        loaded = load_failure_specs("loop-1")
        assert len(loaded) == 1
        assert loaded[0].stuck_skill == "tool_call:x"


# ---------------------------------------------------------------------------
# env_generator_agent
# ---------------------------------------------------------------------------


class TestEnvGeneratorAgent:
    def test_parse_action_from_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        def stub(prompt):
            return '{"op":"compose_skills","params":{"skill_keys":["a","b"],"difficulty":0.6},"difficulty":0.6,"estimated_success_rate":0.15}'
        gen = EnvGeneratorAgent(generate_fn=stub)
        action = gen.produce_action(frontier_skills=["a"])
        assert action.op == "compose_skills"
        assert action.params["skill_keys"] == ["a", "b"]

    def test_default_action_uses_world_when_available(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        w = PersistentWorld("gen-world")
        gen = EnvGeneratorAgent(generate_fn=None, persistent_world=w)
        action = gen.produce_action()
        assert action.op == "persistent_world"

    def test_materialize_compose(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        gen = EnvGeneratorAgent(generate_fn=None)
        action = GeneratorAction(op="compose_skills", params={"skill_keys": ["tool_call:read_file", "tool_call:grep_search"], "difficulty": 0.6})
        task = gen.materialize(action)
        assert task.source == "composed"
        assert "tool_call:read_file" in task.prompt

    def test_generator_reward_prefers_failing_trainee(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        task = TaskSpec(task_id="t1", prompt="p", domain="coding", source="generated")
        r_fail = generator_reward([Rollout(False, 0.0), Rollout(False, 0.0)], task, [], 1000, 100000)
        r_pass = generator_reward([Rollout(True, 1.0), Rollout(True, 1.0)], task, [], 1000, 100000)
        assert r_fail > r_pass

    def test_generator_reward_penalizes_impossible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        task = TaskSpec(task_id="t1", prompt="p", domain="coding", source="generated")
        # 0% pass -> solvability penalty 0.5
        r_impossible = generator_reward([Rollout(False, 0.0), Rollout(False, 0.0)], task, [], 1000, 100000)
        # 15% pass -> in frontier band, no solvability penalty
        r_frontier = generator_reward(
            [Rollout(False, 0.0), Rollout(False, 0.0), Rollout(False, 0.0), Rollout(False, 0.0),
             Rollout(False, 0.0), Rollout(False, 0.0), Rollout(False, 0.0), Rollout(True, 0.5),
             Rollout(False, 0.0), Rollout(False, 0.0)], task, [], 1000, 100000)
        assert r_frontier > r_impossible

    def test_generated_verifier_runs(self):
        verify = 'def verify(answer):\n    return {"correct": answer.strip()=="42", "score": 1.0 if answer.strip()=="42" else 0.0}'
        assert run_generated_verifier(verify, "42")["correct"] is True
        assert run_generated_verifier(verify, "7")["correct"] is False
        # Empty verifier -> error, not a crash.
        assert run_generated_verifier("", "x")["correct"] is False

    def test_generated_verifier_handles_exception(self):
        verify = 'def verify(answer):\n    raise ValueError("boom")'
        res = run_generated_verifier(verify, "x")
        assert res["correct"] is False


# ---------------------------------------------------------------------------
# Adaptive orchestrator end-to-end smoke test
# ---------------------------------------------------------------------------


class TestAdaptiveOrchestrator:
    def test_one_iteration_emits_training_signal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(0)

        class StubModel:
            def __init__(self, responses):
                self.responses = responses
                self.i = 0
            def generate(self, prompt):
                r = self.responses[self.i % len(self.responses)]
                self.i += 1
                return r

        SOLVE = ('<tool>{"name":"check_logs","args":{}}</tool>' * 3
                 + '<tool>{"name":"scale_up","args":{}}</tool><answer>resolved</answer>')
        world = PersistentWorld("orch-world")
        gen = EnvGeneratorAgent(generate_fn=None, persistent_world=world)
        config = AdaptiveConfig(loop_id="test-loop", model_id="stub",
                                rollouts_per_task=4, max_iterations=3, world_task_prob=1.0)
        orch = AdaptiveOrchestrator(config=config, model=StubModel([SOLVE] * 20),
                                    env_generator=gen, persistent_world=world)

        signal = orch.iteration()
        assert isinstance(signal, TrainingSignal)
        assert signal.reward is not None
        assert orch.state.iteration == 1
        assert signal.task["source"] in ("taskset", "mutated", "generated", "persistent_world", "composed")

    def test_multiple_iterations_persist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(1)

        class StubModel:
            def generate(self, prompt):
                return "<answer>42</answer>"

        world = PersistentWorld("orch-world2")
        gen = EnvGeneratorAgent(generate_fn=None, persistent_world=world)
        config = AdaptiveConfig(loop_id="test-loop2", model_id="stub",
                                rollouts_per_task=2, max_iterations=3, world_task_prob=1.0)
        orch = AdaptiveOrchestrator(config=config, model=StubModel(),
                                    env_generator=gen, persistent_world=world)
        sigs = [orch.iteration() for _ in range(3)]
        assert orch.state.iteration == 3
        store = OrchestratorStore()
        assert store.episodes_path("test-loop2").exists()
        # State persisted.
        state = store.load("test-loop2")
        assert state.iteration == 3

    def test_task_source_in_valid_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
        random.seed(2)

        class StubModel:
            def generate(self, prompt):
                return "<answer>x</answer>"

        config = AdaptiveConfig(loop_id="test-loop3", model_id="stub",
                                rollouts_per_task=1, max_iterations=1, world_task_prob=0.0)
        orch = AdaptiveOrchestrator(config=config, model=StubModel())
        signal = orch.iteration()
        assert signal.task["source"] in ("taskset", "mutated", "generated", "persistent_world", "composed")
