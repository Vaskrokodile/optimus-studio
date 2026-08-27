"""Tests for the RL mode package."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Think tags for test fixtures
_TO = chr(60) + "think" + chr(62)
_TC = chr(60) + "/think" + chr(62)


# ---------------------------------------------------------------------------
# Layer 2: RL prompts
# ---------------------------------------------------------------------------


class TestRLPrompts:
    def test_rl_system_prompt_math(self):
        from iloptimus.core.rl_mode.rl_prompts import rl_system_prompt
        prompt = rl_system_prompt("math")
        assert "RL MODE" in prompt
        assert "math" in prompt.lower()
        assert "NO-CODE" in prompt
        assert len(prompt) > 200

    def test_rl_system_prompt_coding(self):
        from iloptimus.core.rl_mode.rl_prompts import rl_system_prompt
        prompt = rl_system_prompt("coding")
        assert "RL MODE" in prompt
        assert "coding" in prompt.lower() or "code" in prompt.lower()

    def test_rl_system_prompt_reasoning(self):
        from iloptimus.core.rl_mode.rl_prompts import rl_system_prompt
        prompt = rl_system_prompt("reasoning")
        assert "RL MODE" in prompt
        assert "reasoning" in prompt.lower()

    def test_rl_system_prompt_agentic(self):
        from iloptimus.core.rl_mode.rl_prompts import rl_system_prompt
        prompt = rl_system_prompt("agentic")
        assert "RL MODE" in prompt
        assert "agent" in prompt.lower()

    def test_rl_system_prompt_unknown_domain_fallback(self):
        from iloptimus.core.rl_mode.rl_prompts import rl_system_prompt
        prompt = rl_system_prompt("unknown_domain")
        assert "RL MODE" in prompt  # falls back to math

    def test_supported_domains(self):
        from iloptimus.core.rl_mode.rl_prompts import SUPPORTED_RL_DOMAINS
        assert "math" in SUPPORTED_RL_DOMAINS
        assert "coding" in SUPPORTED_RL_DOMAINS
        assert "reasoning" in SUPPORTED_RL_DOMAINS
        assert "agentic" in SUPPORTED_RL_DOMAINS


# ---------------------------------------------------------------------------
# Layer 3: RL skills
# ---------------------------------------------------------------------------


class TestRLSkills:
    def test_rl_skills_prompt_nonempty(self):
        from iloptimus.core.rl_mode.rl_skills import rl_skills_prompt
        prompt = rl_skills_prompt()
        assert len(prompt) > 200
        assert "environment" in prompt.lower() or "env" in prompt.lower()
        assert "vllm" in prompt.lower() or "batch" in prompt.lower()
        assert "rl batching" in prompt.lower() or "param-split" in prompt.lower()

    def test_rl_skills_prompt_truncation(self):
        from iloptimus.core.rl_mode.rl_skills import rl_skills_prompt
        prompt = rl_skills_prompt(max_chars=500)
        assert len(prompt) <= 600  # allows some overflow from last block

    def test_list_rl_skills(self):
        from iloptimus.core.rl_mode.rl_skills import list_rl_skills
        skills = list_rl_skills()
        assert len(skills) >= 4
        for skill in skills:
            assert "id" in skill
            assert "chars" in skill
            assert skill["chars"] > 0


# ---------------------------------------------------------------------------
# Layer 4: RL templates
# ---------------------------------------------------------------------------


class TestRLTemplates:
    def test_list_all_templates(self):
        from iloptimus.core.rl_mode.rl_templates import list_env_templates
        templates = list_env_templates()
        assert len(templates) >= 20  # we have 25 templates

    def test_list_by_domain_math(self):
        from iloptimus.core.rl_mode.rl_templates import list_env_templates
        templates = list_env_templates("math")
        assert len(templates) >= 3
        for t in templates:
            assert t.domain == "math"

    def test_list_by_domain_coding(self):
        from iloptimus.core.rl_mode.rl_templates import list_env_templates
        templates = list_env_templates("coding")
        assert len(templates) >= 3
        for t in templates:
            assert t.domain == "coding"

    def test_list_by_domain_reasoning(self):
        from iloptimus.core.rl_mode.rl_templates import list_env_templates
        templates = list_env_templates("reasoning")
        assert len(templates) >= 3
        for t in templates:
            assert t.domain == "reasoning"

    def test_list_by_domain_agentic(self):
        from iloptimus.core.rl_mode.rl_templates import list_env_templates
        templates = list_env_templates("agentic")
        assert len(templates) >= 3
        for t in templates:
            assert t.domain == "agentic"

    def test_get_template_by_name(self):
        from iloptimus.core.rl_mode.rl_templates import get_env_template
        t = get_env_template("proof-golf")
        assert t is not None
        assert t.name == "proof-golf"
        assert t.domain == "math"

    def test_get_template_nonexistent(self):
        from iloptimus.core.rl_mode.rl_templates import get_env_template
        assert get_env_template("nonexistent-env") is None

    def test_template_has_reward_weights(self):
        from iloptimus.core.rl_mode.rl_templates import get_env_template
        t = get_env_template("surgical-patch")
        assert t is not None
        assert "correctness" in t.reward_weights
        assert t.reward_weights["correctness"] == 1.0

    def test_template_has_capabilities(self):
        from iloptimus.core.rl_mode.rl_templates import get_env_template
        t = get_env_template("trajectory-doctor")
        assert t is not None
        assert len(t.capabilities) > 0

    def test_template_supports_rl_batching(self):
        from iloptimus.core.rl_mode.rl_templates import get_env_template
        t = get_env_template("proof-golf")
        assert t is not None
        assert t.supports_rl_batching is True

    def test_recommend_templates_by_capability(self):
        from iloptimus.core.rl_mode.rl_templates import recommend_templates
        recommended = recommend_templates(["debugging", "minimal-changes"])
        assert len(recommended) > 0
        # surgical-patch should be recommended for debugging
        names = [t.name for t in recommended]
        assert "surgical-patch" in names

    def test_template_catalog_prompt(self):
        from iloptimus.core.rl_mode.rl_templates import template_catalog_prompt
        prompt = template_catalog_prompt("math")
        assert "proof-golf" in prompt
        assert len(prompt) > 100

    def test_template_catalog_prompt_truncation(self):
        from iloptimus.core.rl_mode.rl_templates import template_catalog_prompt
        prompt = template_catalog_prompt(max_chars=200)
        assert len(prompt) <= 300


# ---------------------------------------------------------------------------
# Layer 5: RL env builder
# ---------------------------------------------------------------------------


class TestRLEnvBuilder:
    def test_list_buildable_envs(self):
        from iloptimus.core.rl_mode.rl_env_builder import list_buildable_envs
        envs = list_buildable_envs()
        assert len(envs) >= 20
        for env in envs:
            assert "name" in env
            assert "domain" in env
            assert "description" in env

    def test_list_buildable_envs_by_domain(self):
        from iloptimus.core.rl_mode.rl_env_builder import list_buildable_envs
        envs = list_buildable_envs("math")
        assert len(envs) >= 3
        for env in envs:
            assert env["domain"] == "math"

    def test_build_environment_proof_golf(self):
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(template_name="proof-golf", n_problems=5)
        env = build_environment(spec)
        assert env is not None
        # Should have problems loaded
        assert len(env._problems) > 0

    def test_build_environment_surgical_patch(self):
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(template_name="surgical-patch", n_problems=5)
        env = build_environment(spec)
        assert env is not None

    def test_build_environment_unknown_template(self):
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(template_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown environment template"):
            build_environment(spec)

    def test_build_environment_with_overrides(self):
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(
            template_name="proof-golf",
            n_problems=5,
            difficulty=0.8,
            token_budget=512,
        )
        env = build_environment(spec)
        assert env is not None

    def test_env_spec_prompt(self):
        from iloptimus.core.rl_mode.rl_env_builder import env_spec_prompt
        prompt = env_spec_prompt()
        assert "proof-golf" in prompt or "surgical-patch" in prompt
        assert "no code" in prompt.lower()


# ---------------------------------------------------------------------------
# Layer 6: RL batching
# ---------------------------------------------------------------------------


class TestRLBatching:
    def test_rl_batching_config_defaults(self):
        from iloptimus.core.rl_mode.rl_batching import RLBatchingConfig
        config = RLBatchingConfig()
        assert config.n_pipelines == 20
        assert config.lora_param_fraction == 0.05
        assert config.mock_mode is False

    def test_rl_batching_config_custom(self):
        from iloptimus.core.rl_mode.rl_batching import RLBatchingConfig
        config = RLBatchingConfig(
            n_pipelines=5,
            lora_param_fraction=0.1,
            mock_mode=True,
        )
        assert config.n_pipelines == 5
        assert config.lora_param_fraction == 0.1
        assert config.mock_mode is True

    def test_build_pipeline_requests(self):
        from iloptimus.core.rl_mode.rl_batching import build_pipeline_requests
        specs = [
            {"pipeline_id": "p1", "prompt": "Solve x+1=2"},
            {"pipeline_id": "p2", "prompt": "Fix this bug", "n_completions": 4},
        ]
        requests = build_pipeline_requests(specs, n_completions=8)
        assert len(requests) == 2
        assert requests[0].pipeline_id == "p1"
        assert requests[0].n_completions == 8
        assert requests[1].n_completions == 4

    def test_create_rl_batching_rollout_mock(self):
        from iloptimus.core.rl_mode.rl_batching import create_rl_batching_rollout, RLBatchingConfig
        config = RLBatchingConfig(mock_mode=True, mock_generate_fn=lambda p, n: ["mock"] * n)
        rollout = create_rl_batching_rollout(config)
        assert rollout is not None


# ---------------------------------------------------------------------------
# Layer 1: RL orchestrator
# ---------------------------------------------------------------------------


class TestRLOrchestrator:
    def test_rl_mode_spec_defaults(self):
        from iloptimus.core.rl_mode import RLModeSpec
        spec = RLModeSpec()
        assert spec.domain == "math"
        assert spec.n_pipelines == 5
        assert spec.n_rounds == 10
        assert spec.enable_rl_batching is True

    def test_build_failure_driven_rl_spec(self):
        from iloptimus.core.rl_mode import build_failure_driven_rl_spec
        failed = [
            {"capabilities": ["debugging", "code-comprehension"], "error_type": "wrong_answer"},
            {"capabilities": ["conciseness"], "error_type": "truncated"},
        ]
        spec = build_failure_driven_rl_spec(failed, domain="coding")
        assert spec.domain == "coding"
        assert "debugging" in spec.weak_capabilities
        assert "conciseness" in spec.weak_capabilities
        assert "correctness" in spec.weak_capabilities  # from wrong_answer

    def test_build_failure_driven_rl_spec_empty(self):
        from iloptimus.core.rl_mode import build_failure_driven_rl_spec
        spec = build_failure_driven_rl_spec([], domain="math")
        assert spec.domain == "math"
        assert len(spec.weak_capabilities) == 0

    def test_enter_rl_mode_mock(self):
        """Test RL mode with mock generation (no vLLM needed)."""
        from iloptimus.core.rl_mode import enter_rl_mode, RLModeSpec
        spec = RLModeSpec(
            domain="math",
            n_pipelines=2,
            n_rounds=3,
            enable_rl_batching=True,
            mock_mode=True,
            mock_generate_fn=lambda p, n: ["mock response"] * n,
        )
        result = enter_rl_mode(spec)
        assert result.run_id is not None
        assert result.error is None
        assert result.rounds_completed == 3
        assert len(result.reward_history) == 3
        assert result.mean_reward > 0

    def test_enter_rl_mode_with_env_names(self):
        from iloptimus.core.rl_mode import enter_rl_mode, RLModeSpec
        spec = RLModeSpec(
            domain="math",
            env_names=["proof-golf", "symbolic-equation-solver"],
            n_pipelines=2,
            n_rounds=2,
            mock_mode=True,
        )
        result = enter_rl_mode(spec)
        assert result.error is None
        assert "proof-golf" in result.envs_used

    def test_enter_rl_mode_with_weak_capabilities(self):
        from iloptimus.core.rl_mode import enter_rl_mode, RLModeSpec
        spec = RLModeSpec(
            domain="coding",
            weak_capabilities=["debugging", "self-correction"],
            n_pipelines=3,
            n_rounds=2,
            mock_mode=True,
        )
        result = enter_rl_mode(spec)
        assert result.error is None
        assert len(result.envs_used) > 0

    def test_enter_rl_mode_no_batching(self):
        from iloptimus.core.rl_mode import enter_rl_mode, RLModeSpec
        spec = RLModeSpec(
            domain="reasoning",
            n_pipelines=1,
            n_rounds=2,
            enable_rl_batching=False,
            mock_mode=True,
        )
        result = enter_rl_mode(spec)
        assert result.error is None
        assert result.rounds_completed == 2

    def test_rl_mode_result_public(self):
        from iloptimus.core.rl_mode import RLModeResult
        result = RLModeResult(run_id="test123", rounds_completed=5)
        public = result.public()
        assert public["run_id"] == "test123"
        assert public["rounds_completed"] == 5

    def test_list_rl_sessions_empty(self):
        from iloptimus.core.rl_mode import list_rl_sessions
        sessions = list_rl_sessions()
        # May have sessions from previous tests, just check it returns a list
        assert isinstance(sessions, list)

    def test_get_rl_session_nonexistent(self):
        from iloptimus.core.rl_mode import get_rl_session
        assert get_rl_session("nonexistent_id") is None

    def test_rl_mode_prompt(self):
        from iloptimus.core.rl_mode import rl_mode_prompt
        prompt = rl_mode_prompt("math")
        assert "RL MODE" in prompt
        assert len(prompt) > 1000  # system + skills + catalog


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRLIntegration:
    def test_full_prompt_assembly(self):
        """Test that the full RL mode prompt is reasonable size."""
        from iloptimus.core.rl_mode import rl_mode_prompt
        for domain in ["math", "coding", "reasoning", "agentic"]:
            prompt = rl_mode_prompt(domain)
            # Should be comprehensive but not too large for a small model
            assert len(prompt) > 500
            assert len(prompt) < 15000

    def test_env_build_and_reset(self):
        """Test building an environment and calling reset()."""
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(template_name="proof-golf", n_problems=3)
        env = build_environment(spec)
        obs, info = env.reset()
        assert "prompt" in obs
        assert len(obs["prompt"]) > 0

    def test_env_build_and_step(self):
        """Test building an environment and calling step()."""
        from iloptimus.core.rl_mode.rl_env_builder import EnvBuildSpec, build_environment
        spec = EnvBuildSpec(template_name="proof-golf", n_problems=3)
        env = build_environment(spec)
        env.reset()
        obs, reward, terminated, truncated, info = env.step("This is a test answer")
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_template_to_env_pipeline(self):
        """Test the full pipeline: template selection -> env build -> reset -> step."""
        from iloptimus.core.rl_mode import (
            get_env_template, EnvBuildSpec, build_environment,
        )
        # 1. Select template
        template = get_env_template("symbolic-equation-solver")
        assert template is not None
        # 2. Build env
        spec = EnvBuildSpec(template_name=template.name, n_problems=5)
        env = build_environment(spec)
        # 3. Reset and step
        obs, info = env.reset()
        assert "prompt" in obs
        obs, reward, terminated, truncated, info = env.step("x = 5")
        assert isinstance(reward, float)

    def test_rl_templates_cover_all_domains(self):
        """Verify we have templates for all supported domains."""
        from iloptimus.core.rl_mode import SUPPORTED_RL_DOMAINS, list_env_templates
        for domain in SUPPORTED_RL_DOMAINS:
            templates = list_env_templates(domain)
            assert len(templates) >= 3, f"Domain {domain} has only {len(templates)} templates"

    def test_core_api_exposed(self):
        """Test that the core API exposes RL mode functions."""
        from iloptimus.core import (
            enter_rl_mode, RLModeSpec, RLModeResult,
            build_failure_driven_rl_spec, list_rl_sessions,
            get_rl_session, rl_mode_prompt,
        )
        # Just verify they're callable
        assert callable(enter_rl_mode)
        assert callable(build_failure_driven_rl_spec)
        assert callable(list_rl_sessions)
        assert callable(get_rl_session)
        assert callable(rl_mode_prompt)
