"""RL mode: no-code RL environment building and training for the RSI loop.

When the model decides to do RL training, it enters RL mode. The mode provides:
- 60+ pre-built environment templates (no coding needed)
- vLLM batched generation (N responses per task at ~1x speed)
- RL batching: param-split across environments (novel architecture)
- Failure-driven environment selection
- Curriculum-aware difficulty progression

Layers:
1. Orchestrator — enter_rl_mode(), failure-driven spec builder
2. Prompts — RL mode system prompts (math, coding, reasoning, agentic)
3. Skills — env design, vLLM batching, RL batching, curriculum guidance
4. Templates — 60+ no-code environment template catalog
5. Env builder — instantiate environments from template specs
6. RL batching — param-split pipeline manager (novel)
"""

from .rl_prompts import rl_system_prompt, SUPPORTED_RL_DOMAINS
from .rl_skills import rl_skills_prompt, list_rl_skills
from .rl_templates import (
    EnvTemplate,
    get_env_template,
    list_env_templates,
    recommend_templates,
    template_catalog_prompt,
)
from .rl_env_builder import (
    EnvBuildSpec,
    build_environment,
    list_buildable_envs,
    env_spec_prompt,
)
from .rl_batching import (
    RLBatchingConfig,
    RLBatchingResult,
    create_rl_batching_rollout,
    run_rl_batching_round,
    build_pipeline_requests,
)
from .rl_orchestrator import (
    RLModeSpec,
    RLModeResult,
    enter_rl_mode,
    build_failure_driven_rl_spec,
    list_rl_sessions,
    get_rl_session,
    rl_mode_prompt,
)

__all__ = [
    # Orchestrator
    "RLModeSpec", "RLModeResult", "enter_rl_mode",
    "build_failure_driven_rl_spec",
    "list_rl_sessions", "get_rl_session", "rl_mode_prompt",
    # Prompts
    "rl_system_prompt", "SUPPORTED_RL_DOMAINS",
    # Skills
    "rl_skills_prompt", "list_rl_skills",
    # Templates
    "EnvTemplate", "get_env_template", "list_env_templates",
    "recommend_templates", "template_catalog_prompt",
    # Env builder
    "EnvBuildSpec", "build_environment", "list_buildable_envs", "env_spec_prompt",
    # RL batching
    "RLBatchingConfig", "RLBatchingResult",
    "create_rl_batching_rollout", "run_rl_batching_round", "build_pipeline_requests",
]
