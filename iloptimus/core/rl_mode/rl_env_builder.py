"""No-code RL environment builder.

Instantiates rl_factory environments from template specs without writing code.
The model provides a template name and parameters — the builder handles
importing the environment class, creating the problem generator, and
configuring the reward function.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from .rl_templates import EnvTemplate, get_env_template, list_env_templates
from ..rl_factory.core.base import BaseReasoningEnv, Problem
from ..rl_factory.core.reward import RewardConfig
from ..rl_factory.core.anti_patterns import AntiPatternDetector
from ..rl_factory.core.token_budget import TokenBudget


@dataclass
class EnvBuildSpec:
    """Spec for building an RL environment from a template."""

    template_name: str
    # Override defaults
    batch_size: int | None = None
    token_budget: int | None = None
    difficulty: float | None = None
    reward_weights: dict[str, float] | None = None
    # Problem generation
    n_problems: int = 50
    seed: int = 42
    # Extra metadata for the environment
    metadata: dict[str, Any] = field(default_factory=dict)


def build_environment(spec: EnvBuildSpec) -> BaseReasoningEnv:
    """Build an RL environment from a template spec — no code needed.

    1. Looks up the template by name
    2. Imports the environment class
    3. Imports the problem generator
    4. Generates problems
    5. Configures the reward function
    6. Returns the ready-to-use environment
    """
    template = get_env_template(spec.template_name)
    if template is None:
        raise ValueError(f"Unknown environment template: {spec.template_name}")

    # Import the environment class
    env_class = _import_class(template.env_class)

    # Import the generator
    generator = _import_generator(template.generator_name)

    # Generate problems
    import random
    rng = random.Random(spec.seed)
    problems: list[Problem] = []
    for i in range(spec.n_problems):
        try:
            problem = generator(rng.randint(0, 2**31 - 1))
            # Override difficulty if specified
            if spec.difficulty is not None:
                problem = Problem(
                    id=problem.id,
                    prompt=problem.prompt,
                    difficulty=spec.difficulty,
                    metadata=problem.metadata,
                    token_budget=spec.token_budget or problem.token_budget,
                    source=problem.source,
                )
            problems.append(problem)
        except Exception:
            continue

    if not problems:
        raise RuntimeError(f"Generator {template.generator_name} produced no problems")

    # Configure reward
    reward_weights = spec.reward_weights or template.reward_weights
    reward_config = RewardConfig(
        w_correct=reward_weights.get("correctness", 1.0),
        w_efficiency=reward_weights.get("efficiency", 0.3),
        w_anti_pattern=reward_weights.get("anti_pattern", 0.5),
        w_format=reward_weights.get("format", 0.1),
        w_difficulty=reward_weights.get("difficulty", 0.2),
    )

    # Build the environment
    token_budget = spec.token_budget or template.default_token_budget
    batch_size = spec.batch_size or template.default_batch_size

    # Check if the environment is batch-aware
    if template.batch_aware:
        from ..rl_factory.core.batch_base import BatchEnvBase
        env = env_class(
            problems=problems,
            reward_config=reward_config,
            batch_size=batch_size,
        )
    else:
        env = env_class(
            problems=problems,
            reward_config=reward_config,
        )

    return env


def _import_class(dotted_path: str) -> type:
    """Import a class from a dotted path like 'module.path.ClassName'."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Invalid class path: {dotted_path}")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _import_generator(generator_name: str):
    """Import a generator function from the environments package."""
    from ..rl_factory import environments
    return getattr(environments, generator_name)


def list_buildable_envs(domain: str | None = None) -> list[dict[str, Any]]:
    """List all buildable environments with their specs."""
    templates = list_env_templates(domain)
    return [
        {
            "name": t.name,
            "domain": t.domain,
            "description": t.description,
            "capabilities": t.capabilities,
            "batch_aware": t.batch_aware,
            "default_batch_size": t.default_batch_size,
            "default_token_budget": t.default_token_budget,
            "default_difficulty": t.default_difficulty,
        }
        for t in templates
    ]


def env_spec_prompt(max_chars: int = 3_000) -> str:
    """Build a compact prompt showing available environment templates."""
    templates = list_env_templates()
    lines = ["Build RL environments from these templates (no code needed):"]
    for t in templates:
        line = f"- {t.name}: {t.description}"
        if len("\n".join(lines)) + len(line) > max_chars:
            break
        lines.append(line)
    lines.append("\nTo build: specify template_name and optional overrides (batch_size, difficulty, token_budget).")
    return "\n".join(lines)
