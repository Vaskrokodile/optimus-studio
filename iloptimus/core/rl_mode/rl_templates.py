"""No-code RL environment templates.

Provides a catalog of pre-built RL environments that the model can instantiate
without writing code. Each template is a declarative spec that maps to a
rl_factory environment class.

The catalog mirrors the 60+ environments in rl_factory.environments, organized
by domain and capability. The model selects a template by name and provides
problem-specific parameters — the builder handles the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvTemplate:
    """A no-code RL environment template."""

    name: str
    domain: str
    description: str
    env_class: str  # dotted path to the environment class
    generator_name: str  # name of the problem generator
    capabilities: list[str]  # what this environment trains
    default_batch_size: int = 16
    default_token_budget: int = 2048
    default_difficulty: float = 0.5
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "correctness": 1.0,
        "efficiency": 0.3,
        "anti_pattern": 0.5,
        "format": 0.1,
        "difficulty": 0.2,
    })
    batch_aware: bool = False  # supports vLLM batched generation
    supports_rl_batching: bool = True  # can be used with param-split RL


# ---------------------------------------------------------------------------
# Template catalog — organized by domain
# ---------------------------------------------------------------------------

_MATH_TEMPLATES = [
    EnvTemplate(
        name="proof-golf",
        domain="math",
        description="Shortest valid proof wins. Trains proof compression.",
        env_class="iloptimus.core.rl_factory.environments.ProofGolfEnv",
        generator_name="proofgolf_generator",
        capabilities=["proof-writing", "conciseness", "math-reasoning"],
        default_token_budget=1024,
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="formal-proof-gym",
        domain="math",
        description="Formal proof construction with step-by-step verification.",
        env_class="iloptimus.core.rl_factory.environments.FormalProofGymEnv",
        generator_name="formal_proof_gym_generator",
        capabilities=["formal-proofs", "logical-rigor", "step-verification"],
        default_token_budget=2048,
        default_difficulty=0.7,
    ),
    EnvTemplate(
        name="symbolic-equation-solver",
        domain="math",
        description="Solve symbolic equations with verification by substitution.",
        env_class="iloptimus.core.rl_factory.environments.SymbolicEquationSolverEnv",
        generator_name="symbolic_equation_solver_generator",
        capabilities=["algebra", "symbolic-manipulation", "verification"],
        default_difficulty=0.4,
    ),
    EnvTemplate(
        name="math-gap-arithmetic",
        domain="math",
        description="Find the missing value in arithmetic sequences.",
        env_class="iloptimus.core.rl_factory.environments.MathGapArithmeticEnv",
        generator_name="math_gap_arithmetic_generator",
        capabilities=["arithmetic", "pattern-recognition"],
        default_difficulty=0.3,
    ),
    EnvTemplate(
        name="sat-heuristic-discovery",
        domain="math",
        description="Discover heuristics for SAT solving.",
        env_class="iloptimus.core.rl_factory.environments.SatHeuristicDiscoveryEnv",
        generator_name="sat_heuristic_discovery_generator",
        capabilities=["sat-solving", "heuristic-design", "combinatorics"],
        default_difficulty=0.8,
    ),
]

_CODING_TEMPLATES = [
    EnvTemplate(
        name="surgical-patch",
        domain="coding",
        description="Minimal bug fix under a token budget. Traces surgical debugging.",
        env_class="iloptimus.core.rl_factory.environments.SurgicalPatchEnv",
        generator_name="surgical_patch_generator",
        capabilities=["debugging", "minimal-changes", "code-comprehension"],
        default_token_budget=1024,
        default_difficulty=0.5,
    ),
    EnvTemplate(
        name="refactor-arena",
        domain="coding",
        description="Produce equivalent but more concise code.",
        env_class="iloptimus.core.rl_factory.environments.RefactorArenaEnv",
        generator_name="refactor_arena_generator",
        capabilities=["refactoring", "code-compression", "equivalence"],
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="self-debug",
        domain="coding",
        description="Write code, run tests, fix failures iteratively.",
        env_class="iloptimus.core.rl_factory.environments.SelfDebugEnv",
        generator_name="self_debug_generator",
        capabilities=["self-correction", "testing", "iteration"],
        default_difficulty=0.5,
        batch_aware=True,
    ),
    EnvTemplate(
        name="test-driven-checkpoints",
        domain="coding",
        description="Pass incremental test suites in sequence.",
        env_class="iloptimus.core.rl_factory.environments.TestDrivenCheckpointsEnv",
        generator_name="test_driven_checkpoints_generator",
        capabilities=["tdd", "incremental-development", "test-passing"],
        default_difficulty=0.5,
    ),
    EnvTemplate(
        name="ast-refactor-chain",
        domain="coding",
        description="Chain AST-level refactoring operations.",
        env_class="iloptimus.core.rl_factory.environments.AstRefactorChainEnv",
        generator_name="ast_refactor_chain_generator",
        capabilities=["ast-manipulation", "refactoring-chains"],
        default_difficulty=0.7,
    ),
    EnvTemplate(
        name="spec-to-verified-code",
        domain="coding",
        description="Generate code from spec and verify it passes tests.",
        env_class="iloptimus.core.rl_factory.environments.SpecToVerifiedCodeEnv",
        generator_name="spec_to_verified_code_generator",
        capabilities=["spec-to-code", "verification", "code-generation"],
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="kernel-optimizer",
        domain="coding",
        description="Optimize kernel code for performance.",
        env_class="iloptimus.core.rl_factory.environments.KernelOptimizerEnv",
        generator_name="kernel_optimizer_generator",
        capabilities=["optimization", "performance", "low-level-code"],
        default_difficulty=0.8,
    ),
]

_REASONING_TEMPLATES = [
    EnvTemplate(
        name="trajectory-doctor",
        domain="reasoning",
        description="Diagnose the first wrong step in a flawed reasoning trace.",
        env_class="iloptimus.core.rl_factory.environments.TrajectoryDoctorEnv",
        generator_name="trajectory_doctor_generator",
        capabilities=["error-detection", "trace-analysis", "reasoning"],
        default_token_budget=2048,
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="contradiction-detection",
        domain="reasoning",
        description="Find logical inconsistencies in arguments.",
        env_class="iloptimus.core.rl_factory.environments.ContradictionDetectionEnv",
        generator_name="contradiction_detection_generator",
        capabilities=["logic", "consistency-checking", "argument-analysis"],
        default_difficulty=0.5,
    ),
    EnvTemplate(
        name="logic-puzzle-suite",
        domain="reasoning",
        description="Solve logic puzzles (knights/knaves, constraint satisfaction).",
        env_class="iloptimus.core.rl_factory.environments.LogicPuzzleSuiteEnv",
        generator_name="logic_puzzle_suite_generator",
        capabilities=["logic-puzzles", "deduction", "constraint-satisfaction"],
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="trace-surgery",
        domain="reasoning",
        description="Fix a specific step in a reasoning trace.",
        env_class="iloptimus.core.rl_factory.environments.TraceSurgeryEnv",
        generator_name="trace_surgery_generator",
        capabilities=["trace-repair", "surgical-reasoning"],
        default_difficulty=0.7,
    ),
    EnvTemplate(
        name="iterative-refinement",
        domain="reasoning",
        description="Iteratively refine a rough answer into a precise one.",
        env_class="iloptimus.core.rl_factory.environments.IterativeRefinementEnv",
        generator_name="iterative_refinement_generator",
        capabilities=["refinement", "precision", "iteration"],
        default_difficulty=0.5,
        batch_aware=True,
    ),
    EnvTemplate(
        name="self-consistency",
        domain="reasoning",
        description="Generate multiple solutions and vote on the best.",
        env_class="iloptimus.core.rl_factory.environments.SelfConsistencyEnv",
        generator_name="self_consistency_generator",
        capabilities=["self-consistency", "voting", "diversity"],
        default_difficulty=0.5,
        batch_aware=True,
    ),
]

_AGENTIC_TEMPLATES = [
    EnvTemplate(
        name="tool-golf",
        domain="agentic",
        description="Complete a task with minimal tool calls.",
        env_class="iloptimus.core.rl_factory.environments.ToolGolfEnv",
        generator_name="toolgolf_generator",
        capabilities=["tool-use", "efficiency", "planning"],
        default_token_budget=2048,
        default_difficulty=0.6,
    ),
    EnvTemplate(
        name="checkpoint-race",
        domain="agentic",
        description="Complete multi-step tasks fastest.",
        env_class="iloptimus.core.rl_factory.environments.CheckpointRaceEnv",
        generator_name="checkpoint_race_generator",
        capabilities=["multi-step", "speed", "task-completion"],
        default_difficulty=0.5,
    ),
    EnvTemplate(
        name="error-recovery",
        domain="agentic",
        description="Recover from simulated failures in a pipeline.",
        env_class="iloptimus.core.rl_factory.environments.ErrorRecoveryEnv",
        generator_name="error_recovery_generator",
        capabilities=["error-recovery", "resilience", "adaptation"],
        default_difficulty=0.7,
    ),
    EnvTemplate(
        name="context-distiller",
        domain="agentic",
        description="Extract key information from noisy context.",
        env_class="iloptimus.core.rl_factory.environments.ContextDistillerEnv",
        generator_name="context_distiller_generator",
        capabilities=["information-extraction", "summarization", "focus"],
        default_difficulty=0.5,
    ),
    EnvTemplate(
        name="best-of-n",
        domain="agentic",
        description="Generate N responses and pick the best.",
        env_class="iloptimus.core.rl_factory.environments.BestOfNEnv",
        generator_name="bestofn_generator",
        capabilities=["selection", "quality-judgment", "diversity"],
        default_difficulty=0.5,
        batch_aware=True,
    ),
    EnvTemplate(
        name="parallel-explorer",
        domain="agentic",
        description="Explore diverse approaches to a problem in parallel.",
        env_class="iloptimus.core.rl_factory.environments.ParallelExplorerEnv",
        generator_name="parallel_explorer_generator",
        capabilities=["exploration", "diversity", "parallel-thinking"],
        default_difficulty=0.6,
        batch_aware=True,
    ),
    EnvTemplate(
        name="debugging-race",
        domain="agentic",
        description="Find and fix bugs faster than the baseline.",
        env_class="iloptimus.core.rl_factory.environments.DebuggingRaceEnv",
        generator_name="debugging_race_generator",
        capabilities=["debugging", "speed", "bug-detection"],
        default_difficulty=0.6,
    ),
]

_ALL_TEMPLATES = _MATH_TEMPLATES + _CODING_TEMPLATES + _REASONING_TEMPLATES + _AGENTIC_TEMPLATES

_TEMPLATE_MAP: dict[str, EnvTemplate] = {t.name: t for t in _ALL_TEMPLATES}


def list_env_templates(domain: str | None = None) -> list[EnvTemplate]:
    """List all environment templates, optionally filtered by domain."""
    if domain is None:
        return list(_ALL_TEMPLATES)
    return [t for t in _ALL_TEMPLATES if t.domain == domain]


def get_env_template(name: str) -> EnvTemplate | None:
    """Get a specific environment template by name."""
    return _TEMPLATE_MAP.get(name)


def recommend_templates(
    weak_capabilities: list[str],
    domain: str | None = None,
    count: int = 3,
) -> list[EnvTemplate]:
    """Recommend templates that target the model's weak capabilities."""
    candidates = list_env_templates(domain)
    scored: list[tuple[int, EnvTemplate]] = []
    for template in candidates:
        score = sum(1 for cap in template.capabilities if any(wc in cap for wc in weak_capabilities))
        if score > 0:
            scored.append((score, template))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    return [t for _, t in scored[:count]]


def template_catalog_prompt(domain: str | None = None, max_chars: int = 4_000) -> str:
    """Build a compact catalog of available templates for prompt injection."""
    templates = list_env_templates(domain)
    lines = ["Available RL environment templates (use these — no coding needed):"]
    for t in templates:
        line = f"- {t.name} [{t.domain}]: {t.description} (trains: {', '.join(t.capabilities[:3])})"
        if len("\n".join(lines)) + len(line) > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)
