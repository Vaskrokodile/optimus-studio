"""Environment-as-RL-agent (#26) — adversarial LLM environment generator.

The generator is the same local model invoked through the existing inference
path (``run_json_completion``). Its "policy" is its prompt context; its
"reward" is computed after the trainee rollouts and fed back into its context.
It returns one JSON action per call:

    {"op": "mutate_existing|compose_skills|synthesize_new|persistent_world",
     "params": {...}, "difficulty": 0.45, "estimated_success_rate": 0.15}

Per the user's design decision, ``synthesize_new`` MAY emit a generated
``verify.py`` script that is run in the existing sandbox. The generator targets
a 5-30% trainee success rate and is penalized for impossible, trivial,
duplicate, or overly expensive tasks.

See ``RESEARCH_meta_architecture.md`` Section 4.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .persistent_world import TaskSpec
from .rl_factory.environments.environment_mutation import (
    MUTATION_TYPES,
    DynamicsMutator,
)
from .rl_factory.core.base import Problem


# ---------------------------------------------------------------------------
# Generator action / observation
# ---------------------------------------------------------------------------


@dataclass
class GeneratorAction:
    op: str  # mutate_existing | compose_skills | synthesize_new | persistent_world
    params: dict = field(default_factory=dict)
    difficulty: float = 0.5
    estimated_success_rate: float = 0.15

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rollout:
    success: bool
    score: float = 0.0
    tokens: int = 0


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


GENERATOR_PROMPT = """You are the Adversarial Environment Generator for an adaptive RL training loop.
Your goal is to create a task that the Trainee model finds difficult but can still solve.

### Observation
- Frontier skills (under-trained): {frontier_skills}
- Recent failure patterns: {recent_failures}
- Compute budget remaining: {compute_remaining}
- Recent tasks (do not repeat): {recent_task_summaries}

### Allowed actions
Return exactly one JSON object with this shape:
{{
  "op": "mutate_existing" | "compose_skills" | "synthesize_new" | "persistent_world",
  "params": {{
    // mutate_existing: {{"base_task_id": "...", "mutation_type": "hidden_constraint|misleading_info|partial_observability|requirement_change|resource_constraint"}}
    // compose_skills: {{"skill_keys": ["tool_call:read_file", "tool_call:grep_search"], "difficulty": 0.6}}
    // synthesize_new: {{"domain": "coding", "prompt": "...", "verifier": {{"type": "code_execution", "verify_py": "def verify(answer):\\n    ..."}}}}
    // persistent_world: {{"trigger_type": "debugging", "urgency": 0.8}}
  }},
  "difficulty": 0.0-1.0,
  "estimated_success_rate": 0.0-1.0
}}

### Rules
1. Target a trainee success rate between 5% and 30%.
2. The task must be verifiable (has an objective grader or simulator).
3. Do not repeat a recent task; prefer novel skill combinations.
4. If the trainee is failing on `tool_call:grep_search` followed by `tool_call:write_file`, compose a task that requires that exact sequence.
5. If the Persistent World has an active trigger matching a frontier skill, prefer `persistent_world`.

Return only the JSON object."""


# ---------------------------------------------------------------------------
# EnvGeneratorAgent
# ---------------------------------------------------------------------------


class EnvGeneratorAgent:
    """Adversarial LLM environment generator.

    ``generate_fn`` is a callable ``(prompt: str) -> str`` that runs the local
    model. In production this is wired to ``run_json_completion``; in tests a
    stub returns a fixed action.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        persistent_world=None,
        base_problems: list[Problem] | None = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.persistent_world = persistent_world
        self.base_problems = base_problems or []
        self.recent_tasks: list[TaskSpec] = []
        self.recent_embeddings: list[list[float]] = []
        self.recent_rewards: list[float] = []

    # -- observation ------------------------------------------------------

    def _build_observation(
        self,
        frontier_skills: list[str],
        recent_failures: list[dict[str, Any]],
        compute_used: int,
        compute_budget: int,
    ) -> str:
        return GENERATOR_PROMPT.format(
            frontier_skills=json.dumps(frontier_skills[:20]),
            recent_failures=json.dumps(recent_failures[-10:]),
            compute_remaining=max(0, compute_budget - compute_used),
            recent_task_summaries=json.dumps(
                [t.task_id for t in self.recent_tasks[-10:]]
            ),
        )

    # -- action production ------------------------------------------------

    def produce_action(
        self,
        frontier_skills: list[str] | None = None,
        recent_failures: list[dict[str, Any]] | None = None,
        compute_used: int = 0,
        compute_budget: int = 100_000,
    ) -> GeneratorAction:
        """Ask the LLM for the next generator action."""
        if self.generate_fn is None:
            return self._default_action(frontier_skills or [], recent_failures or [])

        prompt = self._build_observation(
            frontier_skills or [],
            recent_failures or [],
            compute_used,
            compute_budget,
        )
        raw = self.generate_fn(prompt)
        return self._parse_action(raw)

    def _parse_action(self, raw: str) -> GeneratorAction:
        """Extract a JSON action from the model output."""
        text = raw.strip()
        start = text.find("{")
        if start < 0:
            return self._default_action([], [])
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            return self._default_action([], [])
        op = str(obj.get("op", "synthesize_new"))
        if op not in ("mutate_existing", "compose_skills", "synthesize_new", "persistent_world"):
            op = "synthesize_new"
        return GeneratorAction(
            op=op,
            params=obj.get("params", {}) or {},
            difficulty=float(obj.get("difficulty", 0.5) or 0.5),
            estimated_success_rate=float(obj.get("estimated_success_rate", 0.15) or 0.15),
        )

    def _default_action(
        self,
        frontier_skills: list[str],
        recent_failures: list[dict[str, Any]],
    ) -> GeneratorAction:
        """Cold-start default: prefer persistent_world, else compose_skills."""
        if self.persistent_world and self.persistent_world.active_triggers():
            return GeneratorAction(
                op="persistent_world",
                params={"trigger_type": self.persistent_world.active_triggers()[0][0]},
                difficulty=0.5,
                estimated_success_rate=0.15,
            )
        skills = frontier_skills[:2] or ["tool_call:read_file", "tool_call:grep_search"]
        return GeneratorAction(
            op="compose_skills",
            params={"skill_keys": skills, "difficulty": 0.6},
            difficulty=0.6,
            estimated_success_rate=0.15,
        )

    # -- task materialization ---------------------------------------------

    def materialize(self, action: GeneratorAction) -> TaskSpec:
        """Turn a generator action into a concrete ``TaskSpec``."""
        if action.op == "persistent_world":
            return self._materialize_world(action)
        if action.op == "mutate_existing":
            return self._materialize_mutate(action)
        if action.op == "compose_skills":
            return self._materialize_compose(action)
        return self._materialize_synthesize(action)

    def _materialize_world(self, action: GeneratorAction) -> TaskSpec:
        if self.persistent_world is None:
            return self._materialize_compose(action)
        focus = action.params.get("trigger_type")
        return self.persistent_world.derive_task(focus_skill=focus)

    def _materialize_mutate(self, action: GeneratorAction) -> TaskSpec:
        base_id = action.params.get("base_task_id", "")
        mutation_type = action.params.get("mutation_type", "hidden_constraint")
        if mutation_type not in MUTATION_TYPES:
            mutation_type = "hidden_constraint"
        base = self._find_base_problem(base_id)
        if base is None:
            return self._materialize_compose(action)
        mutator = DynamicsMutator(base)
        mutated = mutator.mutate(mutation_type=mutation_type)
        return TaskSpec(
            task_id=f"gen-mutate-{mutation_type}-{base.id}",
            prompt=mutated.prompt,
            domain="mutated",
            source="mutated",
            difficulty=action.difficulty,
            metadata={
                "base_task_id": base.id,
                "mutation_type": mutation_type,
                "verifier": {"type": "rl_factory_problem", "problem_id": mutated.id},
            },
        )

    def _materialize_compose(self, action: GeneratorAction) -> TaskSpec:
        skills = action.params.get("skill_keys", []) or ["tool_call:read_file"]
        difficulty = float(action.params.get("difficulty", action.difficulty))
        skill_str = " -> ".join(skills)
        skill_key = "|".join(skills)
        prompt = (
            f"Complete this multi-step task requiring the following skill sequence: {skill_str}.\n"
            f"Use the tools in the listed order. Return your tool calls inside "
            f"<tool>{{\"name\": \"...\", \"args\": {{...}}}}</tool> blocks and the final "
            f"answer inside <answer>...</answer>.\n\n"
            f"Scenario: a service is degraded. Read the relevant log, search for the error "
            f"pattern, then apply the fix. Demonstrate each skill in order."
        )
        return TaskSpec(
            task_id=f"gen-compose-{hashlib.md5(skill_key.encode()).hexdigest()[:8]}",
            prompt=prompt,
            domain="composed",
            source="composed",
            difficulty=difficulty,
            metadata={
                "skill_keys": skills,
                "verifier": {"type": "skill_sequence", "required_skills": skills},
            },
        )

    def _materialize_synthesize(self, action: GeneratorAction) -> TaskSpec:
        domain = action.params.get("domain", "coding")
        prompt = action.params.get("prompt", "Write a function that solves the stated problem.")
        verifier = action.params.get("verifier", {}) or {}
        verify_py = verifier.get("verify_py", "")
        return TaskSpec(
            task_id=f"gen-synth-{hashlib.md5(prompt.encode()).hexdigest()[:8]}",
            prompt=prompt,
            domain=domain,
            source="generated",
            difficulty=action.difficulty,
            metadata={
                "verifier": {
                    "type": verifier.get("type", "code_execution"),
                    "verify_py": verify_py,
                },
            },
        )

    def _find_base_problem(self, base_id: str) -> Problem | None:
        if not self.base_problems:
            return None
        if base_id:
            for p in self.base_problems:
                if p.id == base_id:
                    return p
        return self.base_problems[0]

    # -- reward / feedback ------------------------------------------------

    def record_outcome(
        self,
        task: TaskSpec,
        rollouts: list[Rollout],
        compute_used: int,
        compute_budget: int,
    ) -> float:
        """Compute the generator reward and append the task to recent history."""
        reward = generator_reward(
            rollouts=rollouts,
            task=task,
            recent_tasks=self.recent_tasks,
            compute_used=compute_used,
            compute_budget=compute_budget,
        )
        self.recent_tasks.append(task)
        self.recent_embeddings.append(_task_embedding(task))
        self.recent_rewards.append(reward)
        if len(self.recent_tasks) > 100:
            self.recent_tasks = self.recent_tasks[-100:]
            self.recent_embeddings = self.recent_embeddings[-100:]
            self.recent_rewards = self.recent_rewards[-100:]
        return reward


# ---------------------------------------------------------------------------
# Generator reward
# ---------------------------------------------------------------------------


def generator_reward(
    rollouts: list[Rollout],
    task: TaskSpec,
    recent_tasks: list[TaskSpec],
    compute_used: int,
    compute_budget: int,
    best_expert_score: float = 1.0,
) -> float:
    """Regret-based generator reward.

    reward = regret - solvability_penalty - novelty_penalty - 0.05*compute_cost
    where regret = best_expert_score - mean_trainee_score.
    """
    if not rollouts:
        return -1.0
    pass_rate = sum(1.0 for r in rollouts if r.success) / len(rollouts)
    mean_score = sum(r.score for r in rollouts) / len(rollouts)

    regret = best_expert_score - mean_score

    if pass_rate < 0.05:
        solvability_penalty = 0.5
    elif pass_rate > 0.30:
        solvability_penalty = pass_rate - 0.30
    else:
        solvability_penalty = 0.0

    novelty_penalty = _max_task_similarity(task, recent_tasks)
    compute_cost = compute_used / max(compute_budget, 1)

    return regret - solvability_penalty - novelty_penalty - 0.05 * compute_cost


def _task_embedding(task: TaskSpec) -> list[float]:
    """Cheap bag-of-chars embedding for novelty comparison."""
    text = (task.task_id + " " + task.prompt).lower()
    vec = [0.0] * 16
    for i, ch in enumerate(text):
        vec[i % 16] += ord(ch) / 256.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _max_task_similarity(task: TaskSpec, recent_tasks: list[TaskSpec]) -> float:
    if not recent_tasks:
        return 0.0
    emb = _task_embedding(task)
    max_sim = 0.0
    for prev in recent_tasks[-50:]:
        sim = _cosine(emb, _task_embedding(prev))
        if sim > max_sim:
            max_sim = sim
    return max(0.0, max_sim - 0.70)


# ---------------------------------------------------------------------------
# Generated verify.py runner (per the user's design decision: allowed)
# ---------------------------------------------------------------------------


def run_generated_verifier(verify_py: str, answer: str, timeout: float = 10.0) -> dict[str, Any]:
    """Run a generated ``verify.py`` in a subprocess sandbox.

    The verifier script must define ``verify(answer) -> dict`` returning at
    least ``{"correct": bool, "score": float}``. Returns ``{"error": ...}`` on
    failure. Reuses the same subprocess pattern as ``grader._run_sandbox`` but
    writes the verifier itself to a temp file (the generated-code path the user
    explicitly approved).
    """
    if not verify_py or not verify_py.strip():
        return {"error": "empty_verifier", "correct": False, "score": 0.0}

    harness = verify_py + "\n\n" + (
        "import json, sys\n"
        "if __name__ == '__main__':\n"
        "    ans = sys.argv[1] if len(sys.argv) > 1 else ''\n"
        "    try:\n"
        "        res = verify(ans)\n"
        "        if not isinstance(res, dict):\n"
        "            res = {'correct': bool(res), 'score': float(bool(res))}\n"
        "        print(json.dumps(res))\n"
        "    except Exception as e:\n"
        "        print(json.dumps({'error': str(e), 'correct': False, 'score': 0.0}))\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, answer],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (result.stdout or "").strip()
        lines = out.splitlines()
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        return {"error": (result.stderr or "")[-500:], "correct": False, "score": 0.0}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT", "correct": False, "score": 0.0}
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "EnvGeneratorAgent",
    "GeneratorAction",
    "Rollout",
    "generator_reward",
    "run_generated_verifier",
    "GENERATOR_PROMPT",
]
