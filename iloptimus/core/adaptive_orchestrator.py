"""Adaptive Orchestrator — the main self-evolving RL loop.

Composes the foundation modules (skill graph, capability profiler, frontier
sampler, capability metrics) with the environment systems (env generator agent,
persistent world, session-to-env failure extraction, existing rl_factory
environments) into a single adaptive loop:

    1. Inspect current model capability (skill graph / profiler).
    2. Select a frontier or weak skill (frontier sampler + composition query).
    3. Generate or select an environment (env generator / persistent world /
       existing taskset arm).
    4. Collect trainee rollouts.
    5. Grade them deterministically (grader / state-machine / generated verify.py).
    6. Run counterfactuals / recombination where useful (existing modules).
    7. Update capability and failure memory.
    8. Produce a training signal / curriculum entry.
    9. Persist state and metrics.
   10. Repeat according to budget and loop configuration.

See ``RESEARCH_meta_architecture.md`` Sections 2 and 6.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .capability_metrics import capability_gain, intelligence_density, tool_selection_entropy
from .capability_profiler import CapabilityProfiler
from .env_generator_agent import EnvGeneratorAgent, Rollout, run_generated_verifier
from .frontier_sampler import FrontierProfile, FrontierSampler, RolloutKey, load_frontier_profile
from .grader import GradedResult, build_prompt, grade_response, get_num_tasks
from .persistent_world import PersistentWorld, TaskSpec
from .session_to_env import FailureSpec, append_failure_spec, extract_failure_spec, load_failure_specs
from .skill_graph import SkillGraphProfile
from .storage import atomic_write_json, orchestrator_dir


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    loop_id: str
    model_id: str
    rollouts_per_task: int = 8
    temperature: float = 0.6
    frontier_min: float = 0.05
    frontier_max: float = 0.30
    max_iterations: int = 50
    checkpoint_every: int = 5
    world_task_prob: float = 0.25
    compute_budget: int = 100_000
    model_fingerprint: str = ""


@dataclass
class Episode:
    iteration: int
    task_id: str
    task_source: str  # taskset | mutated | generated | persistent_world | composed
    domain: str
    skill_focus: str
    rollouts: int
    best_score: float
    mean_score: float
    pass_rate: float
    tokens: int
    generator_reward: float
    timestamp: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSignal:
    iteration: int
    task: dict[str, Any]
    rollouts: list[dict[str, Any]]
    reward: float | None
    skill_focus: str
    capability_gain: float
    intelligence_density: float

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorState:
    loop_id: str
    iteration: int = 0
    best_score: float = 0.0
    score_history: list[float] = field(default_factory=list)
    recent_tasks: list[dict[str, Any]] = field(default_factory=list)
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    persistent_world_id: str | None = None
    compute_used: int = 0

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# OrchestratorStore — persistence under ~/.iloptimus/orchestrator/<loop_id>/
# ---------------------------------------------------------------------------


class OrchestratorStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or orchestrator_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def loop_dir(self, loop_id: str) -> Path:
        d = self.root / loop_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def state_path(self, loop_id: str) -> Path:
        return self.loop_dir(loop_id) / "state.json"

    def episodes_path(self, loop_id: str) -> Path:
        return self.loop_dir(loop_id) / "episodes.jsonl"

    def checkpoints_dir(self, loop_id: str) -> Path:
        d = self.loop_dir(loop_id) / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load(self, loop_id: str) -> OrchestratorState:
        path = self.state_path(loop_id)
        if path.exists():
            try:
                return OrchestratorState(**json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return OrchestratorState(loop_id=loop_id)

    def save(self, state: OrchestratorState) -> None:
        atomic_write_json(self.state_path(state.loop_id), state.public())

    def append_episode(self, loop_id: str, episode: Episode) -> None:
        path = self.episodes_path(loop_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode.public(), ensure_ascii=False) + "\n")

    def checkpoint(self, loop_id: str, state: OrchestratorState) -> None:
        atomic_write_json(self.checkpoints_dir(loop_id) / f"{state.iteration}.json", state.public())


# ---------------------------------------------------------------------------
# AdaptiveOrchestrator
# ---------------------------------------------------------------------------


class AdaptiveOrchestrator:
    """The main adaptive self-evolving RL loop."""

    def __init__(
        self,
        config: AdaptiveConfig,
        model: Any,  # object with .generate(prompt) -> str, or a callable
        generate_fn: Callable[[str], str] | None = None,
        profiler: CapabilityProfiler | None = None,
        env_generator: EnvGeneratorAgent | None = None,
        persistent_world: PersistentWorld | None = None,
        store: OrchestratorStore | None = None,
        frontier_profile: FrontierProfile | None = None,
    ) -> None:
        self.config = config
        self.model = model
        # ``generate_fn`` is the trainee generation entry (defaults to model.generate).
        self.generate_fn = generate_fn or getattr(model, "generate", None)
        if self.generate_fn is None:
            # Fall back to treating model itself as a callable.
            self.generate_fn = model  # type: ignore[assignment]
        self.profiler = profiler or CapabilityProfiler()
        self.persistent_world = persistent_world
        self.env_generator = env_generator or EnvGeneratorAgent(
            generate_fn=None,  # generator LLM wired separately below if available
            persistent_world=self.persistent_world,
        )
        self.store = store or OrchestratorStore()
        self.state = self.store.load(config.loop_id)
        if self.state.persistent_world_id and self.persistent_world is None:
            self.persistent_world = PersistentWorld.load(self.state.persistent_world_id)
            self.env_generator.persistent_world = self.persistent_world
        self.frontier_profile = frontier_profile or load_frontier_profile(config.model_id)
        self.skill_profile = self.profiler.load(config.model_id, config.model_fingerprint)

    # -- public API -------------------------------------------------------

    def iteration(self) -> TrainingSignal:
        """Run one orchestrator iteration and return a training signal."""
        self.state.iteration += 1

        # 1-2. Select a frontier skill / task.
        focus_skill, task, selection = self._select_task()

        # 3. Collect rollouts.
        rollouts, graded_list, tokens = self._collect_rollouts(task, selection)

        # 4. Grade (already done in collect); compute pass rate.
        pass_rate = sum(1 for g in graded_list if g.correctness >= 1.0) / max(1, len(graded_list))
        best_score = max((g.score for g in graded_list), default=0.0)
        mean_score = sum(g.score for g in graded_list) / max(1, len(graded_list))

        # 5. Update capability + failure memory.
        self._update_capability(task, selection, graded_list, focus_skill)
        self._extract_failures(task, graded_list, rollouts)

        # 6. Generator reward feedback.
        gen_reward = self.env_generator.record_outcome(
            task=task,
            rollouts=[Rollout(success=g.correctness >= 1.0, score=g.score, tokens=t) for g, t in zip(graded_list, tokens)],
            compute_used=self.state.compute_used,
            compute_budget=self.config.compute_budget,
        )

        # 7. Metrics.
        cap_gain = self._capability_gain_for_task(selection)
        flops = sum(t * 2 * 7 for t in tokens)  # cheap approx: 7B params
        idensity = intelligence_density(cap_gain, sum(tokens), flops)

        # 8. Persist.
        episode = Episode(
            iteration=self.state.iteration,
            task_id=task.task_id,
            task_source=task.source,
            domain=task.domain,
            skill_focus=focus_skill,
            rollouts=len(rollouts),
            best_score=best_score,
            mean_score=mean_score,
            pass_rate=pass_rate,
            tokens=sum(tokens),
            generator_reward=gen_reward,
        )
        self.state.score_history.append(best_score)
        self.state.best_score = max(self.state.best_score, best_score)
        self.state.recent_tasks.append(task.public())
        if len(self.state.recent_tasks) > 50:
            self.state.recent_tasks = self.state.recent_tasks[-50:]
        self.state.compute_used += sum(tokens)
        self.store.append_episode(self.config.loop_id, episode)
        if self.state.iteration % self.config.checkpoint_every == 0:
            self.store.checkpoint(self.config.loop_id, self.state)
        self.store.save(self.state)
        self.profiler.save(self.skill_profile)
        FrontierSampler(self.frontier_profile).save()

        return TrainingSignal(
            iteration=self.state.iteration,
            task=task.public(),
            rollouts=[{"score": g.score, "correctness": g.correctness} for g in graded_list],
            reward=gen_reward,
            skill_focus=focus_skill,
            capability_gain=cap_gain,
            intelligence_density=idensity,
        )

    def run(self, iterations: int | None = None) -> list[TrainingSignal]:
        n = iterations or self.config.max_iterations
        return [self.iteration() for _ in range(n)]

    # -- task selection ---------------------------------------------------

    def _select_task(self) -> tuple[str, TaskSpec, dict[str, Any]]:
        """Pick a focus skill and a task, preferring frontier / world / generator."""
        # Frontier composition from the skill graph.
        comps = self.profiler.suggest_compositions(self.skill_profile, k=5)
        focus_skill = sorted(comps[0].node_ids)[0] if comps else "tool-calling"

        # Decide task source: persistent_world (prob), else generator, else taskset arm.
        use_world = (
            self.persistent_world is not None
            and (
                random.random() < self.config.world_task_prob
                or (self.persistent_world.active_triggers() and not comps)
            )
        )
        if use_world:
            task = self.persistent_world.derive_task(focus_skill=focus_skill)
            return focus_skill, task, {"source": "persistent_world"}

        # Generator action (uses its own LLM if wired; else default).
        action = self.env_generator.produce_action(
            frontier_skills=[focus_skill],
            recent_failures=self.state.recent_failures,
            compute_used=self.state.compute_used,
            compute_budget=self.config.compute_budget,
        )
        if action.op in ("mutate_existing", "compose_skills", "synthesize_new", "persistent_world"):
            task = self.env_generator.materialize(action)
            return focus_skill, task, {"source": task.source, "action": action.public()}

        # Fallback: sample a taskset arm via the frontier sampler.
        key = self._sample_taskset_arm()
        task = TaskSpec(
            task_id=f"taskset-{key.domain}-{key.task_idx}",
            prompt=build_prompt(key.domain, key.task_idx),
            domain=key.domain,
            source="taskset",
            difficulty=0.5,
            metadata={"domain": key.domain, "task_idx": key.task_idx},
        )
        return focus_skill, task, {"source": "taskset", "key": asdict(key)}

    def _sample_taskset_arm(self) -> RolloutKey:
        # Ensure the frontier profile knows the tool domains.
        from .grader import _TOOL_DOMAINS
        for domain in _TOOL_DOMAINS:
            n = get_num_tasks(domain)
            for t in range(n):
                self.frontier_profile.ensure_task(domain, domain, t)
        sampler = FrontierSampler(self.frontier_profile)
        return sampler.sample()

    # -- rollouts ---------------------------------------------------------

    def _collect_rollouts(
        self, task: TaskSpec, selection: dict[str, Any]
    ) -> tuple[list[str], list[GradedResult], list[int]]:
        rollouts: list[str] = []
        graded: list[GradedResult] = []
        tokens: list[int] = []
        for _ in range(self.config.rollouts_per_task):
            response = self._generate(task.prompt)
            tok = self._estimate_tokens(response)
            rollouts.append(response)
            tokens.append(tok)
            graded.append(self._grade(task, response, selection))
        return rollouts, graded, tokens

    def _generate(self, prompt: str) -> str:
        try:
            return self.generate_fn(prompt)
        except Exception:
            return ""

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _grade(self, task: TaskSpec, response: str, selection: dict[str, Any]) -> GradedResult:
        # Persistent world / state-machine tasks: run the projected simulator.
        if task.source == "persistent_world" and "simulator" in task.metadata:
            return self._grade_state_machine(task, response)
        # Generated verify.py tasks.
        if task.source == "generated":
            verifier = task.metadata.get("verifier", {}) or {}
            if verifier.get("type") == "code_execution" and verifier.get("verify_py"):
                answer = self._extract_answer(response)
                res = run_generated_verifier(verifier["verify_py"], answer)
                return GradedResult(
                    score=float(res.get("score", 0.0)),
                    correctness=1.0 if res.get("correct") else 0.0,
                    reasoning_quality=float(res.get("score", 0.0)),
                    info=res,
                )
        # Taskset arm: use the standard grader.
        if task.source == "taskset":
            domain = task.metadata.get("domain", "")
            task_idx = int(task.metadata.get("task_idx", 0))
            try:
                return grade_response(domain, task_idx, response)
            except Exception as e:
                return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0, info={"error": str(e)})
        # Composed / mutated / fallback: a light heuristic grade.
        return self._heuristic_grade(task, response)

    def _grade_state_machine(self, task: TaskSpec, response: str) -> GradedResult:
        from .stateful_environments import StateMachineRuntime, _trajectory_from_response
        sim = task.metadata["simulator"]
        try:
            runtime = StateMachineRuntime(sim)
            actions = _trajectory_from_response(response, runtime.action_names)
            total = 0.0
            for a in actions:
                if runtime.terminated:
                    break
                res = runtime.step(a)
                total += res.reward
            success = runtime.success
            # Normalize reward to [0,1] roughly.
            score = max(0.0, min(1.0, (total + 1.0) / 2.0))
            final_state = runtime.state
            # Persist outcome to the world and advance simulated time.
            if self.persistent_world is not None:
                self.persistent_world.apply_outcome(task, final_state, success)
                self.persistent_world.advance()
            return GradedResult(
                score=score, correctness=1.0 if success else 0.0,
                reasoning_quality=score, info={"final_state": final_state, "outcome": runtime.outcome},
            )
        except Exception as e:
            return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0, info={"error": str(e)})

    def _heuristic_grade(self, task: TaskSpec, response: str) -> GradedResult:
        # Skill-sequence verifier: check that the required skills appear in order.
        verifier = task.metadata.get("verifier", {}) or {}
        if verifier.get("type") == "skill_sequence":
            required = verifier.get("required_skills", [])
            text = response.lower()
            ordered = True
            pos = 0
            for skill in required:
                needle = skill.split(":")[-1].lower()
                idx = text.find(needle, pos)
                if idx < 0:
                    ordered = False
                    break
                pos = idx
            score = 1.0 if ordered else 0.0
            return GradedResult(
                score=score, correctness=1.0 if ordered else 0.0,
                reasoning_quality=score, info={"required_skills": required, "ordered": ordered},
            )
        # Generic: presence of <answer> tag.
        has_answer = "<answer>" in response.lower()
        return GradedResult(
            score=0.5 if has_answer else 0.0,
            correctness=1.0 if has_answer else 0.0,
            reasoning_quality=0.5 if has_answer else 0.0,
            info={},
        )

    def _extract_answer(self, response: str) -> str:
        import re
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else response.strip()

    # -- capability + failure updates ------------------------------------

    def _update_capability(
        self,
        task: TaskSpec,
        selection: dict[str, Any],
        graded: list[GradedResult],
        focus_skill: str,
    ) -> None:
        domain = task.domain
        for g in graded:
            self.profiler.record_rollout(
                self.config.model_id, domain, 0, g,
                fingerprint=self.config.model_fingerprint,
                profile=self.skill_profile,
            )
            # Frontier sampler observation for taskset arms.
            if task.source == "taskset":
                key = RolloutKey(
                    domain=task.metadata.get("domain", ""),
                    task_idx=int(task.metadata.get("task_idx", 0)),
                )
                FrontierSampler(self.frontier_profile).observe(
                    key, g, tokens=self._estimate_tokens(g.info.get("response", "")) if isinstance(g.info, dict) else 0,
                    flops=0.0,
                )

    def _extract_failures(self, task: TaskSpec, graded: list[GradedResult], rollouts: list[str]) -> None:
        for response, g in zip(rollouts, graded):
            if g.correctness >= 1.0:
                continue
            session = {
                "id": f"{task.task_id}-{random.randint(0, 1 << 30)}",
                "domain": task.domain,
                "response": response,
                "errors": [],
            }
            try:
                spec = extract_failure_spec(session, g)
                append_failure_spec(self.config.loop_id, spec)
                self.state.recent_failures.append(spec.public())
            except Exception:
                pass
        if len(self.state.recent_failures) > 50:
            self.state.recent_failures = self.state.recent_failures[-50:]

    def _capability_gain_for_task(self, selection: dict[str, Any]) -> float:
        from collections import deque
        if not self.state.score_history:
            return 0.0
        return capability_gain(deque(self.state.score_history))


__all__ = [
    "AdaptiveConfig",
    "OrchestratorState",
    "Episode",
    "TrainingSignal",
    "OrchestratorStore",
    "AdaptiveOrchestrator",
]
