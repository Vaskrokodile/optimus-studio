"""Model Profiler / Skill Graph — a typed skill DAG with Beta-posterior mastery.

Each skill node keeps a Beta(alpha, beta) posterior over mastery that is updated
from graded rollouts. Compositions (sets of skills exercised together) keep
their own posterior so the system can learn interaction effects beyond the
conjunctive-independence product ``prod(mu(s))``.

The graph is a static ontology (``SKILL_DAG``) augmented at runtime with
co-occurrence composition edges recorded from real rollouts. The curriculum
query ``find_unmastered_compositions`` returns the highest-priority
frontier-adjacent compositions for the adaptive orchestrator.

See ``RESEARCH_skill_graph.md`` for the full design.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .grader import GradedResult

# ---------------------------------------------------------------------------
# Skill ontology
# ---------------------------------------------------------------------------

# DAG as adjacency list. A child may appear under multiple parents (valid DAG).
SKILL_DAG: dict[str, dict[str, list[str]]] = {
    "agentic-intelligence": {
        "children": ["coding", "math", "reasoning", "tool-calling", "agentic"]
    },
    "coding": {
        "children": [
            "code-implementation", "humaneval", "debugging", "refactoring",
            "edge-case-handling", "test-interpretation",
            "performance-profiling", "architecture-selection",
        ]
    },
    "math": {
        "children": [
            "gsm8k", "aime", "arithmetic", "algebra", "combinatorics",
            "geometry", "word-problems", "competition-math",
        ]
    },
    "reasoning": {
        "children": [
            "reasoning-tasks", "logical-deduction", "constraint-reasoning",
            "probabilistic-reasoning", "multi-step-deduction",
            "cross-module-reasoning",
        ]
    },
    "tool-calling": {
        "children": [
            "tool-selection", "argument-construction", "tool-ordering",
            "efficiency", "error-recovery", "distractor-resistance",
            "tool-fs", "tool-sql", "tool-web", "tool-booking",
            "tool-pipeline", "tool-recovery", "tool-distractor",
            "tool-parallel", "tool-interpreter", "tool-devops", "tool-api",
        ]
    },
    # Tool domain leaf nodes + their narrow skill.
    "tool-fs": {"children": ["filesystem-navigation"]},
    "tool-sql": {"children": ["sql-querying"]},
    "tool-web": {"children": ["web-research"]},
    "tool-booking": {"children": ["booking-flows"]},
    "tool-pipeline": {"children": ["pipeline-sequencing"]},
    "tool-recovery": {"children": ["fault-recovery"]},
    "tool-distractor": {"children": ["distractor-resistance"]},
    "tool-parallel": {"children": ["parallel-batching"]},
    "tool-interpreter": {"children": ["code-interpretation"]},
    "tool-devops": {"children": ["devops-triage"]},
    "tool-api": {"children": ["api-reliability"]},
    # Agentic branch.
    "agentic": {
        "children": ["agentic-reasoning", "agentic-coding"]
    },
    "agentic-reasoning": {
        "children": [
            "sustained-planning", "multi-step-deduction", "cross-module-reasoning"
        ]
    },
    "agentic-coding": {
        "children": [
            "codebase-navigation", "dependency-resolution", "api-discovery",
            "rollback", "failure-diagnosis", "refactoring", "debugging",
            "test-interpretation", "performance-profiling", "architecture-selection",
        ]
    },
}

DOMAIN_TO_SKILLS: dict[str, list[str]] = {
    "coding": [
        "code-implementation", "debugging", "refactoring",
        "edge-case-handling", "test-interpretation", "performance-profiling",
    ],
    "reasoning": [
        "reasoning-tasks", "logical-deduction", "constraint-reasoning",
        "probabilistic-reasoning", "combinatorics",
    ],
    "agentic-reasoning": [
        "agentic-reasoning", "sustained-planning",
        "multi-step-deduction", "cross-module-reasoning",
    ],
    "agentic-coding": [
        "agentic-coding", "codebase-navigation", "debugging", "refactoring",
        "dependency-resolution", "api-discovery", "failure-diagnosis",
        "test-interpretation", "performance-profiling", "architecture-selection",
    ],
    "humaneval": [
        "humaneval", "code-implementation", "edge-case-handling",
        "test-interpretation",
    ],
    "gsm8k": ["gsm8k", "word-problems", "arithmetic", "algebra"],
    "aime": ["aime", "competition-math", "combinatorics", "algebra", "geometry"],
    # Tool domains: domain-specific leaf + cross-cutting tool sub-skills.
    "tool-fs": [
        "tool-fs", "filesystem-navigation",
        "tool-selection", "argument-construction", "efficiency", "distractor-resistance",
    ],
    "tool-sql": [
        "tool-sql", "sql-querying",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-web": [
        "tool-web", "web-research",
        "tool-selection", "argument-construction", "efficiency", "distractor-resistance",
    ],
    "tool-booking": [
        "tool-booking", "booking-flows",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-pipeline": [
        "tool-pipeline", "pipeline-sequencing",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-recovery": [
        "tool-recovery", "fault-recovery", "error-recovery",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-distractor": [
        "tool-distractor", "distractor-resistance",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-parallel": [
        "tool-parallel", "parallel-batching",
        "tool-selection", "argument-construction", "tool-ordering", "efficiency",
    ],
    "tool-interpreter": [
        "tool-interpreter", "code-interpretation",
        "tool-selection", "argument-construction", "efficiency",
    ],
    "tool-devops": [
        "tool-devops", "devops-triage",
        "error-recovery", "tool-selection", "argument-construction",
        "tool-ordering", "efficiency", "rollback", "failure-diagnosis",
    ],
    "tool-api": [
        "tool-api", "api-reliability",
        "tool-selection", "argument-construction", "efficiency",
    ],
}

# High-value compositions to seed the curriculum search.
EXAMPLE_COMPOSITIONS: list[set[str]] = [
    {"debugging", "codebase-navigation"},
    {"debugging", "api-discovery"},
    {"refactoring", "performance-profiling"},
    {"tool-ordering", "error-recovery"},
    {"codebase-navigation", "api-discovery", "dependency-resolution", "debugging"},
]


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SkillNodeState:
    node_id: str
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    successes: float = 0.0
    last_seen: float | None = None

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class CompositionState:
    node_ids: frozenset[str]
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    successes: float = 0.0

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def key(self) -> str:
        return "+".join(sorted(self.node_ids))


@dataclass
class SkillGraphProfile:
    model_id: str
    model_fingerprint: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    nodes: dict[str, SkillNodeState] = field(default_factory=dict)
    compositions: dict[str, CompositionState] = field(default_factory=dict)
    rollout_count: int = 0


@dataclass(frozen=True)
class SkillComposition:
    node_ids: frozenset[str]
    target_mastery: float = 0.85
    priority_score: float = 0.0
    distance_to_frontier: int = 0


# ---------------------------------------------------------------------------
# Skill graph (DAG topology helpers)
# ---------------------------------------------------------------------------


class SkillGraph:
    """Topology over the static ``SKILL_DAG`` plus runtime co-occurrence edges."""

    def __init__(self, dag: dict[str, dict[str, list[str]]] | None = None) -> None:
        self.dag = dag if dag is not None else SKILL_DAG
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}
        for parent, spec in self.dag.items():
            self._children[parent] = list(spec.get("children", []))
            for child in spec.get("children", []):
                self._parents.setdefault(child, []).append(parent)
        # Co-occurrence edges recorded from rollouts (undirected adjacency).
        self._cooccur: dict[str, set[str]] = {}

    def children(self, node: str) -> list[str]:
        return self._children.get(node, [])

    def parents(self, node: str) -> list[str]:
        return self._parents.get(node, [])

    def all_nodes(self) -> list[str]:
        seen: set[str] = set(self.dag.keys())
        for spec in self.dag.values():
            seen.update(spec.get("children", []))
        return sorted(seen)

    def record_cooccurrence(self, node_ids: list[str]) -> None:
        """Record that ``node_ids`` were exercised together in one rollout."""
        uniq = [n for n in dict.fromkeys(node_ids) if n]
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                if a == b:
                    continue
                self._cooccur.setdefault(a, set()).add(b)
                self._cooccur.setdefault(b, set()).add(a)

    def adjacent(self, a: str, b: str) -> bool:
        """Two nodes are adjacent if parent/child, sibling, or co-occurring."""
        if a == b:
            return True
        if b in self._children.get(a, []) or a in self._children.get(b, []):
            return True
        if b in self._parents.get(a, []) or a in self._parents.get(b, []):
            return True
        # Siblings: share a parent.
        if set(self._parents.get(a, [])) & set(self._parents.get(b, [])):
            return True
        if b in self._cooccur.get(a, set()):
            return True
        return False

    def ancestors(self, node: str) -> list[str]:
        out: list[str] = []
        stack = list(self._parents.get(node, []))
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(self._parents.get(cur, []))
        return out

    def min_distance_to_set(self, node: str, targets: set[str]) -> int:
        """BFS distance from ``node`` to the nearest target (parent/child edges)."""
        if node in targets:
            return 0
        # BFS over the undirected parent/child graph.
        frontier = {node}
        seen = {node}
        dist = 0
        # Cap the search to avoid pathological cost on a small graph.
        while frontier and dist < 64:
            dist += 1
            nxt: set[str] = set()
            for cur in frontier:
                for nb in self._children.get(cur, []) + self._parents.get(cur, []):
                    if nb in seen:
                        continue
                    if nb in targets:
                        return dist
                    seen.add(nb)
                    nxt.add(nb)
            frontier = nxt
        return dist


# ---------------------------------------------------------------------------
# Rollout -> skill evidence
# ---------------------------------------------------------------------------


def domain_to_active_skills(
    domain: str,
    graded: "GradedResult" | Any,
) -> list[tuple[str, float]]:
    """Return ``(skill_node_id, weight)`` pairs for a graded rollout.

    The primary weight is the graded score in ``[0, 1]``. For tool-calling
    domains the cross-cutting sub-skills (efficiency, distractor-resistance,
    error-recovery) are re-weighted using the breakdown in ``graded.info`` so
    that, e.g., a rollout that hit distractors down-weights ``distractor-resistance``.
    """
    base_skills = DOMAIN_TO_SKILLS.get(domain, [])
    if not base_skills:
        return []

    score = float(getattr(graded, "score", 0.0) or 0.0)
    correctness = float(getattr(graded, "correctness", 0.0) or 0.0)
    info = getattr(graded, "info", {}) or {}

    # Default: every active skill receives the rollout score as evidence.
    weights: dict[str, float] = {s: score for s in base_skills}

    # Tool-calling fine-grained re-weighting from the breakdown dict.
    if domain.startswith("tool-"):
        efficiency = float(info.get("efficiency", 1.0) or 0.0)
        distractor_hits = float(info.get("distractor_hits", 0.0) or 0.0)
        errors = float(len(info.get("errors", []) or []))
        tool_selection = float(info.get("tool_selection", correctness) or 0.0)

        if "efficiency" in base_skills:
            weights["efficiency"] = min(weights["efficiency"], efficiency)
        if "distractor-resistance" in base_skills:
            # Hitting distractors is evidence of weak distractor-resistance.
            weights["distractor-resistance"] = max(0.0, correctness - 0.25 * distractor_hits)
        if "error-recovery" in base_skills:
            weights["error-recovery"] = max(0.0, 1.0 - 0.2 * errors)
        if "tool-selection" in base_skills:
            weights["tool-selection"] = tool_selection
        if "argument-construction" in base_skills:
            weights["argument-construction"] = min(weights["argument-construction"], tool_selection)

    return [(s, max(0.0, min(1.0, w))) for s, w in weights.items()]


# ---------------------------------------------------------------------------
# Curriculum query
# ---------------------------------------------------------------------------


def find_unmastered_compositions(
    graph: SkillGraph,
    profile: SkillGraphProfile,
    k: int = 5,
    threshold: float = 0.85,
    max_size: int = 4,
    frontier_distance: int = 2,
) -> list[SkillComposition]:
    """Return the top-``k`` frontier-adjacent unmastered compositions.

    Search algorithm (see RESEARCH_skill_graph.md section 5):
      1. mastered = nodes with mastery >= threshold
      2. frontier = unmastered nodes within ``frontier_distance`` edges of mastered
      3. candidate compositions of size 1..max_size containing >=1 frontier node,
         all members pairwise adjacent in the DAG
      4. estimate mastery (recorded composition posterior, else product)
      5. priority = gap - 0.2*dist - size_penalty + 0.1*novelty
    """
    mastered = {
        n for n, st in profile.nodes.items()
        if st.mastery >= threshold
    }
    # If nothing is mastered yet, treat the root's children as the frontier seed
    # so the very first compositions are still well-defined.
    if not mastered:
        mastered = set(graph.children("agentic-intelligence")) & set(profile.nodes.keys())

    frontier = set()
    for node in profile.nodes:
        if node in mastered:
            continue
        if graph.min_distance_to_set(node, mastered) <= frontier_distance:
            frontier.add(node)
    if not frontier:
        frontier = set(profile.nodes.keys()) - mastered

    candidates: list[SkillComposition] = []
    frontier_list = sorted(frontier)

    # Size-1 compositions (single frontier skills) first.
    for node in frontier_list:
        candidates.append(_score_composition(graph, profile, frozenset({node}), threshold, mastered))

    # Sizes 2..max_size: combinations of frontier + adjacent nodes.
    all_nodes = sorted(set(profile.nodes.keys()) | frontier)
    for size in range(2, max_size + 1):
        for combo in _combinations_adjacent(graph, frontier_list, all_nodes, size):
            candidates.append(_score_composition(graph, profile, combo, threshold, mastered))

    # Deduplicate by node set, keep best priority.
    best: dict[frozenset[str], SkillComposition] = {}
    for c in candidates:
        cur = best.get(c.node_ids)
        if cur is None or c.priority_score > cur.priority_score:
            best[c.node_ids] = c

    ranked = sorted(best.values(), key=lambda c: c.priority_score, reverse=True)
    return ranked[:k]


def _score_composition(
    graph: SkillGraph,
    profile: SkillGraphProfile,
    node_ids: frozenset[str],
    threshold: float,
    mastered: set[str],
) -> SkillComposition:
    est = _composition_mastery(graph, profile, node_ids)
    gap = max(0.0, threshold - est)
    dist = min(graph.min_distance_to_set(n, mastered) for n in node_ids) if mastered else 0
    attempts = _composition_attempts(profile, node_ids)
    novelty = 1.0 / (1.0 + attempts)
    # Size penalty dominates the gap inflation that comes from composing with
    # unobserved nodes (uniform prior 0.5), so single-node frontier skills rank
    # before larger compositions of the same frontier, per the design doc.
    size_penalty = 0.15 * (len(node_ids) - 1)
    priority = gap - 0.2 * dist - size_penalty + 0.1 * novelty
    return SkillComposition(
        node_ids=node_ids,
        target_mastery=threshold,
        priority_score=priority,
        distance_to_frontier=dist,
    )


def _composition_mastery(
    graph: SkillGraph,
    profile: SkillGraphProfile,
    node_ids: frozenset[str],
) -> float:
    key = "+".join(sorted(node_ids))
    comp = profile.compositions.get(key)
    if comp is not None and comp.attempts > 0:
        return comp.mastery
    # Conjunctive independence: product of per-node mastery (weakest-link).
    product = 1.0
    for n in node_ids:
        st = profile.nodes.get(n)
        product *= st.mastery if st else 0.5  # unobserved -> uniform prior mean
    return product


def _composition_attempts(profile: SkillGraphProfile, node_ids: frozenset[str]) -> int:
    key = "+".join(sorted(node_ids))
    comp = profile.compositions.get(key)
    return comp.attempts if comp else 0


def _combinations_adjacent(
    graph: SkillGraph,
    frontier: list[str],
    pool: list[str],
    size: int,
) -> list[frozenset[str]]:
    """Yield ``size``-subsets of ``pool`` that contain >=1 frontier node and are
    pairwise adjacent in the DAG. Uses a simple recursive extension."""
    out: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()

    def extend(current: tuple[str, ...], candidates: list[str]) -> None:
        if len(current) == size:
            fs = frozenset(current)
            if fs not in seen:
                seen.add(fs)
                out.append(fs)
            return
        for i, c in enumerate(candidates):
            # All members must be adjacent to every existing member.
            if all(graph.adjacent(c, m) for m in current):
                extend(current + (c,), candidates[i + 1:])

    # Seed each combination with a frontier node.
    for f in frontier:
        rest = [n for n in pool if n != f]
        extend((f,), rest)
    return out


__all__ = [
    "SKILL_DAG",
    "DOMAIN_TO_SKILLS",
    "EXAMPLE_COMPOSITIONS",
    "SkillNodeState",
    "CompositionState",
    "SkillGraphProfile",
    "SkillComposition",
    "SkillGraph",
    "domain_to_active_skills",
    "find_unmastered_compositions",
]
