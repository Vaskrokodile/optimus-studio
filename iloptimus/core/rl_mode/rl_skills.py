"""RL mode skills: guidance for building good RL environments.

Compact prompt injections that teach the model how to design effective
RL environments, reward functions, and curricula — without writing code.
"""

from __future__ import annotations

_ENV_DESIGN = """\
## RL environment design skill

- TEMPLATE FIRST: Check the environment catalog before writing code. There are
  60+ pre-built environments covering math, coding, reasoning, and agentic tasks.
  Using a template is always better than coding from scratch.
- REWARD SHAPING: The reward function determines what the model learns.
  - Correctness should be the dominant signal (weight 1.0).
  - Token efficiency bonus rewards conciseness (weight 0.3).
  - Anti-pattern penalty punishes backtracking and filler (weight 0.5).
  - Format bonus rewards following output format (weight 0.1).
  - Difficulty bonus rewards solving harder problems (weight 0.2).
- VERIFIER DESIGN: The verifier must be deterministic and fast.
  - Math: check the boxed answer against ground truth.
  - Code: run tests and check pass rate.
  - Reasoning: check logical validity, not just the conclusion.
  - Never use an LLM as a verifier — it introduces reward noise.
- TOKEN BUDGET: Set a token budget that forces conciseness without making
  the task impossible. Start at 2048 tokens, adjust based on difficulty.
"""

_VLLM_BATCHING = """\
## vLLM batching skill

- PARALLEL GENERATION: For each task, generate N=8-32 responses in parallel
  using vLLM's batch inference. This costs ~the same as generating 1 response.
- AGGREGATION STRATEGIES:
  - Best-of-N: pick the highest-scoring response. Good for correctness.
  - Self-consistency: majority vote on the answer. Good for math.
  - Tournament: pairwise compare responses, pick the winner. Good for quality.
  - Iterative refinement: take the best response, ask the model to improve it.
- DIVERSITY: Use temperature 0.7-0.9 for the batch to get diverse approaches.
  Use temperature 0.3 for the final answer to ensure correctness.
- BATCH SIZE: 16 is a good default. 8 for short tasks, 32 for complex ones.
"""

_RL_BATCHING = """\
## RL batching skill (novel param-split architecture)

- PARAM SPLITTING: The model's parameters are divided into N parts (e.g., 20).
  Each part is trained on a different RL environment via targeted LoRA.
  Only 5% of params are trained per pipeline, so they don't interfere.
- BATCHED GENERATION: All N pipelines' generation requests are batched into
  a SINGLE vLLM call. N pipelines train at ~the same speed as 1 pipeline.
- ATTRIBUTION: Use attribution analysis to determine which layers handle
  which capabilities. Assign each RL environment to the layers most relevant
  to its capability. This maximizes training efficiency.
- ADAPTER SYNC: After each training round, reload all LoRA adapters into vLLM.
  This ensures each pipeline uses the latest version of its adapter.
- PIPELINE COUNT: 20 pipelines is a good default. Each targets a different
  capability or difficulty level. More pipelines = more diversity but more
  memory for adapters.
"""

_CURRICULUM = """\
## Curriculum design skill

- DIFFICULTY LADDER: Start with easy problems (difficulty 0.2), progress to
  medium (0.5), then hard (0.8). Use the curriculum selector to track mastery.
- MASTERY THRESHOLD: Move to the next difficulty level when accuracy exceeds
  80% on the current level. Don't skip levels — that causes training instability.
- FAILURE-FIRST: When the model fails at a specific problem type, generate
  more problems of that type. Failure-driven training is more efficient than
  uniform training.
- REVERSE CURRICULUM: For hard problems, start from the solution and work
  backwards. This helps the model learn the reasoning structure.
- MULTI-OBJECTIVE: Train on multiple objectives simultaneously using RL
  batching. Each pipeline targets a different objective.
"""

_RL_SKILL_BLOCKS = [
    ("env-design", _ENV_DESIGN),
    ("vllm-batching", _VLLM_BATCHING),
    ("rl-batching", _RL_BATCHING),
    ("curriculum", _CURRICULUM),
]


def rl_skills_prompt(max_chars: int = 3_500) -> str:
    """Build the combined RL skills prompt block."""
    parts: list[str] = []
    remaining = max_chars
    for _name, block in _RL_SKILL_BLOCKS:
        if len(block) > remaining:
            parts.append(block[:remaining])
            break
        parts.append(block)
        remaining -= len(block)
    return "\n".join(parts).strip()


def list_rl_skills() -> list[dict[str, str]]:
    return [{"id": name, "chars": len(block)} for name, block in _RL_SKILL_BLOCKS]
