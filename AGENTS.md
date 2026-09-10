# Optimus Studio — agent notes

## Environment
- Python 3.11+ on Windows. Set `$env:PYTHONPATH = "."` before running tests/imports.
- Tests use a temp `ILOPTIMUS_HOME` (set via `monkeypatch.setenv` or `os.environ.setdefault`).
- `scipy` is NOT a dependency; capability metrics use a pure-Python `digamma`.

## Test commands
- New adaptive curriculum tests: `python -m pytest tests/test_adaptive_curriculum.py tests/test_adaptive_orchestrator.py -q`
- Full suite (slow, ~8 min): `python -m pytest tests/ -q --ignore=tests/test_backends.py --ignore=tests/test_inference_generation.py --ignore=tests/test_training_performance.py`
- Fast regression check: `python -m pytest tests/test_rsi_panels.py tests/test_harness_graph.py tests/test_failure_memory.py tests/test_rl_mode.py -q`

## Adaptive RL curriculum architecture
The adaptive layer composes the existing 83 `rl_factory/environments/` Gymnasium
envs (env_mutation, adversarial_counterfactual, trajectory_recombination,
skill_collision, failure_first, ...) with new foundation + orchestration modules:

- `core/capability_metrics.py` — pure-Python EIG / Intelligence Density / digamma.
- `core/skill_graph.py` — typed skill DAG + Beta-posterior compositions.
- `core/capability_profiler.py` — per-model persistent profiles, fingerprint-drift transfer.
- `core/frontier_sampler.py` — Thompson+UCB+EIG sampler over (domain, task) arms, 5-30% band.
- `core/persistent_world.py` (#16) — shared mutable company world, trigger-based task derivation.
- `core/env_generator_agent.py` (#26) — adversarial LLM generator via `run_json_completion`; `synthesize_new` MAY emit generated `verify.py` (user-approved).
- `core/session_to_env.py` (#24) — FailureSpec extraction from session trajectories.
- `core/adaptive_orchestrator.py` — the main loop tying it all together.
- `core/rsi_loops.py` — added `next_task` / `plan_next_composition` / `run_frontier_iteration` + focus fields.
- `server.py` — added `/api/rsi/loops/{id}/adaptive/*` and `/api/worlds/*` endpoints.

## Design decisions (recorded)
1. Generator inference entry point: `run_json_completion` (constrains output to the JSON action schema).
2. Generated verifier code: ALLOWED — `synthesize_new` may emit `verify.py` run in a subprocess sandbox (`run_generated_verifier`).
