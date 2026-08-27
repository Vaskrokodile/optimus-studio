"""
ContextDistiller: Compress a conversation to a minimal summary.

Environment concept:
  The model is given a long conversation or context (multiple turns of
  agent-human interaction, or a long reasoning trace) and must produce a
  minimal summary that preserves ALL information needed to continue the
  task. The summary must be:
    1. Complete: no critical information lost
    2. Concise: significantly shorter than the original
    3. Actionable: the next agent can pick up from the summary

  This trains the model to:
    1. Identify what information is essential vs disposable
    2. Compress verbose exchanges into key facts
    3. Preserve decisions, state, and next steps
    4. Drop filler, repetition, and intermediate dead ends

  This directly attacks the frontier-model pattern of growing context
  windows with redundant information. A 100T model that can distill a
  50K-token conversation into 5K tokens without losing critical info
  saves 90% of context cost on every subsequent turn.

Why this environment is worth using for 10T-100T param models:
  Context distillation is the key to long-running agents. Without it,
  context windows fill up with noise and costs explode linearly. With
  it, agents can run indefinitely at fixed per-turn cost. This is the
  difference between a $100 and $1000 agentic coding session.

Problem types:
  - Coding conversations (debugging sessions, feature requests)
  - Multi-turn reasoning (math, logic, analysis)
  - Agent state (tool calls, results, decisions)
  - Mixed conversations (code + reasoning + decisions)

Verification:
  Each problem has:
    - The original conversation/context
    - A set of "key facts" that MUST appear in the summary
    - The original token count
  The verifier:
    1. Checks that all key facts are present in the summary
    2. Computes compression ratio = original_tokens / summary_tokens
    3. Scores = fact_coverage * compression_factor

Reward design:
  - Fact coverage: fraction of key facts found in the summary (gate: must be >= 0.8)
  - Compression ratio: original / summary (higher = better)
  - If coverage < 0.8: reward = 0 (information loss is unacceptable)
  - If coverage >= 0.8: reward = coverage * min(1.0, compression_ratio / 3.0)
    (3x compression = full reward; beyond 3x = no extra bonus)
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import BaseReasoningEnv, Problem
from iloptimus.core.rl_factory.core.token_budget import estimate_tokens
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------


_CONTEXT_PROBLEMS = [
    {
        "conversation": """User: I need to fix a bug in the user authentication module.

Assistant: I'll help you fix the authentication bug. Let me first look at the auth module. I can see the issue - the `check_password` function is comparing passwords in plaintext instead of using hashed values.

User: Yes, that's a security issue. Can you fix it?

Assistant: I've fixed the `check_password` function to use bcrypt for password comparison. The function now hashes the input password and compares it to the stored hash. I also updated the `hash_password` function to use bcrypt with a salt.

User: Great. Now I need to add rate limiting to the login endpoint.

Assistant: I've added rate limiting using a simple in-memory counter. Each IP address is limited to 5 login attempts per minute. If the limit is exceeded, a 429 Too Many Requests response is returned.

User: Perfect. What's the current state of the code?

Assistant: The auth module now has: 1) bcrypt-based password hashing and comparison, 2) rate limiting on the login endpoint (5 attempts/minute per IP), 3) existing session management unchanged. The next step is to write unit tests for the new functionality.""",
        "key_facts": [
            "bcrypt",
            "password",
            "hash",
            "rate limiting",
            "5 attempts",
            "login",
            "unit tests",
            "auth",
        ],
        "difficulty": 0.3,
    },
    {
        "conversation": """User: I'm getting a TypeError when running my data processing script.

Assistant: Let me look at the error. The TypeError occurs because you're trying to concatenate a string with an integer on line 42. The `process_record` function receives a record where the 'age' field is an integer, but the code tries to add it to a string.

User: How should I fix it?

Assistant: You should convert the integer to a string using `str(record['age'])` before concatenation. I've applied this fix to line 42.

User: Now I'm getting a KeyError for the 'email' field.

Assistant: The KeyError occurs because some records don't have an 'email' field. You should use `record.get('email', '')` instead of `record['email']` to handle missing fields gracefully. I've applied this fix as well.

User: What's the current state?

Assistant: Two bugs have been fixed: 1) TypeError on line 42 - fixed by converting int to str, 2) KeyError for 'email' - fixed by using .get() with default. The script now processes all records without errors. Next step: add logging to track processing statistics.""",
        "key_facts": [
            "TypeError",
            "string",
            "integer",
            "line 42",
            "KeyError",
            "email",
            "get",
            "logging",
        ],
        "difficulty": 0.3,
    },
    {
        "conversation": """User: I need to implement a REST API for managing blog posts.

Assistant: I'll help you build a REST API for blog posts. I've created the following endpoints: GET /posts (list all), GET /posts/:id (get one), POST /posts (create), PUT /posts/:id (update), DELETE /posts/:id (delete).

User: I need to add authentication.

Assistant: I've added JWT-based authentication. Users must include a valid JWT token in the Authorization header. The token is verified by middleware before reaching the route handlers.

User: Now add pagination to the list endpoint.

Assistant: I've added pagination to GET /posts. It accepts 'page' and 'limit' query parameters (default: page=1, limit=10). The response includes the posts, total count, and has_next flag.

User: What's left to do?

Assistant: The API has: 1) CRUD endpoints for posts, 2) JWT authentication, 3) Pagination on the list endpoint. Remaining: add input validation, write API documentation, and set up error handling middleware.""",
        "key_facts": [
            "REST",
            "CRUD",
            "JWT",
            "authentication",
            "pagination",
            "page",
            "limit",
            "validation",
            "documentation",
        ],
        "difficulty": 0.35,
    },
    {
        "conversation": """User: My Docker container keeps crashing on startup.

Assistant: Let me check the container logs. The crash is caused by a missing environment variable `DATABASE_URL`. The application tries to connect to the database on startup but fails because the connection string is not set.

User: I've set the environment variable but it's still crashing.

Assistant: The issue is that the environment variable is set in the host but not passed to the container. You need to add it to the docker-compose.yml file under the 'environment' section, or use the -e flag when running docker run.

User: That worked. Now the container starts but the database migration fails.

Assistant: The migration fails because the database schema already exists from a previous deployment. You need to either drop the existing tables or use a migration tool that handles existing schemas. I recommend using Alembic for schema migrations.

User: What's the current state?

Assistant: Issues resolved: 1) Missing DATABASE_URL - fixed in docker-compose.yml, 2) Migration failure - recommended Alembic for schema management. Current state: container starts successfully, database connection works. Next: implement Alembic migrations and add health checks.""",
        "key_facts": [
            "Docker",
            "DATABASE_URL",
            "environment",
            "docker-compose",
            "migration",
            "Alembic",
            "schema",
            "health check",
        ],
        "difficulty": 0.35,
    },
    {
        "conversation": """User: I need to optimize my Python script that processes a large CSV file.

Assistant: I've analyzed the script. The main bottleneck is the row-by-row processing loop. I've refactored it to use pandas vectorized operations, which should be 50x faster.

User: The memory usage is too high with pandas.

Assistant: The memory issue is because pandas loads the entire file into memory. I've switched to chunked processing using `pd.read_csv(chunksize=10000)`, which processes 10,000 rows at a time.

User: Now I need to add error handling for malformed rows.

Assistant: I've added a try-except block around each chunk processing. Malformed rows are logged and skipped. The script also now reports the total number of skipped rows at the end.

User: Summary?

Assistant: Optimizations applied: 1) Vectorized operations with pandas (50x speedup), 2) Chunked processing (10K rows/chunk) for memory efficiency, 3) Error handling for malformed rows with logging. The script now handles large files efficiently. No remaining issues.""",
        "key_facts": [
            "pandas",
            "vectorized",
            "chunked",
            "chunksize",
            "10000",
            "memory",
            "error handling",
            "malformed",
            "logging",
        ],
        "difficulty": 0.35,
    },
]


def context_distiller_generator(seed: int) -> Problem:
    """Generate a ContextDistiller problem."""
    rng = random.Random(seed)
    template = rng.choice(_CONTEXT_PROBLEMS)

    original_tokens = estimate_tokens(template["conversation"])

    return Problem(
        id=f"context_distill_{rng.randint(0, 99999)}",
        prompt=_format_prompt(template["conversation"], original_tokens),
        difficulty=template["difficulty"],
        metadata={
            "type": "context_distiller",
            "conversation": template["conversation"],
            "key_facts": template["key_facts"],
            "original_tokens": original_tokens,
        },
        token_budget=800,
        source="generated",
    )


def _format_prompt(conversation: str, original_tokens: int) -> str:
    return (
        f"Compress the following conversation into a minimal summary that "
        f"preserves ALL information needed to continue the task.\n\n"
        f"--- CONVERSATION ---\n{conversation}\n--- END ---\n\n"
        f"Original: ~{original_tokens} tokens\n"
        f"Rules:\n"
        f"  - Preserve ALL key facts, decisions, and next steps\n"
        f"  - Remove filler, repetition, and intermediate dead ends\n"
        f"  - Target: 3x+ compression (much shorter than original)\n"
        f"  - Output ONLY the summary — no meta-commentary\n"
        f"  - If information loss occurs, your reward is ZERO"
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ContextDistillerVerifier(Verifier):
    """
    Verifies a ContextDistiller response by:
      1. Checking that all key facts appear in the summary
      2. Computing compression ratio
      3. Scoring = coverage * compression_factor
    """

    def __init__(self, key_facts: list[str], original_tokens: int):
        super().__init__()
        self._key_facts = key_facts
        self._original_tokens = original_tokens

    def verify(self, response: str) -> VerifierResult:
        summary = response.strip()
        summary_lower = summary.lower()
        summary_tokens = estimate_tokens(summary)

        # Check key fact coverage
        facts_found = 0
        facts_missing = []

        for fact in self._key_facts:
            if fact.lower() in summary_lower:
                facts_found += 1
            else:
                facts_missing.append(fact)

        coverage = facts_found / len(self._key_facts) if self._key_facts else 1.0

        # Hard gate: if coverage < 0.8, reward = 0
        if coverage < 0.8:
            return VerifierResult(
                correct=False,
                score=0.0,
                partial_credit={"coverage": coverage, "facts_found": facts_found, "facts_total": len(self._key_facts)},
                diagnostics=f"Coverage too low: {coverage:.0%}. Missing: {', '.join(facts_missing[:3])}",
            )

        # Compute compression ratio
        if summary_tokens > 0:
            compression_ratio = self._original_tokens / summary_tokens
        else:
            compression_ratio = 0.0

        # Compression factor: 3x = full reward, capped
        compression_factor = min(1.0, compression_ratio / 3.0)

        # Final score
        score = coverage * compression_factor
        correct = score >= 0.6

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "coverage": coverage,
                "compression_ratio": compression_ratio,
                "compression_factor": compression_factor,
                "original_tokens": self._original_tokens,
                "summary_tokens": summary_tokens,
                "facts_found": facts_found,
                "facts_total": len(self._key_facts),
            },
            diagnostics=f"Coverage={coverage:.0%} Compression={compression_ratio:.1f}x ({self._original_tokens}→{summary_tokens} tokens). Score={score:.2f}",
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ContextDistillerEnv(BaseReasoningEnv):
    """
    ContextDistiller environment: compress conversation to minimal summary.

    The model receives a conversation and must produce a minimal summary
    that preserves all key facts. Reward = coverage * compression_factor.
    """

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
    ):
        if problems is None and problem_generator is None:
            problem_generator = context_distiller_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ContextDistillerVerifier(
            key_facts=problem.metadata["key_facts"],
            original_tokens=problem.metadata["original_tokens"],
        )

    def _check_format(self, response: str) -> float:
        # Any non-empty response is acceptable format
        if len(response.strip()) > 20:
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> str:
        return response
