"""
ShellQuest: Unix shell CTF skills in a sandboxed environment.

Environment concept:
  The model receives a synthetic shell challenge (e.g. "find the file with
  the SUID bit", "decode this base64 string", "find the hidden flag in this
  fake directory listing"). No real shell is ever executed — all challenges
  are simulated with pure-Python data so everything is sandboxed and safe.

  Challenge types:
    - base64: decode a base64-encoded flag
    - directory_listing: spot the flag hidden in a fake `ls -la` listing
    - suid_find: identify the SUID file from a fake `find` output
    - grep_flag: find the flag in a fake file dump
    - permissions: identify the octal permission string for a described mode
    - env_var: recover a flag stored in a fake environment variable dump

  Verification is rule-based: the model must produce the exact flag/answer.

  reward = answer_correct * 0.8 + challenge_type_named * 0.2

  This is a batch environment: N parallel attempts are scored, and the best
  attempt determines the reward (best-of-N).
"""

from __future__ import annotations

import base64
import random
import re
import string
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Flag generation
# ---------------------------------------------------------------------------


_FLAG_WORDS = [
    "shell", "root", "suid", "hidden", "binary", "exploit", "privesc",
    "kernel", "payload", "overflow", "stack", "heap", "gadget", "rop",
    "syscall", "execve", "chmod", "chown", "sticky", "permission",
]


def _make_flag(rng: random.Random) -> str:
    word = rng.choice(_FLAG_WORDS)
    num = rng.randint(100, 9999)
    return f"flag{{{word}_{num}}}"


# ---------------------------------------------------------------------------
# Challenge generators
# ---------------------------------------------------------------------------


def _gen_base64(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    encoded = base64.b64encode(flag.encode()).decode()
    challenge_text = (
        f"A flag has been base64-encoded. Decode the following string to "
        f"recover it:\n\n{encoded}\n"
    )
    return {
        "type": "base64",
        "challenge_text": challenge_text,
        "flag": flag,
        "hint": "Use `echo <string> | base64 -d` to decode.",
    }


def _gen_directory_listing(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    # Build a fake ls -la listing with the flag embedded in a filename.
    files = [
        "-rw-r--r--  1 root root  4096 Jul  4 10:22 README.md",
        "-rw-r--r--  1 root root  2048 Jul  4 10:22 config.yaml",
        "drwxr-xr-x  2 root root  4096 Jul  4 10:23 src",
        f"-rw-------  1 root root   128 Jul  4 10:24 {flag}.txt",
        "-rw-r--r--  1 root root  1024 Jul  4 10:25 notes.txt",
    ]
    listing = "\n".join(files)
    challenge_text = (
        f"You are given the output of `ls -la` in a target directory. "
        f"Find the hidden flag embedded in the listing.\n\n{listing}\n"
    )
    return {
        "type": "directory_listing",
        "challenge_text": challenge_text,
        "flag": flag,
        "hint": "Look for an unusual filename that matches the flag format.",
    }


def _gen_suid_find(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    # Build a fake `find / -perm -4000` output. The flag is in a comment
    # attached to one of the SUID binaries.
    suid_files = [
        "/usr/bin/sudo",
        "/usr/bin/passwd",
        f"/usr/local/bin/findflag  # {flag}",
        "/usr/bin/chsh",
        "/usr/bin/mount",
    ]
    listing = "\n".join(suid_files)
    challenge_text = (
        f"Review the output of `find / -perm -4000 2>/dev/null`. One of the "
        f"SUID binaries has a flag hidden in its comment.\n\n{listing}\n"
    )
    return {
        "type": "suid_find",
        "challenge_text": challenge_text,
        "flag": flag,
        "hint": "SUID binaries run with the owner's privileges. Inspect each line.",
    }


def _gen_grep_flag(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    # Build a fake file dump with the flag hidden among lines.
    lines = [
        "DEBUG: starting service on port 8080",
        "INFO: connection accepted from 10.0.0.5",
        f"AUTH: token={flag}",
        "INFO: request completed in 12ms",
        "DEBUG: closing socket",
    ]
    dump = "\n".join(lines)
    challenge_text = (
        f"Below is a log dump. Use `grep` to find the line containing the "
        f"flag.\n\n{dump}\n"
    )
    return {
        "type": "grep_flag",
        "challenge_text": challenge_text,
        "flag": flag,
        "hint": "The flag is on a line tagged AUTH:.",
    }


def _gen_permissions(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    # Describe a permission mode and ask for the octal representation.
    modes = [
        ("read/write/execute for owner, read for group, none for others", "740"),
        ("read/execute for owner, read/execute for group, read for others", "755"),
        ("read/write for owner, read for group, read for others", "644"),
        ("read/write/execute for owner, read/write for group, read for others", "764"),
    ]
    desc, octal = rng.choice(modes)
    challenge_text = (
        f"A file has the following permissions: {desc}.\n"
        f"What is the 3-digit octal permission code? "
        f"Submit the code wrapped in the flag format: flag{{<code>}}.\n"
    )
    answer_flag = f"flag{{{octal}}}"
    return {
        "type": "permissions",
        "challenge_text": challenge_text,
        "flag": answer_flag,
        "hint": "r=4, w=2, x=1. Sum per triplet.",
    }


def _gen_env_var(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    # Build a fake `env` dump with the flag in a variable.
    env_lines = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        "HOME=/root",
        f"SECRET_FLAG={flag}",
        "SHELL=/bin/bash",
        "USER=root",
    ]
    dump = "\n".join(env_lines)
    challenge_text = (
        f"Below is the output of `env`. Find the flag stored in an "
        f"environment variable.\n\n{dump}\n"
    )
    return {
        "type": "env_var",
        "challenge_text": challenge_text,
        "flag": flag,
        "hint": "Look for a variable whose value matches the flag format.",
    }


_GENERATORS = {
    "base64": _gen_base64,
    "directory_listing": _gen_directory_listing,
    "suid_find": _gen_suid_find,
    "grep_flag": _gen_grep_flag,
    "permissions": _gen_permissions,
    "env_var": _gen_env_var,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def shell_quest_generator(seed: int) -> Problem:
    """Generate a ShellQuest challenge problem.

    Difficulty scales which challenge types are more likely. An UNLIMITED
    number of distinct challenges can be produced via the seed.
    """
    rng = random.Random(seed)
    difficulty = 0.2 + 0.7 * rng.random()
    if difficulty < 0.4:
        choices = ["base64", "directory_listing", "permissions", "env_var"]
    elif difficulty < 0.7:
        choices = ["base64", "directory_listing", "grep_flag", "env_var", "suid_find"]
    else:
        choices = ["suid_find", "grep_flag", "directory_listing", "permissions"]
    ctype = rng.choice(choices)
    challenge = _GENERATORS[ctype](rng, difficulty)

    prompt = (
        f"{challenge['challenge_text']}\n"
        f"Hint: {challenge['hint']}\n\n"
        f"Respond with the answer in the format ANSWER: <answer>.\n"
        f"If the answer is a flag, use the flag{{...}} format."
    )

    return Problem(
        id=f"shell_quest_{ctype}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": "shell_quest",
            "challenge_type": ctype,
            "challenge_text": challenge["challenge_text"],
            "flag": challenge["flag"],
            "hint": challenge["hint"],
        },
        token_budget=1000,
        source="shell_quest_generator",
    )


shell_quest_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


_TYPE_KEYWORDS = {
    "base64": ["base64", "decode", "b64"],
    "directory_listing": ["ls", "listing", "directory"],
    "suid_find": ["suid", "find", "perm", "4000"],
    "grep_flag": ["grep", "log", "search"],
    "permissions": ["permission", "octal", "chmod", "mode"],
    "env_var": ["env", "environment", "variable"],
}


class ShellQuestVerifier(Verifier):
    """Verify a ShellQuest response by checking the answer/flag.

    reward = answer_correct * 0.8 + challenge_type_named * 0.2
    """

    def __init__(self, flag: str, challenge_type: str):
        super().__init__()
        self._flag = flag.strip()
        self._challenge_type = challenge_type

    def verify(self, response: str) -> VerifierResult:
        answer_correct = self._check_answer(response)
        type_named = self._check_type(response)

        score = answer_correct * 0.8 + type_named * 0.2
        correct = answer_correct >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "answer_correct": answer_correct,
                "challenge_type_named": type_named,
            },
            diagnostics=(
                f"answer={'correct' if answer_correct else 'missed'} "
                f"type={'named' if type_named else 'missed'} "
                f"(expected={self._flag!r})"
            ),
        )

    def _check_answer(self, response: str) -> float:
        """Return 1.0 if the exact flag/answer is present, else partial/0."""
        # Look for ANSWER: <answer> pattern first.
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        candidate = m.group(1).strip() if m else response
        if self._flag in candidate:
            return 1.0
        # Fallback: flag anywhere in the response.
        if self._flag in response:
            return 1.0
        # Partial credit for the flag body without braces.
        body = self._flag.strip("flag{}")
        if body and body in response:
            return 0.5
        return 0.0

    def _check_type(self, response: str) -> float:
        """Return 1.0 if the model names the correct challenge type."""
        keywords = _TYPE_KEYWORDS.get(self._challenge_type, [])
        low = response.lower()
        if any(kw in low for kw in keywords):
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ShellQuestEnv(BatchEnvBase):
    """ShellQuest: Unix shell CTF skills in a sandboxed environment.

    Batch-aware: N parallel attempts are scored, and the best attempt
    determines the reward (best-of-N). All challenges are synthetic and
    sandboxed — no real shell execution or network access.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(
        self,
        problems: Optional[list[Problem]] = None,
        problem_generator=None,
        reward_config=None,
        anti_pattern_detector=None,
        render_mode=None,
        batch_size: int = 16,
    ):
        if problems is None and problem_generator is None:
            problem_generator = shell_quest_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return ShellQuestVerifier(
            flag=problem.metadata["flag"],
            challenge_type=problem.metadata["challenge_type"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"ANSWER:\s*\S+", response, re.IGNORECASE):
            return 1.0
        if "flag{" in response.lower():
            return 0.8
        if len(response.strip()) > 10:
            return 0.3
        return 0.0

    def _aggregate_scores(self, per_sample: list[dict]) -> dict:
        scores = [s["verifier_score"] for s in per_sample]
        corrects = [s["correct"] for s in per_sample]
        best_score = max(scores) if scores else 0.0
        any_correct = any(corrects)

        return {
            "best_score": best_score,
            "any_correct": any_correct,
            "correct_count": sum(corrects),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "reward": best_score,
            "correct": any_correct,
            "diagnostics": (
                f"Best={best_score:.2f} Correct={sum(corrects)}/{len(per_sample)} "
                f"Mean={sum(scores)/len(scores) if scores else 0:.2f}"
            ),
        }

    def _extract_answer(self, response: str) -> str:
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return m.group(1).strip() if m else response
