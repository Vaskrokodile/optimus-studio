"""
BinaryAnalysis: Reverse engineering challenges with encoded/obfuscated data.

Environment concept:
  The model receives a synthetic reverse-engineering challenge. No real
  binaries are used — all data is encoded/obfuscated with pure-Python
  transformations so everything is sandboxed and safe.

  Challenge types:
    - hex_string: decode a hex-encoded ASCII string to recover the flag
    - xor_hex: XOR a hex string with a single-byte key to recover the flag
    - rot_cipher: reverse a ROT-N cipher on the flag
    - base32: decode a base32-encoded flag
    - reverse_string: reverse a backwards-encoded flag string
    - ascii_offsets: each flag char is given as an ASCII code offset list

  Verification is rule-based: the model must produce the exact decoded flag.

  reward = flag_correct * 0.8 + method_named * 0.2

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
    "binary", "reverse", "disasm", "gadget", "payload", "exploit",
    "stack", "heap", "overflow", "register", "opcode", "syscall",
    "decompile", "deobfuscate", "cipher", "decode", "hexdump", "trace",
]


def _make_flag(rng: random.Random) -> str:
    word = rng.choice(_FLAG_WORDS)
    num = rng.randint(100, 9999)
    return f"flag{{{word}_{num}}}"


# ---------------------------------------------------------------------------
# Challenge generators
# ---------------------------------------------------------------------------


def _gen_hex_string(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    encoded = flag.encode().hex()
    return {
        "type": "hex_string",
        "encoded_data": encoded,
        "encoding_method": "hex (ASCII bytes -> hex string)",
        "flag": flag,
        "hint": "Convert each pair of hex digits back to a character.",
    }


def _gen_xor_hex(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    key = rng.randint(1, 255)
    data = flag.encode()
    encoded = bytes(b ^ key for b in data).hex()
    return {
        "type": "xor_hex",
        "encoded_data": encoded,
        "encoding_method": f"single-byte XOR with key 0x{key:02x}, hex-encoded",
        "flag": flag,
        "hint": "XOR each byte with the same single-byte key, then decode ASCII.",
    }


def _gen_rot_cipher(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    shift = rng.randint(1, 25)
    # Only rotate alphabetic characters; preserve flag{} structure is NOT
    # possible with ROT, so we rotate the whole flag including braces.
    out = []
    for ch in flag:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    encoded = "".join(out)
    return {
        "type": "rot_cipher",
        "encoded_data": encoded,
        "encoding_method": f"ROT-{shift} (Caesar shift on letters)",
        "flag": flag,
        "hint": "Try all 25 rotations; the flag format flag{...} will appear.",
    }


def _gen_base32(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    encoded = base64.b32encode(flag.encode()).decode()
    return {
        "type": "base32",
        "encoded_data": encoded,
        "encoding_method": "base32 encoding",
        "flag": flag,
        "hint": "Decode the base32 string to recover ASCII.",
    }


def _gen_reverse_string(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    encoded = flag[::-1]
    return {
        "type": "reverse_string",
        "encoded_data": encoded,
        "encoding_method": "string reversal",
        "flag": flag,
        "hint": "The string is written backwards; reverse it.",
    }


def _gen_ascii_offsets(rng: random.Random, difficulty: float) -> dict:
    flag = _make_flag(rng)
    encoded = ",".join(str(ord(ch)) for ch in flag)
    return {
        "type": "ascii_offsets",
        "encoded_data": encoded,
        "encoding_method": "ASCII code points (comma-separated)",
        "flag": flag,
        "hint": "Each number is the ASCII code of a character.",
    }


_GENERATORS = {
    "hex_string": _gen_hex_string,
    "xor_hex": _gen_xor_hex,
    "rot_cipher": _gen_rot_cipher,
    "base32": _gen_base32,
    "reverse_string": _gen_reverse_string,
    "ascii_offsets": _gen_ascii_offsets,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def binary_analysis_generator(seed: int) -> Problem:
    """Generate a BinaryAnalysis reverse-engineering challenge problem.

    Difficulty scales which encoding methods are more likely. An UNLIMITED
    number of distinct challenges can be produced via the seed.
    """
    rng = random.Random(seed)
    difficulty = 0.2 + 0.7 * rng.random()
    if difficulty < 0.4:
        choices = ["hex_string", "reverse_string", "ascii_offsets", "base32"]
    elif difficulty < 0.7:
        choices = ["hex_string", "rot_cipher", "reverse_string", "base32", "ascii_offsets"]
    else:
        choices = ["xor_hex", "rot_cipher", "hex_string", "ascii_offsets"]
    ctype = rng.choice(choices)
    challenge = _GENERATORS[ctype](rng, difficulty)

    prompt = (
        f"Reverse-engineer the following encoded data to recover the hidden "
        f"flag.\n\n"
        f"Encoded data: {challenge['encoded_data']}\n"
        f"Encoding method: {challenge['encoding_method']}\n"
        f"Hint: {challenge['hint']}\n\n"
        f"Respond with the decoded flag in the format ANSWER: flag{{...}}.\n"
        f"You may briefly state the method you used."
    )

    return Problem(
        id=f"binary_analysis_{ctype}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": "binary_analysis",
            "challenge_type": ctype,
            "encoded_data": challenge["encoded_data"],
            "encoding_method": challenge["encoding_method"],
            "flag": challenge["flag"],
            "hint": challenge["hint"],
        },
        token_budget=1200,
        source="binary_analysis_generator",
    )


binary_analysis_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


_METHOD_KEYWORDS = {
    "hex_string": ["hex", "ascii"],
    "xor_hex": ["xor", "exclusive", "key"],
    "rot_cipher": ["rot", "caesar", "shift", "rotation"],
    "base32": ["base32", "b32"],
    "reverse_string": ["reverse", "backwards", "backward"],
    "ascii_offsets": ["ascii", "code", "ord", "point"],
}


class BinaryAnalysisVerifier(Verifier):
    """Verify a BinaryAnalysis response by checking the decoded flag.

    reward = flag_correct * 0.8 + method_named * 0.2
    """

    def __init__(self, flag: str, challenge_type: str):
        super().__init__()
        self._flag = flag.strip()
        self._challenge_type = challenge_type

    def verify(self, response: str) -> VerifierResult:
        flag_correct = self._check_flag(response)
        method_named = self._check_method(response)

        score = flag_correct * 0.8 + method_named * 0.2
        correct = flag_correct >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "flag_correct": flag_correct,
                "method_named": method_named,
            },
            diagnostics=(
                f"flag={'correct' if flag_correct else 'missed'} "
                f"method={'named' if method_named else 'missed'} "
                f"(expected={self._flag!r})"
            ),
        )

    def _check_flag(self, response: str) -> float:
        """Return 1.0 if the exact flag is present, else partial/0."""
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        candidate = m.group(1).strip() if m else response
        if self._flag in candidate:
            return 1.0
        if self._flag in response:
            return 1.0
        # Partial credit for the flag body without braces.
        body = self._flag.strip("flag{}")
        if body and body in response:
            return 0.5
        return 0.0

    def _check_method(self, response: str) -> float:
        """Return 1.0 if the model names the correct method."""
        keywords = _METHOD_KEYWORDS.get(self._challenge_type, [])
        low = response.lower()
        if any(kw in low for kw in keywords):
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BinaryAnalysisEnv(BatchEnvBase):
    """BinaryAnalysis: reverse engineering challenges with encoded data.

    Batch-aware: N parallel attempts are scored, and the best attempt
    determines the reward (best-of-N). All challenges are synthetic and
    sandboxed — no real binaries or network access.
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
            problem_generator = binary_analysis_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return BinaryAnalysisVerifier(
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
