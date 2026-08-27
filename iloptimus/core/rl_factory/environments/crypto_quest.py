"""
CryptoQuest: Procedurally generated cryptographic CTF challenges.

Environment concept:
  The model receives a ciphertext and a hint about the algorithm used.
  It must decrypt the message to recover a hidden flag (a synthetic CTF
  flag like ``flag{...}``). All challenges are generated locally with
  pure-Python implementations of classical ciphers — no real network
  access, no real exploitation, everything sandboxed.

  Challenge types (parameterized difficulty):
    - Caesar cipher (shift size controls difficulty)
    - Vigenere cipher (key length controls difficulty)
    - XOR cipher (key length + byte range controls difficulty)
    - Simple substitution cipher (alphabet shuffle)
    - Frequency analysis (substitution with a longer ciphertext)
    - Small RSA (textbook RSA with small primes, modular exponentiation)

  Each challenge embeds a flag inside a plaintext message, encrypts it,
  and provides the ciphertext + an algorithm hint. Verification is
  rule-based: the model must produce the exact flag string.

  reward = flag_captured * 0.7 + method_correct * 0.3

  This is a batch environment: N parallel attempts are scored, and the
  best attempt determines the reward (best-of-N). ``method_correct`` is
  awarded when the model names the correct algorithm in its response.

  Security note: All cryptography here is intentionally weak/educational.
  No real secrets are ever handled. The RSA challenges use tiny primes
  (<= ~200) so factorization is trivial and purely synthetic.
"""

from __future__ import annotations

import random
import re
import string
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Crypto primitives (pure Python, educational only)
# ---------------------------------------------------------------------------

_ALPHA = string.ascii_uppercase


def _caesar_encrypt(plaintext: str, shift: int) -> str:
    """Encrypt with a Caesar (monoalphabetic shift) cipher. Preserves non-alpha."""
    out = []
    for ch in plaintext:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)


def _caesar_decrypt(ciphertext: str, shift: int) -> str:
    return _caesar_encrypt(ciphertext, -shift)


def _vigenere_encrypt(plaintext: str, key: str) -> str:
    """Encrypt with a Vigenere cipher using an alphabetic key."""
    key = [k for k in key.upper() if k.isalpha()]
    if not key:
        return plaintext
    out = []
    ki = 0
    for ch in plaintext:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            shift = ord(key[ki % len(key)]) - ord("A")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def _vigenere_decrypt(ciphertext: str, key: str) -> str:
    key = [k for k in key.upper() if k.isalpha()]
    if not key:
        return ciphertext
    out = []
    ki = 0
    for ch in ciphertext:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            shift = ord(key[ki % len(key)]) - ord("A")
            out.append(chr((ord(ch) - base - shift) % 26 + base))
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def _xor_encrypt(plaintext: str, key: str) -> str:
    """XOR each character byte with a repeating key; return hex-encoded output."""
    data = plaintext.encode("utf-8")
    kbytes = key.encode("utf-8")
    out = bytes(b ^ kbytes[i % len(kbytes)] for i, b in enumerate(data))
    return out.hex()


def _xor_decrypt(hex_ct: str, key: str) -> str:
    data = bytes.fromhex(hex_ct)
    kbytes = key.encode("utf-8")
    out = bytes(b ^ kbytes[i % len(kbytes)] for i, b in enumerate(data))
    return out.decode("utf-8", errors="replace")


def _substitution_encrypt(plaintext: str, mapping: dict[str, str]) -> str:
    """Encrypt with a simple substitution cipher using a letter->letter mapping."""
    out = []
    for ch in plaintext:
        if ch.isalpha():
            up = ch.upper()
            sub = mapping.get(up, up)
            out.append(sub if ch.isupper() else sub.lower())
        else:
            out.append(ch)
    return "".join(out)


def _substitution_decrypt(ciphertext: str, mapping: dict[str, str]) -> str:
    inverse = {v: k for k, v in mapping.items()}
    out = []
    for ch in ciphertext:
        if ch.isalpha():
            up = ch.upper()
            orig = inverse.get(up, up)
            out.append(orig if ch.isupper() else orig.lower())
        else:
            out.append(ch)
    return "".join(out)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _small_primes(lo: int, hi: int) -> list[int]:
    return [n for n in range(lo, hi + 1) if _is_prime(n)]


def _modinv(a: int, m: int) -> int:
    """Modular inverse via extended Euclid. Assumes gcd(a, m) == 1."""
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("no modular inverse")
    return x % m


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def _rsa_encrypt(plaintext: str, e: int, n: int) -> str:
    """Textbook RSA on small numeric 'messages'. Each char -> cipher int, comma-joined."""
    return ",".join(str(pow(ord(ch), e, n)) for ch in plaintext)


def _rsa_decrypt(ciphertext: str, d: int, n: int) -> str:
    parts = [p.strip() for p in ciphertext.split(",") if p.strip()]
    return "".join(chr(pow(int(p), d, n)) for p in parts)


# ---------------------------------------------------------------------------
# Challenge generators
# ---------------------------------------------------------------------------

_FLAG_WORDS = [
    "access", "granted", "breach", "secure", "decoded", "cipher", "hidden",
    "secret", "vault", "unlocked", "crypto", "enigma", "phantom", "shadow",
    "matrix", "kernel", "root", "admin", "token", "beacon",
]


def _make_flag(rng: random.Random) -> str:
    """Generate a synthetic CTF flag string."""
    word = rng.choice(_FLAG_WORDS)
    num = rng.randint(100, 9999)
    return f"flag{{{word}_{num}}}"


def _gen_caesar(rng: random.Random, difficulty: float) -> dict:
    shift = rng.randint(1, 25)
    flag = _make_flag(rng)
    plaintext = f"The secret flag is {flag}. Congratulations on decrypting."
    ct = _caesar_encrypt(plaintext, shift)
    return {
        "type": "caesar",
        "ciphertext": ct,
        "hint": "Caesar cipher (monoalphabetic shift)",
        "flag": flag,
        "method": "caesar",
        "params": {"shift": shift},
        "difficulty": difficulty,
    }


def _gen_vigenere(rng: random.Random, difficulty: float) -> dict:
    key_len = rng.randint(3, 4) if difficulty < 0.5 else rng.randint(5, 8)
    key = "".join(rng.choice(_ALPHA) for _ in range(key_len))
    flag = _make_flag(rng)
    plaintext = f"Decrypt complete. The recovered flag is {flag}."
    ct = _vigenere_encrypt(plaintext, key)
    return {
        "type": "vigenere",
        "ciphertext": ct,
        "hint": f"Vigenere cipher (key length {key_len})",
        "flag": flag,
        "method": "vigenere",
        "params": {"key": key},
        "difficulty": difficulty,
    }


def _gen_xor(rng: random.Random, difficulty: float) -> dict:
    key_len = 1 if difficulty < 0.3 else (rng.randint(2, 4) if difficulty < 0.7 else rng.randint(5, 8))
    key = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(key_len))
    flag = _make_flag(rng)
    plaintext = f"XOR decoded. Flag value: {flag}"
    ct = _xor_encrypt(plaintext, key)
    return {
        "type": "xor",
        "ciphertext": ct,
        "hint": f"XOR cipher (key length {key_len}, hex-encoded output)",
        "flag": flag,
        "method": "xor",
        "params": {"key": key},
        "difficulty": difficulty,
    }


def _gen_substitution(rng: random.Random, difficulty: float) -> dict:
    shuffled = list(_ALPHA)
    rng.shuffle(shuffled)
    mapping = dict(zip(_ALPHA, shuffled))
    flag = _make_flag(rng)
    plaintext = f"Substitution solved. Here is the flag: {flag}"
    ct = _substitution_encrypt(plaintext, mapping)
    return {
        "type": "substitution",
        "ciphertext": ct,
        "hint": "Simple substitution cipher (monoalphabetic)",
        "flag": flag,
        "method": "substitution",
        "params": {"mapping": mapping},
        "difficulty": difficulty,
    }


def _gen_frequency(rng: random.Random, difficulty: float) -> dict:
    """Frequency-analysis challenge: substitution with a longer ciphertext."""
    shuffled = list(_ALPHA)
    rng.shuffle(shuffled)
    mapping = dict(zip(_ALPHA, shuffled))
    flag = _make_flag(rng)
    filler = (
        "the quick brown fox jumps over the lazy dog again and again "
        "because cryptography is fun when you practice frequency analysis "
        "on long enough texts that letter distributions become apparent "
        "to anyone patient enough to count them carefully over time "
    )
    plaintext = filler * (2 if difficulty < 0.6 else 3) + f" flag:{flag}"
    ct = _substitution_encrypt(plaintext, mapping)
    return {
        "type": "frequency",
        "ciphertext": ct,
        "hint": "Substitution cipher — use frequency analysis (long ciphertext)",
        "flag": flag,
        "method": "frequency",
        "params": {"mapping": mapping},
        "difficulty": difficulty,
    }


def _gen_rsa(rng: random.Random, difficulty: float) -> dict:
    """Small textbook RSA. Primes kept tiny so factorization is trivial/synthetic."""
    primes = _small_primes(11, 50) if difficulty < 0.6 else _small_primes(11, 200)
    p = rng.choice(primes)
    q = rng.choice(primes)
    while q == p:
        q = rng.choice(primes)
    n = p * q
    phi = (p - 1) * (q - 1)
    # choose e coprime with phi
    e = 3
    while e < phi:
        try:
            _modinv(e, phi)
            break
        except ValueError:
            e += 2
    d = _modinv(e, phi)
    flag = _make_flag(rng)
    # Only encrypt the flag itself to keep ciphertext manageable
    ct = _rsa_encrypt(flag, e, n)
    return {
        "type": "rsa",
        "ciphertext": ct,
        "hint": f"Textbook RSA (n={n}, e={e}) — small primes, factor n to decrypt",
        "flag": flag,
        "method": "rsa",
        "params": {"p": p, "q": q, "e": e, "d": d, "n": n},
        "difficulty": difficulty,
    }


_GENERATORS = {
    "caesar": _gen_caesar,
    "vigenere": _gen_vigenere,
    "xor": _gen_xor,
    "substitution": _gen_substitution,
    "frequency": _gen_frequency,
    "rsa": _gen_rsa,
}


def crypto_quest_generator(seed: int) -> Problem:
    """Generate a CryptoQuest challenge problem.

    Difficulty scales the cipher complexity (key length, prime size, etc.).
    An UNLIMITED number of distinct challenges can be produced via the seed.
    """
    rng = random.Random(seed)
    difficulty = 0.2 + 0.7 * rng.random()
    # Weight selection: easier ciphers more likely at low difficulty
    if difficulty < 0.35:
        choices = ["caesar", "caesar", "xor", "substitution"]
    elif difficulty < 0.65:
        choices = ["caesar", "vigenere", "xor", "substitution", "frequency"]
    else:
        choices = ["vigenere", "frequency", "rsa", "xor"]
    ctype = rng.choice(choices)
    challenge = _GENERATORS[ctype](rng, difficulty)

    prompt = (
        f"Decrypt the message to find the flag:\n"
        f"Ciphertext: {challenge['ciphertext']}\n"
        f"Algorithm: {challenge['hint']}\n"
        f"FLAG: <answer>\n\n"
        f"Respond with the decrypted flag in the format FLAG: flag{{...}}.\n"
        f"You may also briefly state the method you used."
    )

    return Problem(
        id=f"crypto_quest_{ctype}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": "crypto_quest",
            "challenge_type": ctype,
            "flag": challenge["flag"],
            "method": challenge["method"],
            "params": challenge["params"],
            "ciphertext": challenge["ciphertext"],
        },
        token_budget=1500,
        source="generated",
    )


# Prevent pytest from collecting this generator as a test
crypto_quest_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


_METHOD_KEYWORDS = {
    "caesar": ["caesar", "shift"],
    "vigenere": ["vigenere", "vigenère"],
    "xor": ["xor", "exclusive"],
    "substitution": ["substitution", "monoalphabetic"],
    "frequency": ["frequency", "substitution", "statistical"],
    "rsa": ["rsa", "factor", "modular", "exponent"],
}


class CryptoQuestVerifier(Verifier):
    """Verifies a CryptoQuest response by checking the flag and method.

    reward = flag_captured * 0.7 + method_correct * 0.3
    """

    def __init__(self, flag: str, method: str):
        super().__init__()
        self._flag = flag.strip()
        self._method = method

    def verify(self, response: str) -> VerifierResult:
        flag_captured = self._check_flag(response)
        method_correct = self._check_method(response)

        score = flag_captured * 0.7 + method_correct * 0.3
        correct = flag_captured >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "flag_captured": flag_captured,
                "method_correct": method_correct,
            },
            diagnostics=(
                f"flag={'captured' if flag_captured else 'missed'} "
                f"method={'correct' if method_correct else 'missed'} "
                f"(expected flag={self._flag!r})"
            ),
        )

    def _check_flag(self, response: str) -> float:
        """Return 1.0 if the exact flag is present, else 0.0."""
        # Look for FLAG: <flag> pattern first, then any occurrence of the flag
        m = re.search(r"FLAG:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        candidate = m.group(1).strip() if m else response
        if self._flag in candidate:
            return 1.0
        # Fallback: flag anywhere in the response
        if self._flag in response:
            return 1.0
        # Partial credit for the flag body without braces
        body = self._flag.strip("flag{}")
        if body and body in response:
            return 0.5
        return 0.0

    def _check_method(self, response: str) -> float:
        """Return 1.0 if the model names the correct method, else 0.0."""
        keywords = _METHOD_KEYWORDS.get(self._method, [])
        low = response.lower()
        if any(kw in low for kw in keywords):
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CryptoQuestEnv(BatchEnvBase):
    """CryptoQuest: procedurally generated cryptographic CTF challenges.

    Batch-aware: N parallel decryption attempts are scored, and the best
    attempt determines the reward (best-of-N). All challenges are synthetic
    and sandboxed — no real network access or exploitation.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, problems=None, problem_generator=None, reward_config=None,
                 anti_pattern_detector=None, render_mode=None, batch_size: int = 16):
        if problems is None and problem_generator is None:
            problem_generator = crypto_quest_generator
        super().__init__(problems=problems, problem_generator=problem_generator,
                        reward_config=reward_config, anti_pattern_detector=anti_pattern_detector,
                        render_mode=render_mode, batch_size=batch_size)

    def _make_verifier(self, problem: Problem) -> Verifier:
        return CryptoQuestVerifier(
            flag=problem.metadata["flag"],
            method=problem.metadata["method"],
        )

    def _check_format(self, response: str) -> float:
        if re.search(r"FLAG:\s*\S+", response, re.IGNORECASE):
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
        m = re.search(r"FLAG:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        return m.group(1).strip() if m else response
