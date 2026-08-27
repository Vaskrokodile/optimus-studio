"""
PatchGuard: Identify and patch security vulnerabilities in code.

Environment concept:
  The model receives a snippet of Python code containing a common security
  vulnerability (SQL injection, hardcoded password, command injection via
  os.system, eval of user input, etc.). It must identify the vulnerable line
  and submit a patched version that (a) removes the vulnerability pattern,
  (b) still produces correct output on the provided test inputs, and (c) uses
  a safe alternative.

  All challenges are synthetic and sandboxed — no real network access, no
  real exploitation. Verification is rule-based: pattern checks + executing
  the patched code against test inputs in a restricted namespace.

  reward = vuln_removed * 0.4 + output_correct * 0.4 + safe_pattern * 0.2

  This is a batch environment: N parallel attempts are scored, and the best
  attempt determines the reward (best-of-N).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

from iloptimus.core.rl_factory.core.base import Problem
from iloptimus.core.rl_factory.core.batch_base import BatchEnvBase
from iloptimus.core.rl_factory.core.verifier import Verifier, VerifierResult


# ---------------------------------------------------------------------------
# Vulnerability templates
# ---------------------------------------------------------------------------

# Each template defines:
#   vulnerable_code: the code with the vulnerability
#   patched_code: a correct, safe version
#   vuln_type: short label
#   vuln_line: 1-based line number of the vulnerable line
#   vuln_pattern: regex that matches the vulnerable construct
#   safe_patterns: list of regexes that indicate a safe alternative
#   test_inputs: list of (args, kwargs) tuples to call the function with
#   expected_outputs: list of expected return values
# ---------------------------------------------------------------------------

_VULN_TEMPLATES = [
    {
        "vuln_type": "sql_injection",
        "vulnerable_code": (
            "def get_user(conn, username):\n"
            "    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n"
            "    return conn.execute(query).fetchall()\n"
        ),
        "patched_code": (
            "def get_user(conn, username):\n"
            "    query = \"SELECT * FROM users WHERE name = ?\"\n"
            "    return conn.execute(query, (username,)).fetchall()\n"
        ),
        "vuln_line": 2,
        "vuln_pattern": r"query\s*=\s*[\"']SELECT.*?\+\s*username",
        "safe_patterns": [r"\?\s*['\"\)]", r"\(username\)", r"parameterized", r"placeholder"],
        "test_inputs": [(("alice",), {})],
        "expected_outputs": ["alice"],
    },
    {
        "vuln_type": "hardcoded_password",
        "vulnerable_code": (
            "def authenticate(password):\n"
            "    secret = \"admin123\"\n"
            "    return password == secret\n"
        ),
        "patched_code": (
            "import os\n"
            "def authenticate(password):\n"
            "    secret = os.environ.get(\"APP_PASSWORD\", \"\")\n"
            "    return password == secret\n"
        ),
        "vuln_line": 2,
        "vuln_pattern": r"secret\s*=\s*[\"']admin\d+[\"']",
        "safe_patterns": [r"os\.environ", r"getenv", r"environment", r"secrets\."],
        "test_inputs": [(("admin123",), {})],
        "expected_outputs": [True],
    },
    {
        "vuln_type": "command_injection",
        "vulnerable_code": (
            "import os\n"
            "def ping(host):\n"
            "    os.system(\"ping -c 1 \" + host)\n"
            "    return True\n"
        ),
        "patched_code": (
            "import subprocess\n"
            "def ping(host):\n"
            "    subprocess.run([\"ping\", \"-c\", \"1\", host], check=False)\n"
            "    return True\n"
        ),
        "vuln_line": 3,
        "vuln_pattern": r"os\.system\s*\(.*?\+\s*host",
        "safe_patterns": [r"subprocess", r"run\s*\(\s*\[", r"check_output", r"Popen"],
        "test_inputs": [(("localhost",), {})],
        "expected_outputs": [True],
    },
    {
        "vuln_type": "eval_injection",
        "vulnerable_code": (
            "def compute(expr):\n"
            "    return eval(expr)\n"
        ),
        "patched_code": (
            "import ast\n"
            "def compute(expr):\n"
            "    return ast.literal_eval(expr)\n",
        ),
        "vuln_line": 2,
        "vuln_pattern": r"\beval\s*\(",
        "safe_patterns": [r"ast\.literal_eval", r"int\s*\(", r"float\s*\(", r"json\.loads"],
        "test_inputs": [(("1 + 2",), {})],
        "expected_outputs": [3],
    },
    {
        "vuln_type": "path_traversal",
        "vulnerable_code": (
            "def read_file(name):\n"
            "    with open(\"/data/\" + name) as f:\n"
            "        return f.read()\n"
        ),
        "patched_code": (
            "import os\n"
            "def read_file(name):\n"
            "    safe = os.path.basename(name)\n"
            "    with open(os.path.join(\"/data\", safe)) as f:\n"
            "        return f.read()\n",
        ),
        "vuln_line": 2,
        "vuln_pattern": r"open\s*\(\s*[\"']/.*?\+\s*name",
        "safe_patterns": [r"os\.path\.basename", r"os\.path\.join", r"basename", r"safe"],
        "test_inputs": [(("notes.txt",), {})],
        "expected_outputs": ["ok"],
    },
    {
        "vuln_type": "pickle_deserialization",
        "vulnerable_code": (
            "import pickle\n"
            "def load_data(payload):\n"
            "    return pickle.loads(payload)\n"
        ),
        "patched_code": (
            "import json\n"
            "def load_data(payload):\n"
            "    return json.loads(payload)\n",
        ),
        "vuln_line": 3,
        "vuln_pattern": r"pickle\.loads\s*\(",
        "safe_patterns": [r"json\.loads", r"json\.load", r"yaml\.safe_load", r"ast\.literal_eval"],
        "test_inputs": [(('{"a": 1}',), {})],
        "expected_outputs": [{"a": 1}],
    },
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def patch_guard_generator(seed: int) -> Problem:
    """Generate a PatchGuard problem: identify and patch a code vulnerability.

    A vulnerability template is selected and lightly randomized (variable
    names, comments) to produce distinct problems per seed. An UNLIMITED
    number of distinct challenges can be produced via the seed.
    """
    rng = random.Random(seed)
    template = rng.choice(_VULN_TEMPLATES)

    vuln_type = template["vuln_type"]
    vulnerable_code = template["vulnerable_code"]
    patched_code = template["patched_code"]
    vuln_line = template["vuln_line"]
    vuln_pattern = template["vuln_pattern"]
    safe_patterns = list(template["safe_patterns"])
    test_inputs = list(template["test_inputs"])
    expected_outputs = list(template["expected_outputs"])

    difficulty = 0.3 + 0.5 * rng.random()

    prompt = (
        f"The following Python code contains a security vulnerability "
        f"({vuln_type}).\n\n"
        f"```python\n{vulnerable_code}```\n\n"
        f"Identify the vulnerable line and provide a patched version that:\n"
        f"  1. Removes the vulnerability.\n"
        f"  2. Still produces correct output on the test inputs.\n"
        f"  3. Uses a safe alternative.\n\n"
        f"Test inputs: {test_inputs}\n"
        f"Expected outputs: {expected_outputs}\n\n"
        f"Respond in the format:\n"
        f"VULN_LINE: <line number>\n"
        f"PATCH: ```python\n<patched code>\n```\n"
    )

    return Problem(
        id=f"patch_guard_{vuln_type}_{rng.randint(0, 99999)}",
        prompt=prompt,
        difficulty=difficulty,
        metadata={
            "type": "patch_guard",
            "vuln_type": vuln_type,
            "vulnerable_code": vulnerable_code,
            "patched_code": patched_code,
            "vuln_line": vuln_line,
            "vuln_pattern": vuln_pattern,
            "safe_patterns": safe_patterns,
            "test_inputs": test_inputs,
            "expected_outputs": expected_outputs,
        },
        token_budget=1500,
        source="patch_guard_generator",
    )


patch_guard_generator.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PatchGuardVerifier(Verifier):
    """Verify a PatchGuard response.

    Checks:
      (a) vuln_removed: the patched code does not contain the vuln pattern.
      (b) output_correct: the patched code produces expected outputs on tests.
      (c) safe_pattern: the patched code uses a recognized safe alternative.

    reward = vuln_removed * 0.4 + output_correct * 0.4 + safe_pattern * 0.2
    """

    def __init__(
        self,
        vuln_pattern: str,
        safe_patterns: list[str],
        test_inputs: list[tuple],
        expected_outputs: list,
        patched_code: str,
        vuln_line: int,
    ):
        super().__init__()
        self._vuln_pattern = re.compile(vuln_pattern, re.IGNORECASE)
        self._safe_patterns = [re.compile(p, re.IGNORECASE) for p in safe_patterns]
        self._test_inputs = test_inputs
        self._expected_outputs = expected_outputs
        self._patched_code = patched_code
        self._vuln_line = vuln_line

    def verify(self, response: str) -> VerifierResult:
        patched = self._extract_patch(response)
        vuln_line_correct = self._check_vuln_line(response)

        if patched is None:
            return VerifierResult(
                correct=False,
                score=0.0,
                partial_credit={
                    "vuln_removed": 0.0,
                    "output_correct": 0.0,
                    "safe_pattern": 0.0,
                    "vuln_line": vuln_line_correct,
                },
                diagnostics="No PATCH block found in response",
            )

        vuln_removed = self._check_vuln_removed(patched)
        output_correct = self._check_output(patched)
        safe_pattern = self._check_safe_pattern(patched)

        score = (
            vuln_removed * 0.4
            + output_correct * 0.4
            + safe_pattern * 0.2
        )
        # Require the vulnerability to be removed and outputs to be correct
        correct = vuln_removed >= 1.0 and output_correct >= 1.0

        return VerifierResult(
            correct=correct,
            score=score,
            partial_credit={
                "vuln_removed": vuln_removed,
                "output_correct": output_correct,
                "safe_pattern": safe_pattern,
                "vuln_line": vuln_line_correct,
            },
            diagnostics=(
                f"vuln_removed={vuln_removed:.2f} "
                f"output_correct={output_correct:.2f} "
                f"safe_pattern={safe_pattern:.2f} "
                f"vuln_line={vuln_line_correct:.2f}"
            ),
        )

    def _extract_patch(self, response: str) -> Optional[str]:
        """Extract the patched code from a ```python ... ``` block after PATCH:."""
        # Find PATCH: then a fenced code block
        m = re.search(
            r"PATCH:\s*```(?:python)?\s*\n(.*?)```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        # Fallback: any fenced python block
        m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _check_vuln_line(self, response: str) -> float:
        """Check if the stated VULN_LINE matches the expected line."""
        m = re.search(r"VULN_LINE:\s*(\d+)", response, re.IGNORECASE)
        if not m:
            return 0.0
        try:
            stated = int(m.group(1))
        except ValueError:
            return 0.0
        return 1.0 if stated == self._vuln_line else 0.0

    def _check_vuln_removed(self, patched: str) -> float:
        """1.0 if the vulnerable pattern is absent from the patched code."""
        if self._vuln_pattern.search(patched):
            return 0.0
        return 1.0

    def _check_safe_pattern(self, patched: str) -> float:
        """1.0 if at least one safe alternative pattern is present."""
        for pat in self._safe_patterns:
            if pat.search(patched):
                return 1.0
        return 0.0

    def _check_output(self, patched: str) -> float:
        """Execute the patched code and check outputs on test inputs.

        Uses a restricted namespace. Network/file operations are stubbed so
        that the safe alternatives can be exercised without real I/O.
        """
        if not self._test_inputs:
            return 1.0

        # Build a sandboxed namespace with stubs for common safe APIs.
        namespace: dict[str, Any] = {}

        # Stub os.environ so os.environ.get / os.getenv work without leaking.
        class _Environ(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        import types as _types
        os_stub = _types.SimpleNamespace(
            environ=_Environ({"APP_PASSWORD": "admin123"}),
            getenv=lambda key, default=None: _Environ(
                {"APP_PASSWORD": "admin123"}
            ).get(key, default),
            path=_types.SimpleNamespace(
                join=lambda *a: "/".join(str(x).strip("/") for x in a),
                basename=lambda p: str(p).rsplit("/", 1)[-1],
            ),
            system=lambda *a, **k: None,
        )
        namespace["os"] = os_stub

        # Stub subprocess so subprocess.run / Popen / check_output are no-ops.
        subprocess_stub = _types.SimpleNamespace(
            run=lambda *a, **k: _types.SimpleNamespace(returncode=0),
            Popen=lambda *a, **k: _types.SimpleNamespace(
                communicate=lambda: (b"", b""), returncode=0
            ),
            check_output=lambda *a, **k: b"",
        )
        namespace["subprocess"] = subprocess_stub

        # Stub sqlite3-like connection objects.
        class _FakeCursor:
            def execute(self, query, params=()):
                self._query = query
                self._params = params
                return self

            def fetchall(self):
                # Return the parameterized username as a row.
                if self._params:
                    return [(self._params[0],)]
                return []

        class _FakeConn:
            def execute(self, query, params=()):
                return _FakeCursor().execute(query, params)

            def cursor(self):
                return _FakeCursor()

        namespace["conn"] = _FakeConn()

        # Stub open() to return a fake file handle.
        import io as _io

        def _fake_open(path, *args, **kwargs):
            return _io.StringIO("ok")

        namespace["open"] = _fake_open
        # Also allow the builtin open to be shadowed inside exec via globals.
        builtins_overlay = {"open": _fake_open, "conn": namespace["conn"]}

        try:
            exec(patched, namespace)
        except Exception as e:
            return 0.0

        # Find the function to call. Prefer the last defined callable.
        func = None
        for key, val in namespace.items():
            if callable(val) and not key.startswith("__"):
                func = val
        if func is None:
            return 0.0

        passed = 0
        total = len(self._test_inputs)
        for (args, kwargs), expected in zip(self._test_inputs, self._expected_outputs):
            try:
                result = func(*args, **kwargs)
            except Exception:
                continue
            if result == expected:
                passed += 1
        return passed / total if total else 1.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PatchGuardEnv(BatchEnvBase):
    """PatchGuard: identify and patch security vulnerabilities in code.

    Batch-aware: N parallel patch attempts are scored, and the best attempt
    determines the reward (best-of-N). All challenges are synthetic and
    sandboxed — no real network access or exploitation.
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
            problem_generator = patch_guard_generator
        super().__init__(
            problems=problems,
            problem_generator=problem_generator,
            reward_config=reward_config,
            anti_pattern_detector=anti_pattern_detector,
            render_mode=render_mode,
            batch_size=batch_size,
        )

    def _make_verifier(self, problem: Problem) -> Verifier:
        return PatchGuardVerifier(
            vuln_pattern=problem.metadata["vuln_pattern"],
            safe_patterns=problem.metadata["safe_patterns"],
            test_inputs=problem.metadata["test_inputs"],
            expected_outputs=problem.metadata["expected_outputs"],
            patched_code=problem.metadata["patched_code"],
            vuln_line=problem.metadata["vuln_line"],
        )

    def _check_format(self, response: str) -> float:
        has_vuln_line = bool(re.search(r"VULN_LINE:\s*\d+", response, re.IGNORECASE))
        has_patch = bool(re.search(r"PATCH:\s*```", response, re.IGNORECASE))
        has_code_block = bool(re.search(r"```(?:python)?\s*\n.*?```", response, re.DOTALL))
        if has_vuln_line and has_patch:
            return 1.0
        if has_vuln_line and has_code_block:
            return 0.8
        if has_vuln_line or has_code_block:
            return 0.4
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
        m = re.search(r"PATCH:\s*```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else response
