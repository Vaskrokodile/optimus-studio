"""
JavaExecutor — Python wrapper for the persistent JavaArtServer JVM.

Manages a single long-running JVM subprocess that accepts Java source code
via a binary protocol, compiles it in-memory, executes Art.paint(), and
returns PNG bytes. This eliminates per-sample JVM startup (~200ms each).

For ultra-batched RL rollouts, multiple JavaExecutor instances can run in
parallel (one JVM per worker), with a ThreadPoolExecutor dispatching
compilation jobs across them.

Protocol (matches JavaArtServer.java):
  Request:  [4-byte BE length] [UTF-8 source]
  Response: [1-byte status] [4-byte BE length] [payload]
    0 = success → PNG bytes
    1 = compile error → error text
    2 = runtime error → error text
    3 = timeout → "timeout"
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Status codes from JavaArtServer
STATUS_SUCCESS = 0
STATUS_COMPILE_ERROR = 1
STATUS_RUNTIME_ERROR = 2
STATUS_TIMEOUT = 3


@dataclass
class JavaExecutionResult:
    """Result of executing Java painting code."""
    success: bool
    png_bytes: Optional[bytes]
    error: Optional[str]
    error_type: Optional[str]  # "compile", "runtime", "timeout", "transport"
    execution_time_ms: float


class JavaExecutor:
    """
    Wraps a single persistent JavaArtServer JVM subprocess.

    Thread-safe: uses a lock to serialize stdin/stdout access per JVM.
    For parallelism, create multiple JavaExecutor instances and use
    JavaExecutorPool.
    """

    def __init__(
        self,
        server_jar_path: Optional[str] = None,
        java_cmd: str = "java",
        xms: str = "256m",
        xmx: str = "512m",
    ):
        self.java_cmd = java_cmd
        self.xms = xms
        self.xmx = xmx
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._server_source_path: Optional[str] = None

        # The JavaArtServer.java source lives alongside this module
        if server_jar_path is None:
            java_dir = Path(__file__).parent.parent / "java"
            self._server_source_path = str(java_dir / "JavaArtServer.java")

    def start(self) -> None:
        """Compile (if needed) and start the JVM subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            return

        # Compile the server if a .class doesn't exist or is stale
        class_file = self._server_source_path.replace(".java", ".class")
        if not os.path.exists(class_file) or \
           os.path.getmtime(class_file) < os.path.getmtime(self._server_source_path):
            compile_proc = subprocess.run(
                ["javac", "-nowarn", self._server_source_path],
                capture_output=True, text=True, timeout=30,
            )
            if compile_proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to compile JavaArtServer:\n{compile_proc.stderr}"
                )

        # Start the JVM
        self._proc = subprocess.Popen(
            [
                self.java_cmd,
                f"-Xms{self.xms}",
                f"-Xmx{self.xmx}",
                "-XX:+UseSerialGC",  # low memory, single-threaded GC is fine
                "-cp", os.path.dirname(self._server_source_path),
                "JavaArtServer",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it a moment to start
        time.sleep(0.3)
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"JavaArtServer failed to start:\n{stderr}")

    def stop(self) -> None:
        """Terminate the JVM subprocess."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def execute(self, source: str, timeout: float = 15.0) -> JavaExecutionResult:
        """
        Send Java source code to the JVM, get back a PNG or error.

        Args:
            source: Java source code defining a class with
                    `public static BufferedImage paint(int w, int h)`
            timeout: Transport-level timeout in seconds (separate from
                     the JVM's internal 10s execution timeout).

        Returns:
            JavaExecutionResult
        """
        if not self.is_alive():
            self.start()

        start_time = time.time()
        source_bytes = source.encode("utf-8")
        request = struct.pack(">I", len(source_bytes)) + source_bytes

        with self._lock:
            try:
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return JavaExecutionResult(
                    False, None, f"stdin write failed: {e}", "transport",
                    (time.time() - start_time) * 1000,
                )

            # Read response with transport timeout
            try:
                status_byte = self._read_exact(1, timeout)
                status = status_byte[0]

                len_bytes = self._read_exact(4, timeout)
                payload_len = struct.unpack(">I", len_bytes)[0]

                payload = self._read_exact(payload_len, timeout)
            except (TimeoutError, OSError, struct.error) as e:
                # JVM is likely hung — kill and restart
                self.stop()
                return JavaExecutionResult(
                    False, None, f"transport error: {e}", "transport",
                    (time.time() - start_time) * 1000,
                )

        elapsed_ms = (time.time() - start_time) * 1000

        if status == STATUS_SUCCESS:
            return JavaExecutionResult(True, payload, None, None, elapsed_ms)
        else:
            error_text = payload.decode("utf-8", errors="replace")
            error_type = {
                STATUS_COMPILE_ERROR: "compile",
                STATUS_RUNTIME_ERROR: "runtime",
                STATUS_TIMEOUT: "timeout",
            }.get(status, "unknown")
            return JavaExecutionResult(False, None, error_text, error_type, elapsed_ms)

    def _read_exact(self, n: int, timeout: float) -> bytes:
        """Read exactly n bytes from stdout, or raise TimeoutError."""
        data = bytearray()
        deadline = time.time() + timeout
        while len(data) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"read timeout after {timeout}s")
            # Use select-like polling on the pipe
            import select
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(self._proc.stdout.fileno(), n - len(data))
            if not chunk:
                raise OSError("stdout EOF")
            data.extend(chunk)
        return bytes(data)


class JavaExecutorPool:
    """
    Pool of JavaExecutor instances for parallel compilation/execution.

    Uses a round-robin assignment with a ThreadPoolExecutor. Each JVM
    processes one source at a time (serialized by its internal lock),
    so the pool size determines parallelism.
    """

    def __init__(
        self,
        num_workers: int = 4,
        java_cmd: str = "java",
        xms: str = "256m",
        xmx: str = "512m",
    ):
        self.num_workers = num_workers
        self._executors = [
            JavaExecutor(java_cmd=java_cmd, xms=xms, xmx=xmx)
            for _ in range(num_workers)
        ]
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=num_workers)

    def start(self) -> None:
        for ex in self._executors:
            ex.start()

    def stop(self) -> None:
        for ex in self._executors:
            ex.stop()
        self._pool.shutdown(wait=False)

    def execute(self, source: str, timeout: float = 15.0) -> JavaExecutionResult:
        """Execute a single source synchronously (dispatches to a worker)."""
        ex = self._next_executor()
        return ex.execute(source, timeout)

    def execute_batch(
        self, sources: list[str], timeout: float = 15.0
    ) -> list[JavaExecutionResult]:
        """
        Execute a batch of sources in parallel across the pool.

        This is the hot path for RL rollouts: hundreds of generated Java
        programs are dispatched across N JVMs concurrently.
        """
        futures: list[Future[JavaExecutionResult]] = []
        for i, source in enumerate(sources):
            ex = self._executors[i % self.num_workers]
            futures.append(self._pool.submit(ex.execute, source, timeout))

        results = []
        for f in futures:
            results.append(f.result())
        return results

    def _next_executor(self) -> JavaExecutor:
        with self._counter_lock:
            ex = self._executors[self._counter % self.num_workers]
            self._counter += 1
            return ex
