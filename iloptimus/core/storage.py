"""Durable per-user storage for models, environments, and training runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def app_home() -> Path:
    configured = os.environ.get("ILOPTIMUS_HOME")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".iloptimus"


def models_dir() -> Path:
    return app_home() / "models"


def adapters_dir() -> Path:
    return app_home() / "adapters"


def environments_dir() -> Path:
    return app_home() / "environments"


def runs_dir() -> Path:
    return app_home() / "runs"


def learning_dir() -> Path:
    return app_home() / "learning"


def worlds_dir() -> Path:
    """Directory holding Persistent World state (see persistent_world.py)."""
    return app_home() / "worlds"


def orchestrator_dir() -> Path:
    """Directory holding AdaptiveOrchestrator state (see adaptive_orchestrator.py)."""
    return app_home() / "orchestrator"


def profiles_dir() -> Path:
    """Directory holding per-model capability / frontier profiles."""
    return app_home() / "profiles"


def ensure_app_dirs() -> Path:
    root = app_home()
    for path in (
        root,
        models_dir(),
        adapters_dir(),
        environments_dir(),
        runs_dir(),
        learning_dir(),
        worlds_dir(),
        orchestrator_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        # On Windows, os.replace() can fail with PermissionError if another
        # process has the target file open for reading. Retry a few times
        # with a short delay — the lock is transient.
        import sys as _sys
        max_retries = 5 if _sys.platform == "win32" else 1
        for attempt in range(max_retries):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 >= max_retries:
                    raise
                import time as _time
                _time.sleep(0.1 * (attempt + 1))
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
