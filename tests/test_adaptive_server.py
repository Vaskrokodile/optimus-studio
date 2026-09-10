"""HTTP-level tests for the new adaptive-orchestrator and persistent-world
server endpoints added in server.py.

Uses FastAPI's TestClient against the real ``create_app()`` so the routes are
exercised end-to-end (no real model is loaded).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ILOPTIMUS_HOME", os.path.join(os.environ.get("TEMP", "/tmp"), "ilopt_test"))


@pytest.fixture(autouse=True)
def isolated_app_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path / "iloptimus-home"))


@pytest.fixture
def client():
    """FastAPI TestClient. NOTE: ``create_app()`` instantiates ``RsiLoopStore()``
    at construction time, which loads all existing loop JSON files into an
    in-memory dict. To make a loop visible to the server's routes, create it
    BEFORE this fixture is invoked (or use the ``app_factory`` helper)."""
    from fastapi.testclient import TestClient
    from iloptimus.server import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture
def app_factory():
    """Factory that creates a fresh TestClient after loops have been seeded
    on disk. Use this when a test needs to create loops via RsiLoopStore and
    then hit server endpoints that read them."""
    from fastapi.testclient import TestClient
    from iloptimus.server import create_app
    def _make():
        app = create_app()
        return TestClient(app)
    return _make


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestAdaptiveRoutesRegistered:
    def test_world_routes_present(self, client):
        paths = {r.path for r in client.app.routes if hasattr(r, "path")}
        assert "/api/worlds" in paths
        assert "/api/worlds/{world_id}" in paths
        assert "/api/worlds/{world_id}/derive_task" in paths
        assert "/api/worlds/{world_id}/advance" in paths

    def test_adaptive_loop_routes_present(self, client):
        paths = {r.path for r in client.app.routes if hasattr(r, "path")}
        assert "/api/rsi/loops/{loop_id}/adaptive/state" in paths
        assert "/api/rsi/loops/{loop_id}/adaptive/step" in paths


# ---------------------------------------------------------------------------
# Persistent world endpoints
# ---------------------------------------------------------------------------


class TestWorldEndpoints:
    def test_create_and_get_world(self, client):
        # Create a world.
        r = client.post("/api/worlds", json={"world_id": "srv-world"})
        assert r.status_code == 200
        world = r.json()
        assert world["world_id"] == "srv-world"
        assert world["tick"] == 0
        assert "servers" in world

        # GET it back.
        r = client.get("/api/worlds/srv-world")
        assert r.status_code == 200
        assert r.json()["world_id"] == "srv-world"

    def test_get_missing_world_404(self, client):
        r = client.get("/api/worlds/no-such-world")
        assert r.status_code == 404

    def test_list_worlds(self, client):
        client.post("/api/worlds", json={"world_id": "w1"})
        client.post("/api/worlds", json={"world_id": "w2"})
        r = client.get("/api/worlds")
        assert r.status_code == 200
        names = set(r.json()["worlds"])
        assert "w1" in names
        assert "w2" in names

    def test_derive_task_from_world(self, client):
        client.post("/api/worlds", json={"world_id": "task-world"})
        r = client.post("/api/worlds/task-world/derive_task", json={"focus_skill": "debugging"})
        assert r.status_code == 200
        task = r.json()
        assert task["source"] == "persistent_world"
        assert task["task_id"].startswith("world-task-world-")
        assert "simulator" in task["metadata"]

    def test_advance_world(self, client):
        client.post("/api/worlds", json={"world_id": "adv-world"})
        r = client.post("/api/worlds/adv-world/advance", json={"actions": []})
        assert r.status_code == 200
        assert r.json()["tick"] == 1
        # Second advance -> tick 2.
        r = client.post("/api/worlds/adv-world/advance", json={"actions": []})
        assert r.json()["tick"] == 2


# ---------------------------------------------------------------------------
# Adaptive orchestrator endpoints
# ---------------------------------------------------------------------------


class TestAdaptiveLoopEndpoints:
    def _create_adaptive_loop(self):
        from iloptimus.core.rsi_loops import RsiLoopStore
        store = RsiLoopStore()
        loop = store.create({
            "name": "Adaptive test",
            "kind": "adaptive",
            "model_id": "stub-model",
            "objective": "test adaptive loop",
            "time_budget_minutes": 30,
            "max_iterations": 3,
        })
        return loop.id

    def _create_coding_loop(self):
        from iloptimus.core.rsi_loops import RsiLoopStore
        store = RsiLoopStore()
        loop = store.create({
            "name": "Coding loop", "kind": "coding", "model_id": "m",
            "objective": "x", "time_budget_minutes": 30, "max_iterations": 3,
        })
        return loop.id

    def test_get_adaptive_state_404_for_non_adaptive_loop(self, app_factory):
        coding_id = self._create_coding_loop()
        client = app_factory()  # create app AFTER loop is on disk
        r = client.get(f"/api/rsi/loops/{coding_id}/adaptive/state")
        assert r.status_code == 404

    def test_get_adaptive_state_for_new_loop(self, app_factory):
        loop_id = self._create_adaptive_loop()
        client = app_factory()  # create app AFTER loop is on disk
        r0 = client.get(f"/api/rsi/loops/{loop_id}")
        assert r0.status_code == 200, f"loop not found: {r0.status_code} {r0.text[:200]}"
        assert r0.json()["kind"] == "adaptive"
        r = client.get(f"/api/rsi/loops/{loop_id}/adaptive/state")
        assert r.status_code == 200
        state = r.json()
        assert state["loop_id"] == loop_id
        assert state["iteration"] == 0

    def test_adaptive_step_advances_iteration(self, app_factory):
        loop_id = self._create_adaptive_loop()
        client = app_factory()
        r = client.post(f"/api/rsi/loops/{loop_id}/adaptive/step", json={
            "rollouts_per_task": 2,
            "world_task_prob": 1.0,
        })
        assert r.status_code == 200
        signal = r.json()
        assert signal["iteration"] == 1
        assert "task" in signal
        assert signal["task"]["source"] in (
            "taskset", "mutated", "generated", "persistent_world", "composed"
        )
        r = client.get(f"/api/rsi/loops/{loop_id}/adaptive/state")
        assert r.json()["iteration"] == 1
