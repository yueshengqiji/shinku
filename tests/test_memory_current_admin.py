from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from shinku.api.memory_admin import build_memory_admin_router
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


@pytest.fixture
def current_app(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), router=MemoryRouter(),
        clock=lambda: 1790890000, policy=MemoryPolicy(current_understanding_enabled=True))
    service.record_turn(conversation_id="owner", conversation_type="private", message_id="m1", text="记住我不吃香菜。")
    app = FastAPI()
    app.include_router(build_memory_admin_router(service))
    return app, service


def test_current_inspection_is_local_scope_required_and_contains_no_statement(current_app):
    app, service = current_app
    with TestClient(app) as client:
        assert client.get("/memory/current-understanding").status_code == 400
        assert client.post("/memory/current-understanding/rebuild", json={"scope": "private:owner"}).json()["written"] == 0
        assert client.post("/memory/current-understanding/rebuild", json={"scope": "private:owner", "dry_run": False, "confirm": True}).status_code == 200
        result = client.get("/memory/current-understanding", params={"scope": "private:owner"})
        assert result.status_code == 200
        assert "香菜" not in result.text
        assert result.headers["cache-control"] == "no-store"
    with TestClient(app, client=("203.0.113.8", 2000)) as client:
        assert client.get("/memory/current-understanding", params={"scope": "private:owner"}).status_code == 403


def test_rebuild_requires_json_same_origin_and_confirmation(current_app):
    app, _ = current_app
    with TestClient(app) as client:
        path = "/memory/current-understanding/rebuild"
        assert client.post(path, data="scope=private:owner").status_code == 415
        assert client.post(path, json={"scope": "private:owner"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post(path, json={"scope": "private:owner", "dry_run": False}).status_code == 400
        assert client.post(path, json={"scope": "private:owner", "dry_run": "false", "confirm": True}).status_code == 400
        assert client.post(path, json={"scope": "private:owner", "dry_run": False, "confirm": "true"}).status_code == 400


def test_disabled_current_feature_does_not_offer_rebuild(current_app):
    from dataclasses import replace
    app, service = current_app
    service.policy = replace(service.policy, current_understanding_enabled=False)
    with TestClient(app) as client:
        assert client.get("/memory/current-understanding").json()["enabled"] is False
        assert client.post("/memory/current-understanding/rebuild", json={"scope": "private:owner", "dry_run": False, "confirm": True}).status_code == 409


def test_explicit_rebuild_repairs_missing_derived_table_only(current_app):
    app, service = current_app
    with service.store._connect() as conn:
        conn.execute("DROP TABLE memory_current_keys")
    with TestClient(app) as client:
        assert client.get("/memory/current-understanding", params={"scope": "private:owner"}).status_code == 503
        result = client.post("/memory/current-understanding/rebuild", json={"scope": "private:owner", "dry_run": False, "confirm": True, "force": True})
        assert result.status_code == 200
        assert result.json()["written"] == 1
    assert service.current_understanding.read(scope="private:owner", subject="user:owner", query="香菜", now=1790890000)["records"]
