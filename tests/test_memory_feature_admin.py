from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from shinku.api.memory_admin import build_memory_admin_router
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


FLAGS = ("search_keys_enabled", "current_understanding_enabled",
         "evidence_package_enabled", "history_needs_enabled")


def make_app(tmp_path, policy):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"),
                            router=MemoryRouter(), policy=policy)
    app = FastAPI()
    app.include_router(build_memory_admin_router(service))
    return app, service


@pytest.mark.parametrize("enabled", [False, True])
def test_features_reports_loaded_cde_with_b_off_without_reading_data(tmp_path, monkeypatch, enabled):
    policy = MemoryPolicy(current_understanding_enabled=enabled,
                          evidence_package_enabled=enabled, history_needs_enabled=enabled)
    app, service = make_app(tmp_path, policy)

    def forbidden(*args, **kwargs):
        raise AssertionError("status must not access memory data")

    monkeypatch.setattr(service.store, "_connect", forbidden)
    with TestClient(app) as client:
        response = client.get("/memory/features")
    expected = dict.fromkeys(FLAGS, enabled)
    expected["search_keys_enabled"] = False
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"ok": True, "configured": expected,
                               "initialized": expected, "effective": expected,
                               "context_projection":{"configured":False,"initialized":False,
                                   "semantic_configured":False,"policy_version":policy.projection.version,
                                   "last_diagnostics":{},"index_status":{}}}


def test_features_distinguishes_requested_and_available_components(tmp_path):
    app, service = make_app(tmp_path, MemoryPolicy(current_understanding_enabled=True))
    service.current_understanding = None
    with TestClient(app) as client:
        result = client.get("/memory/features").json()
    assert result["configured"]["current_understanding_enabled"] is True
    assert result["initialized"]["current_understanding_enabled"] is False
    assert result["effective"]["current_understanding_enabled"] is False


def test_features_disabled_switch_is_not_effective_with_existing_component(tmp_path):
    app, service = make_app(tmp_path, MemoryPolicy(current_understanding_enabled=True))
    service.policy = replace(service.policy, current_understanding_enabled=False)
    with TestClient(app) as client:
        result = client.get("/memory/features").json()
    assert result["configured"]["current_understanding_enabled"] is False
    assert result["initialized"]["current_understanding_enabled"] is True
    assert result["effective"]["current_understanding_enabled"] is False


def test_features_rejects_nonlocal_requests(tmp_path):
    app, _ = make_app(tmp_path, MemoryPolicy())
    with TestClient(app, client=("203.0.113.8", 12345)) as client:
        response = client.get("/memory/features")
    assert response.status_code == 403
    assert response.json() == {"ok": False, "reason": "local_request_required"}
