from __future__ import annotations

from pathlib import Path

from scripts.check_shinku_cutover import assess


ROOT = Path(__file__).resolve().parents[1]


def test_d5_preflight_is_read_only_and_requires_all_gates() -> None:
    payloads = {
        "http://old/health": {"status": "ok", "project": "legacy"},
        "http://old/health/ready": {"status": "ready", "service": "legacy"},
        "http://new/health": {"status": "ok", "project": "shinku", "version": "0.1.0"},
        "http://new/health/ready": {"status": "ready", "service": "backend"},
    }

    def fake_fetch(url: str) -> dict[str, object]:
        return payloads[url]

    result = assess(ROOT, old_url="http://old", new_url="http://new", fetcher=fake_fetch)
    assert result["read_only"] is True
    assert result["manual_confirmation_required"] is True
    assert result["ready_for_manual_cutover"] is True


def test_d5_preflight_blocks_when_new_service_is_down() -> None:
    def fake_fetch(url: str) -> dict[str, object]:
        if url.startswith("http://new"):
            raise OSError("offline")
        return {"status": "ready" if url.endswith("/health/ready") else "ok"}

    result = assess(ROOT, old_url="http://old", new_url="http://new", fetcher=fake_fetch)
    assert result["checks"]["old_ready"] is True
    assert result["checks"]["new_ready"] is False
    assert result["ready_for_manual_cutover"] is False
