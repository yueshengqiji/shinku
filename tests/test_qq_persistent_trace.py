"""Diagnostic persistence is bounded evidence, never executable recovery."""

import json
from types import SimpleNamespace

import pytest

from shinku.qq.trace import TurnTraceStore


def turn(message="m-1", text="private conversation must not be logged"):
    return SimpleNamespace(conversation_type="group", conversation_id="100002",
                           primary_message_id=message, message_ids=(message,), combined_text=text)


def reply():
    return SimpleNamespace(agent_status="failed", sent=False, delivery_status="suppressed",
                           reason="task_contract_error:contract_repair_failed", final_text="private reply")


def test_terminal_trace_and_validation_cause_survive_recreation(tmp_path):
    path = tmp_path / "turn-traces.json"
    store = TurnTraceStore(path=path, clock=lambda: 100.0)
    trace = store.begin(turn())
    store.stage(trace, "task_contract", status="validation_failed", error="invalid_result_checks",
                field_path="requirements[0].checks.min_sources", selected_tools=["memory_search"])
    store.stage(trace, "task_contract_repair", status="failed", error="invalid_contract_repair_response")
    store.finish(trace, reply=reply())

    restored = TurnTraceStore(path=path, clock=lambda: 101.0).snapshot()[0]
    assert restored["trace_id"] == trace
    assert restored["reason"] == "task_contract_error:contract_repair_failed"
    assert restored["stages"][0]["error"] == "invalid_result_checks"
    assert restored["stages"][0]["field_path"] == "requirements[0].checks.min_sources"
    assert restored["stages"][1]["error"] == "invalid_contract_repair_response"
    assert restored["restored"] is True


def test_sensitive_values_are_not_persisted_under_innocent_keys(tmp_path):
    path = tmp_path / "turn-traces.json"
    secret = "UNIQUE_PRIVATE_SECRET_79"
    store = TurnTraceStore(path=path, clock=lambda: 100.0)
    trace = store.begin(turn(text=secret))
    store.stage(trace, "task_contract", status="validation_failed", error="invalid_result_checks",
                contract={"goal": secret, "scope": [secret], "requirements": [
                    {"description": secret, "targets": ["https://site.test/?token=" + secret],
                     "checks": {"contains": secret}}]},
                extra=secret, credential={"Authorization": "Bearer " + secret},
                argument_value=secret, provider_error="HTTP exception Authorization=" + secret,
                prompt=secret, text=secret, original_request=secret, token=secret)
    store.finish(trace, reply=reply())

    persisted = path.read_text(encoding="utf-8")
    assert secret not in persisted
    assert "Bearer" not in persisted
    assert "site.test" not in persisted
    assert "private reply" not in persisted
    assert "invalid_result_checks" in persisted


def test_expired_rows_are_not_restored(tmp_path):
    path = tmp_path / "turn-traces.json"
    clock = [100.0]
    store = TurnTraceStore(path=path, retention_seconds=60, clock=lambda: clock[0])
    trace = store.begin(turn())
    store.finish(trace, reply=reply())
    clock[0] = 161.0
    assert TurnTraceStore(path=path, retention_seconds=60, clock=lambda: clock[0]).snapshot() == []


def test_record_stage_and_disk_byte_caps_are_enforced(tmp_path):
    path = tmp_path / "turn-traces.json"
    clock = [100.0]
    store = TurnTraceStore(path=path, max_items=2, max_stages=4, max_bytes=8192, clock=lambda: clock[0])
    for index in range(5):
        clock[0] += 1
        trace = store.begin(turn(f"m-{index}"))
        for stage in range(20):
            store.stage(trace, "llm_usage", calls=1, total_tokens=stage,
                        nested={"status": "valid", "input_tokens": 42})
        store.finish(trace, reply=reply())
    assert len(store.snapshot(limit=100)) == 2
    assert all(len(item["stages"]) <= 4 for item in store.snapshot())
    assert path.stat().st_size <= 8192
    restored = TurnTraceStore(path=path, max_items=2, max_stages=4, max_bytes=8192,
                             clock=lambda: clock[0]).snapshot()
    assert len(restored) == 2
    assert [row["message_id"] for row in restored] == ["m-4", "m-3"]


@pytest.mark.parametrize("kind", ["directory", "invalid_json", "oversized"])
def test_persistence_faults_do_not_break_ordinary_trace(kind, tmp_path):
    path = tmp_path / "turn-traces.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid_json":
        path.write_text("{broken", encoding="utf-8")
    else:
        path.write_bytes(b" " * 9000)
    store = TurnTraceStore(path=path, max_bytes=8192, clock=lambda: 100.0)
    trace = store.begin(turn())
    store.stage(trace, "task_contract", status="rejected", error="contract_missing")
    store.finish(trace, reply=reply())
    assert store.snapshot()[0]["reason"] == "task_contract_error:contract_repair_failed"


def test_restored_diagnostics_cannot_restore_approval_or_tool_arguments(tmp_path):
    path = tmp_path / "turn-traces.json"
    store = TurnTraceStore(path=path, clock=lambda: 100.0)
    trace = store.begin(turn())
    store.stage(trace, "task_link", task_id="task-1", mode="new",
                approved=True, arguments={"command": "must not be executable"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["approved"] = True
    payload["items"][0]["context"] = {"agent_permission_mode": "ultra"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = TurnTraceStore(path=path, clock=lambda: 101.0).snapshot()[0]
    assert restored["status"] != "completed"
    assert "context" not in restored
    assert "approved" not in restored
    assert "arguments" not in json.dumps(restored)
    assert not any(stage.get("approved") for stage in restored["stages"])


def test_snapshot_is_deeply_detached_from_store():
    store = TurnTraceStore(clock=lambda: 100.0)
    trace = store.begin(turn())
    store.stage(trace, "llm_usage", nested={"total_tokens": 7})
    first = store.snapshot()[0]
    first["message_ids"].append("injected-message")
    first["stages"][0]["nested"]["total_tokens"] = 999
    second = store.snapshot()[0]
    assert second["message_ids"] == ["m-1"]
    assert second["stages"][0]["nested"]["total_tokens"] == 7
