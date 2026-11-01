"""真实采集入口到重建后的追踪：只保留固定诊断，不扩大来源或执行授权。"""
import json
from types import SimpleNamespace

import pytest

from shinku.memory.context_sources import ContextSourceCheck
from shinku.qq.dialogue_grounding import capture_sources
from shinku.qq.model_stream import call_reply_model
from shinku.qq.trace import TurnTraceStore


def traced(tmp_path):
    path = tmp_path / "trace.json"
    store = TurnTraceStore(path=path, clock=lambda: 100)
    trace = store.begin(SimpleNamespace(conversation_type="private", conversation_id="synthetic",
        primary_message_id="input-1", combined_text="private input must not persist"))
    return path, store, trace


def restored(path):
    return TurnTraceStore(path=path, clock=lambda: 101).snapshot()[0]["stages"]


def test_actual_stream_collector_survives_recreation_without_another_model_call(tmp_path):
    path, store, trace = traced(tmp_path)
    calls = []

    def stream(**options):
        calls.append(options)
        yield "private unfinished output"
        return SimpleNamespace(parsed={"reply": "synthetic"}, first_chunk_ms=12.5,
            transport_chunks=3, rescue_succeeded=False, error="")

    model = SimpleNamespace(runtime=SimpleNamespace(stream_chat_json=stream),
        stream_enabled=lambda: True, effects=SimpleNamespace(trace=store.stage))
    assert call_reply_model(model, trace_id=trace, fallback={}) == {"reply": "synthetic"}
    assert len(calls) == 1
    stage = next(s for s in restored(path) if s["name"] == "chat_stream_complete")
    assert {k: stage[k] for k in ("first_chunk_ms", "chunks", "transport_chunks", "rescued")} == {
        "first_chunk_ms": 12.5, "chunks": 1, "transport_chunks": 3, "rescued": False}
    assert "private unfinished output" not in path.read_text()


def test_actual_source_collector_keeps_both_phases_and_fixed_rejections(tmp_path):
    path, store, trace = traced(tmp_path)
    checks = iter([
        ContextSourceCheck(True, snapshot=(("private-id", "private-scope", "private-version"),),
            read_paths=("unified_projection", "recent_owner_sources")),
        ContextSourceCheck(False, reason="source_revoked", read_paths=("unified_projection",),
            rejected=2, reason_counts=(("source_revoked", 1), ("source_version_changed", 1))),
    ])
    model = SimpleNamespace(source_projection=lambda _: next(checks), user_images=lambda _: [],
        effects=SimpleNamespace(trace=store.stage))
    context = {}
    assert capture_sources(model, context, phase="before", trace_id=trace)
    assert capture_sources(model, context, phase="after", trace_id=trace) is None
    before, after = restored(path)
    assert (before["phase"], before["state"], before["source_count"], before["rejected"]) == (
        "before", "valid", 1, 0)
    assert before["read_paths"] == ["unified_projection", "recent_owner_sources"]
    assert after["phase"] == "after" and after["state"] == "invalid"
    assert after["reason_counts"] == {"source_revoked": 1, "source_version_changed": 1}
    assert after["rejected"] == 2 and after["absence_means"] == "unknown"
    text = path.read_text()
    assert all(s not in text for s in ("private-id", "private-scope", "private-version", "private input"))


@pytest.mark.parametrize("bad", [True, -1, 10**400, float("nan"), float("inf"), "secret_alphanumeric_79"])
def test_new_numeric_fields_reject_invalid_values(tmp_path, bad):
    path, store, trace = traced(tmp_path)
    store.stage(trace, "chat_stream_complete", first_chunk_ms=bad, chunks=bad,
        transport_chunks=bad, rescued="secret_alphanumeric_79")
    store.stage(trace, "context_sources", rejected=bad, reason_counts={"source_revoked": bad})
    stream, source = restored(path)
    assert not any(k in stream for k in ("first_chunk_ms", "chunks", "transport_chunks", "rescued"))
    assert "rejected" not in source and source["reason_counts"] == {}


def test_arbitrary_nested_labels_and_unknown_event_cannot_persist_new_fields(tmp_path):
    path, store, trace = traced(tmp_path)
    secret = "secret_alphanumeric_79"
    fields = dict(state=secret, read_paths=[secret, "recent_owner_sources"], rejected=1,
        reason_counts={secret: 3, "source_revoked": 2}, absence_means=secret,
        checks={"before": {"text": secret}}, source_ids=[secret])
    store.stage(trace, "context_sources", **fields)
    store.stage(trace, "other_event", **fields, first_chunk_ms=1, chunks=1)
    source, other = restored(path)
    assert source["read_paths"] == ["recent_owner_sources"]
    assert source["reason_counts"] == {"source_revoked": 2}
    assert "state" not in source and "absence_means" not in source
    assert not any(k in other for k in (*fields, "first_chunk_ms", "chunks"))
    assert secret not in path.read_text()


def test_missing_old_metrics_stay_missing_and_unobserved_first_chunk_stays_none(tmp_path):
    path, store, trace = traced(tmp_path)
    store.stage(trace, "chat_stream_complete", status="interrupted")
    store.stage(trace, "chat_stream_complete", first_chunk_ms=None, chunks=0, transport_chunks=0)
    old, new = restored(path)
    assert "first_chunk_ms" not in old and "chunks" not in old
    assert new["first_chunk_ms"] is None and new["chunks"] == new["transport_chunks"] == 0
    assert json.loads(path.read_text())["version"] == 1
