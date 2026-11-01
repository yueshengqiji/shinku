from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import math
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from shinku.api.memory_admin import build_memory_admin_router
from shinku.memory import LifecyclePolicy, MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.lifecycle import MemoryLifecycle, digest
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn

T0 = 1_800_000_000
TEXT = "我喜欢安静的环境和自然的红茶。"


def service_for(path, *, mode="active", defer=False, provider=None, **changes):
    clock = [T0]
    life_policy = replace(LifecyclePolicy(mode=mode, defer_l1_embeddings=defer), **changes)
    service = MemoryService(store=MemoryStore(path, embedding_provider=provider, semantic_enabled=bool(provider)), router=MemoryRouter(), clock=lambda: clock[0], policy=MemoryPolicy(lifecycle=life_policy))
    service.record_turn(conversation_id="owner", conversation_type="private", message_id="original", text=TEXT)
    return service, clock, service.store.event_by_source_id(source_id="turn:owner:original", scope="private:owner")


class Provider:
    name = "fixture"
    version = "1"
    dimension = 2
    def __init__(self):
        self.callback = None
    def collection_key(self):
        return "fixture_1_2"
    def embed_texts(self, texts):
        if self.callback is not None:
            callback, self.callback = self.callback, None
            callback()
        return [[1.0, 0.0] if any(term in text for term in ("安静", "宁静")) else [0.0, 1.0] for text in texts]
    def embed_text(self, text):
        return self.embed_texts([text])[0]


def test_deferred_l1_has_ephemeral_paraphrase_recall_but_no_persisted_vectors(tmp_path):
    provider = Provider()
    service, clock, record = service_for(tmp_path / "m.sqlite3", defer=True, provider=provider)
    store = service.store
    assert store.backfill_embeddings() == 0
    matches = store.search(query="宁静的地方", scope=record.scope, now=T0)
    assert matches[0].record_id == record.record_id
    assert matches[0].metadata["semantic_cache"] == "ephemeral_l1"
    assert store.semantic_status()["vectors"] == 0
    assert len(store._l1_semantic_cache) == 1
    clock[0] += 86400
    service.record_turn(conversation_id="owner", conversation_type="private", message_id="repeat", text=TEXT)
    assert store.lifecycle.state(record.record_id)["tier"] == 2
    assert store.backfill_embeddings() == 1
    before = store.semantic_status()["vectors"]
    clock[0] += 86400
    service.record_turn(conversation_id="owner", conversation_type="private", message_id="repeat-2", text=TEXT)
    store.backfill_embeddings()
    assert store.semantic_status()["vectors"] == before


def test_deferred_setting_does_not_disable_unmanaged_summary_index(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3", defer=True, provider=Provider())
    service.store.record_event(conversation_id="owner", scope=record.scope, text="安静的地方", source_id="summary", layer="summary", created_at=T0)
    assert service.store.backfill_embeddings() == 1


@pytest.mark.parametrize("mutation", ["forget", "correct", "delete"])
def test_embedding_snapshot_cannot_commit_after_source_mutates(tmp_path, mutation):
    provider = Provider()
    service, clock, record = service_for(tmp_path / "m.sqlite3", provider=provider)
    store = service.store
    def callback():
        if mutation == "forget":
            store.forget(query="红茶", scope=record.scope, now=T0)
        else:
            with store._connect() as conn:
                if mutation == "correct":
                    conn.execute("UPDATE memory_events SET text='现在的内容已经更正' WHERE event_id=?", (record.record_id.removeprefix("event:"),))
                else:
                    conn.execute("DELETE FROM memory_events WHERE event_id=?", (record.record_id.removeprefix("event:"),))
    provider.callback = callback
    assert store.backfill_embeddings() == 0
    assert store.semantic_status()["vectors"] == 0


def test_ephemeral_cache_is_cleared_by_forget(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3", defer=True, provider=Provider())
    store = service.store
    store.search(query="宁静的地方", scope=record.scope, now=T0)
    assert store._l1_semantic_cache
    store.forget(query="红茶", scope=record.scope, now=T0)
    assert not store._l1_semantic_cache
    assert store.search(query="宁静的地方", scope=record.scope, now=T0) == []


def test_preview_rolls_back_every_sidecar_change(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    before = service.store.lifecycle.state(record.record_id)
    result = service.store.lifecycle.preview(now=T0 + 500 * 86400, scope=record.scope)
    assert result["preview"]
    assert service.store.lifecycle.state(record.record_id) == before
    with service.store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM memory_lifecycle_audit").fetchone()[0] == 0


def test_exponential_strength_uses_time_constant_not_half_life():
    assert MemoryLifecycle.strength_ratio(effective_age=3 * 86400 * math.log(2), time_constant=3 * 86400, count=0) == pytest.approx(0.5)
    assert MemoryLifecycle.strength_ratio(effective_age=3 * 86400, time_constant=3 * 86400, count=0) == pytest.approx(math.exp(-1))
    assert MemoryLifecycle.strength_ratio(effective_age=5, time_constant=3, count=10**9) == MemoryLifecycle.strength_ratio(effective_age=5, time_constant=3, count=3)
    with pytest.raises(ValueError):
        MemoryLifecycle.strength_ratio(effective_age=float("nan"), time_constant=1, count=0)


def test_backup_restore_reapplies_post_backup_forget_and_derivative_links(tmp_path):
    service, clock, record = service_for(tmp_path / "source.sqlite3")
    store = service.store
    store.record_event(conversation_id="owner", scope=record.scope, text="独立的派生摘要。", source_id="derivative", layer="summary", created_at=T0, metadata={"source_record_ids": [record.record_id], "source_content_hashes": {record.record_id: digest(record.text)}})
    backup = tmp_path / "old-backup.sqlite3"
    with closing(sqlite3.connect(store.path)) as source, closing(sqlite3.connect(backup)) as target:
        source.backup(target)
    store.forget(query="红茶", scope=record.scope, purge=True, now=T0)
    payload = store.lifecycle.export_revocations(scope=record.scope)
    assert all("text" not in row for row in payload["revocations"])
    restored = MemoryStore(backup)
    with pytest.raises(ValueError):
        restored.lifecycle.import_revocations(payload, approved_scope="group:other", confirmed=True)
    result = restored.lifecycle.import_revocations(payload, approved_scope=record.scope, confirmed=True)
    assert result["records_disabled"] == 2
    assert restored.search(query="红茶", scope=record.scope, now=T0) == []
    assert restored.event_by_source_id(source_id="derivative", scope=record.scope) is None
    assert not restored.lifecycle.import_revocations(payload, approved_scope=record.scope, confirmed=True)["records_disabled"]


def test_revocation_import_rejects_malformed_rows_atomically(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    payload = {"version": 1, "scope": record.scope, "revocations": [{"scope": record.scope, "source_hash": digest("x"), "content_hash": digest(record.text), "revoked_at": T0}, {"scope": "group:other", "source_hash": digest("y"), "content_hash": digest(record.text), "revoked_at": T0}]}
    with pytest.raises(ValueError):
        service.store.lifecycle.import_revocations(payload, approved_scope=record.scope, confirmed=True)
    assert service.store.lifecycle.status()["revocations"] == 0
    assert service.store.event_by_source_id(source_id=record.source_id, scope=record.scope)


def test_revoking_empty_source_fact_does_not_block_unrelated_empty_source_fact(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    store = service.store
    store.upsert_fact(conversation_id="owner", scope=record.scope, fact_key="tea", text="我喜欢红茶", now=T0)
    store.forget(query="红茶", scope=record.scope, purge=True, now=T0)
    store.upsert_fact(conversation_id="owner", scope=record.scope, fact_key="cat", text="我喜欢猫", now=T0)
    assert any(item.layer == "fact" and item.text == "我喜欢猫" for item in store.search(query="喜欢猫", scope=record.scope, now=T0))


def test_derived_snapshot_rejected_if_parent_forgotten_or_corrected(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    store = service.store
    metadata = {"source_record_ids": [record.record_id], "source_content_hashes": {record.record_id: digest(record.text)}}
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='已经更正的内容' WHERE event_id=?", (record.record_id.removeprefix("event:"),))
    assert not store.record_event(conversation_id="owner", scope=record.scope, text="旧摘要", source_id="stale-summary", layer="summary", created_at=T0, metadata=metadata)
    store.forget(query="更正", scope=record.scope, now=T0)
    assert not store.record_event(conversation_id="owner", scope=record.scope, text="旧摘要", source_id="stale-summary-2", layer="summary", created_at=T0, metadata=metadata)


def test_new_forget_logs_do_not_duplicate_query_plaintext(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    service.store.forget(query="红茶", scope=record.scope, now=T0)
    with service.store._connect() as conn:
        query = conn.execute("SELECT query FROM memory_forget_log").fetchone()[0]
    assert query.startswith("sha256:") and "红茶" not in query


def admin_client(service, **kwargs):
    app = FastAPI()
    app.include_router(build_memory_admin_router(service))
    return TestClient(app, **kwargs)


def test_admin_is_local_only_json_only_and_same_origin(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    remote = admin_client(service, client=("203.0.113.9", 9000))
    assert remote.get("/memory/lifecycle").status_code == 403
    assert remote.post("/memory/lifecycle/scan", json={"scope": record.scope, "confirm": True}).status_code == 403
    local = admin_client(service)
    assert local.post("/memory/lifecycle/preview", content="scope=private:owner").status_code == 415
    assert local.post("/memory/lifecycle/preview", json={"scope": record.scope}, headers={"origin": "https://malicious.invalid"}).status_code == 403
    assert local.post("/memory/lifecycle/preview", json={"scope": record.scope}, headers={"origin": "http://testserver"}).status_code == 200


def test_admin_reclamation_and_pin_require_exact_confirmation(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    local = admin_client(service)
    assert local.post("/memory/lifecycle/scan", json={"scope": record.scope}).status_code == 400
    assert local.post("/memory/lifecycle/scan", json={"scope": record.scope, "confirm": True, "allow_reclaim": True}).status_code == 400
    assert local.post("/memory/lifecycle/protect", json={"scope": "group:other", "record_id": record.record_id, "pinned": True, "confirm": True}).status_code == 404
    assert local.post("/memory/lifecycle/protect", json={"scope": record.scope, "record_id": record.record_id, "pinned": True, "confirm": True}).status_code == 200
    assert service.store.lifecycle.state(record.record_id)["pinned"] == 1
    status = local.get("/memory/lifecycle", params={"scope": record.scope}).json()
    assert status["mode"] == "active"
    assert TEXT not in str(status)


def test_admin_shadow_cannot_physically_reclaim(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3", mode="shadow")
    local = admin_client(service)
    result = local.post("/memory/lifecycle/scan", json={"scope": record.scope, "confirm": True, "allow_reclaim": True, "approved_scope": record.scope, "reclamation_confirmation": "RECLAIM_MEMORY_INDEX_RAW"})
    assert result.status_code == 400
    assert service.store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == TEXT


def turn():
    message = IncomingMessage.from_event({"post_type": "message", "message_type": "private", "user_id": "owner", "message_id": "recall", "message": [{"type": "text", "data": {"text": "你还记得我喜欢什么吗？"}}]})
    return QQTurn(conversation_id="owner", conversation_type="private", primary_message_id="recall", message_ids=("recall",), combined_text=message.text, messages=(message,))


@pytest.mark.parametrize("delivery", ["sent", "failed", "dry_run"])
def test_bridge_only_delivered_answer_counts_as_use(tmp_path, delivery):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    incoming = turn()
    service.prepare_turn(conversation_id="owner", conversation_type="private", user_message=incoming.combined_text, message_meta={"sender_id": "owner", "message_id": "recall"})
    sender = None if delivery == "dry_run" else lambda message: DeliveryResult(delivery == "sent", delivery)
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kwargs: {"speech": "unused"}), system_prompt="fixture", memory_service=service, sender=sender)
    bridge._deliver_text(incoming, TEXT, agent_status="completed", track_chat_flow=False)
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == int(delivery == "sent")


def test_false_final_use_on_correction_does_not_count(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    service.bind_usage_context(scope=record.scope, message_id="q", requester_id="owner", records=[record], user_text="你是不是记错了？")
    assert service.finalize_usage(conversation_id="owner", conversation_type="private", message_id="q", requester_id="owner", final_text=TEXT)["accepted"] == 0


def test_readonly_baseline_and_replay_do_not_depend_on_production(tmp_path):
    from scripts.gray_memory_lifecycle import readonly_baseline, replay
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    before = service.store.lifecycle.state(record.record_id)
    report = readonly_baseline(service.store.path)
    assert report["read_only_sql"] and not report["physical_cleanup_executed"]
    assert report["logical_payload_bytes"]["memory_events"] == len(TEXT.encode())
    assert service.store.lifecycle.state(record.record_id) == before
    result = replay()
    assert result["all_passed"] and result["total"] == 11
    assert result["production_mutations"] == 0


def test_unsafe_summary_keeps_only_original_even_with_scope_reclamation_approval(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3", l1_seconds=1, l2_seconds=2, minimum_idle_turns=1, migration_grace_seconds=1, summary_grace_seconds=1, summary_max_chars=25)
    raw = TEXT + "约定日期是2026年10月3日，时间为14:30，不能改动。" * 8
    service.store.record_event(conversation_id="owner", scope=record.scope, source_id="precision", text=raw, created_at=T0, metadata={"role": "user", "sender_id": "owner"})
    for index in range(3):
        clock[0] += 86400
        service.store.lifecycle.note_interaction(scope=record.scope, evidence_id=f"activity-{index}", now=clock[0])
    result = service.store.lifecycle.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert result["skipped"]["summary_validation_failed_keep_raw"] == 1
    assert service.store.event_by_source_id(source_id="precision", scope=record.scope).text == raw


def test_threshold_change_fingerprint_gives_grace_without_rewinding_memory_age(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3")
    before = service.store.lifecycle.state(record.record_id)
    clock[0] += 100 * 86400
    service.store.lifecycle.policy = replace(service.store.lifecycle.policy, l1_seconds=12345)
    result = service.store.lifecycle.scan(now=clock[0], scope=record.scope)
    after = service.store.lifecycle.state(record.record_id)
    assert result["skipped"]["policy_migration"] == 1
    assert after["policy_version"] != before["policy_version"]
    assert (after["anchor_effective"], after["anchor_turns"]) == (before["anchor_effective"], before["anchor_turns"])


def test_explicit_forget_clears_plaintext_provider_cache_and_pending_context(tmp_path):
    from shinku.providers.embedding import CachedEmbeddingProvider
    provider = CachedEmbeddingProvider(Provider())
    service, clock, record = service_for(tmp_path / "m.sqlite3", provider=provider)
    service.store.search(query="红茶", scope=record.scope, now=T0)
    service.bind_usage_context(scope=record.scope, requester_id="owner", message_id="q", records=[record], user_text="回忆红茶")
    assert provider._stored and service._usage_contexts
    service.store.forget(query="红茶", scope=record.scope, now=T0)
    assert not provider._stored and not service._usage_contexts


def test_foreign_quote_or_at_owner_never_grants_write_to_owner_library(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "m.sqlite3"), router=MemoryRouter(), clock=lambda: T0, policy=MemoryPolicy(master_user_ids=("owner",), lifecycle=LifecyclePolicy(mode="active")))
    for index in range(3):
        service.record_turn(conversation_id="group", conversation_type="group", message_id=f"foreign-{index}", text=TEXT, metadata={"sender_id": "foreign", "shinku_involved": True, "at_targets": ["owner"], "quoted_sender_id": "owner"})
    assert not service.store.search(query="红茶", scope="private:master", now=T0)
    assert service.store.search(query="红茶", scope="group:group", now=T0)


def test_deferred_l1_vectors_live_with_source_and_forget_still_removes_them(tmp_path):
    service, clock, record = service_for(tmp_path / "m.sqlite3", provider=Provider())
    store = service.store
    assert store.backfill_embeddings() == 1
    store.configure_lifecycle(replace(store.lifecycle.policy, defer_l1_embeddings=True))
    # 延期只影响前台生成；即使过宽限且批准扫描，也不能独立删掉仍保留原文的索引。
    store.lifecycle.scan(now=T0, scope=record.scope)
    assert store.semantic_status()["vectors"] == 1
    clock[0] += store.lifecycle.policy.migration_grace_seconds + 1
    preview = store.lifecycle.scan(now=clock[0], scope=record.scope)
    assert not any(row["action"] == "reclaim_l1_vectors" for row in preview["actions"])
    assert store.semantic_status()["vectors"] == 1
    result = store.lifecycle.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert result["logical_vector_bytes_reclaimed"] == 0
    assert store.semantic_status()["vectors"] == 1
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == TEXT
    store.forget(query="", scope=record.scope, record_id=record.record_id, now=clock[0])
    assert store.semantic_status()["vectors"] == 0
