from __future__ import annotations

from dataclasses import replace
import sqlite3
import threading

import pytest

from shinku.memory import LifecyclePolicy, MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.lifecycle import digest


T0 = 1_800_000_000


@pytest.fixture
def setup(tmp_path):
    def create(**changes):
        policy = replace(LifecyclePolicy(mode="active"), **changes)
        clock = [T0]
        store = MemoryStore(tmp_path / f"memory-{len(list(tmp_path.iterdir()))}.sqlite3", lifecycle_policy=policy)
        store.lifecycle.wall_time = lambda: clock[0]
        return store, store.lifecycle, clock
    return create


def event(store, *, text="我喜欢安静的环境和自然的红茶。", source="source-1", scope="private:owner", sender="owner", role="user", created=T0):
    assert store.record_event(conversation_id="owner", scope=scope, text=text, source_id=source, created_at=created, expires_at=created + 60, metadata={"sender_id": sender, "role": role})
    return store.event_by_source_id(source_id=source, scope=scope)


def reinforce(life, record, stamp, evidence="evidence", **kwargs):
    return life.reinforce_remention(record=record, user_text=record.text, actor_id="owner", evidence_id=evidence, now=stamp, session_id="session", **kwargs)


def age_scope(life, clock, *, days=120, turns=20, scope="private:owner"):
    for index in range(turns):
        clock[0] += days * 86400 // turns
        life.note_interaction(scope=scope, evidence_id=f"activity-{clock[0]}-{index}", now=clock[0])


def test_policy_default_shadow_and_explicit_invalid_fail_closed():
    assert MemoryPolicy().lifecycle.mode == "shadow"
    assert MemoryPolicy.from_environment({"SHINKU_MEMORY_LIFECYCLE_MODE": "active"}).lifecycle.mode == "active"
    for values in ({"mode": "typo"}, {"idle_rate": float("nan")}, {"batch_size": 0}, {"l3_count": 2}, {"raw_days": 9}):
        with pytest.raises(ValueError):
            LifecyclePolicy(**values)
    with pytest.raises(ValueError):
        MemoryPolicy.from_environment({"SHINKU_MEMORY_LIFECYCLE_IDLE_RATE": "not-a-number"})


def test_initial_raw_l1_and_separate_last_used(setup):
    store, life, clock = setup()
    record = event(store)
    initial = life.state(record.record_id)
    assert (initial["tier"], initial["representation"]) == (1, "raw")
    store.search(query="红茶", scope=record.scope, now=T0 + 20)
    after = life.state(record.record_id)
    assert after["reinforce_count"] == 0
    assert after["last_reinforced_at"] == 0


def test_effective_repeat_promotes_l2_without_changing_truth_or_scope(setup):
    store, life, clock = setup()
    record = event(store)
    result = reinforce(life, record, T0 + 100)
    assert result["accepted"] and result["applied"]
    assert life.state(record.record_id)["tier"] == 2
    current = store.event_by_source_id(source_id=record.source_id, scope=record.scope)
    assert (current.scope, current.confidence, current.importance) == (record.scope, record.confidence, record.importance)


def test_duplicate_webhook_and_burst_do_not_promote_l3(setup):
    store, life, clock = setup()
    record = event(store)
    assert reinforce(life, record, T0 + 1, evidence="request-1")["accepted"]
    assert reinforce(life, record, T0 + 1, evidence="request-1")["reason"] == "duplicate_evidence"
    for index in range(30):
        assert not reinforce(life, record, T0 + 2 + index, evidence=f"request-{index + 2}")["accepted"]
    assert life.state(record.record_id)["reinforce_count"] == 1
    assert life.state(record.record_id)["tier"] == 2


def test_cross_day_frequency_protects_raw_then_long_span_promotes_l3(setup):
    store, life, clock = setup()
    record = event(store)
    for index in range(3):
        assert reinforce(life, record, T0 + index * 86400, evidence=f"e-{index}")["accepted"]
    assert life.state(record.record_id)["raw_protected_until"] > T0 + 2 * 86400
    assert life.state(record.record_id)["tier"] == 2
    for index, day in enumerate((7, 10, 15), 3):
        assert reinforce(life, record, T0 + day * 86400, evidence=f"e-{index}")["accepted"]
    assert life.state(record.record_id)["tier"] == 3
    age_scope(life, clock, days=1000)
    result = life.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert result["skipped"]["protected_l3"] == 1
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == record.text


def test_shadow_proposes_but_preserves_tier_text_expiry(setup):
    store, life, clock = setup(mode="shadow")
    record = event(store)
    result = reinforce(life, record, T0)
    assert result["proposed_tier"] == 2 and not result["applied"]
    assert life.state(record.record_id)["tier"] == 1
    assert store.expire(now=T0 + 60) == 1
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope) is None
    with pytest.raises(ValueError):
        life.scan(now=T0 + 60, scope=record.scope, allow_reclaim=True, approved_scope=record.scope)


def test_active_managed_lifecycle_not_legacy_ttl_but_fact_validity_still_expires(setup):
    store, life, clock = setup()
    record = event(store)
    store.upsert_fact(conversation_id="owner", scope=record.scope, fact_key="temporary", text="临时工作地址", now=T0, expires_at=T0 + 60)
    assert store.expire(now=T0 + 60) == 1
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope) is not None
    assert not any(item.layer == "fact" for item in store.search(query="地址", scope=record.scope, now=T0 + 60))


def test_one_long_vacation_not_enough_for_eviction(setup):
    store, life, clock = setup()
    record = event(store)
    result = life.scan(now=T0 + 1000 * 86400, scope=record.scope)
    assert result["actions"] == []
    assert result["skipped"]["not_cold"] == 1


def test_activity_other_scope_does_not_age_owner_turn_count(setup):
    store, life, clock = setup()
    record = event(store)
    age_scope(life, clock, days=1000, scope="group:noise")
    assert life.scan(now=clock[0], scope=record.scope)["actions"] == []


def test_wall_clock_rollback_and_restart_do_not_rewind_or_double_count(setup):
    store, life, clock = setup()
    record = event(store)
    life.note_interaction(scope=record.scope, evidence_id="request-1", now=T0)
    life.scan(now=T0 + 100)
    with store._connect() as conn:
        before = dict(conn.execute("SELECT * FROM memory_lifecycle_scopes WHERE scope=?", (record.scope,)).fetchone())
    life.scan(now=T0 - 100)
    restarted = MemoryStore(store.path, lifecycle_policy=life.policy)
    restarted.lifecycle.note_interaction(scope=record.scope, evidence_id="request-1", now=T0 + 100)
    with store._connect() as conn:
        after = dict(conn.execute("SELECT * FROM memory_lifecycle_scopes WHERE scope=?", (record.scope,)).fetchone())
    assert after["effective_seconds"] == before["effective_seconds"]
    assert after["turn_count"] == before["turn_count"]


def test_summary_then_grace_then_exact_scope_reclamation(setup):
    store, life, clock = setup(summary_max_chars=25, migration_grace_seconds=1, l1_seconds=1, l2_seconds=2, summary_grace_seconds=2, minimum_idle_turns=1)
    raw = "我喜欢安静的环境和自然的红茶。" + "这是当天的普通天气闲聊。" * 10
    record = event(store, text=raw)
    age_scope(life, clock, days=3, turns=3)
    first = life.scan(now=clock[0], scope=record.scope)
    assert first["actions"][0]["action"] == "summarize"
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == raw
    assert life.scan(now=clock[0] + 1, scope=record.scope)["actions"] == []
    preview = life.scan(now=clock[0] + 3, scope=record.scope)
    assert not preview["actions"][0]["applied"]
    for kwargs in ({"allow_reclaim": True}, {"scope": record.scope, "approved_scope": "group:other", "allow_reclaim": True}):
        with pytest.raises(ValueError):
            life.scan(now=clock[0] + 3, **kwargs)
    result = life.scan(now=clock[0] + 3, scope=record.scope, approved_scope=record.scope, allow_reclaim=True)
    current = store.event_by_source_id(source_id=record.source_id, scope=record.scope)
    assert len(current.text) < len(raw)
    assert current.metadata["raw_detail_unavailable"] is True
    assert life.state(record.record_id)["representation"] == "summary"
    assert life.state(record.record_id)["summary_source_hash"] == digest(raw)
    assert result["logical_bytes_reclaimed"] == len(raw.encode()) - len(current.text.encode())
    assert not result["file_compacted"] and result["archive_bytes_reclaimed"] == 0


def test_new_reinforcement_cancels_prepared_summary_and_reclamation(setup):
    store, life, clock = setup(summary_max_chars=25, migration_grace_seconds=1, l1_seconds=1, l2_seconds=2, summary_grace_seconds=2, minimum_idle_turns=1)
    record = event(store, text="我喜欢安静的环境和自然的红茶。" + "天气闲聊无关紧要。" * 10)
    age_scope(life, clock, days=3, turns=3)
    assert life.scan(now=clock[0], scope=record.scope)["actions"][0]["action"] == "summarize"
    assert reinforce(life, record, clock[0])["accepted"]
    assert life.state(record.record_id)["summary_at"] == 0
    result = life.scan(now=clock[0] + 10, scope=record.scope, approved_scope=record.scope, allow_reclaim=True)
    assert result["logical_bytes_reclaimed"] == 0
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == record.text


def test_cold_l2_demotes_first_and_does_not_immediately_reclaim(setup):
    store, life, clock = setup(l1_seconds=1, l2_seconds=2, migration_grace_seconds=1, minimum_idle_turns=1)
    record = event(store)
    reinforce(life, record, T0)
    age_scope(life, clock, days=3, turns=3)
    result = life.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert result["actions"][0]["action"] == "demote"
    assert result["logical_bytes_reclaimed"] == 0
    assert life.state(record.record_id)["tier"] == 1


def test_short_indivisible_l1_can_retire_after_second_grace(setup):
    store, life, clock = setup(l1_seconds=1, l2_seconds=2, migration_grace_seconds=1, summary_grace_seconds=2, minimum_idle_turns=1)
    record = event(store)
    age_scope(life, clock, days=3, turns=3)
    result = life.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert result["actions"][0]["action"] == "expire_raw"
    assert result["logical_bytes_reclaimed"] == len(record.text.encode())
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope) is None


def test_lossy_summary_does_not_reclaim_precision_without_grace(setup):
    from shinku.memory.lifecycle import MemoryLifecycle
    assert not MemoryLifecycle.summary_valid("周六见面。限制是不喝甜茶，时间为14:30。", "周六见面。")
    assert not MemoryLifecycle.summary_valid("他说只有周六见面才合适。", "周六见面。")
    assert not MemoryLifecycle.summary_valid("周六见面。", "周日见面。")


def test_false_mentions_foreign_actor_assistant_and_forged_snapshot_rejected(setup):
    store, life, clock = setup()
    record = event(store)
    for text in ("我不再喜欢安静的环境和自然的红茶。", "你是不是记错了我喜欢红茶？", "他说我喜欢安静的环境和自然的红茶。"):
        assert not life.reinforce_remention(record=record, user_text=text, actor_id="owner", evidence_id=text, now=T0)["accepted"]
    assert not life.reinforce_remention(record=record, user_text=record.text, actor_id="other", evidence_id="other", now=T0)["accepted"]
    assistant = event(store, source="assistant", role="assistant")
    forged = replace(assistant, metadata={"role": "user", "sender_id": "owner"})
    assert not reinforce(life, forged, T0)["accepted"]
    foreign = replace(record, scope="group:foreign")
    assert not reinforce(life, foreign, T0)["accepted"]


def test_final_use_requires_quote_current_hash_and_no_supplied_id_magic(setup):
    store, life, clock = setup()
    record = event(store)
    for quote, answer in ((record.record_id, record.record_id), ("不存在的长句引用内容。", "不存在的长句引用内容。"), (record.text, "没采用这条记录")):
        assert not life.reinforce_used(record=record, quote=quote, final_text=answer, actor_id="owner", evidence_id="q", now=T0)["accepted"]
    assert life.reinforce_used(record=record, quote=record.text, final_text="回忆是：" + record.text, actor_id="owner", evidence_id="q", now=T0)["accepted"]
    store.forget(query="红茶", scope=record.scope, now=T0 + 1)
    assert not life.reinforce_used(record=record, quote=record.text, final_text=record.text, actor_id="owner", evidence_id="q-2", now=T0 + 2)["accepted"]


def test_content_correction_resets_old_strength_and_old_snapshot_rejected(setup):
    store, life, clock = setup()
    record = event(store)
    reinforce(life, record, T0)
    rid = record.record_id.removeprefix("event:")
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text=? WHERE event_id=?", ("我现在不再喜欢甜红茶。", rid))
    assert not reinforce(life, record, T0 + 86400)["accepted"]
    life.scan(now=T0 + 86400)
    assert life.state(record.record_id)["reinforce_count"] == 0
    assert life.state(record.record_id)["tier"] == 1


def test_explicit_forget_revokes_l3_and_provenance_derivatives_and_import(setup):
    store, life, clock = setup()
    record = event(store)
    life.set_pin(record_id=record.record_id, scope=record.scope, pinned=True, confirmed=True, now=T0)
    derived = event(store, text="这是压缩后的摘要，未保留原始查询关键词。", source="derived")
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (store._json({"source_record_ids": [record.record_id]}), derived.record_id.removeprefix("event:")))
    assert store.forget(query="红茶", scope=record.scope, purge=True, now=T0 + 1) == (0, 2)
    assert life.state(record.record_id) == {}
    assert not store.record_event(conversation_id="owner", scope=record.scope, text=record.text, source_id="different-import-id", created_at=T0 + 2)
    assert store.record_event(conversation_id="other", scope="group:other", text=record.text, source_id="other-scope-import", created_at=T0 + 2)


def test_evidence_prune_does_not_allow_old_replay(setup):
    store, life, clock = setup()
    record = event(store)
    assert reinforce(life, record, T0, evidence="original")["accepted"]
    stamp = T0 + life.policy.evidence_retention_seconds + 1
    life.scan(now=stamp)
    assert life.status()["retained_evidence"] == 0
    assert not reinforce(life, record, stamp, evidence="original", occurred_at=T0)["accepted"]


def test_single_request_concurrent_duplicate_across_two_store_instances(setup):
    store, life, clock = setup()
    record = event(store)
    other = MemoryStore(store.path, lifecycle_policy=life.policy)
    accepted = []
    def run(target):
        accepted.append(reinforce(target, record, T0, evidence="same-request")["accepted"])
    threads = [threading.Thread(target=run, args=(target,)) for target in (life, other.lifecycle)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert accepted.count(True) == 1
    assert life.state(record.record_id)["reinforce_count"] == 1


def test_failure_inside_reclamation_rolls_back_everything(setup):
    store, life, clock = setup(summary_max_chars=25, migration_grace_seconds=1, l1_seconds=1, l2_seconds=2, summary_grace_seconds=2, minimum_idle_turns=1)
    record = event(store, text="我喜欢安静的环境和自然的红茶。" + "无关紧要的天气闲聊。" * 10)
    age_scope(life, clock, days=3, turns=3)
    life.scan(now=clock[0], scope=record.scope)
    with store._connect() as conn:
        conn.execute("CREATE TRIGGER fail_reclaim BEFORE UPDATE OF content_hash ON memory_lifecycle BEGIN SELECT RAISE(ABORT,'injected'); END;")
    with pytest.raises(sqlite3.IntegrityError):
        life.scan(now=clock[0] + 3, scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == record.text
    assert life.state(record.record_id)["representation"] == "raw"


def test_policy_version_migration_and_legacy_records_start_new_grace(setup):
    store, life, clock = setup()
    record = event(store, created=T0 - 1000 * 86400)
    assert life.state(record.record_id)["enrolled_at"] == T0
    age_scope(life, clock, days=1000, turns=20)
    life.policy = replace(life.policy, version="ebbinghaus-v2")
    result = life.scan(now=clock[0], scope=record.scope)
    assert result["skipped"]["policy_migration"] == 1
    assert life.state(record.record_id)["enrolled_at"] == clock[0]


def test_pin_requires_confirmation_and_scope_and_unpin_has_new_grace(setup):
    store, life, clock = setup()
    record = event(store)
    with pytest.raises(ValueError):
        life.set_pin(record_id=record.record_id, scope=record.scope, pinned=True, confirmed=False, now=T0)
    assert not life.set_pin(record_id=record.record_id, scope="group:wrong", pinned=True, confirmed=True, now=T0)
    assert life.set_pin(record_id=record.record_id, scope=record.scope, pinned=True, confirmed=True, now=T0)
    assert life.set_pin(record_id=record.record_id, scope=record.scope, pinned=False, confirmed=True, now=T0 + 1)
    assert life.state(record.record_id)["tier"] == 2


def test_budget_prevents_new_automatic_raw_protection_not_old_pins(setup):
    store, life, clock = setup(auto_raw_budget_bytes=1)
    record = event(store)
    for index in range(7):
        reinforce(life, record, T0 + index * 3 * 86400, evidence=f"e-{index}")
    state = life.state(record.record_id)
    assert state["tier"] == 2 and state["raw_protected_until"] == 0
    assert life.set_pin(record_id=record.record_id, scope=record.scope, pinned=True, confirmed=True, now=T0 + 30 * 86400)
    assert life.state(record.record_id)["tier"] == 3


def test_service_repeat_counts_once_not_assistant_self_repeat(tmp_path):
    clock = [T0]
    service = MemoryService(store=MemoryStore(tmp_path / "m.sqlite3"), router=MemoryRouter(), clock=lambda: clock[0], policy=MemoryPolicy(lifecycle=LifecyclePolicy(mode="active")))
    kwargs = {"conversation_id": "owner", "conversation_type": "private", "text": "我喜欢安静的环境和自然的红茶。"}
    service.record_turn(**kwargs, message_id="m-1")
    record = service.store.event_by_source_id(source_id="turn:owner:m-1", scope="private:owner")
    clock[0] += 86400
    service.record_turn(**kwargs, message_id="m-2")
    service.record_turn(**kwargs, message_id="m-2")
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 1
    service.record_turn(**kwargs, message_id="m-3", role="assistant")
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 1


def test_service_final_use_only_current_bound_read(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "m.sqlite3"), router=MemoryRouter(), clock=lambda: T0, policy=MemoryPolicy(lifecycle=LifecyclePolicy(mode="active")))
    service.record_turn(conversation_id="owner", conversation_type="private", message_id="m-1", text="我喜欢安静的环境和自然的红茶。")
    context = service.prepare_turn(conversation_id="owner", conversation_type="private", user_message="你还记得我喜欢什么吗？", message_meta={"message_id": "m-2", "sender_id": "owner"})
    record = context.records[0]
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 0
    result = service.finalize_usage(conversation_id="owner", conversation_type="private", message_id="m-2", requester_id="owner", final_text=record.text)
    assert result["accepted"] == 1
    assert service.finalize_usage(conversation_id="owner", conversation_type="private", message_id="m-2", requester_id="owner", final_text=record.text)["accepted"] == 0


def test_reset_drops_lifecycle_but_keeps_compact_reimport_revocations(setup):
    store, life, clock = setup()
    record = event(store)
    reinforce(life, record, T0)
    store.reset_all()
    assert life.status()["retained_evidence"] == 0
    assert life.state(record.record_id) == {}
    assert not store.record_event(conversation_id="owner", scope=record.scope, text=record.text, source_id=record.source_id)
