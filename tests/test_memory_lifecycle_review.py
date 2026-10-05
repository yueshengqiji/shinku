from dataclasses import replace
import threading
import time

import pytest

from shinku.memory import LifecyclePolicy, MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.lifecycle_review import LLMLifecycleReviewer
from shinku.memory.lifecycle import digest
from shinku.api.memory_admin import build_memory_admin_router
from fastapi import FastAPI
from fastapi.testclient import TestClient

T0 = 1_800_000_000
SOURCE = "我喜欢安静的环境和自然的红茶。"
USER = "平时我更爱清静的地方，喝茶也只选红茶。"
FINAL = "这次选清静的茶室，给你点一壶红茶。"


def fixture(tmp_path, reviewer, **changes):
    clock = [T0]
    life = replace(LifecyclePolicy(mode="active", semantic_review_enabled=True), **changes)
    service = MemoryService(store=MemoryStore(tmp_path / "m.sqlite3"), router=MemoryRouter(),
        policy=MemoryPolicy(lifecycle=life), clock=lambda: clock[0], lifecycle_reviewer=reviewer)
    service.record_turn(conversation_id="u", conversation_type="private", message_id="original", text=SOURCE)
    record = service.store.event_by_source_id(source_id="turn:u:original", scope="private:u")
    clock[0] += 86400
    return service, clock, record


def decision(record, kind="remention", **changes):
    return {"record_id": record.record_id, "relation": "same_fact" if kind == "remention" else "operative_use",
        "source_quote": SOURCE, "user_quote": USER, "final_quote": FINAL,
        "same_actor": True, "negated": False, "quoted": False, "hypothetical": False,
        "correction": False, "confidence": .98, **changes}


def enqueue(service, clock, record, kind="remention", text=USER, evidence="next"):
    return service.lifecycle_review.enqueue(scope=record.scope, actor="u", evidence_id=evidence,
        kind=kind, occurred=clock[0], user_text=text, final_text=FINAL, records=(record,))


def test_bound_paraphrase_promotes_not_truth(tmp_path):
    service, clock, record = fixture(tmp_path, lambda _: {"decisions": [decision(record)]})
    assert enqueue(service, clock, record)
    assert service.lifecycle_review.process_one()
    assert service.store.lifecycle.state(record.record_id)["tier"] == 2
    assert service.store.event_by_source_id(source_id=record.source_id, scope=record.scope).confidence == record.confidence
    assert not enqueue(service, clock, record)


@pytest.mark.parametrize("change", [
    {"record_id": "event:invented"}, {"confidence": .89}, {"confidence": float("nan")},
    {"source_quote": "invented source quote"}, {"user_quote": "invented user"},
    {"same_actor": False}, {"negated": True}, {"quoted": True}, {"hypothetical": True},
    {"correction": True}, {"relation": "none"}, {"quoted": None},
])
def test_model_proposal_is_not_authority(tmp_path, change):
    service, clock, record = fixture(tmp_path, lambda _: {"decisions": [decision(record, **change)]})
    assert enqueue(service, clock, record)
    service.lifecycle_review.process_one()
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 0


@pytest.mark.parametrize("text", ["我不再喜欢红茶了", "你是不是记错了？", "他说我喜欢红茶", '“我喜欢红茶”是别人的话', "假如我喜欢红茶会怎么样", "密码是not-real-test-only"])
def test_unsafe_turn_never_calls_model(tmp_path, text):
    service, clock, record = fixture(tmp_path, lambda _: pytest.fail("model must not run"))
    assert not enqueue(service, clock, record, text=text)


def test_timeout_cannot_apply_late_or_spawn_more_requests(tmp_path):
    gate = threading.Event()
    def reviewer(_):
        gate.wait(3)
        return {"decisions": [decision(record)]}
    service, clock, record = fixture(tmp_path, reviewer, review_timeout_seconds=.05)
    assert enqueue(service, clock, record)
    service.lifecycle_review.process_one()
    assert service.lifecycle_review.status()["last_error"] == "review_timeout"
    assert enqueue(service, clock, record, evidence="second")
    assert not service.lifecycle_review.process_one()
    assert service.lifecycle_review.status()["counts"]["calls"] == 1
    gate.set()
    for _ in range(100):
        if not service.lifecycle_review.status()["outstanding_request"]:
            break
        time.sleep(.01)
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 0
    service.lifecycle_review.stop()


@pytest.mark.parametrize("mutation", ["forget", "correction"])
def test_inflight_changed_source_cannot_reinforce(tmp_path, mutation):
    def reviewer(_):
        if mutation == "forget":
            service.store.forget(query="红茶", scope=record.scope, now=clock[0])
        else:
            with service.store._connect() as conn:
                conn.execute("UPDATE memory_events SET text='已经改为喜欢咖啡了' WHERE event_id=?", (record.record_id.removeprefix("event:"),))
        return {"decisions": [decision(record)]}
    service, clock, record = fixture(tmp_path, reviewer)
    assert enqueue(service, clock, record)
    service.lifecycle_review.process_one()
    assert service.store.lifecycle.state(record.record_id).get("reinforce_count", 0) == 0


def test_final_use_requires_real_current_binding(tmp_path):
    service, clock, record = fixture(tmp_path, lambda _: {"decisions": [decision(record, "use")]})
    kwargs = dict(conversation_id="u", conversation_type="private", message_id="answer", requester_id="u", final_text=FINAL)
    assert service.finalize_usage(**kwargs)["accepted"] == 0
    service.bind_usage_context(scope=record.scope, message_id="answer", requester_id="u", records=(record,), user_text=USER)
    assert service.finalize_usage(**kwargs)["semantic_reviews_queued"] == 1
    service.lifecycle_review.process_one()
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 1
    assert service.finalize_usage(**kwargs)["accepted"] == 0


def test_queue_budget_shutdown_and_opt_in(tmp_path):
    service, clock, record = fixture(tmp_path, lambda _: {}, review_queue_size=1, review_daily_calls=1)
    assert enqueue(service, clock, record)
    assert not enqueue(service, clock, record, evidence="overflow")
    service.lifecycle_review.process_one()
    assert enqueue(service, clock, record, evidence="budget")
    service.lifecycle_review.process_one()
    assert service.lifecycle_review.status()["counts"]["daily_budget"] == 1
    service.lifecycle_review.start()
    service.lifecycle_review.stop()
    assert not service.lifecycle_review.status()["worker_running"]
    assert not enqueue(service, clock, record, evidence="stopped")
    assert LifecyclePolicy().semantic_review_enabled is False


def test_review_adapter_uses_existing_aux_without_tools():
    class Runtime:
        def call_aux_json(self, **kwargs):
            assert "tools" not in kwargs and kwargs["fallback"] == {}
            return {"decisions": []}
    assert LLMLifecycleReviewer(Runtime())({}) == {"decisions": []}


def test_secret_not_pinned_by_remember_command(tmp_path):
    service, clock, _ = fixture(tmp_path, None)
    service.record_turn(conversation_id="u", conversation_type="private", message_id="secret", text="记住密码是not-real-test-only")
    with service.store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM memory_facts").fetchone()[0] == 0


def test_shadow_correction_keeps_existing_active_tier(tmp_path):
    service, clock, record = fixture(tmp_path, None)
    service.store.lifecycle.reinforce_remention(record=record, user_text=SOURCE, actor_id="u", evidence_id="r", now=clock[0])
    service.store.lifecycle.policy = replace(service.policy.lifecycle, mode="shadow")
    with service.store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='我现在只喝清水而且喜欢安静的环境。' WHERE event_id=?", (record.record_id.removeprefix("event:"),))
    service.store.lifecycle.enroll(record_id=record.record_id, now=clock[0])
    assert service.store.lifecycle.state(record.record_id)["tier"] == 2
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 0


def test_unrelated_spontaneous_verbatim_echo_is_not_fastpath_use(tmp_path):
    service, clock, record = fixture(tmp_path, None)
    service.bind_usage_context(scope=record.scope, message_id="calc", requester_id="u", records=(record,), user_text="帮我算三加四")
    assert service.finalize_usage(conversation_id="u", conversation_type="private", message_id="calc", requester_id="u", final_text="答案是七。" + SOURCE)["accepted"] == 0
    assert service.store.lifecycle.state(record.record_id)["reinforce_count"] == 0


def test_fact_correction_is_cas_scoped_and_invalidates_views(tmp_path):
    service, clock, _ = fixture(tmp_path, None)
    store = service.store
    store.upsert_fact(conversation_id="u", scope="private:u", fact_key="tea", text="我现在喜欢红茶", source_id="fact-original", now=T0)
    old = next(r for r in store.search(query="红茶", scope="private:u", now=T0) if r.layer == "fact")
    store.record_event(conversation_id="u", scope=old.scope, source_id="derived", text="现在偏好红茶", layer="summary", created_at=T0,
        metadata={"source_record_ids": [old.record_id], "source_content_hashes": {old.record_id: digest(old.text)}})
    args = dict(record_id=old.record_id, scope=old.scope, expected_hash=digest(old.text), replacement="我现在只喝清水", now=clock[0], confirmed=True)
    with pytest.raises(ValueError):
        store.lifecycle.correct_fact(**{**args, "confirmed": False})
    assert not store.lifecycle.correct_fact(**{**args, "scope": "group:foreign"})
    assert not store.lifecycle.correct_fact(**{**args, "expected_hash": "a" * 64})
    assert store.lifecycle.correct_fact(**args)
    assert not store.lifecycle.correct_fact(**args)
    assert store.event_by_source_id(source_id="derived", scope=old.scope) is None
    assert store.lifecycle.state(old.record_id)["content_hash"] == digest(args["replacement"])
    store.upsert_fact(conversation_id="u", scope=old.scope, fact_key="old-backup", text=old.text, now=clock[0])
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM memory_facts WHERE scope=? AND status='active'", (old.scope,)).fetchone()[0] == 1


def test_correction_route_requires_json_confirmation_and_hash(tmp_path):
    service, clock, _ = fixture(tmp_path, None)
    app = FastAPI()
    app.include_router(build_memory_admin_router(service))
    client = TestClient(app)
    assert client.post("/memory/lifecycle/correct-fact", data="x").status_code == 415
    assert client.post("/memory/lifecycle/correct-fact", json={}).status_code == 400
    assert client.get("/memory/lifecycle").json()["semantic_review"]["enabled"]


def test_missing_legacy_fact_author_cannot_be_automatically_reinforced(tmp_path):
    service, clock, _ = fixture(tmp_path, None)
    service.store.upsert_fact(conversation_id="u", scope="private:u", fact_key="tea", text=SOURCE, now=T0)
    fact = next(r for r in service.store.search(query="红茶", scope="private:u", now=T0) if r.layer == "fact")
    assert not enqueue(service, clock, fact)


@pytest.mark.parametrize("with_turn", [True, False])
def test_forget_after_raw_retirement_blocks_backup_resurrection(tmp_path, with_turn):
    life = replace(LifecyclePolicy(mode="active"), l1_seconds=1, l2_seconds=2,
        migration_grace_seconds=1, summary_grace_seconds=1, minimum_idle_turns=1)
    clock = [T0]
    service = MemoryService(store=MemoryStore(tmp_path / "retired.sqlite3"), router=MemoryRouter(),
        clock=lambda: clock[0], policy=MemoryPolicy(lifecycle=life))
    if with_turn:
        service.record_turn(conversation_id="u", conversation_type="private", message_id="original", text=SOURCE)
        source_id = "turn:u:original"
    else:
        source_id = "legacy-import-only"
        service.store.record_event(conversation_id="u", scope="private:u", source_id=source_id,
            text=SOURCE, created_at=T0, metadata={"role": "user", "sender_id": "u"})
    record = service.store.event_by_source_id(source_id=source_id, scope="private:u")
    clock[0] += 86400
    service.store.lifecycle.note_interaction(scope=record.scope, evidence_id="activity", now=clock[0])
    service.store.lifecycle.scan(now=clock[0], scope=record.scope, allow_reclaim=True, approved_scope=record.scope)
    assert service.store.event_by_source_id(source_id=source_id, scope=record.scope) is None
    assert service.store.lifecycle.state(record.record_id)["summary_source_hash"] == digest(SOURCE)
    assert service.store.forget(query="红茶" if with_turn else "", scope=record.scope,
        record_id="" if with_turn else record.record_id, now=clock[0])[0] == 1
    payload = service.store.lifecycle.export_revocations(scope=record.scope)
    restored = MemoryStore(tmp_path / "restored.sqlite3", lifecycle_policy=life)
    restored.record_event(conversation_id="u", scope=record.scope, source_id=source_id, text=SOURCE, created_at=T0)
    restored.lifecycle.import_revocations(payload, approved_scope=record.scope, confirmed=True)
    assert restored.event_by_source_id(source_id=source_id, scope=record.scope) is None


def test_exact_revoke_cannot_touch_foreign_scope_or_match_everything(tmp_path):
    service, clock, record = fixture(tmp_path, None)
    assert service.store.forget(query="", scope="group:foreign", record_id=record.record_id, now=clock[0]) == (0, 0)
    assert service.store.forget(query="", scope=record.scope, record_id="fact:not-present", now=clock[0]) == (0, 0)
    assert service.store.event_by_source_id(source_id=record.source_id, scope=record.scope) is not None


@pytest.mark.parametrize("change", [{"l1_seconds": True}, {"l1_seconds": "7"}, {"review_candidates": 1.5}, {"defer_l1_embeddings": "false"}])
def test_policy_rejects_wrong_types_instead_of_crashing_later(change):
    with pytest.raises(ValueError):
        replace(LifecyclePolicy(), **change)
    with pytest.raises(ValueError):
        LifecyclePolicy.from_environment({"SHINKU_MEMORY_LIFECYCLE_SEMANTIC_REVIEW_ENABLED": "not-a-boolean"})


def test_parallel_review_drains_cannot_start_two_model_calls(tmp_path):
    gate, entered = threading.Event(), threading.Event()
    def reviewer(_):
        entered.set()
        gate.wait(3)
        return {}
    service, clock, record = fixture(tmp_path, reviewer)
    assert enqueue(service, clock, record)
    assert enqueue(service, clock, record, evidence="other-job")
    worker = threading.Thread(target=service.lifecycle_review.process_one)
    worker.start()
    assert entered.wait(1)
    assert not service.lifecycle_review.process_one()
    assert service.lifecycle_review.status()["counts"]["calls"] == 1
    gate.set()
    worker.join(1)
    assert not worker.is_alive()
    service.lifecycle_review.stop()
