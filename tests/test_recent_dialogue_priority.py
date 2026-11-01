"""Recent corrections survive archival retrieval without acquiring authority."""
import json

import pytest

from test_context_projection import NOW, engine, put, req


def pair(store, mid, user, assistant, *, age=60, scope="private:owner", sender="owner"):
    row = put(store, mid, user, age=age, scope=scope, sender=sender)
    reply = put(store, mid + ":assistant", assistant, age=age - 1, scope=scope,
                sender="bot", role="assistant", requester_id=sender,
                delivery_status="sent", delivery_message_ids=["out-" + mid],
                delivery_content_complete=True)
    return row, reply


@pytest.mark.parametrize("max_chars", [3500, 10000])
def test_recent_correction_and_reply_survive_old_photo_seeds(tmp_path, max_chars):
    store, projector = engine(tmp_path, max_chars=max_chars, record_chars=1000, source_limit=8)
    correction, answer = pair(store, "correction", "我不是开玩笑，照片确实是我拍的。", "我刚才却说你在编。")
    old = [put(store, str(i), "以前我给你拍的照片" + str(i), age=8000 + i) for i in range(12)]
    view = projector.project(req("当时给你拍的，这下信了吧", history_mode="retrieve",
        seed_handles=tuple((r.record_id, r.scope, "") for r in old)))
    ids = [r["record_id"] for r in view.records if r["text"]]
    assert ids[:2] == [correction.record_id, answer.record_id]
    assert correction.text in view.rendered and answer.text in view.rendered
    assert view.records[0]["context_priority"] == "recent_dialogue"
    assert view.diagnostics["recent_dialogue_injected"] >= 2
    assert len(view.rendered) <= max_chars


def test_three_recent_pairs_include_serious_restatement_before_image(tmp_path):
    store, projector = engine(tmp_path)
    pairs = [pair(store, str(i), text, "上一条回应", age=500 - i * 100)
             for i, text in enumerate(("我说的是真的，不是临时编的。", "之前我说过这件事。", "这是当时的照片。"))]
    old = put(store, "old", "早先类似照片", age=9000)
    view = projector.project(req("这下信了吧", history_mode="retrieve",
        seed_handles=((old.record_id, old.scope, ""),)))
    assert all(user.text in view.rendered for user, _ in pairs)
    assert view.diagnostics["recent_dialogue_injected"] == 6


def test_protection_does_not_pull_later_queued_messages_into_current_turn(tmp_path):
    store, projector = engine(tmp_path)
    previous, _ = pair(store, "previous", "前一轮澄清", "前一轮回复", age=30)
    put(store, "current", "本轮用户原话", age=20)
    later, _ = pair(store, "later", "下一轮等待处理的纠正", "下一轮回复", age=10)
    view = projector.project(req("本轮用户原话", history_mode="retrieve", message_id="current"))
    protected = [r["record_id"] for r in view.records if r.get("context_priority") == "recent_dialogue"]
    assert previous.record_id in protected
    assert later.record_id not in protected
    assert later.text not in view.rendered


def test_exact_calendar_recall_does_not_prioritize_today(tmp_path):
    store, projector = engine(tmp_path)
    latest, _ = pair(store, "today", "今天的澄清", "今天的回复")
    old = put(store, "yesterday", "昨天的原话", age=86400)
    view = projector.project(req("昨天的原话", history_mode="retrieve", start_at=NOW - 90000,
                                end_at=NOW - 80000, seed_handles=((old.record_id, old.scope, ""),)))
    assert latest.record_id not in [r["record_id"] for r in view.records]
    assert old.text in view.rendered


def test_protected_dialogue_preserves_source_scope_sender_and_revocation(tmp_path):
    store, projector = engine(tmp_path)
    row, _ = pair(store, "own", "本人的澄清", "已送达回应", scope="group:g")
    other, _ = pair(store, "other", "他人的澄清", "他人的回应", scope="group:g", sender="other", age=10)
    hidden, _ = pair(store, "private", "私聊秘密", "私聊回应")
    request = req(scope="group:g", allowed_scopes=("group:g",))
    view = projector.project(request)
    protected = {r["record_id"] for r in view.records if r.get("context_priority") == "recent_dialogue"}
    assert row.record_id in protected and other.record_id not in protected
    assert hidden.text not in view.rendered
    store.forget(scope=row.scope, record_id=row.record_id, query="", now=NOW)
    rebuilt = projector.project(request)
    assert row.text not in rebuilt.rendered


def test_assistant_history_is_proof_of_speech_not_of_its_claim(tmp_path):
    store, projector = engine(tmp_path)
    user, assistant = pair(store, "old", "我是在认真澄清。", "你明明一直在编故事。")
    view = projector.project(req("我说的确实是真的"))
    rows = {r["record_id"]: r for r in view.records}
    assert rows[user.record_id]["evidence_kind"] == "user_statement"
    assert rows[assistant.record_id]["evidence_kind"] == "assistant_utterance"
    assert "只证明说过" in view.rendered
    assert "已核实事实" not in json.dumps(rows[assistant.record_id], ensure_ascii=False)


def test_one_source_budget_keeps_user_correction_before_assistant(tmp_path):
    store, projector = engine(tmp_path, source_limit=1)
    user, _ = pair(store, "old", "我不是那个意思。", "你就是在逗我。")
    view = projector.project(req("请认真听我的澄清。"))
    assert [r["record_id"] for r in view.records] == [user.record_id]


def test_group_correction_recovers_own_pairs_after_noise_without_global_pinning(tmp_path):
    store, projector = engine(tmp_path)
    user, reply = pair(store, "own", "那张照片是我拍的。", "以前的猜测。", scope="group:g")
    for i in range(24):
        put(store, "noise-" + str(i), "群友插话" + str(i), scope="group:g", sender="other", age=40 - i)
    ordinary = projector.project(req("晚饭聊什么", scope="group:g", allowed_scopes=("group:g",)))
    assert user.text not in ordinary.rendered
    corrected = projector.project(req("当时给你拍的，这下信了吧", scope="group:g", allowed_scopes=("group:g",)))
    assert user.text in corrected.rendered and reply.text in corrected.rendered


def test_group_reply_to_foreign_requester_is_not_a_protected_pair(tmp_path):
    store, projector = engine(tmp_path)
    user, reply = pair(store, "own", "本人的澄清", "回给他人的话", scope="group:g")
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET metadata_json=json_set(metadata_json,'$.requester_id','other') WHERE event_id=?",
                     (reply.record_id.split(":", 1)[1],))
    view = projector.project(req("继续澄清", scope="group:g", allowed_scopes=("group:g",)))
    assert user.record_id not in {r["record_id"] for r in view.records if r["context_priority"] == "recent_dialogue"}
