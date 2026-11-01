"""Synthetic private-chat fixtures, independent from real conversations and voice examples."""
from reply_review_fixture import dialogue_review_result
from types import SimpleNamespace
import socket

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.chat_flow import ChatFlowController
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.agent.loop import AgentState

NOW = 1_800_000_000
FIRST = "你要喝茶吗"
ANSWER = "我想要一杯温茶，不用加糖。"
SECOND = "我把茶放在旁边的小桌上了"
NEW_TOPIC = "换个话题，我明天打算把项目的备份目录整理一下，再研究新的部署方案"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("private-context fixtures cannot use the network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def memory(tmp_path):
    ticks = [NOW]
    service = MemoryService(store=MemoryStore(tmp_path / "fixture.sqlite3"),
        router=MemoryRouter(), policy=MemoryPolicy(master_user_ids=("owner",), passive_observe_private=True),
        clock=lambda: ticks[0])
    return service, ticks


def turn(text=SECOND, mid="second", *, sender="owner", conversation="owner", kind="private"):
    message = IncomingMessage.from_event({"post_type": "message", "message_type": kind,
        "user_id": sender, "self_id": "bot", "group_id": conversation if kind == "group" else None,
        "message_id": mid, "message": [{"type": "text", "data": {"text": text}}]})
    return QQTurn(conversation_id=conversation, conversation_type=kind,
        primary_message_id=mid, message_ids=(mid,), combined_text=text, messages=(message,))


def remember_pair(memory, *, age=23, conversation="owner", kind="private", reply=ANSWER,
                  assistant_meta=None, passive=False):
    service, ticks = memory
    ticks[0] = NOW - age - 10
    if passive:
        service.observe_message(turn(FIRST, "first", conversation=conversation, kind=kind).messages[0])
    service.record_turn(conversation_id=conversation, conversation_type=kind,
        message_id="first", text=FIRST, metadata={"sender_id": "owner", "self_id": "bot", "message_id": "first"})
    ticks[0] = NOW - age
    service.record_turn(conversation_id=conversation, conversation_type=kind,
        message_id="first:assistant", text=reply, role="assistant",
        source_id=f"turn:{conversation}:first:assistant",
        metadata={"sender_id": "bot", "self_id": "bot", "requester_id": "owner",
            "message_id": "first", "delivery_status": "sent",
            "delivery_message_ids": ["receipt"], "delivery_content_complete": True,
            **(assistant_meta or {})})
    ticks[0] = NOW


def context(memory, *, text=SECOND, recent=True, conversation="owner", kind="private", meta=None):
    return memory[0].prepare_turn(conversation_id=conversation, conversation_type=kind,
        user_message=text, recent_context=FIRST if recent else "",
        message_meta={"sender_id": "owner", "self_id": "bot",
            "message_id": "second", "topic_continuation": recent, **(meta or {})})


def dialogue(result):
    return result.as_dict().get("recent_dialogue", "")


def test_private_pair_is_fact_context_not_voice_history(memory):
    remember_pair(memory)
    result = context(memory)
    assert result.decision.history_mode == "recent"
    assert FIRST in dialogue(result) and ANSWER in dialogue(result)
    assert ANSWER not in result.rendered
    assert all(row.metadata.get("role") != "assistant" for row in result.records)
    assert result.audit["private_dialogue_count"] == 1


def test_passive_observation_without_role_is_a_valid_user_half(memory):
    remember_pair(memory, passive=True)
    user = memory[0].store.event_by_source_id(source_id="turn:owner:first", scope="private:owner")
    assert user.layer == "observation" and "role" not in user.metadata
    assert ANSWER in dialogue(context(memory))


def test_none_history_mode_keeps_short_dialogue_on_topic_switch(memory):
    remember_pair(memory)
    result = context(memory, text=NEW_TOPIC, recent=False)
    assert result.decision.history_mode == "none"
    assert not result.records and not result.rendered
    assert ANSWER in dialogue(result)


@pytest.mark.parametrize("meta", [
    {"delivery_status": "dry_run"}, {"delivery_status": "failed"},
    {"delivery_status": "rejected"}, {"delivery_message_ids": []},
    {"delivery_message_ids": [""]}, {"delivery_message_ids": "receipt"},
    {"requester_id": "stranger"}, {"sender_id": "otherbot"}, {"self_id": "otherbot"},
])
def test_unverified_or_other_party_reply_is_excluded(memory, meta):
    remember_pair(memory, assistant_meta=meta)
    assert dialogue(context(memory)) == ""


@pytest.mark.parametrize("age", [121, 900, -1])
def test_expired_or_future_reply_is_excluded(memory, age):
    remember_pair(memory, age=age)
    assert dialogue(context(memory)) == ""


def test_partial_delivery_only_includes_delivered_text(memory):
    remember_pair(memory, reply="第一句。\n第三句。", assistant_meta={
        "delivery_status": "partial", "delivery_content_complete": False,
        "delivery_message_ids": ["out-1", "out-3"]})
    value = dialogue(context(memory))
    assert "第一句" in value and "第三句" in value and "第二句" not in value
    assert "部分发送" in value


@pytest.mark.parametrize("change", [
    {"conversation": "stranger"}, {"kind": "group"},
    {"meta": {"sender_id": "stranger"}}, {"meta": {"self_id": "otherbot"}},
    {"meta": {"self_id": ""}},
])
def test_scope_and_identity_are_exact(memory, change):
    remember_pair(memory)
    assert dialogue(context(memory, **change)) == ""


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_forgotten_half_of_pair_cannot_be_restored(memory, role):
    remember_pair(memory)
    source = "turn:owner:first" + (":assistant" if role == "assistant" else "")
    with memory[0].store._lock, memory[0].store._connect() as conn:
        conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id=?", (source,))
    assert dialogue(context(memory)) == ""


def test_context_is_bounded_and_marks_excerpts(memory):
    remember_pair(memory, reply="确实发送的很长回复。" * 300)
    value = dialogue(context(memory))
    assert value and len(value) < 2100 and "节选" in value


def flow_pair(*, ids=("receipt",)):
    ticks = [100.0]
    flow = ChatFlowController(clock=lambda: ticks[0])
    flow.finish_turn(turn(FIRST, "first"), expanded=False, final_text=ANSWER, reply_message_ids=ids)
    ticks[0] += 23
    return flow, ticks


def test_short_private_switch_keeps_previous_actual_reply():
    flow, _ = flow_pair()
    assert ANSWER in flow.begin_turn(turn()).previous_exchange


def test_private_new_topic_remains_fresh_but_has_recent_fact_context():
    flow, _ = flow_pair()
    decision = flow.begin_turn(turn(NEW_TOPIC))
    assert not decision.same_topic and decision.stage == "fresh"
    assert ANSWER in decision.previous_exchange


@pytest.mark.parametrize("change", ["expired", "sender", "conversation", "group", "no_receipt", "future"])
def test_private_flow_exception_is_short_and_isolated(change):
    flow, ticks = flow_pair(ids=() if change == "no_receipt" else ("receipt",))
    current = turn()
    if change == "expired":
        ticks[0] = 221
    elif change == "future":
        ticks[0] = 99
    elif change == "sender":
        current = turn(sender="stranger")
    elif change == "conversation":
        current = turn(conversation="stranger")
    elif change == "group":
        current = turn(kind="group")
    assert not flow.begin_turn(current).previous_exchange


def test_recreated_bridge_restores_reply_from_database(memory):
    remember_pair(memory)
    calls = []
    runtime = SimpleNamespace(call_chat_json=lambda **kw: calls.append(kw) or dialogue_review_result(kw) or {"speech": "先把杯子放稳。"})
    subject = QQAgentBridge(runtime=runtime, system_prompt="fixture", memory_service=memory[0], sender=None)
    subject.handle_turn(turn())
    assert len(calls) == 1
    assert ANSWER in calls[0]["user_prompt"] and SECOND in calls[0]["user_prompt"]
    assert "语气范例" in "\n".join(calls[0]["system_extra_blocks"])


@pytest.mark.parametrize("passive", [False, True])
def test_live_flow_and_durable_context_are_injected_only_once(memory, passive):
    calls, sent = [], []
    replies = iter((ANSWER, "先把杯子放稳。"))
    runtime = SimpleNamespace(call_chat_json=lambda **kw: calls.append(kw) or dialogue_review_result(kw) or {"speech": next(replies)})
    subject = QQAgentBridge(runtime=runtime, system_prompt="fixture", memory_service=memory[0],
        sender=lambda m: sent.append(m) or DeliveryResult(True, "sent", (f"receipt-{len(sent)}",)))
    if passive:
        memory[0].observe_message(turn(FIRST, "first").messages[0])
    subject.handle_turn(turn(FIRST, "first"))
    subject.handle_turn(turn())
    assert calls[1]["user_prompt"].count(ANSWER) == 1
    assert context(memory).audit["private_dialogue_count"] == 1


def test_recreated_bridge_after_observation_and_real_delivery_boundary(memory):
    captured, sent = [], []
    memory[0].observe_message(turn(FIRST, "first").messages[0])
    initial = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: dialogue_review_result(kw) or {"speech": ANSWER}),
        system_prompt="fixture", memory_service=memory[0],
        sender=lambda m: sent.append(m) or DeliveryResult(True, "sent", (f"receipt-{len(sent)}",)))
    initial.handle_turn(turn(FIRST, "first"))
    recreated = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: captured.append(kw) or dialogue_review_result(kw) or {"speech": "unused"}),
        system_prompt="fixture", memory_service=memory[0], sender=None)
    recreated.handle_turn(turn())
    assert captured[0]["user_prompt"].count(ANSWER) == 1


def test_flow_does_not_revive_forgotten_source_at_prompt_boundary(memory):
    remember_pair(memory)
    subject = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "unused"}),
        system_prompt="fixture", memory_service=memory[0])
    flow, _ = flow_pair()
    with memory[0].store._lock, memory[0].store._connect() as conn:
        conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='turn:owner:first:assistant'")
    parts = subject._build_prompt(AgentState(transcript=[{"role": "user", "content": SECOND}], context={
        "qq_turn": turn(), "chat_flow": flow.begin_turn(turn()).as_dict(),
        "memory_context": context(memory).as_dict()}), include_agent_control=False)
    assert ANSWER not in parts.user_prompt


@pytest.mark.parametrize("meta", [{"raw_detail_unavailable": True}, {"record_type": "summary"}])
def test_lost_or_summary_only_raw_reply_is_not_restored(memory, meta):
    remember_pair(memory, assistant_meta=meta)
    assert not dialogue(context(memory))


def test_new_image_does_not_restore_unrelated_previous_pair(memory):
    remember_pair(memory)
    subject = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "unused"}),
        system_prompt="fixture", memory_service=memory[0])
    parts = subject._build_prompt(AgentState(transcript=[{"role": "user", "content": SECOND}], context={
        "qq_turn": turn(), "chat_flow": {"reason": "different_visual_source", "previous_exchange": ""},
        "memory_context": context(memory).as_dict()}), include_agent_control=False)
    assert ANSWER not in parts.user_prompt
