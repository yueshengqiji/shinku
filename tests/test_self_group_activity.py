from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.self_activity import is_self_group_recall, activity_answer
from shinku.memory.tools import MemorySearchTool
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn

NOW = 1791016033
QUESTION = "老婆刚才你在群里和谁说话"
TEXT = "……那就继续当观众吧。看他什么时候能弄完。"


@pytest.fixture
def service(tmp_path):
    return MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), router=MemoryRouter(),
                         policy=MemoryPolicy(master_user_ids=("owner",)), clock=lambda: NOW)


def source(service, *, group="872732158", message="m1", text=TEXT, sender="bot", self_id="bot", sent=True, at=NOW-41, status="sent"):
    service.record_turn(conversation_id=group, conversation_type="group", message_id=message+":assistant",
        role="assistant", text=text, source_id=f"turn:{group}:{message}:assistant",
        metadata={"sender_id": sender, "self_id": self_id, "requester_id": "member", "shinku_involved": True,
                  "attachment_refs": [{"url": "https://example.invalid/?rkey=DO_NOT_LEAK"}]})
    # Deterministic fixture time, not live production data.
    with service.store._connect() as con:
        con.execute("UPDATE memory_events SET created_at=? WHERE source_id=?", (at, f"turn:{group}:{message}:assistant"))
    if sent:
        service.store.confirm_assistant_delivery(scope="group:"+group, source_id=f"turn:{group}:{message}:assistant",
            expected_text=text, delivered_at=at, status=status, requester_name="小明", content_complete=True)


def inspect(service, **kwargs):
    return service.inspect_self_group_activity(conversation_id="owner", conversation_type="private",
        requester_id="owner", query=QUESTION, self_id="bot", **kwargs)


def turn(text=QUESTION, *, sender="owner", group=None, message_id="request"):
    event = {"post_type": "message", "message_type": "group" if group else "private", "self_id": "bot",
             "user_id": sender, "message_id": message_id, "sender": {"nickname": "小明"},
             "message": [{"type": "text", "data": {"text": text}}]}
    if group:
        event["group_id"] = group
    message = IncomingMessage.from_event(event)
    return QQTurn(conversation_id=group or sender, conversation_type=event["message_type"],
        primary_message_id=message_id, message_ids=(message_id,), combined_text=text, messages=(message,))


@pytest.mark.parametrize("text", [QUESTION, "群里你刚才回复了谁", "你刚才有没有在群里说话", "你刚才在哪个群发言", "你今天在群里聊了什么", "真红刚才在群里说了什么", "你刚才回复了哪个群的消息"])
def test_intent_is_own_historical_group_speech(text):
    assert is_self_group_recall(text)


@pytest.mark.parametrize("text", ["你觉得刚才群里他说了什么", "请你去群里说话", "你去群里聊一下", "刚才群里他和谁说话", "你刚才在私聊说了什么", "你在群里说话很可爱", "你刚才说群里没人吗", "你刚才在群里说话很可爱"])
def test_other_speech_current_tasks_and_chitchat_are_not_activity_intent(text):
    assert not is_self_group_recall(text)


def test_original_incident_routes_before_local_private_continuity(service):
    source(service)
    service.router.decide = lambda **_: (_ for _ in ()).throw(AssertionError("wrong local route"))
    result = service.prepare_turn(conversation_id="owner", conversation_type="private", user_message=QUESTION,
        recent_context="刚才断线了，现在回来了", message_meta={"sender_id": "owner", "self_id": "bot", "topic_continuation": True})
    assert result.decision.reason == "self_group_activity"
    assert result.retrieved_count == 1
    assert "小明" in result.rendered
    assert result.records[0].scope == "group:872732158"
    assert "DO_NOT_LEAK" not in json.dumps(result.as_dict())
    with service.store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM memory_events WHERE scope='private:master'").fetchone()[0] == 0


def test_tool_recent_mode_still_uses_explicit_self_activity_query(service):
    source(service)
    result = MemorySearchTool(service).execute(call={"mode": "recent"}, context={"qq_turn": turn()})
    assert result["state_updates"]["memory_inspection"]["mode"] == "self_activity"
    assert "小明" in result["followup_context"]


def test_no_hit_is_unknown_and_never_negative_evidence(service):
    result = inspect(service)
    assert result["absence_means"] == "unknown"
    assert "没法确认" in result["answer"]
    assert "没在群里说话" not in result["answer"]


def test_private_nonowner_cannot_scan_groups(service, monkeypatch):
    source(service)
    monkeypatch.setattr(service.store, "delivered_group_activity", lambda **_: pytest.fail("unauthorized scan"))
    result = service.inspect_self_group_activity(conversation_id="stranger", conversation_type="private",
        requester_id="stranger", query=QUESTION)
    assert result["access"] == "denied" and not result["records"]
    assert "小明" not in result["rendered"]


def test_group_read_is_same_scope_even_for_owner(service):
    source(service, group="100001", text="这里的回复")
    source(service, group="100002", message="m2", text="别群的秘密")
    result = service.inspect_self_group_activity(conversation_id="100001", conversation_type="group",
        requester_id="owner", query="你刚才在群里说了什么")
    assert result["count"] == 1 and result["records"][0]["scope"] == "group:100001"
    assert "别群" not in json.dumps(result, ensure_ascii=False)


def test_group_caller_cannot_ask_for_other_group(service):
    source(service, group="100002")
    result = service.inspect_self_group_activity(conversation_id="100001", conversation_type="group",
        requester_id="owner", query="你刚才在群100002说了什么")
    assert result["access"] == "denied" and not result["records"]


@pytest.mark.parametrize("kwargs", [{"sent": False}, {"sender": "impostor"}, {"self_id": ""}, {"at": NOW-1801}, {"at": NOW+1}, {"self_id": "anotherbot", "sender": "anotherbot"}])
def test_unconfirmed_wrong_actor_out_of_window_and_other_bot_do_not_count(service, kwargs):
    source(service, **kwargs)
    assert inspect(service)["count"] == 0


def test_third_party_and_private_speech_are_not_read(service):
    service.record_turn(conversation_id="872732158", conversation_type="group", message_id="human",
        text="他人的原话绝不外泄", metadata={"sender_id": "member"})
    service.record_turn(conversation_id="member", conversation_type="private", message_id="private",
        text="另一个人的私聊绝不外泄", metadata={"sender_id": "member"})
    source(service)
    assert "绝不外泄" not in json.dumps(inspect(service), ensure_ascii=False)


def test_partial_delivery_proves_only_occurrence_not_full_body(service):
    source(service, status="partial", text="第一段发出了，第二段发送失败")
    result = inspect(service)
    assert result["count"] == 1 and result["records"][0]["text"] == ""
    assert "部分回复" in result["answer"]
    assert "第二段" not in json.dumps(result, ensure_ascii=False)


def test_revoked_source_cannot_be_revived_by_receipt(service):
    source(service)
    service.store.forget(scope="group:872732158", query="当观众", now=NOW)
    assert inspect(service)["count"] == 0


def test_expired_source_is_not_activity_evidence(service):
    source(service)
    with service.store._connect() as con:
        con.execute("UPDATE memory_events SET expires_at=?", (NOW-1,))
    assert inspect(service)["count"] == 0


def test_receipt_cannot_confirm_a_different_body(service):
    source(service, sent=False)
    assert not service.store.confirm_assistant_delivery(scope="group:872732158", source_id="turn:872732158:m1:assistant",
        expected_text="另一个回答", delivered_at=NOW)
    assert inspect(service)["count"] == 0


class Runtime:
    def __init__(self, response):
        self.response, self.calls = response, []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_delivery_boundary_replaces_model_false_denial(service):
    source(service)
    runtime = Runtime({"speech": "……我刚才没在群里说话。你是不是看错了？"})
    bridge = QQAgentBridge(runtime=runtime, system_prompt="persona", memory_service=service, master_user_ids=("owner",))
    result = bridge.handle_turn(turn())
    assert "小明" in result.final_text
    assert "没在群里说话" not in result.final_text
    assert "自己的群聊发送证据" in runtime.calls[0]["user_prompt"]


def test_no_source_replaces_model_fabricated_positive(service):
    bridge = QQAgentBridge(runtime=Runtime({"speech": "我在群里和小王聊了蛋糕。"}), system_prompt="persona", memory_service=service)
    result = bridge.handle_turn(turn())
    assert "没法确认" in result.final_text and "小王" not in result.final_text


@pytest.mark.parametrize("delivery", [None, DeliveryResult(False, "failed"), DeliveryResult(True, "sent", ("qq-1",))])
def test_only_successful_transport_marks_canonical_source(service, delivery):
    bridge = QQAgentBridge(runtime=Runtime({"speech": TEXT}), system_prompt="persona", memory_service=service,
        sender=(lambda _: delivery) if delivery is not None else None, reply_segment_delay_seconds=0)
    result = bridge.handle_turn(turn("群里普通问题", sender="member", group="872732158", message_id="receipt"))
    assert inspect(service)["count"] == int(delivery is not None and delivery.sent)
    assert result.sent == bool(delivery and delivery.sent)


def test_final_boundary_rechecks_revocation_after_prompt(service):
    source(service)
    class RevokingRuntime(Runtime):
        def call_chat_json(self, **kwargs):
            service.store.forget(scope="group:872732158", query="当观众", now=NOW)
            return {"speech": "我刚才回复了小明。"}
    bridge = QQAgentBridge(runtime=RevokingRuntime({}), system_prompt="persona", memory_service=service)
    assert "没法确认" in bridge.handle_turn(turn()).final_text


def test_fast_perception_final_also_uses_same_guard(service):
    source(service)
    bridge = QQAgentBridge(runtime=Runtime({}), system_prompt="persona", memory_service=service)
    result = bridge._deliver_text(turn(), "我没有说话", reason="lightweight_perception_fast_track", record_memory=False)
    assert "小明" in result.final_text and "没有说话" not in result.final_text


def test_configured_bot_identity_fallback_is_safe_for_frozenset(service):
    source(service)
    event = turn()
    event.messages[0].raw_event.pop("self_id")
    bridge = QQAgentBridge(runtime=Runtime({"speech": "没说话"}), system_prompt="persona", memory_service=service, bot_ids=("bot",))
    assert "小明" in bridge.handle_turn(event).final_text


def test_complete_body_requires_separate_receipt_confirmation(service):
    source(service, sent=False)
    assert service.store.confirm_assistant_delivery(scope="group:872732158", source_id="turn:872732158:m1:assistant",
        expected_text=TEXT, delivered_at=NOW)
    result = service.inspect_self_group_activity(conversation_id="owner", conversation_type="private",
        requester_id="owner", query="你刚才在群里说了什么", self_id="bot")
    assert result["count"] == 1 and result["records"][0]["text"] == ""
    assert "不能确认" in result["answer"]


def test_explicit_day_does_not_reuse_current_day_activity(service):
    source(service)
    result = service.inspect_self_group_activity(conversation_id="owner", conversation_type="private",
        requester_id="owner", query="你昨天在群里说了什么", self_id="bot")
    assert result["count"] == 0


def test_raw_reclaimed_activity_never_revives_original_body(service):
    source(service)
    with service.store._connect() as con:
        con.execute("UPDATE memory_events SET metadata_json=json_set(metadata_json,'$.raw_detail_unavailable',json('true'))")
    result = inspect(service)
    assert result["count"] == 1 and result["records"][0]["text"] == ""


def test_private_missing_bot_identity_does_not_scan_all_bots(service, monkeypatch):
    source(service)
    monkeypatch.setattr(service.store, "delivered_group_activity", lambda **_: pytest.fail("unbound bot scan"))
    result = service.inspect_self_group_activity(conversation_id="owner", conversation_type="private",
        requester_id="owner", query=QUESTION)
    assert not result["records"] and "没法确认" in result["answer"]
