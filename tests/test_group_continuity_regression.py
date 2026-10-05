from unittest.mock import patch
from types import SimpleNamespace
from dataclasses import replace

from shinku.agent.loop import AgentState
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.group_context import GroupContextTracker
from shinku.qq.message import IncomingMessage
from shinku.qq.routing import RouteDecision
from shinku.qq.turns import QQTurn
from shinku.memory.service import MemoryService
from shinku.memory.policy import MemoryPolicy
from shinku.memory.store import MemoryStore
from shinku.memory.router import MemoryRouter


def message(mid, text, sender="u1", group="g1", reply=""):
    segments = [{"type": "text", "data": {"text": text}}]
    if reply:
        segments.insert(0, {"type": "reply", "data": {"id": reply}})
    return IncomingMessage.from_event({
        "post_type": "message", "message_type": "group", "group_id": group,
        "user_id": sender, "message_id": mid,
        "sender": {"nickname": sender}, "message": segments,
    })


def turn(msg):
    return QQTurn(conversation_id=msg.conversation_id, conversation_type="group",
                  primary_message_id=msg.message_id, message_ids=(msg.message_id,),
                  combined_text=msg.text, messages=(msg,))


class Runtime:
    def call_chat_json(self, **kwargs):
        return {"text": "test"}


def prompt(bridge, msg):
    return bridge._build_prompt(AgentState(task_id="test", context={"qq_turn": turn(msg)}),
                                include_agent_control=False).user_prompt


def test_reply_anchor_keeps_later_correction_and_two_minute_pause():
    tracker = GroupContextTracker()
    route = RouteDecision(True, "direct", addressed=True)
    tracker.enrich(message("old", "我以为是假的"), route, now=100)
    tracker.enrich(message("new", "查到官方页面了，刚才看错了", sender="u2"), route, now=230)
    followup = message("follow", "那现在呢", reply="old")
    tracker.enrich(followup, route, now=231)
    assert [x["message_id"] for x in followup.raw_event["_shinku_group_context"]["items"]] == ["old", "new"]


def test_sent_reply_follows_group_across_speakers_but_expires_and_isolates():
    bridge = QQAgentBridge(runtime=Runtime(), system_prompt="test",
                           sender=lambda _: DeliveryResult(True, "sent", ("out",)))
    with patch("shinku.qq.agent_bridge.time.time", return_value=100):
        bridge._deliver_text(turn(message("m1", "去查一下")), "搜索失败，暂时无法确认。", record_memory=False)
    with patch("shinku.qq.agent_bridge.time.time", return_value=120):
        result = prompt(bridge, message("m2", "为什么", sender="u2"))
        assert "搜索失败，暂时无法确认" in result
        assert "真红 → u1" in result
        assert "搜索失败，暂时无法确认" not in prompt(bridge, message("m3", "为什么", group="g2"))
    with patch("shinku.qq.agent_bridge.time.time", return_value=401):
        assert "搜索失败，暂时无法确认" not in prompt(bridge, message("m4", "你好"))


def test_failed_or_dry_run_replies_are_not_claimed_as_sent():
    for sender in (None, lambda _: DeliveryResult(False, "failed")):
        bridge = QQAgentBridge(runtime=Runtime(), system_prompt="test", sender=sender)
        bridge._deliver_text(turn(message("m1", "你好")), "测试不能发出的内容", record_memory=False)
        assert "测试不能发出的内容" not in prompt(bridge, message("m2", "然后呢", sender="u2"))


def test_attention_delivery_observer_only_sees_successful_group_sends():
    observed = []
    callback = lambda *args: observed.append(args)
    spoken = replace(turn(message("m1", "大家怎么看", sender="u1")), attention_mode="autonomous")
    success = QQAgentBridge(runtime=Runtime(), system_prompt="test",
                            sender=lambda _: DeliveryResult(True, "sent", ("out-1",)),
                            group_delivery_observer=callback)
    with patch("shinku.qq.agent_bridge.time.time", return_value=100):
        success._deliver_text(spoken, "我看到了", record_memory=False, autonomous_authorized=True)
    assert observed == [("g1", ("u1",), ("out-1",), True, 100)]

    failed = QQAgentBridge(runtime=Runtime(), system_prompt="test",
                           sender=lambda _: DeliveryResult(False, "failed"),
                           group_delivery_observer=callback)
    failed._deliver_text(spoken, "这条没发出去", record_memory=False, autonomous_authorized=True)
    assert len(observed) == 1

    suppressed = success._deliver_text(spoken, "未被模型选中的提示", record_memory=False)
    assert suppressed.delivery_status == "suppressed" and not suppressed.sent
    assert len(observed) == 1


def test_assistant_memory_does_not_take_original_requesters_name(tmp_path):
    # Rendering now revalidates sources even with feature flags off. Exercise
    # the real contract, not an uninitialized service without a source store.
    store = MemoryStore(tmp_path / "memory.sqlite3")
    service = MemoryService(store=store, router=MemoryRouter(), policy=MemoryPolicy(), clock=lambda: 100)
    store.record_event(conversation_id="owner", scope="private:owner", source_id="test", text="我还没查证",
        created_at=100, metadata={"role": "assistant", "sender_id": "owner", "sender_name": "主人"})
    record = store.event_by_source_id(source_id="test", scope="private:owner")
    rendered = service._render((record,), mode="recent")
    assert "真红：我还没查证" in rendered
    assert "主人：我还没查证" not in rendered
