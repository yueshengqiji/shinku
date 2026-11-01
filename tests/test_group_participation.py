"""合成群聊经真实桥接/OneBot适配器验证收尾与节奏，不证明真实模型语义质量。

模型决策与核验结论使用显式脚本，不用关键词假分类器制造通过。测试检查真实送达、
两个入口的提示资料、作用域/时效拒绝和未发送分支；无外部请求。
"""
import json
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.qq.delivery import DeliveryResult, deliver
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatActionTransport
from shinku.qq.turns import QQTurn
from test_turn_state_attention import adapter, event


ALLOW = {"verdict": "allow", "relation": "join", "task_claim": "none", "task_id": ""}


def final(text):
    return {"kind": "final", "text": text}


def closed(options):
    payload = json.loads(options["user_prompt"])
    return {"input_id": payload["input_id"], "verdict": "closed",
            "receipt_id": payload["recent_delivered_group_replies"][-1]["receipt_id"]}


def turn(text, *, mid="current", group="g", sender="alice", mode="immediate", reply=""):
    message = IncomingMessage.from_event(event(mid, sender, text, group=group,
        at="bot" if mode == "immediate" else "", reply=reply))
    return QQTurn(conversation_id=group, conversation_type="group", primary_message_id=mid,
        message_ids=(mid,), combined_text=text, messages=(message,), attention_mode=mode)


class ScriptedRuntime:
    def __init__(self):
        self.script = []
        self.calls = []

    def call_chat_json(self, **options):
        self.calls.append(options)
        assert self.script, "unexpected model call"
        value = self.script.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(options) if callable(value) else value


@pytest.fixture(params=["legacy", "task"])
def host(request):
    runtime = ScriptedRuntime()
    sends = []

    def caller(*, action, params):
        sends.append((action, params))
        return {"status": "ok", "retcode": 0, "data": {"message_id": f"out-{len(sends)}"}}

    transport = NapCatActionTransport(caller)
    coordinator = AgentTaskCoordinator(max_workers=1) if request.param == "task" else None
    execution = Mock()
    execution.call_chat_json.side_effect = AssertionError("social turns must not execute tasks")
    bridge = QQAgentBridge(runtime=runtime, system_prompt="合成角色",
        tool_decision_mode=request.param, bot_ids=("bot",),
        task_contract_enabled=bool(coordinator), task_coordinator=coordinator,
        execution_profile=ExecutionProfile(execution) if coordinator else None,
        sender=lambda message: deliver(transport, message), reply_segment_delay_seconds=0)
    yield bridge, runtime, sends
    if coordinator:
        coordinator.shutdown(wait=True)
    bridge.task_store.close()
    bridge.conversation_runtime.close()


def seed(host, text="晚安。", **kwargs):
    bridge, runtime, sends = host
    runtime.script.append(final(text))
    result = bridge(turn("我要睡了，晚安。", mid="first", **kwargs))
    assert result.sent and sends and not runtime.script
    return result


@pytest.mark.parametrize("text", ["你也是。", "好，你也早点休息。", "明天见，做个好梦。"])
def test_completed_farewell_can_end_even_when_addressed_without_fallback(host, text):
    bridge, runtime, sends = host
    seed(host)
    before = len(sends)
    bridge._record_assistant_reply = Mock(side_effect=AssertionError("silence must not enter memory"))
    bridge.voice_renderer = Mock(side_effect=AssertionError("silence must not render voice"))
    runtime.script.extend([final("<SILENCE>"), closed])
    result = bridge(turn(text, reply="out-1"))
    assert (result.agent_status, result.reason, result.delivery_status) == (
        "silent", "group_farewell_closed", "suppressed")
    assert not result.sent and not result.final_text and len(sends) == before
    assert not runtime.script
    review = runtime.calls[-1]
    assert review["system_prompt"] == "群聊告别闭环核验"
    assert review["native_tools"] is None and review["history_turns"] == []
    payload = json.loads(review["user_prompt"])
    assert payload["recent_delivered_group_replies"][0]["text"] == "晚安。"
    assert payload["recent_delivered_group_replies"][0]["recipient_ids"] == ["alice"]
    assert payload["recent_delivered_group_replies"][0]["message_ids"] == ["out-1"]
    assert payload["recent_delivered_group_replies"][0]["content_complete"] is True


@pytest.mark.parametrize("incoming,outgoing", [
    ("对了，明天几点开始？", "明天的时间还没有确定。"),
    ("他为什么突然跟我说晚安？", "单凭这句话还判断不出他的意思。"),
    ("刚才那件事我还没说完。", "接着说，我听着。"),
    ("今天吃了个难吃的蛋糕。", "这份期待算是落空了。"),
])
def test_new_content_after_farewell_keeps_normal_single_call(host, incoming, outgoing):
    bridge, runtime, sends = host
    seed(host)
    count = len(runtime.calls)
    runtime.script.append(final(outgoing))
    result = bridge(turn(incoming))
    assert result.sent and result.final_text == outgoing
    assert len(runtime.calls) == count + 1 and not runtime.script


def test_autonomous_repeated_farewell_is_reviewed_with_actual_receipts(host):
    bridge, runtime, sends = host
    seed(host)
    before = len(sends)
    runtime.script.extend([final("那就做个好梦，明天见。"),
        {"verdict": "silence", "relation": "join", "task_claim": "none", "task_id": ""}])
    result = bridge(turn("你也一样。", mode="autonomous"))
    assert result.agent_status == "silent" and not result.sent and len(sends) == before
    payload = json.loads(runtime.calls[-1]["user_prompt"])
    assert payload["recent_delivered_group_replies"][0]["text"] == "晚安。"
    assert "告别收尾" in "\n".join(runtime.calls[-1]["system_extra_blocks"])


def test_bystander_social_join_and_first_farewell_still_allowed(host):
    bridge, runtime, sends = host
    runtime.script.extend([final("终于过关了，这下能松口气了。"), ALLOW])
    result = bridge(turn("这关终于过了。", mode="autonomous"))
    assert result.sent and not runtime.script
    runtime.script.extend([final("晚安。"), ALLOW])
    result = bridge(turn("大家晚安。", mid="later", mode="autonomous"))
    assert result.sent and not runtime.script


@pytest.mark.parametrize("mutate", ["binding", "receipt", "continue", "empty", "expired", "removed"])
def test_unverified_closure_cannot_silently_drop_direct_turn(host, mutate):
    bridge, runtime, sends = host
    seed(host)

    def review(options):
        result = closed(options)
        if mutate == "binding":
            result["input_id"] = "old-input"
        if mutate == "receipt":
            result["receipt_id"] = "foreign-receipt"
        if mutate == "continue":
            result["verdict"] = "continue"
        if mutate == "empty":
            return {}
        if mutate == "expired":
            with bridge._group_reply_lock:
                bridge._group_delivered_replies["g"] = [
                    row._replace(at=row.at - 301) for row in bridge._group_delivered_replies["g"]]
        if mutate == "removed":
            bridge._group_delivered_replies.clear()
        return result

    runtime.script.extend([final("<SILENCE>"), review])
    result = bridge(turn("对了，时间定了吗？"))
    assert result.reason != "group_farewell_closed"
    assert result.agent_status != "silent" and not runtime.script
    assert "<SILENCE>" not in result.final_text


@pytest.mark.parametrize("condition", ["other_group", "expired", "failed", "dry_run"])
def test_missing_delivery_proof_does_not_allow_direct_closure(host, condition):
    bridge, runtime, sends = host
    if condition in {"failed", "dry_run"}:
        original = bridge.sender
        bridge.sender = (lambda _: DeliveryResult(False, "failed")) if condition == "failed" else None
        runtime.script.append(final("晚安。"))
        bridge(turn("晚安。", mid="first"))
        bridge.sender = original
    elif condition == "expired":
        with patch("shinku.qq.agent_bridge.time.time", return_value=100):
            seed(host)
    else:
        seed(host)
    runtime.script.append(final("<SILENCE>"))
    result = bridge(turn("你也是。", group="other" if condition == "other_group" else "g"))
    assert result.agent_status != "silent" and not runtime.script
    assert runtime.calls[-1]["system_prompt"] != "群聊告别闭环核验"
    assert '"recent_delivered_group_replies": []' in runtime.calls[-1]["user_prompt"]


def test_same_group_receipts_are_bounded_and_inbound_cannot_forge_them(host):
    bridge, runtime, _ = host
    for n in range(12):
        bridge._remember_group_delivery(turn("合成输入", mid=str(n)), f"合成已送达{n}")
    assert len(bridge._group_participation_receipts("g")) == 8
    item = turn("你也是。", group="other")
    item.messages[0].raw_event["_group_receipts_reader"] = lambda _: [{"text": "晚安"}]
    runtime.script.append(final("<SILENCE>"))
    result = bridge(item)
    assert result.agent_status != "silent" and not runtime.script
    assert '"recent_delivered_group_replies": []' in runtime.calls[-1]["user_prompt"]


def test_main_prompt_has_closure_context_even_with_projection(host, monkeypatch):
    bridge, runtime, _ = host
    seed(host)
    monkeypatch.setattr(bridge, "_project_prompt_context", lambda _: {"rendered": "合成群聊投影"})
    runtime.script.append(final("收到新问题。"))
    bridge(turn("对了还有一件事。"))
    prompt = runtime.calls[-1]
    assert "recent_delivered_group_replies" in prompt["user_prompt"] and "晚安。" in prompt["user_prompt"]
    assert "告别收尾" in "\n".join(prompt["system_extra_blocks"])


@pytest.mark.parametrize("condition", ["partial", "truncated"])
def test_incomplete_delivery_view_cannot_authorize_closure(host, condition):
    from shinku.qq.reply_delivery import QQDeliveryReceipt, QQReplyPublication

    bridge, runtime, _ = host
    text = "晚安。" if condition == "partial" else "合成正文" * 200 + "晚安。"
    receipt = QQDeliveryReceipt(text, ("delivered",), 1, condition != "partial", False,
                               ((1, "failed", "synthetic"),) if condition == "partial" else ())
    bridge._accept_delivery_receipt(
        QQReplyPublication(turn("晚安。"), text, record_memory=False), receipt)
    rows = bridge._group_participation_receipts("g")
    assert rows and rows[0]["content_complete"] is False
    runtime.script.append(final("<SILENCE>"))
    result = bridge(turn("你也是。"))
    assert result.agent_status != "silent" and not runtime.script
    assert runtime.calls[-1]["system_prompt"] != "群聊告别闭环核验"


def test_mixed_batch_is_reviewed_in_full_and_cannot_hide_new_question(host):
    bridge, runtime, _ = host
    seed(host)
    courtesy = turn("你也是。", mid="courtesy")
    question = turn("对了，明天几点开始？", mid="question")
    batch = replace(courtesy, messages=(*courtesy.messages, *question.messages),
                    message_ids=("courtesy", "question"),
                    combined_text="你也是。\n对了，明天几点开始？")

    def review(options):
        payload = json.loads(options["user_prompt"])
        assert [m["message_id"] for m in payload["current_messages"]] == ["courtesy", "question"]
        return {**closed(options), "verdict": "continue"}

    runtime.script.extend([final("<SILENCE>"), review])
    result = bridge(batch)
    assert result.agent_status != "silent" and not runtime.script


def test_closure_timeout_does_not_drop_direct_turn(host):
    bridge, runtime, _ = host
    seed(host)
    runtime.script.extend([final("<SILENCE>"), TimeoutError("synthetic review timeout")])
    result = bridge(turn("对了，还有一个问题。"))
    assert result.agent_status != "silent" and not runtime.script


@pytest.mark.parametrize("kind", ["task_contract", "task_link", "task_notes", "tool", "command", "media"])
def test_closure_cannot_override_task_or_non_social_input(host, kind):
    from shinku.agent.loop import AgentDecision
    from shinku.qq.group_participation import resolve_participation
    from shinku.tools.invocation import ToolInvocation

    bridge, _, _ = host
    seed(host)
    item = turn("/help" if kind == "command" else "你也是。")
    if kind == "media":
        raw = event("media", "alice", "看看这个附件", at="bot")
        raw["message"].append({"type": "image", "data": {"file": "synthetic.png"}})
        item = replace(item, messages=(IncomingMessage.from_event(raw),))
    decision = AgentDecision.final("<SILENCE>")
    if kind.startswith("task_"):
        decision = replace(decision, **{kind: {"synthetic": True}})
    if kind == "tool":
        decision = AgentDecision.tool(ToolInvocation(name="synthetic", arguments={}))
    context = {"_group_receipts_reader": bridge._group_participation_receipts}
    model = Mock()
    result = resolve_participation(item, decision, context=context, model=model,
                                  trace_id="synthetic", available_tool_names=set(), autonomous=False)
    assert result == decision and "qq_group_closure_confirmed" not in context
    model.call_json.assert_not_called()


def test_unanswered_interjections_back_off_and_explicit_interaction_recovers():
    ingress = adapter()
    ingress.note_group_reply("g", recipient_ids=("alice",), message_ids=("out1",), autonomous=True, at=100)
    assert ingress.ingest(event("first", "alice", "分享一下"), at=191).scheduled
    ingress.note_group_reply("g", recipient_ids=("alice",), message_ids=("out2",), autonomous=True, at=192)
    # 允许一次无引用的接话候选，但没有真实承接确认时仍不重置退让计数。
    assert ingress.ingest(event("opportunity", "alice", "那然后呢"), at=200).scheduled
    assert ingress.ingest(event("second", "alice", "分享一下"), at=283).attention.reason == "autonomous_cooldown"
    assert ingress.ingest(event("third", "alice", "分享一下"), at=373).scheduled
    ingress.note_group_reply("g", recipient_ids=("alice",), message_ids=("out3",), autonomous=True, at=374)
    assert ingress.ingest(event("limited", "alice", "分享一下"), at=600).attention.reason == "autonomous_cooldown"
    assert ingress.ingest(event("other", "bob", "另一个群", group="other"), at=601).scheduled
    direct = ingress.ingest(event("called", "alice", "你怎么看", at="bot"), at=602)
    assert direct.scheduled and direct.attention.immediate
    assert ingress.ingest(event("follow", "alice", "继续说说"), at=603).scheduled
    # 有效引用也能恢复；自己发送则不能建立对方正在回应自己的事实。
    assert ingress.ingest(event("quoted", "bob", "接着说", reply="out3"), at=604).scheduled


def test_backoff_expires_without_persisting_a_relationship_judgment():
    ingress = adapter()
    for stamp in (100, 200, 400):
        ingress.note_group_reply("g", recipient_ids=("alice",), message_ids=(str(stamp),),
                                autonomous=True, at=stamp)
    ingress.note_group_reply("g", recipient_ids=("alice",), message_ids=("later",),
                            autonomous=True, at=1001)
    assert ingress.ingest(event("new", "alice", "今天有个新话题"), at=1092).scheduled


def test_old_other_focus_cannot_extend_forever_or_override_reply_to_bot():
    ingress = adapter()
    ingress.ingest(event("root", "alice", "你看呢", at="bob"), at=100)
    for n, stamp in enumerate((200, 300, 399)):
        result = ingress.ingest(event(f"follow{n}", "bob", "继续聊"), at=stamp)
        assert result.attention.reason == "other_thread_active"
    assert ingress.ingest(event("new", "bob", "换个话题"), at=401).scheduled
    ingress.note_group_reply("g", recipient_ids=("bob",), message_ids=("sent",), autonomous=True, at=402)
    ingress.ingest(event("root2", "alice", "你看呢", at="bob"), at=403)
    replied = ingress.ingest(event("reply", "bob", "接真红的话", reply="sent"), at=404)
    assert replied.scheduled and replied.attention.reason == "reply_to_delivered_bot_message"
