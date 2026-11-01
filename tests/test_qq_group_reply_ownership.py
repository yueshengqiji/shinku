"""Ambient messages retain their author and never acquire a task by interjection."""
from reply_review_fixture import dialogue_review_result
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shinku.agent.loop import AgentDecision
from shinku.qq.group_reply_review import review_group_reply
from shinku.qq.turn_context import group_current_context
from test_autonomous_reply_failure import bot, turn
from test_turn_state_attention import adapter, event


ALLOW = {"verdict": "allow", "relation": "join", "task_claim": "none", "task_id": ""}


class Runtime:
    def __init__(self, verdict):
        self.calls = []
        self.verdict = verdict

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        if kwargs["system_prompt"] == "群聊说话人与任务归属核验":
            if isinstance(self.verdict, Exception):
                raise self.verdict
            return self.verdict
        return {"kind": "final", "text": "嗯，刚才没跑完。我碰不到那边环境，需要把启动脚本写出来吗？"}


@pytest.mark.parametrize("verdict", [
    {}, RuntimeError("fixture failure"), {"verdict": "allow"},
    {"verdict": "silence", "relation": "other_dialogue", "task_claim": "own"},
    {**ALLOW, "relation": "addressed"}, {**ALLOW, "task_claim": "own", "task_id": "foreign"},
])
def test_unverified_reply_cannot_send_or_write_an_assistant_memory(verdict):
    runtime, sent = Runtime(verdict), []
    host = bot(runtime, sent)
    host._record_assistant_reply = Mock(side_effect=AssertionError("no ownership-contaminated memory"))
    host.voice_renderer = Mock(side_effect=AssertionError("no audio"))
    reply = host.handle_turn(turn())
    assert reply.agent_status == "silent" and not sent and not reply.final_text
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["native_tools"] is None and runtime.calls[1]["history_turns"] == []


def test_natural_bystander_join_remains_possible():
    runtime, sent = Runtime(ALLOW), []
    original = runtime.call_chat_json
    def call(**options):
        if options["system_prompt"] != "群聊说话人与任务归属核验":
            runtime.calls.append(options)
            review = dialogue_review_result(options)
            if review is not None:
                return review
            return {"kind": "final", "text": "这张画的颜色挺温暖的。"}
        return original(**options)
    runtime.call_chat_json = call
    reply = bot(runtime, sent).handle_turn(turn())
    assert reply.sent and sent and len(runtime.calls) == 2


def test_direct_address_is_not_subject_to_ambient_review():
    runtime, sent = Runtime(RuntimeError("must not review")), []
    reply = bot(runtime, sent).handle_turn(turn(mentioned=True))
    assert reply.sent and len(runtime.calls) == 1
    assert all(row["system_prompt"] != "群聊说话人与任务归属核验" for row in runtime.calls)


def test_own_interjection_does_not_create_a_conversation_or_refresh_it():
    ingress = adapter()
    ingress.ingest(event("m1", "guest", "42/42 全过，现在把它跑起来"), at=100)
    ingress.note_group_reply("g", recipient_ids=("guest",), message_ids=("bot-m1",), autonomous=True, at=101)
    observed = ingress.ingest(event("m2", "guest", "刚才的处理没有完整结束。请再试一次"), at=102)
    assert observed.attention.reason == "possible_bot_followup"
    assert observed.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"] is False
    quoted = ingress.ingest(event("m3", "guest", "真红你怎么看", reply="bot-m1"), at=103)
    assert quoted.scheduled
    followup = ingress.ingest(event("m4", "guest", "能详细解释吗"), at=104)
    assert followup.attention.reason == "bot_thread_followup"
    assert followup.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"] is True


def test_projection_retains_other_members_focus_metadata():
    item = turn()
    message = item.messages[0]
    message.raw_event["_shinku_group_context"] = {
        "current": {"sender_id": "guest", "sender_label": "群友", "autonomous_candidate": True},
        "focus": {"root_sender_id": "alice", "root_sender_label": "Alice",
                  "root_target_ids": ["bob"], "root_target_labels": ["Bob"], "participant_labels": ["Alice", "Bob"]}}
    rendered = group_current_context(item)
    assert "Alice" in rendered and "Bob" in rendered and "未确认" in rendered


def test_confirmation_from_another_message_and_task_scope_are_not_transferable():
    item = turn()
    message = item.messages[0]
    message.raw_event["_shinku_group_context"] = {"current": {
        "message_id": "other-message", "sender_id": "guest", "bot_exchange_confirmed": True}}
    item = replace(item, attention_signals=("bot_thread",))
    options = []
    model = SimpleNamespace(prompt_cache_key="fixture", effects=SimpleNamespace(trace=lambda *a, **k: None),
        call_json=lambda **kw: options.append(kw) or {**ALLOW, "relation": "addressed", "task_claim": "own", "task_id": "t1"})
    ctx = {"agent_task_candidates": [{"task_id": "t1", "requester": "different",
        "session": "group:42", "tool_calls": [{"name": "fixture"}], "contract": {"goal": "other's work"}}]}
    reason = review_group_reply(item, AgentDecision.final("刚才没跑完"), context=ctx, model=model,
        trace_id="", available_tool_names=set())
    assert reason == "group_reply_task_not_owned"
    payload = json.loads(options[0]["user_prompt"])
    assert not payload["confirmed_bot_exchange"] and not payload["own_executed_tasks_same_requester_session"]


def test_confirmed_same_requester_task_can_be_discussed_but_join_cannot_execute():
    from shinku.tools.invocation import ToolInvocation
    item = turn()
    item.messages[0].raw_event["_shinku_group_context"] = {"current": {
        "message_id": item.primary_message_id, "sender_id": "guest", "bot_exchange_confirmed": True}}
    verdict = {**ALLOW, "relation": "addressed", "task_claim": "own", "task_id": "t1"}
    model = SimpleNamespace(prompt_cache_key="fixture", effects=SimpleNamespace(trace=lambda *a, **k: None),
                            call_json=lambda **kw: verdict)
    ctx = {"agent_task_candidates": [{"task_id": "t1", "requester": "guest", "session": "group:42",
        "status": "failed", "tool_calls": [{"name": "fixture"}], "contract": {"goal": "fixture"}}]}
    assert not review_group_reply(item, AgentDecision.final("刚才那一步失败了"), context=ctx,
        model=model, trace_id="", available_tool_names=set())
    verdict.update(ALLOW)
    assert review_group_reply(item, AgentDecision.tool(ToolInvocation("fixture", {})), context=ctx,
        model=model, trace_id="", available_tool_names={"fixture"}) == "group_action_not_addressed"
