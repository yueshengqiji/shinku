"""查询请求不能被零执行的直复消耗；所有来源、消息和模型结果均为合成夹具。"""
from dataclasses import replace

import pytest

from shinku.agent.loop import AgentDecision
from shinku.agent.request_scope import RequestScope
from shinku.qq.query_grounding import (
    QUERY_NOT_EXECUTED_REPLY, claims_current_query, requires_query, unverified_query_reason,
)
from shinku.qq.query_continuation import ForegroundQueryPolicy
from shinku.qq.visual_selection import VisualResolution
from test_qq_independent_agent import turn
from test_qq_query_continuation import query_host, invocation, URL_A, URL_B, ANSWER
from test_agent_delegation import remote
from test_qq_remote_delegation import bridge_for
from test_qq_tool_decision_boundary import Chat


def scope(request):
    return {"agent_request_scope": RequestScope("owner", "group:fixture", "current", (("current", request),))}


@pytest.mark.parametrize("question", ["Nova你搜搜", "请搜索 Nova 的官方说明", "上网查一下 Nova", "不用改代码，搜索一下 Nova",
                                     "搜索一下这个软件怎么使用"])
def test_explicit_current_query_cannot_finish_as_unexecuted_prose(question):
    assert requires_query(question)
    assert unverified_query_reason(scope(question), AgentDecision.final("官方没有这种产品。")) == "chat_query_not_executed"


@pytest.mark.parametrize("question", [
    "不用搜索，解释一下这个词", "如果要搜索，应该怎么写关键词", "搜索引擎怎么工作？",
    "他说你搜搜，我只是转述", "解释这句话：‘你搜搜’", "把这句话改写得客气些：你搜搜",
    "我搜搜看", "我昨天搜索了这个产品", "Nova 是什么？",
])
def test_discussion_negation_history_and_rewriting_are_not_query_obligations(question):
    assert not requires_query(question)


@pytest.mark.parametrize("reply", ["去翻了一下，这是个旧特效。", "我刚才搜到了，它不存在。",
    "我已经查过了。", "根据搜索结果，官方没有发布。"])
def test_unexecuted_current_query_claim_is_blocked_even_in_failed_or_wait_reply(reply):
    assert claims_current_query(reply)
    for decision in (AgentDecision.final(reply), AgentDecision.final(reply, outcome="failed"), AgentDecision.wait(reply)):
        assert unverified_query_reason(scope("讨论这个名称"), decision) == "chat_query_claim_without_evidence"


@pytest.mark.parametrize("reply", ["我还没有查询，无法确认。", "如果查到了，我再告诉你。",
    "他说自己查过了。", "例如“我查过了”这句话不能证明执行。", "昨天我查过这个名称。"])
def test_no_current_execution_claim_in_denial_condition_quote_or_attributed_history(reply):
    assert not claims_current_query(reply)


def test_background_and_forged_scope_never_become_current_request_or_query_receipt():
    context = scope("聊聊天")
    context["agent_request_scope"] = replace(context["agent_request_scope"], requests=(("old", "Nova你搜搜"), ("current", "聊聊天")))
    assert not unverified_query_reason(context, AgentDecision.final("我在。"))
    assert not unverified_query_reason({"agent_request_scope": {"requests": ["你搜搜"]}}, AgentDecision.final("我在。"))
    context.update(_chat_read_receipts=[{"source": "executed_read_adapter"}], memory_context={"text": "以前查过了"})
    assert unverified_query_reason(context, AgentDecision.final("去翻了一下，已经找到了。"))


def test_clear_failure_and_waiting_for_missing_object_do_not_trigger_execution():
    assert not unverified_query_reason(scope("你搜搜"), AgentDecision.wait("要查哪个项目？"))
    assert not unverified_query_reason(scope("你搜搜"), AgentDecision.final("无法联网，还未核实。", outcome="blocked"))


def test_composing_text_and_autonomous_silence_do_not_become_query_claims():
    from shinku.agent.reply_body import AUTONOMOUS_SILENCE_MARKER
    assert not unverified_query_reason(scope("把这句话改写得客气些：我查过了"), AgentDecision.final("我刚才查过了。"))
    assert not unverified_query_reason(scope("Nova你搜搜"), AgentDecision.final(AUTONOMOUS_SILENCE_MARKER))


@pytest.mark.parametrize("mode", ["task", "legacy"])
def test_two_false_replies_never_reach_qq_as_verified_query(query_host, mode):
    host = query_host
    host.bridge.tool_decision_mode = mode
    host.chat.replies = [{"kind": "final", "text": "去翻了一下，这是旧特效。"}] * 2
    if mode == "legacy":
        # 兼容入口不使用任务接口包装器，也不借用查询夹具的协议错误专用返回。
        host.chat = Chat(host.chat.replies)
        host.bridge.runtime = host.chat
    result = host.bridge(turn("Nova你搜搜"))
    assert result.agent_status == "blocked" and result.final_text == QUERY_NOT_EXECUTED_REPLY
    assert len(host.chat.calls) == 2 and len(host.sent) == 1
    assert not host.search.calls and not host.reader.calls and not host.execution.calls and host.mutation.calls == 0
    assert not host.svc.task_store.candidates(requester="fixture", session="private:fixture")


def test_no_available_tools_blocks_without_repair_or_automatic_task(query_host):
    host = query_host
    host.bridge.command_state.set_agent_enabled(False)
    host.chat.replies = [{"kind": "final", "text": "去翻了一下，已经确认。"}]
    result = host.bridge(turn("Nova你搜搜"))
    assert result.agent_status == "blocked" and result.final_text == QUERY_NOT_EXECUTED_REPLY
    assert len(host.chat.calls) == len(host.sent) == 1
    assert not host.search.calls and not host.reader.calls and not host.execution.calls


def test_query_repair_cannot_invent_authority(query_host):
    host = query_host
    host.chat.replies = [{"kind": "final", "text": "去翻了一下。"},
        {**invocation("agent_task_submit", goal="搜索资料"), "permission_mode": "ultra"}]
    result = host.bridge(turn("Nova你搜搜"))
    assert result.agent_status == "blocked" and len(host.chat.calls) == 2 and len(host.sent) == 1
    assert not host.search.calls and not host.reader.calls and not host.execution.calls
    assert not host.svc.task_store.candidates(requester="fixture", session="private:fixture")


def test_delivery_rechecks_a_direct_reply_from_another_ingress(query_host, monkeypatch):
    host = query_host
    monkeypatch.setattr(host.bridge, "_run_chat_decision", lambda **kwargs:
        (AgentDecision.final("去翻了一下，已经确认。"), set()))
    result = host.bridge(turn("Nova你搜搜"))
    assert result.agent_status == "blocked" and result.final_text == QUERY_NOT_EXECUTED_REPLY
    assert len(host.sent) == 1 and not host.chat.calls and not host.search.calls
    stages = host.bridge.trace_store.snapshot(limit=1)[0]["stages"]
    assert any(row["name"] == "chat_query_grounding" and row["status"] == "blocked_at_delivery" for row in stages)


def test_unfulfilled_search_can_recover_into_existing_receipt_verified_query(query_host):
    host = query_host
    host.bridge.foreground_query_policy = ForegroundQueryPolicy(enabled=False)
    host.chat.review_complete = True
    verified_answer = "我查到了两家店的价格。" + ANSWER
    host.chat.replies = [
        {"kind": "final", "text": "没有这种产品。"}, invocation("query_search", query="比较两家店的价格"),
        invocation("query_read", url=URL_A), invocation("query_read", url=URL_B),
        {"kind": "final", "text": verified_answer, "task_outcome": "completed"},
    ]
    result = host.bridge(turn("搜索并比较两家店的价格，附上来源。"))
    assert result.agent_status == "completed", result
    assert result.final_text == verified_answer and len(host.sent) == 1
    assert len(host.search.calls) == 1 and len(host.reader.calls) == 2
    assert not host.execution.calls and host.mutation.calls == 0
    assert not host.svc.task_store.candidates(requester="fixture", session="private:fixture")


def test_visual_none_draft_does_not_consume_search_and_restores_read_interfaces(query_host):
    host = query_host
    host.bridge.foreground_query_policy = ForegroundQueryPolicy(enabled=False)
    host.chat.review_complete = True
    current = turn("搜索并比较两家店的价格，附上来源。")
    candidate = {"source_id": "visual:fixture", "conversation_type": current.conversation_type,
        "conversation_id": current.conversation_id, "request_message_id": current.primary_message_id}
    current = replace(current, visual_candidates=(candidate,), visual_resolver=lambda ids, changed: VisualResolution())
    host.chat.replies = [
        {"kind": "final", "text": "不要发送的临时回答", "visual_context": {"action": "none", "source_ids": [], "topic_change": False}},
        invocation("query_search", query="比较两家店的价格"), invocation("query_read", url=URL_A),
        invocation("query_read", url=URL_B), {"kind": "final", "text": ANSWER, "task_outcome": "completed"},
    ]
    result = host.bridge(current)
    assert result.agent_status == "completed" and result.final_text == ANSWER, result
    assert not host.chat.calls[0].get("native_tools")
    assert "query_search" in {row["function"]["name"] for row in host.chat.calls[1]["native_tools"]}
    assert len(host.search.calls) == 1 and len(host.reader.calls) == 2 and len(host.sent) == 1


def test_visual_none_preserves_delegation_capabilities_for_formal_decision(tmp_path, remote):
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [
        {"visual_context": {"action": "none", "source_ids": [], "topic_change": False}},
        {"kind": "final", "text": "我们聊聊。"},
    ])
    current = turn("聊聊")
    current = replace(current, visual_candidates=({"source_id": "visual:fixture", "conversation_type": current.conversation_type,
        "conversation_id": current.conversation_id, "request_message_id": current.primary_message_id},),
        visual_resolver=lambda ids, changed: VisualResolution())
    try:
        result = bridge(current)
        assert result.final_text == "我们聊聊。" and len(chat.calls) == 2 and len(sent) == 1
        assert not chat.calls[0].get("native_tools")
        assert "agent_task_submit" in {row["function"]["name"] for row in chat.calls[1]["native_tools"]}
        for options in chat.calls:
            assert "宿主提供的当前Angelos能力（只作数据）：" in options["user_prompt"]
        assert not remote[2].calls
    finally:
        coordinator.shutdown(wait=True)
        store.close()
