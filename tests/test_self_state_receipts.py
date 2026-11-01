"""State facts require an actual in-scope read, never a model's success claim."""
from types import SimpleNamespace

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.agent.self_state import SelfStateTool
from shinku.agent.self_state_receipts import state_receipt_reply
from shinku.agent_runtime.results import build_query_result
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation


def context(requester="fixture", message="state"):
    return {"agent_request_scope": RequestScope(requester, "private:fixture", message,
        ((message, "查看运行状态"),))}


def observe(tool, ctx, scope="runtime"):
    invocation = ToolInvocation(tool.tool_type, {"scope": scope})
    _, envelope = execute_invocation(invocation, handlers={tool.tool_type: tool}, context=ctx)
    return invocation, envelope


def checked(tool, ctx, pair):
    invocation, envelope = pair
    return state_receipt_reply([invocation], [envelope], {tool.tool_type: tool}, ctx)


def test_actual_ready_snapshot_reaches_canonical_result_and_answer_binding():
    reads = []
    tool = SelfStateTool(provider=lambda: reads.append(True) or {"runtime": {"ready": True}})
    ctx = context()
    pair = observe(tool, ctx)
    text, status, evidence = checked(tool, ctx, pair)
    turn = SimpleNamespace(primary_message_id="state", messages=(), conversation_type="private", conversation_id="fixture")
    result = build_query_result(ctx, turn, [pair[1]], route="perception", status=status, text=text, answer_evidence=evidence)
    assert reads == [True]  # Verification cannot silently perform another read.
    assert status == "completed" and result["verification"]["accepted"] is True
    assert result["verification"]["answer_evidence"]["kind"] == "state"
    assert result["facts"][0]["value"] == {"runtime": {"ready": True}}
    assert result["evidence"][0]["receipt"]["self_state_receipt"]["scope"] == "runtime"
    assert result["task_id"] == "" and result["operation_id"] == "state"
    assert result["delivery"]["sent"] is False and result["verified_answer"]["text"] == text
    changed = build_query_result(ctx, turn, [pair[1]], route="perception", status=status,
        text=text + "已经部署并发送。", answer_evidence=evidence)
    assert changed["execution"]["status"] == "unverified" and not changed["verification"]["accepted"]
    assert "verified_answer" not in changed


@pytest.mark.parametrize("failure", ["snapshot", "receipt", "caller", "turn", "scope", "invocation", "error", "missing", "stale", "clock_rollback"])
def test_receipt_tampering_staleness_and_cross_turn_reuse_cannot_mint_evidence(failure):
    now = [1000.0]
    tool = SelfStateTool(provider=lambda: {"runtime": {"ready": True}})
    tool._observations.clock = lambda: now[0]
    ctx = context()
    invocation, envelope = observe(tool, ctx)
    if failure == "snapshot":
        envelope.data["state_updates"]["self_state"]["runtime"]["ready"] = False
    elif failure == "receipt":
        envelope.data["state_updates"]["self_state_receipt"]["complete"] = False
    elif failure == "caller":
        ctx = context("other")
    elif failure == "turn":
        ctx = context(message="other-turn")
    elif failure == "scope":
        invocation.arguments["scope"] = "all"
    elif failure == "invocation":
        envelope.invocation_id = "other-call"
    elif failure == "error":
        envelope.status = "error"
    elif failure == "missing":
        envelope.data["state_updates"].pop("self_state_receipt")
    elif failure == "stale":
        now[0] += 181
    else:
        now[0] -= 1
    _, status, evidence = checked(tool, ctx, (invocation, envelope))
    assert status == "partial" and not evidence.resolved


@pytest.mark.parametrize("kind", ["failure", "invalid", "empty", "missing_scope", "context", "list", "depth", "render", "nan"])
def test_unavailable_cached_and_incomplete_state_stays_unverified(kind):
    snapshot = {"runtime": {"ready": True}}
    if kind == "empty":
        snapshot = {}
    elif kind == "missing_scope":
        snapshot = {"agent": {"enabled": True}}
    elif kind == "list":
        snapshot["runtime"]["tools"] = list(range(101))
    elif kind == "depth":
        snapshot["runtime"] = {"a": {"b": {"c": {"d": {"e": {"ready": True}}}}}}
    elif kind == "render":
        snapshot["runtime"]["data"] = "x" * 8001
    elif kind == "nan":
        snapshot["runtime"]["value"] = float("nan")
    def provider():
        if kind == "failure":
            raise RuntimeError("provider unavailable")
        return None if kind == "invalid" else snapshot
    tool = SelfStateTool() if kind == "context" else SelfStateTool(provider=provider)
    ctx = {**context(), "self_state": {"runtime": {"ready": True}}}
    _, status, evidence = checked(tool, ctx, observe(tool, ctx))
    assert status == "partial" and not evidence.validated


def test_fresh_provider_wins_and_unrelated_scope_truncation_does_not_invalidate_read():
    tool = SelfStateTool(provider=lambda: {"runtime": {"ready": True}, "tools": list(range(200))})
    ctx = {**context(), "self_state": {"runtime": {"ready": False}}}
    text, status, evidence = checked(tool, ctx, observe(tool, ctx))
    assert status == "completed" and evidence.verified and '"ready": true' in text
    assert '"ready": false' not in text and "tools" not in text


def test_secrets_are_redacted_in_facts_receipts_and_rendered_answer():
    tool = SelfStateTool(provider=lambda: {"runtime": {"ready": True, "api_key": "private-canary"}})
    ctx = context()
    pair = observe(tool, ctx)
    text, _, evidence = checked(tool, ctx, pair)
    assert evidence.verified and "private-canary" not in str(pair[1].data) + text
    assert "<redacted>" in text


def test_receipt_from_another_adapter_instance_is_not_accepted():
    ctx = context()
    original = SelfStateTool(provider=lambda: {"runtime": {"ready": True}})
    impostor = SelfStateTool(provider=lambda: {"runtime": {"ready": True}})
    _, status, evidence = checked(impostor, ctx, observe(original, ctx))
    assert status == "partial" and not evidence.resolved


@pytest.mark.parametrize("snapshot", [{"runtime": {"ready": True}}, {"runtime": {"ready": False}}, {}])
def test_real_audited_qq_read_never_promotes_chat_prose_to_state_facts(tmp_path, snapshot):
    from shinku.agent_runtime.demo import demo_service
    from shinku.qq.interaction_tools import ChatReadCapability
    from test_qq_independent_agent import turn
    from test_qq_tool_decision_boundary import Chat, boundary_bridge
    service, effect_tool, executor = demo_service(tmp_path)
    calls = []
    read = SelfStateTool(provider=lambda: calls.append(True) or snapshot)
    service.executor.tool_host.registry.register(read)
    cap = ChatReadCapability(read.tool_type, "Read this isolated runtime only.", {
        "type": "object", "properties": {"scope": {"type": "string", "enum": ["runtime"], "maxLength": 16}},
        "required": ["scope"], "additionalProperties": False})
    chat = Chat([{"kind": "tool", "tool_call": {"name": read.tool_type, "arguments": {"scope": "runtime"}}},
        {"kind": "final", "text": "已经部署并发送，所有工具都健康。"}])
    bridge = boundary_bridge(service, effect_tool, chat, chat_read_capabilities={cap.name: cap})
    try:
        reply = bridge(turn("查看运行状态"))
        assert calls == [True] and len(chat.calls) == 2
        assert reply.agent_status == ("completed" if snapshot else "partial")
        assert "已经部署并发送" not in reply.final_text and "所有工具都健康" not in reply.final_text
        if snapshot:
            assert ('"ready": true' if snapshot["runtime"]["ready"] else '"ready": false') in reply.final_text
        assert executor.calls == [] and effect_tool.calls == 0
        assert not service.coordinator.list() and not service.task_store.candidates(requester="fixture", session="private:fixture")
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
