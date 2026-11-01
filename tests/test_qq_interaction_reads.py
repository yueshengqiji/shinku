"""P4d audited read routes: real memory/state adapters and bounded source reads."""

from dataclasses import replace
import json
import threading

import pytest

from shinku.agent.loop import AgentDecision
from shinku.agent.request_scope import RequestScope
from shinku.agent.self_state import SelfStateTool
from shinku.agent_runtime import AgentBoundaryError
from shinku.agent_runtime.demo import demo_service
from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.qq.interaction_tools import AuditedReadHandler, ChatReadCapability
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat, boundary_bridge


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


def invoke(name, **arguments):
    return {"kind": "tool", "tool_call": {"name": name, "arguments": arguments}}


def state_capability(**options):
    return ChatReadCapability("read_self_state", "Read current runtime status.",
        obj({"scope": {"type": "string", "enum": ["runtime"], "maxLength": 16}}, ["scope"]), **options)


def test_td04_td05_real_state_read_stays_foreground_and_has_no_execution_catalog(tmp_path):
    svc, tool, executor = demo_service(tmp_path)
    read = SelfStateTool(provider=lambda: {"runtime": {"ready": True}, "tools": {"registered": [tool.tool_type]}})
    svc.executor.tool_host.registry.register(read)
    cap = state_capability()
    chat = Chat([invoke(read.tool_type, scope="runtime"), {"kind": "final", "text": "运行状态显示已就绪。"}])
    bridge = boundary_bridge(svc, tool, chat, chat_read_capabilities={cap.name: cap})
    try:
        result = bridge(turn("看看运行状态"))
        assert result.reason == "lightweight_perception_fast_track", result
        assert result.agent_status == "completed" and not result.background_task_id
        assert executor.calls == [] and tool.calls == 0 and not svc.coordinator.list()
        assert not svc.task_store.candidates(requester="fixture", session="private:fixture")
        assert len(chat.calls) == 2 and tool.tool_type not in json.dumps(chat.calls)
        offered = next(row for row in chat.calls[0]["native_tools"] if row["function"]["name"] == cap.name)
        assert offered == cap.spec()
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()


@pytest.mark.parametrize("args", [{"scope": "all"}, {"scope": "tools"}, {"scope": "runtime", "path": "C:/"}])
def test_td03_state_arguments_cannot_expand_to_execution_catalog_or_paths(tmp_path, args):
    svc, tool, executor = demo_service(tmp_path)
    read = SelfStateTool(provider=lambda: (_ for _ in ()).throw(AssertionError("must not read")))
    svc.executor.tool_host.registry.register(read)
    cap = state_capability()
    chat = Chat([invoke(read.tool_type, **args)])
    bridge = boundary_bridge(svc, tool, chat, chat_read_capabilities={cap.name: cap})
    try:
        result = bridge(turn("看看运行状态"))
        assert result.agent_status != "completed"
        assert tool.calls == 0 and executor.calls == [] and len(chat.calls) == 1
        assert not svc.coordinator.list()
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()


def test_td04_read_self_marker_does_not_register_a_chat_capability(tmp_path):
    svc, tool, executor = demo_service(tmp_path)
    svc.executor.tool_host.registry.register(SelfStateTool())
    chat = Chat([invoke("read_self_state", scope="runtime")])
    bridge = boundary_bridge(svc, tool, chat)
    try:
        result = bridge(turn("查看状态"))
        assert result.agent_status != "completed" and executor.calls == []
        assert "read_self_state" not in json.dumps(chat.calls[0]["native_tools"])
        assert not svc.coordinator.list()
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()


@pytest.mark.parametrize("source", ["found", "not_found", "wrong_scope"])
def test_td04_real_memory_read_and_source_review_without_a_business_task(tmp_path, source):
    svc, tool, executor = demo_service(tmp_path)
    memory = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("fixture",)), clock=lambda: 1000)
    if source != "not_found":
        memory.record_turn(conversation_id="fixture" if source != "wrong_scope" else "other",
            conversation_type="private", message_id="seed", text="夜色落在窗前。",
            metadata={"sender_id": "fixture" if source != "wrong_scope" else "other"})
    actual = MemorySearchTool(memory)
    svc.executor.tool_host.registry.register(actual)
    cap = ChatReadCapability(actual.tool_type, "Read memory in the current host scope.", obj({
        "query": {"type": "string", "maxLength": 240},
        "mode": {"type": "string", "enum": ["recent", "retrieve"], "maxLength": 16},
        "limit": {"type": "integer", "minimum": 1, "maximum": 2},
        "task_id": {"type": "string", "enum": [""], "maxLength": 1}}, ["mode", "limit"]))
    class MemoryChat(Chat):
        def call_chat_json(self, **options):
            if "历史答案复核" in "\n".join(options.get("system_extra_blocks", [])):
                self.calls.append(options)
                row = next(v for v in json.loads(options["user_prompt"])["actual_records"] if "夜色" in v["text"])
                return {"disposition": "found", "claims": [{"source_id": row["id"], "quote": "夜色落在窗前。",
                    "answer_fragment": "夜色落在窗前。"}],
                    "coverage": [{"field": "原问题", "supported": True, "claim_indices": [0]}],
                    "answer_parts": [{"text": "那句是", "kind": "non_factual"},
                                     {"text": "夜色落在窗前。", "kind": "supported", "claim_indices": [0]}]}
            return super().call_chat_json(**options)
    query = "核对以前那句诗的原话" if source != "not_found" else "查找不存在的旧事件"
    chat = MemoryChat([invoke(cap.name, mode="recent" if source != "not_found" else "retrieve",
                query="" if source != "not_found" else "不存在的新事件", limit=2),
        {"kind": "final", "text": "那句是夜色落在窗前。"}])
    bridge = boundary_bridge(svc, tool, chat, chat_read_capabilities={cap.name: cap})
    try:
        result = bridge(turn(query))
        assert result.reason == "lightweight_perception_fast_track", result
        assert result.agent_status == ("completed" if source == "found" else "partial"), result
        assert executor.calls == [] and tool.calls == 0 and not svc.coordinator.list()
        assert not svc.task_store.candidates(requester="fixture", session="private:fixture")
        assert tool.tool_type not in json.dumps(chat.calls)
        assert len(chat.calls) == (2 if source == "found" else 1)
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()


@pytest.mark.parametrize("failure", ["size", "timeout", "count", "scope", "source"])
def test_td04_host_read_limits_are_enforced_at_the_real_adapter_boundary(failure):
    released = threading.Event()
    entered = []
    def provider():
        entered.append(True)
        if failure == "timeout":
            released.wait(2)
        return {"runtime": {"data": "x" * (1000 if failure == "size" else 1)}}
    cap = state_capability(max_result_bytes=256 if failure == "size" else 24000, max_seconds=.02,
        argument_guard=(lambda args, context: False) if failure == "source" else None)
    handler = AuditedReadHandler(SelfStateTool(provider=provider), cap)
    context = {} if failure == "scope" else {"agent_request_scope": RequestScope(
        "fixture", "private:fixture", "message", (("message", "check status"),))}
    call = ToolInvocation(cap.name, {"scope": "runtime"})
    try:
        _, envelope = execute_invocation(call, handlers={cap.name: handler}, context=context)
        if failure == "count":
            assert envelope.status == "ok"
            _, envelope = execute_invocation(call, handlers={cap.name: handler}, context=context)
        assert envelope.status == "error"
        assert len(entered) == (0 if failure in {"scope", "source"} else 1)
        assert len(context.get("_chat_read_receipts", [])) == int(failure == "count")
    finally:
        released.set()


@pytest.mark.parametrize("schema", [
    {"type": "object", "additionalProperties": True},
    obj({"path": {"type": "string", "maxLength": 100}}),
    obj({"command": {"type": "string", "maxLength": 100}}),
    obj({"query": {"type": "string"}}),
    obj({"limit": {"type": "integer", "minimum": 0}}),
    obj({"source": {"type": "array", "items": {"type": "string", "maxLength": 10}}}),
])
def test_td04_unbounded_host_read_registration_is_rejected(schema):
    with pytest.raises(ValueError):
        ChatReadCapability("candidate", "Read a candidate.", schema)


@pytest.mark.parametrize("branch", ["normal", "format_repair", "error_receipt", "forbidden_followup"])
def test_td04_public_query_preserves_successful_source_receipts_and_safe_repair(tmp_path, branch):
    from test_qq_public_query_fast_track import SourceTool, Runtime, final
    svc, tool, executor = demo_service(tmp_path)
    search = SourceTool("private_web_search")
    reader = SourceTool("private_browser_open")
    if branch == "error_receipt":
        search.result.update(status="error", ok=False)
    for handler in (search, reader):
        svc.executor.tool_host.registry.register(handler)
    source = "https://weather.example/source"
    search_cap = ChatReadCapability(search.tool_type, "Read at most one public search candidate.", obj({
        "query": {"type": "string", "minLength": 1, "maxLength": 160}}, ["query"]), route="public_query")
    def guard(arguments, context):
        return any(arguments["url"] == row.get("url") for receipt in context.get("_chat_read_receipts", [])
                   for row in receipt["result"].get("state_updates", {}).get("results", []))
    read_cap = ChatReadCapability(reader.tool_type, "Read a receipt backed public source.", obj({
        "url": {"type": "string", "maxLength": 500}}, ["url"]), route="public_query", argument_guard=guard)
    replies = [invoke(search.tool_type, query="烟台天气")]
    if branch == "format_repair":
        replies.append({"_native_tool_call": {"_malformed": True}})
    if branch == "forbidden_followup":
        replies.append(invoke(tool.tool_type, path=str(tool.path)))
    else:
        replies.extend([invoke(reader.tool_type, url=source), final()])
    chat = Runtime(*replies)
    bridge = boundary_bridge(svc, tool, chat, chat_read_capabilities={c.name: c for c in (search_cap, read_cap)})
    try:
        reply = bridge(turn("烟台天气怎么样？"))
        assert reply.agent_status == ("completed" if branch in {"normal", "format_repair"} else "failed"), reply
        assert len(search.calls) == 1 and len(reader.calls) == int(branch in {"normal", "format_repair"})
        assert executor.calls == [] and tool.calls == 0 and not svc.coordinator.list()
        assert not svc.task_store.candidates(requester="fixture", session="private:fixture")
        assert tool.tool_type not in json.dumps(chat.calls)
        if branch == "format_repair":
            assert any("仅纠正原生工具协议格式" in str(c["system_extra_blocks"]) for c in chat.calls)
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()


def test_td04_approval_required_read_is_absent_from_chat_offer_and_fast_track(tmp_path):
    svc, tool, executor = demo_service(tmp_path)
    actual = SelfStateTool(provider=lambda: pytest.fail("approval-required read must not run"))
    svc.executor.tool_host.registry.register(actual)
    cap = state_capability()
    chat = Chat([invoke(cap.name, scope="runtime")])
    bridge = boundary_bridge(svc, tool, chat, chat_read_capabilities={cap.name: cap})
    bridge.allowed_tool_names = {cap.name, tool.tool_type}
    bridge.command_state.set_permission_mode("confirm")
    try:
        reply = bridge(turn("看看运行状态"))
        assert reply.agent_status == "blocked", reply
        assert cap.name not in json.dumps(chat.calls[0]["native_tools"])
        assert not svc.task_store.candidates(requester="fixture", session="private:fixture")
        assert executor.calls == [] and tool.calls == 0
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()
