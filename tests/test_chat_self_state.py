"""聊天自查回归：真实装配、SDK 和只读适配器，状态必须属于当前聊天宿主。"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.agent.self_state import SelfStateTool
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.hosts.tool_host import ToolHost
from shinku.qq import assembly_tools
from shinku.qq.assembly import assemble_qq_agent, load_qq_agent_config
from shinku.qq.delegation import DelegationConnection
from shinku.qq.interaction_tools import AuditedReadHandler, ChatReadCapability
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry
from test_agent_conversation_loop import native_sdk
from test_agent_delegation import remote, SECRET, SCOPE
from test_agent_host_bindings import isolated, close_assembly
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat


@pytest.fixture
def assembled(isolated, monkeypatch):
    monkeypatch.setenv("SHINKU_MEMORY_MASTER_USER_IDS", "fixture")
    resources = []
    def build(runtime, **options):
        if not options.get("delegation"):
            options["execution_profile"] = ExecutionProfile(SimpleNamespace(
                call_chat_json=lambda **kwargs: pytest.fail("self-check must not start execution")))
        assembly = assemble_qq_agent(load_qq_agent_config(), runtime=runtime,
                                     tool_decision_mode="task", **options)
        resources.append(assembly)
        assembly.bridge.command_state.set_permission_mode("readonly")
        return assembly.bridge
    yield build
    for assembly in resources:
        if assembly.bridge.delegation:
            assembly.bridge.delegation.close()
        close_assembly(assembly)


def test_real_assembly_offers_its_own_self_state_without_an_explicit_host_catalog(assembled):
    runtime = SimpleNamespace(chat=SimpleNamespace(model="chat-fixture", provider="fixture"),
                              call_chat_json=lambda **kwargs: pytest.fail("assembly must not call the model"))
    bridge = assembled(runtime)
    assert "read_self_state" in bridge.chat_read_capabilities
    handler = bridge.tool_host.registry.get("read_self_state")
    assert type(handler) is SelfStateTool and handler.provider is bridge.self_state_provider
    assert bridge.chat_read_capabilities["read_self_state"].max_calls == 1


@pytest.mark.parametrize("delegated", [False, True])
def test_real_sdk_self_check_stays_local_even_with_angelos_connected(assembled, request, delegated):
    connection = None
    if delegated:
        service, tool, executor, gateway, client, base = request.getfixturevalue("remote")
        connection = DelegationConnection(client.endpoint, "shinku", SECRET, wait_seconds=5)
    def respond(payload, index):
        if index == 1:
            offered = {s["function"]["name"]: s["function"] for s in payload["tools"]}
            assert "read_self_state" in offered
            assert "当前聊天实例" in offered["read_self_state"]["description"]
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "local-state",
                "type": "function", "function": {"name": "read_self_state",
                "arguments": json.dumps({"scope": "runtime"})}}]}, "tool_calls"
        assert index == 2
        # 模型候选不拥有状态事实；最终正文仍由原有读取回执绑定。
        return {"role": "assistant", "content": json.dumps({"kind": "final", "text": "当前模型已读取。"})}, "stop"
    with native_sdk(respond) as (runtime, calls):
        runtime.chat = replace(runtime.chat, model="shinku-fixture-model")
        runtime.aux = runtime.chat
        previous_env = runtime._env
        runtime._env = lambda key, default="": (
            "offline.example.invalid:shinku-fixture-model" if key == "NATIVE_TOOL_PROVIDER_ALLOWLIST"
            else previous_env(key, default))
        bridge = assembled(runtime, delegation=connection)
        result = bridge(turn("你当前使用什么模型？请用自己的状态工具核对。"))
        assert result.agent_status == "completed", result
        assert result.reason == "lightweight_perception_fast_track"
        assert "shinku-fixture-model" in result.final_text and "shinku_chat" in result.final_text
        assert len(calls) == 2 and not result.background_task_id and not result.sent
        assert not bridge.task_coordinator.list()
        assert not bridge.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)
        if delegated:
            assert not bridge.delegation._rows(SCOPE)
            assert not service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)
            assert executor.calls == [] and tool.calls == 0


@pytest.mark.parametrize("barrier", ["non_owner", "disabled", "approval", "blocked", "allowlist", "revoked"])
def test_real_assembly_self_read_keeps_current_authority_gates(assembled, barrier):
    chat = Chat([{"kind": "tool", "tool_call": {"name": "read_self_state", "arguments": {"scope": "runtime"}}},
                 {"kind": "final", "text": "没有可用的自查结果。"}])
    bridge = assembled(chat)
    handler = bridge.tool_host.registry.get("read_self_state")
    reads = []
    original = handler.fresh_provider
    handler.fresh_provider = lambda: reads.append(True) or original()
    if barrier == "disabled":
        bridge.command_state.set_agent_enabled(False)
    elif barrier == "approval":
        bridge.command_state.set_permission_mode("confirm")
    elif barrier == "blocked":
        bridge.policy = replace(bridge.policy, blocked=frozenset({"read_self_state"}))
    elif barrier == "allowlist":
        bridge.command_state.set_permission_mode("traditional")
        bridge.allowed_tool_names = {"another_tool"}
        assert not bridge.policy.requires_approval("read_self_state")
    elif barrier == "revoked":
        original_call = chat.call_chat_json
        def revoke(**kwargs):
            result = original_call(**kwargs)
            bridge.command_state.set_agent_enabled(False)
            return result
        chat.call_chat_json = revoke
    result = bridge(turn("请查看当前模型。", user="stranger" if barrier == "non_owner" else "fixture"))
    assert result.agent_status != "completed", result
    assert not reads and not result.background_task_id
    assert not bridge.task_coordinator.list()
    if chat.calls and barrier != "revoked":
        assert "read_self_state" not in json.dumps(chat.calls[0].get("native_tools") or [])


@pytest.mark.parametrize("kind", ["foreign_provider", "subclass", "no_fresh_provider", "missing"])
def test_automatic_grant_requires_the_assembly_owned_builtin(kind):
    provider = lambda: {"runtime": {"model": "fixture"}}
    class CustomState(SelfStateTool):
        pass
    handler = (CustomState if kind == "subclass" else SelfStateTool)(
        (lambda: {}) if kind == "foreign_provider" else provider,
        fresh_provider=None if kind == "no_fresh_provider" else provider)
    host = ToolHost(ToolRegistry({} if kind == "missing" else {"read_self_state": handler}))
    assert assembly_tools.self_state_read_capabilities(None, provider, host) == {}


def test_explicit_host_scope_is_preserved(assembled):
    cap = ChatReadCapability("read_self_state", "Only runtime.", {
        "type": "object", "properties": {"scope": {"type": "string", "enum": ["runtime"], "maxLength": 16}},
        "required": ["scope"], "additionalProperties": False})
    bridge = assembled(Chat([]), chat_read_capabilities={cap.name: cap})
    assert bridge.chat_read_capabilities == {cap.name: cap}


@pytest.mark.parametrize("arguments", [{}, {"scope": "files"}, {"scope": "runtime", "target": "Angelos"},
    {"scope": "runtime", "path": "/synthetic"}, {"scope": "runtime", "command": "inspect"}])
def test_builtin_chat_capability_rejects_expanded_targets_and_arguments(assembled, arguments):
    bridge = assembled(Chat([]))
    cap = bridge.chat_read_capabilities["read_self_state"]
    handler = bridge.tool_host.registry.get("read_self_state")
    assert not cap.admits(arguments, {})
    assert AuditedReadHandler(handler, cap).normalize_call(arguments) is None


def test_builtin_chat_read_is_fresh_scoped_and_limited_to_one_call(assembled):
    runtime = SimpleNamespace(chat=SimpleNamespace(model="before", provider="fixture"),
                              call_chat_json=lambda **kwargs: pytest.fail("adapter must not call the model"))
    bridge = assembled(runtime)
    runtime.chat.model = "after"
    handler = bridge.tool_host.registry.get("read_self_state")
    cap = bridge.chat_read_capabilities["read_self_state"]
    audited = AuditedReadHandler(handler, cap)
    context = {"agent_request_scope": RequestScope("fixture", "private:fixture", "check", (("check", "model"),)),
               "self_state": {"runtime": {"model": "stale"}}}
    invocation = ToolInvocation("read_self_state", {"scope": "runtime"})
    _, first = execute_invocation(invocation, handlers={cap.name: audited}, context=context)
    updates = first.data["state_updates"]
    assert updates["self_state"] == {"runtime": {"instance": "shinku_chat", "model": "after", "provider": "fixture"}}
    assert updates["self_state_receipt"]["fresh"] and updates["result_valid"]
    _, second = execute_invocation(invocation, handlers={cap.name: audited}, context=context)
    assert second.status == "error" and len(context["_chat_read_receipts"]) == 1


def test_chat_provider_failure_does_not_certify_cached_state(assembled):
    chat = Chat([{"kind": "tool", "tool_call": {"name": "read_self_state", "arguments": {"scope": "runtime"}}},
                 {"kind": "final", "text": "已确认当前模型。"}])
    bridge = assembled(chat)
    def fail():
        raise RuntimeError("synthetic snapshot failure")
    bridge.tool_host.registry.get("read_self_state").fresh_provider = fail
    result = bridge(turn("核实自己的当前模型。"))
    assert result.agent_status == "partial", result
    assert "不能确认完整答案" in result.final_text and "已确认当前模型" not in result.final_text
    assert not bridge.task_coordinator.list()
