"""真实 CLI/QQ 公共装配与工具执行的插件集成；模型为合成响应，禁止实际网络发送。"""
from contextlib import contextmanager
from dataclasses import replace
import json
from types import ModuleType

from fastapi.testclient import TestClient
import pytest

from shinku import cli
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.extensions import RuntimeExtension
from shinku.plugins import PluginContext, PluginContributions, PluginDefinition, PluginManager, PluginManifest
from shinku.plugins.examples.local_clock import LocalClockPlugin, LocalClockTool
from shinku.plugins.host import with_plugins
from shinku.qq.assembly import QQAgentConfigError, assemble_qq_agent, load_qq_agent_config
from shinku.qq.host_bindings import QQAgentHostBindings
from shinku.qq.interaction_tools import ChatReadCapability
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry

from test_agent_host_bindings import isolated, capture_real_assembly, close_assembly
from test_plugin_manager import Resource, definition, states
from test_qq_tool_decision_boundary import Chat
from test_qq_owner_access import _turn


@pytest.mark.parametrize("text", [
    "请查一下上海当前时间。",
    '不要取消任务，“取消”只是引用，请查一下上海时间。',
    "聊天改写里出现了写文件，顺便查一下上海时间。",
    "没有找到异常。请再查一下上海时间。",
])
def test_plugin_reaches_real_cli_and_chat_read_without_extra_owner_gate(isolated, tmp_path, monkeypatch, text):
    # HTTP 桌面入口原本要求单一身份；补齐真实入口的合成绑定，不放宽其已有规则。
    monkeypatch.setenv("SHINKU_MEMORY_MASTER_USER_IDS", "fixture-owner")
    monkeypatch.setenv("SHINKU_QQ_BOT_IDS", "fixture-bot")
    chat = Chat([{"kind": "tool", "tool_call": {"name": "local_clock", "arguments": {"timezone": "Asia/Shanghai"}}},
                 {"kind": "final", "text": "已经查看了上海当前时间。"}])
    execution = Chat([])
    seen, managers = capture_real_assembly(monkeypatch), []
    events = []
    class Clock(LocalClockPlugin):
        def close(self):
            events.append("closed")

    @contextmanager
    def factory(config):
        # 插件适配器仍是公共宿主，不引入旧私用扩展专属门禁。
        with PluginManager(PluginContext("shinku", "fixture", config=config), [
            PluginDefinition(PluginManifest("clock"), Clock),
        ]) as manager:
            managers.append(manager)
            assert not chat.calls and not execution.calls
            yield QQAgentHostBindings(chat, ExecutionProfile(execution), runtime_extension=manager.runtime_extension())

    def serve(app, **kwargs):
        assert states(managers[0])["clock"]["status"] == "active"
        response = TestClient(app).post("/api/pet/chat", json={"text": text})
        assert response.status_code == 200 and response.json()["ok"], response.text
        assert "当前时间" in response.json()["reply"]
        bridge = seen[0].bridge
        assert not bridge.owner_gate_required
        assert bridge.sender is None and execution.calls == []
        assert text in json.dumps(chat.calls[0], ensure_ascii=False)
        assert "local_clock" in {tool["function"]["name"] for tool in chat.calls[0]["native_tools"]}
        assert "lightweight_perception_fast_track" == bridge.trace_store.snapshot(limit=1)[0]["reason"]
    monkeypatch.setattr("uvicorn.run", serve)
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("external connections forbidden"))
    try:
        assert cli.main(["serve"], agent_bindings=factory) == 0
        assert events == ["closed"]
        assert seen[0].bridge.tool_host.registry.get("local_clock") is None
    finally:
        for assembly in seen:
            close_assembly(assembly)


@pytest.mark.parametrize("failure", ["assembly", "server"])
def test_cli_failure_unwinds_plugin_resources(isolated, monkeypatch, failure):
    resource = Resource()
    seen = capture_real_assembly(monkeypatch)
    @contextmanager
    def factory(config):
        with PluginManager(PluginContext("shinku", config=config), [definition(factory=lambda: resource)]) as manager:
            yield QQAgentHostBindings(Chat([]), ExecutionProfile(Chat([])), runtime_extension=manager.runtime_extension())
    if failure == "assembly":
        def reject(*args, **kwargs):
            raise QQAgentConfigError("fixture_rejection")
        monkeypatch.setattr(cli, "assemble_qq_agent", reject)
    def serve(*args, **kwargs):
        raise RuntimeError("fixture_server_error")
    monkeypatch.setattr("uvicorn.run", serve)
    try:
        if failure == "assembly":
            assert cli.main(["serve"], agent_bindings=factory) == 2
        else:
            with pytest.raises(RuntimeError, match="fixture_server_error"):
                cli.main(["serve"], agent_bindings=factory)
        assert resource.closed == 1
    finally:
        for assembly in seen:
            close_assembly(assembly)


def test_legacy_private_owner_requirement_is_preserved(isolated, monkeypatch):
    legacy = RuntimeExtension("legacy", ModuleType("legacy"))
    config = load_qq_agent_config()
    with PluginManager(PluginContext("shinku"), []) as manager:
        managed = manager.runtime_extension(legacy=legacy)
        assert managed.owner_gate_required
        with pytest.raises(QQAgentConfigError, match="private_master_user_ids_missing"):
            assemble_qq_agent(config, runtime=Chat([]), runtime_extension=managed)


def test_public_plugin_preserves_assembly_without_private_owner_configuration(isolated):
    config = load_qq_agent_config()
    with PluginManager(PluginContext("shinku"), [PluginDefinition(PluginManifest("clock"), LocalClockPlugin)]) as manager:
        for extension in (None, manager.runtime_extension()):
            assembly = assemble_qq_agent(config, runtime=Chat([]), runtime_extension=extension)
            try:
                assert not assembly.bridge.owner_gate_required
                assert not assembly.bridge.master_user_ids and assembly.bridge.sender is None
            finally:
                close_assembly(assembly)


@pytest.mark.parametrize("blocked", [False, True])
def test_registered_plugin_uses_existing_execution_policy(blocked):
    registry = ToolRegistry()
    with PluginManager(PluginContext("angelos"), [definition()]) as manager:
        manager.register_tools(registry)
        policy = ExecutionPolicy(blocked=frozenset({"fixture_echo"}) if blocked else frozenset())
        result, envelope = execute_invocation(ToolInvocation("fixture_echo", {"text": "没有找到异常，不要取消"}),
            handlers=registry.view(), context={"permission_mode": "ultra"}, policy=policy)
        if blocked:
            assert result is None and envelope.executed is False
        else:
            assert result["call"]["text"] == "没有找到异常，不要取消" and envelope.executed


def test_read_name_conflict_retracts_only_plugin_tools_and_keeps_host_read():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    read = ChatReadCapability("clock_read", "宿主的合成只读项", schema)
    class Reads(Resource):
        def activate(self, context):
            return PluginContributions(tools={"clock_tool": LocalClockTool()}, chat_read_capabilities={read.name: read})
    registry = ToolRegistry()
    with PluginManager(PluginContext("shinku"), [definition(factory=Reads)]) as manager:
        extension = manager.runtime_extension()
        extension.register_tools(registry=registry, config=None)
        assert manager.read_capabilities({read.name: read}) == {read.name: read}
        assert registry.names() == () and states(manager)["sample"]["status"] == "failed"


def test_host_binding_default_has_no_plugin_opt_in_and_rejects_wrong_dependency():
    host = QQAgentHostBindings(Chat([]), ExecutionProfile(Chat([])))
    assert "runtime_extension" not in host.assembly_options()
    with pytest.raises(ValueError, match="runtime_extension_required"):
        replace(host, runtime_extension=object())


@pytest.mark.parametrize("config_mode", ["valid", "missing", "malformed"])
def test_existing_host_factory_can_load_selected_json_without_per_plugin_core_changes(isolated, monkeypatch, tmp_path, config_mode):
    order = []
    @contextmanager
    def original(config):
        order.append("base_enter")
        try:
            yield QQAgentHostBindings(Chat([{"kind": "final", "text": "普通聊天照常回复。"}]), ExecutionProfile(Chat([])))
        finally:
            order.append("base_exit")
    original.delegation_enabled = True
    config_path = tmp_path / "plugins.json"
    if config_mode == "valid":
        config_path.write_text(json.dumps({"plugins": [
            {"id": "broken", "factory": "missing_optional_fixture:create"},
            {"id": "clock", "factory": "shinku.plugins.examples.local_clock:create"},
        ]}))
    elif config_mode == "malformed":
        config_path.write_text("invalid JSON")
    seen = capture_real_assembly(monkeypatch)
    factory = with_plugins(original, config_path, instance_id="fixture")
    assert factory.delegation_enabled is True
    def serve(app, **kwargs):
        order.append("serve")
        extension = seen[0].extension
        if config_mode == "valid":
            assert states(extension.manager)["broken"]["status"] == "failed"
            assert seen[0].bridge.tool_host.registry.get("local_clock") is not None
        else:
            assert states(extension.manager)["plugin-configuration"]["status"] == "failed"
        reply = seen[0].bridge.handle_turn(_turn("fixture", "你好，普通聊天。"))
        assert reply.final_text == "普通聊天照常回复。" and reply.agent_status == "completed"
    monkeypatch.setattr("uvicorn.run", serve)
    try:
        assert cli.main(["serve"], agent_bindings=factory) == 0
        assert order == ["base_enter", "serve", "base_exit"]
        assert seen[0].extension.manager.diagnostics()["closed"]
        assert seen[0].bridge.tool_host.registry.get("local_clock") is None
    finally:
        for assembly in seen:
            close_assembly(assembly)
