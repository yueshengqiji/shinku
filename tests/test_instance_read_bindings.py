"""独立启动器的聊天只读声明：真实装配与适配调用，模型响应为合成输入。"""
from dataclasses import replace
import io
import os
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from shinku.api import create_app
from shinku.extensions import RuntimeExtension, RuntimeExtensionError
from shinku.instance import claim_instance, load_instance
from shinku.instance_runtime import bindings, terminal_loop
from shinku.qq.assembly import assemble_qq_agent, load_qq_agent_config
from shinku.qq.interaction_tools import ChatReadCapability
from test_instance_console import settings
from test_instance_launcher import manifest, close
from test_qq_owner_access import _turn
from test_qq_tool_decision_boundary import Chat


class Clock:
    tool_type = "fixture_clock"
    agent_permission_class = "read"
    agent_progress_class = "observe"
    agent_fast_perception = True
    agent_observation_required_fields = ("iso",)
    agent_observation_receipt_fields = ("iso",)
    calls = 0

    def normalize_call(self, call):
        return {"type": self.tool_type}

    def execute(self, *, call, context):
        self.calls += 1
        return {"tool_type": self.tool_type, "ok": True,
                "followup_context": "合成时间：2030-01-02T03:04:05+08:00",
                "state_updates": {"iso": "2030-01-02T03:04:05+08:00"}}


@pytest.mark.parametrize("channel", ["desktop", "terminal"])
def test_explicit_read_reaches_model_and_adapter_without_executor(tmp_path, monkeypatch, channel):
    actual = Clock()
    capability = ChatReadCapability(actual.tool_type, "Read fixture clock.",
        {"type": "object", "properties": {}, "additionalProperties": False})
    module = ModuleType("fixture_instance_read_extension")
    module.register_tools = lambda registry, config: registry.register(actual)
    module.build_chat_read_capabilities = lambda config: {capability.name: capability}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    clients = []

    class Client(Chat):
        def __init__(self, **kwargs):
            super().__init__([{"kind": "tool", "tool_call": {"name": capability.name, "arguments": {}}},
                              {"kind": "final", "text": "现在是合成时间 03:04。"}] if not clients else [])
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("shinku.llm.runtime.LLMRuntime", Client)
    spec = replace(load_instance(manifest(tmp_path)), extension_module=module.__name__)
    with claim_instance(spec), patch.dict(os.environ, spec.environment(), clear=True):
        config = replace(load_qq_agent_config(), memory_judge_enabled=False)
        with bindings(config) as host:
            assembly = assemble_qq_agent(config, **host.assembly_options())
            try:
                if channel == "desktop":
                    client = TestClient(create_app(settings(tmp_path), qq_agent_assembly=assembly))
                    result = client.post("/api/pet/chat", json={"text": "用工具查看当前时间"})
                    assert result.status_code == 200 and result.json()["ok"]
                    assert "03:04" in result.json()["reply"]
                else:
                    output = io.StringIO()
                    assert terminal_loop(assembly.bridge, input_stream=io.StringIO("用工具查看当前时间\n"),
                                         output_stream=output, error_stream=io.StringIO()) == 0
                    assert "03:04" in output.getvalue()
                offered = clients[0].calls[0]["native_tools"]
                assert next(v for v in offered if v["function"]["name"] == capability.name) == capability.spec()
                assert actual.calls == 1 and clients[1].calls == []
                assert not assembly.task_coordinator.list()
                assert assembly.bridge.trace_store.snapshot(limit=1)[0]["reason"] == "lightweight_perception_fast_track"
                # 同一个已登记只读能力不会绕过宿主账号边界。
                clients[0].replies.append({"kind": "final", "text": "你好。"})
                stranger = _turn("other", "请查时间", conversation_type="private")
                stranger = replace(stranger, messages=tuple(replace(m, raw_event={"self_id": spec.bot_id})
                                                           for m in stranger.messages))
                assembly.bridge.handle_turn(stranger)
                assert actual.calls == 1
                assert not clients[0].calls[-1].get("native_tools")
                # 显式工具集合仍有排除效力，不能因为目录非空而放开全部工具。
                assembly.bridge.allowed_tool_names = {"memory_search"}
                assert actual.tool_type not in assembly.bridge._tool_names_for_context(
                    {"audience_context": {"is_master": True}})
            finally:
                close(assembly)
    assert all(client.closed for client in clients)


def test_extension_read_declaration_is_opt_in_and_errors_are_sanitized():
    module = ModuleType("fixture")
    extension = RuntimeExtension("fixture", module)
    assert extension.build_chat_read_capabilities(config=None) == {}
    module.build_chat_read_capabilities = 3
    with pytest.raises(RuntimeExtensionError, match="not_callable"):
        extension.build_chat_read_capabilities(config=None)
    def broken(config):
        raise ValueError("fixture credential must not appear")
    module.build_chat_read_capabilities = broken
    with pytest.raises(RuntimeExtensionError) as error:
        extension.build_chat_read_capabilities(config=None)
    assert str(error.value) == "runtime_extension_read_capabilities_failed:ValueError"


def test_invalid_read_declaration_cannot_become_a_host_binding(tmp_path, monkeypatch):
    module = ModuleType("fixture_invalid_read_extension")
    module.build_chat_read_capabilities = lambda config: {"unaudited": object()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = replace(load_instance(manifest(tmp_path)), extension_module=module.__name__)
    with patch.dict(os.environ, spec.environment(), clear=True):
        config = load_qq_agent_config()
        # 构造实际 SDK 客户端但禁止任何连接；错误声明必须在启动阶段拒绝。
        monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("network forbidden"))
        with pytest.raises(ValueError, match="host_bindings_read_capabilities_invalid"):
            with bindings(config):
                pytest.fail("invalid declaration accepted")
