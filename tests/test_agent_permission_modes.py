from __future__ import annotations

from types import SimpleNamespace
import unittest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandInterpreter, QQCommandState
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry
from shinku.hosts.tool_host import ToolHost


class _Runtime:
    def call_chat_json(self, **kwargs):
        del kwargs
        return {"speech": "完成。"}


class _ReadHandler:
    tool_type = "read_data"
    agent_permission_class = "read"
    agent_progress_class = "observe"

    def tool_metadata(self):
        return SimpleNamespace(description="只读测试工具", input_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    def normalize_call(self, call):
        return dict(call) if isinstance(call, dict) else None

    def execute(self, *, call, context):
        del context
        return {"tool_type": self.tool_type, "followup_context": str(call.get("query") or "")}


class _WriteHandler:
    tool_type = "write_data"
    agent_permission_class = "change"
    agent_progress_class = "change"
    agent_auto_confirm_in_full_access = True

    def tool_metadata(self):
        return SimpleNamespace(
            description="写入测试工具；需要 confirm=true",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "confirm": {"type": "boolean"}},
                "required": ["path", "confirm"],
            },
        )

    def normalize_call(self, call):
        if not isinstance(call, dict) or not call.get("path") or call.get("confirm") is not True:
            return None
        return dict(call)

    def execute(self, *, call, context):
        del context
        return {"tool_type": self.tool_type, "followup_context": f"写入 {call['path']}"}


class _UltraHandler(_WriteHandler):
    tool_type = "private_shell"
    agent_ultra_only = True


def _bridge(*, state: QQCommandState | None = None) -> QQAgentBridge:
    handlers = {
        "read_data": _ReadHandler(),
        "write_data": _WriteHandler(),
        "private_shell": _UltraHandler(),
    }
    return QQAgentBridge(
        runtime=_Runtime(),
        system_prompt="persona",
        tool_host=ToolHost(ToolRegistry(handlers)),
        allowed_tool_names={"read_data"},
        policy=ExecutionPolicy(risk_by_tool={"read_data": "low", "write_data": "high", "private_shell": "high"}),
        command_state=state or QQCommandState(),
    )


class AgentPermissionModeTests(unittest.TestCase):
    def test_permission_mode_command_is_persisted_and_does_not_enable_agent(self) -> None:
        state = QQCommandState(agent_enabled=False)
        interpreter = QQCommandInterpreter(state)

        result = interpreter.execute("/agent permission full", sender_id="master")

        self.assertEqual(result.status, "agent_permission_full")
        self.assertEqual(state.permission_mode(), "full")
        self.assertFalse(state.is_agent_enabled())
        self.assertIn("/agent permission traditional", result.reply)

    def test_permission_mode_command_is_owner_only_when_private_gate_is_enabled(self) -> None:
        state = QQCommandState(owner_ids=("master",))
        interpreter = QQCommandInterpreter(state, require_owner=True)

        result = interpreter.execute("/agent permission ultra", sender_id="other")

        self.assertTrue(result.quiet)
        self.assertEqual(state.permission_mode(), "traditional")

    def test_permission_mode_persists_and_invalid_saved_value_fails_closed(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "command_state.json"
            state = QQCommandState(state_path=path)
            self.assertTrue(state.set_permission_mode("full"))
            self.assertEqual(QQCommandState(state_path=path).permission_mode(), "full")
            path.write_text(json.dumps({"version": 1, "permission_mode": "unbounded"}), encoding="utf-8")
            self.assertEqual(QQCommandState(state_path=path).permission_mode(), "traditional")

    def test_permission_mode_reverts_in_memory_when_persistence_fails(self) -> None:
        state = QQCommandState()
        state._save = lambda: False  # type: ignore[method-assign]

        self.assertFalse(state.set_permission_mode("ultra"))
        self.assertEqual(state.permission_mode(), "traditional")

    def test_mode_policy_and_tool_visibility_are_distinct(self) -> None:
        bridge = _bridge()
        state = bridge.command_state

        self.assertEqual(bridge._tool_names_for_context(None), {"read_data"})
        self.assertTrue(bridge._policy_for_permission_mode("traditional").requires_approval("write_data"))

        state.set_permission_mode("readonly")
        self.assertEqual(bridge._tool_names_for_context(None), {"read_data"})
        readonly = bridge._policy_for_permission_mode("readonly")
        self.assertTrue(readonly.accepts("read_data"))
        self.assertFalse(readonly.accepts("write_data"))
        self.assertFalse(readonly.requires_approval("read_data"))

        state.set_permission_mode("confirm")
        confirm = bridge._policy_for_permission_mode("confirm")
        self.assertTrue(confirm.requires_approval("read_data"))
        self.assertTrue(confirm.requires_approval("write_data"))

        state.set_permission_mode("full")
        self.assertEqual(bridge._tool_names_for_context(None), {"read_data", "write_data"})
        full = bridge._policy_for_permission_mode("full")
        self.assertFalse(full.requires_approval("write_data"))
        self.assertTrue(full.accepts("write_data"))
        self.assertFalse(full.accepts("private_shell"))

        state.set_permission_mode("ultra")
        self.assertEqual(bridge._tool_names_for_context(None), {"read_data", "write_data", "private_shell"})
        ultra = bridge._policy_for_permission_mode("ultra")
        self.assertFalse(ultra.requires_approval("private_shell"))
        self.assertTrue(ultra.accepts("private_shell"))

    def test_full_mode_removes_authorization_only_confirm_parameter_from_model_schema(self) -> None:
        bridge = _bridge()
        specs = bridge.tool_host.registry.native_specs(allowed_tool_names={"write_data"})

        transformed = bridge._tool_specs_for_permission_mode(specs, "full")
        parameters = transformed[0]["function"]["parameters"]

        self.assertNotIn("confirm", parameters["properties"])
        self.assertEqual(parameters.get("required", []), ["path"])

    def test_full_mode_schema_transform_reaches_agent_followup_planner(self) -> None:
        bridge = _bridge()
        loop = bridge.tool_host.build_loop(
            runtime=_Runtime(),
            system_prompt="test",
            allowed_tool_names={"write_data"},
            tool_specs_transform=lambda specs: bridge._tool_specs_for_permission_mode(specs, "full"),
        )

        parameters = loop.planner.tool_specs[0]["function"]["parameters"]
        self.assertNotIn("confirm", parameters["properties"])
        self.assertEqual(parameters.get("required", []), ["path"])

    def test_full_mode_auto_confirms_only_marked_private_handler(self) -> None:
        handler = _WriteHandler()
        invocation = ToolInvocation("write_data", {"path": "C:/tmp/out.txt"})
        _, envelope = execute_invocation(
            invocation,
            handlers={"write_data": handler},
            context={"agent_permission_mode": "full"},
        )
        self.assertEqual(envelope.status, "ok")

        _, traditional_envelope = execute_invocation(
            invocation,
            handlers={"write_data": handler},
            context={"agent_permission_mode": "traditional"},
        )
        self.assertEqual(traditional_envelope.status, "error")


if __name__ == "__main__":
    unittest.main()
