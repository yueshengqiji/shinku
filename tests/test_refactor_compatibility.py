"""Integration boundaries that must survive responsibility extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, get_type_hints
from unittest.mock import patch

from shinku.agent.verification import SmartCompletionVerifier
from shinku.qq.assembly import QQAgentConfig, assemble_qq_agent


class RefactorCompatibilityTests(unittest.TestCase):
    def test_assembled_self_state_keeps_sender_and_live_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            persona = root / "persona.md"
            persona.write_text("隔离测试人设", encoding="utf-8")
            config = QQAgentConfig(
                enabled=True,
                send_enabled=True,
                persona_file=str(persona),
                chat_api_key="test-only",
                chat_base_url="http://invalid.invalid/v1",
                chat_model_name="before",
                memory_enabled=False,
                command_state_path=str(root / "commands.json"),
                agent_task_store_path=str(root / "background.sqlite3"),
                agent_state_store_path=str(root / "tasks.sqlite3"),
                conversation_state_store_path=str(root / "conversation.sqlite3"),
                trace_store_path=str(root / "traces.json"),
            )
            runtime = SimpleNamespace(
                chat=SimpleNamespace(model="before"),
                call_chat_json=lambda **kwargs: self.fail("unexpected model call"),
            )
            sender = lambda **kwargs: self.fail("unexpected QQ send")
            assembly = assemble_qq_agent(config, runtime=runtime, sender=sender)
            runtime.chat = SimpleNamespace(model="after")
            tool = assembly.bridge.tool_host.registry.get("read_self_state")
            result = tool.execute(call={"scope": "all"}, context={})
            state = result["state_updates"]["self_state"]
            self.assertEqual(state["runtime"]["model"], "after")
            self.assertTrue(state["agent"]["sender_configured"])
            self.assertIn("read_self_state", state["tools"]["registered"])

    def test_verifier_target_helpers_remain_replaceable_at_the_facade(self) -> None:
        self.assertEqual(
            SmartCompletionVerifier._extract_targets("打开 https://example.invalid/a"),
            ["https://example.invalid/a"],
        )
        with patch.object(SmartCompletionVerifier, "_valid_http_target", return_value=False):
            self.assertEqual(
                SmartCompletionVerifier._extract_targets("打开 https://example.invalid/a"),
                [],
            )

        class NoFileTargets(SmartCompletionVerifier):
            @staticmethod
            def _is_file_mention(*args, **kwargs):
                return False

        self.assertTrue(SmartCompletionVerifier._explicit_target_roles(["输出：report.md"]))
        self.assertEqual(NoFileTargets._explicit_target_roles(["输出：report.md"]), {})

    def test_public_verifier_exports_resolve_to_the_facade(self) -> None:
        """拆分后的公开导出必须指向实际定义，不能遗留在规则模块。"""
        from shinku.agent import verification
        from shinku.agent.completion import CompletionVerdict

        self.assertEqual(
            {name: getattr(verification, name) for name in verification.__all__},
            {"SmartCompletionVerifier": SmartCompletionVerifier, "CompletionVerdict": CompletionVerdict},
        )

    def test_assembly_owner_gate_annotations_resolve(self) -> None:
        """延迟注解仍须能解析，不能因拆分丢失 Sequence 导入。"""
        from shinku.qq.assembly_config import _validate_private_owner_gate

        self.assertEqual(get_type_hints(_validate_private_owner_gate)["owner_ids"], Sequence[str])
