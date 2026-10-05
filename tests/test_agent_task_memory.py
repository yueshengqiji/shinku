from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


class EmptyRuntime:
    def call_chat_json(self, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "final", "text": ""}


def _private_turn(text: str, *, user_id: str = "master-1", message_id: str = "m-1") -> Any:
    from shinku.qq.message import IncomingMessage
    from shinku.qq.turns import QQTurn

    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": user_id,
        "message_id": message_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    message = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id=user_id,
        conversation_type="private",
        primary_message_id=message_id,
        message_ids=(message_id,),
        combined_text=text,
        messages=(message,),
    )


class AgentTaskMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        self.memory = MemoryService(
            store=self.store,
            router=MemoryRouter(),
            policy=MemoryPolicy(master_user_ids=("master-1",)),
            clock=lambda: 2_000_000_000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_terminal_owner_task_is_indexed_durably_and_idempotently(self) -> None:
        metadata = {"sender_id": "master-1", "shinku_involved": True}
        args = {
            "conversation_id": "master-1",
            "conversation_type": "private",
            "requester_id": "master-1",
            "task_id": "qq:master-1:m-1",
            "goal": "排查私用 Agent 的依赖安装失败",
            "status": "blocked",
            "outcome": "确认是代理连接失败，尚未完成安装。",
            "reason": "network_timeout",
            "rounds": 4,
            "tool_outcomes": [{"name": "shell", "status": "error"}],
            "metadata": metadata,
        }

        self.assertTrue(self.memory.record_agent_task_summary(**args))
        self.assertFalse(self.memory.record_agent_task_summary(**args))

        matches = self.store.search(
            query="依赖安装失败",
            scope=MemoryService.OWNER_SCOPE,
            limit=5,
            now=2_000_000_000 + 400 * 86400,
            layers={"episodic"},
        )
        task_records = [item for item in matches if item.metadata.get("record_type") == "agent_task_summary"]
        self.assertEqual(len(task_records), 1)
        self.assertIn("尚未完成安装", task_records[0].text)
        self.assertEqual(task_records[0].metadata["task_status"], "blocked")
        self.assertEqual(task_records[0].metadata["task_id"], "qq:master-1:m-1")

    def test_non_owner_task_never_enters_owner_memory(self) -> None:
        stored = self.memory.record_agent_task_summary(
            conversation_id="group-1",
            conversation_type="group",
            requester_id="member-2",
            task_id="qq:group-1:m-2",
            goal="查询某个群成员的配置",
            status="completed",
            outcome="查询完成。",
            tool_outcomes=[{"name": "read_config", "status": "ok"}],
            metadata={"sender_id": "member-2", "shinku_involved": True, "at_targets": ["bot-9"]},
        )

        self.assertFalse(stored)
        self.assertEqual(
            len(
                self.store.timeline(
                    scope=MemoryService.OWNER_SCOPE,
                    limit=10,
                    now=2_000_000_000,
                    layers=("episodic",),
                    include_facts=False,
                )
            ),
            0,
        )

    def test_summary_redacts_labeled_credentials(self) -> None:
        stored = self.memory.record_agent_task_summary(
            conversation_id="master-1",
            conversation_type="private",
            requester_id="master-1",
            task_id="qq:master-1:secret-test",
            goal="检查密钥配置",
            status="completed",
            outcome="配置已保存，API key=do-not-store-this-value",
            tool_outcomes=[{"name": "config_read", "status": "ok"}],
            metadata={"sender_id": "master-1", "shinku_involved": True},
        )

        self.assertTrue(stored)
        summary = next(
            item
            for item in self.store.timeline(
                scope=MemoryService.OWNER_SCOPE,
                limit=10,
                now=2_000_000_000,
                layers=("episodic",),
                include_facts=False,
            )
            if item.metadata.get("record_type") == "agent_task_summary"
        )
        self.assertNotIn("do-not-store-this-value", summary.text)
        self.assertIn("[已省略]", summary.text)

    def test_bridge_records_summary_and_safe_detail_without_raw_tool_payload(self) -> None:
        try:
            from shinku.qq.agent_bridge import QQAgentBridge
        except ImportError as exc:
            self.skipTest(f"QQ bridge dependencies are unavailable: {type(exc).__name__}")

        bridge = QQAgentBridge(
            runtime=EmptyRuntime(),
            system_prompt="test",
            memory_service=self.memory,
            master_user_ids=("master-1",),
        )
        turn = _private_turn("帮我检查记忆检索为何漏掉昨天的 Agent 任务")
        result = SimpleNamespace(
            pending_approval=None,
            status="completed",
            final_text="原因是历史查询只匹配了旧任务的标题；我已记录修复结论。",
            reason="",
            rounds=3,
            transcript=(
                {
                    "role": "assistant",
                    "kind": "tool",
                    "tool_calls": [
                        {
                            "id": "call-memory-1",
                            "name": "memory_search",
                            "arguments": {"api_key": "DO_NOT_ARCHIVE_THIS"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "memory_search",
                    "invocation_id": "call-memory-1",
                    "content": "PRIVATE_RAW_TOOL_RESULT",
                },
            ),
            tool_envelopes=(
                SimpleNamespace(
                    invocation_id="call-memory-1",
                    status="ok",
                    data={"tool_type": "memory_search", "followup_context": "PRIVATE_RAW_TOOL_RESULT"},
                ),
            ),
        )

        bridge._finish_result(
            turn,
            result,
            context={},
            initial_transcript=[],
            task_id="qq:master-1:m-1",
        )

        summaries = [
            item
            for item in self.store.timeline(
                scope=MemoryService.OWNER_SCOPE,
                limit=10,
                now=2_000_000_000,
                layers=("episodic",),
                include_facts=False,
            )
            if item.metadata.get("record_type") == "agent_task_summary"
        ]
        self.assertEqual(len(summaries), 1)
        self.assertIn("昨天的 Agent 任务", summaries[0].text)
        self.assertIn("memory_search", summaries[0].text)
        self.assertNotIn("PRIVATE_RAW_TOOL_RESULT", summaries[0].text)
        opened = self.memory.inspect_agent_task(
            requester_id="master-1",
            task_id="qq:master-1:m-1",
        )
        detail = opened["detail_record"]
        self.assertIsNotNone(detail)
        self.assertEqual(detail["metadata"]["tool_steps"], [{"name": "memory_search", "status": "ok"}])
        self.assertNotIn("PRIVATE_RAW_TOOL_RESULT", detail["text"])
        self.assertNotIn("DO_NOT_ARCHIVE_THIS", detail["text"])
        generic_search = self.store.search(
            query="memory_search",
            scope=MemoryService.OWNER_SCOPE,
            limit=10,
            now=2_000_000_000,
        )
        self.assertFalse(any(item.metadata.get("record_type") == "agent_task_detail" for item in generic_search))

    def test_task_detail_expires_while_summary_card_remains(self) -> None:
        now = [2_000_000_000]
        memory = MemoryService(
            store=self.store,
            router=MemoryRouter(),
            policy=MemoryPolicy(
                master_user_ids=("master-1",),
                agent_task_detail_ttl_seconds=3600,
            ),
            clock=lambda: now[0],
        )
        common = {
            "conversation_id": "master-1",
            "conversation_type": "private",
            "requester_id": "master-1",
            "task_id": "qq:master-1:expires",
            "metadata": {"sender_id": "master-1"},
        }
        memory.record_agent_task_summary(
            **common,
            goal="短期详情过期测试",
            status="completed",
            outcome="摘要仍保留",
            tool_outcomes=[{"name": "memory_search", "status": "ok"}],
        )
        memory.record_agent_task_detail(
            **common,
            status="completed",
            rounds=2,
            steps=[{"name": "memory_search", "status": "ok"}],
        )
        now[0] += 3601

        opened = memory.inspect_agent_task(requester_id="master-1", task_id="qq:master-1:expires")

        self.assertTrue(opened["found"])
        self.assertFalse(opened["detail_available"])
        self.assertIn("摘要仍保留", opened["rendered"])

    def test_task_detail_capacity_forgets_oldest_detail_but_keeps_index_cards(self) -> None:
        memory = MemoryService(
            store=self.store,
            router=MemoryRouter(),
            policy=MemoryPolicy(master_user_ids=("master-1",), agent_task_detail_max_records=1),
            clock=lambda: 2_000_000_000,
        )
        for task_id in ("task-old", "task-new"):
            common = {
                "conversation_id": "master-1",
                "conversation_type": "private",
                "requester_id": "master-1",
                "task_id": task_id,
                "metadata": {"sender_id": "master-1"},
            }
            memory.record_agent_task_summary(
                **common,
                goal=f"任务 {task_id}",
                status="completed",
                outcome="已完成",
                tool_outcomes=[{"name": "memory_search", "status": "ok"}],
            )
            memory.record_agent_task_detail(
                **common,
                status="completed",
                rounds=1,
                steps=[{"name": "memory_search", "status": "ok"}],
            )

        old = memory.inspect_agent_task(requester_id="master-1", task_id="task-old")
        new = memory.inspect_agent_task(requester_id="master-1", task_id="task-new")
        self.assertTrue(old["found"])
        self.assertFalse(old["detail_available"])
        self.assertTrue(new["detail_available"])

    def test_forgetting_task_summary_also_forgets_linked_detail(self) -> None:
        common = {
            "conversation_id": "master-1",
            "conversation_type": "private",
            "requester_id": "master-1",
            "task_id": "task-forget-linked",
            "metadata": {"sender_id": "master-1"},
        }
        self.memory.record_agent_task_summary(
            **common,
            goal="遗忘联动测试任务",
            status="completed",
            outcome="任务结果已保存",
            tool_outcomes=[{"name": "memory_search", "status": "ok"}],
        )
        self.memory.record_agent_task_detail(
            **common,
            status="completed",
            rounds=2,
            steps=[{"name": "memory_search", "status": "ok"}],
        )

        forgotten, purged = self.store.forget(
            query="遗忘联动测试任务",
            scope=MemoryService.OWNER_SCOPE,
            now=2_000_000_000,
        )

        self.assertEqual((forgotten, purged), (2, 0))
        opened = self.memory.inspect_agent_task(
            requester_id="master-1",
            task_id="task-forget-linked",
        )
        self.assertFalse(opened["found"])

    def test_owner_can_open_exact_task_card_from_private_or_group_context(self) -> None:
        self.assertTrue(
            self.memory.record_agent_task_summary(
                conversation_id="master-1",
                conversation_type="private",
                requester_id="master-1",
                task_id="qq:master-1:task-open",
                goal="修复 Agent 任务回忆",
                status="completed",
                outcome="检索现在能按任务 ID 找到对应档案。",
                tool_outcomes=[{"name": "memory_search", "status": "ok"}],
                metadata={"sender_id": "master-1", "shinku_involved": True},
            )
        )

        opened = self.memory.inspect_agent_task(
            requester_id="master-1",
            task_id="qq:master-1:task-open",
        )

        self.assertTrue(opened["found"])
        self.assertEqual(opened["status"], "completed")
        self.assertIn("检索现在能按任务 ID 找到对应档案", opened["rendered"])
        self.assertEqual(opened["record"]["metadata"]["record_type"], "agent_task_summary")

    def test_owner_can_retrieve_group_task_from_private_and_group_without_exposing_detail(self) -> None:
        common = {
            "conversation_id": "group-42",
            "conversation_type": "group",
            "requester_id": "master-1",
            "task_id": "qq:group-42:cross-scope-task",
            "metadata": {
                "sender_id": "master-1",
                "shinku_involved": True,
                "at_targets": ["shinku-bot"],
            },
        }
        self.assertTrue(
            self.memory.record_agent_task_summary(
                **common,
                goal="修复跨会话记忆检索回归",
                status="completed",
                outcome="群聊任务已经完成，语义检索可以从主人记忆中找到结论。",
                tool_outcomes=[{"name": "memory_search", "status": "ok"}],
            )
        )
        self.assertTrue(
            self.memory.record_agent_task_detail(
                **common,
                status="completed",
                rounds=2,
                steps=[
                    {"name": "memory_search", "status": "ok"},
                    {"name": "memory_forget", "status": "skipped"},
                ],
            )
        )

        private_result = self.memory.inspect(
            conversation_id="master-1",
            conversation_type="private",
            requester_id="master-1",
            query="跨会话记忆检索回归 群聊任务结论",
            mode="retrieve",
        )
        group_result = self.memory.inspect(
            conversation_id="group-42",
            conversation_type="group",
            requester_id="master-1",
            query="跨会话记忆检索回归 群聊任务结论",
            mode="retrieve",
        )
        ordinary_recent = self.memory.inspect(
            conversation_id="master-1",
            conversation_type="private",
            requester_id="master-1",
            query="跨会话记忆检索回归 群聊任务结论",
            mode="recent",
        )

        for result in (private_result, group_result):
            self.assertTrue(result["owner_memory"])
            self.assertIn(MemoryService.OWNER_SCOPE, result["retrieval_scopes"])
            task_cards = [
                item for item in result["records"]
                if item["metadata"].get("record_type") == "agent_task_summary"
            ]
            self.assertEqual(len(task_cards), 1)
            self.assertIn("【Agent任务记录】", result["rendered"])
            self.assertNotIn("群成员：Agent任务目标", result["rendered"])
            self.assertNotIn("memory_forget", result["rendered"])
        self.assertNotIn(MemoryService.OWNER_SCOPE, ordinary_recent["retrieval_scopes"])
        self.assertNotIn("群聊任务已经完成", ordinary_recent["rendered"])

        exact = self.memory.inspect_agent_task(
            requester_id="master-1",
            task_id="qq:group-42:cross-scope-task",
        )
        self.assertTrue(exact["detail_available"])
        self.assertIn("memory_forget", exact["rendered"])

    def test_non_owner_cannot_retrieve_owner_task_from_private_or_group(self) -> None:
        self.memory.record_agent_task_summary(
            conversation_id="group-42",
            conversation_type="group",
            requester_id="master-1",
            task_id="qq:group-42:owner-only-task",
            goal="主人私有的跨会话安全检查",
            status="completed",
            outcome="只有主人可以回忆这条任务结论。",
            tool_outcomes=[{"name": "memory_search", "status": "ok"}],
            metadata={"sender_id": "master-1", "shinku_involved": True},
        )

        private_result = self.memory.inspect(
            conversation_id="member-2",
            conversation_type="private",
            requester_id="member-2",
            query="主人私有的跨会话安全检查任务结论",
            mode="retrieve",
        )
        group_result = self.memory.inspect(
            conversation_id="group-42",
            conversation_type="group",
            requester_id="member-2",
            query="主人私有的跨会话安全检查任务结论",
            mode="retrieve",
        )

        for result in (private_result, group_result):
            self.assertFalse(result["owner_memory"])
            self.assertNotIn(MemoryService.OWNER_SCOPE, result["retrieval_scopes"])
            self.assertNotIn("只有主人可以回忆", result["rendered"])

    def test_non_owner_and_unknown_task_receive_same_not_found_response(self) -> None:
        self.memory.record_agent_task_summary(
            conversation_id="master-1",
            conversation_type="private",
            requester_id="master-1",
            task_id="qq:master-1:private-task",
            goal="主人私有任务",
            status="completed",
            outcome="私有结果",
            tool_outcomes=[{"name": "read_file", "status": "ok"}],
            metadata={"sender_id": "master-1"},
        )

        denied = self.memory.inspect_agent_task(
            requester_id="member-2",
            task_id="qq:master-1:private-task",
        )
        missing = self.memory.inspect_agent_task(
            requester_id="master-1",
            task_id="not-a-real-task",
        )

        self.assertFalse(denied["found"])
        self.assertFalse(missing["found"])
        self.assertEqual(denied["rendered"], missing["rendered"])
        self.assertNotIn("私有结果", denied["rendered"])

    def test_memory_search_tool_can_open_task_id_without_searching_other_scopes(self) -> None:
        from shinku.memory.tools import MemorySearchTool

        self.memory.record_agent_task_summary(
            conversation_id="master-1",
            conversation_type="private",
            requester_id="master-1",
            task_id="qq:master-1:tool-open",
            goal="检查任务档案工具入口",
            status="completed",
            outcome="入口已就绪。",
            tool_outcomes=[{"name": "memory_search", "status": "ok"}],
            metadata={"sender_id": "master-1"},
        )
        turn = _private_turn("这个任务详细结果呢？")
        tool = MemorySearchTool(self.memory)

        result = tool.execute(
            call={"task_id": "qq:master-1:tool-open"},
            context={"qq_turn": turn},
        )

        self.assertIn("入口已就绪", result["followup_context"])
        self.assertTrue(result["state_updates"]["memory_inspection"]["found"])

        non_owner_turn = _private_turn("打开这个任务", user_id="member-2", message_id="m-2")
        denied = tool.execute(
            call={"task_id": "qq:master-1:tool-open"},
            context={"qq_turn": non_owner_turn},
        )
        self.assertFalse(denied["state_updates"]["memory_inspection"]["found"])
        self.assertNotIn("入口已就绪", denied["followup_context"])


if __name__ == "__main__":
    unittest.main()
