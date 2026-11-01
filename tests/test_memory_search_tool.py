from __future__ import annotations

from reply_review_fixture import dialogue_review_result

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.registry import ToolRegistry
from shinku.tools.execution import ExecutionPolicy


class _NativeRuntime:
    def __init__(self):
        self.calls = 0

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls += 1
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        if self.calls == 1:
            return {
                "_native_tool_call": {
                    "tool_calls": [
                        {
                            "name": "memory_search",
                            "arguments": {"mode": "recent", "limit": 2},
                            "id": "memory-call-1",
                        }
                    ]
                }
            }
        return {"speech": "我看过了。"}


def _turn() -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-native-memory",
            "message_id": "m-native-memory",
            "message": [{"type": "text", "data": {"text": "查看记忆"}}],
        }
    )
    return QQTurn(
        conversation_id="u-native-memory",
        conversation_type="private",
        primary_message_id="m-native-memory",
        message_ids=("m-native-memory",),
        combined_text="查看记忆",
        messages=(message,),
    )


class MemorySearchToolTests(unittest.TestCase):
    def _service(self, root: str) -> MemoryService:
        store = MemoryStore(Path(root) / "memory.sqlite3")
        return MemoryService(
            store=store,
            router=MemoryRouter(deterministic_guards=True),
            policy=MemoryPolicy(passive_observe_private=True),
            clock=lambda: 2_000_000_000,
        )

    def test_explicit_inspection_reads_current_scope_without_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self._service(temporary)
            service.store.record_event(
                conversation_id="u-1",
                scope="private:u-1",
                text="上次讨论了图片发送问题。",
                source_id="event-1",
                layer="observation",
                created_at=1_999_999_990,
            )
            service.store.upsert_fact(
                conversation_id="u-1",
                scope="private:u-1",
                fact_key="explicit:habit",
                text="用户希望重要记忆长期保留。",
                source_id="fact-1",
                now=1_999_999_989,
            )
            tool = MemorySearchTool(service)
            self.assertEqual(tool.normalize_call({"query": "记忆迁移"})["mode"], "retrieve")
            result = tool.execute(
                call={"mode": "recent", "limit": 8},
                context={"qq_turn": SimpleNamespace(conversation_id="u-1", conversation_type="private")},
            )
            self.assertIn("图片发送问题", result["followup_context"])
            self.assertIn("重要记忆长期保留", result["followup_context"])
            self.assertEqual(result["state_updates"]["memory_inspection"]["scope"], "private:u-1")

    def test_retrieve_and_missing_scope_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self._service(temporary)
            service.store.record_event(
                conversation_id="g-1",
                scope="group:g-1",
                text="群里讨论了记忆迁移。",
                source_id="event-2",
                layer="observation",
                created_at=1_999_999_990,
            )
            tool = MemorySearchTool(service)
            result = tool.execute(
                call={"mode": "retrieve", "query": "记忆迁移", "limit": 3},
                context={"qq_turn": SimpleNamespace(conversation_id="g-1", conversation_type="group")},
            )
            self.assertEqual(result["state_updates"]["memory_inspection"]["count"], 1)
            missing = tool.execute(call={}, context={})
            self.assertIn("没有绑定 QQ 会话", missing["followup_context"])

    def test_agent_can_execute_memory_tool_and_continue_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self._service(temporary)
            runtime = _NativeRuntime()
            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="persona",
                tool_host=ToolHost(ToolRegistry({"memory_search": MemorySearchTool(service)})),
                policy=ExecutionPolicy(risk_by_tool={"memory_search": "low"}),
            )
            result = bridge.handle_turn(_turn())
            self.assertEqual(result.agent_status, "partial")
            self.assertIn("没找到", result.final_text)
            self.assertEqual(runtime.calls, 1)

    def test_date_recall_uses_a_day_range_and_sees_both_sides_of_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            now = int(time.mktime((2026, 9, 21, 12, 0, 0, -1, -1, -1)))
            service = MemoryService(
                store=MemoryStore(Path(temporary) / "memory.sqlite3"),
                router=MemoryRouter(deterministic_guards=True),
                policy=MemoryPolicy(passive_observe_private=True),
                clock=lambda: now,
            )
            day = int(time.mktime((2026, 9, 19, 12, 0, 0, -1, -1, -1)))
            service.store.record_event(
                conversation_id="u-date",
                scope="private:u-date",
                text="我们讨论了图片发送问题。",
                source_id="user-date",
                layer="episodic",
                created_at=day + 3600,
                metadata={"role": "user", "source": "qq_turn"},
            )
            service.store.record_event(
                conversation_id="u-date",
                scope="private:u-date",
                text="我会先检查出站分条链路。",
                source_id="assistant-date",
                layer="episodic",
                created_at=day,
                metadata={"role": "assistant", "source": "qq_turn"},
            )
            service.store.record_event(
                conversation_id="u-date",
                scope="private:u-date",
                text="这条在日期范围外，不应该被选中。",
                source_id="outside-date",
                layer="episodic",
                created_at=day - 3 * 86400,
                metadata={"role": "assistant", "source": "qq_turn"},
            )
            context = service.prepare_turn(
                conversation_id="u-date",
                conversation_type="private",
                user_message="看看9月19号下午我们聊了什么",
            )
            self.assertEqual(context.decision.history_mode, "summary")
            self.assertTrue(context.records)
            self.assertIn("图片发送问题", context.rendered)
            self.assertIn("出站分条链路", context.rendered)
            self.assertNotIn("日期范围外", context.rendered)

    def test_relative_date_recall_finds_an_older_topic_and_adjacent_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            now = int(time.mktime((2026, 9, 21, 19, 40, 0, -1, -1, -1)))
            service = MemoryService(
                store=MemoryStore(Path(temporary) / "memory.sqlite3"),
                router=MemoryRouter(deterministic_guards=True),
                policy=MemoryPolicy(passive_observe_private=True),
                clock=lambda: now,
            )
            day = int(time.mktime((2026, 9, 19, 12, 0, 0, -1, -1, -1)))
            service.store.record_event(
                conversation_id="u-relative",
                scope="private:u-relative",
                text="前几天我问过你怎么形容自己。",
                source_id="relative-user",
                layer="episodic",
                created_at=day + 60,
                metadata={"role": "user", "source": "qq_turn"},
            )
            service.store.record_event(
                conversation_id="u-relative",
                scope="private:u-relative",
                text="我说自己安静、冷静，不需要急着证明什么。",
                source_id="relative-assistant",
                layer="episodic",
                created_at=day + 75,
                metadata={"role": "assistant", "source": "qq_turn"},
            )
            for index in range(8):
                service.store.record_event(
                    conversation_id="u-relative",
                    scope="private:u-relative",
                    text=f"当天后面出现的无关噪声 {index}",
                    source_id=f"relative-noise-{index}",
                    layer="episodic",
                    created_at=day + 3600 + index,
                    metadata={"role": "user", "source": "qq_turn"},
                )

            context = service.prepare_turn(
                conversation_id="u-relative",
                conversation_type="private",
                user_message="你还记得前天我问你你形容自己是什么吗？",
            )

            self.assertEqual(context.decision.history_mode, "summary")
            self.assertIn("形容自己", context.rendered)
            self.assertIn("安静、冷静", context.rendered)
            self.assertNotIn("无关噪声", context.rendered)
            self.assertEqual(context.audit["fallback"], "date_query_match")

    def test_empty_date_window_does_not_leak_records_from_other_days(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            now = int(time.mktime((2026, 9, 21, 19, 40, 0, -1, -1, -1)))
            service = MemoryService(
                store=MemoryStore(Path(temporary) / "memory.sqlite3"),
                router=MemoryRouter(deterministic_guards=True),
                policy=MemoryPolicy(passive_observe_private=True),
                clock=lambda: now,
            )
            service.store.record_event(
                conversation_id="u-empty-date",
                scope="private:u-empty-date",
                text="今天刚讨论了一个新话题。",
                source_id="today-only",
                layer="episodic",
                created_at=now - 60,
                metadata={"role": "user", "source": "qq_turn"},
            )
            context = service.prepare_turn(
                conversation_id="u-empty-date",
                conversation_type="private",
                user_message="你还记得前天我问过的那个旧话题吗？",
            )
            self.assertEqual(context.records, ())
            self.assertEqual(context.audit["fallback"], "date_timeline_empty")

    def test_assistant_reply_is_recorded_once_as_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self._service(temporary)
            sent, calls = [], []
            bridge = QQAgentBridge(
                runtime=type("Runtime", (), {"call_chat_json": lambda self, **kwargs:
                    calls.append(kwargs) or dialogue_review_result(kwargs) or {"speech": "真红的完整回复。"}})(),
                system_prompt="persona",
                sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out",)),
                memory_service=service,
            )
            # Use the normal delivery contract rather than relying on a
            # transport implementation detail in the assertion below.
            result = bridge.handle_turn(_turn())
            self.assertTrue(result.final_text)
            # 主生成和审核两次调用，只保存实际交付的最终助手正文。
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.agent_status, "completed")
            rows = service.store.timeline(
                scope="private:u-native-memory",
                limit=20,
                now=2_000_000_000,
                layers=("episodic",),
                include_facts=False,
            )
            assistant_rows = [row for row in rows if row.metadata.get("role") == "assistant"]
            self.assertEqual(len(assistant_rows), 1)
            self.assertEqual(assistant_rows[0].text, "真红的完整回复。")


if __name__ == "__main__":
    unittest.main()
