from __future__ import annotations

import json
import unittest

from shinku.llm.runtime_audit import RuntimeAuditMixin
from shinku.llm.turn_usage import bind_qq_turn_usage
from shinku.qq.agent_bridge import QQAgentBridge, QQAgentReply
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionPolicy
from shinku.qq.attention_trace import AttentionTraceStore
from shinku.qq.message import IncomingMessage
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.trace import TurnTraceStore
from shinku.qq.turns import QQTurn


class AttentionTraceTests(unittest.TestCase):
    def test_ingress_records_rejected_for_autonomous_reply_without_message_text(self) -> None:
        trace_store = AttentionTraceStore(max_items=5)
        adapter = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(),
            attention_policy=AttentionPolicy(autonomous_enabled=False),
            attention_trace_store=trace_store,
        )
        message = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-1",
                "message_id": "m-1",
                "time": 100,
                "self_id": "bot-1",
                "raw_message": "这段原文不能进入注意力诊断",
                "message": [{"type": "text", "data": {"text": "这段原文不能进入注意力诊断"}}],
            }
        )

        result = adapter.ingest_message(message, at=100)
        snapshot = trace_store.snapshot()
        row = snapshot["items"][0]

        self.assertTrue(result.admitted)
        self.assertFalse(result.scheduled)
        self.assertEqual(row["attention"]["reason"], "autonomous_reply_disabled")
        self.assertFalse(row["queued"])
        self.assertEqual(row["text_chars"], len(message.text))
        self.assertEqual(row["topic_state"]["group_context_count"], 0)
        self.assertNotIn("这段原文不能进入注意力诊断", json.dumps(snapshot, ensure_ascii=False))
        self.assertEqual(snapshot["summary"]["not_scheduled"], 1)

    def test_duplicate_ingress_is_observable_without_scheduling(self) -> None:
        trace_store = AttentionTraceStore(max_items=5)
        adapter = QQIngressAdapter(attention_trace_store=trace_store)
        event = {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-1",
            "user_id": "u-1",
            "message_id": "m-duplicate",
            "time": 100,
            "self_id": "bot-1",
            "raw_message": "普通群聊内容",
            "message": [{"type": "text", "data": {"text": "普通群聊内容"}}],
        }
        first = adapter.ingest(event, at=100)
        second = adapter.ingest(event, at=101)

        rows = trace_store.snapshot()["items"]
        self.assertTrue(first.admitted)
        self.assertFalse(second.admitted)
        self.assertEqual(rows[0]["admission"]["reason"], "duplicate_event")
        self.assertNotIn("fingerprint", rows[0]["admission"])
        self.assertFalse(rows[0]["attention"]["should_schedule"])

    def test_trace_store_is_bounded_and_summarizes_reasons(self) -> None:
        store = AttentionTraceStore(max_items=2)
        for index in range(3):
            store.record(
                observed_at=float(index),
                conversation_id="g-1",
                conversation_type="group",
                message_id=f"m-{index}",
                sender_id="u-1",
                text_chars=8,
                has_media=False,
                is_poke=False,
                admission={"accepted": True, "reason": "accepted"},
                route={"should_reply": False, "reason": "unaddressed_group_message"},
                attention={"should_schedule": index == 2, "reason": f"reason-{index}"},
                queued=index == 2,
            )
        snapshot = store.snapshot(limit=10)
        self.assertEqual([item["message_id"] for item in snapshot["items"]], ["m-2", "m-1"])
        self.assertEqual(snapshot["summary"]["scheduled"], 1)
        self.assertEqual(snapshot["summary"]["window_items"], 2)

    def test_provider_usage_is_attached_only_inside_qq_turn_context(self) -> None:
        reports: list[dict[str, object]] = []

        class RuntimeAuditHarness(RuntimeAuditMixin):
            def __init__(self) -> None:
                self.metrics: dict[str, int] = {}

            def _add_metric(self, key: str, value: int = 1) -> None:
                self.metrics[key] = self.metrics.get(key, 0) + int(value)

        audit = RuntimeAuditHarness()
        with bind_qq_turn_usage("turn-1", lambda report: reports.append(dict(report))):
            audit._record_tokens(
                {"prompt_tokens": 18, "completion_tokens": 4, "total_tokens": 22, "prompt_cache_hit_tokens": 9},
                metric_suffix="chat",
            )
        audit._record_tokens(
            {"prompt_tokens": 7, "completion_tokens": 2},
            metric_suffix="aux",
        )

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["trace_id"], "turn-1")
        self.assertEqual(reports[0]["channel"], "chat")
        self.assertEqual(reports[0]["input_tokens"], 18)
        self.assertEqual(reports[0]["output_tokens"], 4)
        self.assertEqual(reports[0]["cache_read_tokens"], 9)

    def test_bridge_turn_trace_receives_correlated_model_usage(self) -> None:
        bridge = object.__new__(QQAgentBridge)
        bridge.trace_store = TurnTraceStore()

        def fake_turn_handler(turn: QQTurn, *, trace_id: str = "") -> QQAgentReply:
            from shinku.llm.turn_usage import report_qq_turn_usage

            report_qq_turn_usage(
                channel="chat",
                calls=1,
                input_tokens=31,
                output_tokens=6,
                total_tokens=37,
            )
            return QQAgentReply("completed", sent=False, reason="test_only")

        bridge._handle_turn = fake_turn_handler
        turn = QQTurn(
            conversation_id="g-1",
            conversation_type="group",
            primary_message_id="m-1",
            message_ids=("m-1", "m-2"),
            combined_text="测试内容不应存入 trace",
            messages=(),
        )

        bridge.handle_turn(turn)

        row = bridge.trace_store.snapshot()[0]
        usage = next(stage for stage in row["stages"] if stage["name"] == "llm_usage")
        self.assertEqual(row["message_ids"], ["m-1", "m-2"])
        self.assertEqual(usage["total_tokens"], 37)
        self.assertNotIn("测试内容不应存入 trace", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
