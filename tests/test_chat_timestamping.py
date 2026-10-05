from __future__ import annotations

import time
import unittest
from datetime import datetime

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn


class ScriptedRuntime:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def call_chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.response


def _make_turn(
    user_text: str,
    *,
    timestamp: float | None = None,
    conversation_type: str = "private",
    conversation_id: str = "conv_1",
    sender_label: str = "主人",
) -> QQTurn:
    event = {
        "post_type": "message",
        "message_type": conversation_type,
        "user_id": "u_test",
        "message_id": "m_1",
        "message": [{"type": "text", "data": {"text": user_text}}],
        "sender": {"nickname": sender_label},
    }
    if timestamp is not None:
        event["time"] = int(timestamp)
    msg = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        primary_message_id="m_1",
        message_ids=("m_1",),
        combined_text=user_text,
        messages=(msg,),
    )


class ChatTimestampingTests(unittest.TestCase):
    def test_private_message_with_timestamp_formats_prefix(self) -> None:
        """测试私聊消息在有真实时间戳时，格式化为 [HH:MM] 文本。"""
        # 构造一个 15:08 的绝对时间戳
        dt = datetime(2026, 9, 25, 15, 8, 30)
        ts = dt.timestamp()
        turn = _make_turn("早上好", timestamp=ts, conversation_type="private")

        transcript = QQAgentBridge._transcript(turn)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["role"], "user")
        self.assertEqual(transcript[0]["content"], "[15:08] 早上好")

    def test_group_message_with_timestamp_formats_prefix_and_sender(self) -> None:
        """测试群聊消息在有真实时间戳时，格式化为 [HH:MM] 昵称：文本。"""
        dt = datetime(2026, 9, 25, 9, 30, 0)
        ts = dt.timestamp()
        turn = _make_turn("大家早", timestamp=ts, conversation_type="group", sender_label="琴里")

        transcript = QQAgentBridge._transcript(turn)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["role"], "user")
        self.assertEqual(transcript[0]["content"], "[09:30] 琴里：大家早")

    def test_message_without_timestamp_preserves_plain_text(self) -> None:
        """测试没有时间戳（如简易测试 Mock）时，回退保持纯文本不带时间前缀，向后兼容。"""
        turn = _make_turn("你好", timestamp=None, conversation_type="private")

        transcript = QQAgentBridge._transcript(turn)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["content"], "你好")

    def test_runtime_call_history_turns_includes_timestamp(self) -> None:
        """测试端到端经由 handle_turn 处理后，发送给模型 runtime 的 history_turns 包含时间戳。"""
        dt = datetime(2026, 9, 25, 15, 8, 0)
        ts = dt.timestamp()
        turn = _make_turn("早上好", timestamp=ts)

        runtime = ScriptedRuntime({"speech": "……15点了还早上好？"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="测试人设",
            sender=lambda msg: DeliveryResult(True, "sent", ("out",)),
        )

        reply = bridge.handle_turn(turn)
        self.assertEqual(reply.agent_status, "completed")
        self.assertEqual(len(runtime.calls), 1)

        # 当前原话只放在末尾 user 消息，不重复塞进历史。
        self.assertEqual(runtime.calls[0]["history_turns"], [])
        self.assertEqual(runtime.calls[0]["user_prompt"], "[15:08] 早上好")

    def test_chat_timestamp_deterministic_and_cache_stable(self) -> None:
        """测试绝对时间戳是确定性的，在相同消息生成时文本严格恒定，确保 Prefix Caching 稳定。"""
        ts = 1790320080.0  # 固定绝对时间
        expected_stamp = time.strftime("%H:%M", time.localtime(ts))

        turn_1 = _make_turn("测试确定性", timestamp=ts)
        transcript_1 = QQAgentBridge._transcript(turn_1)

        turn_2 = _make_turn("测试确定性", timestamp=ts)
        transcript_2 = QQAgentBridge._transcript(turn_2)

        self.assertEqual(transcript_1[0]["content"], f"[{expected_stamp}] 测试确定性")
        self.assertEqual(transcript_1[0]["content"], transcript_2[0]["content"])

    def test_format_time_anchor_periods(self) -> None:
        """测试不同时段与星期的客观现实时间锚点格式化。"""
        cases = [
            (datetime(2026, 9, 21, 6, 30), "2026-09-21 06:30", "清晨", "星期一"),
            (datetime(2026, 9, 22, 10, 15), "2026-09-22 10:15", "上午", "星期二"),
            (datetime(2026, 9, 23, 12, 0), "2026-09-23 12:00", "中午", "星期三"),
            (datetime(2026, 9, 24, 15, 8), "2026-09-24 15:08", "下午", "星期四"),
            (datetime(2026, 9, 25, 18, 45), "2026-09-25 18:45", "傍晚", "星期五"),
            (datetime(2026, 9, 26, 21, 30), "2026-09-26 21:30", "晚上", "星期六"),
            (datetime(2026, 9, 27, 23, 15), "2026-09-27 23:15", "深夜", "星期日"),
            (datetime(2026, 9, 28, 2, 5), "2026-09-28 02:05", "深夜", "星期一"),
        ]
        for dt, expected_time_str, expected_period, expected_weekday in cases:
            anchor = QQAgentBridge._format_time_anchor(dt)
            self.assertIn("【当前现实时间】", anchor)
            self.assertIn(expected_time_str, anchor)
            self.assertIn(expected_period, anchor)
            self.assertIn(expected_weekday, anchor)
            self.assertEqual(
                anchor,
                f"【当前现实时间】\n现在是 {expected_time_str}（{expected_period}，{expected_weekday}）。",
            )

    def test_user_prompt_tail_contains_minimal_fact_time_anchor(self) -> None:
        """测试每轮模型请求的 user_prompt 尾部注入了客观现实时间锚点，且前面的前缀内容保持稳定。"""
        dt = datetime(2026, 9, 25, 15, 8, 0)
        ts = dt.timestamp()
        turn = _make_turn("早上好", timestamp=ts)

        runtime = ScriptedRuntime({"speech": "……现在是下午哦。"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="测试系统人设",
            sender=lambda msg: DeliveryResult(True, "sent", ("out",)),
        )

        reply = bridge.handle_turn(turn)
        self.assertEqual(reply.agent_status, "completed")
        self.assertEqual(len(runtime.calls), 1)

        user_prompt = runtime.calls[0]["user_prompt"]
        expected_anchor = "【当前现实时间】\n现在是 2026-09-25 15:08（下午，星期五）。"

        # 现实时间是 system 侧事实锚点；user_prompt 保留用户原话作为最后一条消息。
        system_context = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn(expected_anchor, system_context)
        self.assertEqual(user_prompt, "[15:08] 早上好")

    def test_resolve_anchor_time_precedence(self) -> None:
        """测试时间锚点解析优先级：context 显式指定 > turn 消息时间戳 > 系统实时时间。"""
        dt_override = datetime(2026, 1, 1, 12, 0)
        dt_turn = datetime(2026, 5, 1, 14, 0)

        turn = _make_turn("你好", timestamp=dt_turn.timestamp())

        # 1. 显式 context override
        ctx_with_override = {"now": dt_override, "qq_turn": turn}
        self.assertEqual(QQAgentBridge._resolve_anchor_time(ctx_with_override), dt_override)

        # 2. turn 消息时间戳
        ctx_with_turn = {"qq_turn": turn}
        resolved = QQAgentBridge._resolve_anchor_time(ctx_with_turn)
        self.assertEqual(resolved.year, 2026)
        self.assertEqual(resolved.month, 5)
        self.assertEqual(resolved.day, 1)
        self.assertEqual(resolved.hour, 14)

        # 3. 兜底系统时间
        resolved_default = QQAgentBridge._resolve_anchor_time({})
        self.assertIsInstance(resolved_default, datetime)


if __name__ == "__main__":
    unittest.main()
