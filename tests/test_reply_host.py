from __future__ import annotations

import unittest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.qq.reply_host import ReplyHost


class _Runtime:
    def __init__(self, response):
        self.response = response

    def call_chat_json(self, **kwargs):
        return self.response


def _turn(text: str = "测试") -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": "reply-host-user",
            "message_id": "reply-host-in",
            "message": [{"type": "text", "data": {"text": text}}],
        }
    )
    return QQTurn(
        conversation_id="reply-host-user",
        conversation_type="private",
        primary_message_id="reply-host-in",
        message_ids=("reply-host-in",),
        combined_text=text,
        messages=(message,),
    )


class ReplyHostTests(unittest.TestCase):
    def test_period_boundary_is_split_then_final_period_is_removed(self) -> None:
        host = ReplyHost(strip_final_periods=True)
        self.assertEqual(host.shape_text("第一句。第二句！"), ["第一句", "第二句！"])

        sent = []
        bridge = QQAgentBridge(
            runtime=_Runtime({"speech": "第一句。第二句！"}),
            system_prompt="persona",
            sender=lambda message: sent.append(message) or DeliveryResult(
                True, "sent", (f"out-{len(sent)}",)
            ),
            reply_host=ReplyHost(strip_final_periods=True),
        )
        result = bridge.handle_turn(_turn())
        self.assertTrue(result.sent)
        self.assertEqual([item.segments[0].text for item in sent], ["第一句", "第二句！"])
        self.assertEqual(result.delivery.message_ids, ("out-1", "out-2"))

    def test_final_period_stripping_can_be_disabled(self) -> None:
        host = ReplyHost(strip_final_periods=False)
        self.assertEqual(host.shape_text("第一句。第二句。"), ["第一句。", "第二句。"])

    def test_compact_multi_sentence_reply_preserves_all_sentence_boundaries(self) -> None:
        host = ReplyHost(strip_final_periods=True)
        shaped = host.shape_text(
            "你啊……\n知道了。我陪你到睡着。明天要我喊你吗？",
            user_message="陪我睡一会儿",
        )
        self.assertEqual(
            shaped,
            ["你啊……", "知道了", "我陪你到睡着", "明天要我喊你吗？"],
        )

    def test_no_sentence_count_cap_even_for_longer_technical_replies(self) -> None:
        host = ReplyHost(strip_final_periods=True)
        sentences = [f"第{index}句确认完成。" for index in range(1, 13)]
        shaped = host.shape_text("".join(sentences), user_message="帮我检查代码")
        self.assertEqual(shaped, [f"第{index}句确认完成" for index in range(1, 13)])

    def test_comfort_reply_splits_each_authored_sentence(self) -> None:
        host = ReplyHost(strip_final_periods=True)
        shaped = host.shape_text(
            "……原来是说那件事。因为碰不到你啊。"
            "就算你想，我现在也只能像这样陪你说说话。"
            "……真是的，大半夜非要让我把这种话说出来。"
            "快点躺好，我在这里陪着你"
        )
        self.assertEqual(
            shaped,
            [
                "……原来是说那件事",
                "因为碰不到你啊",
                "就算你想，我现在也只能像这样陪你说说话",
                "……真是的，大半夜非要让我把这种话说出来",
                "快点躺好，我在这里陪着你",
            ],
        )

    def test_long_or_technical_reply_preserves_every_sentence_boundary(self) -> None:
        host = ReplyHost(long_trigger_chars=20, strip_final_periods=True)
        shaped = host.shape_text("端口已经恢复。接下来检查日志。最后再做一次回归测试。")
        self.assertEqual(
            shaped,
            ["端口已经恢复", "接下来检查日志", "最后再做一次回归测试"],
        )


if __name__ == "__main__":
    unittest.main()
