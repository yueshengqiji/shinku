from __future__ import annotations

import unittest

from shinku.agent.loop import AgentRunResult
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatVisualInput
from shinku.qq.turns import QQTurn


class _Runtime:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class QQAgentBridgeTests(unittest.TestCase):
    @staticmethod
    def _address_turn() -> QQTurn:
        message = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "private",
                "user_id": "u-19",
                "message_id": "m-address",
                "message": [{"type": "text", "data": {"text": "老婆"}}],
            }
        )
        return QQTurn(
            conversation_id="u-19",
            conversation_type="private",
            primary_message_id="m-address",
            message_ids=("m-address",),
            combined_text="老婆",
            messages=(message,),
        )

    @staticmethod
    def _turn() -> QQTurn:
        message = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "private",
                "user_id": "u-19",
                "message_id": "m-19",
                "message": [
                    {"type": "text", "data": {"text": "帮我看看"}},
                    {"type": "image", "data": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }
        )
        image = NapCatVisualInput(
            source_kind="url",
            reference="data:image/png;base64,aGVsbG8=",
            data_url="data:image/png;base64,aGVsbG8=",
        )
        return QQTurn(
            conversation_id="u-19",
            conversation_type="private",
            primary_message_id="m-19",
            message_ids=("m-19",),
            combined_text="帮我看看",
            messages=(message,),
            visual_inputs=(("m-19", (image,)),),
        )

    def test_runs_agent_with_persona_transcript_and_images_then_sends_one_reply(self) -> None:
        runtime = _Runtime({"speech": "看到了。"})
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-1",)),
            prompt_cache_key="shinku:qq:v1",
        )

        result = bridge.handle_turn(self._turn())

        self.assertEqual((result.agent_status, result.final_text, result.sent), ("completed", "看到了。", True))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].conversation_id, "u-19")
        self.assertEqual(sent[0].segments[0].text, "看到了。")
        self.assertIn("真红主人设", runtime.calls[0]["system_prompt"])
        self.assertIn("聊天节奏与表达变化", runtime.calls[0]["system_prompt"])
        self.assertEqual(runtime.calls[0]["user_images"], [{
            "data_url": "data:image/png;base64,aGVsbG8=", "source_message_id": "m-19",
            "source_sender_id": "u-19", "source_sender_label": "", "reference_kind": "direct",
        }])
        self.assertEqual(runtime.calls[0]["history_turns"], [])
        self.assertIn("图片附件", runtime.calls[0]["user_prompt"])

    def test_default_sender_is_dry_run(self) -> None:
        bridge = QQAgentBridge(runtime=_Runtime({"speech": "只模拟。"}), system_prompt="p")
        result = bridge.handle_turn(self._turn())
        self.assertEqual((result.final_text, result.sent, result.delivery_status), ("只模拟。", False, "dry_run"))

    def test_terminal_agent_failure_is_delivered_instead_of_silently_dropped(self) -> None:
        sent = []
        bridge = QQAgentBridge(
            runtime=_Runtime({"speech": "不会调用到。"}),
            system_prompt="p",
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-failed",)),
        )

        result = bridge._finish_result(
            self._turn(),
            AgentRunResult(status="blocked", rounds=4, reason="no_progress"),
            context={},
            initial_transcript=[],
            task_id="task-failed",
        )

        self.assertEqual((result.agent_status, result.sent), ("blocked", True))
        self.assertIn("没有完成", result.final_text)
        expected = bridge.reply_host.shape(result.final_text).messages
        self.assertEqual(tuple(message.segments[0].text for message in sent), expected)

    def test_invalid_runtime_result_does_not_send(self) -> None:
        runtime = _Runtime({"kind": "wait", "text": "还缺一份资料。"})
        bridge = QQAgentBridge(runtime=runtime, system_prompt="p")
        result = bridge.handle_turn(self._turn())
        self.assertEqual((result.agent_status, result.final_text, result.delivery_status), ("waiting_user", "还缺一份资料。", "dry_run"))

    def test_partial_and_background_handoffs_are_delivered_with_truthful_status(self) -> None:
        for status in ("partial", "waiting_background", "failed"):
            with self.subTest(status=status):
                sent = []
                bridge = QQAgentBridge(runtime=_Runtime({}), system_prompt="测试",
                    sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("test-out",)))
                result = bridge._finish_result(self._turn(), AgentRunResult(status=status, rounds=2,
                    final_text="测试任务仍有缺项，尚未全部完成。", completion={"status": status, "missing": ["second"]}),
                    context={}, initial_transcript=[], task_id="test-only")
                self.assertEqual((result.agent_status, result.sent), (status, True))
                self.assertEqual(len(sent), 1)

    def test_address_only_turn_gets_contextual_guidance_without_reply_template(self) -> None:
        runtime = _Runtime({"speech": "知道了。"})
        bridge = QQAgentBridge(runtime=runtime, system_prompt="真红主人设")

        result = bridge.handle_turn(self._address_turn())

        self.assertEqual(result.final_text, "知道了。")
        prompt = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("本轮主要是点名", prompt)
        self.assertIn("先检查当前对话有没有尚未结束的话题", prompt)
        self.assertIn("不要机械复用上一轮的确认句", prompt)
        self.assertIn("招呼后的自然接话", prompt)
        self.assertNotIn("请回复：嗯，我在。什么事？", prompt)

    def test_address_only_turn_can_use_recent_loaded_tool_change_as_a_soft_topic(self) -> None:
        runtime = _Runtime({"speech": "刚好有件事想告诉你。"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            self_state_provider=lambda: {
                "changes": {
                    "last_startup": {
                        "tools_added": ["private_desktop_screenshot"],
                        "added_files": [],
                        "modified_files": [],
                    },
                    "since_startup": {"restart_required": False},
                }
            },
        )

        bridge.handle_turn(self._address_turn())

        prompt = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("private_desktop_screenshot", prompt)
        self.assertIn("语境自然时轻轻提到", prompt)
        self.assertIn("不要列清单", prompt)


if __name__ == "__main__":
    unittest.main()
