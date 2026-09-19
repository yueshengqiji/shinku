from __future__ import annotations

import unittest

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
        self.assertEqual(runtime.calls[0]["system_prompt"], "真红主人设")
        self.assertEqual(runtime.calls[0]["user_images"], [{"data_url": "data:image/png;base64,aGVsbG8="}])
        self.assertIn("图片附件", runtime.calls[0]["history_turns"][0]["content"])

    def test_default_sender_is_dry_run(self) -> None:
        bridge = QQAgentBridge(runtime=_Runtime({"speech": "只模拟。"}), system_prompt="p")
        result = bridge.handle_turn(self._turn())
        self.assertEqual((result.final_text, result.sent, result.delivery_status), ("只模拟。", False, "dry_run"))

    def test_invalid_runtime_result_does_not_send(self) -> None:
        runtime = _Runtime({"kind": "wait", "text": "还缺一份资料。"})
        bridge = QQAgentBridge(runtime=runtime, system_prompt="p")
        result = bridge.handle_turn(self._turn())
        self.assertEqual((result.agent_status, result.final_text, result.delivery_status), ("waiting_user", "还缺一份资料。", "dry_run"))


if __name__ == "__main__":
    unittest.main()
