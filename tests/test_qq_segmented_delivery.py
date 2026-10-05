from __future__ import annotations

import unittest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.config import NapCatConnectionConfig
from shinku.qq.delivery import DeliveryResult, build_reply
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatActionTransport, NapCatHttpActionCaller
from shinku.qq.reply_host import ReplyHost
from shinku.qq.turns import QQTurn


class _Runtime:
    def call_chat_json(self, **kwargs):
        return {"speech": "unused"}


def _turn() -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": "owner-1",
            "message_id": "incoming-1",
            "message": [{"type": "text", "data": {"text": "看一下"}}],
        }
    )
    return QQTurn(
        conversation_id="owner-1",
        conversation_type="private",
        primary_message_id="incoming-1",
        message_ids=("incoming-1",),
        combined_text="看一下",
        messages=(message,),
    )


class SegmentedQQDeliveryTests(unittest.TestCase):
    def _bridge(self, sender):
        return QQAgentBridge(
            runtime=_Runtime(),
            system_prompt="test",
            sender=sender,
            reply_host=ReplyHost(),
        )

    def test_attempts_every_segment_after_an_ambiguous_timeout(self) -> None:
        attempted: list[str] = []

        def sender(message):
            text = message.segments[0].text
            attempted.append(text)
            if len(attempted) == 1:
                raise TimeoutError("response timed out")
            # NapCat can accept a send without returning a message id.
            return DeliveryResult(True, "sent")

        result = self._bridge(sender)._deliver_text(
            _turn(),
            "我看过了。确实和你说的一样。后面还有一句。",
            agent_status="completed",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertEqual(attempted, ["我看过了。", "确实和你说的一样。", "后面还有一句。"])
        self.assertFalse(result.sent)
        self.assertEqual(result.delivery_status, "partial")
        self.assertIn("segment_1:transport_error:TimeoutError", result.reason)

    def test_attempts_later_segments_when_a_middle_segment_is_rejected(self) -> None:
        attempted: list[str] = []

        def sender(message):
            text = message.segments[0].text
            attempted.append(text)
            if len(attempted) == 2:
                return DeliveryResult(False, "rejected", error="test rejection")
            return DeliveryResult(True, "sent", (f"out-{len(attempted)}",))

        result = self._bridge(sender)._deliver_text(
            _turn(),
            "第一句。第二句。第三句。",
            agent_status="completed",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertEqual(attempted, ["第一句。", "第二句。", "第三句。"])
        self.assertFalse(result.sent)
        self.assertEqual(result.delivery_status, "partial")
        self.assertEqual(result.delivery.message_ids, ("out-1", "out-3"))

    def test_success_without_message_ids_counts_as_delivered(self) -> None:
        attempted: list[str] = []

        def sender(message):
            attempted.append(message.segments[0].text)
            return DeliveryResult(True, "sent")

        result = self._bridge(sender)._deliver_text(
            _turn(),
            "第一句。第二句。",
            agent_status="completed",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertEqual(attempted, ["第一句。", "第二句。"])
        self.assertTrue(result.sent)
        self.assertEqual(result.delivery_status, "sent")

    def test_napcat_timeout_defaults_to_thirty_seconds(self) -> None:
        self.assertEqual(NapCatConnectionConfig().timeout, 30.0)
        self.assertEqual(NapCatConnectionConfig.from_env({}).timeout, 30.0)

        observed: list[float] = []

        class _Response:
            def read(self) -> bytes:
                return b'{"status":"ok","retcode":0}'

        def opener(request, *, timeout):
            observed.append(timeout)
            return _Response()

        caller = NapCatHttpActionCaller("http://napcat.test", opener=opener)
        result = NapCatActionTransport(caller).send(
            build_reply("owner-1", "收到", conversation_type="private")
        )
        self.assertEqual(observed, [30.0])
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
