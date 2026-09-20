from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryRouter, MemoryService, MemoryStore
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionBatcher
from shinku.qq.delivery import DeliveryResult
from shinku.qq.host import NapCatHost
from shinku.qq.napcat import NapCatVisualInputBridge
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.turns import NapCatTurnDispatcher


class _Runtime:
    def __init__(self, text: str = "收到。") -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "final", "text": self.text}


class IndependentGrayJointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            clock=lambda: 1_800_000_000,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_event(self, *, sender=None):
        runtime = _Runtime("联合回归已收到。")
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="独立主人设",
            memory_service=self.memory,
            sender=sender,
        )
        ingress = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(),
            batcher=AttentionBatcher(window_seconds=2.0),
        )
        host = NapCatHost(
            ingress=ingress,
            visual_bridge=NapCatVisualInputBridge(),
            clock=lambda: 1.0,
        )
        replies = []
        dispatcher = NapCatTurnDispatcher(
            handler=lambda turn: replies.append(bridge.handle_turn(turn)),
            flush_batch=host.flush,
            timer_factory=lambda delay, callback: None,
        )
        host.set_turn_dispatcher(dispatcher)
        event = {
            "post_type": "message",
            "message_type": "group",
            "self_id": 3599477026,
            "group_id": "g-joint",
            "user_id": "u-joint",
            "message_id": "joint-1",
            "message": [
                {"type": "at", "data": {"qq": "3599477026"}},
                {"type": "text", "data": {"text": "请确认联合回归"}},
            ],
        }
        host.handle_event(event, at=1.0)
        receipt = dispatcher.flush()
        return receipt, replies, runtime

    def test_real_onebot_event_reaches_memory_agent_and_sender(self) -> None:
        sent = []

        def send(message):
            sent.append(message)
            return DeliveryResult(True, "sent", ("joint-reply",))

        receipt, replies, runtime = self._run_event(sender=send)

        self.assertEqual((receipt.dispatched, len(replies)), (1, 1))
        self.assertEqual((replies[0].delivery_status, replies[0].sent), ("sent", True))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].segments[0].text, "联合回归已收到。")
        self.assertEqual(len(runtime.calls), 1)
        records = self.memory.store.search(query="请确认联合回归", scope="group:g-joint")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "请确认联合回归")

    def test_sender_disabled_keeps_memory_and_never_delivers(self) -> None:
        receipt, replies, _runtime = self._run_event(sender=None)

        self.assertEqual(receipt.dispatched, 1)
        self.assertEqual((replies[0].delivery_status, replies[0].sent), ("dry_run", False))
        records = self.memory.store.search(query="请确认联合回归", scope="group:g-joint")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "请确认联合回归")


if __name__ == "__main__":
    unittest.main()
