from __future__ import annotations

import unittest

from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionPolicy
from shinku.qq.commands import QQCommandState


def _event(message_id: str, sender_id: str, text: str, *, group_id: str = "group-1", at_other: bool = False) -> dict:
    segments = []
    if at_other:
        segments.append({"type": "at", "data": {"qq": "someone-else"}})
    segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": group_id,
        "user_id": sender_id,
        "message_id": message_id,
        "message": segments,
    }


def _adapter(draw: float) -> QQIngressAdapter:
    return QQIngressAdapter(
        attention_policy=AttentionPolicy(
            autonomous_enabled=True,
            autonomous_probability=0.12,
            owner_user_ids=frozenset({"owner"}),
            random_source=lambda: draw,
        )
    )


class AutonomousAttentionScoringTests(unittest.TestCase):
    def test_owner_and_question_get_small_bounded_local_bonuses(self) -> None:
        owner = _adapter(0.19).ingest(_event("owner-1", "owner", "今天聊什么"), at=1.0)
        other = _adapter(0.19).ingest(_event("other-1", "other", "今天聊什么"), at=1.0)
        question = _adapter(0.16).ingest(_event("question-1", "other", "今天聊什么？"), at=1.0)
        all_signals = _adapter(0.31).ingest(_event("all-1", "owner", "今天聊什么？"), at=1.0)

        self.assertTrue(owner.scheduled)
        self.assertAlmostEqual(owner.attention.probability, 0.20)
        self.assertEqual(owner.attention.signals, ("owner",))
        recorded = _adapter(0.19)
        recorded.ingest(_event("trace-1", "owner", "今天聊什么"), at=1.0)
        attention_trace = recorded.attention_trace_store.snapshot(limit=1)["items"][0]["attention"]
        self.assertAlmostEqual(attention_trace["probability"], 0.20)
        self.assertEqual(attention_trace["signals"], ["owner"])
        self.assertFalse(other.scheduled)
        self.assertAlmostEqual(other.attention.probability, 0.12)
        self.assertTrue(question.scheduled)
        self.assertAlmostEqual(question.attention.probability, 0.17)
        self.assertFalse(all_signals.scheduled)
        self.assertAlmostEqual(all_signals.attention.probability, 0.25)

    def test_recent_directed_thread_boosts_same_speaker_only(self) -> None:
        adapter = _adapter(0.21)
        addressed = adapter.ingest(_event("address-1", "owner", "真红，听我说"), at=1.0)
        followup = adapter.ingest(_event("follow-1", "owner", "还有一件事"), at=2.0)
        other = _adapter(0.21)
        other.ingest(_event("address-2", "owner", "真红，听我说"), at=1.0)
        stranger = other.ingest(_event("follow-2", "other", "还有一件事"), at=2.0)

        self.assertEqual(addressed.route.reason, "direct_name")
        self.assertTrue(followup.scheduled)
        self.assertAlmostEqual(followup.attention.probability, 0.30)
        self.assertIn("directed_followup", followup.attention.signals)
        self.assertFalse(stranger.scheduled)

    def test_combined_bonuses_are_capped_and_followup_expires(self) -> None:
        adapter = _adapter(0.31)
        adapter.ingest(_event("address-cap", "owner", "真红，听我说"), at=1.0)
        capped = adapter.ingest(_event("follow-cap", "owner", "还有一件事？"), at=2.0)
        self.assertTrue(capped.scheduled)
        self.assertAlmostEqual(capped.attention.probability, 0.32)

        expired = _adapter(0.21)
        expired.ingest(_event("address-expire", "owner", "真红，听我说"), at=1.0)
        late = expired.ingest(_event("follow-expire", "owner", "还有一件事"), at=122.0)
        self.assertFalse(late.scheduled)
        self.assertAlmostEqual(late.attention.probability, 0.20)

    def test_autonomous_cooldown_is_per_group_and_never_blocks_direct_address(self) -> None:
        adapter = _adapter(0.0)
        first = adapter.ingest(_event("first", "other", "在讨论什么"), at=1.0)
        # Scheduling is not a send: a model silence/failure must not start cooldown.
        unsent = adapter.ingest(_event("unsent", "other", "你怎么看"), at=1.5)
        self.assertNotEqual(unsent.attention.reason, "autonomous_cooldown")
        adapter.note_group_reply(
            "group-1", recipient_ids=("other",), message_ids=("bot-1",), autonomous=True, at=1.6,
        )
        cooled = adapter.ingest(_event("second", "owner", "想听听你的看法？"), at=2.0)
        direct = adapter.ingest(_event("third", "owner", "真红，帮我看一下"), at=3.0)
        other_group = adapter.ingest(_event("fourth", "other", "另一个群", group_id="group-2"), at=4.0)
        later = adapter.ingest(_event("fifth", "other", "又有新消息", group_id="group-1"), at=92.0)

        self.assertTrue(first.scheduled)
        self.assertEqual(cooled.attention.reason, "autonomous_cooldown")
        self.assertTrue(direct.scheduled)
        self.assertEqual(direct.route.reason, "direct_name")
        self.assertTrue(other_group.scheduled)
        self.assertTrue(later.scheduled)

    def test_at_other_and_group_off_are_not_bypassed_by_owner_bonus(self) -> None:
        adapter = _adapter(0.0)
        state = QQCommandState(autonomous_reply_enabled=True)
        state.set_autonomous_reply_enabled(False, "group-off")
        adapter.set_command_state(state)

        at_other = adapter.ingest(_event("at-other", "owner", "你觉得呢？", at_other=True), at=1.0)
        group_off = adapter.ingest(_event("group-off", "owner", "你觉得呢？", group_id="group-off"), at=2.0)

        self.assertEqual(at_other.attention.reason, "autonomous_at_other_member")
        self.assertEqual(group_off.attention.reason, "autonomous_reply_disabled")

    def test_zero_base_probability_stays_off_even_for_owner(self) -> None:
        adapter = QQIngressAdapter(
            attention_policy=AttentionPolicy(
                autonomous_enabled=True,
                autonomous_probability=0.0,
                owner_user_ids=frozenset({"owner"}),
                random_source=lambda: 0.0,
            )
        )
        result = adapter.ingest(_event("zero", "owner", "有人在吗？"), at=1.0)
        self.assertFalse(result.scheduled)
        self.assertEqual(result.attention.probability, 0.0)


if __name__ == "__main__":
    unittest.main()
