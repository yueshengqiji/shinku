from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionPolicy
from shinku.qq.attention_classifier import FEATURE_NAMES, LinearAttentionClassifier
from shinku.qq.commands import QQCommandState


def event(mid: str, sender: str, text: str, *, group: str = "g", at: str = "", reply: str = "") -> dict:
    segments = []
    if at:
        segments.append({"type": "at", "data": {"qq": at}})
    if reply:
        segments.append({"type": "reply", "data": {"id": reply}})
    segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message", "message_type": "group", "group_id": group,
        "user_id": sender, "message_id": mid, "self_id": "bot", "message": segments,
    }


def adapter() -> QQIngressAdapter:
    return QQIngressAdapter(attention_policy=AttentionPolicy(
        autonomous_enabled=True, autonomous_probability=0.0,
        owner_user_ids=frozenset({"owner"}), mode="turn_state",
    ))


class TurnStateAttentionTests(unittest.TestCase):
    def test_live_policy_defaults_to_turn_state(self) -> None:
        policy = AttentionPolicy.from_environment({"SHINKU_QQ_AUTONOMOUS_REPLY_PROBABILITY": "0.12"})
        self.assertEqual(policy.mode, "turn_state")

    def test_quiet_open_question_is_candidate_without_random_draw(self) -> None:
        result = adapter().ingest(event("m1", "guest", "大家怎么看？"), at=100.0)
        self.assertTrue(result.scheduled)
        self.assertEqual(result.attention.reason, "group_open_question")

    def test_plain_statement_is_candidate_for_model_decision(self) -> None:
        result = adapter().ingest(event("m1", "guest", "今天吃了饺子"), at=100.0)
        self.assertTrue(result.scheduled)
        self.assertEqual(result.attention.reason, "group_open_turn")

    def test_directed_other_exchange_does_not_steal_turn(self) -> None:
        a = adapter()
        a.ingest(event("m1", "blue", "你觉得呢？", at="alice"), at=100.0)
        result = a.ingest(event("m2", "alice", "我觉得可以？"), at=101.0)
        self.assertFalse(result.scheduled)
        self.assertEqual(result.attention.reason, "other_thread_active")

    def test_recent_bot_exchange_continues_and_send_starts_cooldown(self) -> None:
        a = adapter()
        first = a.ingest(event("m1", "owner", "大家怎么看？"), at=100.0)
        self.assertTrue(first.scheduled)
        # A candidate that ends in model silence does not start cooldown.
        second = a.ingest(event("m2", "guest", "你们怎么看？"), at=101.0)
        self.assertNotEqual(second.attention.reason, "autonomous_cooldown")
        a.note_group_reply("g", recipient_ids=("owner",), message_ids=("bot-m1",), autonomous=True, at=102.0)
        followup = a.ingest(event("m3", "owner", "那具体怎么做"), at=103.0)
        self.assertEqual(followup.attention.reason, "possible_bot_followup")
        self.assertFalse(followup.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"])
        cooled = a.ingest(event("m4", "guest", "另一件事呢？"), at=104.0)
        self.assertEqual(cooled.attention.reason, "autonomous_cooldown")
        reply = a.ingest(event("m5", "guest", "刚才那句再说说", reply="bot-m1"), at=105.0)
        self.assertEqual(reply.attention.reason, "reply_to_delivered_bot_message")

    def test_group_switch_still_wins(self) -> None:
        a = adapter()
        state = QQCommandState(autonomous_reply_enabled=True)
        state.set_autonomous_reply_enabled(False, "g")
        a.set_command_state(state)
        result = a.ingest(event("m1", "owner", "大家怎么看？"), at=100.0)
        self.assertEqual(result.attention.reason, "autonomous_reply_disabled")

    def test_classifier_is_shadow_by_default_and_requires_labeled_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "attention.json"
            path.write_text(json.dumps({
                "schema": 1, "features": list(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 8.0, "trained_examples": 120,
            }), encoding="utf-8")
            classifier = LinearAttentionClassifier.load(path)
            shadow = QQIngressAdapter(attention_policy=AttentionPolicy(
                autonomous_enabled=True, mode="turn_state", classifier=classifier,
            ))
            ignored = shadow.ingest(event("m1", "guest", "今天吃了饺子"), at=100.0)
            self.assertTrue(ignored.scheduled)
            self.assertGreater(ignored.attention.classifier_score or 0, 0.9)
            self.assertEqual(set(ignored.attention.features), set(FEATURE_NAMES))
            active = QQIngressAdapter(attention_policy=AttentionPolicy(
                autonomous_enabled=True, mode="turn_state", classifier=classifier,
                classifier_mode="active",
            ))
            chosen = active.ingest(event("m2", "guest", "今天吃了饺子"), at=100.0)
            self.assertEqual(chosen.attention.reason, "classifier_candidate")
            at_other = active.ingest(event("m3", "guest", "你说呢？", at="alice"), at=101.0)
            self.assertEqual(at_other.attention.reason, "autonomous_at_other_member")

    def test_offline_training_exports_safe_json_model(self) -> None:
        try:
            import sklearn  # noqa: F401 - optional offline trainer dependency
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        from scripts.train_attention_classifier import main

        with tempfile.TemporaryDirectory() as raw:
            labels = Path(raw) / "labels.jsonl"
            output = Path(raw) / "model.json"
            rows = []
            for index in range(200):
                label = index % 2
                features = {name: 0.0 for name in FEATURE_NAMES}
                features["owner"] = float(label)
                features["question"] = float(label)
                rows.append(json.dumps({"label": label, "features": features}))
            labels.write_text("\n".join(rows), encoding="utf-8")
            with patch("sys.argv", ["train_attention_classifier.py", str(labels), str(output)]):
                self.assertEqual(main(), 0)
            trained = LinearAttentionClassifier.load(output)
            self.assertEqual(trained.trained_examples, 160)


if __name__ == "__main__":
    unittest.main()
