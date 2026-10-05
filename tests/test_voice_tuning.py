from __future__ import annotations

import unittest
from pathlib import Path

from shinku.persona.loader import load_persona
from shinku.qq.chat_flow import ChatFlowController
from shinku.qq.message import IncomingMessage
from shinku.qq.reply_host import ReplyHost, clean_voice_text
from shinku.qq.turns import QQTurn


def _make_turn(text: str, *, sender_id: str = "u-voice", message_id: str = "m-voice") -> QQTurn:
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": sender_id,
        "message_id": message_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    msg = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id=sender_id,
        conversation_type="private",
        primary_message_id=message_id,
        message_ids=(message_id,),
        combined_text=text,
        messages=(msg,),
    )


class VoiceTuningTests(unittest.TestCase):
    def test_clean_voice_text_smooths_dashes_outside_code(self) -> None:
        raw = "漏洞在最后半句——中间那一步他没给，这比论证省事多了。"
        cleaned = clean_voice_text(raw)
        self.assertNotIn("——", cleaned)
        self.assertIn("，", cleaned)
        self.assertEqual(cleaned, "漏洞在最后半句，中间那一步他没给，这比论证省事多了。")

    def test_clean_voice_text_preserves_code_fences(self) -> None:
        raw = "正文解释——看这段代码：\n```python\n# step 1 —— step 2\ndef test():\n    pass\n```\n结论在这里。"
        cleaned = clean_voice_text(raw)
        self.assertIn("# step 1 —— step 2", cleaned)
        self.assertTrue(cleaned.startswith("正文解释，看这段代码："))

    def test_clean_voice_text_trims_debater_closer(self) -> None:
        raw = "真要辩就拿出真正站得住的证据，说吧。"
        cleaned = clean_voice_text(raw)
        self.assertNotIn("说吧", cleaned)
        self.assertTrue(cleaned.endswith("。"))

    def test_reply_host_shapes_with_cleaned_text(self) -> None:
        host = ReplyHost()
        shaped = host.shape_text("前面的问题不大——主要是这句没讲清。")
        for segment in shaped:
            self.assertNotIn("——", segment)

    def test_chat_flow_detects_status_sharing_anchor(self) -> None:
        controller = ChatFlowController()
        turn = _make_turn("我刚忙完，今天好累啊")
        decision = controller.begin_turn(turn)
        self.assertIn("不当照护员", decision.voice_anchor)
        self.assertIn("两拍接住", decision.voice_anchor)

    def test_chat_flow_detects_debate_anchor(self) -> None:
        controller = ChatFlowController()
        turn = _make_turn("你这根本就是胡说，漏洞在逻辑本身")
        decision = controller.begin_turn(turn)
        self.assertIn("观点争论", decision.voice_anchor)
        self.assertIn("严禁把对方的话拆解成前半句/后半段", decision.voice_anchor)

    def test_chat_flow_ordinary_chat_has_no_anchor(self) -> None:
        controller = ChatFlowController()
        turn = _make_turn("明天会下雨吗")
        decision = controller.begin_turn(turn)
        self.assertEqual(decision.voice_anchor, "")

    def test_clean_voice_text_replaces_cynical_coax(self) -> None:
        raw = "……突然说这个干什么。少拿这种话来哄我。"
        cleaned = clean_voice_text(raw)
        self.assertNotIn("哄我", cleaned)
        self.assertEqual(cleaned, "……突然说这个干什么。……头转过去。")

    def test_chat_flow_detects_praise_anchor(self) -> None:
        controller = ChatFlowController()
        turn = _make_turn("谁让真红你这么可爱的")
        decision = controller.begin_turn(turn)
        self.assertIn("轻微难为情", decision.voice_anchor)
        self.assertIn("不要套用固定的惊讶反问", decision.voice_anchor)

    def test_repeated_recent_affection_changes_praise_anchor_to_familiar_warmth(self) -> None:
        controller = ChatFlowController()
        first = _make_turn("你今天真可爱", message_id="m-affection-1")
        first_decision = controller.begin_turn(first)
        self.assertIn("轻微难为情", first_decision.voice_anchor)
        controller.finish_turn(first, final_text="……突然夸这个干什么。", expanded=False)

        second = _make_turn("你刚才说我可爱，我也喜欢你", message_id="m-affection-2")
        second_decision = controller.begin_turn(second)
        self.assertNotIn("近期频繁", second_decision.voice_anchor)
        controller.finish_turn(second, final_text="嗯，我听见了。", expanded=False)

        third = _make_turn("老婆，我也喜欢你", message_id="m-affection-3")
        third_decision = controller.begin_turn(third)
        self.assertIn("近期频繁", third_decision.voice_anchor)
        self.assertIn("不要每次都表现得意外或害羞", third_decision.voice_anchor)

    def test_intimate_address_alone_does_not_trigger_a_surprised_reaction(self) -> None:
        controller = ChatFlowController()
        decision = controller.begin_turn(_make_turn("老婆，帮我看一下这个报错"))
        self.assertIn("称呼本身是熟悉的称谓", decision.voice_anchor)
        self.assertNotIn("轻微难为情", decision.voice_anchor)

    def test_persona_contains_three_beat_rhythm_and_anchors(self) -> None:
        persona_path = Path("private/profile/shinku/persona.md")
        doc = load_persona(persona_path)
        self.assertTrue(doc.configured)
        self.assertIn("三拍结构", doc.text)
        self.assertIn("具体反应 → 自己的判断或理由 →（必要时）一个真问题", doc.text)
        self.assertIn("不用破折号补刀", doc.text)
        self.assertIn("绝不做催人睡觉、叮嘱休息的照护员", doc.text)
        self.assertIn("少拿这种话来哄我", doc.text)
        self.assertIn("熟悉的亲密称呼和近期反复表达的喜欢", doc.text)
        self.assertIn("害羞只是偶尔出现的真实反应", doc.text)


if __name__ == "__main__":
    unittest.main()
