from __future__ import annotations

import unittest

from shinku.qq.chat_flow import (
    CHAT_FLOW_SYSTEM_GUIDANCE,
    ChatFlowController,
    append_chat_flow_guidance,
    is_address_only_message,
)
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def _turn(text: str, *, message_id: str, sender_id: str = "u-1", conversation_type: str = "private") -> QQTurn:
    event = {
        "post_type": "message",
        "message_type": conversation_type,
        "user_id": sender_id,
        "group_id": "g-1" if conversation_type == "group" else None,
        "message_id": message_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    message = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id="g-1" if conversation_type == "group" else sender_id,
        conversation_type=conversation_type,
        primary_message_id=message_id,
        message_ids=(message_id,),
        combined_text=text,
        messages=(message,),
    )


class ChatFlowTests(unittest.TestCase):
    def test_address_only_signal_leaves_real_requests_to_the_model(self) -> None:
        for value in (
            "真红",
            "真红真红老婆",
            "真红？在吗",
            "老婆啊",
            "在吗",
            "早",
            "午安",
            "晚安",
            "你好",
            "",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_address_only_message(value))
        for value in ("老婆我想学网安", "帮我查一下更新", "你为什么生气", "真红看这张图"):
            with self.subTest(value=value):
                self.assertFalse(is_address_only_message(value))

    def test_address_only_decision_is_a_soft_context_signal(self) -> None:
        controller = ChatFlowController()
        decision = controller.begin_turn(_turn("老婆", message_id="m-address"))
        self.assertTrue(decision.address_only)
        self.assertTrue(decision.as_dict()["address_only"])
        self.assertFalse(decision.allow_expansion)

    def test_new_turn_then_same_topic_followup_then_cooldown(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)

        first = _turn("端口为什么没有启动", message_id="m-1")
        decision = controller.begin_turn(first)
        self.assertEqual(decision.stage, "fresh")
        self.assertFalse(decision.allow_expansion)
        controller.finish_turn(first, final_text="先检查服务是否还在运行。", expanded=False)

        clock.value += 5
        followup = _turn("那日志要看哪里", message_id="m-2")
        decision = controller.begin_turn(followup)
        self.assertEqual(decision.stage, "followup")
        self.assertTrue(decision.same_topic)
        self.assertTrue(decision.allow_expansion)
        controller.finish_turn(followup, final_text="我在这里，继续看日志。", expanded=True)

        clock.value += 5
        next_turn = _turn("还是刚才那个端口", message_id="m-3")
        decision = controller.begin_turn(next_turn)
        self.assertEqual(decision.stage, "cooldown")
        self.assertFalse(decision.allow_expansion)
        self.assertEqual(decision.reason, "same_topic_after_expansion")
        controller.finish_turn(next_turn, final_text="先这样。", expanded=False)
        clock.value += 5
        resumed = _turn("那日志还要继续看吗", message_id="m-4")
        decision = controller.begin_turn(resumed)
        self.assertEqual(decision.stage, "followup")
        self.assertTrue(decision.allow_expansion)

    def test_topic_switch_resets_and_group_state_is_per_sender(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("图片加载失败", message_id="m-1", conversation_type="group")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="先检查图片地址。", expanded=False)

        clock.value += 2
        other_sender = _turn("今天吃什么", message_id="m-2", sender_id="u-2", conversation_type="group")
        decision = controller.begin_turn(other_sender)
        self.assertEqual(decision.stage, "fresh")
        self.assertEqual(decision.reason, "new_topic_or_expired")

        clock.value += 2
        switched = _turn("今天晚上我们要不要一起去电影院看一部新的科幻电影", message_id="m-3", conversation_type="group")
        decision = controller.begin_turn(switched)
        self.assertEqual(decision.stage, "fresh")
        self.assertFalse(decision.same_topic)

    def test_repeated_voice_habit_is_soft_guidance_not_a_filter(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock, voice_window=4)
        first = _turn("今天有点累", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="嗯……我在这里。", expanded=False)
        clock.value += 2
        second = _turn("还是有点累", message_id="m-2")
        decision = controller.begin_turn(second)
        self.assertEqual(decision.habit_guidance, "")
        controller.finish_turn(second, final_text="嗯，我在这里。", expanded=False)
        clock.value += 2
        third = _turn("现在好多了", message_id="m-3")
        decision = controller.begin_turn(third)
        self.assertIn("同类开场", decision.habit_guidance)
        self.assertIn("自然换一种开头", decision.habit_guidance)

    def test_repeated_you_address_opening_gets_dynamic_hint_only_after_repeat(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock, voice_window=4)
        first = _turn("你还好吗", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="你啊……又把事情拖到最后了。", expanded=False)

        clock.value += 2
        second = _turn("我只是忘了", message_id="m-2")
        decision = controller.begin_turn(second)
        self.assertEqual(decision.habit_guidance, "")
        controller.finish_turn(second, final_text="你啊，这种借口倒是很熟练。", expanded=False)

        clock.value += 2
        third = _turn("那现在怎么办", message_id="m-3")
        decision = controller.begin_turn(third)
        self.assertIn("重复使用同类开场", decision.habit_guidance)
        self.assertIn("自然换一种开头", decision.habit_guidance)

    def test_different_opening_families_do_not_trigger_hint(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock, voice_window=4)
        first = _turn("今天有点累", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="嗯……先休息一下。", expanded=False)

        clock.value += 2
        second = _turn("现在好多了", message_id="m-2")
        decision = controller.begin_turn(second)
        self.assertEqual(decision.habit_guidance, "")
        controller.finish_turn(second, final_text="你啊，恢复得还挺快。", expanded=False)

        clock.value += 2
        third = _turn("那继续做事吧", message_id="m-3")
        decision = controller.begin_turn(third)
        self.assertEqual(decision.habit_guidance, "")

    def test_system_guidance_is_stable_and_idempotent(self) -> None:
        value = append_chat_flow_guidance("真红主人设")
        self.assertIn(CHAT_FLOW_SYSTEM_GUIDANCE, value)
        self.assertIn("单独点名", value)
        self.assertIn("不要把点名自动回答成同一套确认句", value)
        self.assertEqual(append_chat_flow_guidance(value), value)


if __name__ == "__main__":
    unittest.main()
