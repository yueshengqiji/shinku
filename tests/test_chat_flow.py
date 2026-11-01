from __future__ import annotations
from reply_review_fixture import dialogue_review_result

import unittest

from shinku.qq.chat_flow import (
    CHAT_FLOW_SYSTEM_GUIDANCE,
    ChatFlowController,
    append_chat_flow_guidance,
    is_address_only_message,
)
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
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
        self.assertIn("上一轮真红：先这样。", decision.previous_exchange)

    def test_short_followup_carries_only_the_immediately_previous_exchange(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("过来陪我睡睡觉", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(
            first,
            final_text="……我现在没办法陪你做那种事。不过，可以陪你说说话。",
            expanded=False,
        )

        clock.value += 22
        followup = _turn("为什么没有办法", message_id="m-2")
        decision = controller.begin_turn(followup)

        self.assertEqual(decision.stage, "followup")
        self.assertIn("上一轮用户：过来陪我睡睡觉", decision.previous_exchange)
        self.assertIn("上一轮真红：……我现在没办法陪你做那种事", decision.previous_exchange)

        controller.finish_turn(followup, final_text="因为现在不方便。", expanded=False)
        clock.value += 2
        next_followup = _turn("具体哪里不方便", message_id="m-3")
        next_decision = controller.begin_turn(next_followup)
        self.assertIn("上一轮用户：为什么没有办法", next_decision.previous_exchange)
        self.assertIn("上一轮真红：因为现在不方便。", next_decision.previous_exchange)

    def test_short_new_topic_does_not_attach_unrelated_private_exchange(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("请记住：我喜欢安静的环境", message_id="m-1", conversation_type="group")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="收到。", expanded=False)

        clock.value += 2
        unrelated = _turn("今天继续做项目", message_id="m-2", conversation_type="group")
        decision = controller.begin_turn(unrelated)

        self.assertEqual(decision.stage, "followup")
        self.assertEqual(decision.previous_exchange, "")

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
        self.assertEqual(decision.previous_exchange, "")

        clock.value += 2
        switched = _turn("今天晚上我们要不要一起去电影院看一部新的科幻电影", message_id="m-3", conversation_type="group")
        decision = controller.begin_turn(switched)
        self.assertEqual(decision.stage, "fresh")
        self.assertFalse(decision.same_topic)
        self.assertEqual(decision.previous_exchange, "")

    def test_repeated_voice_habit_is_soft_guidance_not_a_filter(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock, voice_window=4)
        first = _turn("今天有点累", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="嗯……我在这里。", expanded=False)
        clock.value += 2
        second = _turn("还是有点累", message_id="m-2")
        decision = controller.begin_turn(second)
        self.assertIn("前几轮已经明确表达过陪伴", decision.habit_guidance)
        self.assertNotIn("同类开场", decision.habit_guidance)
        controller.finish_turn(second, final_text="嗯，我在这里。", expanded=False)
        clock.value += 2
        third = _turn("现在好多了", message_id="m-3")
        decision = controller.begin_turn(third)
        self.assertIn("同类开场", decision.habit_guidance)
        self.assertIn("无须为了变化而换词", decision.habit_guidance)

    def test_soft_ack_keeps_original_reply_and_does_not_imply_hearing_failure(self) -> None:
        controller = ChatFlowController()
        first = _turn("下午没课了", message_id="m-soft-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="嗯，下午有空了。", expanded=False, reply_message_ids=("r-soft-1",))
        second = _turn("想一起看电影", message_id="m-soft-2")
        decision = controller.begin_turn(second)
        self.assertEqual(decision.habit_guidance, "")
        self.assertIn("嗯，下午有空了。", decision.previous_exchange)
        controller.finish_turn(second, final_text="……嗯。一起看吧。", expanded=False, reply_message_ids=("r-soft-2",))
        third = _turn("那就看上次那部", message_id="m-soft-3")
        decision = controller.begin_turn(third)
        self.assertIn("具体内容", decision.habit_guidance)
        self.assertIn("仍可自然保留", decision.habit_guidance)
        self.assertIn("无须为了变化而换词", decision.habit_guidance)
        self.assertNotIn("再说一遍", decision.habit_guidance)
        self.assertNotIn("没接好", decision.habit_guidance)
        self.assertIn("……嗯。一起看吧。", decision.previous_exchange)

    def test_agreement_is_not_a_new_banned_opening(self) -> None:
        controller = ChatFlowController()
        for index in range(3):
            turn = _turn("这样更清楚", message_id=f"m-agree-{index}")
            decision = controller.begin_turn(turn)
            self.assertEqual(decision.habit_guidance, "")
            controller.finish_turn(turn, final_text="确实，这样更清楚。", expanded=False)

    def test_recent_companionship_and_physical_limit_are_not_repeated_on_ellipsis(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("过来陪我睡睡觉", message_id="m-comfort-1")
        controller.begin_turn(first)
        controller.finish_turn(
            first,
            final_text=(
                "因为碰不到你啊。就算你想，我现在也只能像这样陪你说说话。"
                "快点躺好，我在这里陪着你。"
            ),
            expanded=False,
        )

        clock.value += 5
        followup = _turn("上面我说的事情啊", message_id="m-comfort-2")
        decision = controller.begin_turn(followup)

        self.assertTrue(decision.same_topic)
        self.assertIn("不要再补", decision.habit_guidance)
        self.assertIn("不要再重复限制说明", decision.habit_guidance)

    def test_direct_reask_or_direct_boundary_question_remains_answerable(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("过来陪我睡睡觉", message_id="m-comfort-3")
        controller.begin_turn(first)
        controller.finish_turn(
            first,
            final_text="我在这里陪着你。碰不到你，但可以陪你说话。",
            expanded=False,
        )

        clock.value += 2
        direct_request = _turn("再陪我一会儿", message_id="m-comfort-4")
        decision = controller.begin_turn(direct_request)
        self.assertNotIn("不要再补", decision.habit_guidance)

        controller.finish_turn(direct_request, final_text="好，再陪一会儿。", expanded=False)
        clock.value += 2
        direct_question = _turn("为什么碰不到我", message_id="m-comfort-5")
        decision = controller.begin_turn(direct_question)
        self.assertNotIn("不再重复限制说明", decision.habit_guidance)

    def test_companionship_echo_guidance_is_scoped_to_the_same_topic(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("陪我睡一会儿", message_id="m-comfort-6")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="我在这里陪着你。", expanded=False)

        clock.value += 2
        unrelated = _turn("换个完全无关的话题，我想问一下端口现在恢复了吗", message_id="m-unrelated")
        decision = controller.begin_turn(unrelated)
        self.assertFalse(decision.same_topic)
        self.assertNotIn("不要再补", decision.habit_guidance)

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
        self.assertIn("同类开场", decision.habit_guidance)
        self.assertIn("无须为了变化而换词", decision.habit_guidance)

    def test_listening_ack_gets_a_one_turn_nudge_across_topic_switch(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock, voice_window=4)
        first = _turn("今天有点累", message_id="m-1")
        controller.begin_turn(first)
        controller.finish_turn(first, final_text="……听到了。", expanded=False)

        clock.value += 2
        second = _turn("换个话题，周末我想去海边走走，你觉得哪段时间人会少一点？", message_id="m-2")
        decision = controller.begin_turn(second)
        self.assertFalse(decision.same_topic)
        self.assertIn("听到了/再说一遍", decision.habit_guidance)
        self.assertIn("不要再次声称没接好", decision.habit_guidance)
        controller.finish_turn(second, final_text="周六早一点出发会清静些。", expanded=False)

        clock.value += 2
        third = _turn("好，那我再看看天气", message_id="m-3")
        decision = controller.begin_turn(third)
        # An ordinary answer breaks the acknowledgement habit; the hint must
        # not linger for the rest of the conversation window.
        self.assertEqual(decision.habit_guidance, "")

    def test_false_repair_opening_is_marked_and_discouraged(self) -> None:
        clock = _Clock()
        controller = ChatFlowController(clock=clock)
        first = _turn("真红", message_id="m-repair-1")
        controller.begin_turn(first)
        controller.finish_turn(
            first,
            final_text="……刚才那句我没接好。你再说一遍？",
            expanded=False,
        )

        clock.value += 2
        second = _turn("这次听清楚了吧", message_id="m-repair-2")
        decision = controller.begin_turn(second)
        self.assertIn("当前消息本身缺失/无法理解", decision.habit_guidance)
        self.assertIn("直接回应这条新消息", decision.habit_guidance)

    def test_local_bridge_gray_run_injects_ack_guard_then_clears_it(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.responses = iter((
                    "……听到了。",
                    "周六早一点出发会清静些。",
                    "好，晚点再看天气。",
                ))
                self.calls = []

            def call_chat_json(self, **kwargs):
                self.calls.append(kwargs)
                review = dialogue_review_result(kwargs)
                if review is not None:
                    return review
                return {"speech": next(self.responses)}

        runtime = Runtime()
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", (f"out-{len(sent)}",)),
        )

        turns = (
            _turn("你刚才听见了吗？", message_id="gray-1"),
            _turn("换个话题，周末我想去海边走走，你觉得哪段时间人会少一点？", message_id="gray-2"),
            _turn("好，那我晚点再查天气", message_id="gray-3"),
        )
        results = [bridge.handle_turn(turn) for turn in turns]

        self.assertEqual([item.final_text for item in results], [
            "……听到了。",
            "周六早一点出发会清静些。",
            "好，晚点再看天气。",
        ])
        self.assertEqual(len(runtime.calls), 3)
        main_calls = [row for row in runtime.calls if not str(row.get("prompt_cache_key", "")).endswith(":dialogue-review:v5")]
        self.assertEqual(len(main_calls), 3)
        second_prompt = "\n".join(main_calls[1]["system_extra_blocks"])
        third_prompt = "\n".join(main_calls[2]["system_extra_blocks"])
        self.assertIn("听到了/再说一遍", second_prompt)
        self.assertIn("不要再次声称没接好", second_prompt)
        self.assertNotIn("听到了/再说一遍", third_prompt)
        self.assertEqual(len(sent), 3)

    def test_system_guidance_is_stable_and_idempotent(self) -> None:
        value = append_chat_flow_guidance("真红主人设")
        self.assertIn(CHAT_FLOW_SYSTEM_GUIDANCE, value)
        self.assertIn("单独点名", value)
        self.assertIn("不要把点名自动回答成同一套确认句", value)
        self.assertEqual(append_chat_flow_guidance(value), value)


if __name__ == "__main__":
    unittest.main()
