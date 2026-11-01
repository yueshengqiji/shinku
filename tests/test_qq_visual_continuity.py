from __future__ import annotations
from reply_review_fixture import dialogue_review_result

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.chat_flow import ChatFlowController
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatVisualInputBridge
from test_qq_visual_provenance import Harness, batch, message


REACTIONS = (
    "老婆给的好", "真好看", "这也太可爱了吧", "不错诶", "我很喜欢",
    "好喜欢啊", "太丑了", "哈哈哈", "我要大的", "那我选小的",
)


@pytest.mark.parametrize("text", REACTIONS)
@pytest.mark.parametrize("kind", ("private", "group"))
def test_short_same_author_reaction_keeps_the_actual_image_and_provenance(text, kind):
    h = Harness()
    h.receive(message("image", text="", image=True, kind=kind))
    h.now += 5
    turn = h.choose(h.receive(message(text=text, kind=kind)))
    assert turn.visual_context_inherited
    assert turn.visual_context_source_message_id == "image"
    image = QQAgentBridge._images(turn)[0]
    assert image["source_message_id"] == "image"
    assert image["source_sender_id"] == "owner"
    assert image["reference_kind"] == "inherited"
    assert not turn.visuals_for("query")


@pytest.mark.parametrize("text", REACTIONS)
def test_other_author_reaction_never_borrows_the_image_without_exact_quote(text):
    h = Harness()
    h.receive(message("image", "other", text="", image=True), scheduled=False)
    assert not QQAgentBridge._images(h.receive(message(text=text)))


@pytest.mark.parametrize("text", (
    "好", "嗯", "这是什么", "今天心情很好", "晚饭很好吃", "老婆帮我查天气",
    "今天天气好", "老婆给的好，不过今晚吃什么", "好看，换个话题聊工作",
    "难道联想不到月亮这个形态是被我们炸出来的吗",
))
def test_unrelated_or_ambiguous_text_still_closes_implicit_visual_continuity(text):
    h = Harness()
    h.receive(message("image", text="", image=True))
    current = h.receive(message(text=text))
    assert not QQAgentBridge._images(current) and current.visual_candidates
    # LB02: a generic phrase cannot close the topic; the decision confirms it.
    h.choose(current, action="none", topic_change=True)
    assert not h.receive(message("later", text="真好看")).visual_candidates
    assert h.choose(h.receive(message("quoted", reply="image"))).visual_context_inherited


def test_reaction_window_is_shorter_than_the_explicit_image_reference_ttl():
    h = Harness(visual_context_ttl_seconds=120)
    h.receive(message("image", text="", image=True))
    h.now += 61
    assert not QQAgentBridge._images(h.receive(message(text="老婆给的好")))
    assert h.choose(h.receive(message("quoted", reply="image"))).visual_context_inherited


def test_natural_reactions_use_existing_followup_budget_without_resetting_it():
    h = Harness(visual_followup_turns=1)
    h.receive(message("image", text="", image=True))
    assert h.choose(h.receive(message(text="老婆给的好"))).visual_context_inherited
    assert not h.receive(message("second", text="真好看")).visual_candidates


def test_exhausted_new_image_does_not_fall_back_to_an_older_own_image():
    h = Harness(visual_followup_turns=1)
    h.receive(message("older", text="", image=True))
    h.receive(message("latest", text="", image=True))
    assert h.choose(h.receive(message(text="真好看")), "latest").visual_context_source_message_id == "latest"
    assert not h.receive(message("second", text="好喜欢啊")).visual_candidates


def test_nonaddressed_same_author_reaction_keeps_the_thread_but_topic_change_closes_it():
    h = Harness()
    h.receive(message("image", text="", image=True), scheduled=False)
    h.receive(message("reaction", text="老婆给的好"), scheduled=False)
    assert h.choose(h.receive(message())).visual_context_inherited
    h.receive(message("topic", text="晚饭吃什么"), scheduled=False)
    later = h.receive(message("later", text="真好看"))
    assert later.visual_candidates
    h.choose(later, action="none", topic_change=True)
    assert not h.receive(message("closed", text="真好看")).visual_candidates


def test_secondary_member_reaction_cannot_select_primary_members_image():
    h = Harness()
    h.receive(message("image", text="", image=True), scheduled=False)
    primary = message(text="晚饭吃什么")
    secondary = message("other-query", "other", "真好看")
    h.submit(primary)
    h.submit(secondary)
    h.pending.append(batch(primary, secondary))
    h.dispatcher.flush()
    assert not QQAgentBridge._images(h.turns[-1])


@pytest.mark.parametrize("text", REACTIONS)
def test_image_only_previous_answer_is_carried_to_the_matching_visual_followup(text):
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    assert first.combined_text == ""
    controller.finish_turn(first, final_text="刚好一人一杯，你要大的还是小的？", expanded=False)
    h.now += 5
    current = h.choose(h.receive(message(text=text, kind="private")))
    decision = controller.begin_turn(current)
    assert decision.same_topic
    assert "上一轮用户：[图片附件" in decision.previous_exchange
    assert "上一轮真红：刚好一人一杯" in decision.previous_exchange


def test_missing_current_image_does_not_erase_the_image_only_previous_answer():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    h.now += 5
    current = h.receive(message(text="老婆给的好", kind="private"))
    current = replace(current, visual_inputs=(), visual_context_inherited=False, visual_context_source_message_id="")
    assert "上一轮真红：刚好一人一杯。" in controller.begin_turn(current).previous_exchange


@pytest.mark.parametrize("text", ("今晚吃什么", "今天心情很好", "真红帮我查天气", "那我们明天再说"))
def test_image_only_exchange_is_not_attached_to_an_unrelated_short_topic(text):
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    h.now += 5
    current = h.receive(message(text=text, kind="private"))
    decision = controller.begin_turn(current)
    assert decision.reason == "visual_context_candidate" and not decision.same_topic
    assert "刚好一人一杯" in decision.previous_exchange
    assert not QQAgentBridge._images(current)


def test_visual_exchange_does_not_leak_across_senders_or_conversations():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True))
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    assert controller.begin_turn(h.receive(message(sender="other", text="真好看"))).previous_exchange == ""
    assert controller.begin_turn(h.receive(message(group="other-group", text="真好看"))).previous_exchange == ""


def test_a_new_direct_picture_does_not_receive_the_previous_pictures_answer():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    current = h.receive(message("new-image", text="真好看", image=True, kind="private"))
    assert controller.begin_turn(current).previous_exchange == ""


def test_picture_exchange_reactions_do_not_outlive_the_short_reaction_window():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    h.now += 61
    current = h.receive(message(text="老婆给的好", kind="private"))
    assert current.visual_candidates and not current.visual_context_inherited
    assert "刚好一人一杯" in controller.begin_turn(current).previous_exchange


def test_failed_picture_input_preserves_the_actual_previous_failure_not_a_fake_description():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    raw = dict(message("image", text="", image=True, kind="private").raw_event)
    raw["message"] = [{"type": "image", "data": {"url": "https://image.test/unread.png"}}]
    first = h.receive(IncomingMessage.from_event(raw))
    assert not QQAgentBridge._images(first)
    controller.finish_turn(first, final_text="这张还没能读取，暂时不能判断内容。", expanded=False)
    h.now += 5
    current = h.receive(message(text="真好看", kind="private"))
    decision = controller.begin_turn(current)
    assert "上一轮真红：这张还没能读取" in decision.previous_exchange
    assert "一人一杯" not in decision.previous_exchange


def test_same_source_followup_works_even_when_the_first_response_finished_after_reaction_arrival():
    h = Harness()
    controller = ChatFlowController(clock=lambda: h.now)
    first = h.receive(message("image", text="", image=True, kind="private"))
    h.now += 5
    # Dispatcher builds the followup while the first serial handler is still generating.
    current = h.receive(message(text="老婆给的好", kind="private"))
    h.now += 90
    controller.finish_turn(first, final_text="刚好一人一杯。", expanded=False)
    assert "上一轮真红：刚好一人一杯。" in controller.begin_turn(current).previous_exchange


def test_bridge_passes_image_and_previous_answer_and_does_not_claim_an_unread_image():
    calls = []
    def respond(**kw):
        calls.append(kw)
        review = dialogue_review_result(kw)
        if review is not None:
            return review
        main_count = sum(not str(row.get("prompt_cache_key", "")).endswith(":dialogue-review:v5") for row in calls)
        if main_count == 1:
            return {"speech": "刚好一人一杯，你要大的还是小的？"}
        if main_count == 2:
            import json
            block = next(item for item in kw["system_extra_blocks"] if item.startswith("【图片候选待确认】"))
            rows = json.loads(block[block.index('{"candidates"'):])["candidates"]
            return {"kind": "final", "text": "", "visual_context": {"action": "select", "source_ids": [rows[0]["source_id"]]}}
        return {"speech": "那就收好，给你大的。"}
    runtime = SimpleNamespace(call_chat_json=respond)
    bridge = QQAgentBridge(runtime=runtime, system_prompt="真红", memory_service=None, sender=None)
    h = Harness()
    h.dispatcher.handler = bridge.handle_turn
    first = message("image", text="", image=True, kind="private")
    h.submit(first)
    h.pending.append(batch(first))
    h.dispatcher.flush()
    h.now += 5
    followup = message(text="老婆给的好", kind="private")
    h.submit(followup)
    h.pending.append(batch(followup))
    h.dispatcher.flush()
    assert len(calls) == 3
    main_calls = [row for row in calls if not str(row.get("prompt_cache_key", "")).endswith(":dialogue-review:v5")]
    assert len(main_calls) == 3
    assert not main_calls[1]["user_images"]
    assert main_calls[2]["user_images"][0]["source_message_id"] == "image"
    assert "上一轮真红：刚好一人一杯" in main_calls[2]["user_prompt"]
    assert "老婆给的好" in main_calls[2]["user_prompt"]
    assert "不等于之前读图失败" in "\n".join(calls[2]["system_extra_blocks"])
