from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionBatch, AttentionDecision
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatVisualInputBridge
from shinku.qq.turns import NapCatTurnDispatcher
from test_contract_c2_5b import _runtime


IMAGE = "data:image/png;base64,aGVsbG8="


def message(mid="query", sender="owner", text="这张图是谁发的", *, image=False, group="g",
            kind="group", reply="", quote=None):
    segments = [{"type": "text", "data": {"text": text}}]
    if reply:
        segments.insert(0, {"type": "reply", "data": {"id": reply}})
    if image:
        segments.append({"type": "image", "data": {"url": IMAGE}})
    result = IncomingMessage.from_event({
        "post_type": "message", "message_type": kind, "group_id": group,
        "message_id": mid, "user_id": sender, "sender": {"nickname": sender + "-name"},
        "message": segments,
    })
    return result.with_quoted_message(quote) if quote is not None else result


def batch(*messages, primary=0):
    return AttentionBatch(messages[0].conversation_id, messages,
                          tuple(AttentionDecision(True, True, 90, "mentioned_bot") for _ in messages), primary)


class Harness:
    def __init__(self, **kwargs):
        self.turns = []
        self.pending = []
        self.now = 10.0
        self.dispatcher = NapCatTurnDispatcher(
            handler=self.turns.append, flush_batch=lambda: self.pending.pop(0) if self.pending else None,
            window_seconds=0, timer_factory=lambda *args: SimpleNamespace(start=lambda: None, cancel=lambda: None),
            clock=lambda: self.now, **kwargs,
        )

    def submit(self, msg, *, scheduled=True, admitted=True, closed=None):
        return self.dispatcher.submit(SimpleNamespace(
            ingress=SimpleNamespace(message=msg, admitted=admitted, closed_batch=closed),
            visual=NapCatVisualInputBridge().build(msg), scheduled=scheduled,
        ))

    def receive(self, msg, *, scheduled=True):
        self.submit(msg, scheduled=scheduled)
        if scheduled:
            self.pending.append(batch(msg))
            self.dispatcher.flush()
            return self.turns[-1]


@pytest.mark.parametrize("text", [
    "难道联想不到月亮这个形态是被我们炸出来的吗", "这是什么", "刚才怎么了", "那是谁",
    "它怎么了", "你看到我了吗", "你能看清文字吗", "这个人怎么样", "我买了图书",
    "什么内容", "那我们明天再说", "图谋不轨", "这是地图", "这是我们刚才讨论的话题", "这图书值得看吗",
])
def test_generic_words_do_not_pull_old_image_even_from_same_sender(text):
    h = Harness()
    h.receive(message("image", image=True))
    turn = h.receive(message(text=text))
    assert not turn.visual_context_inherited
    assert QQAgentBridge._images(turn) == []


@pytest.mark.parametrize("text", ["这张图是谁发的", "那张照片像谁", "上一张截图里是什么", "图里是谁", "你看这是谁"])
def test_same_author_explicit_visual_followup_keeps_original_key_and_author(text):
    h = Harness()
    h.receive(message("image", image=True))
    turn = h.receive(message(text=text))
    assert turn.visual_context_inherited
    assert not turn.visuals_for("query")
    assert turn.visuals_for("image")
    images = QQAgentBridge._images(turn)
    assert images[0]["source_message_id"] == "image"
    assert images[0]["source_sender_id"] == "owner"
    assert images[0]["reference_kind"] == "inherited"
    transcript = QQAgentBridge._transcript(turn)
    assert "原图片来源" in transcript[0]["content"]
    assert "图片附件" not in transcript[-1]["content"]


def test_original_incident_other_member_image_then_owner_moon_text_has_no_image():
    h = Harness()
    h.receive(message("526594253", "3498798489", "看看我给你画的", image=True), scheduled=False)
    turn = h.receive(message("1339966470", "3545447411", "难道联想不到月亮这个形态是被我们炸出来的吗"))
    assert not turn.visual_context_inherited
    assert not QQAgentBridge._images(turn)
    assert "图片" not in QQAgentBridge._transcript(turn)[0]["content"]
    assert "attachment_refs" not in QQAgentBridge._memory_message_meta(turn)


def test_cross_author_ambiguous_visual_reference_does_not_inherit_without_quote():
    h = Harness()
    h.receive(message("image", "other", image=True), scheduled=False)
    assert not h.receive(message(text="刚才那张图是谁发的")).visual_context_inherited


def test_exact_cross_author_quote_selects_original_not_latest_image():
    h = Harness()
    h.receive(message("original", "other", image=True), scheduled=False)
    h.receive(message("latest", "third", image=True), scheduled=False)
    turn = h.receive(message(reply="original"))
    assert turn.visual_context_source_message_id == "original"
    assert QQAgentBridge._images(turn)[0]["source_sender_id"] == "other"
    assert not turn.visuals_for("query")
    meta = QQAgentBridge._memory_message_meta(turn)
    assert not meta["has_media"]
    assert meta["attachment_refs"][0]["source_sender_id"] == "other"
    assert meta["attachment_refs"][0]["reference_kind"] == "inherited"


def test_each_sender_retains_their_own_recent_image_after_other_member_posts():
    h = Harness()
    h.receive(message("owner-image", image=True), scheduled=False)
    h.receive(message("other-image", "other", image=True), scheduled=False)
    assert h.receive(message()).visual_context_source_message_id == "owner-image"


def test_no_cross_group_inheritance_even_if_message_ids_collide():
    h = Harness()
    h.receive(message("same-id", "other", image=True), scheduled=False)
    turn = h.receive(message(group="different", reply="same-id"))
    assert not turn.visual_context_inherited
    assert not QQAgentBridge._images(turn)


def test_direct_attachment_keys_are_conversation_scoped():
    h = Harness()
    first = message("same-id", image=True)
    second = message("same-id", "other", image=True, group="different")
    h.submit(first)
    h.submit(second)
    h.pending.extend([batch(first), batch(second)])
    h.dispatcher.flush()
    h.dispatcher.flush()
    assert [QQAgentBridge._images(t)[0]["source_sender_id"] for t in h.turns] == ["owner", "other"]


def test_unrelated_unscheduled_topic_change_closes_implicit_followup_but_exact_quote_still_works():
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    h.receive(message("topic", text="晚饭吃什么"), scheduled=False)
    assert not h.receive(message()).visual_context_inherited
    assert h.receive(message("explicit", reply="image")).visual_context_inherited


@pytest.mark.parametrize("reply", ["", "image"])
def test_expired_image_cannot_be_inherited(reply):
    h = Harness(visual_context_ttl_seconds=5)
    h.receive(message("image", image=True), scheduled=False)
    h.now = 20
    assert not h.receive(message(reply=reply)).visual_context_inherited


def test_followup_budget_is_not_reset_by_inherited_questions():
    h = Harness(visual_followup_turns=1)
    h.receive(message("image", image=True), scheduled=False)
    assert h.receive(message()).visual_context_inherited
    assert not h.receive(message("second")).visual_context_inherited


def test_secondary_sender_text_cannot_trigger_primary_image_inheritance():
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    primary = message(text="什么内容")
    secondary = message("second", "other", "这张图片")
    h.submit(primary)
    h.submit(secondary)
    h.pending.append(batch(primary, secondary))
    h.dispatcher.flush()
    assert not h.turns[-1].visual_context_inherited


def test_new_image_closing_old_question_batch_cannot_leak_backwards():
    h = Harness()
    old = message("old-question")
    h.submit(old)
    h.submit(message("future-image", image=True), closed=batch(old))
    assert not h.turns[-1].visual_context_inherited


def test_rejected_event_does_not_seed_visual_cache():
    h = Harness()
    assert not h.submit(message("image", image=True), admitted=False).accepted
    assert not h.receive(message(reply="image")).visual_context_inherited


def test_image_retention_is_bounded():
    h = Harness()
    for index in range(20):
        h.receive(message(str(index), "other", image=True), scheduled=False)
    assert len(h.dispatcher._visual_context) == 8
    assert not h.dispatcher._visuals
    assert not h.receive(message(reply="0")).visual_context_inherited


def test_quoted_and_own_images_keep_distinct_origins_and_order_in_model_prompt():
    original = message("original", "other", image=True)
    current = message(image=True, reply="original", quote=original)
    h = Harness()
    turn = h.receive(current)
    images = QQAgentBridge._images(turn)
    assert [item["source_sender_id"] for item in images] == ["owner", "other"]
    assert [item["reference_kind"] for item in images] == ["direct", "quoted"]
    assert not turn.visual_context_inherited
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "收到"}), system_prompt="真红")
    parts = bridge._build_prompt(AgentState(
        transcript=QQAgentBridge._transcript(turn), context={"qq_turn": turn, "user_images": images},
    ))
    ledger = parts.user_prompt.split("【本轮实际图片来源，按附加图片顺序编号】\n")[1].split("\n\n")[0]
    sources = json.loads(ledger)
    assert [s["image_index"] for s in sources] == [1, 2]
    assert [s["source_sender_id"] for s in sources] == ["owner", "other"]
    assert "当前提问者不等于原始发送者" in "".join(parts.system_extra_blocks)
    assert "图片属于同一行" not in "".join(parts.system_extra_blocks)


def test_unknown_quote_author_is_not_replaced_by_current_sender_or_message():
    quote = {"attachments": [{"kind": "image", "url": IMAGE}]}
    turn = Harness().receive(message(quote=quote))
    image = QQAgentBridge._images(turn)[0]
    assert image["source_sender_id"] == ""
    assert image["source_message_id"] == ""
    assert image["reference_kind"] == "quoted"


def test_private_visual_followup_remains_supported():
    h = Harness()
    h.receive(message("image", image=True, kind="private"))
    assert h.receive(message(kind="private", text="你看这是谁")).visual_context_inherited


def test_prompt_ledger_matches_runtime_five_image_limit():
    h = Harness()
    msg = message(image=True)
    raw = dict(msg.raw_event)
    raw["message"] = [{"type": "image", "data": {"url": IMAGE}} for _ in range(7)]
    turn = h.receive(IncomingMessage.from_event(raw))
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "收到"}), system_prompt="真红")
    parts = bridge._build_prompt(AgentState(transcript=[], context={"qq_turn": turn}))
    ledger = parts.user_prompt.split("【本轮实际图片来源，按附加图片顺序编号】\n")[1].split("\n\n")[0]
    assert len(json.loads(ledger)) == 5


def test_quoted_attachment_does_not_seed_new_cache_under_quoter():
    h = Harness()
    h.receive(message(quote=message("original", "other", image=True)))
    assert not h.receive(message("second")).visual_context_inherited


@pytest.mark.parametrize("known_sender", [False, True])
def test_embedded_quote_keeps_original_author_or_unknown_instead_of_quoter(known_sender):
    original = {"id": "original", "message": [{"type": "image", "data": {"url": IMAGE}}]}
    if known_sender:
        original["user_id"] = "other"
        original["sender"] = {"nickname": "other-name"}
    msg = IncomingMessage.from_event({
        "post_type": "message", "message_type": "group", "group_id": "g", "user_id": "owner",
        "message_id": "query", "message": [{"type": "reply", "data": original}],
    })
    image = QQAgentBridge._images(Harness().receive(msg))[0]
    assert image["source_sender_id"] == ("other" if known_sender else "")
    assert image["source_sender_label"] == ("other-name" if known_sender else "")
    assert image["source_message_id"] == "original"


def test_unresolved_quote_keeps_provenance_without_model_image():
    original = message("original", "other", image=True)
    raw = original.as_dict()
    raw["attachments"][0]["url"] = "https://image.test/pending.png"
    turn = Harness().receive(message(reply="original", quote=raw))
    assert not QQAgentBridge._images(turn)
    assert turn.as_dict()["visual_inputs"]["query"][0]["source_sender_id"] == "other"
    assert "尚未成功读取" in QQAgentBridge._transcript(turn)[0]["content"]


def test_memory_of_secondary_message_does_not_claim_primary_inherited_attachment():
    h = Harness()
    h.receive(message("image", "other", image=True), scheduled=False)
    primary = message(reply="image")
    secondary = message("second", "third", "好")
    h.submit(primary)
    h.submit(secondary)
    h.pending.append(batch(primary, secondary))
    h.dispatcher.flush()
    meta = QQAgentBridge._memory_message_meta(h.turns[-1], message=secondary)
    assert "attachment_refs" not in meta
    assert "visual_context_inherited" not in meta


def test_actual_multimodal_payload_places_source_metadata_next_to_each_image():
    original = message("original", "other", image=True)
    turn = Harness().receive(message(image=True, reply="original", quote=original))
    runtime = _runtime()
    payload = runtime._build_payload(
        bundle=runtime.chat, system_prompt="persona", user_prompt="当前问题", temperature=0.2,
        user_images=QQAgentBridge._images(turn),
    )
    content = payload["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["text", "text", "image_url", "text", "image_url"]
    assert json.loads(content[1]["text"].split("\n", 1)[1])["source_sender_id"] == "owner"
    assert json.loads(content[3]["text"].split("\n", 1)[1])["source_sender_id"] == "other"


def test_unknown_source_still_has_explicit_unknown_marker_in_multimodal_payload():
    turn = Harness().receive(message(quote={"attachments": [{"kind": "image", "url": IMAGE}]}))
    runtime = _runtime()
    items = runtime._coerce_image_items(QQAgentBridge._images(turn))
    assert items[0]["type"] == "text"
    assert json.loads(items[0]["text"].split("\n", 1)[1])["source_sender_id"] == ""
    assert items[1]["type"] == "image_url"


def test_source_markers_do_not_increase_runtime_image_limit():
    runtime = _runtime()
    items = runtime._coerce_image_items([
        {"data_url": IMAGE, "source_message_id": str(index)} for index in range(7)
    ])
    assert sum(item["type"] == "image_url" for item in items) == 5
    assert sum(item["type"] == "text" for item in items) == 5
