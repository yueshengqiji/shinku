"""LA02-LA05/LA08: semantic choices use real host/media/material boundaries."""
from reply_review_fixture import dialogue_review_result
import base64
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget, current_task_budget
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime.materials import TaskMaterialSession, TaskMaterialStore
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionPolicy
from shinku.qq.delivery import DeliveryResult
from shinku.qq.host import NapCatHost
from shinku.qq.napcat import NapCatImageMaterializer, NapCatVisualInputBridge
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.task_materials import sources_for_turn
from shinku.qq.turns import NapCatTurnDispatcher
from shinku.qq.visual_selection import apply_selection
from test_qq_visual_provenance import Harness, message


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6J6sAAAAASUVORK5CYII=")


def candidate_payload(options):
    block = next((item for item in options.get("system_extra_blocks", ())
                  if item.startswith("【图片候选待确认】")), "")
    return json.loads(block[block.index('{"candidates"'):]) if block else None


def choice(turn, action="select", ids=None, changed=False):
    return {"visual_context": {"action": action,
        "source_ids": [turn.visual_candidates[0]["source_id"]] if ids is None and action == "select" else ids or [],
        "topic_change": changed}}


@pytest.mark.parametrize("text", ["这是什么", "那是谁", "细节能展开吗", "换一种表述完全没有匹配词", "晚饭吃什么"])
@pytest.mark.parametrize("delay", [1, 31, 61, 240])
def test_lexical_miss_or_window_never_confirms_or_closes_image_topic(text, delay):
    h = Harness(visual_context_ttl_seconds=300)
    h.receive(message("image", image=True), scheduled=False)
    h.now += delay
    turn = h.receive(message(text=text))
    assert turn.visual_candidates[0]["message_id"] == "image"
    assert not turn.visual_context_inherited and not QQAgentBridge._images(turn)
    assert h.dispatcher._visual_context[("group", "g", "image")].followup_open
    assert turn.visual_candidates[0]["sender_id"] == "owner"


def test_same_group_other_sender_is_a_legal_semantic_candidate():
    h = Harness()
    h.receive(message("image", "other", image=True), scheduled=False)
    raw = h.receive(message(text="这是什么"))
    turn = h.choose(raw)
    assert QQAgentBridge._images(turn)[0]["source_sender_id"] == "other"
    assert not turn.visuals_for("query")


@pytest.mark.parametrize("scope", [("group", "another"), ("private", "g")])
def test_candidates_and_history_do_not_cross_conversation_or_transport(scope):
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    turn = h.receive(message(group=scope[1], kind=scope[0]))
    assert not turn.visual_candidates and not turn.visual_history
    foreign = "visual:" + h.dispatcher._visual_context[("group", "g", "image")].version
    assert apply_selection({"qq_turn": turn}, choice(turn, ids=[foreign])) == ("rejected", "visual_source_not_offered")


def test_unknown_image_author_never_becomes_candidate():
    h = Harness()
    h.receive(message("image", "", image=True), scheduled=False)
    assert not h.receive(message()).visual_candidates


@pytest.mark.parametrize("change", ["replace", "remove", "supersede"])
def test_selection_revalidates_source_version_and_availability(change):
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    turn = h.receive(message())
    if change == "remove":
        assert h.dispatcher.revoke_visual_source("group", "g", "image")
    else:
        h.now += 1
        h.receive(message("image" if change == "replace" else "new", image=True), scheduled=False)
    context = {"qq_turn": turn}
    assert apply_selection(context, choice(turn)) == ("rejected", "visual_source_changed")
    assert not context.get("user_images")


@pytest.mark.parametrize("payload", [
    {}, {"action": "select", "source_ids": ["forged"]}, {"action": "select", "source_ids": []},
    {"action": "none", "source_ids": ["forged"]}, {"action": "clarify", "source_ids": [], "topic_change": 1},
    {"action": [], "source_ids": []},
])
def test_missing_or_forged_choices_cannot_read_or_attach(payload):
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    turn = h.receive(message())
    context = {"qq_turn": turn}
    result, error = apply_selection(context, {"visual_context": payload})
    assert result == "rejected" and error
    assert not context.get("user_images")
    assert h.dispatcher._visual_context[("group", "g", "image")].remaining_turns == h.dispatcher.visual_followup_turns


def test_duplicate_ids_and_single_use_selection_cannot_spend_twice():
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    turn = h.receive(message())
    token = turn.visual_candidates[0]["source_id"]
    assert apply_selection({"qq_turn": turn}, choice(turn, ids=[token, token]))[0] == "rejected"
    assert apply_selection({"qq_turn": turn}, choice(turn))[0] == "select"
    assert apply_selection({"qq_turn": turn}, choice(turn)) == ("rejected", "visual_selection_already_consumed")


@pytest.mark.parametrize("field,value", [("conversation_id", "foreign"), ("conversation_type", "private"),
                                        ("primary_message_id", "another-request")])
def test_resolver_cannot_be_transplanted_to_another_scope_or_request(field, value):
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    raw = h.receive(message())
    copied = replace(raw, **{field: value})
    assert apply_selection({"qq_turn": copied}, choice(raw)) == ("rejected", "visual_selection_scope_conflict")
    assert h.choose(raw).visual_context_inherited


def test_future_image_and_history_do_not_enter_an_already_arrived_question():
    from test_qq_visual_provenance import batch
    h = Harness()
    h.receive(message("older", "other", image=True), scheduled=False)
    query = message()
    h.submit(query)
    h.receive(message("future", "third", image=True), scheduled=False)
    h.pending.append(batch(query))
    h.dispatcher.flush()
    turn = h.turns[-1]
    assert [row["message_id"] for row in turn.visual_candidates] == ["older"]
    assert "future" not in [row["message_id"] for row in turn.visual_history]


def test_total_image_read_budget_is_checked_before_preparation():
    from shinku.qq.message import IncomingMessage
    from test_qq_visual_provenance import IMAGE
    h = Harness()
    event = dict(message("image", image=True).raw_event)
    event["message"] = [{"type": "image", "data": {"url": IMAGE}} for _ in range(6)]
    h.receive(IncomingMessage.from_event(event), scheduled=False)
    turn = h.receive(message())
    assert apply_selection({"qq_turn": turn}, choice(turn)) == ("rejected", "visual_read_budget")
    assert h.dispatcher._visual_context[("group", "g", "image")].remaining_turns == h.dispatcher.visual_followup_turns


def test_no_image_this_turn_does_not_close_unless_model_confirms_topic_change():
    h = Harness()
    h.receive(message("image", image=True), scheduled=False)
    h.choose(h.receive(message(text="晚饭吃什么")), action="none")
    current = h.receive(message("later", text="另一个说法"))
    assert current.visual_candidates
    h.choose(current, action="none", topic_change=True)
    assert not h.receive(message("closed")).visual_candidates
    assert h.choose(h.receive(message("quoted", reply="image"))).visual_context_inherited


@pytest.fixture
def pipeline(tmp_path):
    source = tmp_path / "fixture.png"
    source.write_bytes(PNG)
    calls, replies, sent, turns, budgets = [], [], [], [], []
    script = []
    clock = [100.0]

    def respond(**options):
        calls.append(options)
        budgets.append(current_task_budget())
        review = dialogue_review_result(options)
        if review is not None:
            return review
        pending = candidate_payload(options)
        if pending:
            action, changed = script.pop(0) if script else ("select", False)
            return {"kind": "final", "text": "你指哪张图片？" if action == "clarify" else "按当前话题回答。",
                "visual_context": {"action": action,
                    "source_ids": [pending["candidates"][0]["source_id"]] if action == "select" else [],
                    "topic_change": changed}}
        return {"speech": "图片像素为白色。" if options.get("user_images") else "按当前话题回答。"}

    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=respond), system_prompt="fixture",
        sender=lambda output: sent.append(output) or DeliveryResult(True, "sent", (f"out-{len(sent)}",)),
        memory_service=None, bot_ids=("bot",), master_user_ids=("owner",),
        task_store=AgentTaskStore(tmp_path / "tasks.sqlite"))
    ingress = QQIngressAdapter(route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot"})),
                               attention_policy=AttentionPolicy(autonomous_enabled=False))
    media = NapCatVisualInputBridge(materialize=NapCatImageMaterializer(allowed_roots=(str(tmp_path),)))
    host = NapCatHost(ingress=ingress, visual_bridge=media, clock=lambda: clock[0])
    def handle(turn):
        turns.append(turn)
        replies.append(bridge.handle_turn(turn))
    dispatcher = NapCatTurnDispatcher(handler=handle, flush_batch=host.flush,
        timer_factory=lambda *_: SimpleNamespace(start=lambda: None, cancel=lambda: None),
        clock=lambda: clock[0], visual_context_ttl_seconds=5)
    host.set_turn_dispatcher(dispatcher)

    def send(mid, text="", sender="owner", image=False, directed=False, group="g", reply=""):
        segments = [{"type": "text", "data": {"text": text}}]
        if image:
            segments.append({"type": "image", "data": {"local_path": str(source)}})
        if directed:
            segments.insert(0, {"type": "at", "data": {"qq": "bot"}})
        if reply:
            segments.insert(0, {"type": "reply", "data": {"id": reply}})
        result = host.handle_event({"post_type": "message", "message_type": "group", "group_id": group,
            "self_id": "bot", "user_id": sender, "message_id": mid, "time": int(clock[0]), "message": segments})
        if result.scheduled:
            dispatcher.flush()
        return result

    yield SimpleNamespace(source=source, calls=calls, replies=replies, sent=sent, turns=turns,
        script=script, budgets=budgets, clock=clock, send=send, bridge=bridge, dispatcher=dispatcher)
    dispatcher.close()
    bridge.task_store.close()


@pytest.mark.parametrize("action,expected_calls,expected_images", [("select", 2, True), ("clarify", 1, False), ("none", 2, False)])
def test_inbound_model_choice_real_media_and_one_test_delivery(pipeline, action, expected_calls, expected_images):
    p = pipeline
    p.send("image", "第一张图片", sender="other", image=True)
    p.clock[0] += 1
    if action == "clarify":
        p.send("second", "另一张图片", sender="third", image=True)
    p.script.append((action, action == "none"))
    p.send("question", "这是什么", directed=True)
    assert len(p.calls) == expected_calls and len(p.sent) == 1
    assert not p.calls[0].get("user_images") and p.calls[0].get("native_tools") is None
    assert bool(p.calls[-1].get("user_images")) == expected_images
    assert p.replies[-1].sent and p.replies[-1].delivery_status == "sent"
    trace = p.bridge.trace_store.snapshot(limit=1)[0]
    resolution = next(row for row in trace["stages"] if row["name"] == "visual_selection")
    assert resolution["status"] == action and resolution["extra_model_calls"] == expected_calls - 1
    if expected_images:
        image = p.calls[1]["user_images"][0]
        assert image["source_sender_id"] == "other" and image["source_message_id"] == "image"
        assert base64.b64decode(image["data_url"].split(",", 1)[1]) == PNG
        assert "本轮实际图片来源" in p.calls[1]["user_prompt"]


def test_no_candidates_chat_keeps_one_model_call_and_no_task(pipeline):
    p = pipeline
    p.send("question", "普通闲聊", directed=True)
    assert len(p.calls) == 1 and len(p.sent) == 1
    assert not candidate_payload(p.calls[0]) and not p.bridge.task_store.candidates(requester="owner", session="group:g")


def test_same_generic_question_after_confirmed_topic_switch_has_no_selected_image(pipeline):
    p = pipeline
    p.send("image", "图片话题", image=True)
    p.send("topic", "先换到晚饭的话题", sender="owner")
    p.script.append(("none", True))
    p.send("question", "这是什么", directed=True)
    payload = candidate_payload(p.calls[0])
    assert any("晚饭" in row["text"] for row in payload["recent_messages"])
    assert not p.calls[0].get("user_images") and len(p.sent) == 1 and len(p.calls) == 2
    assert not p.dispatcher._visual_context[("group", "g", "image")].followup_open


def test_same_generic_question_without_source_preserves_the_gap(pipeline):
    p = pipeline
    p.bridge.runtime = SimpleNamespace(call_chat_json=lambda **kw: p.calls.append(kw) or dialogue_review_result(kw) or {"speech": "目前没有可读图片，请发图或引用原消息。"})
    p.send("question", "这是什么", directed=True)
    assert len(p.calls) == 1 and len(p.sent) == 1
    assert not candidate_payload(p.calls[0]) and not p.calls[0].get("user_images")
    assert "没有可读图片" in p.replies[-1].final_text


@pytest.mark.parametrize("failure", ["deleted", "changed", "exhausted"])
def test_real_expired_source_gaps_never_fall_back_or_claim_picture(pipeline, failure):
    p = pipeline
    p.send("older", image=True)
    p.send("image", image=True)
    p.clock[0] += 6
    if failure == "deleted":
        p.source.unlink()
    elif failure == "changed":
        p.source.write_bytes(PNG + b"changed")
    else:
        ctx = p.dispatcher._visual_context[("group", "g", "image")]
        ctx.remaining_turns = 0
    p.send("question", "这是什么", directed=True)
    assert len(p.calls) == 1 and not p.calls[0].get("user_images")
    assert len(p.sent) == 1
    if failure != "exhausted":
        assert "无法核实或读取" in p.replies[-1].final_text
    else:
        assert not candidate_payload(p.calls[0])


def test_expired_cache_bounded_reread_uses_real_bytes_and_budget(pipeline):
    p = pipeline
    p.send("image", image=True)
    p.clock[0] += 6
    p.bridge.recovery_enabled = True
    p.bridge.task_budget_limits = {"max_tools": 1}
    p.send("question", "换一种说法", directed=True)
    assert len(p.calls) == 2 and p.budgets[-1].tools == 1 and len(p.sent) == 1
    assert base64.b64decode(p.calls[-1]["user_images"][0]["data_url"].split(",", 1)[1]) == PNG
    trace = p.bridge.trace_store.snapshot(limit=1)[0]
    assert next(row for row in trace["stages"] if row["name"] == "visual_selection")["reads"] == 1
    p.clock[0] += 1
    p.send("another-question", "同一张继续说明", directed=True)
    trace = p.bridge.trace_store.snapshot(limit=1)[0]
    assert next(row for row in trace["stages"] if row["name"] == "visual_selection")["reads"] == 0
    assert len(p.calls) == 4 and len(p.sent) == 2


def test_source_revoked_during_real_reload_never_reaches_the_answer_model(pipeline):
    p = pipeline
    p.send("image", image=True)
    p.clock[0] += 6
    original_loader = p.dispatcher._visual_loader
    def revoked_loader(msg):
        result = original_loader(msg)
        assert p.dispatcher.revoke_visual_source("group", "g", "image")
        return result
    p.dispatcher.set_visual_loader(revoked_loader)
    p.send("question", "这是什么", directed=True)
    assert len(p.calls) == len(p.sent) == 1
    assert not p.calls[0].get("user_images") and "无法核实或读取" in p.replies[-1].final_text


def test_read_budget_denies_loader_and_keeps_gap(pipeline):
    p = pipeline
    p.send("image", image=True)
    p.clock[0] += 6
    # Read validation is the host boundary, independently of model admission.
    p.dispatcher.handler = p.turns.append
    p.send("question", "这是什么", directed=True)
    budget = TaskBudget(max_tools=1, tools=1)
    context = {"qq_turn": p.turns[-1]}
    with bind_task_budget(budget):
        assert apply_selection(context, choice(p.turns[-1])) == ("rejected", "visual_read_budget")
    assert budget.tools == 1 and not context.get("user_images")


def test_selected_sources_reach_existing_material_store_without_reparenting(tmp_path):
    h = Harness()
    h.receive(message("one", "other", image=True), scheduled=False)
    h.receive(message("two", "third", image=True), scheduled=False)
    raw = h.receive(message())
    context = {"qq_turn": raw}
    assert apply_selection(context, choice(raw, ids=[row["source_id"] for row in raw.visual_candidates]))[0] == "select"
    turn = context["qq_turn"]
    ledger = AgentTaskStore(tmp_path / "materials.sqlite")
    try:
        task = ledger.save(new_task(task_id="fixture-task", requester="owner", session="group:g",
            request=turn.combined_text, message_id=turn.primary_message_id, contract=None))
        sources, gaps = sources_for_turn(turn, task, context)
        assert not gaps and len(sources) == 2
        store = TaskMaterialStore(ledger)
        package = store.package(task, sources, authorization_ref="fixture")
        task["material_package"] = package
        ledger.save(task)
        session = TaskMaterialSession(store, task, still_authorized=lambda: True)
        pictures = session.media(supported=True)
        assert len(pictures) == 2
        assert {row.provenance["sender_id"] for row in sources} == {"other", "third"}
        assert all(base64.b64decode(row["data_url"].split(",", 1)[1]) == b"hello" for row in pictures)
    finally:
        ledger.close()
