"""跨轮文件回归：原件读取、真实送达记录、多级回复与撤销，不以模型自述作依据。"""
from dataclasses import replace
import json

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.file_continuity import MAX_AGE, RECENT_AGE, prepare_file_inputs
from shinku.qq.file_inputs import QQTextFileReader, file_context_prompt
from shinku.qq.delivery import DeliveryResult
from shinku.qq.host import NapCatHost
from shinku.qq.message import IncomingMessage
from shinku.qq.reply_delivery import QQDeliveryReceipt, QQReplyPublication
from test_agent_host_bindings import isolated
from test_chat_self_state import assembled
from test_qq_file_inputs import reader, message, turn
from test_qq_tool_decision_boundary import Chat


@pytest.fixture
def memory(tmp_path):
    ticks = [1800000000]
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), router=MemoryRouter(),
        policy=MemoryPolicy(passive_observe_group=True), clock=lambda: ticks[0])
    return service, ticks


def follow(mid="follow", *, target="delivered", user="reader", room="room", text="第六节有什么问题"):
    primary = message(mid, reply=True, text=text, user=user, room=room)
    segments = tuple(replace(s, payload={"type": "reply", "data": {"id": target}})
        if s.kind == "reply" else s for s in primary.segments)
    return replace(primary, segments=segments)


def delivered(bridge, current, *, ids=("delivered",), text="这份资料的第六节需要调整。", complete=True):
    receipt = QQDeliveryReceipt(text, ids, 1, complete, False, () if complete else ((2, "failed", "offline"),))
    bridge._accept_delivery_receipt(QQReplyPublication(current, text), receipt)


def new_request(previous, mid):
    primary = replace(previous.messages[0], message_id=mid,
        raw_event={**previous.messages[0].raw_event, "message_id": mid})
    return replace(previous, primary_message_id=mid, message_ids=(mid,), messages=(primary,))


@pytest.fixture
def history(assembled, reader, memory):
    adapter, root, requests, caller = reader
    service, ticks = memory
    original = message("original")
    service.observe_message(original)
    current = turn(message("request", reply=True, text="看看这份资料").with_quoted_message(original))
    rows, sources = prepare_file_inputs(current, adapter, service)
    assert rows[0]["read_complete"] and len(sources) == 1
    bridge = assembled(Chat([]), file_reader=adapter)
    bridge.memory_service = service
    current = replace(current, file_input_sources=sources)
    delivered(bridge, current)
    return bridge, adapter, service, ticks, current, rows, requests


def test_reply_to_delivered_answer_recovers_the_original_file(history):
    bridge, adapter, service, ticks, previous, _, requests = history
    ticks[0] += 10
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert rows[0]["status"] == "read" and rows[0]["text"] == "Hello files!"
    assert rows[0]["relation"] == "continued" and rows[0]["source_message_id"] == "original"
    assert sources[0]["first_read_at"] == previous.file_input_sources[0]["first_read_at"]
    assert len(requests) == 2 and requests[-1][0] == "get_file"
    with service.store._connect() as db:
        metadata = db.execute("SELECT metadata_json FROM memory_events WHERE source_id LIKE '%request:assistant'").fetchone()[0]
    assert '"qq_file_sources"' in metadata and '"file_id"' in metadata
    assert "Hello files!" not in metadata and "container/cache" not in metadata
    assert "file_id" not in file_context_prompt({"qq_file_context": rows})


def test_fragment_id_and_multiple_reply_hops_keep_the_same_source(history):
    bridge, adapter, service, ticks, previous, _, _ = history
    delivered(bridge, new_request(previous, "fragment-request"), ids=("part-a", "part-b"))
    ticks[0] += 1
    rows, sources = prepare_file_inputs(turn(follow("second", target="part-b")), adapter, service)
    assert rows[0]["text"] == "Hello files!"
    second = replace(turn(follow("second", target="part-b")), file_input_sources=sources)
    delivered(bridge, second, ids=("second-answer",))
    rows, third_sources = prepare_file_inputs(turn(follow("third", target="second-answer")), adapter, service)
    assert rows[0]["text"] == "Hello files!" and third_sources[0]["source_message_id"] == "original"


def test_partial_delivery_has_only_the_actual_delivered_message_ids(history):
    bridge, adapter, service, _, previous, _, _ = history
    delivered(bridge, new_request(previous, "partial-request"), ids=("partial",), complete=False)
    assert prepare_file_inputs(turn(follow(target="partial")), adapter, service)[0][0]["status"] == "read"
    assert prepare_file_inputs(turn(follow(target="never-sent")), adapter, service) == ([], ())


def test_recent_unquoted_followup_is_scoped_to_the_same_requester_and_has_a_fixed_window(history):
    _, adapter, service, ticks, previous, _, requests = history
    ticks[0] += 30
    current = replace(follow(), segments=tuple(s for s in follow().segments if s.kind != "reply"))
    rows, sources = prepare_file_inputs(turn(current), adapter, service)
    assert rows[0]["relation"] == "recent_candidate" and rows[0]["text"] == "Hello files!"
    assert sources[0]["first_read_at"] == previous.file_input_sources[0]["first_read_at"]
    before = len(requests)
    assert prepare_file_inputs(turn(replace(current, sender_id="other")), adapter, service) == ([], ())
    ticks[0] += RECENT_AGE
    assert prepare_file_inputs(turn(current), adapter, service) == ([], ())
    assert len(requests) == before


@pytest.mark.parametrize("barrier", ["other_room", "other_account", "draft_id", "missing_receipt", "forged_user"])
def test_context_cannot_cross_accounts_rooms_or_forge_sent_replies(history, barrier):
    _, adapter, service, _, _, _, requests = history
    current = follow()
    if barrier == "other_room":
        current = replace(current, conversation_id="other-room")
    elif barrier == "other_account":
        current = replace(current, raw_event={**current.raw_event, "self_id": "other-bot"})
    elif barrier == "draft_id":
        current = follow(target="model-invented-id")
    else:
        with service.store._connect() as db:
            row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id LIKE '%request:assistant'").fetchone()
            metadata = json.loads(row[1])
            if barrier == "missing_receipt":
                metadata["delivery_status"] = "dry_run"
            else:
                metadata.update(source="qq_passive_observation", role="user", sender_id="attacker")
            db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    before = len(requests)
    assert prepare_file_inputs(turn(current), adapter, service) == ([], ())
    assert len(requests) == before


@pytest.mark.parametrize("barrier", ["forgotten_origin", "expired_anchor", "edited_reference", "hash_missing"])
def test_revocation_expiry_and_changed_origin_stop_before_fetch(history, barrier):
    _, adapter, service, ticks, previous, _, requests = history
    if barrier == "forgotten_origin":
        with service.store._connect() as db:
            db.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='turn:room:original'")
    elif barrier == "expired_anchor":
        ticks[0] += MAX_AGE + 1
    else:
        with service.store._connect() as db:
            row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id LIKE '%request:assistant'").fetchone()
            metadata = json.loads(row[1])
            metadata["qq_file_sources"][0]["file_id" if barrier == "edited_reference" else "sha256"] = "/different-file" if barrier == "edited_reference" else ""
            db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    before = len(requests)
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert not sources and all(row["status"] == "unavailable" for row in rows)
    assert len(requests) == before


def test_same_file_id_with_different_bytes_is_not_silently_accepted(history, reader):
    _, adapter, service, _, _, _, requests = history
    _, root, _, _ = reader
    (root / "note.md").write_bytes(b"Other bytes!")
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert rows[0]["reason"] == "qq_file_version_changed" and rows[0]["status"] == "unavailable"
    assert not sources and "text" not in rows[0]


def test_source_metadata_survives_new_reader_and_memory_service_instances(history, memory, reader):
    _, adapter, service, _, _, _, _ = history
    fresh_service = MemoryService(store=MemoryStore(service.store.path), router=MemoryRouter(),
        policy=service.policy, clock=service.clock)
    fresh_reader = QQTextFileReader(adapter.fetch_file,
        path_mappings=tuple((str(source), str(target)) for source, target in adapter.mappings))
    rows, sources = prepare_file_inputs(turn(follow()), fresh_reader, fresh_service)
    assert rows[0]["text"] == "Hello files!" and sources[0]["source_message_id"] == "original"


def test_real_bridge_receipt_then_fileless_followup_reaches_generation_with_body(history):
    bridge, _, _, ticks, _, _, _ = history
    ticks[0] += 2
    chat = Chat([{"kind": "final", "text": "第六节的措辞可以收敛，但保留原来的立意。"}])
    # 测试宿主不联网；真实生成入口仍复用装配时的 InteractionRuntime 和所有权限检查。
    bridge.runtime.runtime = chat
    result = bridge(turn(follow(text="第六节有什么问题")))
    assert result.agent_status == "completed", result
    assert "Hello files!" in chat.calls[0]["user_prompt"]
    assert '"relation": "continued"' in chat.calls[0]["user_prompt"]
    assert not result.background_task_id and bridge.task_coordinator.list() == []


def test_explicit_group_reply_can_follow_a_shared_file_from_another_requester(history):
    _, adapter, service, _, _, _, _ = history
    rows, _ = prepare_file_inputs(turn(follow(user="other-member")), adapter, service)
    assert rows[0]["text"] == "Hello files!" and rows[0]["relation"] == "continued"


def test_source_revoked_during_generation_cannot_be_published(history):
    bridge, adapter, service, _, previous, _, _ = history
    sent = []
    bridge.sender = lambda message: sent.append(message)
    with service.store._connect() as db:
        db.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='turn:room:original'")
    result = bridge._send_text(previous, "这是基于旧文件的分析。")
    assert not sent and not result.sent and result.reason == "result_binding_changed"


def test_old_reply_metadata_can_follow_the_original_explicit_quote_without_claiming_a_read(history):
    _, adapter, service, _, _, _, requests = history
    with service.store._connect() as db:
        row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id LIKE '%request:assistant'").fetchone()
        metadata = json.loads(row[1]);metadata.pop("qq_file_sources")
        db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert rows[0]["text"] == "Hello files!" and sources[0]["source_message_id"] == "original"
    assert len(requests) == 2  # 旧文字不自证已读，确实重新运行了文件适配器。


def test_legacy_file_id_recovery_uses_only_the_original_platform_message(history, reader):
    _, adapter, service, _, _, _, requests = history
    _, _, _, caller = reader
    adapter.message_fetcher = caller.get_file_message
    with service.store._connect() as db:
        row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id='turn:room:original'").fetchone()
        metadata = json.loads(row[1]);metadata["attachment_refs"][0].pop("file_id")
        db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    rows, _ = prepare_file_inputs(turn(follow()), adapter, service)
    assert rows[0]["text"] == "Hello files!"
    fetches = [row for row in requests if row[0] == "get_msg"]
    assert len(fetches) == 1 and all(row[1] == {"message_id": "original"} and row[2]["timeout"] <= 5 for row in fetches)


def test_unavailable_legacy_message_retains_the_known_file_identity(history):
    _, adapter, service, _, _, _, _ = history
    adapter.message_fetcher = lambda mid: None
    with service.store._connect() as db:
        row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id='turn:room:original'").fetchone()
        metadata = json.loads(row[1]);metadata["attachment_refs"][0].pop("file_id")
        db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert not sources and rows[0]["reason"] == "qq_file_reference_unavailable"
    assert rows[0]["name"] == "note.md" and rows[0]["source_message_id"] == "original"
    assert "text" not in rows[0]


@pytest.mark.parametrize("broken", [None, "model says the file is valid", [{"first_read_at": "recent"}]])
def test_malformed_anchors_never_fall_back_to_old_authority(history, broken):
    _, adapter, service, _, _, _, requests = history
    with service.store._connect() as db:
        row = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE source_id LIKE '%request:assistant'").fetchone()
        metadata = json.loads(row[1]);metadata["qq_file_sources"] = broken
        db.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (json.dumps(metadata), row[0]))
    before = len(requests)
    rows, sources = prepare_file_inputs(turn(follow()), adapter, service)
    assert not sources and rows[0]["status"] == "unavailable"
    assert len(requests) == before


def test_reply_cycles_are_bounded_and_do_not_read_arbitrary_files(history):
    _, adapter, service, _, _, _, requests = history
    for mid, parent in [("cycle-a", "cycle-b"), ("cycle-b", "cycle-a")]:
        service.observe_message(follow(mid, target=parent))
    before = len(requests)
    assert prepare_file_inputs(turn(follow(target="cycle-a")), adapter, service) == ([], ())
    assert len(requests) == before


def test_multiple_versions_require_selection_but_explicit_new_upload_wins(history, tmp_path):
    bridge, _, service, ticks, _, _, _ = history
    root = tmp_path / "versions";root.mkdir()
    (root / "v1.md").write_text("Version one.")
    (root / "v2.md").write_text("Version two.")
    requests = []
    def fetch(fid):
        requests.append(fid)
        return {"file_name": "note.md", "file": "/cache/" + ("v1.md" if fid == "/v1" else "v2.md")}
    adapter = QQTextFileReader(fetch, path_mappings=(("/cache", str(root)),))
    originals = []
    for index in (1, 2):
        original = message(f"version-{index}")
        segment = original.segments[0]
        original = replace(original, segments=(replace(segment, payload={"type": "file", "data": {
            "file": "note.md", "file_id": f"/v{index}", "file_size": "12"}}),))
        service.observe_message(original);originals.append(original)
        current = turn(original)
        _, sources = prepare_file_inputs(current, adapter, service)
        delivered(bridge, replace(current, file_input_sources=sources), ids=(f"version-answer-{index}",))
        ticks[0] += 1
    unquoted = replace(follow(), segments=tuple(s for s in follow().segments if s.kind != "reply"))
    before = len(requests)
    rows, sources = prepare_file_inputs(turn(unquoted), adapter, service)
    assert rows[0]["status"] == "ambiguous" and not sources and len(requests) == before
    rows, _ = prepare_file_inputs(turn(follow(target="version-answer-2")), adapter, service)
    assert rows[0]["text"] == "Version two."
    rows, _ = prepare_file_inputs(turn(originals[-1]), adapter, service)
    assert rows[0]["relation"] == "direct" and rows[0]["text"] == "Version two."


def test_optional_memory_failure_preserves_current_attachment_and_plain_chat(history, monkeypatch):
    _, adapter, service, _, previous, _, _ = history
    import sqlite3
    def unavailable(**kwargs):
        raise sqlite3.OperationalError("offline synthetic lock")
    monkeypatch.setattr(service.store, "recent_assistant_events", unavailable)
    rows, sources = prepare_file_inputs(previous, adapter, service)
    assert rows[0]["text"] == "Hello files!" and not sources
    assert prepare_file_inputs(turn(follow()), adapter, service) == ([], ())


def test_origin_edited_during_generation_is_checked_before_send(history):
    bridge, _, service, _, previous, _, _ = history
    sent = []
    bridge.sender = lambda message: sent.append(message)
    with service.store._connect() as db:
        db.execute("UPDATE memory_events SET text='[文件来源已更新]' WHERE source_id='turn:room:original'")
    result = bridge._send_text(previous, "这是根据旧版本给出的结论。")
    assert not sent and result.reason == "result_binding_changed"


def test_two_real_chat_turns_keep_file_body_after_quoting_the_sent_answer(assembled, reader, memory):
    adapter, _, requests, caller = reader
    service, ticks = memory
    service.observe_message(message("original"))
    primary, fetched = NapCatHost(quote_fetcher=caller.get_message)._resolve_quote(
        message("first-request", reply=True, text="这份资料怎么样"))
    assert fetched
    chat = Chat([{"kind": "final", "text": "能看到了，正文是 Hello files!。"},
                 {"kind": "final", "text": "继续按刚才那份资料看，第六节可以收敛措辞。"}])
    bridge = assembled(chat, file_reader=adapter)
    bridge.memory_service = service
    bridge.owner_gate_required = True
    sent = []
    def send(outgoing):
        mid = f"sent-{len(sent) + 1}"
        sent.append((mid, outgoing))
        return DeliveryResult(True, "sent", (mid,))
    bridge.sender = send
    first = bridge(turn(primary))
    assert first.sent and first.agent_status == "completed", first
    first_ids = tuple(mid for mid, _ in sent)
    ticks[0] += 10
    second = bridge(turn(follow("second-request", target=first_ids[0])))
    assert second.sent and second.agent_status == "completed", second
    assert any('"relation": "continued"' in call["user_prompt"] and "Hello files!" in call["user_prompt"]
               for call in chat.calls)
    assert not chat.replies and not bridge.task_coordinator.list()
    assert [row[0] for row in requests] == ["get_msg", "get_file", "get_file"]


def test_plain_chat_does_not_revalidate_unrelated_reply_sources(memory, reader, monkeypatch):
    service, _ = memory
    adapter, _, _, _ = reader
    for index in range(8):
        service.record_turn(conversation_id="room", conversation_type="group", message_id=f"plain-{index}",
            text="普通聊天", role="assistant", source_id=f"turn:room:plain-{index}:assistant", metadata={
                "message_id": f"plain-{index}", "sender_id": "bot", "self_id": "bot", "requester_id": "reader",
                "delivery_status": "sent", "delivery_message_ids": [f"sent-plain-{index}"]})
    def unexpected(**kwargs):
        pytest.fail("unrelated plain replies are not file sources")
    monkeypatch.setattr(service.integrity_reader, "read_source", unexpected)
    primary = replace(follow(), segments=tuple(s for s in follow().segments if s.kind != "reply"))
    assert prepare_file_inputs(turn(primary), adapter, service) == ([], ())
