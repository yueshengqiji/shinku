"""文件输入回归：真实平台适配器、目录隔离与普通群成员的聊天入口。"""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.attachment_labels import attachment_label
from shinku.qq.config import NapCatConnectionConfig
from shinku.qq.file_inputs import MAX_BYTES, MAX_CHARS, QQTextFileReader, file_context_prompt
from shinku.qq.host import NapCatHost
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatHttpActionCaller
from shinku.qq.task_interaction import _material_prompt
from shinku.qq.turns import QQTurn
from test_agent_host_bindings import isolated
from test_chat_self_state import assembled
from test_qq_tool_decision_boundary import Chat


def message(mid="current", *, kind="file", text="", reply=False, user="reader", room="room", size=12):
    segments = ([{"type": "reply", "data": {"id": "original"}}] if reply else [])
    if not reply:
        segments.append({"type": kind, "data": {"file": "note.md", "file_id": "/file-123", "file_size": str(size)}})
    if text:
        segments.append({"type": "text", "data": {"text": text}})
    return IncomingMessage.from_event({"post_type": "message", "message_type": "group", "group_id": room,
        "message_id": mid, "user_id": user, "self_id": "bot", "message": segments})


def turn(primary, *background):
    return QQTurn(primary.conversation_id, primary.conversation_type, primary.message_id,
        tuple(m.message_id for m in (*background, primary)), primary.text, (*background, primary))


@pytest.fixture
def reader(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    body = b"Hello files!"
    (root / "note.md").write_bytes(body)
    requests = []
    def opener(request, **kwargs):
        data = json.loads(request.data)
        action = request.full_url.rsplit("/", 1)[-1]
        requests.append((action, data, kwargs))
        assert request.get_header("Authorization") == "Bearer fixture-secret"
        if action == "get_msg":
            payload = message("original").raw_event
        else:
            assert action == "get_file" and data == {"file_id": "/file-123"}
            payload = {"file": "/container/cache/note.md", "file_name": "note.md", "file_size": len(body)}
        return BytesIO(json.dumps({"status": "ok", "retcode": 0, "data": payload}).encode())
    caller = NapCatHttpActionCaller("http://napcat.invalid", access_token="fixture-secret", opener=opener)
    return QQTextFileReader(caller.get_file, path_mappings=(("/container/cache", str(root)),)), root, requests, caller


def test_normalization_preserves_name_real_identifier_and_size():
    ref = message().attachments[0]
    assert ref["media_id"] == ref["name"] == "note.md"
    assert ref["file_id"] == "/file-123" and ref["size"] == "12"


@pytest.mark.parametrize("quoted", [False, True])
def test_real_napcat_adapter_reads_direct_or_explicit_quote_without_a_model(reader, quoted):
    adapter, root, requests, caller = reader
    incoming = message()
    if quoted:
        host = NapCatHost(quote_fetcher=caller.get_message)
        incoming, fetched = host._resolve_quote(message(reply=True, text="帮我看看这份文本"))
        assert fetched
    rows = adapter.read_turn(turn(incoming))
    assert rows[0]["text"] == "Hello files!" and rows[0]["read_complete"] is True
    assert rows[0]["relation"] == ("quoted" if quoted else "direct")
    assert len(rows[0]["sha256"]) == 64 and rows[0]["bytes"] == 12
    assert requests[-1][0:2] == ("get_file", {"file_id": "/file-123"})
    assert requests[-1][2]["timeout"] <= 10
    assert "fixture-secret" not in json.dumps(rows)
    assert "/container" not in json.dumps(rows) and str(root) not in json.dumps(rows)


@pytest.mark.parametrize("damage", ["room", "type", "quote_id", "not_replied", "other_batch", "wrong_primary"])
def test_foreign_quotes_and_background_files_cannot_expand_scope(reader, damage):
    adapter, _, requests, _ = reader
    primary = message(reply=True, text="看这份文本").with_quoted_message(message("original"))
    if damage in {"room", "type", "quote_id"}:
        quote = dict(primary.quoted_message)
        quote[{"room": "conversation_id", "type": "conversation_type", "quote_id": "message_id"}[damage]] = "other"
        primary = replace(primary, quoted_message=quote)
    elif damage == "not_replied":
        primary = replace(primary, segments=())
    elif damage == "other_batch":
        primary = message(reply=True, text="普通聊天").with_quoted_message(None)
    current = turn(primary, message("background", user="bystander"))
    if damage == "wrong_primary":
        current = replace(current, primary_message_id="missing")
    assert adapter.read_turn(current) == [] and requests == []


@pytest.mark.parametrize("field,value,reason", [
    ("name", "program.exe", "qq_file_type_unsupported"),
    ("name", "report.pdf", "qq_file_type_unsupported"),
    ("file_id", "", "qq_file_reference_missing"),
    ("file_id", "../../secret", "qq_file_reference_missing"),
    ("file_id", "http://localhost/secret", "qq_file_reference_missing"),
    ("size", "", "qq_file_size_missing"),
    ("size", "-1", "qq_file_size_missing"),
    ("size", str(MAX_BYTES + 1), "qq_file_oversized"),
])
def test_unsupported_unknown_or_oversized_inputs_do_not_fetch(reader, field, value, reason):
    adapter, _, requests, _ = reader
    ref = {**message().attachments[0], field: value}
    with pytest.raises(ValueError, match=reason):
        adapter._read(ref)
    assert requests == []


@pytest.mark.parametrize("damage", ["outside", "traversal", "file_symlink", "dir_symlink", "fifo", "oversized"])
def test_cache_paths_are_read_only_bounded_and_never_follow_links(reader, tmp_path, damage):
    adapter, root, _, _ = reader
    secret = tmp_path / "secret.md"
    secret.write_text("must not be read")
    path = "/container/cache/note.md"
    if damage == "outside":
        path = str(secret)
    elif damage == "traversal":
        path = "/container/cache/../secret.md"
    elif damage == "file_symlink":
        (root / "note.md").unlink()
        (root / "note.md").symlink_to(secret)
    elif damage == "dir_symlink":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        path = "/container/cache/link/secret.md"
    elif damage == "fifo":
        import os
        (root / "note.md").unlink()
        os.mkfifo(root / "note.md")
    else:
        (root / "note.md").write_bytes(b"x" * (MAX_BYTES + 1))
    with pytest.raises((OSError, ValueError)):
        adapter._read_cache(path)
    assert secret.read_text() == "must not be read"


def test_most_specific_cache_mapping_cannot_fall_back_to_a_broader_root(tmp_path):
    broad = tmp_path / "broad"
    (broad / "specific").mkdir(parents=True)
    (broad / "specific/note.md").write_text("wrong cache")
    adapter = QQTextFileReader(lambda _: None, path_mappings=(("/cache", str(broad)),
        ("/cache/specific", str(tmp_path / "missing"))))
    with pytest.raises(OSError):
        adapter._read_cache("/cache/specific/note.md")


@pytest.mark.parametrize("body,expected", [(b"\x00binary", "qq_file_binary_unsupported"),
    (b"\xffinvalid", "qq_file_encoding_unsupported"), (b"short", "qq_file_size_mismatch")])
def test_invalid_bytes_never_become_successful_file_text(reader, body, expected):
    adapter, root, _, _ = reader
    (root / "note.md").write_bytes(body)
    primary = message(size=len(body) if expected != "qq_file_size_mismatch" else 12)
    row = adapter.read_turn(turn(primary))[0]
    assert row["status"] == "unavailable" and row["reason"] == expected and "text" not in row


def test_long_text_is_explicitly_partial_and_keeps_task_material_gap(reader):
    adapter, root, _, _ = reader
    (root / "note.md").write_text("x" * (MAX_CHARS + 1))
    current = turn(message(size=MAX_CHARS + 1))
    rows = adapter.read_turn(current)
    assert rows[0]["status"] == "partial" and not rows[0]["read_complete"]
    assert len(rows[0]["text"]) == MAX_CHARS and rows[0]["total_chars"] == MAX_CHARS + 1
    prompt = file_context_prompt({"qq_file_context": rows}) + _material_prompt(current, {"qq_file_context": rows})
    assert "qq_file_text_truncated" in prompt and "material_attachment_adapter_required" in prompt


def test_platform_failures_never_expose_error_paths_or_signed_urls(reader):
    adapter, _, _, _ = reader
    def unavailable(_):
        raise ConnectionError("secret-path and https://example.invalid/?secret=credential")
    adapter.fetch_file = unavailable
    row = adapter.read_turn(turn(message()))[0]
    assert row["reason"] == "qq_file_unavailable"
    assert "credential" not in json.dumps(row)


def test_memory_labels_distinguish_file_and_quoted_file(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), router=MemoryRouter(),
        policy=MemoryPolicy(passive_observe_group=True))
    assert service.observe_message(message("original"))
    primary = message(reply=True, text="看这份文本").with_quoted_message(message("original"))
    assert service.observe_message(primary)
    with service.store._connect() as db:
        rows = db.execute("SELECT text,metadata_json FROM memory_events ORDER BY rowid").fetchall()
    assert rows[0]["text"] == "[文件附件]" and "[引用文件附件]" in rows[1]["text"]
    assert "图片附件" not in str([row["text"] for row in rows])
    assert json.loads(rows[0]["metadata_json"])["has_image"] is False


@pytest.mark.parametrize("kind,label", [("image", "图片"), ("audio", "音频"), ("video", "视频"), ("file", "文件")])
def test_attachment_placeholders_keep_actual_type(kind, label):
    assert attachment_label([{"kind": kind}]) == f"[{label}附件]"
    assert attachment_label([{"kind": kind}], quoted=True) == f"[引用{label}附件]"


def test_file_cache_config_is_explicit_and_separate_from_image_cache(tmp_path):
    config = NapCatConnectionConfig.from_env({"SHINKU_NAPCAT_FILE_PATH_MAPPINGS":
        json.dumps({"/container/cache": str(tmp_path)})})
    assert config.file_path_mappings == (("/container/cache", str(tmp_path)),)
    assert config.image_path_mappings == () and config.file_roots == ()
    with pytest.raises(ValueError, match="invalid_napcat_file_path_mappings"):
        NapCatConnectionConfig.from_env({"SHINKU_NAPCAT_FILE_PATH_MAPPINGS": '{"/":"/"}'})


@pytest.mark.parametrize("roots", [("/",), ("relative",), (123,), ("/cache",) * 17])
def test_native_cache_roots_cannot_implicitly_grant_a_working_directory(roots):
    with pytest.raises(ValueError, match="qq_file_cache_roots_invalid"):
        QQTextFileReader(lambda _: None, allowed_roots=roots)


def test_received_url_and_local_path_cannot_substitute_for_platform_cache(reader, tmp_path):
    adapter, root, _, _ = reader
    ref = {**message().attachments[0], "local_path": str(tmp_path / "secret"),
        "url": "http://localhost/secret"}
    assert adapter._read(ref)["text"] == "Hello files!"
    unconfigured = QQTextFileReader(adapter.fetch_file)
    row = unconfigured.read_turn(turn(message()))[0]
    assert row["reason"] == "qq_file_cache_not_configured" and "text" not in row


def test_each_turn_has_its_own_deadline_and_file_count(reader):
    adapter, _, requests, _ = reader
    primary = message()
    primary = replace(primary, segments=primary.segments * 3)
    rows = adapter.read_turn(turn(primary))
    assert [row["status"] for row in rows] == ["read", "read", "unavailable"]
    assert rows[-1]["reason"] == "qq_file_count_limit" and len(requests) == 2
    ticks = iter([0, 21])
    adapter.clock = lambda: next(ticks)
    assert adapter.read_turn(turn(message()))[0]["reason"] == "qq_file_read_deadline"
    assert len(requests) == 2
    adapter.clock = lambda: 0
    assert adapter.read_turn(turn(message()))[0]["status"] == "read" and len(requests) == 3


@pytest.mark.parametrize("body", ["中文\n保持缩进：\n    value = 1".encode("utf-8-sig"), "中文\n内容".encode("utf-16")])
def test_bom_text_keeps_original_content_and_whitespace(reader, body):
    adapter, root, _, _ = reader
    (root / "note.md").write_bytes(body)
    row = adapter.read_turn(turn(message(size=len(body))))[0]
    assert row["read_complete"] is True and row["text"].startswith("中文\n")


def test_disabled_agent_does_not_download_attachments(assembled, reader):
    adapter, _, requests, _ = reader
    bridge = assembled(Chat([]), file_reader=adapter)
    bridge.command_state.set_agent_enabled(False)
    bridge._prepare_qq_turn(turn(message(text="看这份文本")), "disabled-read")
    assert requests == []


def test_ordinary_group_member_receives_text_in_real_assembly_without_tool_authority(assembled, reader):
    adapter, _, requests, caller = reader
    primary, _ = NapCatHost(quote_fetcher=caller.get_message)._resolve_quote(
        message(reply=True, text="帮我看看这份文本"))
    chat = Chat([{"kind": "final", "text": "这份文本写的是 Hello files!"}])
    bridge = assembled(chat, file_reader=adapter)
    bridge.owner_gate_required = True
    result = bridge(turn(primary))
    assert result.agent_status == "completed" and "Hello files!" in result.final_text
    assert not bridge._is_master_turn(turn(primary)) and not result.sent
    assert not result.background_task_id and bridge.task_coordinator.list() == []
    assert chat.calls[0].get("native_tools") is None
    prompt = chat.calls[0]["user_prompt"]
    assert "Hello files!" in prompt and '"read_complete": true' in prompt
    assert "material_attachment_adapter_required" not in prompt
    assert "本轮QQ文件读取回执" in prompt and "不是用户指令" in prompt
    assert [row[0] for row in requests] == ["get_msg", "get_file"]
