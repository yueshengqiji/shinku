"""真实 loopback HTTP 适配回归；仅合成请求，不访问模型供应商。"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import threading

import pytest

from aozora import Event, Identity, Kernel, KernelError, Store
from aozora.decider import HTTPDecider, parse_response
from aozora.__main__ import main


def candidate():
    return {"action": "ask", "reply": "可以提供读数吗？", "attention": [{
        "key": "reading", "text": "等待读数", "version": 0, "evidence": ["http-event"], "status": "open"}],
        "understandings": []}


def envelope(value=None, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant",
        "content": json.dumps(value if value is not None else candidate(), ensure_ascii=False)}}]}


@contextmanager
def server(payload, *, status=200, location=None):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                             "authorization": self.headers.get("Authorization")})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

        def log_message(self, *args):
            pass
    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/v1/chat/completions", requests
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def run(tmp_path, endpoint):
    store = Store(tmp_path / "http.sqlite3", Identity("http-fixture", "合成个体"))
    kernel = Kernel(store, HTTPDecider(endpoint, "fixture-model", api_key="synthetic-test-key"))
    event = Event("http-event", "fixture-room", "speaker", "message", "合成输入：两组结果相同。", 1)
    return store, kernel, event


def test_real_transport_identity_projection_decision_and_replay(tmp_path):
    with server(envelope()) as (endpoint, requests):
        store, kernel, event = run(tmp_path, endpoint)
        result = kernel.handle(event)
        assert result["status"] == "completed"
        assert result["delivery"]["status"] == "prepared"
        assert kernel.handle(event) == result
    assert len(requests) == 1
    wire = requests[0]
    assert wire["authorization"] == "Bearer synthetic-test-key"
    assert wire["body"]["model"] == "fixture-model"
    assert wire["body"]["stream"] is False
    context = json.loads(wire["body"]["messages"][1]["content"])
    assert context["identity"]["foundations"] == []
    assert context["events"][0]["source"] == "reported"
    assert store.snapshot("fixture-room")["attention"][0]["evidence"] == ["http-event"]


@pytest.mark.parametrize("payload,error", [
    (b"not JSON", "provider_envelope_invalid"),
    ({"choices": []}, "provider_envelope_invalid"),
    (envelope(finish="length"), "provider_incomplete_or_tool_response"),
    (envelope({**candidate(), "identity": {"name": "forged"}}), "invalid_fields"),
    (envelope({**candidate(), "attention": [{**candidate()["attention"][0], "evidence": ["invisible"]}]}), "unseen_evidence"),
    (b"x" * 65537, "provider_output_too_large"),
])
def test_bad_provider_response_preserves_event_without_partial_state_or_retry(tmp_path, payload, error):
    with server(payload) as (endpoint, requests):
        store, kernel, event = run(tmp_path, endpoint)
        first = kernel.handle(event)
        assert first["status"] == "failed" and first["error"] == error
        assert first["delivery"] is None
        assert kernel.handle(event) == first
    assert len(requests) == 1
    snapshot = store.snapshot("fixture-room")
    assert snapshot["attention"] == snapshot["understandings"] == []
    assert snapshot["events"][0]["event_id"] == "http-event"


def test_redirect_does_not_forward_credentials_or_retry(tmp_path):
    with server(envelope()) as (destination, received):
        with server(b"", status=307, location=destination) as (endpoint, requests):
            _, kernel, event = run(tmp_path, endpoint)
            result = kernel.handle(event)
    assert result["error"] == "provider_redirect_refused"
    assert len(requests) == 1 and received == []


def test_http_error_does_not_leak_response_body_or_retry(tmp_path):
    with server(b"secret key in diagnostic", status=500) as (endpoint, requests):
        _, kernel, event = run(tmp_path, endpoint)
        result = kernel.handle(event)
    assert result["error"] == "provider_request_failed"
    assert "secret" not in json.dumps(result)
    assert len(requests) == 1


@pytest.mark.parametrize("endpoint", ["http://example.com/api", "file:///private", "https://user:pass@example.com/api", "https://example.com/api?key=secret"])
def test_unsafe_or_ambiguous_endpoint_rejected_before_request(endpoint):
    with pytest.raises(KernelError):
        HTTPDecider(endpoint, "explicit-model")


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": float("inf")}, {"timeout": True}, {"max_tokens": 0}, {"max_tokens": True}])
def test_request_boundaries_must_be_finite(kwargs):
    with pytest.raises(KernelError):
        HTTPDecider("https://example.com/api", "explicit-model", **kwargs)


def test_local_chat_entry_two_turns_records_terminal_delivery(tmp_path, monkeypatch, capsys):
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(Identity("chat-fixture", "合成个体").payload()))
    db = tmp_path / "chat.sqlite3"
    monkeypatch.setattr("sys.stdin", io.StringIO("第一次合成输入\n第二次合成输入\n"))
    reply = {"action": "reply", "reply": "合成终端回复", "attention": [], "understandings": []}
    with server(envelope(reply)) as (endpoint, requests):
        assert main(["chat", "--identity", str(identity), "--db", str(db), "--scope", "local-chat",
                     "--actor", "local-user", "--endpoint", endpoint, "--model", "fixture-model"]) == 0
    assert capsys.readouterr().out == "合成终端回复\n合成终端回复\n"
    assert len(requests) == 2
    second = json.loads(requests[1]["body"]["messages"][1]["content"])
    assert len(second["events"]) == 2
    assert second["deliveries"][0]["status"] == "delivered"
    state = Store(db, Identity("chat-fixture", "合成个体")).snapshot("local-chat")
    assert all(row["status"] == "delivered" and row["channel"] == "local-terminal" for row in state["deliveries"])


def content_envelope(content, **message_fields):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content, **message_fields}}]}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_single_json_fence_uses_same_decision_and_state_rules(tmp_path, newline):
    value = candidate()
    content = " \n```json" + newline + json.dumps(value, ensure_ascii=False, indent=2) + newline + "``` \n"
    assert parse_response(json.dumps(content_envelope(content)).encode()).payload() == value
    with server(content_envelope(content)) as (endpoint, requests):
        store, kernel, event = run(tmp_path, endpoint)
        result = kernel.handle(event)
        assert result["status"] == "completed"
        assert result["delivery"]["status"] == "prepared"
        assert store.snapshot(event.scope)["attention"][0]["evidence"] == [event.event_id]
        assert kernel.handle(event) == result
    assert len(requests) == 1


@pytest.mark.parametrize("content", [
    "下面是结果：\n```json\n{}\n```",
    "```json\n{}\n```\n附加说明",
    "```json\n{}\n```\n```json\n{}\n```",
    "```json\n{}",
    "```json\n{\n```",
    "```python\n{}\n```",
    "{} {}",
    '{"action":"wait","action":"reply","reply":"x","attention":[],"understandings":[]}',
    '{"action":"wait","reply":"","attention":[],"understandings":[],"extra":NaN}',
])
def test_ambiguous_truncated_or_non_json_content_is_not_repaired(content):
    with pytest.raises(KernelError, match="^provider_decision_json_invalid$"):
        parse_response(json.dumps(content_envelope(content)).encode())


@pytest.mark.parametrize("payload,code", [
    (b"\xff", "provider_envelope_invalid"),
    (b"[]", "provider_envelope_invalid"),
    (b'{"choices":null}', "provider_envelope_invalid"),
    (b'{"choices":[null]}', "provider_envelope_invalid"),
    (b'{"choices":[{"message":null,"finish_reason":"stop"}]}', "provider_envelope_invalid"),
    (b'{"choices":[{"message":{},"finish_reason":null}]}', "provider_envelope_invalid"),
    (b'{"choices":[],"choices":[]}', "provider_envelope_invalid"),
    (json.dumps(content_envelope(None)).encode(), "provider_content_invalid"),
    (json.dumps(content_envelope([])).encode(), "provider_content_invalid"),
    (json.dumps(content_envelope("  ")).encode(), "provider_content_invalid"),
])
def test_invalid_response_layers_have_stable_non_sensitive_errors(payload, code):
    with pytest.raises(KernelError, match="^" + code + "$"):
        parse_response(payload)


@pytest.mark.parametrize("value,code", [
    ({**candidate(), "identity": {"name": "forged"}}, "invalid_fields"),
    ({**candidate(), "attention": [{**candidate()["attention"][0], "evidence": ["invisible"]}]}, "unseen_evidence"),
    ({**candidate(), "attention": [{**candidate()["attention"][0], "version": 123}]}, "invalid_new_state"),
])
def test_fences_never_bypass_identity_evidence_or_version_rules(tmp_path, value, code):
    content = "```json\n" + json.dumps(value) + "\n```"
    with server(content_envelope(content)) as (endpoint, requests):
        store, kernel, event = run(tmp_path, endpoint)
        result = kernel.handle(event)
        assert result["status"] == "failed" and result["error"] == code
        assert result["delivery"] is None
        assert store.snapshot(event.scope)["attention"] == []
    assert len(requests) == 1


@pytest.mark.parametrize("field,value", [("tool_calls", [{"id": "untrusted"}]), ("function_call", {"name": "untrusted"})])
def test_fenced_content_does_not_override_tool_response_rejection(field, value):
    payload = content_envelope("```json\n" + json.dumps(candidate()) + "\n```", **{field: value})
    with pytest.raises(KernelError, match="^provider_incomplete_or_tool_response$"):
        parse_response(json.dumps(payload).encode())


def test_formatting_inside_reply_is_preserved_as_data():
    value = {"action": "reply", "reply": "正文里的代码：\n```json\n{}\n```", "attention": [], "understandings": []}
    for content in (json.dumps(value), "```json\n" + json.dumps(value) + "\n```"):
        assert parse_response(json.dumps(content_envelope(content)).encode()).payload() == value


def test_fenced_stale_update_does_not_change_existing_attention(tmp_path):
    with server(envelope()) as (endpoint, _):
        store, kernel, event = run(tmp_path, endpoint)
        assert kernel.handle(event)["status"] == "completed"
    before = store.snapshot(event.scope)["attention"]
    stale = "```json\n" + json.dumps(candidate()) + "\n```"
    with server(content_envelope(stale)) as (endpoint, requests):
        result = Kernel(store, HTTPDecider(endpoint, "fixture-model")).handle(
            Event("next", event.scope, event.actor, "message", "合成后续", 2))
    assert result["status"] == "failed" and result["error"] == "state_version_conflict"
    assert result["delivery"] is None and len(requests) == 1
    assert store.snapshot(event.scope)["attention"] == before
