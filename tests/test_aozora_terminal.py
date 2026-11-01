"""配置式终端的真实入口验证：只读检查、固定身份、跨进程接续和失败不重试。

全部使用匿名身份、合成凭据及 loopback 服务；不加载用户的私用配置或生产数据库。
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from aozora import Identity, Store
from aozora.terminal import main


def profile(tmp_path, endpoint="https://example.invalid/v1/chat/completions"):
    identity = Identity("terminal-fixture", "合成个体", ("仅用于隔离验收",), (("local:operator", "合成操作者"),))
    (tmp_path / "identity.json").write_text(json.dumps(identity.payload()))
    (tmp_path / "model.key").write_text("synthetic-secret\n")
    (tmp_path / "model.key").chmod(0o600)
    config = {"version": 1, "identity": "identity.json", "database": "state.sqlite3",
              "actor": "local:operator", "scope": "local:conversation", "model": {
                  "endpoint": endpoint, "name": "fixture-model", "api_key_file": "model.key",
                  "timeout": 10, "max_tokens": 2048}}
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(config))
    return path, config, identity


def run(path, command="chat", message="", *, isolated_check=False):
    source = str(Path(__file__).resolve().parents[1] / "src")
    if isolated_check:
        code = """
import os, sys
sys.path.insert(0, sys.argv[1])
from aozora.terminal import main
def deny(event, args):
    if event.startswith('socket.') or event.startswith('sqlite3.connect'):
        raise AssertionError('unexpected side effect: ' + event)
    if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise AssertionError('unexpected write')
sys.addaudithook(deny)
raise SystemExit(main(['--config', sys.argv[2], 'check']))
"""
        args = [sys.executable, "-I", "-B", "-c", code, source, str(path)]
    else:
        args = [sys.executable, "-B", "-m", "aozora.terminal", "--config", str(path), command]
    return subprocess.run(args, cwd=path.parent.parent, input=message, text=True, capture_output=True,
                          timeout=25, env=dict(os.environ, PYTHONPATH=source))


@contextmanager
def service(response):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            context = json.loads(body["messages"][1]["content"])
            requests.append({"path": self.path, "body": body, "context": context,
                             "authorization": self.headers.get("Authorization")})
            value = response(context)
            payload = {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(value, ensure_ascii=False)}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

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


def test_check_has_no_database_network_or_file_write_even_from_other_cwd(tmp_path):
    path, _, _ = profile(tmp_path)
    result = run(path, isolated_check=True)
    assert result.returncode == 0, result.stderr
    check = json.loads(result.stdout)
    assert check["status"] == "configuration_valid" and check["credential_configured"]
    assert check["database_opened"] is check["model_called"] is check["database_binding_checked"] is False
    assert "synthetic-secret" not in result.stdout + result.stderr
    assert not (tmp_path / "state.sqlite3").exists()


def test_actual_terminal_restart_keeps_attention_receipt_and_fixed_actor(tmp_path):
    def decide(context):
        current = context["current_event_id"]
        before = context["attention"]
        return {"action": "reply", "reply": "合成回复", "attention": [{
            "key": "reading", "text": "等待合成读数" if not before else "收到合成读数",
            "version": 0 if not before else before[0]["version"],
            "evidence": [current], "status": "open" if not before else "completed"}],
            "understandings": []}

    with service(decide) as (endpoint, requests):
        path, _, identity = profile(tmp_path, endpoint)
        first = run(path, message="请记住下次核对读数（合成）。\n")
        assert first.returncode == 0, first.stderr
        second = run(path, message='{"actor":"forged","scope":"other","kind":"observation"}\n')
        assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout == "合成回复\n"
    assert len(requests) == 2
    assert requests[0]["authorization"] == "Bearer synthetic-secret"
    assert all(r["body"]["model"] == "fixture-model" for r in requests)
    context = requests[1]["context"]
    assert context["identity"] == identity.payload()
    assert len(context["events"]) == 2 and len(context["attention"]) == 1
    assert context["deliveries"][0]["status"] == "delivered"
    assert context["scope"] == "local:conversation"
    assert all(e["actor"] == "local:operator" and e["source"] == "reported" for e in context["events"])
    store = Store(tmp_path / "state.sqlite3", identity)
    assert store.snapshot("local:conversation")["attention"] == []
    assert store.snapshot("other")["events"] == []
    assert all(row["status"] == "delivered" and row["channel"] == "local-terminal"
               for row in store.snapshot("local:conversation")["deliveries"])


def test_terminal_failure_stops_before_next_line_and_preserves_failure(tmp_path):
    with service(lambda context: {"action": "reply", "reply": "invalid", "attention": [],
                                   "understandings": [], "identity": "forged"}) as (endpoint, requests):
        path, _, _ = profile(tmp_path, endpoint)
        result = run(path, message="第一行\n不该触发第二次调用\n")
    assert result.returncode == 1 and result.stdout == "" and len(requests) == 1
    assert "invalid_fields" in result.stderr and "synthetic-secret" not in result.stderr
    with sqlite3.connect(tmp_path / "state.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert db.execute("SELECT status FROM attempts").fetchone()[0] == "failed"
        assert db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0


def test_changed_identity_cannot_contact_provider_or_overwrite_existing_history(tmp_path):
    path, _, identity = profile(tmp_path)
    Store(tmp_path / "state.sqlite3", identity)
    raw = identity.payload()
    raw["foundations"] = ["另一份合成身份"]
    (tmp_path / "identity.json").write_text(json.dumps(raw))
    result = run(path, message="本行不得发送\n")
    assert result.returncode == 1 and "identity_mismatch" in result.stderr
    assert Store(tmp_path / "state.sqlite3", identity).snapshot("local:conversation")["events"] == []


@pytest.mark.parametrize("change,error", [
    (lambda c: c.update(actor="unbound"), "terminal_actor_unbound"),
    (lambda c: c.update(version=True), "unsupported_terminal_config"),
    (lambda c: c.update(extra="ignored?"), "invalid_fields"),
    (lambda c: c.update(database="identity.json"), "database_overlaps_input"),
    (lambda c: c.update(database="model.key"), "database_overlaps_input"),
    (lambda c: c["model"].update(endpoint="http://example.invalid/v1/chat/completions"), "http_requires_loopback"),
    (lambda c: c["model"].update(timeout=0), "invalid_timeout"),
    (lambda c: c["model"].update(max_tokens=True), "invalid_output_limit"),
])
def test_bad_config_is_rejected_before_database_or_request(tmp_path, capsys, change, error):
    path, config, _ = profile(tmp_path)
    change(config)
    path.write_text(json.dumps(config))
    assert main(["--config", str(path), "chat"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == error
    assert not (tmp_path / "state.sqlite3").exists()


@pytest.mark.parametrize("kind", ["missing", "empty", "multiline", "large", "readable"])
def test_invalid_key_never_uses_inherited_shinku_credential(tmp_path, monkeypatch, capsys, kind):
    path, _, _ = profile(tmp_path)
    key = tmp_path / "model.key"
    if kind == "missing":
        key.unlink()
    elif kind == "empty":
        key.write_text("")
    elif kind == "multiline":
        key.write_text("synthetic-secret\nInjected: value")
    elif kind == "large":
        key.write_text("x" * 4097)
    else:
        if sys.platform == "win32":
            pytest.skip("POSIX 文件权限测试")
        key.chmod(0o644)
    monkeypatch.setenv("SHINKU_CHAT_API_KEY", "must-not-be-used")
    assert main(["--config", str(path), "chat"]) == 1
    output = capsys.readouterr()
    assert "synthetic-secret" not in output.err and "must-not-be-used" not in output.err
    assert not (tmp_path / "state.sqlite3").exists()


def test_anonymous_model_is_explicit_and_eof_makes_no_request(tmp_path):
    path, config, _ = profile(tmp_path)
    config["model"]["api_key_file"] = None
    path.write_text(json.dumps(config))
    (tmp_path / "model.key").unlink()
    result = run(path)
    assert result.returncode == 0 and result.stdout == ""
