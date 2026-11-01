"""角色控制台的真实公共装配与 loopback NapCat 适配验收，不调用模型服务。"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shinku.api import create_app
from shinku.config import load_settings
from shinku.instance import claim_instance, load_instance
from shinku.qq.config import NapCatConnectionConfig
from shinku.qq.host import NapCatHost
from test_instance_launcher import manifest, make_assembly, close, event_rows


def settings(tmp_path):
    return load_settings({f"SHINKU_{name}_ROOT": str(tmp_path / name.lower())
                          for name in ("DATA", "CONFIG", "LOG")})


def test_two_console_instances_keep_names_memory_and_http_receipts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit instance must not read default character assets")
    monkeypatch.setattr("shinku.api.pet._load_character_data", forbidden)
    monkeypatch.setattr("shinku.api.debug_hub._find_electron_pids", forbidden)
    rigs = []
    try:
        for name in ("child", "parent"):
            spec = load_instance(manifest(tmp_path, name))
            with claim_instance(spec):
                assembly, runtime = make_assembly(spec)
                rigs.append(assembly)
                client = TestClient(create_app(settings(tmp_path / name), qq_agent_assembly=assembly))
                state = client.get("/debug/hub/state").json()
                assert state["character"] == {"id": name, "name": name}
                assert state["pet"]["available"] is False
                status = client.get("/api/pet/status").json()
                assert status["character"]["name"] == name
                assert status["character"]["dialogue"]["local_click_lines"] == []
                assert status["character"]["live2d"]["loaded"] is False
                assert client.get("/api/pet/live2d/anything.model3.json").status_code == 404
                for action in ("launch", "stop", "restart", "status"):
                    assert client.post("/control-center/pet/control", json={"action": action}).status_code == 409
                result = client.post("/api/pet/chat", json={"text": name + "独有的话题", "conversation_id": "same"})
                assert result.status_code == 200 and result.json()["ok"]
                assert "爸爸" in json.dumps(runtime.calls, ensure_ascii=False)
                assistant = [json.loads(r["metadata_json"]) for r in event_rows(assembly)
                             if json.loads(r["metadata_json"]).get("role") == "assistant"]
                assert assistant and all(m["sender_name"] == name and m["delivery_channel"] == "desktop_http" for m in assistant)
                assert assembly.bridge.sender is None
        assert "parent独有" not in json.dumps(event_rows(rigs[0]), ensure_ascii=False)
        assert "child独有" not in json.dumps(event_rows(rigs[1]), ensure_ascii=False)
    finally:
        for assembly in rigs:
            close(assembly)


@pytest.mark.parametrize("online,account,valid,expected", [
    (True, "10002", True, "online"), (False, "10002", True, "offline"),
    (True, "10003", True, "account_mismatch"), (True, "10002", False, "unavailable")])
def test_connection_uses_real_http_caller_and_checks_account(tmp_path, online, account, valid, expected):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.path, self.headers.get("Authorization")))
            data = {"online": online} if self.path == "/get_status" else {"user_id": account}
            body = json.dumps({"status": "ok" if valid else "failed", "retcode": 0 if valid else 1, "data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = NapCatConnectionConfig(base_url=f"http://127.0.0.1:{server.server_port}", access_token="fixture-secret")
        app = create_app(settings(tmp_path), napcat_host=NapCatHost(), napcat_config=config,
                         qq_agent_assembly=SimpleNamespace(bridge=SimpleNamespace(bot_ids=("10002",))))
        client = TestClient(app)
        assert client.get("/api/qq/self-check").status_code == 200 and not calls
        response = client.get("/api/qq/connection")
        assert response.json()["status"] == expected
        assert "fixture-secret" not in response.text
        assert response.json()["observed_at"]
        assert client.get("/api/qq/connection").json() == response.json()
        assert calls == [("/get_status", "Bearer fixture-secret"), ("/get_login_info", "Bearer fixture-secret")]
        remote = TestClient(app, client=("203.0.113.10", 1234))
        assert remote.get("/api/qq/connection").status_code == 403 and len(calls) == 2
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
