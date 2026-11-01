"""独立源包与本地 CLI 验收；复制的包禁止依赖 shinku，演示禁止联网。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from aozora.__main__ import main


def test_demo_restarts_and_reports_only_synthetic_results(tmp_path, capsys):
    destination = tmp_path / "demo"
    assert main(["demo", "--directory", str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "synthetic_offline_demo"
    assert result["model_called"] is result["external_delivery"] is False
    assert result["first"]["status"] == result["after_restart"]["status"] == "completed"
    assert result["after_restart"]["delivery"]["status"] == "prepared"
    assert result["state"]["attention"] == []
    assert result["state"]["understandings"][0]["source"] == "model_interpretation"
    assert main(["demo", "--directory", str(destination)]) == 1
    capsys.readouterr()
    assert main(["inspect", "--identity", str(destination / "identity.example.json"),
                 "--db", str(destination / "state.sqlite3"), "--scope", "synthetic-room"]) == 0
    assert len(json.loads(capsys.readouterr().out)["events"]) == 2
    assert main(["status", "--identity", str(destination / "identity.example.json"),
                 "--db", str(destination / "state.sqlite3"), "--scope", "synthetic-room", "--event-id", "demo-1"]) == 0
    assert json.loads(capsys.readouterr().out)["delivery"]["status"] == "prepared"


def test_standalone_copy_runs_with_shinku_import_and_all_network_forbidden(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/aozora"
    isolated = tmp_path / "standalone"
    isolated.mkdir()
    shutil.copytree(source, isolated / "aozora", ignore=shutil.ignore_patterns("__pycache__"))
    code = """
import importlib.abc, json, pathlib, socket, sys
sys.path.insert(0, sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'shinku', 'shinku_private', 'openai', 'requests'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Deny())
def deny(event, args):
    if event.startswith('socket.'):
        raise AssertionError('network is forbidden: ' + event)
sys.addaudithook(deny)
from aozora.__main__ import main
raise SystemExit(main(['demo', '--directory', sys.argv[2]]))
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(isolated), str(tmp_path / "output")],
                            cwd=isolated, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["after_restart"]["status"] == "completed"


def test_missing_explicit_key_does_not_create_database(tmp_path, capsys, monkeypatch):
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"agent_id": "fixture", "name": "合成个体", "foundations": [], "relationships": {}}))
    monkeypatch.delenv("AOZORA_FIXTURE_MISSING_KEY", raising=False)
    db = tmp_path / "not-created.sqlite3"
    assert main(["turn", "--identity", str(identity), "--db", str(db), "--endpoint", "https://example.com/api",
                 "--model", "fixture", "--api-key-env", "AOZORA_FIXTURE_MISSING_KEY", "--event", "unused.json"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "api_key_environment_missing"
    assert not db.exists()


def test_module_help_is_independent_and_has_no_default_network(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-B", "-m", "aozora", "--help"], env=env,
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "inspect" in result.stdout and "chat" in result.stdout
