"""Build the canonical frontend into a wheel and import outside the checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

pytest.importorskip("build")

ROOT = Path(__file__).resolve().parents[1]


def test_wheel_has_frontend_and_serves_it_outside_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md", "AGENTS.md"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "frontend", source / "frontend", ignore=shutil.ignore_patterns("node_modules"))
    environment = dict(os.environ, PIP_NO_INDEX="1", UV_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    sdist = subprocess.run([sys.executable, "-B", "-m", "build", "--sdist", "--no-isolation", str(source),
                            "--outdir", str(tmp_path / "sdist")], env=environment, capture_output=True, text=True)
    assert sdist.returncode == 0, sdist.stdout + sdist.stderr
    unpacked = tmp_path / "unpacked"
    with tarfile.open(next((tmp_path / "sdist").glob("*.tar.gz"))) as archive:
        archive.extractall(unpacked, filter="data")
    build_source = next(unpacked.iterdir())
    build = subprocess.run([sys.executable, "-B", "-m", "build", "--wheel", "--no-isolation", str(build_source),
                            "--outdir", str(tmp_path / "wheel")], env=environment, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next((tmp_path / "wheel").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        assert all(f"shinku/_frontend/control-center/{name}" in archive.namelist()
                   for name in ("index.html", "app.css", "app.js"))
        archive.extractall(installed)
    environment = {key: value for key, value in environment.items() if not key.startswith(("SHINKU_", "LLM_"))}
    environment["PYTHONPATH"] = str(installed)
    environment["SHINKU_ENV_FILE"] = os.devnull
    script = '''
import json, tempfile, re
from pathlib import Path
import shinku
from fastapi.testclient import TestClient
from shinku.api import create_app
from shinku.config import load_settings
from shinku.frontend import control_center_root, angelos_root
from shinku.angelos.app import create_angelos_app
with tempfile.TemporaryDirectory() as d:
    settings = load_settings({"SHINKU_DATA_ROOT": d+"/data", "SHINKU_CONFIG_ROOT": d+"/config", "SHINKU_LOG_ROOT": d+"/logs"})
    with TestClient(create_app(settings)) as client:
        result = {p: client.get(p).status_code for p in ("/health", "/debug/hub", "/debug/hub/assets/app.css", "/debug/hub/assets/app.js")}
        assert all(v == 200 for v in result.values()), result
        assert client.get("/debug/hub").content == (control_center_root()/"index.html").read_bytes()
        with TestClient(create_angelos_app(Path(d)/"agent-data")) as agent_client:
            for p in ("/angelos", "/angelos/assets/app.css", "/angelos/assets/app.js", "/angelos/assets/context.js", "/angelos/assets/layout.js", "/angelos/assets/activity.js", "/angelos/assets/usage.js", "/angelos/assets/workspace.js", "/angelos/assets/editor.js"):
                assert agent_client.get(p).status_code == 200
            assert agent_client.get("/angelos").content == (angelos_root()/"index.html").read_bytes()
            # 以真正 HTML 的资源引用验证发布包，避免新增脚本只在源码检出可用。
            html = agent_client.get("/angelos").text
            for asset in re.findall(r'(?:src|href)="(/angelos/assets/[^"?]+)', html):
                assert agent_client.get(asset).status_code == 200, asset
        print(json.dumps({"module": shinku.__file__, "root": str(control_center_root()), "responses": result}))
'''
    process = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                             env=environment, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert Path(result["module"]).is_relative_to(installed)
    assert Path(result["root"]).is_relative_to(installed)
