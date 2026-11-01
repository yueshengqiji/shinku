"""开发批处理的真实跨进程串行、等待超时和环境隔离回归。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux desktop batch launcher")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_desktop_batch.py"


def command(root, code, timeout=2):
    return [sys.executable, str(SCRIPT), "--inside", "--tmp-root", str(root),
            "--lock-timeout", str(timeout), "--", sys.executable, "-c", code]


def test_command_gets_disk_temp_and_preserves_working_directory_and_exit_code(tmp_path):
    code = "import os,tempfile,json;print(json.dumps([os.getcwd(),tempfile.gettempdir(),os.getenv('OMP_NUM_THREADS')]));raise SystemExit(7)"
    result = subprocess.run(command(tmp_path / "tmp", code), cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 7
    cwd, temp, threads = json.loads(result.stdout)
    assert cwd == str(tmp_path)
    assert Path(temp).parent == tmp_path / "tmp"
    assert threads == "2"
    assert Path(temp).is_dir()  # 失败产物不会被启动器悄悄删除。


def test_two_processes_cannot_run_batch_commands_at_the_same_time(tmp_path):
    marker = tmp_path / "started"
    first_code = f"from pathlib import Path;import time;Path({str(marker)!r}).touch();time.sleep(.5);print(time.monotonic())"
    first = subprocess.Popen(command(tmp_path / "tmp", first_code), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        second = subprocess.run(command(tmp_path / "tmp", "import time;print(time.monotonic())"), capture_output=True, text=True, timeout=5)
        first_out, _ = first.communicate(timeout=3)
        assert first.returncode == second.returncode == 0
        assert float(second.stdout) >= float(first_out)
    finally:
        if first.poll() is None:
            first.kill(); first.wait()


def test_lock_timeout_does_not_start_command_or_create_another_temp_directory(tmp_path):
    import fcntl
    root = tmp_path / "tmp"
    with (tmp_path / "batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(command(root, "raise SystemExit(99)", timeout=.1), capture_output=True, text=True, timeout=3)
    assert result.returncode == 75
    assert not list(root.glob("batch-*"))
