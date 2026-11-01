#!/usr/bin/env python3
"""在 Linux 桌面上串行运行开发批处理，并交给独立的 systemd 资源组。

仅用于测试、审计和迁移命令，正常后端不经过这里。外层负责 scope 生命周期，
scope 内的进程持有跨会话文件锁；实际命令继承锁，即使外层退出也不会提前放行。
临时产物留在用户缓存的磁盘目录，便于失败后排查，不自动清除测试证据。
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def acquire_slot(path: Path, timeout: float) -> int:
    """跨进程等待唯一批处理槽；锁等待有期限，且不提前创建测试临时目录。"""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    deadline = time.monotonic() + timeout
    announced = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if not announced:
                    print("另一轮后台批处理正在运行，等待执行槽。", file=sys.stderr, flush=True)
                    announced = True
                if time.monotonic() >= deadline:
                    raise TimeoutError("后台批处理槽等待超时；原任务继续运行，请稍后重试。")
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    except BaseException:
        os.close(fd)
        raise


def execute_inside(args: argparse.Namespace) -> int:
    root = args.tmp_root.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = acquire_slot(root.parent / "batch.lock", args.lock_timeout)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 75
    try:
        job_tmp = tempfile.mkdtemp(prefix="batch-", dir=root)
        env = os.environ.copy()
        env.update({"TMPDIR": job_tmp, "TMP": job_tmp, "TEMP": job_tmp})
        # 只约束批处理的数值计算线程，避免一轮测试再扩张到全部桌面核心。
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "2"
        os.nice(10)
        # 锁随真实命令存活，而不是随启动器存活；不通过 shell 拼接命令。
        os.set_inheritable(fd, True)
        print(f"后台批处理临时目录：{job_tmp}", file=sys.stderr, flush=True)
        os.execvpe(args.command[0], args.command, env)
    finally:
        os.close(fd)
    return 127


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-timeout", type=float, default=120)
    parser.add_argument("--tmp-root", type=Path, default=Path.home() / ".cache/shinku-runtime/tmp")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command or not 0 <= args.lock_timeout <= 3600:
        parser.error("需要 -- 后的命令，锁等待时间必须在 0–3600 秒之间")
    if sys.platform != "linux":
        parser.error("此开发启动器仅用于 Linux；其他系统直接运行原命令")
    if args.inside:
        return execute_inside(args)
    unit = f"shinku-batch-{uuid.uuid4().hex}.scope"
    command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
               "--slice=shinku-background.slice", f"--unit={unit}",
               "--expand-environment=no", sys.executable, str(Path(__file__).resolve()),
               "--inside", "--lock-timeout", str(args.lock_timeout),
               "--tmp-root", str(args.tmp_root), "--", *args.command]
    process = None
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        # 正常退出、取消和命令失败都回收 scope，不能遗留脱离命令根进程的子进程。
        subprocess.run(["systemctl", "--user", "stop", unit],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
