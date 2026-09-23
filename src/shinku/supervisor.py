"""只管理由 Shinku 自己启动的后端子进程的轻量监督器。

监督器不扫描系统进程、不按名称杀进程，也不修改端口或 NapCat 配置。
它只启动 ``python -m shinku.cli serve --service backend``，读取本地
``/health/live``，并在子进程异常退出或持续不健康时按有限次数重启。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

from .config import Settings, load_settings

__all__ = ["ShinkuSupervisor", "SupervisorConfig", "probe_health"]


def probe_health(url: str, *, timeout: float = 2.0) -> bool:
    """读取本地存活探针，只把 ``status=ok`` 视为健康。"""

    request = Request(str(url).rstrip("/") + "/health/live", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local endpoint
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return isinstance(payload, Mapping) and str(payload.get("status") or "").casefold() == "ok"


@dataclass(frozen=True)
class SupervisorConfig:
    service: str = "backend"
    max_restarts: int = 3
    health_interval_seconds: float = 5.0
    startup_timeout_seconds: float = 30.0
    restart_backoff_seconds: float = 2.0
    shutdown_timeout_seconds: float = 5.0
    health_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.service not in {"backend"}:
            raise ValueError("only the implemented backend service can be supervised")
        if self.max_restarts < -1:
            raise ValueError("max_restarts must be -1 or non-negative")
        for name in (
            "health_interval_seconds",
            "startup_timeout_seconds",
            "restart_backoff_seconds",
            "shutdown_timeout_seconds",
            "health_timeout_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


class ShinkuSupervisor:
    """Run and supervise exactly one child command owned by this object."""

    def __init__(
        self,
        config: SupervisorConfig | None = None,
        *,
        settings: Settings | None = None,
        process_factory: Callable[..., Any] | None = None,
        health_probe: Callable[[str], bool] | None = None,
        sleep: Callable[[float], Any] | None = None,
        monotonic: Callable[[], float] | None = None,
        command: Sequence[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or SupervisorConfig()
        self.settings = settings or load_settings()
        self._process_factory = process_factory or subprocess.Popen
        self._health_probe = health_probe or (
            lambda url: probe_health(url, timeout=self.config.health_timeout_seconds)
        )
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._command = tuple(command or self.default_command())
        self._environ = dict(environ) if environ is not None else None
        self._child: Any = None

    def default_command(self) -> tuple[str, ...]:
        return (sys.executable, "-m", "shinku.cli", "serve", "--service", self.config.service)

    @property
    def health_url(self) -> str:
        return self.settings.url(self.config.service)

    @property
    def child(self) -> Any:
        return self._child

    def run(self) -> int:
        """Run until the child exits cleanly, restart budget is exhausted, or Ctrl-C."""

        restarts = 0
        try:
            while True:
                self._child = self._start_child()
                outcome = self._monitor_child()
                self._child = None
                if outcome == 0:
                    return 0
                if self.config.max_restarts != -1 and restarts >= self.config.max_restarts:
                    return int(outcome or 1)
                restarts += 1
                self._sleep(self.config.restart_backoff_seconds)
        except KeyboardInterrupt:
            self._stop_child()
            return 130
        finally:
            if self._child is not None:
                self._stop_child()
                self._child = None

    def _start_child(self) -> Any:
        kwargs: dict[str, Any] = {"stdin": None}
        if self._environ is not None:
            kwargs["env"] = dict(self._environ)
        return self._process_factory(list(self._command), **kwargs)

    def _monitor_child(self) -> int:
        child = self._child
        started = self._monotonic()
        next_probe = started
        failed_probes = 0
        while True:
            code = child.poll()
            if code is not None:
                return int(code)
            now = self._monotonic()
            if now >= next_probe:
                healthy = False
                try:
                    healthy = bool(self._health_probe(self.health_url))
                except Exception:
                    healthy = False
                next_probe = now + self.config.health_interval_seconds
                if healthy:
                    failed_probes = 0
                else:
                    failed_probes += 1
                    timed_out = now - started >= self.config.startup_timeout_seconds
                    if timed_out and failed_probes >= 3:
                        self._stop_child()
                        return 1
            self._sleep(min(0.5, self.config.health_interval_seconds))

    def _stop_child(self) -> None:
        child = self._child
        if child is None or getattr(child, "poll", lambda: 0)() is not None:
            return
        terminate = getattr(child, "terminate", None)
        if callable(terminate):
            terminate()
        wait = getattr(child, "wait", None)
        if not callable(wait):
            return
        try:
            wait(timeout=self.config.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            kill = getattr(child, "kill", None)
            if callable(kill):
                kill()
            try:
                wait(timeout=self.config.shutdown_timeout_seconds)
            except Exception:
                pass
        except Exception:
            pass
