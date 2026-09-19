"""后端应用工厂。

为什么用工厂而不是模块级 ``app`` 对象
--------------------------------------
工厂让"配置"成为入参而不是模块副产物：测试可以传一份指向临时目录的 ``Settings``，
不用改环境变量、不用碰真实数据根。C 阶段加入依赖注入（引擎、锚点服务）时，
注入点就在这里，不需要改调用方。

B1 只提供 ``/health`` 与 ``/health/ready``。它们的作用不是"看起来像个服务"，
而是**证明新树能在没有旧项目的情况下起来**：
进程能装、能启动、能回答问题、能报告自己解析到的是哪套配置。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from .. import names
from ..config import Settings, load_settings
from ..qq.host import NapCatHost
from ..qq.webhook import create_napcat_webhook_router

SERVICE = "backend"


def _probe_writable(path: Path) -> tuple[bool, str]:
    """就绪探针：目录能不能建、能不能在里头写文件。

    只做**最小**写入并立刻清理，不留残余。
    """

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot create {path}: {exc}"
    probe = path / ".shinku-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"cannot write under {path}: {exc}"
    return True, "ok"


def create_app(
    settings: Settings | None = None,
    *,
    napcat_host: NapCatHost | None = None,
    napcat_webhook_path: str = "/events/napcat",
    napcat_event_token: str = "",
) -> FastAPI:
    resolved = settings or load_settings()

    app = FastAPI(
        title="Shinku",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        # 关掉 schema 本身。只关 docs_url/redoc_url 是不够的——`/openapi.json` 仍会
        # 提供完整路由清单，等于把内部面暴露出来。这是干净环境验收实际测出来的缺口。
        openapi_url=None,
    )
    app.state.settings = resolved
    if napcat_host is not None:
        app.state.napcat_host = napcat_host
        app.include_router(
            create_napcat_webhook_router(
                napcat_host,
                path=napcat_webhook_path,
                event_token=napcat_event_token,
            )
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        """存活探针：返回本次解析到的配置快照。

        把端口和路径回显出来是有意的——旧项目吃过"以为连的是这台、其实是另一台"的亏。
        """

        return {
            "status": "ok",
            "project": resolved.project,
            "version": resolved.version,
            "service": SERVICE,
            "pid": os.getpid(),
            "bind": resolved.binds[SERVICE],
            "port": resolved.ports[SERVICE],
            "allow_lan": resolved.allow_lan,
            "data_root": str(resolved.data_root),
            "config_root": str(resolved.config_root),
            "log_root": str(resolved.log_root),
            "log_file": str(resolved.log_path(SERVICE)),
            "correlation_header": names.CORRELATION_ID_HEADER,
        }

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        ok, detail = _probe_writable(resolved.data_root)
        return {
            "status": "ready" if ok else "not-ready",
            "service": SERVICE,
            "data_root": str(resolved.data_root),
            "detail": detail,
        }

    return app


__all__ = ["SERVICE", "create_app"]
