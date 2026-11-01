"""后端应用工厂。

为什么用工厂而不是模块级 ``app`` 对象
--------------------------------------
工厂让"配置"成为入参而不是模块副产物：测试可以传一份指向临时目录的 ``Settings``，
不用改环境变量、不用碰真实数据根。C 阶段加入依赖注入（引擎、锚点服务）时，
注入点就在这里，不需要改调用方。

B1 先提供 ``/health`` 与 ``/health/ready``；当前后端还挂载本机控制中枢和 QQ Webhook。
健康端点的作用不是"看起来像个服务"，而是**证明新树能在没有旧项目的情况下起来**：
进程能装、能启动、能回答问题、能报告自己解析到的是哪套配置。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from .. import names
from ..config import Settings, load_settings
from ..health import build_basic_health_payload
from .debug_hub import build_debug_hub_router
from .agent_work import build_agent_work_router
from .observability import build_observability_router
from .qq_diagnostics import build_qq_diagnostics_router
from .memory_admin import build_memory_admin_router
from .capabilities import build_capabilities_router
from .model_services import build_model_services_router
from .pet import build_pet_router
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
    qq_agent_assembly: object | None = None,
    napcat_config: object | None = None,
    model_services_controller: object | None = None,
    agent_task_coordinator: object | None = None,
) -> FastAPI:
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bridge = getattr(qq_agent_assembly, "bridge", None)
        memory_service = getattr(bridge, "memory_service", None)
        memory_store = getattr(memory_service, "store", None)
        lifecycle = getattr(memory_store, "lifecycle", None)
        review = getattr(memory_service, "lifecycle_review", None)
        projection = getattr(memory_service, "projection", None)
        projection_start = getattr(projection, "start", None)
        if callable(projection_start):
            projection_start()
        review_start = getattr(review, "start", None)
        if callable(review_start):
            review_start()
        starter = getattr(lifecycle, "start_worker", None)
        if callable(starter):
            starter()
        try:
            yield
        finally:
            # Close admission and signal cooperative cancellation before
            # stopping services used by task handlers. In-flight blocking I/O
            # cannot be forcefully preempted; reboot never replays handlers.
            coordinator = agent_task_coordinator or getattr(qq_agent_assembly, "task_coordinator", None)
            shutdown = getattr(coordinator, "shutdown", None)
            if callable(shutdown):
                shutdown(wait=False)
            for owner, method in ((projection, "stop"), (review, "stop"), (lifecycle, "stop_worker"), (memory_store, "stop_embedding_worker")):
                closer = getattr(owner, method, None)
                if callable(closer):
                    closer()

    app = FastAPI(
        title="Shinku",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        # 关掉 schema 本身。只关 docs_url/redoc_url 是不够的——`/openapi.json` 仍会
        # 提供完整路由清单，等于把内部面暴露出来。这是干净环境验收实际测出来的缺口。
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(build_observability_router(settings=resolved, assembly=qq_agent_assembly))
    app.include_router(build_qq_diagnostics_router(napcat_host=napcat_host, napcat_config=napcat_config, assembly=qq_agent_assembly))
    # 控制中枢的状态面和写操作都只允许本机请求；模型服务路由另外再做一次来源校验。
    app.include_router(
        build_debug_hub_router(
            settings=resolved,
            assembly=qq_agent_assembly,
            napcat_host=napcat_host,
            napcat_config=napcat_config,
            model_services_controller=model_services_controller,
        )
    )
    if model_services_controller is not None:
        store = getattr(model_services_controller, "store", None)
        config_module = getattr(model_services_controller, "config_module", None)
        if store is not None and config_module is not None:
            app.state.model_services_controller = model_services_controller
            app.include_router(
                build_model_services_router(
                    store=store,
                    config_module=config_module,
                    engine=model_services_controller,
                )
            )
    coordinator = agent_task_coordinator or getattr(qq_agent_assembly, "task_coordinator", None)
    if coordinator is not None:
        app.include_router(build_agent_work_router(coordinator))
    if qq_agent_assembly is not None:
        bridge = getattr(qq_agent_assembly, "bridge", None)
        memory_service = getattr(bridge, "memory_service", None)
        if memory_service is not None:
            app.include_router(build_memory_admin_router(memory_service))
        app.include_router(build_capabilities_router(qq_agent_assembly))
    app.include_router(build_pet_router(settings=resolved, assembly=qq_agent_assembly))
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

        payload = build_basic_health_payload()
        payload.update({
            "project": resolved.project,
            "version": resolved.version,
            "service": SERVICE,
            "bind": resolved.binds[SERVICE],
            "port": resolved.ports[SERVICE],
            "allow_lan": resolved.allow_lan,
            "data_root": str(resolved.data_root),
            "config_root": str(resolved.config_root),
            "log_root": str(resolved.log_root),
            "log_file": str(resolved.log_path(SERVICE)),
            "correlation_header": names.CORRELATION_ID_HEADER,
        })
        return payload

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
