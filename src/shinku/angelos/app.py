"""Angelos 独立 ASGI 组合根，没有控制中枢、QQ Webhook 或消息调度器。"""
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from shinku.api.angelos import build_angelos_router


def create_angelos_app(data_root, *, host=None):
    router, workbench = build_angelos_router(SimpleNamespace(data_root=Path(data_root)), host=host)
    @asynccontextmanager
    async def lifespan(app):
        workbench.start()
        try:
            # 自动维护仅由本机可信工厂显式注入；未配置时无线程、模型或任务。
            maintenance = getattr(host, 'maintenance', None)
            if maintenance is not None:
                maintenance.start()
            yield
        finally:
            maintenance = getattr(host, 'maintenance', None)
            if maintenance is not None:
                maintenance.stop()
            workbench.close()
    app = FastAPI(title='Angelos', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.include_router(router)
    gateway = getattr(host, 'delegation', None)
    if gateway is not None:
        from shinku.api.agent_delegation import build_delegation_router
        app.include_router(build_delegation_router(gateway))
    app.state.angelos = workbench
    @app.get('/')
    def root():
        return RedirectResponse('/angelos')
    return app
