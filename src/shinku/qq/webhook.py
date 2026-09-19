"""NapCat HTTP 入站 webhook 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .host import NapCatHost

__all__ = ["create_napcat_webhook_router"]


def create_napcat_webhook_router(
    host: NapCatHost,
    *,
    path: str = "/events/napcat",
    event_token: str = "",
) -> APIRouter:
    """建立一个只负责接收事件的 FastAPI router。

    它不创建 app、不监听端口，也不负责发送回复；外层应用可以按自己的服务布局
    include 这个 router。``event_token`` 只做入站 Bearer 校验，不会回显。
    """

    if not isinstance(host, NapCatHost):
        raise TypeError("host must be a NapCatHost")
    route_path = str(path or "").strip()
    if not route_path.startswith("/"):
        raise ValueError("napcat webhook path must start with '/'")
    expected_token = str(event_token or "").strip()
    router = APIRouter()

    @router.post(route_path)
    async def receive_event(request: Request) -> JSONResponse:
        if expected_token:
            authorization = str(request.headers.get("authorization") or "").strip()
            if authorization != f"Bearer {expected_token}":
                return JSONResponse({"ok": False, "error": "event_unauthorized"}, status_code=401)
        try:
            result = host.handle_event(await request.body())
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "invalid_napcat_event"}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "scheduled": result.scheduled,
                "visual_ready": result.visual.ready,
                "visual_pending": result.visual.pending,
            },
            status_code=200,
        )

    return router
