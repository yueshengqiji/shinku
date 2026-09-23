"""独立版的存活、指标和最近指标入口。"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__all__ = ["build_observability_router"]


def _local_only(request: Request) -> bool:
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "").strip().lower() in {"127.0.0.1", "::1", "localhost", "testclient"}


def build_observability_router(*, settings: Any, assembly: Any = None) -> APIRouter:
    router = APIRouter()

    @router.get("/health/live", response_model=None)
    def live() -> dict[str, Any]:
        # 存活探针只回答进程是否能同步响应，不触碰数据库、NapCat 或模型。
        return {"status": "ok", "service": "backend", "project": getattr(settings, "project", "shinku"), "pid": os.getpid()}

    @router.get("/metrics", response_model=None)
    def metrics(request: Request) -> JSONResponse:
        if not _local_only(request):
            return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
        return JSONResponse(_metrics_snapshot(assembly), headers={"Cache-Control": "no-store"})

    @router.get("/metrics/recent", response_model=None)
    def recent_metrics(request: Request, days: int = 7) -> JSONResponse:
        if not _local_only(request):
            return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
        bridge = getattr(assembly, "bridge", None)
        runtime = getattr(bridge, "runtime", None)
        getter = getattr(runtime, "recent_metric_days", None)
        try:
            rows = getter(max(1, min(31, int(days or 7)))) if callable(getter) else []
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"metrics_error:{type(exc).__name__}"}, status_code=500)
        return JSONResponse({"ok": True, "days": rows}, headers={"Cache-Control": "no-store"})

    return router


def _metrics_snapshot(assembly: Any) -> dict[str, Any]:
    bridge = getattr(assembly, "bridge", None)
    runtime = getattr(bridge, "runtime", None)
    counters = _safe_call(getattr(runtime, "snapshot_metrics", None), {})
    persistence = _safe_call(getattr(runtime, "metrics_persistence_metadata", None), {})
    last_error = _safe_call(getattr(runtime, "snapshot_last_error", None), {})
    command_state = getattr(bridge, "command_state", None)
    return {
        "ok": True,
        "pid": os.getpid(),
        "runtime": {
            "counters": counters if isinstance(counters, dict) else {},
            "persistence": persistence if isinstance(persistence, dict) else {},
            "last_error": last_error if isinstance(last_error, dict) else {},
        },
        "agent": {
            "attached": bridge is not None,
            "responses_enabled": bool(_safe_call(getattr(command_state, "responses_enabled", None), True)),
            "agent_enabled": bool(_safe_call(getattr(command_state, "is_agent_enabled", None), True)),
        },
    }


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default
