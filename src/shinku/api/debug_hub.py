"""独立版 Shinku 的本机控制中枢。

这个面板只允许本机访问，不复用旧项目的前端。它把当前进程、QQ Agent、记忆库、
模型服务、LLM 计数器和最近日志放到一页上，并提供受限的本机模型/Agent 控制。
密钥只作为本机表单输入和服务端落盘值流转，不会出现在状态响应或日志中。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..frontend import control_center_root
from ..character import SHINKU_CHARACTER
from ..health import build_basic_health_payload

from . import debug_hub_controls, debug_hub_pet
from .debug_hub_logs import _tail, _log_files, _safe_call
from .debug_hub_snapshots import _agent_snapshot, _trace_snapshot, _attention_snapshot, _memory_routing_snapshot, _model_services_snapshot
from .debug_hub_pet import _find_electron_pids
from .debug_hub_controls import _audit_permission_mode_change

__all__ = ["build_debug_hub_router"]


_LOCAL_CALLER_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _local_only(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    return host in _LOCAL_CALLER_HOSTS


def _forbidden() -> JSONResponse:
    return JSONResponse(
        {"ok": False, "status": "forbidden", "reason": "local_request_required"},
        status_code=403,
        headers={"Cache-Control": "no-store"},
    )


def _pet_snapshot(settings: Any, assembly: Any) -> dict[str, Any]:
    # 显式实例尚未配置桌宠资产，不能读取或控制默认角色的共享桌宠目录。
    character = getattr(getattr(assembly, "bridge", None), "character", SHINKU_CHARACTER)
    if not character.legacy_audience:
        return {"available": False, "status": "not_configured", "character_name": character.name}
    return debug_hub_pet.pet_snapshot(
        settings,
        assembly,
        find_pids=_find_electron_pids,
        trace_snapshot=_trace_snapshot,
    )


def _state(
    *,
    settings: Any,
    assembly: Any,
    napcat_host: Any,
    napcat_config: Any,
    model_services_controller: Any = None,
) -> dict[str, Any]:
    health = build_basic_health_payload()
    health.update(
        {
            "project": getattr(settings, "project", "shinku"),
            "version": getattr(settings, "version", ""),
            "service": "backend",
            "bind": settings.binds.get("backend", "127.0.0.1"),
            "port": settings.ports.get("backend"),
            "allow_lan": getattr(settings, "allow_lan", False),
            "data_root": str(getattr(settings, "data_root", "")),
            "config_root": str(getattr(settings, "config_root", "")),
            "log_root": str(getattr(settings, "log_root", "")),
        }
    )
    logs = []
    for path in _log_files(settings):
        try:
            stat = path.stat()
            logs.append(
                {
                    "name": path.name,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "lines": _tail(path),
                }
            )
        except OSError:
            continue

    napcat = _safe_call(getattr(napcat_config, "diagnostics", None), {})
    if not isinstance(napcat, dict):
        napcat = {}
    return {
        "ok": True,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "read_only": False,
        "health": health,
        "character": {"id": getattr(getattr(assembly, "bridge", None), "character", SHINKU_CHARACTER).character_id,
                      "name": getattr(getattr(assembly, "bridge", None), "character", SHINKU_CHARACTER).name},
        "process": {"pid": os.getpid()},
        "agent": _agent_snapshot(assembly),
        "pet": _pet_snapshot(settings, assembly),
        "model_services": _model_services_snapshot(model_services_controller),
        "napcat": {
            "configured": napcat_host is not None,
            "diagnostics": napcat,
            "transport": type(getattr(napcat_host, "transport", None)).__name__
            if napcat_host is not None and getattr(napcat_host, "transport", None) is not None
            else "unconfigured",
            "dispatcher_attached": napcat_host is not None
            and getattr(napcat_host, "turn_dispatcher", None) is not None,
        },
        "logs": logs,
    }


_FRONTEND_ROOT = control_center_root()
_FRONTEND_ASSETS = frozenset({"index.html", "app.css", "app.js"})


def _frontend_asset(name: str) -> Path:
    asset = str(name or "").strip()
    if asset not in _FRONTEND_ASSETS:
        raise FileNotFoundError("unknown_debug_hub_asset")
    path = _FRONTEND_ROOT / asset
    if not path.is_file():
        raise FileNotFoundError(f"missing_debug_hub_asset:{asset}")
    return path


def build_debug_hub_router(
    *,
    settings: Any,
    assembly: Any = None,
    napcat_host: Any = None,
    napcat_config: Any = None,
    model_services_controller: Any = None,
) -> APIRouter:
    """装配控制中枢路由。所有端点都要求请求来自本机。"""

    router = APIRouter()

    def _asset_response(request: Request, name: str, media_type: str) -> Any:
        if not _local_only(request):
            return _forbidden()
        try:
            path = _frontend_asset(name)
        except FileNotFoundError:
            return JSONResponse(
                {"ok": False, "status": "frontend_asset_missing", "asset": name},
                status_code=500,
                headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
            )
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
        )

    @router.get("/debug/hub", response_model=None)
    def debug_hub(request: Request) -> Any:
        return _asset_response(request, "index.html", "text/html")

    @router.get("/debug/hub/assets/app.css", response_model=None)
    def debug_hub_css(request: Request) -> Any:
        return _asset_response(request, "app.css", "text/css")

    @router.get("/debug/hub/assets/app.js", response_model=None)
    def debug_hub_js(request: Request) -> Any:
        return _asset_response(request, "app.js", "text/javascript")

    @router.get("/debug/hub/state", response_model=None)
    def debug_hub_state(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            _state(
                settings=settings,
                assembly=assembly,
                napcat_host=napcat_host,
                napcat_config=napcat_config,
                model_services_controller=model_services_controller,
            ),
            headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
        )

    @router.get("/debug/hub/traces", response_model=None)
    def debug_hub_traces(request: Request, limit: int = 30) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            {"ok": True, "traces": _trace_snapshot(assembly, limit=limit)},
            headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
        )

    @router.get("/debug/hub/attention", response_model=None)
    def debug_hub_attention(request: Request, limit: int = 80) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            _attention_snapshot(napcat_host, assembly, limit=limit),
            headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
        )

    @router.get("/debug/hub/memory-routing", response_model=None)
    def debug_hub_memory_routing(request: Request, limit: int = 60) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            _memory_routing_snapshot(assembly, limit=limit),
            headers={"Cache-Control": "no-store", "X-Shinku-Debug-Mode": "control"},
        )

    @router.get("/control-center/agent", response_model=None)
    def agent_control_state(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            {"ok": True, "agent": _agent_snapshot(assembly)},
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/control-center/agent/control", response_model=None)
    async def control_agent(request: Request) -> JSONResponse:
        return await debug_hub_controls.control_agent(
            request,
            assembly=assembly,
            local_only=_local_only,
            forbidden=_forbidden,
            snapshot=_agent_snapshot,
            audit_mode=_audit_permission_mode_change,
        )

    async def _do_control_pet(action: str) -> JSONResponse:
        character = getattr(getattr(assembly, "bridge", None), "character", SHINKU_CHARACTER)
        if not character.legacy_audience:
            return JSONResponse({"ok": False, "status": "instance_pet_not_configured"}, status_code=409)
        return await debug_hub_pet.control_pet(
            action,
            settings=settings,
            assembly=assembly,
            find_pids=_find_electron_pids,
            snapshot=_pet_snapshot,
        )

    @router.get("/control-center/pet", response_model=None)
    def pet_state(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(
            {"ok": True, "pet": _pet_snapshot(settings, assembly)},
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/control-center/pet/launch", response_model=None)
    async def launch_pet(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return await _do_control_pet("launch")

    @router.post("/control-center/pet/control", response_model=None)
    async def control_pet(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        action = str(payload.get("action", "") if isinstance(payload, dict) else "").strip().casefold()
        return await _do_control_pet(action)

    return router
