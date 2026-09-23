"""独立版 Shinku 的本机控制中枢。

这个面板只允许本机访问，不复用旧项目的前端。它把当前进程、QQ Agent、记忆库、
模型服务、LLM 计数器和最近日志放到一页上，并提供受限的本机模型/Agent 控制。
密钥只作为本机表单输入和服务端落盘值流转，不会出现在状态响应或日志中。
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..health import build_basic_health_payload

__all__ = ["build_debug_hub_router"]


_LOCAL_CALLER_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_LOG_LINES = 120
_MAX_LOG_BYTES = 256 * 1024
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(token\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(password\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
)


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


def _redact_line(value: str) -> str:
    result = str(value or "")
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _tail(path: Path, *, limit: int = _MAX_LOG_LINES) -> list[str]:
    """读日志末尾，避免控制中枢因日志文件变大而占满内存。"""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _MAX_LOG_BYTES), os.SEEK_SET)
            payload = handle.read(_MAX_LOG_BYTES)
    except OSError:
        return []
    text = payload.decode("utf-8", errors="replace")
    lines = [_redact_line(line.rstrip()) for line in text.splitlines() if line.strip()]
    return lines[-max(1, min(int(limit), _MAX_LOG_LINES)) :]


def _log_files(settings: Any) -> list[Path]:
    root = Path(getattr(settings, "log_root", "") or "")
    if not root.is_dir():
        return []
    candidates = {
        Path(settings.log_path("backend")),
        Path(settings.log_path("backend", error=True)),
    }
    # 外部启动器可能把 stdout/stderr 重定向到带端口或批次名的文件。
    candidates.update(root.glob("backend*.log"))
    existing = [path for path in candidates if path.is_file()]
    return sorted(existing, key=lambda item: item.stat().st_mtime, reverse=True)[:8]


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default


def _agent_snapshot(assembly: Any) -> dict[str, Any]:
    if assembly is None:
        return {"enabled": False, "status": "not_attached"}

    diagnostics = _safe_call(getattr(assembly, "diagnostics", None), {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    bridge = getattr(assembly, "bridge", None)
    runtime = getattr(bridge, "runtime", None)
    metrics = _safe_call(getattr(runtime, "snapshot_metrics", None), {})
    last_error = _safe_call(getattr(runtime, "snapshot_last_error", None), {})
    memory_service = getattr(bridge, "memory_service", None)
    memory_store = getattr(memory_service, "store", None)
    memory_stats = _safe_call(getattr(memory_store, "stats", None), {})
    memory_path = str(getattr(memory_store, "path", "") or "")
    tool_host = getattr(bridge, "tool_host", None)
    registry = getattr(tool_host, "registry", None)
    names_getter = getattr(registry, "names", None)
    tool_names = _safe_call(names_getter, ())
    if not isinstance(tool_names, (list, tuple, set)):
        tool_names = ()
    tool_names = tuple(str(item) for item in tool_names if str(item).strip())
    runtime_health = {}
    health_getter = getattr(tool_host, "runtime_health", None)
    if callable(health_getter):
        health_value = _safe_call(lambda: health_getter(runtime), None)
        as_dict = getattr(health_value, "as_dict", None)
        if callable(as_dict):
            runtime_health = _safe_call(as_dict, {})
        elif isinstance(health_value, dict):
            runtime_health = health_value
    chat_bundle = getattr(runtime, "chat", None)
    aux_bundle = getattr(runtime, "aux", None)
    command_state = getattr(bridge, "command_state", None)
    task_coordinator = getattr(assembly, "task_coordinator", None)
    task_snapshot = _safe_call(getattr(task_coordinator, "list", None), []) if task_coordinator is not None else []
    if not isinstance(task_snapshot, list):
        task_snapshot = []
    return {
        "enabled": True,
        "status": "attached",
        "process": {
            "pid": os.getpid(),
            "mode": "embedded",
            "service": "backend",
            "note": "Agent 与 Shinku backend 共用一个进程，不是单独的 agent.exe。",
        },
        "controls": {
            "agent_enabled": bool(_safe_call(getattr(command_state, "is_agent_enabled", None), True)),
            "stopped": bool(_safe_call(getattr(command_state, "is_stopped", None), False)),
            "responses_enabled": bool(_safe_call(getattr(command_state, "responses_enabled", None), True)),
        },
        "live_models": {
            "chat": str(getattr(chat_bundle, "model", "") or ""),
            "aux": str(getattr(aux_bundle, "model", "") or ""),
        },
        "diagnostics": diagnostics,
        "runtime": {
            "metrics": metrics if isinstance(metrics, dict) else {},
            "last_error": last_error if isinstance(last_error, dict) else {},
        },
        "memory": {
            "path": memory_path,
            "stats": memory_stats if isinstance(memory_stats, dict) else {},
        },
        "bridge": {
            "sender_configured": getattr(bridge, "sender", None) is not None,
            "allowed_tool_count": len(tool_names),
            "tool_names": list(tool_names),
            "tool_allowlist": sorted(getattr(bridge, "allowed_tool_names", set()) or set()),
            "runtime_health": runtime_health,
            "max_rounds": getattr(bridge, "max_rounds", None),
            "history_turns": getattr(bridge, "history_turns", None),
            "recent_context_messages": getattr(bridge, "recent_context_messages", None),
        },
        "tasks": {
            "count": len(task_snapshot),
            "items": task_snapshot[:50],
        },
    }


def _trace_snapshot(assembly: Any, *, limit: int = 30) -> list[dict[str, Any]]:
    """读取 QQ Agent 的有限结构化回合追踪，不触碰聊天原文。"""

    bridge = getattr(assembly, "bridge", None)
    store = getattr(bridge, "trace_store", None)
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        rows = snapshot(limit=max(1, min(100, int(limit or 30))))
    except Exception:
        return []
    return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _model_services_snapshot(controller: Any) -> dict[str, Any]:
    if controller is None:
        return {"available": False, "status": "not_attached", "providers": []}
    try:
        from ..providers.catalog import public_model_services_snapshot

        snapshot = public_model_services_snapshot(
            controller.store,
            controller.config_module,
            source="local_control_center",
        )
        snapshot["available"] = True
        return snapshot
    except Exception as exc:
        return {
            "available": True,
            "status": "error",
            "reason": str(exc)[:240],
            "providers": [],
        }


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
        "process": {"pid": os.getpid()},
        "agent": _agent_snapshot(assembly),
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


_FRONTEND_ROOT = Path(__file__).resolve().parents[3] / "frontend" / "control-center"
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
        if not _local_only(request):
            return _forbidden()
        bridge = getattr(assembly, "bridge", None)
        state = getattr(bridge, "command_state", None)
        if state is None:
            return JSONResponse(
                {"ok": False, "status": "agent_not_attached", "reason": "qq_agent_not_attached"},
                status_code=409,
            )
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        action = str(payload.get("action", "") if isinstance(payload, dict) else "").strip().casefold()
        if action in {"start", "resume", "on"}:
            state.set_agent_enabled(True)
            state.set_stopped(False)
            result = "started"
        elif action in {"stop", "pause"}:
            state.set_stopped(True)
            result = "stopped"
        elif action in {"off", "disable"}:
            state.set_agent_enabled(False)
            result = "agent_disabled"
        elif action in {"status", ""}:
            result = "status"
        else:
            return JSONResponse(
                {"ok": False, "status": "invalid_action", "reason": "use start, stop, on, off, or status"},
                status_code=400,
            )
        return JSONResponse(
            {"ok": True, "status": result, "agent": _agent_snapshot(assembly)},
            headers={"Cache-Control": "no-store"},
        )

    return router
