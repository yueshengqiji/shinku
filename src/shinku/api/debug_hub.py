"""独立版 Shinku 的本机控制中枢。

这个面板只允许本机访问，不复用旧项目的前端。它把当前进程、QQ Agent、记忆库、
模型服务、LLM 计数器和最近日志放到一页上，并提供受限的本机模型/Agent 控制。
密钥只作为本机表单输入和服务端落盘值流转，不会出现在状态响应或日志中。
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..health import build_basic_health_payload
from ..qq.commands import AGENT_PERMISSION_LABELS, AGENT_PERMISSION_MODES

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


def _audit_permission_mode_change(bridge: Any, previous_mode: str, permission_mode: str) -> None:
    """Record control-center permission changes through the private audit sink."""

    if previous_mode == permission_mode:
        return
    tool_host = getattr(bridge, "tool_host", None)
    registry = getattr(tool_host, "registry", None)
    handlers = _safe_call(getattr(registry, "view", None), {})
    if not isinstance(handlers, dict):
        return
    for handler in handlers.values():
        config = getattr(handler, "config", None)
        audit = getattr(config, "audit", None)
        if not callable(audit):
            continue
        try:
            audit(
                "agent_permission_mode_changed",
                previous_mode=previous_mode,
                permission_mode=permission_mode,
                source="control_center",
            )
        except Exception:
            pass
        return


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
    permission_mode = str(_safe_call(getattr(command_state, "permission_mode", None), "traditional") or "traditional")
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
            "permission_mode": permission_mode,
            "permission_mode_label": AGENT_PERMISSION_LABELS.get(permission_mode, AGENT_PERMISSION_LABELS["traditional"]),
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
            "lifecycle": _safe_call(getattr(getattr(memory_store, "lifecycle", None), "status", None), {}),
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


def _attention_snapshot(napcat_host: Any, assembly: Any, *, limit: int = 80) -> dict[str, Any]:
    """Join content-free ingress decisions with any resulting QQ turn trace."""

    ingress = getattr(napcat_host, "ingress", None)
    store = getattr(ingress, "attention_trace_store", None)
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot):
        return {"available": False, "summary": {}, "items": []}
    try:
        payload = snapshot(limit=max(1, min(300, int(limit or 80))))
    except Exception:
        return {"available": False, "summary": {}, "items": []}
    if not isinstance(payload, dict):
        return {"available": False, "summary": {}, "items": []}

    traces = _trace_snapshot(assembly, limit=100)
    by_message_id: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in traces:
        message_ids = trace.get("message_ids")
        if not isinstance(message_ids, (list, tuple)):
            message_ids = [trace.get("message_id", "")]
        for message_id in message_ids:
            identifier = str(message_id or "").strip()
            if identifier:
                key = (str(trace.get("conversation_id") or "").strip(), identifier)
                by_message_id[key] = trace

    items: list[dict[str, Any]] = []
    for raw_item in payload.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        trace_key = (
            str(item.get("conversation_id") or "").strip(),
            str(item.get("message_id") or "").strip(),
        )
        trace = by_message_id.get(trace_key)
        if trace is not None:
            stages = trace.get("stages") if isinstance(trace.get("stages"), list) else []
            usage_stages = [
                stage for stage in stages
                if isinstance(stage, dict) and stage.get("name") == "llm_usage"
            ]
            usage: dict[str, Any] = {"calls": 0}
            for stage in usage_stages:
                usage["calls"] += max(0, int(stage.get("calls") or 0))
                for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens"):
                    if key in stage:
                        usage[key] = int(usage.get(key, 0)) + max(0, int(stage.get(key) or 0))
                channel = str(stage.get("channel") or "").strip()
                if channel:
                    channels = usage.setdefault("channels", {})
                    channels[channel] = int(channels.get(channel, 0)) + max(0, int(stage.get("calls") or 0))
            item["turn"] = {
                "trace_id": trace.get("trace_id", ""),
                "status": trace.get("status", ""),
                "agent_status": trace.get("agent_status", ""),
                "sent": bool(trace.get("sent")),
                "delivery_status": trace.get("delivery_status", ""),
                "reason": trace.get("reason", ""),
                "elapsed_ms": trace.get("elapsed_ms"),
                "llm_usage": usage if usage_stages else {"available": False},
            }
        elif not bool((item.get("attention") or {}).get("should_schedule")):
            item["turn"] = {"status": "not_scheduled", "llm_usage": {"calls": 0}}
        else:
            item["turn"] = {"status": "pending_or_trace_expired", "llm_usage": {"available": False}}
        items.append(item)
    return {
        "available": True,
        "summary": payload.get("summary", {}),
        "items": items,
    }


def _memory_routing_snapshot(assembly: Any, *, limit: int = 60) -> dict[str, Any]:
    """Expose bounded memory-routing metadata without query or message text."""

    bridge = getattr(assembly, "bridge", None)
    memory_service = getattr(bridge, "memory_service", None)
    store = getattr(memory_service, "store", None)
    reader = getattr(store, "recent_routing_audits", None)
    if not callable(reader):
        return {"available": False, "summary": {}, "items": []}
    try:
        rows = reader(limit=max(1, min(100, int(limit or 60))))
    except Exception:
        return {"available": False, "summary": {}, "items": []}
    if not isinstance(rows, list):
        return {"available": False, "summary": {}, "items": []}

    safe_reasons = {
        "empty_message",
        "forget_command",
        "explicit_summary_request",
        "explicit_recall_request",
        "obvious_short_chat",
        "current_task_request",
        "local_recent_topic",
        "local_summary_intent",
        "local_no_history_signal",
        "no_memory_judge_configured",
        "memory_judge_error",
        "memory_router_fallback",
        "model_memory_judgement",
    }
    items: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    model_judged = 0
    retrieved = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        mode = str(row.get("history_mode") or "none").strip().lower()
        if mode not in {"none", "recent", "retrieve", "summary"}:
            mode = "none"
        used_model = bool(row.get("used_model"))
        raw_reason = str(row.get("reason") or "")
        reason = raw_reason if raw_reason in safe_reasons else ("model_judgement" if used_model else "local_decision")
        signals = decision.get("local_signals")
        if not isinstance(signals, (list, tuple)):
            signals = ()
        safe_signals = [
            str(value)[:48]
            for value in signals[:8]
            if str(value).replace("_", "").isalnum()
        ]
        item = {
            "audit_id": str(row.get("audit_id") or "")[:80],
            "conversation_id": str(row.get("conversation_id") or "")[:160],
            "scope": str(row.get("scope") or "")[:160],
            "created_at": max(0, int(row.get("created_at") or 0)),
            "history_mode": mode,
            "need_retrieval": bool(row.get("need_retrieval")),
            "used_model": used_model,
            "confidence": max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
            "reason": reason,
            "local_signals": safe_signals,
            "retrieved_count": max(0, int(row.get("retrieved_count") or 0)),
            "injected_chars": max(0, int(row.get("injected_chars") or 0)),
        }
        items.append(item)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        model_judged += int(used_model)
        retrieved += int(item["retrieved_count"] > 0)
    return {
        "available": True,
        "summary": {
            "window_items": len(items),
            "mode_counts": mode_counts,
            "model_judged": model_judged,
            "with_results": retrieved,
        },
        "items": items,
    }


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


def _find_electron_pids() -> list[int]:
    project_root = Path(__file__).resolve().parents[3]
    runtime_file = project_root / "private" / "desktop-pet" / "pet_runtime.json"

    live_electron_pids: list[int] = []
    try:
        res = subprocess.run(
            ["tasklist", "/fi", "imagename eq electron.exe", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for row in csv.reader(io.StringIO(res.stdout)):
            if row and len(row) > 1 and row[0].lower() == "electron.exe":
                try:
                    live_electron_pids.append(int(row[1]))
                except ValueError:
                    pass
    except Exception:
        pass

    if runtime_file.is_file():
        try:
            with runtime_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            recorded_pid = int(data.get("pid", 0))
            if recorded_pid and recorded_pid in live_electron_pids:
                return [recorded_pid] + [p for p in live_electron_pids if p != recorded_pid]
        except Exception:
            pass

    return live_electron_pids


def _pet_snapshot(settings: Any, assembly: Any) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    pet_dir = project_root / "private" / "desktop-pet"
    settings_file = pet_dir / "settings.json"
    live2d_dir = project_root / "private" / "assets" / "live2d" / "shinku"

    settings_data = {
        "scale": 1.0,
        "opacity": 1.0,
        "alwaysOnTop": True,
        "soundEnabled": True,
    }
    if settings_file.is_file():
        try:
            with settings_file.open("r", encoding="utf-8") as f:
                settings_data.update(json.load(f))
        except Exception:
            pass

    live2d_model_found = False
    live2d_model_name = ""
    if live2d_dir.is_dir():
        for mf in live2d_dir.rglob("*.model*.json"):
            live2d_model_found = True
            live2d_model_name = mf.name
            break

    pids = _find_electron_pids()
    is_running = len(pids) > 0

    traces = _trace_snapshot(assembly, limit=50)
    last_turn = None
    for trace in traces:
        if trace.get("conversation_type") == "desktop":
            time_str = ""
            ts = trace.get("finished_at") or trace.get("started_at")
            if ts:
                try:
                    time_str = datetime.fromtimestamp(ts, timezone.utc).astimezone().strftime("%H:%M:%S")
                except Exception:
                    pass
            last_turn = {
                "trace_id": trace.get("trace_id", ""),
                "elapsed_ms": trace.get("elapsed_ms", 0),
                "input_chars": trace.get("input_chars", 0),
                "reply_chars": trace.get("reply_chars", 0),
                "status": trace.get("status", ""),
                "time": time_str,
            }
            break

    return {
        "available": True,
        "status": "online" if is_running else "offline",
        "pid": pids[0] if pids else None,
        "all_pids": pids,
        "character_name": "二阶堂真红",
        "worldview": "风之滨 (风之浜) / 伴侣",
        "window": {"width": 340, "height": 560},
        "live2d": {"loaded": live2d_model_found, "model": live2d_model_name},
        "settings": settings_data,
        "last_turn": last_turn,
        "launcher_available": (pet_dir / "start_pet.cmd").is_file(),
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
        elif action in {"permission", "permission_mode"}:
            mode = str(payload.get("mode", "") if isinstance(payload, dict) else "").strip().casefold()
            if mode not in AGENT_PERMISSION_MODES:
                return JSONResponse(
                    {"ok": False, "status": "invalid_permission_mode", "reason": "use traditional, readonly, confirm, full, or ultra"},
                    status_code=400,
                )
            if mode in {"full", "ultra"} and payload.get("acknowledge_risk") is not True:
                return JSONResponse(
                    {"ok": False, "status": "explicit_acknowledgement_required", "reason": "confirm_full_or_ultra_in_control_center"},
                    status_code=400,
                )
            previous_mode = str(_safe_call(getattr(state, "permission_mode", None), "traditional") or "traditional")
            if not state.set_permission_mode(mode):
                return JSONResponse(
                    {"ok": False, "status": "permission_mode_save_failed", "reason": "permission_mode_not_saved"},
                    status_code=500,
                )
            _audit_permission_mode_change(bridge, previous_mode, mode)
            result = f"permission_{mode}"
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

    async def _do_control_pet(action: str) -> JSONResponse:
        project_root = Path(__file__).resolve().parents[3]
        pet_dir = project_root / "private" / "desktop-pet"
        start_cmd = pet_dir / "start_pet.cmd"

        if action == "launch":
            pids = _find_electron_pids()
            if pids:
                return JSONResponse({"ok": True, "status": "already_running", "pids": pids, "pet": _pet_snapshot(settings, assembly)})
            if not start_cmd.is_file():
                return JSONResponse({"ok": False, "status": "launcher_not_found"}, status_code=404)
            flags = 0x00000008 | 0x00000200
            subprocess.Popen(
                ["cmd.exe", "/c", str(start_cmd)],
                cwd=str(pet_dir),
                creationflags=flags,
                close_fds=True,
            )
            return JSONResponse({"ok": True, "status": "launched", "pet": _pet_snapshot(settings, assembly)})

        elif action in {"stop", "kill", "quit"}:
            pids = _find_electron_pids()
            if not pids:
                return JSONResponse({"ok": True, "status": "not_running", "pet": _pet_snapshot(settings, assembly)})
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True, timeout=3)
                except Exception:
                    pass
            runtime_file = pet_dir / "pet_runtime.json"
            if runtime_file.is_file():
                try:
                    runtime_file.unlink()
                except Exception:
                    pass
            return JSONResponse({"ok": True, "status": "stopped", "pet": _pet_snapshot(settings, assembly)})

        elif action == "restart":
            pids = _find_electron_pids()
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True, timeout=3)
                except Exception:
                    pass
            import asyncio
            await asyncio.sleep(0.5)
            flags = 0x00000008 | 0x00000200
            subprocess.Popen(
                ["cmd.exe", "/c", str(start_cmd)],
                cwd=str(pet_dir),
                creationflags=flags,
                close_fds=True,
            )
            return JSONResponse({"ok": True, "status": "restarted", "pet": _pet_snapshot(settings, assembly)})

        elif action in {"status", ""}:
            return JSONResponse({"ok": True, "status": "status", "pet": _pet_snapshot(settings, assembly)})

        else:
            return JSONResponse(
                {"ok": False, "status": "invalid_action", "reason": "use launch, stop, restart, or status"},
                status_code=400,
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
