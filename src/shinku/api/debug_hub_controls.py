"""Local Agent control and audit recording."""
from __future__ import annotations
from typing import Any, Callable
from fastapi import Request
from fastapi.responses import JSONResponse
from ..qq.commands import AGENT_PERMISSION_MODES
from .debug_hub_logs import _safe_call

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


async def control_agent(
    request: Request,
    *,
    assembly: Any,
    local_only: Callable[..., Any],
    forbidden: Callable[..., Any],
    snapshot: Callable[..., Any],
    audit_mode: Callable[..., Any],
) -> JSONResponse:
    if not local_only(request):
        return forbidden()
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
        previous_mode = str(
            _safe_call(getattr(state, 'permission_mode', None), 'traditional') or 'traditional',
        )
        if not state.set_permission_mode(mode):
            return JSONResponse(
                {"ok": False, "status": "permission_mode_save_failed", "reason": "permission_mode_not_saved"},
                status_code=500,
            )
        audit_mode(bridge, previous_mode, mode)
        result = f"permission_{mode}"
    elif action in {"status", ""}:
        result = "status"
    else:
        return JSONResponse(
            {"ok": False, "status": "invalid_action", "reason": "use start, stop, on, off, or status"},
            status_code=400,
        )
    return JSONResponse(
        {"ok": True, "status": result, "agent": snapshot(assembly)},
        headers={"Cache-Control": "no-store"},
    )
