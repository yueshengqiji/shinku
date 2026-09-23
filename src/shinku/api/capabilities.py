"""本机能力目录与待授权请求的只读查询。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__all__ = ["build_capabilities_router"]


def build_capabilities_router(assembly: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/capabilities", response_model=None)
    def capabilities(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        return JSONResponse({"ok": True, "capabilities": _capability_rows(assembly)}, headers={"Cache-Control": "no-store"})

    @router.get("/capabilities/{tool_name}", response_model=None)
    def capability(tool_name: str, request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        rows = [row for row in _capability_rows(assembly) if row["name"] == str(tool_name or "").strip()]
        if not rows:
            return JSONResponse({"ok": False, "reason": "capability_not_found"}, status_code=404)
        return JSONResponse({"ok": True, "capability": rows[0]}, headers={"Cache-Control": "no-store"})

    @router.get("/capabilities/approvals/{conversation_id}", response_model=None)
    def approval(conversation_id: str, request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        bridge = getattr(assembly, "bridge", None)
        store = getattr(bridge, "approval_store", None)
        record = _safe_call(lambda: store.get(conversation_id), None) if store is not None else None
        if record is None:
            return JSONResponse({"ok": True, "pending": False, "approval": None}, headers={"Cache-Control": "no-store"})
        request_obj = getattr(record, "request", None)
        request_data = _safe_call(getattr(request_obj, "as_dict", None), {})
        return JSONResponse(
            {"ok": True, "pending": True, "approval": request_data if isinstance(request_data, dict) else {}},
            headers={"Cache-Control": "no-store"},
        )

    return router


def _capability_rows(assembly: Any) -> list[dict[str, Any]]:
    bridge = getattr(assembly, "bridge", None)
    host = getattr(bridge, "tool_host", None)
    registry = getattr(host, "registry", None)
    names = _safe_call(getattr(registry, "names", None), ())
    policy = getattr(bridge, "policy", None)
    allowed = set(getattr(bridge, "allowed_tool_names", set()) or set())
    rows: list[dict[str, Any]] = []
    for name in names if isinstance(names, (list, tuple, set)) else ():
        handler = _safe_call(lambda current=name: registry.get(current), None)
        metadata = _safe_call(getattr(handler, "tool_metadata", None), None)
        description = ""
        schema: Mapping[str, Any] = {}
        if metadata is not None:
            description = str(getattr(metadata, "description", "") or "").strip()[:900]
            raw_schema = getattr(metadata, "input_schema", {})
            if isinstance(raw_schema, Mapping):
                schema = {str(key): value for key, value in raw_schema.items() if str(key) not in {"x-secret", "secret"}}
        risk = str(policy.risk_of(name) if policy is not None and callable(getattr(policy, "risk_of", None)) else "unknown")
        rows.append(
            {
                "name": str(name),
                "description": description,
                "risk": risk,
                "requires_approval": bool(policy.requires_approval(name)) if policy is not None and callable(getattr(policy, "requires_approval", None)) else None,
                "allowlisted": not allowed or str(name) in allowed,
                "input_schema": schema,
            }
        )
    return rows


def _guard(request: Request) -> JSONResponse | None:
    client = getattr(request, "client", None)
    if str(getattr(client, "host", "") or "").strip().lower() not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
    return None


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default
