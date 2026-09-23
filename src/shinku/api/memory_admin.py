"""本机记忆时间线、摘要和显式维护入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__all__ = ["build_memory_admin_router"]


def build_memory_admin_router(service: Any) -> APIRouter:
    if service is None:
        raise TypeError("memory service is required")
    router = APIRouter()

    @router.get("/memory/stats", response_model=None)
    def stats(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        return JSONResponse({"ok": True, "stats": _safe_call(getattr(service.store, "stats", None), {})})

    @router.get("/memory/timeline", response_model=None)
    def timeline(
        request: Request,
        conversation_id: str = "",
        conversation_type: str = "private",
        query: str = "",
        mode: str = "recent",
        limit: int = 8,
    ) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        if not str(conversation_id or "").strip():
            return JSONResponse({"ok": False, "reason": "conversation_id_required"}, status_code=400)
        try:
            result = service.inspect(
                conversation_id=str(conversation_id).strip(),
                conversation_type=str(conversation_type or "private").strip(),
                query=str(query or "")[:500],
                mode=str(mode or "recent").strip(),
                limit=max(1, min(50, int(limit or 8))),
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"memory_inspection_failed:{type(exc).__name__}"}, status_code=500)
        return JSONResponse({"ok": True, **result}, headers={"Cache-Control": "no-store"})

    @router.get("/memory/facts", response_model=None)
    def facts(request: Request, conversation_id: str = "", conversation_type: str = "private", limit: int = 20) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        if not str(conversation_id or "").strip():
            return JSONResponse({"ok": False, "reason": "conversation_id_required"}, status_code=400)
        scope = service.scope_for(conversation_id, conversation_type)
        rows = service.store.timeline(scope=scope, limit=max(20, min(200, int(limit or 20))), include_facts=True)
        selected = [row.as_dict() for row in rows if str(getattr(row, "layer", "")) == "fact"][: max(1, min(50, int(limit or 20)))]
        return JSONResponse({"ok": True, "scope": scope, "facts": selected}, headers={"Cache-Control": "no-store"})

    @router.post("/memory/rollup", response_model=None)
    async def rollup(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope = str(payload.get("scope") or "").strip()
        if not scope:
            conversation_id = str(payload.get("conversation_id") or "").strip()
            conversation_type = str(payload.get("conversation_type") or "private").strip()
            scope = service.scope_for(conversation_id, conversation_type) if conversation_id else ""
        if not scope:
            return JSONResponse({"ok": False, "reason": "scope_or_conversation_id_required"}, status_code=400)
        try:
            created = service.store.rollup_daily_summaries(scope=scope, max_lines=max(1, min(100, int(payload.get("max_lines", 60) or 60))))
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"memory_rollup_failed:{type(exc).__name__}"}, status_code=500)
        return JSONResponse({"ok": True, "scope": scope, "created": created})

    @router.post("/memory/maintenance", response_model=None)
    async def maintenance(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        result = _safe_call(getattr(service, "maintenance", None), {})
        return JSONResponse({"ok": True, **(result if isinstance(result, Mapping) else {})})

    @router.post("/memory/forget", response_model=None)
    async def forget(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        if payload.get("confirm") is not True:
            return JSONResponse({"ok": False, "reason": "explicit_confirm_required"}, status_code=400)
        query = str(payload.get("query") or "").strip()
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not query or not conversation_id:
            return JSONResponse({"ok": False, "reason": "query_and_conversation_id_required"}, status_code=400)
        scope = service.scope_for(conversation_id, str(payload.get("conversation_type") or "private"))
        purge = bool(payload.get("purge"))
        try:
            forgotten, purged = service.store.forget(query=query[:240], scope=scope, purge=purge)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"memory_forget_failed:{type(exc).__name__}"}, status_code=500)
        return JSONResponse({"ok": True, "scope": scope, "forgotten": forgotten, "purged": purged, "purge": purge})

    return router


def _guard(request: Request) -> JSONResponse | None:
    client = getattr(request, "client", None)
    if str(getattr(client, "host", "") or "").strip().lower() not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
    return None


async def _json(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default
