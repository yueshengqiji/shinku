"""本机记忆时间线、摘要和显式维护入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__all__ = ["build_memory_admin_router"]


def build_memory_admin_router(service: Any) -> APIRouter:
    if service is None:
        raise TypeError("memory service is required")
    router = APIRouter()

    @router.get("/memory/features", response_model=None)
    def features(request: Request) -> JSONResponse:
        """Report loaded switches and components without reading memory data."""
        denied = _guard(request)
        if denied is not None:
            return denied
        policy = getattr(service, "policy", None)
        names = ("search_keys_enabled", "current_understanding_enabled",
                 "evidence_package_enabled", "history_needs_enabled")
        initialized = {
            "search_keys_enabled": bool(getattr(service.store, "search_keys_enabled", False)
                                        and getattr(service.store, "search_index", None) is not None),
            "current_understanding_enabled": getattr(service, "current_understanding", None) is not None,
            "evidence_package_enabled": getattr(service, "evidence_reader", None) is not None,
            "history_needs_enabled": getattr(service, "history_workbench", None) is not None,
        }
        configured = {name: bool(getattr(policy, name, False)) for name in names}
        effective = {name: configured[name] and initialized[name] for name in names}
        return JSONResponse({"ok": True, "configured": configured, "initialized": initialized,
                             "effective": effective}, headers={"Cache-Control": "no-store"})

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

    @router.get("/memory/lifecycle", response_model=None)
    def lifecycle_status(request: Request, scope: str = "", limit: int = 20) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        lifecycle = service.store.lifecycle
        result = lifecycle.status()
        review = getattr(service, "lifecycle_review", None)
        if review is not None:
            result["semantic_review"] = review.status()
        if scope:
            result["inspection"] = lifecycle.inspect(scope=scope, limit=limit)
        return JSONResponse({"ok": True, **result}, headers={"Cache-Control": "no-store"})

    @router.post("/memory/lifecycle/preview", response_model=None)
    async def lifecycle_preview(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope = str(payload.get("scope") or "").strip()
        if not scope:
            return JSONResponse({"ok": False, "reason": "exact_scope_required"}, status_code=400)
        result = service.store.lifecycle.preview(scope=scope, now=int(service.clock()))
        return JSONResponse({"ok": True, **result}, headers={"Cache-Control": "no-store"})

    @router.post("/memory/lifecycle/scan", response_model=None)
    async def lifecycle_scan(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope = str(payload.get("scope") or "").strip()
        if not scope or payload.get("confirm") is not True:
            return JSONResponse({"ok": False, "reason": "scope_and_explicit_confirmation_required"}, status_code=400)
        reclaim = payload.get("allow_reclaim") is True
        if reclaim and (payload.get("approved_scope") != scope or payload.get("reclamation_confirmation") != "RECLAIM_MEMORY_INDEX_RAW"):
            return JSONResponse({"ok": False, "reason": "independent_reclamation_confirmation_required"}, status_code=400)
        try:
            result = service.store.lifecycle.scan(now=int(service.clock()), scope=scope, allow_reclaim=reclaim, approved_scope=str(payload.get("approved_scope") or ""))
        except ValueError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **result}, headers={"Cache-Control": "no-store"})

    @router.post("/memory/lifecycle/protect", response_model=None)
    async def lifecycle_protect(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope = str(payload.get("scope") or "").strip()
        record_id = str(payload.get("record_id") or "").strip()
        if not scope or not record_id or payload.get("confirm") is not True or not isinstance(payload.get("pinned"), bool):
            return JSONResponse({"ok": False, "reason": "scope_record_pin_and_confirmation_required"}, status_code=400)
        try:
            applied = service.store.lifecycle.set_pin(record_id=record_id, scope=scope, pinned=payload["pinned"], now=int(service.clock()), confirmed=True)
        except ValueError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
        return JSONResponse({"ok": applied, "scope": scope, "record_id": record_id}, status_code=200 if applied else 404)

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

    @router.post("/memory/lifecycle/correct-fact", response_model=None)
    async def lifecycle_correct_fact(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        try:
            applied = service.store.lifecycle.correct_fact(record_id=str(payload.get("record_id") or ""),
                scope=str(payload.get("scope") or ""), expected_hash=str(payload.get("expected_hash") or ""),
                replacement=str(payload.get("replacement") or ""), now=int(service.clock()), confirmed=payload.get("confirm") is True)
        except ValueError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
        return JSONResponse({"ok": applied, "reason": "corrected" if applied else "stale_or_wrong_scope"}, status_code=200 if applied else 409)

    @router.post("/memory/lifecycle/revoke-record", response_model=None)
    async def lifecycle_revoke_record(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope, record_id = str(payload.get("scope") or ""), str(payload.get("record_id") or "")
        if not scope or not record_id or payload.get("confirm") is not True:
            return JSONResponse({"ok": False, "reason": "exact_scope_record_confirmation_required"}, status_code=400)
        try:
            forgotten, _ = service.store.forget(query="", scope=scope, record_id=record_id, now=int(service.clock()))
        except ValueError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
        return JSONResponse({"ok": forgotten > 0, "forgotten": forgotten}, status_code=200 if forgotten else 404)

    @router.get("/memory/current-understanding", response_model=None)
    def current_understanding(request: Request, scope: str = "", limit: int = 20) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied
        view = getattr(service, "current_understanding", None)
        if view is None or not service.policy.current_understanding_enabled:
            return JSONResponse({"ok": True, "enabled": False}, headers={"Cache-Control": "no-store"})
        if not scope.strip():
            return JSONResponse({"ok": False, "reason": "exact_scope_required"}, status_code=400)
        try:
            result = view.inspect(scope=scope, now=int(service.clock()), limit=limit)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": "inspection_failed", "error_type": type(exc).__name__}, status_code=503)
        return JSONResponse({"ok": True, "enabled": True, **result}, headers={"Cache-Control": "no-store"})

    @router.post("/memory/current-understanding/rebuild", response_model=None)
    async def current_understanding_rebuild(request: Request) -> JSONResponse:
        denied = _lifecycle_write_guard(request)
        if denied is not None:
            return denied
        payload = await _json(request)
        scope = str(payload.get("scope") or "").strip()
        view = getattr(service, "current_understanding", None)
        if view is None or not service.policy.current_understanding_enabled:
            return JSONResponse({"ok": False, "reason": "current_view_disabled"}, status_code=409)
        dry_run = payload.get("dry_run", True)
        if not scope or not isinstance(dry_run, bool) or not dry_run and payload.get("confirm") is not True:
            return JSONResponse({"ok": False, "reason": "exact_scope_preview_or_confirmation_required"}, status_code=400)
        try:
            result = view.rebuild(scope=scope, now=int(service.clock()), dry_run=dry_run,
                                  force=payload.get("force") is True)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": "rebuild_failed", "error_type": type(exc).__name__}, status_code=503)
        return JSONResponse({"ok": True, **result}, headers={"Cache-Control": "no-store"})

    return router


def _guard(request: Request) -> JSONResponse | None:
    client = getattr(request, "client", None)
    if str(getattr(client, "host", "") or "").strip().lower() not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
    return None


def _lifecycle_write_guard(request: Request) -> JSONResponse | None:
    denied = _guard(request)
    if denied is not None:
        return denied
    origin = request.headers.get("origin", "")
    if origin:
        source = urlsplit(origin)
        destination = urlsplit(str(request.url))
        if (source.scheme, source.netloc) != (destination.scheme, destination.netloc):
            return JSONResponse({"ok": False, "reason": "same_origin_required"}, status_code=403)
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        return JSONResponse({"ok": False, "reason": "json_content_type_required"}, status_code=415)
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
