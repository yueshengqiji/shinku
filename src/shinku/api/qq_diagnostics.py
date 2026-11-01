"""NapCat/QQ 的只读诊断入口。"""

from __future__ import annotations

from typing import Any
import threading
import time
from datetime import datetime, timezone
from ..qq.napcat import NapCatHttpActionCaller

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

__all__ = ["build_qq_diagnostics_router"]


def build_qq_diagnostics_router(*, napcat_host: Any = None, napcat_config: Any = None, assembly: Any = None) -> APIRouter:
    router = APIRouter()
    # 连接探测与纯装配自检分开；单实例锁与短缓存限制并发轮询，不调用模型或发送消息。
    cache = {}
    lock = threading.Lock()

    @router.get("/api/qq/connection", response_model=None)
    def connection(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        with lock:
            now = time.monotonic()
            if now >= cache.get("expires", 0):
                payload = {"ok": True, "status": "unconfigured"}
                if napcat_host is not None and napcat_config is not None and not napcat_config.validate():
                    payload["status"] = "unavailable"
                    try:
                        caller = NapCatHttpActionCaller(napcat_config.base_url,
                            access_token=napcat_config.token_value(), timeout=2)
                        status = caller(action="get_status", params={})
                        login = caller(action="get_login_info", params={})
                        valid = all(item.get("status") == "ok" and item.get("retcode") == 0
                                    and isinstance(item.get("data"), dict) for item in (status, login))
                        if valid:
                            account = str(login["data"].get("user_id") or "")
                            bots = getattr(getattr(assembly, "bridge", None), "bot_ids", ())
                            payload["status"] = ("account_mismatch" if not account or account not in bots
                                else "online" if status["data"].get("online") is True else "offline")
                    except Exception:
                        # 请求错误可能带 token/地址；前端只获得稳定状态，不返回任意异常文本。
                        pass
                payload["observed_at"] = datetime.now(timezone.utc).isoformat()
                cache.update(payload=payload, expires=time.monotonic() + 10)
            result = dict(cache["payload"])
        ingress = getattr(napcat_host, "ingress", None)
        result["received_events"] = len(getattr(ingress, "_seen_event_fingerprints", {}) or {})
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @router.get("/api/qq/napcat/status", response_model=None)
    def napcat_status(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        return JSONResponse(_snapshot(napcat_host, napcat_config), headers={"Cache-Control": "no-store"})

    @router.get("/api/qq/self-check", response_model=None)
    def self_check(request: Request) -> JSONResponse:
        if not _local_only(request):
            return _forbidden()
        snapshot = _snapshot(napcat_host, napcat_config)
        checks = snapshot["checks"]
        return JSONResponse(
            {"ok": all(item["ok"] for item in checks), "checks": checks, "note": "只检查本地装配状态，不主动访问 NapCat。"},
            status_code=200,
            headers={"Cache-Control": "no-store"},
        )

    return router


def _snapshot(host: Any, config: Any) -> dict[str, Any]:
    config_diag = _safe_call(getattr(config, "diagnostics", None), {})
    ingress = getattr(host, "ingress", None)
    policy = getattr(ingress, "ingress_policy", None)
    checks = [
        {"name": "host_attached", "ok": host is not None, "detail": "NapCatHost 已装配" if host is not None else "未装配"},
        {"name": "config_valid", "ok": bool(config is None or not _safe_call(getattr(config, "validate", None), ())), "detail": config_diag.get("errors", []) if isinstance(config_diag, dict) else []},
        {"name": "webhook_decoder", "ok": host is not None and getattr(host, "decoder", None) is not None, "detail": "decoder"},
        {"name": "ingress_guard", "ok": policy is not None, "detail": type(policy).__name__ if policy is not None else "missing"},
        {"name": "quote_fetcher", "ok": host is not None and callable(getattr(host, "quote_fetcher", None)), "detail": "get_msg boundary" if host is not None and callable(getattr(host, "quote_fetcher", None)) else "not configured"},
        {"name": "turn_dispatcher", "ok": host is not None and getattr(host, "turn_dispatcher", None) is not None, "detail": "dispatcher"},
    ]
    return {
        "ok": all(item["ok"] for item in checks),
        "configured": host is not None,
        "config": config_diag if isinstance(config_diag, dict) else {},
        "host": {
            "transport": type(getattr(host, "transport", None)).__name__ if host is not None else "unconfigured",
            "last_observer_error": str(getattr(host, "last_observer_error", "") or ""),
            "last_quote_error": str(getattr(host, "last_quote_error", "") or ""),
            "seen_event_count": len(getattr(ingress, "_seen_event_fingerprints", {}) or {}) if ingress is not None else 0,
        },
        "checks": checks,
    }


def _local_only(request: Request) -> bool:
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "").strip().lower() in {"127.0.0.1", "::1", "localhost", "testclient"}


def _forbidden() -> JSONResponse:
    return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default
