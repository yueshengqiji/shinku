"""模型服务控制中心路由。

这一层的活只有一件：把"配哪家的模型服务"这件只能进后台改的事搬到 HTTP 上，
并且**只让本机来按这些按钮**。四个路由围着一个注册表转：

- 整页快照（目录里每家的状态 + 当前激活的那家，**不含密钥**）
- 按供应商保存配置（可以顺手激活）
- 刷新某家的模型清单（模型名本来就是这一步要发现的，所以不要求先填模型）
- 切换当前激活项

**为什么不"顺便保留单服务端点"。** 旧的真红实现里有一组
``/control-center/model-service``（单数）端点，注释写的是给老版本控制台兼容。
真红自己的控制台走的是多供应商那组路径，全仓没有第二处调用单数端点——
唯一的消费方在另一个产品的前端。搬进来等于替别人的旧版本留一盏永远不灭的灯，
本批不留。

**依赖注入的边界：** `store`、`config_module`、`engine`、度量、日志、探测回调
是宿主提供的东西，走注入；其余（归一化、动词→设置、写回项目配置、错误脱敏、
JSON 信封）是本仓自己的模块，直接 import。分界线是"这个东西的形状是不是本仓定的"。

导入方向：`api` → `providers` → `contracts`。不反向依赖。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..contracts.http import json_response
from ..providers.catalog import (
    ModelServiceConfigStore,
    apply_model_service_settings,
    effective_settings_from_config,
    probe_model_ids,
    public_model_services_snapshot,
    settings_from_mapping,
)
from ..providers.service import normalize_model_ids, redact_provider_error

#: 只有本机呼得到写操作。`testclient` 是 Starlette 测试客户端的默认来源地址，
#: 少了它所有测试都会被闸门挡在外面。
_LOCAL_CALLER_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})

#: 表单里"要不要顺便激活"可以写成布尔，也可以写成这些词。
_AFFIRMATIVE_WORDS = frozenset({"1", "true", "yes", "on", "enabled"})

#: 还没存过配置就点"刷新模型列表"时的提示。
_UNSAVED_PROVIDER_HINT = "先保存该供应方配置，再刷新模型列表。"

#: 度量名。改动会打断既有监控面板的名字。
_METRIC_PAGE = "model_services.read"
_METRIC_CONFIG = "model_services.save"
_METRIC_MODELS = "model_services.models"
_METRIC_SELECT = "model_services.select"

#: 结构化日志的事件名。
_EVENT_CONFIG = "model_services_save"
_EVENT_MODELS = "model_services_models"
_EVENT_SELECT = "model_services_select"


class ModelServiceEngine(Protocol):
    """自动装载这一侧的移交点：保存之后要有人把新配置灌进运行时。"""

    def reload_model_services(self) -> dict[str, Any]: ...


class RequestMetrics(Protocol):
    """请求度量记录面。"""

    def observe_request(self, name: str, *, duration_ms: float, ok: bool) -> None: ...


#: 结构化日志。缺省就是"没人监听"。
LogEvent = Callable[..., Any]

#: 模型清单探测。返回一串模型 id；重整由 `normalize_model_ids` 负责。
ModelProbe = Callable[[Any], list[str]]


def _caller_is_local(request: Request) -> bool:
    """这条请求是不是从本机来的。取不到来源一律当"不是"。"""

    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    return host in _LOCAL_CALLER_HOSTS


async def _json_body(request: Request) -> dict[str, Any]:
    """请求体必须是 JSON 对象——不是对象、解析失败，一律折成空载荷。"""

    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _emit_metric(
    metrics: RequestMetrics | None,
    name: str,
    started_at: float,
    ok: bool,
) -> None:
    """记一条度量。**观测设施本身出故障不能带崩业务。**"""

    if metrics is None or not hasattr(metrics, "observe_request"):
        return
    try:
        metrics.observe_request(
            name,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
            ok=ok,
        )
    except Exception:
        return


def _emit_log(log_event: LogEvent | None, event: str, **fields: Any) -> None:
    """写一条结构化日志。同上，写崩了就当没人监听。"""

    if log_event is None:
        return
    try:
        log_event(event, **fields)
    except Exception:
        return


def _truthy(value: Any) -> bool:
    """把表单里的"要不要"折成布尔。布尔原样返回，其余按真值词表认。"""

    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _AFFIRMATIVE_WORDS


def _remote_caller() -> JSONResponse:
    """非本机来源的统一拒绝响应。四个写路由共用这套信封。"""

    return json_response(
        {"ok": False, "status": "forbidden", "reason": "local_request_required"},
        status_code=403,
    )


def _stored_api_key_for(
    store: ModelServiceConfigStore,
    config_module: Any,
    provider_id: str,
) -> str:
    """这家供应商现在用着的密钥。

    表单通常不带密钥，密钥要从已经存下的那份继承；**换供应商时不继承**——
    否则会把 A 家的密钥糊到 B 家身上。注册表读不出东西（损坏或还没存过）
    当作"没存过"，再退回环境变量推导出来的当前配置。

    环境变量推导出来的那一份有个前提：它得**正好就是**这家供应商，否则也是空串。
    """

    try:
        saved = store.get_provider(provider_id)
    except Exception:
        saved = None
    if saved is not None:
        return saved.api_key
    effective = effective_settings_from_config(config_module)
    requested = str(provider_id or "").strip().lower()
    return effective.api_key if effective.provider_id == requested else ""


def _picked(values: Mapping[str, Any], *keys: str) -> str:
    """按顺序取第一个说得通的键。几个客户端版本各有自己的写法。"""

    for key in keys:
        raw = values.get(key)
        if raw is not None:
            return str(raw).strip()
    return ""


def build_model_services_router(
    *,
    store: ModelServiceConfigStore,
    config_module: Any,
    engine: ModelServiceEngine,
    runtime_metrics: RequestMetrics | None = None,
    log_event: LogEvent | None = None,
    model_probe: ModelProbe = probe_model_ids,
) -> APIRouter:
    """装配模型服务控制中心那四个路由。"""

    router = APIRouter()

    @router.get("/control-center/model-services")
    async def read_model_services_page() -> JSONResponse:
        started_at = time.perf_counter()
        try:
            page = public_model_services_snapshot(store, config_module)
        except Exception as exc:
            _emit_metric(runtime_metrics, _METRIC_PAGE, started_at, False)
            return json_response(
                {
                    "ok": False,
                    "status": "invalid_config",
                    "reason": redact_provider_error(exc),
                },
                status_code=500,
            )
        _emit_metric(runtime_metrics, _METRIC_PAGE, started_at, True)
        return json_response(page)

    @router.post("/control-center/model-services/{provider_id}/config")
    async def save_provider_config(provider_id: str, request: Request) -> JSONResponse:
        started_at = time.perf_counter()
        if not _caller_is_local(request):
            _emit_metric(runtime_metrics, _METRIC_CONFIG, started_at, False)
            return _remote_caller()

        payload = await _json_body(request)
        # 密钥要在"把供应商覆盖进载荷"之前取，否则换供应商就串味了。
        existing = _stored_api_key_for(store, config_module, provider_id)
        payload["providerId"] = provider_id
        reload_result: Any = None
        try:
            settings = settings_from_mapping(
                payload,
                existing_api_key=existing,
                require_model=False,
            )
            activate = _truthy(payload.get("activate", payload.get("setActive", False)))
            if activate and not settings.chat_model:
                raise ValueError("model_service_model_missing")
            store.save_provider(settings, set_active=activate)
            if activate:
                apply_model_service_settings(config_module, settings)
                reload_result = await asyncio.to_thread(engine.reload_model_services)
        except Exception as exc:
            _emit_metric(runtime_metrics, _METRIC_CONFIG, started_at, False)
            _emit_log(
                log_event,
                _EVENT_CONFIG,
                status="failed",
                provider_id=provider_id,
            )
            return json_response(
                {
                    "ok": False,
                    "status": "invalid_config",
                    "reason": redact_provider_error(exc, api_key=existing),
                }
            )

        _emit_metric(runtime_metrics, _METRIC_CONFIG, started_at, True)
        _emit_log(
            log_event,
            _EVENT_CONFIG,
            status="saved",
            provider_id=settings.provider_id,
            protocol=settings.protocol,
            chat_model=settings.chat_model,
            active=activate,
        )
        return json_response(
            {
                **public_model_services_snapshot(store, config_module),
                "refresh": True,
                "runtime": reload_result,
            }
        )

    @router.post("/control-center/model-services/{provider_id}/models")
    async def refresh_provider_models(provider_id: str, request: Request) -> JSONResponse:
        started_at = time.perf_counter()
        if not _caller_is_local(request):
            _emit_metric(runtime_metrics, _METRIC_MODELS, started_at, False)
            return _remote_caller()

        payload = await _json_body(request)
        try:
            stored = store.get_provider(provider_id)
        except Exception as exc:
            _emit_metric(runtime_metrics, _METRIC_MODELS, started_at, False)
            return json_response(
                {
                    "ok": False,
                    "status": "invalid_config",
                    "reason": redact_provider_error(exc),
                }
            )
        if stored is None:
            _emit_metric(runtime_metrics, _METRIC_MODELS, started_at, False)
            return json_response(
                {
                    "ok": False,
                    "status": "provider_not_configured",
                    "reason": _UNSAVED_PROVIDER_HINT,
                }
            )

        candidate: Any = None
        try:
            # 允许控制中心带着还没落盘的字段来刷新，缺口用已存的那份补齐；
            # 密钥永远取已存的——这一步不改变"在用哪家服务"。
            payload.setdefault("protocol", stored.protocol)
            payload.setdefault("baseUrl", stored.base_url)
            payload.setdefault("chatModel", stored.chat_model)
            payload.setdefault("useForVision", stored.use_for_vision)
            payload.setdefault("visionModel", stored.vision_model)
            payload.setdefault("timeoutSeconds", stored.timeout_seconds)
            payload["providerId"] = provider_id
            candidate = settings_from_mapping(
                payload,
                existing_api_key=stored.api_key,
                require_model=False,
            )
            # 探测只需要端点加密钥；模型名本来就是这一步要找出来的东西。
            if not candidate.endpoint_configured:
                raise ValueError("model_service_config_incomplete")
            models = await asyncio.to_thread(model_probe, candidate)
            normalized = normalize_model_ids(models)
            now = int(time.time())
            store.save_model_probe(candidate.provider_id, normalized, timestamp=now)
        except Exception as exc:
            error = redact_provider_error(exc, api_key=stored.api_key)
            # 失败也要落一次探测记录，控制台得看得到"上次探测黄了"。
            try:
                store.save_model_probe(
                    stored.provider_id,
                    [],
                    error=error,
                    timestamp=int(time.time()),
                )
            except Exception:
                pass
            _emit_metric(runtime_metrics, _METRIC_MODELS, started_at, False)
            _emit_log(
                log_event,
                _EVENT_MODELS,
                status="failed",
                provider_id=provider_id,
            )
            return json_response(
                {"ok": False, "status": "request_failed", "reason": error}
            )

        _emit_metric(runtime_metrics, _METRIC_MODELS, started_at, True)
        _emit_log(
            log_event,
            _EVENT_MODELS,
            status="ok",
            provider_id=candidate.provider_id,
            count=len(normalized),
        )
        return json_response(
            {
                "ok": True,
                "status": "available",
                "providerId": candidate.provider_id,
                "models": normalized,
                "count": len(normalized),
                "lastModelProbeAt": now,
            }
        )

    @router.post("/control-center/model-services/select")
    async def select_model_service(request: Request) -> JSONResponse:
        started_at = time.perf_counter()
        if not _caller_is_local(request):
            _emit_metric(runtime_metrics, _METRIC_SELECT, started_at, False)
            return _remote_caller()

        payload = await _json_body(request)
        provider_id = _picked(payload, "providerId", "provider_id")
        model_id = _picked(payload, "modelId", "model_id", "chatModel", "model")
        try:
            settings = store.activate(provider_id, model_id)
            apply_model_service_settings(config_module, settings)
            reload_result = await asyncio.to_thread(engine.reload_model_services)
        except Exception as exc:
            _emit_metric(runtime_metrics, _METRIC_SELECT, started_at, False)
            _emit_log(
                log_event,
                _EVENT_SELECT,
                status="failed",
                provider_id=provider_id,
            )
            return json_response(
                {
                    "ok": False,
                    "status": "select_failed",
                    "reason": redact_provider_error(exc),
                }
            )

        _emit_metric(runtime_metrics, _METRIC_SELECT, started_at, True)
        _emit_log(
            log_event,
            _EVENT_SELECT,
            status="selected",
            provider_id=settings.provider_id,
            chat_model=settings.chat_model,
        )
        return json_response(
            {
                **public_model_services_snapshot(store, config_module),
                "activeProviderId": settings.provider_id,
                "activeModel": settings.chat_model,
                "runtime": reload_result or {"status": "reloaded"},
            }
        )

    return router


__all__ = ["build_model_services_router"]
