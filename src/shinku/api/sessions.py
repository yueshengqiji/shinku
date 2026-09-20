"""会话 HTTP 路由。

会话是"某个人 + 某个角色包 + 某段时间的对话"的载体。这一层只做三件事：
按身份定位会话、把会话及其周边（列表、消息、最近一次评测结果）打包返回、以及改名。

**依赖全部注入。** 这一层不导入存储实现、不导入度量实现、不导入错误信封的构造器——
它只声明自己真正用到的方法。这样做的代价是工厂签名变长，收益是这一层可以脱离
整个后端被单独测试，也避免了为了写一个路由而先迁移整棵存储。

被注入的三样各有来源：

- 会话读取面：旧实现的存储对象（按 C5 迁移）
- ``normalize_pack_id``：角色包标识的归一化规则（属存储模块）
- ``build_error_payload``：错误信封的构造器（属宿主契约模块）
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from shinku.runtime_limits import RuntimeLimits

#: 会话列表与消息列表的返回上限。
SESSION_LIST_LIMIT = 50
MESSAGE_LIMIT = 120


@dataclass(frozen=True)
class _SessionLimits:
    list_limit: int = SESSION_LIST_LIMIT
    message_limit: int = MESSAGE_LIMIT

#: 非法 JSON 时回显的原因截断长度——不要把这个当成错误详情通道。
_REASON_CHARS = 160

#: 角色包标识的键序。三个键名来自不同客户端版本，顺序不可调整。
_PACK_ID_KEYS = ("character_pack_id", "characterPackId", "character_pack")

#: 请求体里还可能夹带当前视觉状态，角色包标识要往里再看一眼。
_VISUAL_KEY = "current_visual"

#: 度量名。改动会打断既有的监控面板。
_METRIC_LIST = "sessions_list"
_METRIC_ENSURE = "sessions_ensure"
_METRIC_RENAME = "sessions_rename"


class SessionStore(Protocol):
    """会话存储的读取面——本路由只用到这些方法。"""

    def ensure_session(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        character_pack_id: str = "",
        display_title: str | None = None,
    ) -> Any: ...

    def get_session(self, profile_user_id: str, session_id: str) -> Any | None: ...

    def get_character_session(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        character_pack_id: str = "",
    ) -> Any | None: ...

    def list_sessions(
        self,
        profile_user_id: str,
        limit: int = SESSION_LIST_LIMIT,
        character_pack_id: str | None = None,
    ) -> list[Any]: ...

    def rename_session(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        display_title: str,
    ) -> Any | None: ...

    def get_session_messages(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        character_pack_id: str | None = None,
        limit: int = MESSAGE_LIMIT,
    ) -> list[Any]: ...

    def get_latest_eval_turn_for_session(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        character_pack_id: str | None = None,
    ) -> dict[str, Any] | None: ...


class MetricsRecorder(Protocol):
    """请求度量记录面。"""

    def observe_request(self, name: str, *, duration_ms: float, ok: bool) -> None: ...


#: 从请求或请求体里解析出 ``(session_id, profile_user_id)``——顺序不可交换。
IdentityResolver = Callable[[Request | Mapping[str, Any]], "tuple[str, str]"]

#: 角色包标识的归一化。形状不合规时返回空串。
PackIdNormalizer = Callable[[Any], str]

#: 错误信封构造器。形状见 `docs/contracts/c1_2_shared_modules.md` §3.7。
ErrorPayloadBuilder = Callable[..., "dict[str, Any]"]

#: 结构化日志。
LogEvent = Callable[..., None]


def _latest_final_json(
    sessions: SessionStore,
    profile_user_id: str,
    session_id: str,
    character_pack_id: str | None,
) -> dict[str, Any] | None:
    """最近一次评测回合里的 ``final_json``；没有或形状不对则为 ``None``。"""

    turn = sessions.get_latest_eval_turn_for_session(
        profile_user_id=profile_user_id,
        session_id=session_id,
        character_pack_id=character_pack_id,
    )
    value = turn.get("final_json") if turn else None
    return value if isinstance(value, dict) else None


def build_sessions_router(
    *,
    sessions: SessionStore,
    metrics: MetricsRecorder,
    log_event: LogEvent,
    resolve_identity: IdentityResolver,
    normalize_pack_id: PackIdNormalizer,
    build_error_payload: ErrorPayloadBuilder,
    limits: RuntimeLimits | None = None,
) -> APIRouter:
    """装配会话路由。"""

    router = APIRouter()
    runtime_limits = limits or RuntimeLimits.from_environment()
    session_limits = _SessionLimits(
        list_limit=runtime_limits.session_list_limit,
        message_limit=runtime_limits.session_message_limit,
    )

    def record(name: str, started: float, ok: bool) -> None:
        metrics.observe_request(name, duration_ms=(time.perf_counter() - started) * 1000, ok=ok)

    def bad_request(error: str, message: str) -> JSONResponse:
        """构造 400 响应。错误信封必须自带 ``no-store``——这类响应不该被缓存。"""

        return JSONResponse(
            build_error_payload(error=error, message=message, retryable=False),
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )

    def pack_from_mapping(payload: Mapping[str, Any]) -> str | None:
        """从映射里取角色包标识：先看顶层，再看不顶层就没有的嵌套视觉状态。"""

        for key in _PACK_ID_KEYS:
            if key in payload:
                return normalize_pack_id(payload.get(key))
        visual = payload.get(_VISUAL_KEY)
        if isinstance(visual, dict):
            for key in _PACK_ID_KEYS:
                if key in visual:
                    return normalize_pack_id(visual.get(key))
        return None

    def pack_from_query(request: Request) -> str | None:
        """从查询参数里取角色包标识。查询参数不做嵌套查找。"""

        for key in _PACK_ID_KEYS:
            if key in request.query_params:
                return normalize_pack_id(request.query_params.get(key))
        return None

    def session_state(
        *,
        profile_user_id: str,
        session_id: str,
        character_pack_id: str | None,
        ensure: bool,
        title: str | None = None,
    ) -> dict[str, Any]:
        """一次会话的完整视图。会话不存在且不要求创建时抛 404。"""

        if ensure:
            session = sessions.ensure_session(
                profile_user_id=profile_user_id,
                session_id=session_id,
                character_pack_id=character_pack_id or "",
                display_title=title,
            )
        elif character_pack_id is not None:
            session = sessions.get_character_session(
                profile_user_id=profile_user_id,
                session_id=session_id,
                character_pack_id=character_pack_id,
            )
        else:
            session = sessions.get_session(profile_user_id, session_id)

        if session is None:
            raise HTTPException(status_code=404, detail="session not found")

        return {
            "session": session,
            "sessions": sessions.list_sessions(
                profile_user_id=profile_user_id,
                limit=session_limits.list_limit,
                character_pack_id=character_pack_id,
            ),
            "messages": sessions.get_session_messages(
                profile_user_id=profile_user_id,
                session_id=session_id,
                character_pack_id=character_pack_id,
                limit=session_limits.message_limit,
            ),
            "latest_final_json": _latest_final_json(
                sessions, profile_user_id, session_id, character_pack_id
            ),
        }

    @router.get("/sessions")
    async def list_sessions_endpoint(request: Request) -> JSONResponse:
        started = time.perf_counter()
        session_id, profile_user_id = resolve_identity(request)
        pack_id = pack_from_query(request)
        try:
            listed = sessions.list_sessions(
                profile_user_id=profile_user_id,
                limit=session_limits.list_limit,
                character_pack_id=pack_id,
            )
        except Exception as exc:
            record(_METRIC_LIST, started, False)
            log_event(
                "sessions_list_error",
                session_id=session_id,
                profile_user_id=profile_user_id,
                message=str(exc),
            )
            # 存储故障是后端自己的问题，不能在这里被降级成空列表——原样抛出。
            raise
        record(_METRIC_LIST, started, True)
        return JSONResponse({"sessions": listed, "current_session_id": session_id})

    @router.post("/sessions/ensure")
    async def ensure_session_endpoint(request: Request) -> JSONResponse:
        started = time.perf_counter()
        try:
            payload = await request.json()
        except Exception as exc:
            record(_METRIC_ENSURE, started, False)
            return bad_request("invalid_json", f"无法读取会话请求：{str(exc)[:_REASON_CHARS]}")
        if not isinstance(payload, dict):
            record(_METRIC_ENSURE, started, False)
            return bad_request(
                "invalid_payload", "/sessions/ensure payload must be a JSON object"
            )

        session_id, profile_user_id = resolve_identity(payload)
        try:
            state = session_state(
                profile_user_id=profile_user_id,
                session_id=session_id,
                character_pack_id=pack_from_mapping(payload),
                ensure=True,
                title=str(payload.get("display_title") or "").strip() or None,
            )
        except Exception as exc:
            record(_METRIC_ENSURE, started, False)
            log_event(
                "sessions_ensure_error",
                session_id=session_id,
                profile_user_id=profile_user_id,
                message=str(exc),
            )
            raise
        record(_METRIC_ENSURE, started, True)
        return JSONResponse(state)

    @router.post("/sessions/rename")
    async def rename_session_endpoint(request: Request) -> JSONResponse:
        started = time.perf_counter()
        # 这里不接非法 JSON：改名是显式动作，请求体读不出来就是调用方的问题。
        payload = await request.json()
        session_id, profile_user_id = resolve_identity(payload)
        pack_id = pack_from_mapping(payload)
        title = str(payload.get("display_title") or "").strip()
        if not title:
            record(_METRIC_RENAME, started, False)
            raise HTTPException(status_code=400, detail="display_title is required")

        renamed = sessions.rename_session(
            profile_user_id=profile_user_id,
            session_id=session_id,
            display_title=title,
        )
        if renamed is None:
            record(_METRIC_RENAME, started, False)
            raise HTTPException(status_code=404, detail="session not found")

        record(_METRIC_RENAME, started, True)
        return JSONResponse(
            {
                "session": renamed,
                "sessions": sessions.list_sessions(
                    profile_user_id=profile_user_id,
                    limit=session_limits.list_limit,
                    character_pack_id=pack_id,
                ),
            }
        )

    return router


__all__ = ["build_sessions_router"]
