"""跨服务请求关联 ID。

一次调用会穿过后端与多个回环宿主，日志要能把它们串成一条链，靠的就是这个 ID。
两条约束决定了下边的写法：

1. **头名是线路协议**，所以它不在本模块定义——唯一定义处在 :mod:`shinku.names`；
2. **调用方给的值不可信**。只有形状合规的值才被采纳，其余一律换成新生成的值——
   于是日志里不会出现调用方随手塞进来的内容。
"""

from __future__ import annotations

import contextlib
import contextvars
import string
import uuid
from collections.abc import Iterator
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware

from ..names import CORRELATION_ID_HEADER

#: 采纳的调用方 ID 长度区间（闭区间，按字符计）。
MIN_ID_CHARS = 8
MAX_ID_CHARS = 128

#: 采纳的字符集合 = 字母 + 数字 + 四种分隔符。
ID_CHARACTERS = frozenset(string.ascii_letters + string.digits + "._:-")

_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "shinku.correlation_id",
    default="",
)


def _new_id() -> str:
    """新造一个关联 ID：32 位、小写十六进制、无连字符。"""

    return uuid.uuid4().hex


def _is_adoptable(candidate: str) -> bool:
    """长度落在闭区间内，且每个字符都在允许集合里。"""

    if not MIN_ID_CHARS <= len(candidate) <= MAX_ID_CHARS:
        return False
    return ID_CHARACTERS.issuperset(candidate)


def normalize_correlation_id(value: Any) -> str:
    """采纳调用方给的关联 ID；不合形状（含未提供）就新造一个。"""

    candidate = str(value or "").strip()
    return candidate if _is_adoptable(candidate) else _new_id()


def current_correlation_id() -> str:
    """本请求的关联 ID；无请求上下文时为空串。"""

    return _context.get("")


def correlation_headers() -> dict[str, str]:
    """本次出站调用应当带上的关联头。上下文为空则现取一个新值。"""

    return {CORRELATION_ID_HEADER: current_correlation_id() or _new_id()}


@contextlib.contextmanager
def _binding(correlation_id: str) -> Iterator[None]:
    """在当前上下文里绑定关联 ID，离开作用域时无条件恢复。

    单独抽出来是为了让"绑定范围"和"响应处理"在中间件里各自成段——
    绑定必须在最内层结清，不能靠在末尾补一句 reset 来保证。
    """

    token = _context.set(correlation_id)
    try:
        yield
    finally:
        _context.reset(token)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """每次请求都绑定关联 ID；同一个值在响应上回显。

    处理顺序：读头 → 归一化 → 绑定上下文 → 写 ``request.state.correlation_id``
    → 调用下游 → 回显 → 无条件解绑。

    下游抛异常时不设置响应头，但上下文仍会被恢复——这正是 :func:`_binding`
    用 ``with`` 而不是手写 ``try/finally`` 的原因：绑定范围由语法保证。
    """

    async def dispatch(self, request, call_next):  # type: ignore[override]
        correlation_id = normalize_correlation_id(request.headers.get(CORRELATION_ID_HEADER))
        request.state.correlation_id = correlation_id
        with _binding(correlation_id):
            response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response


__all__ = [
    "CorrelationIdMiddleware",
    "correlation_headers",
    "current_correlation_id",
    "normalize_correlation_id",
]
