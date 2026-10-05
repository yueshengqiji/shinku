"""Optional per-QQ-turn correlation for provider-reported token usage.

The LLM runtime remains usable outside QQ. When a QQ turn binds a callback,
usage reports are attached to that turn's existing diagnostic trace.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

__all__ = ["bind_qq_turn_usage", "report_qq_turn_usage"]


_ContextValue = tuple[str, Callable[[Mapping[str, Any]], None]]
_current_context: ContextVar[_ContextValue | None] = ContextVar(
    "shinku.qq_turn_usage_context",
    default=None,
)


@contextmanager
def bind_qq_turn_usage(
    trace_id: str,
    callback: Callable[[Mapping[str, Any]], None],
) -> Iterator[None]:
    """Bind a content-free usage callback for one synchronous QQ turn."""

    identifier = str(trace_id or "").strip()
    if not identifier or not callable(callback):
        yield
        return
    token: Token[_ContextValue | None] = _current_context.set((identifier, callback))
    try:
        yield
    finally:
        _current_context.reset(token)


def report_qq_turn_usage(**values: Any) -> None:
    """Send provider usage to the current QQ trace, if this is a QQ call."""

    current = _current_context.get()
    if current is None:
        return
    trace_id, callback = current
    payload = {"trace_id": trace_id, **values}
    try:
        callback(payload)
    except Exception:
        # Diagnostics must not turn a model response into a failed turn.
        return
