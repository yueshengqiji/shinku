"""QQ 短窗口回合调度边界。

入站 webhook 不应直接调用模型：第一条消息需要等短窗口结束，后续相邻消息还要
合并成一轮，同时图片输入必须跟着各自的消息保留下来。本模块只负责这个调度边界，
不认识 LLM、persona 或具体工具；真正的 Agent 处理器由调用方注入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import Any, Protocol, TYPE_CHECKING

from shinku.runtime_limits import RuntimeLimits

from .attention import AttentionBatch
from .napcat import NapCatVisualInput

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免 host ↔ turns 循环导入
    from .host import NapCatHostResult

__all__ = ["QQTurn", "TurnDispatcher", "TurnDispatchReceipt", "NapCatTurnDispatcher"]


@dataclass(frozen=True)
class QQTurn:
    """已经结束合并窗口、可以交给 Agent 的一轮 QQ 输入。"""

    conversation_id: str
    conversation_type: str
    primary_message_id: str
    message_ids: tuple[str, ...]
    combined_text: str
    messages: tuple[Any, ...]
    visual_inputs: tuple[tuple[str, tuple[NapCatVisualInput, ...]], ...] = ()

    def visuals_for(self, message_id: str) -> tuple[NapCatVisualInput, ...]:
        key = str(message_id or "").strip()
        for current_id, inputs in self.visual_inputs:
            if current_id == key:
                return inputs
        return ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type,
            "primary_message_id": self.primary_message_id,
            "message_ids": list(self.message_ids),
            "combined_text": self.combined_text,
            "visual_inputs": {
                message_id: [
                    {
                        "source_kind": item.source_kind,
                        "reference": item.reference,
                        "ready": item.ready,
                    }
                    for item in inputs
                ]
                for message_id, inputs in self.visual_inputs
            },
        }


@dataclass(frozen=True)
class TurnDispatchReceipt:
    """一次入站提交的可观察调度结果。"""

    accepted: bool
    dispatched: int = 0
    pending: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "dispatched": self.dispatched,
            "pending": self.pending,
            "error": self.error,
        }


class TurnDispatcher(Protocol):
    def submit(self, result: "NapCatHostResult") -> TurnDispatchReceipt: ...

    def flush(self) -> TurnDispatchReceipt: ...


class NapCatTurnDispatcher:
    """把 ``NapCatHostResult`` 变成可注入 Agent handler 的短窗口回合。"""

    def __init__(
        self,
        *,
        handler: Callable[[QQTurn], Any],
        flush_batch: Callable[[], AttentionBatch | None],
        window_seconds: float | None = None,
        timer_factory: Callable[[float, Callable[[], None]], Any] | None = None,
    ) -> None:
        if not callable(handler):
            raise TypeError("turn handler must be callable")
        if not callable(flush_batch):
            raise TypeError("flush_batch must be callable")
        self.handler = handler
        self.flush_batch = flush_batch
        configured = (
            RuntimeLimits.from_environment().qq_batch_window_seconds
            if window_seconds is None
            else window_seconds
        )
        self.window_seconds = max(0.0, float(configured))
        self._timer_factory = timer_factory or threading.Timer
        self._lock = threading.RLock()
        self._timer: Any = None
        self._visuals: dict[str, tuple[NapCatVisualInput, ...]] = {}
        self._last_error = ""

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def submit(self, result: "NapCatHostResult") -> TurnDispatchReceipt:
        ingress = getattr(result, "ingress", None)
        message = getattr(ingress, "message", None)
        if ingress is None or message is None:
            return TurnDispatchReceipt(False, error="turn_result_missing_ingress")

        message_id = str(getattr(message, "message_id", "") or "").strip()
        visual = getattr(result, "visual", None)
        inputs = tuple(getattr(visual, "inputs", ()) or ())
        closed = getattr(ingress, "closed_batch", None)
        dispatched = 0
        with self._lock:
            if message_id:
                self._visuals[message_id] = inputs
            if closed is not None:
                dispatched += self._dispatch_locked(closed)
            scheduled = bool(getattr(result, "scheduled", False))
            if scheduled:
                self._arm_timer_locked()
            return TurnDispatchReceipt(
                True,
                dispatched=dispatched,
                pending=scheduled,
                error=self._last_error,
            )

    def flush(self) -> TurnDispatchReceipt:
        with self._lock:
            self._cancel_timer_locked()
            batch = self.flush_batch()
            dispatched = self._dispatch_locked(batch) if batch is not None else 0
            return TurnDispatchReceipt(
                True,
                dispatched=dispatched,
                pending=batch is None and bool(self._visuals),
                error=self._last_error,
            )

    def close(self) -> TurnDispatchReceipt:
        return self.flush()

    def _arm_timer_locked(self) -> None:
        self._cancel_timer_locked()
        timer = self._timer_factory(self.window_seconds, self._timer_callback)
        self._timer = timer
        start = getattr(timer, "start", None)
        if callable(start):
            start()

    def _cancel_timer_locked(self) -> None:
        timer, self._timer = self._timer, None
        cancel = getattr(timer, "cancel", None)
        if callable(cancel):
            cancel()

    def _timer_callback(self) -> None:
        try:
            self.flush()
        except Exception as exc:  # pragma: no cover - 防护线程不向 timer 泄漏异常
            with self._lock:
                self._last_error = f"flush_error:{type(exc).__name__}"

    def _dispatch_locked(self, batch: AttentionBatch | None) -> int:
        if batch is None:
            return 0
        turn = self._build_turn_locked(batch)
        try:
            self.handler(turn)
        except Exception as exc:
            self._last_error = f"handler_error:{type(exc).__name__}"
        return 1

    def _build_turn_locked(self, batch: AttentionBatch) -> QQTurn:
        message_ids = tuple(str(getattr(item, "message_id", "") or "").strip() for item in batch.messages)
        visuals = tuple(
            (message_id, self._visuals.pop(message_id, ()))
            for message_id in message_ids
            if message_id
        )
        primary = batch.primary_message
        return QQTurn(
            conversation_id=str(batch.conversation_id or "").strip(),
            conversation_type=str(getattr(primary, "conversation_type", "unknown") or "unknown").strip(),
            primary_message_id=str(getattr(primary, "message_id", "") or "").strip(),
            message_ids=message_ids,
            combined_text=batch.combined_text,
            messages=tuple(batch.messages),
            visual_inputs=visuals,
        )
