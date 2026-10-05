"""QQ 短窗口回合调度边界。

入站 webhook 不应直接调用模型：第一条消息需要等短窗口结束，后续相邻消息还要
合并成一轮，同时图片输入必须跟着各自的消息保留下来。本模块只负责这个调度边界，
不认识 LLM、persona 或具体工具；真正的 Agent 处理器由调用方注入。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import re
import threading
import time
from typing import Any, Protocol, TYPE_CHECKING

from shinku.runtime_limits import RuntimeLimits

from .attention import AttentionBatch
from .napcat import NapCatVisualInput
from .visual_continuity import VISUAL_REACTION_WINDOW_SECONDS, is_short_visual_reaction

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免 host ↔ turns 循环导入
    from .host import NapCatHostResult

__all__ = ["QQTurn", "TurnDispatcher", "TurnDispatchReceipt", "NapCatTurnDispatcher"]


_VISUAL_FOLLOWUP_RE = re.compile(
    r"(?:(?:这|那|上一|前一|刚才(?:的)?)(?:一)?(?:张|幅)?(?:图片|照片|截图|图(?!书|谋|灵|案|鉴)|画面)"
    r"|(?:图片|照片|截图|画面|图)(?:里|中|上|内)"
    r"|(?:看|看看|看清|描述)(?:一下)?[，,\s]*(?:这|那)(?:是|个是|里面|里边))",
    re.IGNORECASE,
)


@dataclass
class _VisualContext:
    """A bounded, conversation-local visual reference for follow-up turns."""

    created_at: float
    source_message_id: str
    inputs: tuple[NapCatVisualInput, ...]
    remaining_turns: int
    source_sender_id: str
    followup_open: bool = True


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
    visual_context_inherited: bool = False
    visual_context_source_message_id: str = ""
    attention_mode: str = ""
    attention_signals: tuple[str, ...] = ()
    response_message_id: str = ""
    response_mention_ids: tuple[str, ...] | None = None

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
                        **self.visual_source(message_id, item),
                    }
                    for item in inputs
                ]
                for message_id, inputs in self.visual_inputs
            },
            "visual_context_inherited": self.visual_context_inherited,
            "visual_context_source_message_id": self.visual_context_source_message_id,
            "attention_mode": self.attention_mode,
            "attention_signals": list(self.attention_signals),
        }

    def visual_source(self, message_id: str, item: NapCatVisualInput) -> dict[str, str]:
        """Original attachment author, never the author of an inheriting question."""
        inherited = self.visual_context_inherited and message_id == self.visual_context_source_message_id
        source_id = item.source_message_id or ("" if item.reference_kind == "quoted" else message_id)
        source = next((m for m in self.messages if str(getattr(m, "message_id", "")) == source_id), None)
        # Missing quote provenance must remain unknown, not become the quoter.
        fallback = source if item.reference_kind != "quoted" else None
        return {
            "source_message_id": source_id,
            "source_sender_id": item.source_sender_id or str(getattr(fallback, "sender_id", "") or ""),
            "source_sender_label": item.source_sender_label or str(getattr(fallback, "sender_label", "") or ""),
            "reference_kind": "inherited" if inherited else item.reference_kind,
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
        visual_context_ttl_seconds: float | None = None,
        visual_followup_turns: int | None = None,
        clock: Callable[[], float] | None = None,
        async_handler: bool = False,
        max_workers: int = 4,
    ) -> None:
        if not callable(handler):
            raise TypeError("turn handler must be callable")
        if not callable(flush_batch):
            raise TypeError("flush_batch must be callable")
        self.handler = handler
        self.flush_batch = flush_batch
        limits = RuntimeLimits.from_environment()
        configured = limits.qq_batch_window_seconds if window_seconds is None else window_seconds
        self.window_seconds = max(0.0, float(configured))
        configured_ttl = (
            limits.qq_visual_context_ttl_seconds
            if visual_context_ttl_seconds is None
            else visual_context_ttl_seconds
        )
        self.visual_context_ttl_seconds = max(1.0, float(configured_ttl))
        configured_followups = (
            limits.qq_visual_followup_turns
            if visual_followup_turns is None
            else visual_followup_turns
        )
        self.visual_followup_turns = max(1, min(8, int(configured_followups)))
        self._clock = clock or time.monotonic
        self._timer_factory = timer_factory or threading.Timer
        self._lock = threading.RLock()
        self._timer: Any = None
        self._visuals: dict[tuple[str, str, str], tuple[NapCatVisualInput, ...]] = {}
        self._visual_context: dict[tuple[str, str, str], _VisualContext] = {}
        self._last_error = ""
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="shinku-qq-turn")
            if async_handler else None
        )
        self._conversation_futures: dict[tuple[str, str], Future[Any]] = {}
        self._pending_turn_counts: dict[tuple[str, str], int] = {}

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def submit(self, result: "NapCatHostResult") -> TurnDispatchReceipt:
        ingress = getattr(result, "ingress", None)
        message = getattr(ingress, "message", None)
        if ingress is None or message is None:
            return TurnDispatchReceipt(False, error="turn_result_missing_ingress")
        if not bool(getattr(ingress, "admitted", True)):
            return TurnDispatchReceipt(False, error="turn_result_not_admitted")

        message_id = str(getattr(message, "message_id", "") or "").strip()
        visual = getattr(result, "visual", None)
        inputs = tuple(
            replace(item, source_message_id=message_id,
                    source_sender_id=str(getattr(message, "sender_id", "") or ""),
                    source_sender_label=str(getattr(message, "sender_label", "") or ""))
            if item.reference_kind == "direct" else item
            for item in (getattr(visual, "inputs", ()) or ())
        )
        closed = getattr(ingress, "closed_batch", None)
        dispatched = 0
        with self._lock:
            now = self._now()
            self._prune_visual_context_locked(now)
            # The incoming event may close an older batch. Do not let that
            # batch see an image which arrived after its question.
            if closed is not None:
                dispatched += self._dispatch_locked(closed, now=now)
            scheduled = bool(getattr(result, "scheduled", False))
            visual_key = (*self._conversation_key(message), message_id)
            if message_id and inputs and scheduled:
                self._visuals[visual_key] = inputs
            direct_inputs = tuple(item for item in inputs if item.reference_kind == "direct")
            if direct_inputs and message_id:
                # A new own image supersedes older implicit topics. Older
                # sources remain available only through their exact QQ quote.
                for key, context in self._visual_context.items():
                    if key[:2] == visual_key[:2] and context.source_sender_id == str(getattr(message, "sender_id", "") or ""):
                        context.followup_open = False
                self._visual_context[visual_key] = _VisualContext(
                    created_at=now,
                    source_message_id=message_id,
                    inputs=direct_inputs,
                    remaining_turns=self.visual_followup_turns,
                    source_sender_id=str(getattr(message, "sender_id", "") or ""),
                )
                # Bound passive image retention even in busy groups.
                group_keys = [key for key in self._visual_context if key[:2] == visual_key[:2]]
                for key in group_keys[:-8]:
                    self._visual_context.pop(key, None)
            elif not inputs and not getattr(message, "reply_message_ids", ()):
                text = str(getattr(message, "text_without_at_mentions", getattr(message, "text", "")) or "").strip()
                if text:
                    # Even non-addressed topic changes close this author's
                    # implicit image thread; exact quotes remain available.
                    for key, context in self._visual_context.items():
                        if (key[:2] == visual_key[:2]
                            and context.source_sender_id == str(getattr(message, "sender_id", "") or "")
                            and not self._is_visual_followup(text, context, now=now)):
                            context.followup_open = False
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
            now = self._now()
            self._prune_visual_context_locked(now)
            dispatched = self._dispatch_locked(batch, now=now) if batch is not None else 0
            return TurnDispatchReceipt(
                True,
                dispatched=dispatched,
                pending=batch is None and bool(self._visuals),
                error=self._last_error,
            )

    def close(self) -> TurnDispatchReceipt:
        receipt = self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        return receipt

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

    def _dispatch_locked(self, batch: AttentionBatch | None, *, now: float | None = None) -> int:
        if batch is None:
            return 0
        turn = self._build_turn_locked(batch, now=self._now() if now is None else now)
        if self._executor is not None:
            key = (turn.conversation_type, turn.conversation_id)
            if turn.attention_mode == "autonomous" and (
                    self._pending_turn_counts.get(key, 0) >= 2 or sum(self._pending_turn_counts.values()) >= 24):
                self._last_error = "autonomous_queue_full"
                return 0
            previous = self._conversation_futures.get(key)
            self._pending_turn_counts[key] = self._pending_turn_counts.get(key, 0) + 1
            future = self._executor.submit(self._handle_after, previous, turn)
            self._conversation_futures[key] = future
            future.add_done_callback(lambda completed, key=key: self._forget_completed(key, completed))
            return 1
        try:
            self.handler(turn)
        except Exception as exc:
            self._last_error = f"handler_error:{type(exc).__name__}"
        return 1

    def _handle_after(self, previous: Future[Any] | None, turn: QQTurn) -> None:
        if previous is not None:
            try:
                previous.result()
            except Exception:
                pass
        try:
            self.handler(turn)
        except Exception as exc:
            with self._lock:
                self._last_error = f"handler_error:{type(exc).__name__}"

    def _forget_completed(self, key: tuple[str, str], completed: Future[Any]) -> None:
        with self._lock:
            self._pending_turn_counts[key] = max(0, self._pending_turn_counts.get(key, 1) - 1)
            if not self._pending_turn_counts[key]:
                self._pending_turn_counts.pop(key, None)
            if self._conversation_futures.get(key) is completed:
                self._conversation_futures.pop(key, None)

    def _build_turn_locked(self, batch: AttentionBatch, *, now: float) -> QQTurn:
        message_ids = tuple(str(getattr(item, "message_id", "") or "").strip() for item in batch.messages)
        direct_visuals = tuple(
            (message_id, self._visuals.pop((*self._conversation_key(message), message_id), ()))
            for message, message_id in zip(batch.messages, message_ids)
            if message_id
        )
        visuals = list(direct_visuals)
        inherited = False
        inherited_source = ""
        if not any(inputs for _, inputs in direct_visuals):
            inherited_inputs, inherited_source = self._inherit_visual_context_locked(batch, now=now)
            primary_id = str(getattr(batch.primary_message, "message_id", "") or "").strip()
            if inherited_inputs and primary_id:
                # Keep the source message key; do not reparent to the requester.
                self._visuals.pop((*self._conversation_key(batch.primary_message), inherited_source), None)
                visuals.append((inherited_source, inherited_inputs))
                inherited = True
        primary = batch.primary_message
        # A short-window batch can contain both an autonomous candidate and
        # a later explicit address.  Any immediate decision makes the whole
        # turn user-directed, so the model must not silently discard it.
        attention_mode = ""
        if batch.decisions:
            attention_mode = (
                "immediate"
                if any(decision.immediate for decision in batch.decisions)
                else "autonomous"
            )
        return QQTurn(
            conversation_id=str(batch.conversation_id or "").strip(),
            conversation_type=str(getattr(primary, "conversation_type", "unknown") or "unknown").strip(),
            primary_message_id=str(getattr(primary, "message_id", "") or "").strip(),
            message_ids=message_ids,
            combined_text=batch.combined_text,
            messages=tuple(batch.messages),
            visual_inputs=tuple(visuals),
            visual_context_inherited=inherited,
            visual_context_source_message_id=inherited_source,
            attention_mode=attention_mode,
            attention_signals=tuple(dict.fromkeys(signal for decision in batch.decisions for signal in decision.signals)),
        )

    def _inherit_visual_context_locked(
        self,
        batch: AttentionBatch,
        *,
        now: float,
    ) -> tuple[tuple[NapCatVisualInput, ...], str]:
        primary = batch.primary_message
        key = self._conversation_key(primary)
        sender_id = str(getattr(primary, "sender_id", "") or "").strip()
        contexts = [(context_key, context) for context_key, context in self._visual_context.items()
                    if context_key[:2] == key and 0 <= now - context.created_at <= self.visual_context_ttl_seconds
                    and context.remaining_turns > 0]
        reply_ids = tuple(getattr(primary, "reply_message_ids", ()) or ())
        context_entry = None
        if reply_ids:
            # Exact message references may cross authors, but never groups.
            context_entry = next((entry for entry in reversed(contexts)
                                  if entry[1].source_message_id in reply_ids), None)
        else:
            own = [entry for entry in contexts if sender_id and entry[1].source_sender_id == sender_id]
            # Only the primary author's body can establish image intent.
            text = str(getattr(primary, "text_without_at_mentions", getattr(primary, "text", "")) or "").strip()
            context_entry = next((entry for entry in reversed(own) if entry[1].followup_open), None)
            if context_entry is None or not self._is_visual_followup(text, context_entry[1], now=now):
                for _, context in own:
                    context.followup_open = False
                return (), ""
        if context_entry is None:
            return (), ""
        context_key, context = context_entry
        remaining = context.remaining_turns - 1
        if remaining <= 0:
            self._visual_context.pop(context_key, None)
        else:
            context.remaining_turns = remaining
        return context.inputs, context.source_message_id

    @staticmethod
    def _is_visual_followup(text: str, context: _VisualContext, *, now: float) -> bool:
        return bool(_VISUAL_FOLLOWUP_RE.search(text)) or (
            0 <= now - context.created_at <= VISUAL_REACTION_WINDOW_SECONDS
            and is_short_visual_reaction(text)
        )

    def _prune_visual_context_locked(self, now: float) -> None:
        expired = [
            key
            for key, context in self._visual_context.items()
            if now - context.created_at > self.visual_context_ttl_seconds
        ]
        for key in expired:
            context = self._visual_context.pop(key, None)
            if context is not None:
                self._visuals.pop(key, None)

    @staticmethod
    def _conversation_key(message: Any) -> tuple[str, str]:
        return (
            str(getattr(message, "conversation_type", "unknown") or "unknown").strip().lower(),
            str(getattr(message, "conversation_id", "") or "").strip(),
        )

    def _now(self) -> float:
        try:
            return float(self._clock())
        except Exception:
            return time.monotonic()
