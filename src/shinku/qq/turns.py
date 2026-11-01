"""QQ 短窗口回合调度边界。

入站 webhook 不应直接调用模型：第一条消息需要等短窗口结束，后续相邻消息还要
合并成一轮，同时图片输入必须跟着各自的消息保留下来。本模块只负责这个调度边界，
不认识 LLM、persona 或具体工具；真正的 Agent 处理器由调用方注入。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import re
import hashlib
import json
import threading
import time
from typing import Any, Protocol, TYPE_CHECKING

from shinku.runtime_limits import RuntimeLimits

from .attention import AttentionBatch
from .napcat import NapCatVisualInput
from .visual_selection import VisualResolution
from .visual_continuity import (
    VISUAL_QUESTION_WINDOW_SECONDS, VISUAL_REACTION_WINDOW_SECONDS,
    is_short_visual_question, is_short_visual_reaction,
)

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免 host ↔ turns 循环导入
    from .host import NapCatHostResult
    from .group_participation import GroupContinuation

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
    superseded: bool = False
    message: Any = None
    version: str = ""
    content_hashes: tuple[str, ...] = ()
    sequence: int = 0
    cached_at: float = 0.0


@dataclass(frozen=True)
class _AmbientMessage:
    """后续审查的短快照；不延长 raw_event、附件或媒体字节的生命周期。"""

    message_id: str
    sender_id: str
    text: str
    at_target_ids: tuple[str, ...]
    reply_message_ids: tuple[str, ...]
    has_media: bool
    unreadable: bool


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
    autonomous_guard: Callable[[str], str] | None = field(default=None, repr=False, compare=False)
    visual_candidates: tuple[dict[str, Any], ...] = ()
    visual_history: tuple[dict[str, Any], ...] = ()
    visual_resolver: Callable[..., VisualResolution] | None = field(default=None, repr=False, compare=False)
    # 仅宿主桌面适配器提供；随本轮传递，绝不替换并发 QQ 请求的全局发送器。
    local_delivery: Callable[..., Any] | None = field(default=None, repr=False, compare=False)
    # 调度器持有有界入站窗口；既有群聊审查读取/确认它，发送出口重新核对，模型不能改游标。
    ambient_updates: Callable[[], tuple[Any, ...]] | None = field(default=None, repr=False, compare=False)
    ambient_reviewed: Callable[[tuple[str, ...]], bool] | None = field(default=None, repr=False, compare=False)
    group_continuation: GroupContinuation | None = field(default=None, repr=False, compare=False)
    # 仅既有归属审查接受的聊天草稿设置；社交时效检查不能取消独立任务/工具结果的送达。
    ambient_reply_reviewed: bool = field(default=False, repr=False, compare=False)
    # 仅宿主实际读取产生的来源/版本，随本轮送达写入既有消息metadata；不包含正文。
    file_input_sources: tuple[dict[str, Any], ...] = field(default=(), repr=False, compare=False)

    def suppression_reason(self, phase: str) -> str:
        """Only ambient group replies may expire or yield to a newer turn."""
        if (self.attention_mode != "autonomous" or self.conversation_type not in {"group", "guild"}
                or self.autonomous_guard is None):
            return ""
        try:
            return str(self.autonomous_guard(phase) or "")
        except Exception:
            return "autonomous_guard_failed"

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
            "visual_candidates": list(self.visual_candidates),
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
        self.autonomous_queue_ttl_seconds = limits.qq_autonomous_queue_ttl_seconds
        self._clock = clock or time.monotonic
        self._timer_factory = timer_factory or threading.Timer
        self._lock = threading.RLock()
        self._timer: Any = None
        self._visuals: dict[tuple[str, str, str], tuple[NapCatVisualInput, ...]] = {}
        self._visual_context: dict[tuple[str, str, str], _VisualContext] = {}
        self._visual_history: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._visual_loader: Callable[..., Any] | None = None
        self._message_sequences: dict[tuple[str, str, str], int] = {}
        self._ingress_sequence = 0
        self.visual_source_retention_seconds = max(600.0, min(86400.0, self.visual_context_ttl_seconds * 4))
        self._last_error = ""
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="shinku-qq-turn")
            if async_handler else None
        )
        self._conversation_futures: dict[tuple[str, str], Future[Any]] = {}
        self._pending_turn_counts: dict[tuple[str, str], int] = {}
        self._conversation_queues: dict[tuple[str, str], list[QQTurn]] = {}
        self._running_conversations: set[tuple[str, str]] = set()
        self._turn_revisions: dict[tuple[str, str], int] = {}
        self._latest_scheduled_messages: dict[tuple[str, str], str] = {}
        self._ambient_arrivals: dict[tuple[str, str], list[tuple[int, _AmbientMessage]]] = {}
        self._ambient_history_floor: dict[tuple[str, str], int] = {}
        self._explicit_sequences: dict[tuple[str, str], int] = {}
        self._explicit_turn_revisions: dict[tuple[str, str], int] = {}

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
            self._ingress_sequence += 1
            self._message_sequences[(*self._conversation_key(message), message_id)] = self._ingress_sequence
            key = self._conversation_key(message)
            if message.conversation_type in {"group", "guild"}:
                arrivals = self._ambient_arrivals.setdefault(key, [])
                # 超长正文保留一个越界字符供 guard 拒绝，不能截断后照常确认整条已审。
                # 该窗口只服务短文本时效审查，不复制附件/引用原文或入站原始 payload。
                arrivals.append((self._ingress_sequence, _AmbientMessage(message_id, message.sender_id,
                    message.text[:2001], tuple(message.at_target_ids[:16]), tuple(message.reply_message_ids[:16]),
                    message.has_media, len(message.at_target_ids) > 16 or len(message.reply_message_ids) > 16)))
                if len(arrivals) > 32:
                    self._ambient_history_floor[key] = arrivals.pop(0)[0]
                if bool(getattr(getattr(ingress, "attention", None), "immediate", False)):
                    self._explicit_sequences[key] = self._ingress_sequence
                while len(self._ambient_arrivals) > 512:
                    old = next(iter(self._ambient_arrivals))
                    self._ambient_arrivals.pop(old)
                    self._ambient_history_floor.pop(old, None)
                    self._explicit_sequences.pop(old, None)
            while len(self._message_sequences) > 2048:
                self._message_sequences.pop(next(iter(self._message_sequences)))
            # The incoming event may close an older batch. Do not let that
            # batch see an image which arrived after its question.
            if closed is not None:
                dispatched += self._dispatch_locked(closed, now=now)
            scheduled = bool(getattr(result, "scheduled", False))
            if scheduled and message_id and message.conversation_type in {"group", "guild"}:
                key = self._conversation_key(message)
                # 新候选只替换尚未开始的旧候选；运行中的普通回复交给既有审查结合后续消息判断。
                self._latest_scheduled_messages.pop(key, None)
                self._latest_scheduled_messages[key] = message_id
                while len(self._latest_scheduled_messages) > 512:
                    self._latest_scheduled_messages.pop(next(iter(self._latest_scheduled_messages)))
            visual_key = (*self._conversation_key(message), message_id)
            if message_id and inputs and scheduled:
                self._visuals[visual_key] = inputs
            direct_inputs = tuple(item for item in inputs if item.reference_kind == "direct")
            if direct_inputs and message_id:
                # A new own image supersedes older implicit topics. Older
                # sources remain available only through their exact QQ quote.
                for key, context in self._visual_context.items():
                    if key[:2] == visual_key[:2] and context.source_sender_id == str(getattr(message, "sender_id", "") or ""):
                        context.superseded = True
                self._visual_context[visual_key] = _VisualContext(
                    created_at=now,
                    source_message_id=message_id,
                    inputs=direct_inputs,
                    remaining_turns=self.visual_followup_turns,
                    source_sender_id=str(getattr(message, "sender_id", "") or ""),
                    message=message, version=self._visual_version(visual_key, direct_inputs, now, self._ingress_sequence),
                    content_hashes=tuple(self._image_hash(item) for item in direct_inputs),
                    sequence=self._ingress_sequence,
                    cached_at=now,
                )
                while len(self._visual_context) > 512:
                    self._visual_context.pop(next(iter(self._visual_context)))
                # Bound passive image retention even in busy groups.
                group_keys = [key for key in self._visual_context if key[:2] == visual_key[:2]]
                for key in group_keys[:-8]:
                    self._visual_context.pop(key, None)
            # Text is retained as evidence; a lexical miss cannot close a topic.
            if message_id:
                history = self._visual_history.setdefault(visual_key[:2], [])
                history.append({"message_id": message_id, "sender_id": str(getattr(message, "sender_id", "") or ""),
                    "text": str(getattr(message, "text_without_at_mentions", getattr(message, "text", "")) or "")[:600],
                    "has_image": bool(direct_inputs)})
                history[-1]["sequence"] = self._ingress_sequence
                del history[:-8]
                while len(self._visual_history) > 256:
                    self._visual_history.pop(next(iter(self._visual_history)))
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
            queue = self._conversation_queues.get(key, [])
            obsolete = [item for item in queue if item.attention_mode == "autonomous"] if turn.attention_mode == "autonomous" else []
            # Count both the running turn and all queued turns. Explicit calls
            # bypass ambient admission limits and retain queue precedence.
            if turn.attention_mode == "autonomous" and (
                self._pending_turn_counts.get(key, 0) - len(obsolete) >= 2
                or sum(self._pending_turn_counts.values()) - len(obsolete) >= 24
            ):
                self._last_error = "autonomous_queue_full"
                return 0
            # 先检查替换后的容量，再删除旧候选，避免新候选仍被拒绝时连旧候选也误删。
            if obsolete:
                queue[:] = [item for item in queue if item.attention_mode != "autonomous"]
                self._pending_turn_counts[key] -= len(obsolete)
                self._last_error = "autonomous_queued_turn_coalesced"
            revision = self._turn_revisions.get(key, 0) + 1
            self._turn_revisions[key] = revision
            if turn.attention_mode != "autonomous":
                # 精简适配器可能只提供合并批次；显式优先级仍可由实际调度的回合确认。
                self._explicit_turn_revisions[key] = revision
            if turn.attention_mode == "autonomous":
                queued_at = self._now()
                turn = self._bind_ambient_window(turn, key, revision, queued_at)
            # Explicit calls keep their order and precede queued ambient turns.
            # One worker drains each conversation; waiting turns do not occupy
            # other executor threads while a previous model request is running.
            queue = self._conversation_queues.setdefault(key, queue)
            index = len(queue) if turn.attention_mode == "autonomous" else next(
                (index for index, item in enumerate(queue) if item.attention_mode == "autonomous"), len(queue),
            )
            queue.insert(index, turn)
            self._pending_turn_counts[key] = self._pending_turn_counts.get(key, 0) + 1
            if key not in self._running_conversations:
                self._running_conversations.add(key)
                try:
                    future = self._executor.submit(self._drain_conversation, key)
                except RuntimeError:
                    queue.remove(turn)
                    self._running_conversations.discard(key)
                    self._turn_revisions.pop(key, None)
                    self._explicit_turn_revisions.pop(key, None)
                    count = self._pending_turn_counts[key] - 1
                    if count:
                        self._pending_turn_counts[key] = count
                    else:
                        self._pending_turn_counts.pop(key, None)
                    if not queue:
                        self._conversation_queues.pop(key, None)
                    self._last_error = "turn_executor_closed"
                    return 0
                self._conversation_futures[key] = future
                future.add_done_callback(lambda completed, key=key: self._forget_completed(key, completed))
            return 1
        try:
            self.handler(turn)
        except Exception as exc:
            self._last_error = f"handler_error:{type(exc).__name__}"
        return 1

    def _bind_ambient_window(self, turn, key, revision, queued_at):
        """每轮独立审查游标，状态由调度器锁保护，不让普通补充反复中止模型生成。

        queue 阶段合并陈旧候选；reply 阶段保留生成、仍接受显式抢占；publish 阶段要求
        原有审查覆盖全部后续消息。最多审16条；总期限取120秒与队列期限较大者，缓存丢失、
        更新溢出或超期均保守退出，
        不无限追赶新消息、不重试生成。每段发送仍单独检查，已发送内容不宣称可撤回。
        """
        origin = max((self._message_sequences.get((*key, mid), 0) for mid in turn.message_ids),
                     default=0) or self._ingress_sequence
        reviewed = origin
        history = self._ambient_arrivals.setdefault(key, [])

        def updates():
            with self._lock:
                if self._ambient_arrivals.get(key) is not history:
                    raise ValueError("autonomous_updates_unavailable")
                if self._ambient_history_floor.get(key, 0) > origin:
                    raise ValueError("autonomous_updates_overflow")
                rows = tuple(m for seq, m in self._ambient_arrivals.get(key, ()) if seq > origin)
                if len(rows) > 16:
                    raise ValueError("autonomous_updates_overflow")
                # 后续图片未在本轮读取；长正文也不能截掉尾部更正后冒充整条已审。
                if any(m.unreadable or m.has_media or len(m.text) > 2000 for m in rows):
                    raise ValueError("autonomous_updates_unreadable")
                return rows

        def acknowledge(message_ids):
            nonlocal reviewed
            with self._lock:
                if tuple(m.message_id for m in updates()) != message_ids:
                    return False
                reviewed = max((seq for seq, _ in self._ambient_arrivals.get(key, ())), default=origin)
                return True

        def guard(phase):
            with self._lock:
                if (self._explicit_sequences.get(key, 0) > origin
                        or self._explicit_turn_revisions.get(key, 0) > revision):
                    return "autonomous_turn_superseded"
                if phase == "queue":
                    latest = self._latest_scheduled_messages.get(key)
                    if self._turn_revisions.get(key) != revision or latest and latest not in turn.message_ids:
                        return "autonomous_turn_superseded"
                    if self._now() - queued_at > self.autonomous_queue_ttl_seconds:
                        return "autonomous_queue_expired"
                # 模型生成和既有审查期间不因普通补充而作废；未审到的新消息不能越过发送出口。
                if self._now() - queued_at > max(120.0, self.autonomous_queue_ttl_seconds):
                    return "autonomous_reply_expired"
                try:
                    updates()
                except ValueError as exc:
                    return str(exc)
                if phase == "publish" and any(seq > reviewed for seq, _ in self._ambient_arrivals.get(key, ())):
                    return "autonomous_updates_unreviewed"
                return ""

        return replace(turn, autonomous_guard=guard, ambient_updates=updates, ambient_reviewed=acknowledge)

    def _drain_conversation(self, key: tuple[str, str]) -> None:
        while True:
            with self._lock:
                queue = self._conversation_queues.get(key)
                if not queue:
                    self._conversation_queues.pop(key, None)
                    self._running_conversations.discard(key)
                    return
                turn = queue.pop(0)
            try:
                self.handler(turn)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"handler_error:{type(exc).__name__}"
            finally:
                with self._lock:
                    count = max(0, self._pending_turn_counts.get(key, 1) - 1)
                    if count:
                        self._pending_turn_counts[key] = count
                    else:
                        self._pending_turn_counts.pop(key, None)

    def _forget_completed(self, key: tuple[str, str], completed: Future[Any]) -> None:
        with self._lock:
            if self._conversation_futures.get(key) is completed:
                self._conversation_futures.pop(key, None)
                if completed.cancelled():
                    self._conversation_queues.pop(key, None)
                    self._pending_turn_counts.pop(key, None)
                    self._running_conversations.discard(key)
                if key not in self._running_conversations:
                    self._turn_revisions.pop(key, None)
                    self._explicit_turn_revisions.pop(key, None)

    def _build_turn_locked(self, batch: AttentionBatch, *, now: float) -> QQTurn:
        message_ids = tuple(str(getattr(item, "message_id", "") or "").strip() for item in batch.messages)
        direct_visuals = tuple(
            (message_id, self._visuals.pop((*self._conversation_key(message), message_id), ()))
            for message, message_id in zip(batch.messages, message_ids)
            if message_id
        )
        visuals = list(direct_visuals)
        # Direct attachments and exact quote bytes are already input evidence.
        # Retained images become metadata candidates until the chat decision
        # selects them and the host validates the original again.
        candidates = () if any(inputs for _, inputs in direct_visuals) else self._visual_candidates_locked(batch, now=now)
        resolver = None
        if candidates:
            offered = tuple(dict(row) for row in candidates)
            used = False
            def resolver(source_ids, changed):
                nonlocal used
                with self._lock:
                    if used:
                        return VisualResolution(error="visual_selection_already_consumed")
                    used = True
                return self._resolve_visual_candidates(batch, offered, source_ids, changed)
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
            visual_candidates=candidates,
            visual_history=tuple(row for row in self._visual_history.get(self._conversation_key(primary), ())
                if row["sequence"] <= self._message_sequences.get((*self._conversation_key(primary),
                    str(getattr(primary, "message_id", ""))), -1)) if candidates else (),
            visual_resolver=resolver,
            attention_mode=attention_mode,
            attention_signals=tuple(dict.fromkeys(signal for decision in batch.decisions for signal in decision.signals)),
        )

    def set_visual_loader(self, loader: Callable[..., Any]) -> None:
        self._visual_loader = loader

    def revoke_visual_source(self, conversation_type: str, conversation_id: str, message_id: str) -> bool:
        """Drop an authoritative source; pending choices must revalidate it."""
        key = (str(conversation_type).strip().casefold(), str(conversation_id).strip(), str(message_id).strip())
        with self._lock:
            present = self._visual_context.pop(key, None) is not None
            self._visuals.pop(key, None)
            history = self._visual_history.get(key[:2], [])
            history[:] = [row for row in history if row["message_id"] != key[2]]
            return present

    @staticmethod
    def _image_hash(item: NapCatVisualInput) -> str:
        return hashlib.sha256(item.data_url.encode()).hexdigest() if item.ready else ""

    @staticmethod
    def _visual_version(key, inputs, created, sequence):
        body = [key, created, sequence, [(item.source_kind, item.reference, item.source_sender_id,
                               NapCatTurnDispatcher._image_hash(item)) for item in inputs]]
        return hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()

    def _visual_candidates_locked(self, batch, *, now):
        primary = batch.primary_message
        scope = self._conversation_key(primary)
        replies = tuple(getattr(primary, "reply_message_ids", ()) or ())
        request_sequence = self._message_sequences.get((*scope, str(getattr(primary, "message_id", ""))))
        rows = []
        for key, ctx in self._visual_context.items():
            exact = ctx.source_message_id in replies
            if (key[:2] != scope or not ctx.source_sender_id or ctx.remaining_turns <= 0
                    or request_sequence is None or ctx.sequence > request_sequence or ctx.created_at > now
                    or not exact and (not ctx.followup_open or ctx.superseded)
                    or replies and not exact):
                continue
            hint = self._is_visual_followup(str(getattr(primary, "text_without_at_mentions", "")), ctx, now=now)
            rows.append({"source_id": "visual:" + ctx.version, "message_id": ctx.source_message_id,
                "conversation_type": scope[0], "conversation_id": scope[1],
                "request_message_id": str(getattr(primary, "message_id", "")),
                "sender_id": ctx.source_sender_id, "sender_label": str(getattr(ctx.message, "sender_label", "")),
                "version": ctx.version, "relation": "quoted" if exact else "candidate",
                "age_seconds": max(0.0, now - ctx.created_at), "lexical_hint": hint,
                "same_sender": ctx.source_sender_id == str(getattr(primary, "sender_id", "")),
                "image_count": len(ctx.inputs), "cached": all(item.ready for item in ctx.inputs),
                "source_text": str(getattr(ctx.message, "text_without_at_mentions", ""))[:600]})
        # Quotation, lexical matches and recency rank candidates, never approve them.
        rows.sort(key=lambda r: (r["relation"] != "quoted", not r["lexical_hint"], r["age_seconds"]))
        return tuple(rows[:5])

    def _resolve_visual_candidates(self, batch, offered, source_ids, changed):
        scope = self._conversation_key(batch.primary_message)
        allowed = {row["source_id"]: row for row in offered}
        if any(source not in allowed for source in source_ids):
            return VisualResolution(error="visual_source_not_offered")
        with self._lock:
            self._prune_visual_context_locked(self._now())
            if not source_ids:
                if changed:
                    for key, ctx in self._visual_context.items():
                        if key[:2] == scope and "visual:" + ctx.version in allowed:
                            ctx.followup_open = False
                return VisualResolution()
            entries = []
            for source in source_ids:
                row = allowed[source]
                key = (*scope, row["message_id"])
                ctx = self._visual_context.get(key)
                if (ctx is None or ctx.version != row["version"] or ctx.source_sender_id != row["sender_id"]
                        or not ctx.source_sender_id or ctx.remaining_turns <= 0
                        or row["relation"] != "quoted" and (not ctx.followup_open or ctx.superseded)):
                    return VisualResolution(error="visual_source_changed")
                entries.append((key, ctx))
            if sum(len(ctx.inputs) for _, ctx in entries) > 5:
                return VisualResolution(error="visual_read_budget")
        prepared, reads = [], 0
        for key, ctx in entries:
            inputs = ctx.inputs
            if not all(item.ready for item in inputs):
                if self._visual_loader is None or ctx.message is None:
                    return VisualResolution(error="visual_source_unavailable", reads=reads)
                reads += 1
                try:
                    from shinku.agent.budget import current_task_budget
                    budget = current_task_budget()
                    if budget is not None:
                        budget.use_tool()
                    loaded = self._visual_loader(ctx.message)
                    inputs = tuple(item for item in loaded.inputs if item.reference_kind == "direct")
                except Exception as exc:
                    from shinku.agent.budget import TaskBudgetExceeded
                    if isinstance(exc, TaskBudgetExceeded):
                        return VisualResolution(error="visual_read_budget", reads=reads - 1)
                    return VisualResolution(error="visual_read_failed", reads=reads)
            if (len(inputs) != len(ctx.inputs) or not all(item.ready for item in inputs)
                    or any(item.source_message_id != ctx.source_message_id or item.source_sender_id != ctx.source_sender_id
                           for item in inputs)):
                return VisualResolution(error="visual_source_unavailable", reads=reads)
            if any(old and old != self._image_hash(item) for old, item in zip(ctx.content_hashes, inputs)):
                return VisualResolution(error="visual_source_changed", reads=reads)
            prepared.append((key, ctx, inputs))
        with self._lock:
            self._prune_visual_context_locked(self._now())
            for key, ctx, _ in prepared:
                if (self._visual_context.get(key) is not ctx or ctx.remaining_turns <= 0
                        or "visual:" + ctx.version not in allowed
                        or allowed["visual:" + ctx.version]["relation"] != "quoted" and (not ctx.followup_open or ctx.superseded)):
                    return VisualResolution(error="visual_source_changed", reads=reads)
            for _, ctx, inputs in prepared:
                ctx.remaining_turns -= 1
                ctx.inputs = inputs
                ctx.cached_at = self._now()
                ctx.content_hashes = tuple(self._image_hash(item) for item in inputs)
            return VisualResolution(tuple((ctx.source_message_id, tuple(replace(item, reference_kind="inherited")
                for item in inputs)) for _, ctx, inputs in prepared), reads=reads)

    @staticmethod
    def _is_visual_followup(text: str, context: _VisualContext, *, now: float) -> bool:
        """Compatibility name for a ranking hint, never a semantic gate."""
        return bool(_VISUAL_FOLLOWUP_RE.search(text)) or (
            0 <= now - context.created_at <= VISUAL_REACTION_WINDOW_SECONDS
            and is_short_visual_reaction(text)
        ) or (
            0 <= now - context.created_at <= VISUAL_QUESTION_WINDOW_SECONDS
            and is_short_visual_question(text)
        )

    def _prune_visual_context_locked(self, now: float) -> None:
        for key, ctx in tuple(self._visual_context.items()):
            if now - ctx.created_at > self.visual_source_retention_seconds:
                self._visual_context.pop(key, None)
                self._visuals.pop(key, None)
            elif now - ctx.cached_at > self.visual_context_ttl_seconds:
                # Evict bytes, retain the bounded authoritative message/ref.
                # Expiry is a read state, never a semantic topic decision.
                ctx.inputs = tuple(replace(item, data_url="") for item in ctx.inputs)

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
