"""宿主持有的累计用量账本，由嵌套同步模型工具继承。

有限模式同时执行准入上限，持续模式只计量；两者均不是费用账单。
缺少usage或SDK失败时保留输入字节与输出预留估计，真实usage到达后再结算。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import threading
import time
import math
from typing import Any


class TaskBudgetExceeded(RuntimeError):
    pass


_current: ContextVar[TaskBudget | None] = ContextVar("shinku_task_budget", default=None)
_output_reservation: ContextVar[int | None] = ContextVar("model_output_reservation", default=None)


@contextmanager
def output_reservation(tokens):
    """输出不设SDK限额时，按供应商最大能力预留；有限模式的未知能力占满剩余额度直至usage结算。"""
    token = _output_reservation.set(tokens)
    try:
        yield
    finally:
        _output_reservation.reset(token)


_reservation_observer: ContextVar[Any] = ContextVar('shinku_budget_reservation_observer', default=None)


@contextmanager
def observe_budget_reservations(observer):
    """Trusted synchronous checkpoint, after reservation and before SDK IO.

    Failure retains the conservative reservation and prevents the provider
    request. Ordinary calls have no observer and keep their existing behavior.
    """
    token = _reservation_observer.set(observer)
    try:
        yield
    finally:
        _reservation_observer.reset(token)


@dataclass
class TaskBudget:
    max_requests: int | None = 400
    max_tokens: int | None = 4000000
    max_tools: int | None = 256
    max_active_seconds: float | None = 7200.0
    requests: int = 0
    tokens: int = 0
    tools: int = 0
    reported_tokens: int = 0
    estimated_requests: int = 0
    active_seconds: float = 0.0
    reason: str = ""
    managed: bool = False
    profile: str = "legacy"
    last_reservation: int = 0
    rejected_reservation: int = 0
    _lock: Any = field(default_factory=threading.RLock, repr=False)
    _depth: int = field(default=0, repr=False)
    _started: float = field(default=0.0, repr=False)

    def __post_init__(self):
        # 只有可信宿主可用四个null选择持续计量；既不以超大数字伪装无限，
        # 也不允许半损坏的预算悄悄扩大。旧有限档位仍严格校验。
        caps = (self.max_requests, self.max_tokens, self.max_tools, self.max_active_seconds)
        if all(v is None for v in caps):
            return
        if (any(v is None for v in caps)
                or any(type(v) is not int or v < 1 for v in caps[:3])
                or self.max_tools > 1024
                or type(self.max_active_seconds) not in (int, float)
                or not math.isfinite(self.max_active_seconds) or self.max_active_seconds <= 0):
            raise ValueError("invalid_task_budget")

    @property
    def metered_only(self):
        """累计计数始终保留；持续模式不因累计数停止，不影响上下文窗口。"""
        return self.max_requests is None

    def stop_reason(self) -> str:
        with self._lock:
            elapsed = self.active_seconds + (time.monotonic() - self._started if self._depth else 0)
            if not self.reason and self.max_active_seconds is not None and elapsed >= self.max_active_seconds:
                self.reason = "task_time_budget_exhausted"
            return self.reason

    def _deny(self, reason):
        self.reason = self.reason or reason
        raise TaskBudgetExceeded(self.reason)

    def reserve_request(self, payload) -> int:
        # UTF-8 byte length is intentionally pessimistic for an unknown tokenizer,
        # but base64 image data URLs must not be counted as 1 byte = 1 token.
        # An image typically consumes 256-1600 tokens in multimodal vision models.
        messages = payload.get("messages", [])
        image_count = 0

        def _strip_large_data_urls(item: Any) -> Any:
            nonlocal image_count
            if isinstance(item, str):
                if item.startswith("data:image/"):
                    image_count += 1
                    return "<image_data_url>"
                return item
            if isinstance(item, dict):
                return {k: _strip_large_data_urls(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [_strip_large_data_urls(v) for v in item]
            return item

        sanitized_messages = _strip_large_data_urls(messages)
        input_bound = len(json.dumps(sanitized_messages, ensure_ascii=False, default=str).encode("utf-8"))
        input_bound += image_count * 1600
        input_bound += len(json.dumps(payload.get("tools", []), ensure_ascii=False, default=str).encode("utf-8"))
        explicit = payload.get("max_completion_tokens") or payload.get("max_tokens")
        reserved = _output_reservation.get()
        output_bound = int(explicit or reserved or 32768)
        reservation = input_bound + max(1, output_bound)
        with self._lock:
            if self.stop_reason():
                self._deny(self.reason)
            if self.max_requests is not None and self.requests >= self.max_requests:
                self._deny("task_model_budget_exhausted")
            if self.max_tokens is not None and not explicit and reserved == 0:
                # 未知供应商不能以32768的猜测制造额外可用预算。
                reservation = max(input_bound + 1, self.max_tokens - self.tokens)
            if self.max_tokens is not None and self.tokens + reservation > self.max_tokens:
                self.rejected_reservation = reservation
                self._deny("task_token_budget_exhausted")
            self.requests += 1
            self.tokens += reservation
            self.estimated_requests += 1
            self.last_reservation = reservation
            self.rejected_reservation = 0
        observer = _reservation_observer.get()
        if callable(observer):
            observer(self)
        return reservation

    def stage(self) -> str:
        """Advisory pressure only; admission checks enforce the actual hard caps.

        A previous large request is not proof that the next operation cannot
        fit. Checkpoint/yield must not turn available budget into a manual stop.
        """
        with self._lock:
            if not self.managed or self.metered_only:
                return "working"
            elapsed = self.active_seconds + (time.monotonic() - self._started if self._depth else 0)
            ratios = (self.requests / self.max_requests, self.tokens / self.max_tokens,
                      self.tools / self.max_tools, elapsed / self.max_active_seconds)
            return "warning" if max(ratios) >= .70 else "working"

    def reconcile(self, reservation, response):
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        def value(key):
            return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        total = value("total_tokens")
        if total is None:
            inp, out = value("prompt_tokens"), value("completion_tokens")
            if inp is not None and out is not None:
                total = int(inp) + int(out)
        if total is not None:
            with self._lock:
                count = max(0, int(total))
                self.tokens += count - reservation
                self.reported_tokens += count
                self.estimated_requests -= 1
                if self.max_tokens is not None and self.tokens >= self.max_tokens:
                    self.reason = self.reason or "task_token_budget_exhausted"

    def use_tool(self):
        with self._lock:
            if self.stop_reason():
                self._deny(self.reason)
            if self.max_tools is not None and self.tools >= self.max_tools:
                self._deny("task_tool_budget_exhausted")
            self.tools += 1

    def snapshot(self):
        with self._lock:
            return {key: getattr(self, key) for key in (
                "max_requests", "max_tokens", "max_tools", "max_active_seconds",
                "requests", "tokens", "tools", "reported_tokens", "estimated_requests", "reason") } | {
                "active_seconds": self.active_seconds + (time.monotonic() - self._started if self._depth else 0),
                "limit_mode": "metered" if self.metered_only else "bounded",
                "token_accounting": "reported_usage_or_conservative_reservation",
                "profile": self.profile, "managed": self.managed, "stage": self.stage(),
                "last_reservation": self.last_reservation,
                "rejected_reservation": self.rejected_reservation,
            }

    def absorb_prior(self, snapshot):
        """Merge a HOST ledger on explicit semantic continuation, never model notes."""
        if not isinstance(snapshot, dict):
            if snapshot is not None:
                self.reason = self.reason or "task_budget_snapshot_invalid"
            return
        with self._lock:
            def valid(key, val, *, limit=False):
                if key in {"active_seconds", "max_active_seconds"}:
                    numeric = type(val) in (int, float) and math.isfinite(val)
                else:
                    numeric = type(val) is int
                if not numeric or (val <= 0 if limit else val < 0):
                    self.reason = self.reason or "task_budget_snapshot_invalid"
                    return False
                return True
            for key in ("requests", "tokens", "tools", "reported_tokens", "estimated_requests", "active_seconds"):
                if key in snapshot and valid(key, snapshot[key]):
                    setattr(self, key, getattr(self, key) + snapshot[key])
            keys = ("max_requests", "max_tokens", "max_tools", "max_active_seconds")
            unlimited = snapshot.get("limit_mode") == "metered" and all(
                key in snapshot and snapshot[key] is None for key in keys)
            # 旧有限任务不会因宿主升级丢掉上限；有限宿主恢复持续任务时取更严上限。
            merged = {key: getattr(self, key) for key in keys}
            if not unlimited:
                for key in keys:
                    if key in snapshot and valid(key, snapshot[key], limit=True):
                        current = merged[key]
                        merged[key] = snapshot[key] if current is None else min(current, snapshot[key])
            # 损坏的半份上限不能留在对象中，否则保存诊断时stage也会二次崩溃。
            if any(v is None for v in merged.values()) and not all(v is None for v in merged.values()):
                self.reason = self.reason or "task_budget_snapshot_invalid"
            else:
                for key, value in merged.items():
                    setattr(self, key, value)
            self.reason = self.reason or str(snapshot.get("reason") or "")
            reservation = snapshot.get("last_reservation", 0)
            if type(reservation) is not int or reservation < 0:
                self.reason = self.reason or "task_budget_snapshot_invalid"
            else:
                self.last_reservation = max(self.last_reservation, reservation)
            if self.max_requests is not None and self.requests > self.max_requests:
                self.reason = self.reason or "task_model_budget_exhausted"
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                self.reason = self.reason or "task_token_budget_exhausted"
            if self.max_tools is not None and self.tools > self.max_tools:
                self.reason = self.reason or "task_tool_budget_exhausted"


def current_task_budget():
    return _current.get()


@contextmanager
def bind_task_budget(budget):
    token = _current.set(budget)
    if budget is not None:
        with budget._lock:
            if budget._depth == 0:
                budget._started = time.monotonic()
            budget._depth += 1
    try:
        yield budget
    finally:
        if budget is not None:
            with budget._lock:
                budget._depth -= 1
                if budget._depth == 0:
                    budget.active_seconds += time.monotonic() - budget._started
        _current.reset(token)


def metered_completion(create, payload):
    budget = current_task_budget()
    # 独立Agent/无限额结构化调用由输出预留器保护累计预算，不能再偷偷注入32k上限。
    if budget and not budget.metered_only and _output_reservation.get() is None and not payload.get("max_tokens") and not payload.get("max_completion_tokens"):
        payload = {**payload, "max_tokens": 32768}
    reservation = budget.reserve_request(payload) if budget else 0
    result = create(**payload)
    if budget:
        if payload.get("stream"):
            def monitored_stream():
                latest_usage = None
                try:
                    for chunk in result:
                        usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
                        if usage is not None:
                            latest_usage = chunk
                        yield chunk
                finally:
                    if latest_usage is not None:
                        budget.reconcile(reservation, latest_usage)
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
            return monitored_stream()
        budget.reconcile(reservation, result)
    return result
