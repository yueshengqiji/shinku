"""Host-owned cumulative task limits, inherited by nested synchronous model tools.

Counters are admission guards, not billing. Missing/streaming usage and failed
SDK attempts retain a conservative input-byte + output-limit reservation.
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


@dataclass
class TaskBudget:
    max_requests: int = 120
    max_tokens: int = 600000
    max_tools: int = 128
    max_active_seconds: float = 1800.0
    requests: int = 0
    tokens: int = 0
    tools: int = 0
    reported_tokens: int = 0
    estimated_requests: int = 0
    active_seconds: float = 0.0
    reason: str = ""
    _lock: Any = field(default_factory=threading.RLock, repr=False)
    _depth: int = field(default=0, repr=False)
    _started: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if (self.max_requests < 1 or self.max_tokens < 1 or not 1 <= self.max_tools <= 128
                or not math.isfinite(self.max_active_seconds) or self.max_active_seconds <= 0):
            raise ValueError("invalid_task_budget")

    def stop_reason(self) -> str:
        with self._lock:
            elapsed = self.active_seconds + (time.monotonic() - self._started if self._depth else 0)
            if not self.reason and elapsed >= self.max_active_seconds:
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
        output_bound = int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 32768)
        reservation = input_bound + max(1, output_bound)
        with self._lock:
            if self.stop_reason():
                self._deny(self.reason)
            if self.requests >= self.max_requests:
                self._deny("task_model_budget_exhausted")
            if self.tokens + reservation > self.max_tokens:
                self._deny("task_token_budget_exhausted")
            self.requests += 1
            self.tokens += reservation
            self.estimated_requests += 1
        return reservation

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
                if self.tokens >= self.max_tokens:
                    self.reason = self.reason or "task_token_budget_exhausted"

    def use_tool(self):
        with self._lock:
            if self.stop_reason():
                self._deny(self.reason)
            if self.tools >= self.max_tools:
                self._deny("task_tool_budget_exhausted")
            self.tools += 1

    def snapshot(self):
        with self._lock:
            return {key: getattr(self, key) for key in (
                "max_requests", "max_tokens", "max_tools", "max_active_seconds",
                "requests", "tokens", "tools", "reported_tokens", "estimated_requests", "reason") } | {
                "active_seconds": self.active_seconds + (time.monotonic() - self._started if self._depth else 0),
                "token_accounting": "reported_usage_or_conservative_reservation",
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
            for key in ("max_requests", "max_tokens", "max_tools", "max_active_seconds"):
                if key in snapshot and valid(key, snapshot[key], limit=True):
                    setattr(self, key, min(getattr(self, key), snapshot[key]))
            self.reason = self.reason or str(snapshot.get("reason") or "")
            if self.requests > self.max_requests:
                self.reason = self.reason or "task_model_budget_exhausted"
            if self.tokens >= self.max_tokens:
                self.reason = self.reason or "task_token_budget_exhausted"
            if self.tools > self.max_tools:
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
    if budget and not payload.get("max_tokens") and not payload.get("max_completion_tokens"):
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
