"""显式开启的请求内检索诊断，不保存正文、不修改检索或生命周期。

调用方用 ``capture_retrieval`` 包住一次 Store/Service/Projection 调用，结束后
自行读取快照。各检索阶段只上报计数和有界来源定位信息；不写数据库、记录
metadata、模型上下文或全局 last_status。诊断不是来源权限或实际使用证据。

读取顺序：capture_retrieval -> observe/query -> RetrievalTrace.snapshot。
ContextVar 管理嵌套；线程/异步任务所有权防止继承的上下文混入另一个请求。
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import math
import threading
from typing import Any, Iterator, Sequence


def _owner() -> tuple[int, Any]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), 8) if math.isfinite(value) else None


class RetrievalTrace:
    """有界事件账本；watch 只帮助观察指定候选，不触发任何额外来源读取。"""

    def __init__(self, *, event_limit: int, record_limit: int, watch: Sequence[tuple[str, str]]):
        if (type(event_limit) is not int or not 1 <= event_limit <= 256
                or type(record_limit) is not int or not 1 <= record_limit <= 128):
            raise ValueError("invalid_retrieval_trace_budget")
        if len(watch) > 16 or any(
            not isinstance(item, tuple) or len(item) != 2
            or any(not isinstance(value, str) or len(value) > 240 for value in item)
            for item in watch
        ):
            raise ValueError("invalid_retrieval_trace_watch")
        self.event_limit, self.record_limit = event_limit, record_limit
        self.watch = tuple(dict.fromkeys(watch))
        self.owner = _owner()
        self.events: list[dict[str, Any]] = []
        self.dropped_events = 0
        self.closed = False

    def add(self, stage: str, records: Sequence[Any], details: dict[str, Any]) -> None:
        if self.closed or _owner() != self.owner:
            return
        if len(self.events) >= self.event_limit:
            self.dropped_events += 1
            return
        # 只保留固定调用点给出的标量。正文与查询不允许通过 details 混入。
        allowed = {"scope", "limit", "mode", "reason", "query_hash", "query_chars",
                   "semantic_enabled", "require_term_overlap", "rendered_chars",
                   "requested", "rejected", "source_count", "candidate_count",
                   "vector_truncated", "lexical_truncated", "index_ready", "state", "partial_count"}
        values = {key: value for key, value in details.items()
                  if key in allowed and isinstance(value, (str, int, float, bool))
                  and (not isinstance(value, str) or len(value) <= 240)
                  and (not isinstance(value, float) or math.isfinite(value))}
        rows = []
        seen = set()
        # 扫描现有候选确认 watch 是否存在；不再查库，也不把采样遗漏当未召回。
        for index, record in enumerate(records):
            if isinstance(record, tuple):
                record_id, scope = record[:2]
                row = {"record_id": record_id, "scope": scope}
            else:
                record_id, scope = record.record_id, record.scope
                row = {"record_id": record_id, "scope": scope, "layer": record.layer,
                       "score": _number(record.score),
                       "representation": "summary" if record.metadata.get("raw_detail_unavailable")
                       or "summary" in record.layer else "source"}
                components = record.metadata.get("ranking_components")
                if isinstance(components, dict):
                    row["ranking_components"] = {
                        key: _number(components[key]) for key in ("semantic", "exact", "lexical", "auxiliary")
                        if key in components
                    }
            key = (str(record_id), str(scope))
            if key in self.watch:
                seen.add(key)
            if index < self.record_limit:
                rows.append(row)
        self.events.append({"stage": stage, "count": len(records), "sample": rows,
                            "sample_truncated": len(records) > len(rows),
                            "watch": [{"record_id": rid, "scope": scope, "present": (rid, scope) in seen}
                                      for rid, scope in self.watch], **values})

    def snapshot(self) -> dict[str, Any]:
        """返回独立 JSON 数据，调用方修改快照不会污染后续观察。"""
        import copy
        return {"version": "retrieval-trace-v1", "events": copy.deepcopy(self.events),
                "dropped_events": self.dropped_events, "complete": self.dropped_events == 0,
                "event_limit": self.event_limit, "record_limit": self.record_limit}


_current: ContextVar[RetrievalTrace | None] = ContextVar("shinku_retrieval_trace", default=None)


def enabled() -> bool:
    trace = _current.get()
    return trace is not None and not trace.closed and trace.owner == _owner()


def observe(stage: str, records: Sequence[Any] = (), **details: Any) -> None:
    """默认空操作；固定阶段调用不获取模型、数据库或持久化资源。"""
    trace = _current.get()
    if trace is not None:
        trace.add(stage, records, details)


def query(stage: str, text: str, **details: Any) -> None:
    if enabled():
        observe(stage, query_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                query_chars=len(text), **details)


@contextmanager
def capture_retrieval(*, event_limit: int = 64, record_limit: int = 32,
                      watch: Sequence[tuple[str, str]] = ()) -> Iterator[RetrievalTrace]:
    """显式隔离一次诊断；异常时也恢复父上下文，不吞掉业务异常。"""
    trace = RetrievalTrace(event_limit=event_limit, record_limit=record_limit, watch=watch)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        trace.closed = True
        trace.owner = (0, None)
        _current.reset(token)
