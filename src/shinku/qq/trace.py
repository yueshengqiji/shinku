"""本机 QQ Agent 回合的结构化追踪缓冲区。

追踪用于控制中枢排查“这一轮为什么这样回复”，不是聊天记忆，也不是提示词
审计。它只保留有限的元数据，不保存用户原文、system prompt、工具参数或密钥。
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
import os
from collections import deque
from collections.abc import Mapping
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any

__all__ = ["TurnTraceStore"]


_SENSITIVE_KEYS = frozenset({
    "api_key",
    "arguments",
    "content",
    "cookie",
    "password",
    "prompt",
    "secret",
    "text",
    "token",
})

# Disk persistence uses an allow-list, not just a secret-key deny-list. Free
# text under e.g. "error" can itself contain provider credentials or dialogue.
_CONTROL_KEYS = frozenset({"name", "status", "agent_status", "delivery_status", "reason", "error",
    "field_path", "kind", "mode", "history_mode", "effect", "outcome", "model_outcome",
    "goal_status", "phase", "cause", "error_type", "source_kind", "failure_kind", "task_id",
    "trace_id", "source_message_id", "conversation_type", "conversation_id", "message_id", "rejection_code"})
_CONTROL_LIST_KEYS = frozenset({"selected_tools", "tool_names", "effects", "missing_fields", "field_paths", "message_ids", "error_codes"})
_METRIC_KEYS = frozenset({"started_at", "finished_at", "elapsed_ms", "input_chars", "reply_chars",
    "sent", "restored", "count", "attempt", "calls", "rounds", "slices", "total_tokens", "input_tokens",
    "output_tokens", "cached_tokens", "extra_model_calls", "retrieved_count", "injected_chars",
    "requested_tool_count", "prior_tool_count", "request_count", "http_status", "accepted", "valid",
    "verified", "result_count", "missing_count", "attachment_count", "ready_count", "model_image_count",
    "max_requests", "max_tokens", "max_tools", "max_active_seconds", "requests", "tokens", "tools",
    "active_seconds", "remaining_requests", "remaining_tokens", "remaining_tools", "requirements_count",
    "targets_count", "deliverables_count", "unknown_fields_count", "tool_count", "source_read",
    "segment_count", "message_id_count", "source_count", "read_chunks", "bound_claims"})
_NESTED_KEYS = frozenset({"budget", "usage", "verification", "completion", "nested", "contract_shape"})
_CONTROL_VALUE = re.compile(r"[A-Za-z0-9_.:\[\]/\-]{0,160}\Z")
_CONTROL_KEYS |= {'binding', 'span_id', 'parent_id', 'request_id'}
_METRIC_KEYS |= {'duration_ms', 'elapsed_seconds', 'cache_hit_tokens', 'cache_miss_tokens',
                 'cache_creation_tokens', 'reasoning_tokens', 'timing_truncated', 'timing_error'}


def _event_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """补充已有事件的安全字段；不把任意嵌套映射或相似键名放入通用白名单。"""
    clean: dict[str, Any] = {}
    if value.get("name") == "chat_stream_complete":
        first = value.get("first_chunk_ms")
        # None表示未取得观测，不能伪造为0；布尔值不是耗时或片段计数。
        if "first_chunk_ms" in value and (first is None or
                type(first) in (int, float) and 0 <= first <= 2**53 - 1 and math.isfinite(first)):
            clean["first_chunk_ms"] = first
        for key in ("chunks", "transport_chunks"):
            if type(value.get(key)) is int and 0 <= value[key] <= 2**31 - 1:
                clean[key] = value[key]
        if type(value.get("rescued")) is bool:
            clean["rescued"] = value["rescued"]
    elif value.get("name") == "context_sources":
        from shinku.memory.context_sources import (
            CONTEXT_SOURCE_READ_PATHS, CONTEXT_SOURCE_REJECTION_REASONS,
        )
        if value.get("state") in ("valid", "invalid"):
            clean["state"] = value["state"]
        if value.get("absence_means") == "unknown":
            clean["absence_means"] = "unknown"
        paths = value.get("read_paths")
        if isinstance(paths, (list, tuple)):
            clean["read_paths"] = [p for p in paths[:30]
                                   if isinstance(p, str) and p and p in CONTEXT_SOURCE_READ_PATHS]
        if type(value.get("rejected")) is int and 0 <= value["rejected"] <= 2**31 - 1:
            clean["rejected"] = value["rejected"]
        counts = value.get("reason_counts")
        if isinstance(counts, Mapping):
            clean["reason_counts"] = {k: v for k, v in list(counts.items())[:64]
                if isinstance(k, str) and k in CONTEXT_SOURCE_REJECTION_REASONS
                and type(v) is int and 0 <= v <= 2**31 - 1}
    return clean


def _response_protocol_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """只投影聊天响应的固定元数据；独立于其他事件，不扩展通用嵌套字段。"""
    clean: dict[str, Any] = {}
    if value.get("name") != "chat_response_protocol":
        return clean
    stages = value.get("stages")
    known = {"native_call", "json_object", "partial_object", "plain_reply_candidate",
        "parse_invalid", "empty_response", "transport_error", "output_incomplete",
        "final_text_json_recovered", "wait_text_json_recovered"}
    if isinstance(stages, (list, tuple)):
        clean["stages"] = [s for s in stages[:30] if isinstance(s, str) and s in known]
    if (isinstance(value.get("finish_reason"), str)
            and value["finish_reason"] in {"stop", "tool_calls", "length", "content_filter", "unknown"}):
        clean["finish_reason"] = value["finish_reason"]
    shape = value.get("response_shape")
    if isinstance(shape, Mapping):
        fields = {"transport": {"stream", "json"},
            "body_shape": {"empty", "object", "array", "fenced", "text", "other"},
            "native_call": {"none", "present", "malformed"}}
        detail = {k: shape[k] for k, choices in fields.items()
                  if isinstance(shape.get(k), str) and shape[k] in choices}
        if type(shape.get("body_chars")) is int and 0 <= shape["body_chars"] <= 2**31 - 1:
            detail["body_chars"] = shape["body_chars"]
        clean["response_shape"] = detail
    return clean


def _disk_details(value: Mapping[str, Any], *, depth: int = 0) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    if depth > 4:
        return clean
    for key, item in list(value.items())[:80]:
        if key in _CONTROL_KEYS and isinstance(item, str) and _CONTROL_VALUE.fullmatch(item):
            clean[key] = item
        elif key in _CONTROL_LIST_KEYS and isinstance(item, (list, tuple)):
            clean[key] = [entry for entry in item[:30] if isinstance(entry, str) and _CONTROL_VALUE.fullmatch(entry)]
        elif key in _METRIC_KEYS and (item is None or isinstance(item, bool)
                or isinstance(item, (int, float)) and math.isfinite(item)):
            clean[key] = item
        elif key in _NESTED_KEYS and isinstance(item, Mapping):
            clean[key] = _disk_details(item, depth=depth + 1)
    clean.update(_event_diagnostics(value))
    clean.update(_response_protocol_diagnostics(value))
    return clean


def contract_shape(value: Any) -> dict[str, Any]:
    """Field presence/types, never description, target or check values."""
    if not isinstance(value, Mapping):
        return {"status": "missing" if value is None else "invalid_type"}
    fields = []
    known = {"goal", "deliverables", "scope", "requirements", "id", "description", "effect", "targets", "checks", "target_selector"}
    check_keys = {"contains", "sha256", "min_bytes", "http_status", "page_contains", "receipt_required",
                  "required_fields", "min_sources", "require_dates", "date_kinds", "source_pages", "unchanged"}
    def add(path, item):
        kind = "object" if isinstance(item, Mapping) else "array" if isinstance(item, (list, tuple)) else type(item).__name__
        fields.append(path + ":" + kind)
    for key in ("goal", "deliverables", "scope", "requirements"):
        if key in value:
            add(key, value[key])
    rows = value.get("requirements", [])
    if isinstance(rows, list):
        for i, row in enumerate(rows[:24]):
            if not isinstance(row, Mapping):
                add(f"requirements[{i}]", row)
                continue
            for key in sorted(set(row) & known):
                add(f"requirements[{i}].{key}", row[key])
            checks = row.get("checks")
            if isinstance(checks, Mapping):
                for key in sorted(set(checks) & check_keys):
                    add(f"requirements[{i}].checks.{key}", checks[key])
    return {"status": "present", "field_paths": fields[:30],
            "requirements_count": len(rows) if isinstance(rows, list) else 0,
            "unknown_fields_count": len(set(value) - known)}


def _safe_detail(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, (list, tuple, set)):
        return [_safe_detail(item) for item in list(value)[:30]]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_detail(item)
            for key, item in list(value.items())[:30]
            if str(key).strip().lower() not in _SENSITIVE_KEYS
        }
    return str(value)[:160]


class TurnTraceStore:
    """线程安全的最近回合追踪缓冲区。"""

    def __init__(self, *, max_items: int = 100, clock: Any = time.time,
                 path: str | Path | None = None, retention_seconds: float = 7 * 86400,
                 max_stages: int = 128, max_bytes: int = 2 * 1024 * 1024) -> None:
        self.max_items = max(1, min(1000, int(max_items or 100)))
        self.clock = clock
        self._items: deque[dict[str, Any]] = deque(maxlen=self.max_items)
        self._lock = threading.RLock()
        self.path = Path(path) if path else None
        self.retention_seconds = max(1.0, min(30 * 86400, float(retention_seconds)))
        self.max_stages = max(1, min(256, int(max_stages)))
        self.max_bytes = max(1024, min(8 * 1024 * 1024, int(max_bytes)))
        self.persistence_error = ""
        self._restore()

    def begin(self, turn: Any) -> str:
        trace_id = uuid.uuid4().hex
        now = float(self.clock())
        record = {
            "trace_id": trace_id,
            "started_at": now,
            "finished_at": None,
            "elapsed_ms": None,
            "conversation_type": str(getattr(turn, "conversation_type", "") or "").strip().lower(),
            "conversation_id": str(getattr(turn, "conversation_id", "") or "").strip(),
            "message_id": str(getattr(turn, "primary_message_id", "") or "").strip(),
            "message_ids": [
                str(item).strip()
                for item in tuple(getattr(turn, "message_ids", ()) or ())[:30]
                if str(item).strip()
            ],
            "status": "running",
            "agent_status": "",
            "reason": "",
            "sent": False,
            "delivery_status": "not_sent",
            "input_chars": len(str(getattr(turn, "combined_text", "") or "")),
            "stages": [],
        }
        with self._lock:
            self._expire()
            self._items.append(record)
            self._persist()
        return trace_id

    @contextmanager
    def observe(self, trace_id):
        """每个同步回合/后台切片显式绑定；复用追踪存储，不创建业务状态。"""
        from shinku.phase_timing import bind_observer, clean_span, PHASES
        from shinku.llm.request_ledger import bind_provider_calls, provider_call_context
        from shinku.agent_runtime.evaluation import provider_usage, summarize_calls

        def emit(event, *, provider=False):
            try:
                if event.get('binding') != trace_id:
                    raise ValueError('trace_timing_binding_mismatch')
                if provider:
                    row = {k: event.get(k) for k in ('binding', 'request_id', 'status', 'elapsed_seconds')}
                    row['phase'] = event.get('phase') if event.get('phase') in PHASES else 'unclassified'
                    row['usage'] = {k: (event.get('usage') or {}).get(k) for k in provider_usage({})}
                    summarize_calls([dict(row, status='unknown' if row['status'] == 'started' else row['status'])])
                else:
                    row = clean_span(event)
                self.stage(trace_id, 'provider_request' if provider else 'phase_timing', **row)
            except Exception:
                with self._lock:
                    record = self._find(trace_id)
                    if record is not None:
                        record['timing_error'] = True
        upstream_binding, upstream = provider_call_context()
        def provider_event(row):
            emit(row, provider=True)
            # 保留已有外部验收账本；它继续使用原绑定，不能被新展示采集器吞掉。
            if callable(upstream):
                try:
                    upstream(dict(row, binding=upstream_binding))
                except Exception:
                    pass
        with bind_observer(trace_id, emit), bind_provider_calls(trace_id, provider_event):
            yield

    def stage(self, trace_id: str, name: str, **details: Any) -> None:
        identifier = str(trace_id or "").strip()
        if not identifier:
            return
        with self._lock:
            record = self._find(identifier)
            if record is None:
                return
            elapsed = round(max(0.0, (float(self.clock()) - float(record["started_at"])) * 1000), 1)
            clean = {
                str(key): _safe_detail(value)
                for key, value in details.items()
                if str(key).strip().lower() not in _SENSITIVE_KEYS
            }
            record["stages"].append({"name": str(name or "stage")[:80], "elapsed_ms": elapsed, **clean})
            if len(record['stages']) > self.max_stages:
                record['timing_truncated'] = True
            del record["stages"][:-self.max_stages]
            self._persist()

    def finish(self, trace_id: str, reply: Any = None, **details: Any) -> None:
        identifier = str(trace_id or "").strip()
        if not identifier:
            return
        with self._lock:
            record = self._find(identifier)
            if record is None:
                return
            now = float(self.clock())
            record["finished_at"] = now
            record["elapsed_ms"] = round(max(0.0, (now - float(record["started_at"])) * 1000), 1)
            if reply is not None:
                record["agent_status"] = str(getattr(reply, "agent_status", "") or "")
                record["sent"] = bool(getattr(reply, "sent", False))
                record["delivery_status"] = str(getattr(reply, "delivery_status", "not_sent") or "not_sent")
                record["reason"] = str(getattr(reply, "reason", "") or "")[:240]
                record["reply_chars"] = len(str(getattr(reply, "final_text", "") or ""))
                record["status"] = "completed" if record["agent_status"] in {"completed", "command"} else record["agent_status"] or "finished"
            for key, value in details.items():
                if str(key).strip().lower() not in _SENSITIVE_KEYS:
                    record[str(key)] = _safe_detail(value)
            if record["status"] == "running":
                record["status"] = "finished"
            self._persist()

    def snapshot(self, *, limit: int = 30) -> list[dict[str, Any]]:
        wanted = max(1, min(self.max_items, int(limit or 30)))
        with self._lock:
            self._expire()
            rows = list(self._items)[-wanted:]
            from shinku.phase_timing import summarize_phases
            snapshots = [self._copy(row) for row in reversed(rows)]
            for row in snapshots:
                row['performance'] = summarize_phases(
                    [s for s in row['stages'] if s.get('name') == 'phase_timing'],
                    [s for s in row['stages'] if s.get('name') == 'provider_request'],
                    truncated=row.get('timing_truncated') or row.get('timing_error') or bool(self.persistence_error))
                row['performance']['delegatedTaskIds'] = sorted({s['task_id'] for s in row['stages']
                    if s.get('name') == 'agent_delegation' and s.get('task_id')})
            return snapshots

    def _expire(self) -> None:
        cutoff = float(self.clock()) - self.retention_seconds
        self._items = deque((row for row in self._items if row["started_at"] >= cutoff), maxlen=self.max_items)

    def _disk_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        clean = _disk_details(record)
        stages = record.get("stages", [])
        clean["stages"] = [_disk_details(stage) for stage in stages[-self.max_stages:] if isinstance(stage, Mapping)]
        return clean

    def _restore(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            if self.path.stat().st_size > self.max_bytes:
                raise ValueError("trace_file_too_large")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("items"), list):
                raise ValueError("trace_file_invalid")
            for raw in payload["items"][-self.max_items:]:
                if not isinstance(raw, dict):
                    continue
                row = self._disk_record(raw)
                if (not row.get("trace_id") or not isinstance(row.get("started_at"), (int, float))
                        or isinstance(row["started_at"], bool)):
                    continue
                row["restored"] = True
                if row.get("status") == "running":
                    row["status"] = "interrupted"
                self._items.append(row)
            self._expire()
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            self.persistence_error = type(exc).__name__

    def _persist(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self._expire()
            rows = [self._disk_record(row) for row in self._items]
            while True:
                data = json.dumps({"version": 1, "items": rows}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                if len(data) <= self.max_bytes:
                    break
                if len(rows) > 1:
                    rows.pop(0)
                elif rows and rows[0].get("stages"):
                    rows[0]["stages"].pop(0)
                    rows[0]['timing_truncated'] = True
                else:
                    rows = []
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self.persistence_error = ""
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            # Diagnostic I/O never stops a conversation or grants execution.
            self.persistence_error = type(exc).__name__
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _find(self, trace_id: str) -> dict[str, Any] | None:
        for item in reversed(self._items):
            if item.get("trace_id") == trace_id:
                return item
        return None

    @staticmethod
    def _copy(record: Mapping[str, Any]) -> dict[str, Any]:
        # A diagnostic consumer must not be able to mutate live trace identity
        # or nested evidence through a returned snapshot.
        return copy.deepcopy(dict(record))
