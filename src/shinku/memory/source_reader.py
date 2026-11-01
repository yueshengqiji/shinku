"""Exact revocable source reading shared by optional memory views.

No derived schema, inference, or extra authority; every delivery rechecks sources.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any
from . import retrieval_trace

def fingerprint(record: Any, conversation_id: str) -> str:
    value = [record.record_id, record.scope, record.source_id, record.layer, record.text,
             record.created_at, conversation_id, record.metadata]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class MemorySourceReader:
    def __init__(self, store: Any):
        self.store = store

    @staticmethod
    def _reject(rejections, reason):
        """可选的请求内原因收集；不保存正文、记录 ID 或全局最后状态。"""
        if rejections is not None and len(rejections) < 16:
            rejections.append(reason)
        return None

    def _snapshot(self, conn: Any, record_id: str, scope: str, *, _path=(), _budget=None,
                  _now=None, _rejections=None):
        budget = [128] if _budget is None else _budget
        if record_id in _path or len(_path) >= 8 or budget[0] <= 0:
            return self._reject(_rejections, "source_dependency_budget")
        budget[0] -= 1
        try:
            kind, bare = self.store.lifecycle._identity(record_id)
        except ValueError:
            return self._reject(_rejections, "invalid_source_handle")
        row = self.store.lifecycle._record(conn, kind, bare)
        if row is None:
            return self._reject(_rejections, "source_missing")
        if row["scope"] != scope:
            return self._reject(_rejections, "scope_not_allowed")
        if row["status"] != "active":
            reason = {"forgotten": "source_revoked", "expired": "source_expired"}.get(
                row["status"], "source_inactive")
            return self._reject(_rejections, reason)
        if _now is not None and row["expires_at"] and row["expires_at"] <= _now:
            managed = (kind == "event" and self.store.lifecycle.policy.mode == "active"
                       and conn.execute("SELECT 1 FROM memory_lifecycle WHERE record_type='event' AND record_id=?", (bare,)).fetchone())
            if not managed:
                return self._reject(_rejections, "source_expired")
        if str(row["source_id"]).startswith("owner:agent-task-detail:"):
            return self._reject(_rejections, "source_not_readable")
        if self.store.lifecycle.revoked(conn, scope=scope, source_id=row["source_id"], text=row["text"]):
            return self._reject(_rejections, "source_revoked")
        record = (self.store._event_record(row, score=0.0) if kind == "event"
                  else self.store._fact_record(row, score=0.0))
        if record.metadata.get("experience_schema") == 1:
            projector = getattr(self.store, "task_experience", None)
            if projector is None or not projector.validate_record(record):
                return self._reject(_rejections, "source_dependency_invalid")
        parents = record.metadata.get("source_record_ids", [])
        hashes = record.metadata.get("source_content_hashes", {})
        if not isinstance(parents, list) or len(parents) > 128 or not isinstance(hashes, dict):
            return self._reject(_rejections, "source_dependency_invalid")
        for parent_id in parents:
            if not isinstance(parent_id, str) or not hashes.get(parent_id):
                return self._reject(_rejections, "source_dependency_invalid")
            parent = self._snapshot(conn, parent_id, scope, _path=(*_path, record_id), _budget=budget,
                                    _now=_now, _rejections=_rejections)
            if parent is None:
                return None
            if hashlib.sha256(parent[0].text.encode("utf-8")).hexdigest() != hashes[parent_id]:
                return self._reject(_rejections, "source_dependency_changed")
        if (kind == "fact" or record.metadata.get("owner_participant_id")) and record.metadata.get("owner_origin_record_id"):
            # 当前理解的事实副本与参与者经历副本都依赖原件版本。
            # 原件修改或撤销即失效；这里只核验，不返回外部作用域的正文。
            meta = record.metadata
            try:
                origin_kind, origin_id = self.store.lifecycle._identity(meta["owner_origin_record_id"])
            except (ValueError, KeyError):
                return self._reject(_rejections, "owner_origin_unavailable")
            origin = self.store.lifecycle._record(conn, origin_kind, origin_id)
            if origin_kind != kind or origin is None or origin["status"] != "active" or origin["scope"] != meta.get("owner_origin_scope"):
                return self._reject(_rejections, "owner_origin_unavailable")
            if _now is not None and origin["expires_at"] and origin["expires_at"] <= _now:
                return self._reject(_rejections, "source_expired")
            if self.store.lifecycle.revoked(conn, scope=origin["scope"], source_id=origin["source_id"], text=origin["text"]):
                return self._reject(_rejections, "source_revoked")
            origin_record = (self.store._fact_record(origin, score=0.0) if kind == "fact"
                             else self.store._event_record(origin, score=0.0))
            if fingerprint(origin_record, str(origin["conversation_id"])) != meta.get("owner_origin_version"):
                return self._reject(_rejections, "owner_origin_changed")
        return record, fingerprint(record, str(row["conversation_id"])), str(row["conversation_id"])

    def read_source(self, *, record_id: str, scope: str, now: int,
                    expected_hash: str = ""):
        """Exact caller-authorized scope only; full retained source, no lost raw revival."""
        return self.read_sources(requests=((record_id, scope, expected_hash),), now=now)[0]

    def read_sources(self, *, requests: tuple[tuple[str, str, str], ...], now: int):
        """One bounded batch of exact handles, checked independently, never cached.

        Requests carry (record ID, caller-authorized scope, expected version).
        The caller supplies already bounded candidates. A missing/changed source
        occupies its original slot as None. Repeat callers must revalidate; this
        only shares expiry work and a read transaction, not source authority.
        """
        if not requests:
            return ()
        self.store.expire(now=now)
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN")
            snapshots = [self._snapshot(conn, record_id, scope) for record_id, scope, _ in requests]
        result = tuple(self._reading_source(snapshot) if snapshot is not None
                     and (not expected_hash or snapshot[1] == expected_hash) else None
                     for snapshot, (_, _, expected_hash) in zip(snapshots, requests))
        if retrieval_trace.enabled():
            retrieval_trace.observe("source.validated", tuple(row for row in result if row is not None),
                                    requested=len(requests), rejected=sum(row is None for row in result))
        return result

    @staticmethod
    def _reading_source(snapshot):
        record, source_hash, _ = snapshot
        return replace(record, metadata={**record.metadata, "source_version": source_hash,
                       "evidence_kind": "fact_record" if record.layer == "fact" else "retained_source",
                       "reading_representation": "summary" if (record.metadata.get("raw_detail_unavailable")
                           or "summary" in record.layer or record.metadata.get("record_type") in {"summary", "agent_task_summary"}) else "source"})

    def read_rounds(self, records: tuple[Any, ...], *, now: int, limit: int):
        """Revalidate hits; append only explicit reply edges inside the same origin.

        No inference from chronological proximity in a multi-user conversation.
        The limit is shared by hits and neighbours; all primary hits take priority.
        """
        primary = []
        fresh_sources = self.read_sources(requests=tuple((record.record_id, record.scope,
            record.metadata.get("source_version", "")) for record in records), now=now)
        for record, fresh in zip(records, fresh_sources):
            if fresh is not None:
                # query_bridge 只是本次查询的展示依赖，渲染端继续核对桥接版本；
                # 不落入来源正文或持久事实，也不能授予新的读取范围。
                primary.append(replace(fresh, score=record.score,
                    metadata={**fresh.metadata, **{key: value for key, value in record.metadata.items()
                              if key in {"search_hints", "search_generator", "retrieval_method", "semantic_similarity", "lexical_score", "current_understanding", "evidence_selection", "query_bridge"}}}))
        selected = {record.record_id: record for record in primary[:limit]}
        for record in primary:
            if len(selected) >= limit:
                break
            role = record.metadata.get("role")
            source = record.source_id
            if role == "assistant" and source.startswith(("turn:", "owner:turn:")) and source.endswith(":assistant"):
                target = source.removesuffix(":assistant")
            elif role == "user" and source.startswith(("turn:", "owner:turn:")):
                target = source + ":assistant"
            else:
                continue
            with self.store._lock, self.store._connect() as conn:
                original = self._snapshot(conn, record.record_id, record.scope)
                row = conn.execute("SELECT event_id,conversation_id FROM memory_events WHERE source_id=? AND scope=? AND status='active'",
                                   (target, record.scope)).fetchone()
                adjacent = self._snapshot(conn, "event:" + row["event_id"], record.scope) if row is not None else None
            if (adjacent is None or original is None or original[1] != record.metadata["source_version"]
                    or adjacent[2] != original[2] or adjacent[0].source_id != target):
                continue
            neighbor = self.read_source(record_id=adjacent[0].record_id, scope=record.scope, now=now,
                                        expected_hash=adjacent[1])
            if neighbor is not None and neighbor.metadata.get("role") in {"user", "assistant"} and neighbor.metadata.get("role") != role:
                selected.setdefault(neighbor.record_id, replace(neighbor, metadata={**neighbor.metadata, "reading_relation": "explicit_reply", "reading_parent": record.record_id}))
        return tuple(selected.values())
