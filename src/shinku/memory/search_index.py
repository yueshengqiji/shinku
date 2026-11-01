"""Local search hints, never substitute facts or reading evidence.

Every derived row binds content AND provenance. Generation is outside the SQLite
lock; commit and delivery recheck the source. No remote model or raw-turn fallback.
"""
from __future__ import annotations

import json
from typing import Any

from .source_reader import MemorySourceReader

VERSION = "local-topic-hints-v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_search_keys (
 record_id TEXT PRIMARY KEY, scope TEXT NOT NULL, source_hash TEXT NOT NULL,
 generator_version TEXT NOT NULL, hints_json TEXT NOT NULL,
 updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_search_keys_scope ON memory_search_keys(scope);
"""
# These are vocabulary labels, not assertions (especially not permanent preferences).
# Trigger words must occur in the source; the reader receives the original text.
TOPICS = (
    ("饮食限制", ("胃", "辣", "忌口", "吃饭", "饮食", "川菜", "香菜", "食物")),
    ("饮品偏好", ("咖啡", "红茶", "绿茶", "喝茶", "饮料", "饮品")),
    ("环境噪声", ("安静", "音乐", "嘈杂", "吵闹", "噪声", "背景音乐")),
    ("花费预算", ("预算", "费用", "花费", "价格", "成本", "元", "收费")),
    ("响应耗时", ("延迟", "毫秒", "耗时", "速度", "响应时间", "性能")),
    ("图像尺寸", ("分辨率", "尺寸", "像素", "宽度", "高度", "画幅")),
    ("设置参数", ("参数", "强度", "采样", "阈值", "设置", "配置")),
    ("选择理由", ("理由", "原因", "因为", "为何", "为什么", "选择", "放弃")),
)


def hints(text: str) -> tuple[str, ...]:
    body = str(text).lower()
    return tuple(label for label, words in TOPICS if any(word in body for word in words))


class MemorySearchIndex(MemorySourceReader):
    def __init__(self, store: Any):
        self.store = store
        with store._lock, store._connect() as conn:
            conn.executescript(SCHEMA)
        store.on_forget(self.invalidate)

    def rebuild(self, *, scope: str, now: int, limit: int = 800,
                dry_run: bool = True, force: bool = False) -> dict[str, Any]:
        if not str(scope).strip():
            raise ValueError("exact_scope_required")
        self.store.expire(now=now)
        schema_missing = False
        if force and not dry_run:
            # Explicit repair of a missing derived table; never rewrite sources.
            with self.store._lock, self.store._connect() as conn:
                schema_missing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search_keys'").fetchone() is None
                conn.executescript(SCHEMA)
        bound = max(1, min(self.store.event_scan_limit + self.store.fact_scan_limit, int(limit)))
        with self.store._lock, self.store._connect() as conn:
            ids = ["event:" + row[0] for row in conn.execute(
                "SELECT event_id FROM memory_events WHERE scope=? AND status='active' "
                "AND layer IN ('observation','episodic','legacy_raw','legacy_lesson') "
                "AND source_id NOT LIKE 'owner:agent-task-detail:%' ORDER BY created_at DESC LIMIT ?",
                (scope, bound))]
            ids += ["fact:" + row[0] for row in conn.execute(
                "SELECT fact_id FROM memory_facts WHERE scope=? AND status='active' ORDER BY updated_at DESC LIMIT ?",
                (scope, min(bound, self.store.fact_scan_limit)))]
            snapshots = [self._snapshot(conn, rid, scope) for rid in ids]
            schema_missing = schema_missing or conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search_keys'").fetchone() is None
            existing = {} if schema_missing and force and dry_run else {
                row["record_id"]: (row["source_hash"], row["generator_version"]) for row in conn.execute(
                "SELECT record_id,source_hash,generator_version FROM memory_search_keys WHERE scope=?", (scope,))}
        pending = [snap for snap in snapshots if snap is not None and
                   (force or existing.get(snap[0].record_id) != (snap[1], VERSION))]
        written = stale = failed = 0
        if not dry_run:
            generated = []
            for record, source_hash, _ in pending:
                try:
                    labels = hints(record.text)  # No SQL lock during generation.
                except Exception:
                    failed += 1
                    continue
                generated.append((record, source_hash, labels))
            if generated:
                with self.store._lock, self.store._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for record, source_hash, labels in generated:
                        current = self._snapshot(conn, record.record_id, scope)
                        if current is None or current[1] != source_hash:
                            stale += 1
                            continue
                        conn.execute("INSERT OR REPLACE INTO memory_search_keys VALUES (?,?,?,?,?,?)",
                                     (record.record_id, scope, source_hash, VERSION,
                                      json.dumps(labels, ensure_ascii=False), now))
                        written += 1
        return {"scope": scope, "scanned": len(ids), "pending": len(pending), "written": written,
                "stale_rejected": stale, "failed": failed, "dry_run": dry_run, "force": force,
                "generator_version": VERSION, "remote_calls": 0,
                "scan_bounded": True, "scan_limit": bound, "schema_missing": schema_missing}

    def inspect(self, *, scope: str, limit: int = 20) -> dict[str, Any]:
        """Bounded metadata only. Caller owns authorization to this exact scope."""
        if not str(scope).strip():
            raise ValueError("exact_scope_required")
        with self.store._lock, self.store._connect() as conn:
            rows = conn.execute("SELECT record_id,source_hash,generator_version,updated_at "
                                "FROM memory_search_keys WHERE scope=? ORDER BY updated_at DESC LIMIT ?",
                                (scope, max(1, min(100, int(limit))))).fetchall()
            items = []
            for row in rows:
                current = self._snapshot(conn, row["record_id"], scope)
                state = ("source_unavailable" if current is None else
                         "stale" if row["source_hash"] != current[1] or row["generator_version"] != VERSION else "current")
                items.append({**dict(row), "state": state})
        return {"scope": scope, "entries": items, "includes_source_text": False}

    def matches(self, *, records: list[Any], query: str, scope: str, now: int) -> dict[str, tuple[str, ...]]:
        # Existing candidate bounds, permissions and date windows remain authoritative.
        query_hints = set(hints(query))
        if not query_hints:
            return {}
        with self.store._lock, self.store._connect() as conn:
            result = {}
            for record in records:
                row = conn.execute("SELECT * FROM memory_search_keys WHERE record_id=? AND scope=?",
                                   (record.record_id, scope)).fetchone()
                current = self._snapshot(conn, record.record_id, scope)
                if row is None or current is None or row["source_hash"] != current[1] or row["generator_version"] != VERSION:
                    continue
                try:
                    labels = json.loads(row["hints_json"])
                    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                        continue
                    # Treat damaged/tampered derived rows as unavailable, not evidence.
                    if tuple(labels) != hints(current[0].text):
                        continue
                except (ValueError, TypeError):
                    continue
                found = tuple(sorted(query_hints.intersection(labels)))
                if found:
                    result[record.record_id] = found
            return result

    def invalidate(self, *, scope: str, record_ids: set[str]) -> None:
        with self.store._lock, self.store._connect() as conn:
            if record_ids:
                conn.executemany("DELETE FROM memory_search_keys WHERE scope=? AND record_id=?",
                                 [(scope, rid) for rid in record_ids])
            elif scope:
                conn.execute("DELETE FROM memory_search_keys WHERE scope=?", (scope,))
            else:
                conn.execute("DELETE FROM memory_search_keys")
