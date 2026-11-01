"""Small, rebuildable current understanding with explicit source authority.

Local recognizers only support a bounded vocabulary. Unsupported statements stay
ordinary sources; inference, quoted speech and unknown authors never become facts.
Different statements remain visible conflicts; only the existing confirmed
correction operation replaces a fact. No remote extraction, graph or new truth DB.
"""
from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any

from .source_reader import MemorySourceReader
from .lifecycle import sensitive

VERSION = "explicit-current-view-v4"
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_current_keys (
 record_id TEXT PRIMARY KEY, scope TEXT NOT NULL, source_hash TEXT NOT NULL,
 generator_version TEXT NOT NULL, claims_json TEXT NOT NULL, updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_current_keys_scope ON memory_current_keys(scope);
"""
_INDIRECT = re.compile(r'他说|她说|听说|转述|据说|假如|假设|如果|比如|也许|可能|猜测|建议|你说|[“”"「」]')
_CONDITIONS = re.compile(r"工作时|休息时|平时|这周|本周|今天|暂时|只在[^，。；;]{1,20}(?:里|时)")
_TEMPORAL = re.compile(r"这周|本周|上周|下周|今天|明天|昨天|暂时|最近|近期|这几天|这个月|本月|上个月|下个月|今年|明年|\d{1,2}月\d{1,2}(?:日|号)?")
_ACTIVITY = re.compile(r"(?:工作|办公|学习|阅读|休息|写(?:报告|材料|文章|长篇文章)|写作)(?:的时候|时)")
_PERSONAL_IF = re.compile(r"^如果我[^，。；;？！?]{1,24}[，,](?:我)?(?:就)?")
_TOPICS = (
    ("diet:香菜", ("香菜",)), ("diet:辣", ("辣",)),
    ("diet:川菜", ("川菜",)), ("diet:面食", ("面食",)),
    ("drink:咖啡", ("咖啡",)), ("drink:选茶", ("喝茶", "红茶", "绿茶", "乌龙茶")),
    ("environment:噪声", ("环境", "安静", "热闹", "噪声")),
    ("environment:背景音乐", ("背景音乐", "轻音乐", "音乐", "背景声音")),
)


def claims(record: Any) -> list[dict[str, Any]]:
    """从本人来源读取有限偏好词汇，保留整句条件，不判断条件已经成立。

    “如果我…”仅在明确保存的偏好中作为条件解析；普通假设、引用和未知
    作者仍不产生当前理解。派生版本变化只触发重建，不改原文或确认状态。
    """
    meta = record.metadata
    author = str(meta.get("sender_id") or "").strip()
    body = record.text.strip()
    confirmed = record.layer == "fact" and meta.get("confirmation_kind") == "explicit_memory_request"
    personal_if = _PERSONAL_IF.match(body) if confirmed else None
    indirect_body = body.removeprefix("如果") if personal_if else body
    if meta.get("role") != "user" or not author or _INDIRECT.search(indirect_body) or sensitive(body):
        return []
    activity = _ACTIVITY.match(body)
    if not (re.match(r"^(?:我|本人|工作时我|休息时我|平时我|这周我|本周我|今天我)", body)
            or activity and body[activity.end():].startswith("我") or personal_if):
        return []
    if record.layer == "fact" and not confirmed:
        return []  # Legacy/third-party facts do not acquire new confirmation.
    # 仅在分号后明确重新给出本人/活动条件时拆分；否则保留整句，避免把
    # “这周不吃辣；也不喝咖啡”后半句误标成永久约束。原文始终完整交付。
    segments = [part.strip() for part in re.split(r"[；;]", body) if part.strip()]
    if not all(re.match(r"^(?:我|本人|平时|这周|本周|今天)", part) or _ACTIVITY.match(part)
               or confirmed and _PERSONAL_IF.match(part) for part in segments):
        segments = [body]
    found = []
    for segment in segments:
        conditions = _CONDITIONS.findall(segment) + _TEMPORAL.findall(segment) + _ACTIVITY.findall(segment)
        if _PERSONAL_IF.match(segment):
            conditions.append(re.split(r"[，,]", segment, maxsplit=1)[0])
        condition = "、".join(dict.fromkeys(conditions))
        temporal_uncertain = bool(_TEMPORAL.search(segment))
        for attribute, terms in _TOPICS:
            if not any(term in segment for term in terms):
                continue
            found.append({"subject": "user:" + author, "attribute": attribute,
                          "value": body, "condition": condition, "condition_text": segment,
                          "confirmation": "explicit_memory_request" if confirmed else "unconfirmed_observation",
                          "temporal_status": "needs_time_resolution" if temporal_uncertain else "not_specified",
                          "valid_from": None, "valid_until": None,
                          "corrected_at": meta.get("corrected_at") if meta.get("correction_confirmation") == "explicit" else None})
    return found[:8]


def relevant(claim: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    if any(word in query for word in ("偏好", "习惯", "限制", "当前理解")):
        return True
    return any(attribute == claim["attribute"] and any(term in query for term in terms)
               for attribute, terms in _TOPICS)


class CurrentUnderstanding(MemorySourceReader):
    def __init__(self, store: Any):
        super().__init__(store)
        with store._lock, store._connect() as conn:
            conn.executescript(SCHEMA)
        store.on_forget(self.invalidate)
        self.last_status: dict[str, Any] = {}

    def _bounded(self, conn: Any, scope: str, limit: int):
        fact_limit = min(max(1, int(limit)), self.store.fact_scan_limit)
        event_limit = min(max(1, int(limit)), self.store.event_scan_limit)
        ids = ["fact:" + row[0] for row in conn.execute(
            "SELECT fact_id FROM memory_facts WHERE scope=? AND status='active' ORDER BY updated_at DESC LIMIT ?",
            (scope, fact_limit))]
        ids += ["event:" + row[0] for row in conn.execute(
            "SELECT event_id FROM memory_events WHERE scope=? AND status='active' "
            "AND layer IN ('observation','episodic') AND source_id NOT LIKE 'owner:agent-task-%' "
            "ORDER BY created_at DESC LIMIT ?", (scope, event_limit))]
        return [snap for rid in ids if (snap := self._snapshot(conn, rid, scope)) is not None]

    def rebuild(self, *, scope: str, now: int, limit: int = 800,
                dry_run: bool = True, force: bool = False) -> dict[str, Any]:
        if not str(scope).strip():
            raise ValueError("exact_scope_required")
        self.store.expire(now=now)
        schema_missing = False
        if force and not dry_run:
            with self.store._lock, self.store._connect() as conn:
                schema_missing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_current_keys'").fetchone() is None
                conn.executescript(SCHEMA)
        with self.store._lock, self.store._connect() as conn:
            snapshots = self._bounded(conn, scope, limit)
            schema_missing = schema_missing or conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_current_keys'").fetchone() is None
            existing = {} if schema_missing and force and dry_run else {
                row["record_id"]: (row["source_hash"], row["generator_version"]) for row in conn.execute(
                "SELECT record_id,source_hash,generator_version FROM memory_current_keys WHERE scope=?", (scope,))}
        pending = [snap for snap in snapshots if force or existing.get(snap[0].record_id) != (snap[1], VERSION)]
        written = stale = failed = 0
        if not dry_run:
            generated = []
            for record, version, _ in pending:
                try:
                    generated.append((record, version, claims(record)))
                except Exception:
                    failed += 1
            if generated:
                with self.store._lock, self.store._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for record, version, value in generated:
                        fresh = self._snapshot(conn, record.record_id, scope)
                        if fresh is None or fresh[1] != version:
                            stale += 1
                            continue
                        conn.execute("INSERT OR REPLACE INTO memory_current_keys VALUES (?,?,?,?,?,?)",
                                     (record.record_id, scope, version, VERSION,
                                      json.dumps(value, ensure_ascii=False), now))
                        written += 1
        return {"scope": scope, "scanned": len(snapshots), "pending": len(pending), "written": written,
                "stale_rejected": stale, "failed": failed, "dry_run": dry_run, "force": force,
                "generator_version": VERSION, "remote_calls": 0, "scan_bounded": True, "schema_missing": schema_missing}

    def read(self, *, scope: str, subject: str, query: str, now: int, limit: int = 5):
        if not scope or not subject:
            raise ValueError("exact_scope_and_subject_required")
        fallback = ""
        try:
            self.last_status = self.rebuild(scope=scope, now=now, dry_run=False)
        except Exception as exc:
            fallback = "validated_source_parse"
            self.last_status = {"fallback": fallback, "error_type": type(exc).__name__}
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN")
            snapshots = self._bounded(conn, scope, self.store.event_scan_limit)
            cached = {} if fallback else {row["record_id"]: row for row in conn.execute(
                "SELECT * FROM memory_current_keys WHERE scope=?", (scope,))}
            selected = []
            for record, version, _ in snapshots:
                actual = claims(record)
                row = cached.get(record.record_id)
                if row is not None:
                    try:
                        cache_ok = (row["source_hash"] == version and row["generator_version"] == VERSION
                                    and json.loads(row["claims_json"]) == actual)
                    except (TypeError, ValueError):
                        cache_ok = False
                    if not cache_ok:
                        fallback = "validated_source_parse"
                matching = [claim for claim in actual if claim["subject"] == subject and relevant(claim, query)]
                if matching:
                    fresh = self._reading_source((record, version, ""))
                    selected.append((fresh, matching))
        entries = []
        for record, parsed in selected:
            for claim in parsed:
                entries.append({**claim, "record_id": record.record_id, "source_id": record.source_id,
                                "source_version": record.metadata["source_version"],
                                "state": "confirmed" if claim["confirmation"] == "explicit_memory_request" else "unconfirmed"})
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            groups.setdefault((entry["subject"], entry["attribute"], entry["condition"]), []).append(entry)
        conflicts = []
        for key, values in groups.items():
            if len({value["value"] for value in values}) > 1:
                conflicts.append({"attribute": key[1], "condition": key[2],
                                  "record_ids": [value["record_id"] for value in values],
                                  "reason": "different_statements_require_confirmation"})
                for entry in values:
                    entry["state"] = "conflict_pending_confirmation"
        bound = max(1, min(self.store.max_results, int(limit)))
        records = tuple(replace(record, metadata={**record.metadata, "current_understanding":
            [entry for entry in entries if entry["record_id"] == record.record_id]})
            for record, _ in selected[:bound])
        delivered = {record.record_id for record in records}
        return {"records": records, "entries": [entry for entry in entries if entry["record_id"] in delivered],
                "conflicts": conflicts, "scope": scope, "subject": subject, "fallback": fallback,
                "scan_bounded": True, "read_truncated": len(selected) > bound,
                "checked_sources": len(snapshots)}

    def inspect(self, *, scope: str, now: int, limit: int = 20):
        if not scope:
            raise ValueError("exact_scope_required")
        self.store.expire(now=now)
        with self.store._lock, self.store._connect() as conn:
            rows = conn.execute("SELECT record_id,source_hash,generator_version,updated_at FROM memory_current_keys "
                                "WHERE scope=? ORDER BY updated_at DESC LIMIT ?",
                                (scope, max(1, min(100, int(limit))))).fetchall()
            result = []
            for row in rows:
                current = self._snapshot(conn, row["record_id"], scope)
                state = "source_unavailable" if current is None else "current" if (
                    row["source_hash"] == current[1] and row["generator_version"] == VERSION) else "source_version_changed"
                result.append({**dict(row), "state": state})
        return {"scope": scope, "entries": result, "includes_source_text": False,
                "generator_version": VERSION, "last_status": self.last_status}

    def invalidate(self, *, scope: str, record_ids: set[str]):
        with self.store._lock, self.store._connect() as conn:
            if record_ids:
                conn.executemany("DELETE FROM memory_current_keys WHERE scope=? AND record_id=?",
                                 [(scope, record_id) for record_id in record_ids])
            elif scope:
                conn.execute("DELETE FROM memory_current_keys WHERE scope=?", (scope,))
            else:
                conn.execute("DELETE FROM memory_current_keys")
