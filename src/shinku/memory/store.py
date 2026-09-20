"""Small SQLite store for the independent Shinku memory layers."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import MemoryRecord

__all__ = ["MemoryStore"]


_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")


class MemoryStore:
    """Persistence with explicit layers and lifecycle fields.

    The source database is never involved.  Migration writes through the same
    API so imported records get the same scope, status and provenance rules as
    new records.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_query_terms: int = 12,
        event_scan_limit: int = 800,
        fact_scan_limit: int = 120,
        max_results: int = 12,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.configure(
            max_query_terms=max_query_terms,
            event_scan_limit=event_scan_limit,
            fact_scan_limit=fact_scan_limit,
            max_results=max_results,
        )
        self._initialize()

    def configure(
        self,
        *,
        max_query_terms: int = 12,
        event_scan_limit: int = 800,
        fact_scan_limit: int = 120,
        max_results: int = 12,
    ) -> None:
        self.max_query_terms = max(1, min(64, int(max_query_terms or 12)))
        self.event_scan_limit = max(10, min(10000, int(event_scan_limit or 800)))
        self.fact_scan_limit = max(10, min(5000, int(fact_scan_limit or 120)))
        self.max_results = max(1, min(100, int(max_results or 12)))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memory_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    text TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_used_at INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.25,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    expires_at INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_id TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_facts (
                    fact_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    last_used_at INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.8,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    status TEXT NOT NULL DEFAULT 'active',
                    expires_at INTEGER NOT NULL DEFAULT 0,
                    source_id TEXT NOT NULL DEFAULT ''
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_fact_key
                    ON memory_facts(scope, fact_key);
                CREATE TABLE IF NOT EXISTS memory_forget_log (
                    forget_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    query TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    action TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_scope_time
                    ON memory_events(scope, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_scope_status
                    ON memory_facts(scope, status, updated_at DESC);
                """
            )

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    def _terms(self, query: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for term in _TERM_RE.findall(str(query or "").lower()):
            if any("\u4e00" <= char <= "\u9fff" for char in term):
                pieces = [term] if len(term) <= 4 else []
                pieces.extend(term[index : index + 2] for index in range(len(term) - 1))
            else:
                pieces = [term]
            for piece in pieces:
                if piece not in seen:
                    seen.add(piece)
                    terms.append(piece)
        return terms[: self.max_query_terms]

    def record_turn(
        self,
        *,
        conversation_id: str,
        scope: str,
        role: str,
        content: str,
        source_id: str,
        created_at: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        text = str(content or "").strip()
        if not text or not source_id:
            return False
        stamp = int(created_at or time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memory_turns
                    (turn_id, conversation_id, scope, role, content, created_at, source_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid.uuid4().hex, str(conversation_id), str(scope), str(role or "user"), text, stamp, str(source_id), self._json(metadata)),
            )
            return cursor.rowcount > 0

    def record_event(
        self,
        *,
        conversation_id: str,
        scope: str,
        text: str,
        source_id: str,
        layer: str = "episodic",
        created_at: int | None = None,
        importance: float = 0.25,
        confidence: float = 0.5,
        expires_at: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        body = str(text or "").strip()
        if not body or not source_id:
            return False
        stamp = int(created_at or time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memory_events
                    (event_id, conversation_id, scope, text, layer, created_at,
                     importance, confidence, expires_at, source_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    str(conversation_id),
                    str(scope),
                    body,
                    str(layer or "episodic"),
                    stamp,
                    max(0.0, min(1.0, float(importance))),
                    max(0.0, min(1.0, float(confidence))),
                    max(0, int(expires_at or 0)),
                    str(source_id),
                    self._json(metadata),
                ),
            )
            return cursor.rowcount > 0

    def upsert_fact(
        self,
        *,
        conversation_id: str,
        scope: str,
        fact_key: str,
        text: str,
        source_id: str = "",
        importance: float = 0.85,
        confidence: float = 0.9,
        expires_at: int = 0,
        now: int | None = None,
    ) -> None:
        stamp = int(now or time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_facts
                    (fact_id, conversation_id, scope, fact_key, text, created_at, updated_at,
                     importance, confidence, expires_at, source_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(scope, fact_key) DO UPDATE SET
                    text=excluded.text,
                    conversation_id=excluded.conversation_id,
                    updated_at=excluded.updated_at,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    expires_at=excluded.expires_at,
                    source_id=excluded.source_id,
                    status='active'
                """,
                (
                    uuid.uuid4().hex,
                    str(conversation_id),
                    str(scope),
                    str(fact_key),
                    str(text).strip(),
                    stamp,
                    stamp,
                    max(0.0, min(1.0, float(importance))),
                    max(0.0, min(1.0, float(confidence))),
                    max(0, int(expires_at or 0)),
                    str(source_id),
                ),
            )

    def search(self, *, query: str, scope: str, limit: int = 5, now: int | None = None) -> list[MemoryRecord]:
        terms = self._terms(query)
        if not terms:
            return []
        stamp = int(now or time.time())
        self.expire(now=stamp)
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                f"SELECT * FROM memory_events WHERE scope = ? AND status = 'active' ORDER BY created_at DESC LIMIT {self.event_scan_limit}",
                (str(scope),),
            ).fetchall()
            fact_rows = conn.execute(
                f"SELECT * FROM memory_facts WHERE scope = ? AND status = 'active' ORDER BY updated_at DESC LIMIT {self.fact_scan_limit}",
                (str(scope),),
            ).fetchall()
        candidates: list[MemoryRecord] = []
        for row in event_rows:
            candidates.append(self._score_event(row, terms, stamp))
        for row in fact_rows:
            candidates.append(self._score_fact(row, terms, stamp))
        candidates = [item for item in candidates if item.score > 0.0]
        candidates.sort(key=lambda item: (item.score, item.importance, item.created_at), reverse=True)
        selected = candidates[: max(1, min(self.max_results, int(limit or 5)))]
        if selected:
            with self._lock, self._connect() as conn:
                ids = [item.record_id for item in selected]
                conn.executemany(
                    "UPDATE memory_events SET last_used_at = ? WHERE event_id = ?",
                    [(stamp, item) for item in ids if item.startswith("event:")],
                )
                conn.executemany(
                    "UPDATE memory_facts SET last_used_at = ? WHERE fact_id = ?",
                    [(stamp, item.record_id.removeprefix("fact:")) for item in selected if item.record_id.startswith("fact:")],
                )
        return selected

    def _score_event(self, row: sqlite3.Row, terms: list[str], now: int) -> MemoryRecord:
        text = str(row["text"] or "")
        overlap = sum(1 for term in terms if term in text.lower())
        age_days = max(0.0, (now - int(row["created_at"] or now)) / 86400.0)
        decay = 0.5 ** (age_days / 30.0)
        score = overlap * 0.55 + float(row["importance"] or 0.0) * decay * 0.45
        return MemoryRecord(
            record_id=f"event:{row['event_id']}",
            layer=str(row["layer"] or "episodic"),
            scope=str(row["scope"] or ""),
            text=text,
            created_at=int(row["created_at"] or 0),
            score=score,
            importance=float(row["importance"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            source_id=str(row["source_id"] or ""),
            metadata=self._loads(row["metadata_json"]),
        )

    def _score_fact(self, row: sqlite3.Row, terms: list[str], now: int) -> MemoryRecord:
        text = str(row["text"] or "")
        overlap = sum(1 for term in terms if term in text.lower())
        score = overlap * 0.7 + float(row["importance"] or 0.0) * float(row["confidence"] or 0.0) * 0.3
        return MemoryRecord(
            record_id=f"fact:{row['fact_id']}",
            layer="fact",
            scope=str(row["scope"] or ""),
            text=text,
            created_at=int(row["updated_at"] or 0),
            score=score,
            importance=float(row["importance"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            source_id=str(row["source_id"] or ""),
            metadata={"fact_key": str(row["fact_key"] or "")},
        )

    def expire(self, *, now: int | None = None) -> int:
        stamp = int(now or time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_events SET status='expired'
                WHERE status='active' AND expires_at > 0 AND expires_at <= ?
                """,
                (stamp,),
            )
            return int(cursor.rowcount or 0)

    def forget(self, *, query: str, scope: str, purge: bool = False, now: int | None = None) -> tuple[int, int]:
        terms = self._terms(query)
        if not terms:
            return 0, 0
        stamp = int(now or time.time())
        forgotten = 0
        purged = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, text, layer FROM memory_events WHERE scope = ? AND status = 'active'",
                (str(scope),),
            ).fetchall()
            fact_rows = conn.execute(
                "SELECT fact_id, text FROM memory_facts WHERE scope = ? AND status = 'active'",
                (str(scope),),
            ).fetchall()
            event_ids = [str(row["event_id"]) for row in rows if all(term in str(row["text"]).lower() for term in terms)]
            fact_ids = [str(row["fact_id"]) for row in fact_rows if all(term in str(row["text"]).lower() for term in terms)]
            if purge:
                conn.executemany("DELETE FROM memory_events WHERE event_id = ?", [(item,) for item in event_ids])
                conn.executemany("DELETE FROM memory_facts WHERE fact_id = ?", [(item,) for item in fact_ids])
                purged = len(event_ids) + len(fact_ids)
            else:
                conn.executemany("UPDATE memory_events SET status='forgotten' WHERE event_id = ?", [(item,) for item in event_ids])
                conn.executemany("UPDATE memory_facts SET status='forgotten', updated_at=? WHERE fact_id = ?", [(stamp, item) for item in fact_ids])
                forgotten = len(event_ids) + len(fact_ids)
            conn.executemany(
                "INSERT INTO memory_forget_log (forget_id, scope, query, record_id, layer, created_at, action) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (uuid.uuid4().hex, str(scope), str(query), item, "event", stamp, "purge" if purge else "forget")
                    for item in event_ids
                ]
                + [
                    (uuid.uuid4().hex, str(scope), str(query), item, "fact", stamp, "purge" if purge else "forget")
                    for item in fact_ids
                ],
            )
        return forgotten, purged

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            return {
                "turns": int(conn.execute("SELECT COUNT(*) FROM memory_turns").fetchone()[0]),
                "active_events": int(conn.execute("SELECT COUNT(*) FROM memory_events WHERE status='active'").fetchone()[0]),
                "active_facts": int(conn.execute("SELECT COUNT(*) FROM memory_facts WHERE status='active'").fetchone()[0]),
                "forgotten": int(conn.execute("SELECT COUNT(*) FROM memory_forget_log").fetchone()[0]),
            }

    @staticmethod
    def _loads(value: Any) -> dict[str, Any]:
        try:
            parsed = json.loads(str(value or "{}"))
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
