"""Read-only importer from the previous Shinku SQLite memory database."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .store import MemoryStore

__all__ = ["migrate_legacy_sqlite"]


def migrate_legacy_sqlite(source: str | Path, target: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Import old records without changing the source database.

    Semantic summaries are imported as low-priority historical events.  Their
    ``stable_facts_json`` is retained in metadata, never promoted to facts.
    Vector indexes are intentionally not copied; the independent store must
    rebuild them from cleaned records later.
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    report: dict[str, Any] = {"source": str(source_path), "target": str(target), "dry_run": dry_run, "imported": {}}
    read_uri = f"file:{source_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(read_uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        store = None if dry_run else MemoryStore(target)
        report["imported"]["chat_messages"] = _import_messages(conn, store)
        report["imported"]["memory_summaries"] = _import_summaries(conn, store)
        report["imported"]["memory_semantic_summaries"] = _import_semantic(conn, store)
        report["imported"]["memory_lessons"] = _import_lessons(conn, store)
        report["imported"]["context_states"] = _import_states(conn, store)
    if store is not None:
        report["stats"] = store.stats()
    return report


def _rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    try:
        return list(conn.execute(f"SELECT * FROM {table}"))
    except sqlite3.OperationalError:
        return []


def _scope(row: sqlite3.Row) -> str:
    profile = str(row["profile_user_id"] or "unknown")
    session = str(row["session_id"] if "session_id" in row.keys() else profile or profile)
    if profile == "master":
        # The standalone runtime treats the single owner as a private scope.
        # A future multi-owner deployment can remap this during import.
        return "private:master"
    if profile.startswith("qq_group_shared_"):
        return f"group:{profile.removeprefix('qq_group_shared_')}"
    if session.startswith("qq_group_shared_"):
        return f"group:{session.removeprefix('qq_group_shared_')}"
    return f"legacy:{profile}:{session}"


def _write_event(store: MemoryStore | None, **kwargs: Any) -> None:
    if store is not None:
        store.record_event(**kwargs)


def _import_messages(conn: sqlite3.Connection, store: MemoryStore | None) -> int:
    count = 0
    for row in _rows(conn, "chat_messages"):
        text = str(row["content"] or "").strip()
        source = str(row["source_id"] or "").strip()
        if not text or not source:
            continue
        _write_event(
            store,
            conversation_id=str(row["session_id"] or "legacy"),
            scope=_scope(row),
            text=text,
            source_id=f"legacy:message:{source}",
            layer="legacy_raw",
            created_at=int(row["timestamp"] or time.time()),
            importance=0.12,
            confidence=0.35,
            metadata={"role": str(row["role"] or "unknown"), "profile_user_id": str(row["profile_user_id"] or "")},
        )
        count += 1
    return count


def _import_summaries(conn: sqlite3.Connection, store: MemoryStore | None) -> int:
    count = 0
    for row in _rows(conn, "memory_summaries"):
        text = str(row["diary_summary"] or "").strip()
        source = str(row["summary_id"] or "").strip()
        if not text or not source:
            continue
        _write_event(
            store,
            conversation_id=str(row["session_id"] or "legacy"),
            scope=_scope(row),
            text=text,
            source_id=f"legacy:summary:{source}",
            layer="legacy_summary",
            created_at=int(row["timestamp"] or time.time()),
            importance=min(0.28, float(row["importance"] or 0.4) * 0.35),
            confidence=0.45,
            metadata={"legacy_summary_id": source},
        )
        count += 1
    return count


def _import_semantic(conn: sqlite3.Connection, store: MemoryStore | None) -> int:
    count = 0
    for row in _rows(conn, "memory_semantic_summaries"):
        body = str(row["semantic_summary"] or "").strip()
        source = str(row["semantic_id"] or "").strip()
        if not body or not source:
            continue
        metadata = {
            "legacy_semantic_id": source,
            "legacy_stable_facts": _loads(row["stable_facts_json"]),
            "legacy_topics": _loads(row["recurring_topics_json"]),
            "legacy_people": _loads(row["important_people_json"]),
        }
        _write_event(
            store,
            conversation_id=str(row["session_id"] or "legacy"),
            scope=_scope(row),
            text=body,
            source_id=f"legacy:semantic:{source}",
            layer="legacy_semantic",
            created_at=int(row["timestamp"] or time.time()),
            importance=min(0.32, float(row["importance"] or 0.5) * 0.35),
            confidence=0.4,
            metadata=metadata,
        )
        count += 1
    return count


def _import_lessons(conn: sqlite3.Connection, store: MemoryStore | None) -> int:
    count = 0
    for row in _rows(conn, "memory_lessons"):
        text = str(row["correct_fact"] or "").strip()
        source = str(row["lesson_id"] or "").strip()
        if not text or not source:
            continue
        _write_event(
            store,
            conversation_id=str(row["session_id"] or "legacy"),
            scope=_scope(row),
            text=text,
            source_id=f"legacy:lesson:{source}",
            layer="legacy_lesson",
            created_at=int(row["updated_at"] or row["created_at"] or time.time()),
            importance=min(0.55, float(row["importance"] or 0.6) * 0.7),
            confidence=0.55,
            metadata={"lesson_type": str(row["lesson_type"] or ""), "trigger": str(row["trigger_text"] or "")},
        )
        count += 1
    return count


def _import_states(conn: sqlite3.Connection, store: MemoryStore | None) -> int:
    count = 0
    for row in _rows(conn, "context_states"):
        text = str(row["state_text"] or "").strip()
        source = str(row["state_id"] or "").strip()
        if not text or not source:
            continue
        _write_event(
            store,
            conversation_id=str(row["session_id"] if "session_id" in row.keys() else row["profile_user_id"] or "legacy"),
            scope=_scope(row),
            text=text,
            source_id=f"legacy:state:{source}",
            layer="legacy_state",
            created_at=int(row["updated_at"] or time.time()),
            importance=0.35,
            confidence=float(row["confidence"] or 0.5),
            expires_at=int(row["expires_at"] or 0),
            metadata={"topic_key": str(row["topic_key"] or "")},
        )
        count += 1
    return count


def _loads(value: Any) -> Any:
    try:
        return json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
