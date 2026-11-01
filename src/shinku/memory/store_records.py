"""Store responsibilities: records."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from .models import MemoryRecord

from .store_models import MemoryStoreRecordsDependencies


def record_turn(
    *,
    conversation_id: str,
    scope: str,
    role: str,
    content: str,
    source_id: str,
    created_at: int | None=None,
    metadata: dict[str, Any] | None=None,
    deps: MemoryStoreRecordsDependencies,
) -> bool:
    text = str(content or "").strip()
    if not text or not source_id:
        return False
    stamp = int(created_at or time.time())
    with deps.lock(), deps.connect() as conn:
        cursor = conn.execute(
            """
                INSERT OR IGNORE INTO memory_turns
                    (turn_id, conversation_id, scope, role, content, created_at, source_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
            (uuid.uuid4().hex, str(conversation_id), str(scope), str(role or "user"), text, stamp, str(source_id), deps.json(metadata)),
        )
        return cursor.rowcount > 0


def record_routing_audit(
    *,
    conversation_id: str,
    scope: str,
    created_at: int,
    decision: dict[str, Any],
    retrieved_count: int=0,
    injected_chars: int=0,
    deps: MemoryStoreRecordsDependencies,
) -> bool:
    """Persist one cheap, non-prompt audit row for memory routing."""

    data = dict(decision or {})
    with deps.lock(), deps.connect() as conn:
        cursor = conn.execute(
            """INSERT INTO memory_routing_audit
                   (audit_id, conversation_id, scope, created_at, history_mode,
                    need_retrieval, used_model, confidence, reason,
                    retrieved_count, injected_chars, decision_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                str(conversation_id),
                str(scope),
                int(created_at),
                str(data.get("history_mode") or "none"),
                1 if bool(data.get("need_retrieval")) else 0,
                1 if bool(data.get("used_model")) else 0,
                max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
                str(data.get("reason") or "")[:240],
                max(0, int(retrieved_count or 0)),
                max(0, int(injected_chars or 0)),
                deps.json(data),
            ),
        )
        return cursor.rowcount > 0


def recent_routing_audits(
    *,
    scope: str | None=None,
    limit: int=20,
    deps: MemoryStoreRecordsDependencies,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    clause = ""
    if scope:
        clause = " WHERE scope = ?"
        params.append(str(scope))
    params.append(max(1, min(200, int(limit or 20))))
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM memory_routing_audit{clause} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "audit_id": str(row["audit_id"] or ""),
                "conversation_id": str(row["conversation_id"] or ""),
                "scope": str(row["scope"] or ""),
                "created_at": int(row["created_at"] or 0),
                "history_mode": str(row["history_mode"] or "none"),
                "need_retrieval": bool(row["need_retrieval"]),
                "used_model": bool(row["used_model"]),
                "confidence": float(row["confidence"] or 0.0),
                "reason": str(row["reason"] or ""),
                "retrieved_count": int(row["retrieved_count"] or 0),
                "injected_chars": int(row["injected_chars"] or 0),
                "decision": deps.loads(row["decision_json"]),
            }
        )
    return result


def record_event(
    *,
    conversation_id: str,
    scope: str,
    text: str,
    source_id: str,
    layer: str='episodic',
    created_at: int | None=None,
    importance: float=0.25,
    confidence: float=0.5,
    expires_at: int=0,
    metadata: dict[str, Any] | None=None,
    deps: MemoryStoreRecordsDependencies,
) -> bool:
    body = str(text or "").strip()
    if not body or not source_id:
        return False
    stamp = int(created_at or time.time())
    with deps.lock(), deps.connect() as conn:
        if deps.lifecycle().revoked(conn, scope=str(scope), source_id=str(source_id), text=body):
            return False
        meta = metadata if isinstance(metadata, dict) else {}
        parents = meta.get("source_record_ids", [])
        if not isinstance(parents, list) or len(parents) > 128:
            return False
        hashes = meta.get("source_content_hashes", {})
        if not isinstance(hashes, dict):
            return False
        for parent_id in parents:
            try:
                parent_type, parent_bare = deps.lifecycle()._identity(str(parent_id))
            except ValueError:
                return False
            parent = deps.lifecycle()._record(conn, parent_type, parent_bare)
            if parent is None or parent["status"] != "active" or str(parent["scope"]) != str(scope):
                return False
            expected = hashes.get(str(parent_id))
            if expected and hashlib.sha256(str(parent["text"]).encode("utf-8")).hexdigest() != expected:
                return False
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
                deps.json(metadata),
            ),
        )
        if cursor.rowcount > 0 and deps.lifecycle().policy.enabled:
            row = conn.execute("SELECT * FROM memory_events WHERE source_id=?", (str(source_id),)).fetchone()
            deps.lifecycle()._enroll(conn, "event", row, int(deps.lifecycle().wall_time()))
        return cursor.rowcount > 0


def event_by_source_id(
    *,
    source_id: str,
    scope: str,
    deps: MemoryStoreRecordsDependencies,
) -> MemoryRecord | None:
    """Read one active event by its stable source key inside one scope."""

    source = str(source_id or "").strip()
    selected_scope = str(scope or "").strip()
    if not source or not selected_scope:
        return None
    with deps.lock(), deps.connect() as conn:
        row = conn.execute(
            "SELECT * FROM memory_events WHERE source_id = ? AND scope = ? AND status = 'active' LIMIT 1",
            (source, selected_scope),
        ).fetchone()
    return deps.event_record(row, score=1.0) if row is not None else None


def prune_agent_task_details(
    *,
    scope: str,
    max_records: int,
    deps: MemoryStoreRecordsDependencies,
) -> int:
    """Forget the oldest active detail archives past the configured cap."""

    selected_scope = str(scope or "").strip()
    cap = max(1, int(max_records or 1))
    if not selected_scope:
        return 0
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute(
            "SELECT event_id FROM memory_events "
            "WHERE scope = ? AND status = 'active' AND layer = 'episodic' "
            "AND source_id LIKE 'owner:agent-task-detail:%' "
            "ORDER BY created_at DESC, rowid DESC",
            (selected_scope,),
        ).fetchall()
        forgotten_ids = [str(row["event_id"]) for row in rows[cap:]]
        if not forgotten_ids:
            return 0
        conn.executemany(
            "UPDATE memory_events SET status = 'forgotten' WHERE event_id = ?",
            [(item,) for item in forgotten_ids],
        )
        conn.executemany(
            "DELETE FROM memory_embeddings WHERE record_type = 'event' AND record_id = ?",
            [(item,) for item in forgotten_ids],
        )
        return len(forgotten_ids)


def promote_scope_to_owner(
    *,
    source_scope: str,
    owner_scope: str='private:master',
    deps: MemoryStoreRecordsDependencies,
) -> dict[str, int]:
    """Copy an already isolated private scope into the owner library.

        This is intentionally limited to a private scope whose identity is
        configured as the owner.  It never promotes group spectators, and the
        deterministic source prefix makes startup migration idempotent.
        """

    source_scope = str(source_scope or "").strip()
    owner_scope = str(owner_scope or "").strip()
    if not source_scope or not owner_scope or source_scope == owner_scope:
        return {"events": 0, "facts": 0}
    copied_events = 0
    copied_facts = 0
    with deps.lock(), deps.connect() as conn:
        event_rows = conn.execute(
            "SELECT * FROM memory_events WHERE scope = ? AND status = 'active'",
            (source_scope,),
        ).fetchall()
        for row in event_rows:
            source_id = f"owner:import:{row['source_id']}"
            if deps.lifecycle().revoked(conn, scope=owner_scope, source_id=source_id, text=row["text"]):
                continue
            metadata = deps.loads(row["metadata_json"])
            metadata.update({"owner_memory": True, "owner_source_scope": source_scope, "owner_import": True})
            cursor = conn.execute(
                """
                    INSERT OR IGNORE INTO memory_events
                        (event_id, conversation_id, scope, text, layer, created_at,
                         last_used_at, importance, confidence, expires_at, status,
                         source_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                (
                    uuid.uuid5(uuid.NAMESPACE_URL, source_id).hex,
                    "master",
                    owner_scope,
                    str(row["text"] or ""),
                    str(row["layer"] or "episodic"),
                    int(row["created_at"] or 0),
                    int(row["last_used_at"] or 0),
                    float(row["importance"] or 0.0),
                    float(row["confidence"] or 0.0),
                    int(row["expires_at"] or 0),
                    source_id,
                    deps.json(metadata),
                ),
            )
            copied_events += max(0, int(cursor.rowcount))
        fact_rows = conn.execute(
            "SELECT * FROM memory_facts WHERE scope = ? AND status = 'active'",
            (source_scope,),
        ).fetchall()
        for row in fact_rows:
            key = str(row["fact_key"] or "").strip()
            if not key:
                continue
            source_id = f"owner:import:{row['source_id']}"
            if deps.lifecycle().revoked(conn, scope=owner_scope, source_id=source_id, text=row["text"]):
                continue
            prior = conn.execute("SELECT status,updated_at FROM memory_facts WHERE scope=? AND fact_key=?", (owner_scope, key)).fetchone()
            if prior is not None and (prior["status"] != "active" or prior["updated_at"] >= row["updated_at"]):
                continue
            cursor = conn.execute(
                """
                    INSERT INTO memory_facts
                        (fact_id, conversation_id, scope, fact_key, text, created_at,
                         updated_at, last_used_at, importance, confidence, status,
                         expires_at, source_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                    ON CONFLICT(scope, fact_key) DO UPDATE SET
                        text=excluded.text,
                        conversation_id=excluded.conversation_id,
                        updated_at=excluded.updated_at,
                        last_used_at=excluded.last_used_at,
                        importance=excluded.importance,
                        confidence=excluded.confidence,
                        status='active',
                        expires_at=excluded.expires_at,
                        source_id=excluded.source_id,
                        metadata_json=excluded.metadata_json
                    """,
                (
                    uuid.uuid5(uuid.NAMESPACE_URL, f"owner:fact:{source_scope}:{key}").hex,
                    "master",
                    owner_scope,
                    key,
                    str(row["text"] or ""),
                    int(row["created_at"] or 0),
                    int(row["updated_at"] or 0),
                    int(row["last_used_at"] or 0),
                    float(row["importance"] or 0.0),
                    float(row["confidence"] or 0.0),
                    int(row["expires_at"] or 0),
                    f"owner:import:{row['source_id']}",
                    row["metadata_json"],
                ),
            )
            copied_facts += max(0, int(cursor.rowcount))
    return {"events": copied_events, "facts": copied_facts}


def upsert_fact(
    *,
    conversation_id: str,
    scope: str,
    fact_key: str,
    text: str,
    source_id: str='',
    importance: float=0.85,
    confidence: float=0.9,
    expires_at: int=0,
    now: int | None=None,
    metadata: dict[str, Any] | None=None,
    deps: MemoryStoreRecordsDependencies,
) -> None:
    stamp = int(now or time.time())
    with deps.lock(), deps.connect() as conn:
        if deps.lifecycle().revoked(conn, scope=str(scope), source_id=str(source_id), text=str(text).strip()):
            return
        conn.execute(
            """
                INSERT INTO memory_facts
                    (fact_id, conversation_id, scope, fact_key, text, created_at, updated_at,
                     importance, confidence, expires_at, source_id, metadata_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(scope, fact_key) DO UPDATE SET
                    text=excluded.text,
                    conversation_id=excluded.conversation_id,
                    updated_at=excluded.updated_at,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    expires_at=excluded.expires_at,
                    source_id=excluded.source_id,
                    metadata_json=excluded.metadata_json,
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
                deps.json(metadata or {}),
            ),
        )

        if deps.lifecycle().policy.enabled:
            row = conn.execute("SELECT * FROM memory_facts WHERE scope=? AND fact_key=?", (str(scope), str(fact_key))).fetchone()
            deps.lifecycle()._enroll(conn, "fact", row, stamp)
