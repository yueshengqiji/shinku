"""Store responsibilities: maintenance."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid


from .store_models import MemoryStoreMaintenanceDependencies


def rollup_daily_summaries(
    *,
    scope: str,
    now: int | None=None,
    max_lines: int=60,
    source_layers: set[str] | tuple[str, ...]=('observation', 'episodic'),
    deps: MemoryStoreMaintenanceDependencies,
) -> int:
    """Create compact local summaries for completed conversation days.

        This is intentionally deterministic: it does not call a model and it
        never replaces the underlying observation evidence.
        """

    stamp = int(now or time.time())
    current_day = time.strftime("%Y-%m-%d", time.localtime(stamp))
    layers = tuple(dict.fromkeys(str(item).strip() for item in source_layers if str(item).strip()))
    if not layers:
        return 0
    layer_clause = " AND layer IN (" + ",".join("?" for _ in layers) + ")"
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute(
            f"""SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active'
                     AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}
                   ORDER BY created_at ASC""",
            (str(scope), *layers),
        ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(int(row["created_at"] or stamp)))
        if day < current_day:
            grouped.setdefault(day, []).append(row)
    created = 0
    for day, day_rows in grouped.items():
        lines: list[str] = []
        included_rows = day_rows[: max(1, min(100, int(max_lines or 60)))]
        for row in included_rows:
            metadata = deps.loads(row["metadata_json"])
            role = str(metadata.get("role") or "").strip().lower()
            sender = str(
                metadata.get("sender_name")
                or metadata.get("sender_id")
                or ("助手" if role == "assistant" else "用户" if role == "user" else "群成员")
            ).strip()
            text = str(row["text"] or "").replace("\n", " ").strip()
            if text:
                lines.append(f"{sender}: {text[:160]}")
        if not lines:
            continue
        scope_label = (
            "群聊摘要" if str(scope).startswith("group:")
            else "私聊摘要" if str(scope).startswith("private:")
            else "会话摘要"
        )
        summary = f"{day} {scope_label}（{len(day_rows)} 条记录）\n" + "\n".join(lines)
        if deps.record_event(
            conversation_id=str(day_rows[0]["conversation_id"] or ""),
            scope=scope,
            text=summary[:12000],
            source_id=f"summary:{scope}:{day}",
            layer="summary",
            created_at=int(day_rows[-1]["created_at"] or stamp),
            importance=0.62,
            confidence=0.72,
            expires_at=0,
            metadata={"summary_day": day, "source": "local_daily_rollup", "observation_count": len(day_rows), "source_record_ids": [f"event:{row['event_id']}" for row in included_rows], "source_content_hashes": {f"event:{row['event_id']}": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() for row in included_rows}},
        ):
            created += 1
    return created


def expire(*, now: int | None=None, deps: MemoryStoreMaintenanceDependencies) -> int:
    stamp = int(now or time.time())
    with deps.lock(), deps.connect() as conn:
        lifecycle_clause = " AND NOT (layer IN ('observation','episodic','legacy_raw','legacy_lesson') AND source_id NOT LIKE 'owner:agent-task-%')" if deps.lifecycle().policy.mode == "active" else ""
        cursor = conn.execute(
            f"""
                UPDATE memory_events SET status='expired'
                WHERE status='active' AND expires_at > 0 AND expires_at <= ?{lifecycle_clause}
                """,
            (stamp,),
        )
        facts = conn.execute("UPDATE memory_facts SET status='expired' WHERE status='active' AND expires_at>0 AND expires_at<=?", (stamp,))
        return int(cursor.rowcount or 0) + int(facts.rowcount or 0)


def forget(
    *,
    query: str,
    scope: str,
    purge: bool=False,
    now: int | None=None,
    record_id: str='',
    deps: MemoryStoreMaintenanceDependencies,
) -> tuple[int, int]:
    terms = deps.terms(query)
    exact_kind, exact_bare = deps.lifecycle()._identity(record_id) if record_id else ("", "")
    if not terms and not record_id:
        return 0, 0
    stamp = int(now or time.time())
    forgotten = 0
    purged = 0
    with deps.lock(), deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT event_id,text,layer,metadata_json,source_id FROM memory_events WHERE scope=? AND status IN ('active','expired')",
            (str(scope),),
        ).fetchall()
        fact_rows = conn.execute(
            "SELECT fact_id,text,source_id FROM memory_facts WHERE scope=? AND status IN ('active','expired')",
            (str(scope),),
        ).fetchall()
        # A summary/retired event may no longer contain the forgotten
        # term. Its separately retained scoped source turn proves linkage.
        turn_matches = []
        if terms and not record_id:
            turn_matches = [row for row in conn.execute("SELECT source_id,content FROM memory_turns WHERE scope=?", (str(scope),))
                            if all(term in str(row["content"]).lower() for term in terms)]
            for turn in turn_matches:
                deps.lifecycle().revoke(conn, scope=str(scope), source_id=turn["source_id"], text=turn["content"], now=stamp)
        source_ids = {str(row["source_id"]) for row in turn_matches}
        event_ids = [str(row["event_id"]) for row in rows if
                     exact_kind == "event" and row["event_id"] == exact_bare] if record_id else [
                     str(row["event_id"]) for row in rows if all(term in str(row["text"]).lower() for term in terms) or row["source_id"] in source_ids]
        fact_ids = [str(row["fact_id"]) for row in fact_rows if exact_kind == "fact" and row["fact_id"] == exact_bare] if record_id else [
                    str(row["fact_id"]) for row in fact_rows if all(term in str(row["text"]).lower() for term in terms) or row["source_id"] in source_ids]
        matched_event_ids = set(event_ids)
        matched_task_ids = {
            str(metadata.get("task_id") or "")
            for row in rows
            if str(row["event_id"]) in matched_event_ids
            for metadata in (deps.loads(row["metadata_json"]),)
            if metadata.get("record_type") in {"agent_task_summary", "agent_task_detail"}
            and str(metadata.get("task_id") or "").strip()
        }
        if matched_task_ids:
            linked_ids = {
                str(row["event_id"])
                for row in rows
                for metadata in (deps.loads(row["metadata_json"]),)
                if metadata.get("record_type") in {"agent_task_summary", "agent_task_detail"}
                and str(metadata.get("task_id") or "") in matched_task_ids
            }
            event_ids = list(dict.fromkeys((*event_ids, *linked_ids)))
        # Source-linked summaries may omit the original query. Remove
        # those derivatives by provenance, not by similarity or scope mixing.
        while True:
            known = set(event_ids)
            known_records = {f"event:{item}" for item in event_ids} | {f"fact:{item}" for item in fact_ids}
            linked = [str(row["event_id"]) for row in rows
                      for metadata in (deps.loads(row["metadata_json"]),)
                      for parents in (metadata.get("source_record_ids", []),)
                      if (isinstance(parents, list) and known_records.intersection(str(item) for item in parents))
                      or (known_records and metadata.get("source") == "local_daily_rollup" and not parents)]
            event_ids = list(dict.fromkeys((*event_ids, *linked)))
            if set(event_ids) == known:
                break
        for record_type, selected_ids, selected_rows, pk in (("event", event_ids, rows, "event_id"), ("fact", fact_ids, fact_rows, "fact_id")):
            for row in selected_rows:
                rid = str(row[pk])
                if rid not in selected_ids:
                    continue
                state = conn.execute("SELECT summary_source_hash FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid)).fetchone()
                deps.lifecycle().revoke(conn, scope=str(scope), source_id=row["source_id"], text=row["text"], now=stamp, original_hash=state["summary_source_hash"] if state else "")
                conn.execute("DELETE FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid))
                conn.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type=? AND record_id=?", (record_type, rid))
                conn.execute("DELETE FROM memory_embeddings WHERE record_type=? AND record_id=?", (record_type, rid))
                if record_type == "event":
                    deps.set_l1_semantic_cache({key: value for key, value in deps.l1_semantic_cache().items() if key[1] != rid})
        if purge:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_task_experience_versions'").fetchone():
                conn.executemany("UPDATE memory_task_experience_versions SET payload='{}' WHERE scope=? AND source_id=?",
                    [(str(scope), row["source_id"]) for row in rows if row["event_id"] in event_ids])
            conn.executemany("DELETE FROM memory_events WHERE event_id = ?", [(item,) for item in event_ids])
            conn.executemany("DELETE FROM memory_facts WHERE fact_id = ?", [(item,) for item in fact_ids])
            conn.executemany("DELETE FROM memory_embeddings WHERE record_type = 'event' AND record_id = ?", [(item,) for item in event_ids])
            conn.executemany("DELETE FROM memory_embeddings WHERE record_type = 'fact' AND record_id = ?", [(item,) for item in fact_ids])
            purged = len(event_ids) + len(fact_ids)
        else:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_task_experience_versions'").fetchone():
                conn.executemany("UPDATE memory_task_experience_versions SET payload='{}' WHERE scope=? AND source_id=?",
                    [(str(scope), row["source_id"]) for row in rows if row["event_id"] in event_ids])
            conn.executemany("UPDATE memory_events SET status='forgotten' WHERE event_id = ?", [(item,) for item in event_ids])
            conn.executemany("UPDATE memory_facts SET status='forgotten', updated_at=? WHERE fact_id = ?", [(stamp, item) for item in fact_ids])
            forgotten = len(event_ids) + len(fact_ids)
        conn.executemany(
            "INSERT INTO memory_forget_log (forget_id, scope, query, record_id, layer, created_at, action) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (uuid.uuid4().hex, str(scope), "sha256:" + hashlib.sha256(str(query).encode("utf-8")).hexdigest(), item, "event", stamp, "purge" if purge else "forget")
                for item in event_ids
            ]
            + [
                (uuid.uuid4().hex, str(scope), "sha256:" + hashlib.sha256(str(query).encode("utf-8")).hexdigest(), item, "fact", stamp, "purge" if purge else "forget")
                for item in fact_ids
            ],
        )
    with deps.lock():
        deps.clear_derived_caches(scope=str(scope), record_ids={*(f"event:{item}" for item in event_ids), *(f"fact:{item}" for item in fact_ids)})
    return forgotten, purged


def reset_all(*, deps: MemoryStoreMaintenanceDependencies) -> dict[str, int]:
    """Clear every memory record while keeping the database schema.

        This is intentionally a low-level destructive operation.  The private
        command layer is responsible for authorization and explicit confirmation.
        """

    tables = (
        "memory_turns",
        "memory_events",
        "memory_facts",
        "memory_embeddings",
        "memory_forget_log",
        "memory_routing_audit",
    )
    with deps.lock(), deps.connect() as conn:
        deps.l1_semantic_cache().clear()
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        stamp = int(time.time())
        for table in ("memory_events", "memory_facts"):
            for row in conn.execute(f"SELECT scope,source_id,text FROM {table}").fetchall():
                deps.lifecycle().revoke(conn, scope=row["scope"], source_id=row["source_id"], text=row["text"], now=stamp)
        # Compact revocations intentionally survive a reset, so importing
        # an old backup cannot undo the user's explicit forgetting.
        for table in ("memory_lifecycle", "memory_lifecycle_evidence", "memory_lifecycle_audit", "memory_lifecycle_activity", "memory_lifecycle_scopes"):
            conn.execute(f"DELETE FROM {table}")
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_task_experience_versions'").fetchone():
            conn.execute("UPDATE memory_task_experience_versions SET payload='{}'")
    with deps.lock():
        deps.clear_derived_caches(scope="", record_ids=set())
    return counts


def stats(*, deps: MemoryStoreMaintenanceDependencies) -> dict[str, int]:
    with deps.lock(), deps.connect() as conn:
        return {
            "turns": int(conn.execute("SELECT COUNT(*) FROM memory_turns").fetchone()[0]),
            "active_events": int(conn.execute("SELECT COUNT(*) FROM memory_events WHERE status='active'").fetchone()[0]),
            "active_facts": int(conn.execute("SELECT COUNT(*) FROM memory_facts WHERE status='active'").fetchone()[0]),
            "forgotten": int(conn.execute("SELECT COUNT(*) FROM memory_forget_log").fetchone()[0]),
            "routing_audits": int(conn.execute("SELECT COUNT(*) FROM memory_routing_audit").fetchone()[0]),
            "embedding_vectors": int(conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]),
        }
