"""Atomic card + dedupe + correction history in the existing memory SQLite."""
from __future__ import annotations

import hashlib
import json
import uuid

from shinku.agent_runtime.contracts import canonical_hash


class ExperienceStore:
    def __init__(self, store, *, max_versions=4096):
        self.store = store
        self.max_versions = max(1, min(16384, int(max_versions)))
        with store._lock, store._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memory_task_experience_versions (
                projection_hash TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                scope TEXT NOT NULL, created INTEGER NOT NULL, payload TEXT NOT NULL)""")

    def commit(self, snapshot, text, *, scope, now, check_current):
        source_id = "owner:agent-task:" + snapshot["task_id"]
        invalidated = set()
        with self.store._lock, self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not check_current():
                return {"status": "rejected", "reason": "experience_source_changed"}
            if self.store.lifecycle.revoked(db, scope=scope, source_id=source_id, text=text):
                return {"status": "rejected", "reason": "experience_revoked"}
            old = db.execute("SELECT * FROM memory_events WHERE source_id=?", (source_id,)).fetchone()
            meta = self.store._loads(old["metadata_json"]) if old else {}
            if old:
                if (old["scope"] != scope or meta.get("experience_schema") != 1
                        or meta.get("source_scope") != snapshot["scope"]):
                    return {"status": "rejected", "reason": "experience_identity_conflict"}
                if old["status"] != "active":
                    return {"status": "rejected", "reason": "experience_revoked_or_expired"}
                if meta.get("projection_hash") == snapshot["projection_hash"]:
                    if old["text"] != text:
                        return {"status": "rejected", "reason": "experience_card_integrity_failed"}
                    return {"status": "duplicate", "record_id": "event:" + old["event_id"]}
                if (snapshot["attempt"], snapshot["version"]) < (meta["attempt"], meta["version"]):
                    return {"status": "rejected", "reason": "experience_stale_revision"}
                if ((snapshot["attempt"], snapshot["version"]) == (meta["attempt"], meta["version"])
                        and meta["result_hash"] != snapshot["result_hash"]):
                    return {"status": "rejected", "reason": "experience_result_conflict"}
            version_exists = db.execute("SELECT payload FROM memory_task_experience_versions WHERE projection_hash=?",
                                       (snapshot["projection_hash"],)).fetchone()
            if not version_exists and db.execute("SELECT count(*) FROM memory_task_experience_versions").fetchone()[0] >= self.max_versions:
                return {"status": "rejected", "reason": "experience_capacity"}
            metadata = {"experience_schema": 1, "record_type": "agent_task_summary",
                "role": "agent_task", "source": "canonical_task_experience", "task_id": snapshot["task_id"],
                "task_status": snapshot["execution_status"], "owner_memory": True,
                "owner_source_scope": snapshot["scope"]["session"], "source_scope": snapshot["scope"],
                "attempt": snapshot["attempt"], "version": snapshot["version"],
                "result_hash": snapshot["result_hash"], "projection_hash": snapshot["projection_hash"],
                "text_hash": hashlib.sha256(text.encode()).hexdigest(), "snapshot": snapshot,
                "supersedes": meta.get("projection_hash", ""), "historical_snapshot": True,
                "is_current_project_fact": False, "raw_detail_unavailable": True}
            db.execute("INSERT OR IGNORE INTO memory_task_experience_versions VALUES (?,?,?,?,?)",
                (snapshot["projection_hash"], source_id, scope, now,
                 json.dumps({"snapshot": snapshot, "text": text, "supersedes": metadata["supersedes"]}, ensure_ascii=False)))
            if old:
                rid = old["event_id"]
                self.store._discard_cached_text(old["text"])
                db.execute("UPDATE memory_events SET text=?,created_at=?,metadata_json=? WHERE event_id=?",
                           (text, now, self.store._json(metadata), rid))
                invalidated.add("event:" + rid)
                # Dependent summaries must not keep a corrected old card.
                candidates = db.execute("SELECT event_id,metadata_json FROM memory_events WHERE scope=? AND status='active'", (scope,)).fetchall()
                while True:
                    before = len(invalidated)
                    for candidate in candidates:
                        parents = self.store._loads(candidate["metadata_json"]).get("source_record_ids", [])
                        if isinstance(parents, list) and invalidated.intersection(parents):
                            invalidated.add("event:" + candidate["event_id"])
                    if before == len(invalidated):
                        break
                for identity in invalidated:
                    key = identity.removeprefix("event:")
                    if key != rid:
                        db.execute("UPDATE memory_events SET status='superseded' WHERE event_id=?", (key,))
                    self.store.lifecycle._drop_vectors(db, "event", key)
                    db.execute("DELETE FROM memory_lifecycle WHERE record_type='event' AND record_id=?", (key,))
                    db.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type='event' AND record_id=?", (key,))
            else:
                rid = uuid.uuid4().hex
                db.execute("""INSERT INTO memory_events
                    (event_id,conversation_id,scope,text,layer,created_at,importance,confidence,expires_at,source_id,metadata_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (rid, "master", scope, text, "episodic", now, .62, .9, 0, source_id, self.store._json(metadata)))
            # Cross-database consistency is optimistic, not a distributed txn.
            # Re-read immediately before committing; recall revalidates again.
            if not check_current():
                raise ValueError("experience_source_changed_during_commit")
        if invalidated:
            self.store._clear_derived_caches(scope=scope, record_ids=invalidated)
        return {"status": "stored", "record_id": "event:" + rid,
                "projection_hash": snapshot["projection_hash"]}

    def history(self, *, task_id, scope, limit=32):
        with self.store._lock, self.store._connect() as db:
            head = db.execute("SELECT * FROM memory_events WHERE source_id=? AND scope=? AND status='active'",
                              ("owner:agent-task:" + task_id, scope)).fetchone()
            if head is None or self.store.lifecycle.revoked(db, scope=scope, source_id=head["source_id"], text=head["text"]):
                return []
            rows = db.execute("SELECT payload FROM memory_task_experience_versions WHERE source_id=? AND scope=? ORDER BY created DESC,rowid DESC LIMIT ?",
                ("owner:agent-task:" + task_id, scope, max(1, min(128, int(limit))))).fetchall()
            result = [json.loads(row[0]) for row in rows]
            for row in result:
                if row and (canonical_hash({k: v for k, v in row["snapshot"].items() if k != "projection_hash"})
                            != row["snapshot"]["projection_hash"]):
                    raise ValueError("experience_history_integrity_failed")
            return [row for row in result if row]
