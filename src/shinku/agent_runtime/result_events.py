"""Opt-in result outbox in the authoritative business task database.

The task update and event insertion share a transaction. Coordinator and
conversation records do not. Events keep immutable, redacted projections and
exact source hashes; a consumer must recheck the current task before effects.
No tools, provider requests, timers or external sends are replayed here.
"""
from __future__ import annotations

import json
import re
import uuid

from .contracts import AgentScope, canonical_hash

TERMINAL = frozenset({"completed", "partial", "failed", "blocked", "canceled", "unverified"})
UNRESOLVED = frozenset({"pending", "processing", "failed", "uncertain"})


class ResultEventStore:
    def __init__(self, task_store, *, max_entries=4096, max_attempts=3, lease_seconds=300):
        self.store = task_store
        self.max_entries = max(1, min(16384, int(max_entries)))
        self.max_attempts = max(1, min(8, int(max_attempts)))
        self.lease_seconds = max(1, min(3600, float(lease_seconds)))
        with task_store._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_result_events (
                event_id TEXT PRIMARY KEY, scope TEXT NOT NULL, task_id TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS agent_result_events_scope ON agent_result_events(scope, status, created)")
        # Explicit host opt-in. Legacy/default stores neither migrate nor emit.
        task_store.result_events = self

    @staticmethod
    def key(result):
        return canonical_hash({"task_id": result["task_id"], "scope": result["scope"],
            "attempt": result["attempt"], "version": result["version"], "type": "result_ready"})

    @staticmethod
    def _scope(scope):
        return canonical_hash({"requester": scope.requester, "session": scope.session})

    @staticmethod
    def _encode(row):
        row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
        return json.dumps(row, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(payload):
        try:
            row = json.loads(payload)
            if (row["record_hash"] != canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
                    or row["projection_hash"] != canonical_hash(row["canonical_result"])):
                raise ValueError()
            return row
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("result_event_integrity_failed") from exc

    def _write(self, db, row):
        db.execute("UPDATE agent_result_events SET status=?, payload=? WHERE event_id=? AND scope=?",
            (row["status"], self._encode(row), row["event_id"], self._scope(AgentScope(**row["scope"]))))

    def _expire(self, db):
        cutoff = self.store.clock() - self.store.ttl_seconds
        for (payload,) in db.execute("SELECT payload FROM agent_result_events WHERE created<? AND status IN ('pending','processing','uncertain','failed')", (cutoff,)).fetchall():
            row = self._decode(payload)
            row.update(previous_status=row["status"], previous_reason=row["reason"],
                status="expired", reason="result_retention_expired", claim_id="", lease_until=0)
            self._write(db, row)

    def append_in_transaction(self, db, task):
        result = task.get("execution_result")
        if (not result or result.get("execution", {}).get("status") not in TERMINAL
                or result.get("attempt") != task.get("execution_attempt", 0)
                or result.get("version") != task["revision"]):
            return None  # Retained historical results are not new finalizations.
        scope = AgentScope(task["requester"], task["session"])
        if result.get("task_id") != task["task_id"]:
            raise ValueError("result_event_binding_invalid")
        return self._enqueue(db, result, scope)

    def _enqueue(self, db, result, scope):
        from shinku.result_expression.facts import archive_result, validate_result
        error = validate_result(result, scope)
        if error or not result.get("task_id") or result["execution"]["status"] not in TERMINAL:
            raise ValueError(error or "result_event_binding_invalid")
        key = self.key(result)
        stored = db.execute("SELECT scope, payload FROM agent_result_events WHERE event_id=?", (key,)).fetchone()
        if stored:
            if stored[0] != self._scope(scope):
                raise ValueError("result_event_scope_collision")
            row = self._decode(stored[1])
            if row["source_hash"] != result["result_hash"]:
                raise ValueError("result_event_content_conflict")
            return row
        if db.execute("SELECT count(*) FROM agent_result_events").fetchone()[0] >= self.max_entries:
            # Keep tombstones. Capacity is diagnostic, never permission to
            # evict a receipt and repeat model calls or external delivery.
            raise ValueError("result_event_capacity")
        projection = archive_result(result)
        projection.pop("execution_report", None)
        row = {"event_id": key, "type": "result_ready", "task_id": result["task_id"],
            "scope": result["scope"], "attempt": result["attempt"], "version": result["version"],
            "source_hash": result["result_hash"], "canonical_result": projection,
            "projection_hash": canonical_hash(projection), "status": "pending",
            "reason": "", "attempts": 0, "claim_id": "", "lease_until": 0,
            "created": self.store.clock(), "updated": self.store.clock(),
            "expression_id": "", "delivery": {"status": "not_sent", "sent": False}}
        db.execute("INSERT INTO agent_result_events VALUES (?,?,?,?,?,?)", (key, self._scope(scope),
            result["task_id"], row["status"], row["created"], self._encode(row)))
        return row

    def enqueue(self, result, *, scope):
        """Idempotent explicit intake, not a production historical backfill."""
        with self.store._transaction() as db:
            return self._enqueue(db, result, scope)

    def get(self, event_id, *, scope):
        with self.store._transaction() as db:
            self._expire(db)
            row = db.execute("SELECT payload FROM agent_result_events WHERE event_id=? AND scope=?",
                (event_id, self._scope(scope))).fetchone()
            return self._decode(row[0]) if row else None

    def pending(self, *, scope, limit=32):
        with self.store._transaction() as db:
            self._expire(db)
            rows = db.execute("SELECT payload FROM agent_result_events WHERE scope=? AND status IN ('pending','processing') ORDER BY created, rowid LIMIT ?",
                (self._scope(scope), max(1, min(128, int(limit))))).fetchall()
            return [self._decode(row[0]) for row in rows]

    def diagnostics(self, *, scope, limit=32):
        with self.store._transaction() as db:
            self._expire(db)
            rows = db.execute("SELECT payload FROM agent_result_events WHERE scope=? AND status NOT IN ('pending','done') ORDER BY created, rowid LIMIT ?",
                (self._scope(scope), max(1, min(128, int(limit))))).fetchall()
            return [self._decode(row[0]) for row in rows]

    def committed_after_restart(self, task, work):
        """Distinguish generic worker restart interruption from execution failure.

        An independently committed result/event is authoritative only for the
        exact worker binding. This does not clear a cancellation, repair a
        failed verification, or treat a real worker failure as success.
        """
        if (not work or work.get("status") != "failed" or work.get("worker_active")
                or work.get("cancel_requested") or not work.get("metadata", {}).get("recovered_after_restart")
                or work.get("metadata", {}).get("agent_task_id") != task["task_id"]
                or work.get("profile_user_id") != task["requester"] or work.get("session_id") != task["session"]
                or work.get("attempt") != task.get("execution_work_attempt", task.get("execution_attempt", 0))):
            return False
        result = task.get("execution_result") or {}
        scope = AgentScope(task["requester"], task["session"])
        from shinku.result_expression.facts import validate_result
        if (validate_result(result, scope) or result.get("task_id") != task["task_id"]
                or (result.get("attempt"), result.get("version")) != (task.get("execution_attempt", 0), task["revision"])
                or result.get("execution", {}).get("status") != task.get("status")
                or task.get("status") not in TERMINAL or task.get("cancel_requested")):
            return False
        if task["status"] == "completed" and (result["verification"].get("accepted") is not True
                or task.get("completion_attempt") != task.get("execution_attempt")):
            return False
        event = self.get(self.key(result), scope=scope)
        return bool(event and event["source_hash"] == result["result_hash"] and event["status"] != "expired")

    def delivery_view(self, task, *, scope):
        result = task.get("execution_result") or {}
        if (not result or (result.get("attempt"), result.get("version")) != (
                task.get("execution_attempt", 0), task["revision"])):
            return None
        event = self.get(self.key(result), scope=scope)
        if not event or event["source_hash"] != result["result_hash"]:
            return None
        receipt = event["delivery"]
        return {"event_id": event["event_id"], "event_status": event["status"],
            "reason": event["reason"], "status": receipt["status"], "sent": receipt.get("sent") is True,
            "message_ids": list(receipt.get("message_ids", [])), "attempt": event["attempt"], "version": event["version"],
            "source": "agent_result_events"}

    def claim(self, event_id, *, scope):
        with self.store._transaction() as db:
            self._expire(db)
            stored = db.execute("SELECT payload FROM agent_result_events WHERE event_id=? AND scope=?",
                (event_id, self._scope(scope))).fetchone()
            if not stored:
                return None
            row, now = self._decode(stored[0]), self.store.clock()
            if row["status"] not in {"pending", "processing"} or (row["status"] == "processing" and row["lease_until"] > now):
                return None
            if row["attempts"] >= self.max_attempts:
                row.update(status="failed", reason="result_consumer_retry_exhausted", claim_id="", lease_until=0)
                self._write(db, row)
                return None
            row.update(status="processing", attempts=row["attempts"] + 1, claim_id=uuid.uuid4().hex,
                lease_until=now + self.lease_seconds, updated=now)
            self._write(db, row)
            return row

    def owns(self, claim, *, scope):
        row = self.get(claim["event_id"], scope=scope)
        return bool(row and row["status"] == "processing" and row["claim_id"] == claim["claim_id"]
            and row["lease_until"] > self.store.clock())

    def _change_claim(self, claim, scope, change):
        with self.store._transaction() as db:
            self._expire(db)
            stored = db.execute("SELECT payload FROM agent_result_events WHERE event_id=? AND scope=?",
                (claim["event_id"], self._scope(scope))).fetchone()
            if not stored:
                return False
            row = self._decode(stored[0])
            if (row["status"] != "processing" or row["claim_id"] != claim["claim_id"]
                    or row["lease_until"] <= self.store.clock()):
                return False
            change(row)
            row.update(updated=self.store.clock(), claim_id="", lease_until=0)
            self._write(db, row)
            return True

    def finish(self, claim, *, scope, status, reason, expression_id="", delivery=None):
        if status not in {"done", "stale", "uncertain", "failed", "suppressed"}:
            raise ValueError("result_event_status_invalid")
        safe_reason = reason if isinstance(reason, str) and re.fullmatch(r"[a-zA-Z0-9_:.-]{0,128}", reason) else "result_consumer_error"
        def change(row):
            row.update(status=status, reason=safe_reason, expression_id=expression_id)
            if delivery is not None:
                row["delivery"] = {"status": delivery.get("status", "uncertain"),
                    "sent": delivery.get("sent") is True,
                    "message_ids": [str(v)[:256] for v in delivery.get("message_ids", [])[:32]]}
        return self._change_claim(claim, scope, change)

    def retry(self, claim, *, scope, reason):
        def change(row):
            row.update(status="pending" if row["attempts"] < self.max_attempts else "failed",
                reason=reason if re.fullmatch(r"[a-zA-Z0-9_:.-]{0,128}", reason) else "result_consumer_error")
        return self._change_claim(claim, scope, change)

    def prune_tasks_in_transaction(self, db, now):
        self._expire(db)
        # 送达事件终结不代表人工等待终结；等待任务及其来源必须保留到显式处置。
        pinned = ("SELECT task_id FROM agent_result_events WHERE status IN ('pending','processing','uncertain','failed') "
                  "UNION SELECT task_id FROM agent_conversation_tasks WHERE json_extract(payload, '$.status') = 'waiting_user'")
        db.execute(f"DELETE FROM agent_conversation_tasks WHERE task_id NOT IN ({pinned}) AND (updated<? OR updated>?)",
            (now - self.store.ttl_seconds, now + 300))
        count = db.execute(f"SELECT count(*) FROM agent_conversation_tasks WHERE task_id IN ({pinned})").fetchone()[0]
        db.execute(f"DELETE FROM agent_conversation_tasks WHERE task_id IN (SELECT task_id FROM agent_conversation_tasks WHERE task_id NOT IN ({pinned}) ORDER BY updated DESC, rowid DESC LIMIT -1 OFFSET ?)",
            (max(0, self.store.max_entries - count),))
