"""Durable, finite batches; resuming never resets consumed work or grants access.

The runner knows nothing about tools, providers or document formats. Every
external call is reserved transactionally before it starts. Completed chunks
are immutable and failed/unknown calls remain billed against the same budget.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable, Any
import uuid


class BatchTaskError(ValueError):
    def __init__(self, message):
        super().__init__(message)
        # Only domain codes may enter durable diagnostics, never arbitrary text.
        self.code = message if isinstance(message, str) and re.fullmatch(
            r"(?:batch|document|pdf|translation|source|invalid|cannot|empty|encrypted|chinese)_[a-z0-9_]{1,90}", message) else type(self).__name__


@dataclass(frozen=True)
class BatchBudget:
    max_calls: int = 80
    max_seconds: float = 1800.0
    max_output_chars: int = 800_000

    def __post_init__(self):
        if (type(self.max_calls) is not int or not 1 <= self.max_calls <= 160
                or isinstance(self.max_seconds, bool) or not 1 <= self.max_seconds <= 3600
                or type(self.max_output_chars) is not int or not 1024 <= self.max_output_chars <= 2_000_000):
            raise BatchTaskError("invalid_batch_budget")


class BatchTaskStore:
    def __init__(self, path: str | Path, *, clock=time.time, create=True):
        self.path = Path(path)
        self.clock = clock
        if not create:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS batch_jobs (
                id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
                owner TEXT NOT NULL, session TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS batch_chunks (
                job TEXT NOT NULL, ordinal INTEGER NOT NULL, page INTEGER NOT NULL,
                source TEXT NOT NULL, source_hash TEXT NOT NULL, translated TEXT,
                calls INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job, ordinal))""")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    @staticmethod
    def _write(db, job):
        db.execute("UPDATE batch_jobs SET payload=? WHERE id=?",
                   (json.dumps(job, ensure_ascii=False, separators=(",", ":")), job["job_id"]))

    @contextmanager
    def read_transaction(self):
        db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    def _get(self, db, job_id, owner, session):
        row = db.execute("SELECT payload FROM batch_jobs WHERE id=? AND owner=? AND session=?",
                         (job_id, owner, session)).fetchone()
        if not row:
            raise BatchTaskError("batch_task_not_found")
        return json.loads(row[0])

    def create(self, *, owner: str, session: str, metadata: dict, chunks: list[tuple[int, str]],
               budget: BatchBudget | None = None) -> dict:
        if not owner or not session or not chunks or len(chunks) > 512:
            raise BatchTaskError("invalid_batch_scope_or_chunks")
        if any(type(page) is not int or page < 1 or not isinstance(text, str) or not text.strip()
               or len(text) > 6000 for page, text in chunks):
            raise BatchTaskError("invalid_batch_chunk")
        if sum(len(text) for _, text in chunks) > 600_000:
            raise BatchTaskError("batch_source_too_large")
        budget = budget or BatchBudget()
        identity = {"owner": owner, "session": session, "metadata": metadata,
                    "chunks": [(page, hashlib.sha256(text.encode()).hexdigest()) for page, text in chunks]}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM batch_jobs WHERE fingerprint=?", (fingerprint,)).fetchone()
            if row:
                return json.loads(row[0])
            now = float(self.clock())
            job = {"job_id": uuid.uuid4().hex, "status": "queued", "created_at": now,
                   "updated_at": now, "metadata": metadata, "budget": asdict(budget),
                   "calls_used": 0, "active_seconds": 0.0, "output_chars": 0,
                   "chunk_count": len(chunks), "lease": "", "lease_until": 0,
                   "call_started": None, "reason": "", "artifact": {}, "cancel_requested": False}
            db.execute("INSERT INTO batch_jobs VALUES (?, ?, ?, ?, ?)",
                       (job["job_id"], fingerprint, owner, session, json.dumps(job, ensure_ascii=False)))
            db.executemany("INSERT INTO batch_chunks(job,ordinal,page,source,source_hash) VALUES(?,?,?,?,?)",
                           [(job["job_id"], i, page, text, hashlib.sha256(text.encode()).hexdigest())
                            for i, (page, text) in enumerate(chunks)])
            return job

    def get(self, job_id: str, *, owner: str, session: str) -> dict:
        with self.read_transaction() as db:
            job = self._get(db, job_id, owner, session)
            done = db.execute("SELECT count(*) FROM batch_chunks WHERE job=? AND translated IS NOT NULL",
                              (job_id,)).fetchone()[0]
        # No original text, translations, credentials or permission grants.
        return {**job, "done_chunks": done, "owner": owner, "session": session,
                "lease_active": bool(job["lease"] and job["lease_until"] > float(self.clock()))}

    def chunks(self, job_id, *, owner, session) -> list[dict]:
        with self.read_transaction() as db:
            self._get(db, job_id, owner, session)
            return [dict(row) for row in db.execute("SELECT * FROM batch_chunks WHERE job=? ORDER BY ordinal", (job_id,))]

    def cancel(self, job_id, *, owner, session):
        with self.transaction() as db:
            job = self._get(db, job_id, owner, session)
            if job["status"] != "completed":
                job["cancel_requested"] = True
                job["status"] = "canceled"
                job["reason"] = "user_canceled"
                self._write(db, job)
        return self.get(job_id, owner=owner, session=session)

    def claim(self, job_id, *, owner, session) -> str:
        token = uuid.uuid4().hex
        with self.transaction() as db:
            job = self._get(db, job_id, owner, session)
            now = float(self.clock())
            if job["cancel_requested"] or job["status"] == "completed":
                return ""
            if job["lease"] and job["lease_until"] > now:
                raise BatchTaskError("batch_task_already_running")
            # A crashed request is not free. Its reserved call was counted;
            # conservatively charge its bounded lease interval as active time.
            if job["call_started"] is not None:
                job["active_seconds"] += min(900.0, max(0.0, now - job["call_started"]))
            job.update(lease=token, lease_until=now + 900, call_started=None,
                       status="running", updated_at=now, reason="")
            self._write(db, job)
        return token

    def _leased(self, db, job_id, owner, session, token):
        job = self._get(db, job_id, owner, session)
        if job["lease"] != token:
            raise BatchTaskError("batch_lease_lost")
        return job

    def reserve(self, job_id, *, owner, session, token):
        with self.transaction() as db:
            job = self._leased(db, job_id, owner, session, token)
            if job["cancel_requested"]:
                return None
            row = db.execute("SELECT * FROM batch_chunks WHERE job=? AND translated IS NULL ORDER BY ordinal LIMIT 1",
                             (job_id,)).fetchone()
            if row is None:
                return None
            if job["calls_used"] >= job["budget"]["max_calls"]:
                job.update(status="paused", reason="call_budget_exhausted")
                self._write(db, job)
                return None
            if job["active_seconds"] >= job["budget"]["max_seconds"]:
                job.update(status="paused", reason="time_budget_exhausted")
                self._write(db, job)
                return None
            now = float(self.clock())
            job.update(calls_used=job["calls_used"] + 1, call_started=now, lease_until=now + 900)
            db.execute("UPDATE batch_chunks SET calls=calls+1 WHERE job=? AND ordinal=?", (job_id, row["ordinal"]))
            self._write(db, job)
            return dict(row)

    def finish_call(self, job_id, *, owner, session, token, chunk, translated=None, reason=""):
        with self.transaction() as db:
            job = self._leased(db, job_id, owner, session, token)
            now = float(self.clock())
            started = job["call_started"] if job["call_started"] is not None else now
            job["active_seconds"] += max(0.0, now - float(started))
            job["call_started"] = None
            job["updated_at"] = now
            if job["cancel_requested"]:
                self._write(db, job)
                return
            if translated is not None:
                if job["output_chars"] + len(translated) > job["budget"]["max_output_chars"]:
                    job.update(status="paused", reason="output_budget_exhausted")
                else:
                    changed = db.execute("UPDATE batch_chunks SET translated=? WHERE job=? AND ordinal=? AND translated IS NULL",
                                         (translated, job_id, chunk["ordinal"])).rowcount
                    if changed:
                        job["output_chars"] += len(translated)
                    job["reason"] = ""
            else:
                job["reason"] = reason[:160]
            self._write(db, job)

    def finish(self, job_id, *, owner, session, token, artifact=None, reason=""):
        with self.transaction() as db:
            job = self._leased(db, job_id, owner, session, token)
            if job["cancel_requested"]:
                job.update(status="canceled", reason="user_canceled")
            elif artifact is not None:
                missing = db.execute("SELECT count(*) FROM batch_chunks WHERE job=? AND translated IS NULL", (job_id,)).fetchone()[0]
                if missing:
                    raise BatchTaskError("cannot_finish_missing_chunks")
                job.update(status="completed", artifact=artifact, reason="")
            elif job["status"] == "running":
                job.update(status="paused", reason=reason or job["reason"] or "interrupted")
            job.update(lease="", lease_until=0, updated_at=float(self.clock()))
            self._write(db, job)
        return self.get(job_id, owner=owner, session=session)


class BatchRunner:
    """One tool invocation executes many durable units under one total budget."""
    def __init__(self, store: BatchTaskStore):
        self.store = store

    def run(self, job_id, *, owner, session, translate: Callable, validate: Callable,
            publish: Callable, guard: Callable[[], None] = lambda: None,
            on_progress: Callable[[dict], None] = lambda _job: None):
        guard()
        token = self.store.claim(job_id, owner=owner, session=session)
        if not token:
            return self.store.get(job_id, owner=owner, session=session)
        failures = 0
        try:
            while True:
                guard()
                chunk = self.store.reserve(job_id, owner=owner, session=session, token=token)
                if chunk is None:
                    break
                try:
                    translated = validate(chunk, translate(chunk))
                except Exception as exc:
                    self.store.finish_call(job_id, owner=owner, session=session, token=token,
                                           chunk=chunk, reason=exc.code if isinstance(exc, BatchTaskError) else type(exc).__name__)
                    failures += 1
                    if failures >= 2:
                        break
                    continue
                self.store.finish_call(job_id, owner=owner, session=session, token=token,
                                       chunk=chunk, translated=translated)
                failures = 0
                current = self.store.get(job_id, owner=owner, session=session)
                on_progress(current)
                if current["status"] != "running":
                    break
            current = self.store.get(job_id, owner=owner, session=session)
            artifact = None
            if current["status"] == "running" and current["done_chunks"] == current["chunk_count"]:
                guard()
                artifact = publish(current, self.store.chunks(job_id, owner=owner, session=session))
            return self.store.finish(job_id, owner=owner, session=session, token=token, artifact=artifact)
        except Exception as exc:
            # Do not serialize arbitrary exception text (it may contain secrets).
            try:
                self.store.finish(job_id, owner=owner, session=session, token=token, reason=type(exc).__name__)
            except BatchTaskError:
                pass
            raise
