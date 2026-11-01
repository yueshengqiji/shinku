"""Evidence-driven memory lifespan, independent of retrieval and fact truth.

All mutation uses the MemoryStore lock and short SQLite transactions. External
summarizers are deliberately not implicit: v1 only proposes source-extractive
summaries. Reclamation requires an explicit scope and independent approval.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, replace
import math
from typing import Any, TYPE_CHECKING

from .lifecycle_policy import LifecyclePolicy
from .models import MemoryRecord

if TYPE_CHECKING:
    from .store import MemoryStore


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_lifecycle (
 record_type TEXT NOT NULL, record_id TEXT NOT NULL, scope TEXT NOT NULL,
 content_hash TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 tier INTEGER NOT NULL DEFAULT 1, representation TEXT NOT NULL DEFAULT 'raw',
 pinned INTEGER NOT NULL DEFAULT 0, raw_protected_until INTEGER NOT NULL DEFAULT 0,
 enrolled_at INTEGER NOT NULL, first_reinforced_at INTEGER NOT NULL DEFAULT 0,
 last_reinforced_at INTEGER NOT NULL DEFAULT 0, reinforce_count INTEGER NOT NULL DEFAULT 0,
 distinct_days INTEGER NOT NULL DEFAULT 0, last_day INTEGER NOT NULL DEFAULT -1,
 anchor_effective REAL NOT NULL DEFAULT 0, anchor_turns INTEGER NOT NULL DEFAULT 0,
 summary_text TEXT NOT NULL DEFAULT '', summary_source_hash TEXT NOT NULL DEFAULT '',
 summary_at INTEGER NOT NULL DEFAULT 0, raw_reclaimed INTEGER NOT NULL DEFAULT 0,
 policy_version TEXT NOT NULL, last_scan INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(record_type, record_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_scan ON memory_lifecycle(scope,last_scan,record_id);
CREATE TABLE IF NOT EXISTS memory_lifecycle_scopes (
 scope TEXT PRIMARY KEY, wall_checkpoint INTEGER NOT NULL,
 effective_seconds REAL NOT NULL DEFAULT 0, turn_count INTEGER NOT NULL DEFAULT 0,
 active_until INTEGER NOT NULL DEFAULT 0, replay_floor INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memory_lifecycle_activity (
 scope TEXT NOT NULL, evidence_id TEXT NOT NULL, created_at INTEGER NOT NULL,
 PRIMARY KEY(scope,evidence_id)
);
CREATE TABLE IF NOT EXISTS memory_lifecycle_evidence (
 scope TEXT NOT NULL, record_type TEXT NOT NULL, record_id TEXT NOT NULL,
 evidence_id TEXT NOT NULL, kind TEXT NOT NULL, created_at INTEGER NOT NULL,
 session_hash TEXT NOT NULL, PRIMARY KEY(scope,record_type,record_id,evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_evidence_time ON memory_lifecycle_evidence(created_at);
CREATE TABLE IF NOT EXISTS memory_lifecycle_audit (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
 record_id TEXT NOT NULL, created_at INTEGER NOT NULL, action TEXT NOT NULL,
 mode TEXT NOT NULL, reason TEXT NOT NULL, before_tier INTEGER NOT NULL,
 after_tier INTEGER NOT NULL, policy_version TEXT NOT NULL,
 logical_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memory_revocations (
 scope TEXT NOT NULL, source_hash TEXT NOT NULL, content_hash TEXT NOT NULL,
 revoked_at INTEGER NOT NULL, PRIMARY KEY(scope,source_hash,content_hash)
);
"""


def digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    # Punctuation at the edges is harmless, negation/tense/attribution is not.
    return re.sub(r"\s+", " ", str(value)).strip(" \t\r\n。，,.！!").casefold()


def sensitive(value: str) -> bool:
    return bool(re.search(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|bearer|cookie)\b|密码|密钥|访问令牌", value))


class MemoryLifecycle:
    def __init__(self, store: "MemoryStore", policy: LifecyclePolicy | None = None) -> None:
        self.store = store
        self.policy = policy or LifecyclePolicy(mode="off")
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._scan_lock = threading.Lock()
        self._last_error = ""
        self.wall_time = time.time

    @staticmethod
    def _record(conn, record_type: str, record_id: str):
        if record_type == "event":
            return conn.execute("SELECT * FROM memory_events WHERE event_id=?", (record_id,)).fetchone()
        if record_type == "fact":
            return conn.execute("SELECT * FROM memory_facts WHERE fact_id=?", (record_id,)).fetchone()
        return None

    @staticmethod
    def _identity(record_id: str) -> tuple[str, str]:
        kind, _, bare = str(record_id).partition(":")
        if kind not in {"event", "fact"} or not bare:
            raise ValueError("invalid_memory_record_id")
        return kind, bare

    @staticmethod
    def _eligible(row, record_type: str) -> bool:
        if row is None or row["status"] != "active":
            return False
        if record_type == "event":
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (ValueError, TypeError):
                metadata = {}
            if isinstance(metadata, dict) and metadata.get("experience_schema") == 1:
                return False  # Historical cards use source validity, not persona reinforcement.
        if record_type == "fact":
            return True
        # Task archives, pre-existing summaries and temporary states have
        # separate retention/validity rules and must not silently become L3.
        return row["layer"] in {"observation", "episodic", "legacy_raw", "legacy_lesson"} and not str(row["source_id"]).startswith("owner:agent-task-")

    @staticmethod
    def revoked(conn, *, scope: str, source_id: str, text: str) -> bool:
        source_hash = digest(source_id) if source_id else ""
        # 主人共享副本的遗忘同步约束精确同源原件；不能按相同正文跨用户撤销。
        # 复用持久化墓碑，另一连接/进程下一次读取即生效，无需等待缓存通知。
        if source_id.startswith("turn:") and conn.execute(
            "SELECT 1 FROM memory_revocations WHERE scope='private:master' AND source_hash=? LIMIT 1",
            (digest("owner:" + source_id),)).fetchone() is not None:
            return True
        return conn.execute("SELECT 1 FROM memory_revocations WHERE scope=? AND ((source_hash<>'' AND source_hash=?) OR content_hash=?) LIMIT 1", (scope, source_hash, digest(text))).fetchone() is not None

    @staticmethod
    def revoke(conn, *, scope: str, source_id: str, text: str, now: int, original_hash: str = "") -> None:
        for content_hash in dict.fromkeys((digest(text), original_hash)):
            if content_hash:
                conn.execute("INSERT OR IGNORE INTO memory_revocations VALUES (?,?,?,?)", (scope, digest(source_id) if source_id else "", content_hash, now))

    def _clock(self, conn, scope: str, now: int, evidence_id: str = "", occurred_at: int | None = None):
        conn.execute("INSERT OR IGNORE INTO memory_lifecycle_scopes(scope,wall_checkpoint) VALUES (?,?)", (scope, now))
        clock = conn.execute("SELECT * FROM memory_lifecycle_scopes WHERE scope=?", (scope,)).fetchone()
        checkpoint = int(clock["wall_checkpoint"])
        stamp = max(now, checkpoint)
        active = max(0, min(stamp, int(clock["active_until"])) - checkpoint)
        elapsed = float(clock["effective_seconds"]) + active + (stamp - checkpoint - active) * self.policy.idle_rate
        turns = int(clock["turn_count"])
        active_until = int(clock["active_until"])
        if evidence_id:
            occurred = now if occurred_at is None else int(occurred_at)
            # A replay older than the retained idempotence window is never
            # relabelled as a new interaction. Future/rollback events fail shut.
            if max(int(clock["replay_floor"]), stamp - self.policy.evidence_retention_seconds) <= occurred <= stamp:
                added = conn.execute("INSERT OR IGNORE INTO memory_lifecycle_activity VALUES (?,?,?)", (scope, digest(evidence_id), occurred)).rowcount
                if added and occurred >= checkpoint:
                    turns += 1
                    active_until = max(active_until, stamp + self.policy.active_window_seconds)
        conn.execute("UPDATE memory_lifecycle_scopes SET wall_checkpoint=?,effective_seconds=?,turn_count=?,active_until=? WHERE scope=?", (stamp, elapsed, turns, active_until, scope))
        return {"wall_checkpoint": stamp, "effective_seconds": elapsed, "turn_count": turns, "replay_floor": int(clock["replay_floor"])}

    def note_interaction(self, *, scope: str, evidence_id: str, now: int, occurred_at: int | None = None) -> None:
        if not self.policy.enabled or not scope or not evidence_id:
            return
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._clock(conn, scope, now, evidence_id, occurred_at)

    def _enroll(self, conn, record_type: str, row, now: int) -> bool:
        if not self._eligible(row, record_type):
            return False
        if self.revoked(conn, scope=row["scope"], source_id=row["source_id"], text=row["text"]):
            table, pk = ("memory_events", "event_id") if record_type == "event" else ("memory_facts", "fact_id")
            conn.execute(f"UPDATE {table} SET status='forgotten' WHERE {pk}=?", (row[pk],))
            conn.execute("DELETE FROM memory_embeddings WHERE record_type=? AND record_id=?", (record_type, row[pk]))
            return False
        rid = str(row["event_id"] if record_type == "event" else row["fact_id"])
        existing = conn.execute("SELECT * FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid)).fetchone()
        body_hash = digest(row["text"])
        clock = self._clock(conn, row["scope"], now)
        if existing is not None:
            if existing["content_hash"] != body_hash:
                # Corrections are new content versions, not reinforcement of
                # old truth. Old summary/counters/protection cannot carry over.
                conn.execute("""UPDATE memory_lifecycle SET content_hash=?,version=version+1,
                  reinforce_count=0,distinct_days=0,last_day=-1,first_reinforced_at=0,last_reinforced_at=0,
                  tier=?,representation='raw',raw_reclaimed=0,raw_protected_until=0,
                  summary_text='',summary_source_hash='',summary_at=0,enrolled_at=?,
                  anchor_effective=?,anchor_turns=?,policy_version=? WHERE record_type=? AND record_id=?""",
                  (body_hash, int(existing["tier"]) if self.policy.mode == "shadow" else (3 if existing["pinned"] else 1), now, clock["effective_seconds"], clock["turn_count"], self.policy.revision, record_type, rid))
                conn.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type=? AND record_id=?", (record_type, rid))
            return True
        conn.execute("""INSERT INTO memory_lifecycle(record_type,record_id,scope,content_hash,tier,pinned,
          enrolled_at,anchor_effective,anchor_turns,policy_version) VALUES (?,?,?,?,?,?,?,?,?,?)""",
          (record_type, rid, row["scope"], body_hash, 3 if record_type == "fact" else 1, int(record_type == "fact"), now, clock["effective_seconds"], clock["turn_count"], self.policy.revision))
        return True

    def enroll(self, *, record_id: str, now: int) -> bool:
        if not self.policy.enabled:
            return False
        kind, bare = self._identity(record_id)
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._enroll(conn, kind, self._record(conn, kind, bare), now)

    def state(self, record_id: str) -> dict[str, Any]:
        kind, bare = self._identity(record_id)
        with self.store._lock, self.store._connect() as conn:
            row = conn.execute("SELECT * FROM memory_lifecycle WHERE record_type=? AND record_id=?", (kind, bare)).fetchone()
        if row is None:
            return {}
        return {key: row[key] for key in row.keys() if key != "summary_text"}

    def preview(self, *, now: int, scope: str) -> dict[str, Any]:
        if not scope:
            raise ValueError("preview_scope_required")
        # A detached policy + rolled-back transaction: no global policy flip,
        # no new levels, counters, migration anchors or audit records persist.
        preview = MemoryLifecycle(self.store, replace(self.policy, mode="shadow"))
        result = preview._scan(now=now, scope=scope, allow_reclaim=False, rollback=True)
        return {**result, "preview": True, "configured_mode": self.policy.mode}

    @staticmethod
    def strength_ratio(*, effective_age: float, time_constant: float, count: int) -> float:
        if not math.isfinite(effective_age) or not math.isfinite(time_constant) or time_constant <= 0:
            raise ValueError("invalid_decay_time")
        tau = time_constant * (1 + min(3, max(0, count)) * 0.5)
        return math.exp(-max(0, effective_age) / tau)

    @staticmethod
    def _user_source(record: MemoryRecord, actor_id: str) -> bool:
        if record.layer == "fact":
            # Legacy facts have no independently verifiable author column.
            # Explicitly pinned facts need no automatic reinforcement anyway.
            return False
        meta = record.metadata
        return meta.get("role") == "user" and bool(actor_id) and str(meta.get("sender_id") or "") == actor_id

    def reinforce_remention(self, *, record: MemoryRecord, user_text: str, actor_id: str, evidence_id: str, now: int, session_id: str = "", occurred_at: int | None = None) -> dict[str, Any]:
        if not self._user_source(record, actor_id) or len(normalized(user_text)) < 8 or normalized(user_text) != normalized(record.text):
            return {"accepted": False, "reason": "unconfirmed_remention"}
        if any(mark in user_text for mark in ("?", "？", "\"", "“", "”", "他说", "她说", "听说")):
            return {"accepted": False, "reason": "quoted_or_questioned_remention"}
        if re.search(r"假如|假设|如果我|更正|记错|不再|不是这样|没有说过|别记|忘记|忘掉", user_text):
            return {"accepted": False, "reason": "hypothesis_correction_or_revocation"}
        return self._reinforce(record=record, evidence_id=evidence_id, kind="user_remention", now=now, session_id=session_id, occurred_at=occurred_at)

    def reinforce_used(self, *, record: MemoryRecord, quote: str, final_text: str, actor_id: str, evidence_id: str, now: int, session_id: str = "", occurred_at: int | None = None) -> dict[str, Any]:
        # Only source-bound verbatim use is supported automatically in v1.
        # Mere supplied IDs, prompt injection and semantic guesses never count.
        if not self._user_source(record, actor_id) or len(quote) < 12 or len(quote) > 1000 or quote not in record.text or quote not in final_text:
            return {"accepted": False, "reason": "unverified_final_use"}
        return self._reinforce(record=record, evidence_id=evidence_id, kind="answer_use", now=now, session_id=session_id, occurred_at=occurred_at)

    def reinforce_reviewed(self, *, record: MemoryRecord, decision: dict, user_text: str, final_text: str, actor_id: str, evidence_id: str, now: int, kind: str, occurred_at: int) -> dict[str, Any]:
        """A review is an opinion, accepted only with host-bound verbatim spans.

        Public IDs alone confer no authority. The worker supplies a real scoped
        snapshot; the final transaction still checks author, hash and status.
        """
        try:
            confidence = float(decision.get("confidence", 0))
        except (ValueError, TypeError):
            confidence = 0
        relation = "same_fact" if kind == "remention" else "operative_use"
        source_quote, user_quote, final_quote = (str(decision.get(key) or "") for key in ("source_quote", "user_quote", "final_quote"))
        if (not self._user_source(record, actor_id) or not math.isfinite(confidence)
            or confidence < self.policy.review_min_confidence or confidence > 1
            or decision.get("record_id") != record.record_id or decision.get("relation") != relation
            or decision.get("same_actor") is not True
            or any(decision.get(flag) is not False for flag in ("negated", "quoted", "hypothetical", "correction"))
            or not 8 <= len(source_quote) <= 1000 or source_quote not in record.text
            or not 4 <= len(user_quote) <= 1000 or user_quote not in user_text
            or (kind != "remention" and (not 8 <= len(final_quote) <= 1500 or final_quote not in final_text))):
            return {"accepted": False, "reason": "unbound_or_uncertain_semantic_review"}
        return self._reinforce(record=record, evidence_id=evidence_id, kind="reviewed_remention" if kind == "remention" else "reviewed_answer_use", now=now, session_id=record.scope, occurred_at=occurred_at)

    def _reinforce(self, *, record: MemoryRecord, evidence_id: str, kind: str, now: int, session_id: str, occurred_at: int | None) -> dict[str, Any]:
        if not self.policy.enabled or not evidence_id or len(evidence_id) > 512:
            return {"accepted": False, "reason": "disabled_or_invalid_evidence"}
        record_type, rid = self._identity(record.record_id)
        occurred = now if occurred_at is None else int(occurred_at)
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._record(conn, record_type, rid)
            if not self._eligible(row, record_type) or row["scope"] != record.scope or digest(row["text"]) != digest(record.text):
                return {"accepted": False, "reason": "stale_or_inactive_source"}
            if record_type == "fact" and row["expires_at"] and row["expires_at"] <= now:
                return {"accepted": False, "reason": "fact_validity_expired"}
            if not self._enroll(conn, record_type, row, now):
                return {"accepted": False, "reason": "revoked_source"}
            # The model-facing snapshot cannot forge role/author metadata.
            if record_type == "event":
                actual_meta = self.store._loads(row["metadata_json"])
                if actual_meta.get("role") != "user" or actual_meta.get("sender_id") != record.metadata.get("sender_id"):
                    return {"accepted": False, "reason": "untrusted_author_snapshot"}
            if sensitive(record.text):
                return {"accepted": False, "reason": "sensitive_data_not_auto_protected"}
            state = conn.execute("SELECT * FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid)).fetchone()
            clock = self._clock(conn, record.scope, now)
            if occurred < max(clock["replay_floor"], now - self.policy.evidence_retention_seconds) or occurred > now or occurred < int(state["last_reinforced_at"]):
                return {"accepted": False, "reason": "stale_or_future_evidence"}
            evidence_hash = digest(evidence_id)
            if conn.execute("SELECT 1 FROM memory_lifecycle_evidence WHERE scope=? AND record_type=? AND record_id=? AND evidence_id=?", (record.scope, record_type, rid, evidence_hash)).fetchone():
                return {"accepted": False, "reason": "duplicate_evidence"}
            day = occurred // 86400  # Explicit UTC policy; does not depend on host TZ.
            daily = conn.execute("SELECT COUNT(*) FROM memory_lifecycle_evidence WHERE scope=? AND record_type=? AND record_id=? AND created_at>=? AND created_at<?", (record.scope, record_type, rid, day * 86400, (day + 1) * 86400)).fetchone()[0]
            if daily >= self.policy.daily_reinforcement_cap or (state["last_reinforced_at"] and occurred - int(state["last_reinforced_at"]) < self.policy.repeat_interval_seconds):
                return {"accepted": False, "reason": "burst_or_interval_guard"}
            conn.execute("INSERT INTO memory_lifecycle_evidence VALUES (?,?,?,?,?,?,?)", (record.scope, record_type, rid, evidence_hash, kind, occurred, digest(session_id)))
            count = min(1000, int(state["reinforce_count"]) + 1)
            days = min(1000, int(state["distinct_days"]) + int(day != state["last_day"]))
            first = int(state["first_reinforced_at"] or occurred)
            target = 3 if state["pinned"] or (count >= self.policy.l3_count and days >= self.policy.l3_days and occurred - first >= self.policy.l3_span_seconds) else max(2, int(state["tier"]))
            recent = conn.execute("SELECT COUNT(*),COUNT(DISTINCT created_at/86400) FROM memory_lifecycle_evidence WHERE record_type=? AND record_id=? AND created_at>=?", (record_type, rid, now - self.policy.reinforcement_window_seconds)).fetchone()
            protected = int(state["raw_protected_until"])
            if recent[0] >= self.policy.raw_count and recent[1] >= self.policy.raw_days:
                protected = max(protected, now + self.policy.raw_protection_seconds)
            # Budget stops new automatic permanent/raw protection, never
            # silently evicts previously promised L3 or manual pins.
            if target > state["tier"] or protected > state["raw_protected_until"]:
                used = self._protected_bytes(conn, now)
                additional = 0 if state["tier"] == 3 or state["pinned"] or state["raw_protected_until"] > now else len(record.text.encode("utf-8"))
                if not state["pinned"] and used + additional > self.policy.auto_raw_budget_bytes:
                    target = max(2, int(state["tier"]))
                    protected = int(state["raw_protected_until"])
                    self._audit(conn, state, now, "budget_warning", "auto_raw_budget_exceeded", int(state["tier"]))
            conn.execute("""UPDATE memory_lifecycle SET first_reinforced_at=?,last_reinforced_at=?,reinforce_count=?,
              distinct_days=?,last_day=?,anchor_effective=?,anchor_turns=?,tier=?,raw_protected_until=?,
              summary_text='',summary_source_hash=CASE WHEN raw_reclaimed=1 THEN summary_source_hash ELSE '' END,summary_at=0,version=version+1 WHERE record_type=? AND record_id=?""",
              (first, occurred, count, days, day, clock["effective_seconds"], clock["turn_count"], target if self.policy.mode == "active" else state["tier"], protected if self.policy.mode == "active" else state["raw_protected_until"], record_type, rid))
            self._audit(conn, state, now, "reinforce", kind, target)
            self._prune(conn, now)
            return {"accepted": True, "proposed_tier": target, "applied": self.policy.mode == "active", "count": count, "distinct_days": days}

    @staticmethod
    def _protected_bytes(conn, now: int) -> int:
        return int(conn.execute("""SELECT COALESCE(SUM(length(CAST(m.text AS BLOB))),0) FROM memory_lifecycle l
          JOIN memory_events m ON l.record_type='event' AND l.record_id=m.event_id
          WHERE m.status='active' AND (l.tier=3 OR l.pinned=1 OR l.raw_protected_until>?)""", (now,)).fetchone()[0])

    def _audit(self, conn, state, now: int, action: str, reason: str, after: int, logical_bytes: int = 0) -> None:
        conn.execute("""INSERT INTO memory_lifecycle_audit(scope,record_id,created_at,action,mode,reason,before_tier,after_tier,policy_version,logical_bytes)
          VALUES (?,?,?,?,?,?,?,?,?,?)""", (state["scope"], f"{state['record_type']}:{state['record_id']}", now, action, self.policy.mode, reason, int(state["tier"]), after, self.policy.revision, logical_bytes))

    @staticmethod
    def _drop_vectors(conn, kind: str, rid: str) -> int:
        saved = int(conn.execute("SELECT COALESCE(sum(length(vector_blob)),0) FROM memory_embeddings WHERE record_type=? AND record_id=?", (kind, rid)).fetchone()[0])
        conn.execute("DELETE FROM memory_embeddings WHERE record_type=? AND record_id=?", (kind, rid))
        return saved

    def _prune(self, conn, now: int) -> None:
        cutoff = now - self.policy.evidence_retention_seconds
        conn.execute("UPDATE memory_lifecycle_scopes SET replay_floor=max(replay_floor,?)", (cutoff,))
        conn.execute("DELETE FROM memory_lifecycle_evidence WHERE created_at<?", (cutoff,))
        conn.execute("DELETE FROM memory_lifecycle_activity WHERE created_at<?", (cutoff,))
        conn.execute("DELETE FROM memory_lifecycle_audit WHERE sequence NOT IN (SELECT sequence FROM memory_lifecycle_audit ORDER BY sequence DESC LIMIT ?)", (self.policy.audit_limit,))

    @staticmethod
    def extract_summary(text: str, max_chars: int) -> str:
        # Safe local fallback: only complete verbatim sentences, no model
        # paraphrase. If all detail cannot fit, do not pretend the synopsis
        # preserved it all; reclamation below requires a stronger validation.
        lines = [line.strip() for line in re.split(r"(?<=[。！？!?\n])", text) if line.strip()]
        result = []
        size = 0
        for line in lines:
            if size + len(line) > max_chars:
                break
            result.append(line)
            size += len(line)
        return "".join(result) if result and size < len(text) else ""

    @staticmethod
    def summary_valid(text: str, summary: str) -> bool:
        # Reclamation is fail-closed: important numbers/negation/conditions
        # and named speakers must remain. Do not use guessed model truth.
        markers = re.findall(r"\d+(?:[.:/年月日%-]\d+)*|不[^，。！？\n]{0,16}|没[^，。！？\n]{0,16}|(?:如果|只有|除非|但是|他说|她说)[^。！？\n]{0,80}", text)
        return bool(summary) and len(summary.encode("utf-8")) < len(text.encode("utf-8")) and all(piece in summary for piece in markers) and all(piece in text for piece in re.split(r"(?<=[。！？!?])", summary) if piece)

    def scan(self, *, now: int, scope: str = "", allow_reclaim: bool = False, approved_scope: str = "") -> dict[str, Any]:
        if allow_reclaim and (not scope or approved_scope != scope or self.policy.mode != "active"):
            raise ValueError("reclamation_requires_active_mode_and_exact_scope_approval")
        if not self.policy.enabled:
            return {"mode": "off", "scanned": 0, "actions": [], "logical_bytes_reclaimed": 0}
        if not self._scan_lock.acquire(blocking=False):
            return {"mode": self.policy.mode, "busy": True, "scanned": 0, "actions": []}
        try:
            return self._scan(now=now, scope=scope, allow_reclaim=allow_reclaim)
        finally:
            self._scan_lock.release()

    def _scan(self, *, now: int, scope: str, allow_reclaim: bool, rollback: bool = False) -> dict[str, Any]:
        actions = []
        reclaimed = 0
        vector_reclaimed = 0
        skipped = Counter()
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            clause = " AND scope=?" if scope else ""
            params = (scope,) if scope else ()
            # Bounded enrollment of the pre-existing database. No invented
            # reuse history and no aging before the enrollment grace starts.
            for kind, table, pk in (("event", "memory_events", "event_id"), ("fact", "memory_facts", "fact_id")):
                eligibility = " AND layer IN ('observation','episodic','legacy_raw','legacy_lesson') AND source_id NOT LIKE 'owner:agent-task-%'" if kind == "event" else ""
                rows = conn.execute(f"SELECT * FROM {table} m WHERE status='active'{clause}{eligibility} AND NOT EXISTS (SELECT 1 FROM memory_lifecycle l WHERE l.record_type=? AND l.record_id=m.{pk}) ORDER BY {pk} LIMIT ?", (*params, kind, self.policy.batch_size)).fetchall()
                for row in rows:
                    self._enroll(conn, kind, row, now)
            states = conn.execute(f"SELECT * FROM memory_lifecycle WHERE 1=1{clause} ORDER BY last_scan,record_id LIMIT ?", (*params, self.policy.batch_size)).fetchall()
            scan_sequence = conn.execute("SELECT COALESCE(MAX(last_scan),0)+1 FROM memory_lifecycle").fetchone()[0]
            for state in states:
                kind, rid = state["record_type"], state["record_id"]
                row = self._record(conn, kind, rid)
                conn.execute("UPDATE memory_lifecycle SET last_scan=? WHERE record_type=? AND record_id=?", (scan_sequence, kind, rid))
                if not self._eligible(row, kind):
                    skipped["inactive"] += 1
                    continue
                if digest(row["text"]) != state["content_hash"]:
                    self._enroll(conn, kind, row, now)
                    skipped["content_changed"] += 1
                    continue
                clock = self._clock(conn, state["scope"], now)
                age = max(0, clock["effective_seconds"] - float(state["anchor_effective"]))
                turns = max(0, clock["turn_count"] - int(state["anchor_turns"]))
                if state["policy_version"] != self.policy.revision:
                    # Parameter migration starts a new safe grace/checkpoint,
                    # not immediate mass eviction under a different policy.
                    conn.execute("UPDATE memory_lifecycle SET policy_version=?,enrolled_at=? WHERE record_type=? AND record_id=?", (self.policy.revision, now, kind, rid))
                    skipped["policy_migration"] += 1
                    continue
                if state["pinned"] or state["tier"] == 3:
                    skipped["protected_l3"] += 1
                    continue
                if state["raw_protected_until"] > now or now - state["enrolled_at"] < self.policy.migration_grace_seconds:
                    skipped["protected_or_grace"] += 1
                    continue
                # 保留的原文仍需要可检索表示；不再仅因处于 L1 就回收有效向量。
                # 向量随原文撤销、摘要替代或实际回收而失效，不改变以下冷却/晋级条件。
                base_threshold = self.policy.l2_seconds if state["tier"] == 2 else self.policy.l1_seconds
                threshold = base_threshold
                # Bounded reinforcement gain (not cumulative immortality).
                threshold *= 1 + min(3, int(state["reinforce_count"])) * 0.5
                strength = self.strength_ratio(effective_age=age, time_constant=base_threshold, count=int(state["reinforce_count"]))
                if strength > math.exp(-1) or turns < self.policy.minimum_idle_turns:
                    skipped["not_cold"] += 1
                    continue
                action, reason, target = "", "", int(state["tier"])
                if state["tier"] == 2:
                    action, reason, target = "demote", "l2_cold", 1
                    if self.policy.mode == "active":
                        conn.execute("UPDATE memory_lifecycle SET tier=1,anchor_effective=?,anchor_turns=?,version=version+1 WHERE record_type=? AND record_id=?", (clock["effective_seconds"], clock["turn_count"], kind, rid))
                elif state["raw_reclaimed"]:
                    action, reason = "expire", "summary_cold"
                    if self.policy.mode == "active" and allow_reclaim:
                        saved = len(str(row["text"]).encode("utf-8"))
                        self.store._discard_cached_text(str(row["text"]))
                        conn.execute("UPDATE memory_events SET text='',status='expired',metadata_json='{}' WHERE event_id=? AND status='active'", (rid,))
                        conn.execute("UPDATE memory_lifecycle SET content_hash=?,representation='discarded',version=version+1 WHERE record_type=? AND record_id=?", (digest(""), kind, rid))
                        vector_reclaimed += self._drop_vectors(conn, kind, rid)
                        reclaimed += saved
                        self._audit(conn, state, now, action, reason, target, saved)
                elif not state["summary_text"]:
                    summary = self.extract_summary(row["text"], self.policy.summary_max_chars)
                    if not self.summary_valid(row["text"], summary):
                        if summary or len(str(row["text"])) > self.policy.summary_max_chars:
                            # A failed/doubtful compression is NOT a licence
                            # to throw away the only valid source instead.
                            skipped["summary_validation_failed_keep_raw"] += 1
                            continue
                        # A short indivisible event cannot benefit from a
                        # lossy synopsis. Keep it, then permit ordinary L1
                        # retirement only after a second safety grace.
                        if age < threshold + self.policy.summary_grace_seconds:
                            skipped["summary_not_loss_safe"] += 1
                            continue
                        action, reason = "expire_raw", "unreinforced_l1_retirement"
                        if self.policy.mode == "active" and allow_reclaim:
                            saved = len(str(row["text"]).encode("utf-8"))
                            self.store._discard_cached_text(str(row["text"]))
                            conn.execute("UPDATE memory_events SET text='',status='expired',metadata_json='{}' WHERE event_id=? AND status='active'", (rid,))
                            conn.execute("UPDATE memory_lifecycle SET summary_source_hash=content_hash,content_hash=?,representation='discarded',raw_reclaimed=1,version=version+1 WHERE record_type=? AND record_id=?", (digest(""), kind, rid))
                            vector_reclaimed += self._drop_vectors(conn, kind, rid)
                            reclaimed += saved
                            self._audit(conn, state, now, action, reason, target, saved)
                    else:
                        action, reason = "summarize", "local_extractive_verified"
                    if action == "summarize" and self.policy.mode == "active":
                        conn.execute("UPDATE memory_lifecycle SET summary_text=?,summary_source_hash=?,summary_at=?,version=version+1 WHERE record_type=? AND record_id=?", (summary, state["content_hash"], now, kind, rid))
                elif now - state["summary_at"] >= self.policy.summary_grace_seconds:
                    if state["summary_source_hash"] != digest(row["text"]) or not self.summary_valid(row["text"], state["summary_text"]):
                        skipped["stale_summary"] += 1
                        continue
                    action, reason = "reclaim_raw", "verified_summary_grace_elapsed"
                    if self.policy.mode == "active" and allow_reclaim:
                        summary = str(state["summary_text"])
                        self.store._discard_cached_text(str(row["text"]))
                        saved = len(str(row["text"]).encode("utf-8")) - len(summary.encode("utf-8"))
                        # One transaction: text+provenance+state+indices. Raw
                        # archive/log/backup copies are deliberately untouched.
                        metadata = self.store._loads(row["metadata_json"])
                        metadata["lifecycle_representation"] = "summary"
                        metadata["summary_source_hash"] = state["summary_source_hash"]
                        metadata["raw_detail_unavailable"] = True
                        conn.execute("UPDATE memory_events SET text=?,metadata_json=? WHERE event_id=? AND status='active'", (summary, self.store._json(metadata), rid))
                        conn.execute("UPDATE memory_lifecycle SET content_hash=?,representation='summary',raw_reclaimed=1,summary_text='',anchor_effective=?,anchor_turns=?,version=version+1 WHERE record_type=? AND record_id=?", (digest(summary), clock["effective_seconds"], clock["turn_count"], kind, rid))
                        vector_reclaimed += self._drop_vectors(conn, kind, rid)
                        reclaimed += saved
                        self._audit(conn, state, now, action, reason, target, saved)
                if action:
                    actions.append({"record_id": f"{kind}:{rid}", "scope": state["scope"], "action": action, "reason": reason, "strength_ratio": strength, "effective_age_seconds": age, "idle_turns": turns, "before_tier": state["tier"], "proposed_tier": target, "applied": self.policy.mode == "active" and (action not in {"reclaim_raw", "expire", "expire_raw"} or allow_reclaim)})
                    if not (action in {"reclaim_raw", "expire_raw", "expire"} and self.policy.mode == "active" and allow_reclaim):
                        self._audit(conn, state, now, action, reason, target)
            self._prune(conn, now)
            if rollback:
                conn.rollback()
        return {"mode": self.policy.mode, "scanned": len(states), "actions": actions, "skipped": dict(skipped), "logical_bytes_reclaimed": reclaimed + vector_reclaimed, "logical_text_bytes_reclaimed": reclaimed, "logical_vector_bytes_reclaimed": vector_reclaimed, "archive_bytes_reclaimed": 0, "file_compacted": False}

    def set_pin(self, *, record_id: str, scope: str, pinned: bool, now: int, confirmed: bool) -> bool:
        if not confirmed or self.policy.mode != "active":
            raise ValueError("pin_requires_active_and_confirmation")
        kind, rid = self._identity(record_id)
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._record(conn, kind, rid)
            if not self._eligible(row, kind) or row["scope"] != scope:
                return False
            if pinned and sensitive(row["text"]):
                raise ValueError("sensitive_data_not_permanently_protected")
            if not self._enroll(conn, kind, row, now):
                return False
            state = conn.execute("SELECT * FROM memory_lifecycle WHERE record_type=? AND record_id=?", (kind, rid)).fetchone()
            clock = self._clock(conn, scope, now)
            conn.execute("UPDATE memory_lifecycle SET pinned=?,tier=?,enrolled_at=?,anchor_effective=?,anchor_turns=?,version=version+1 WHERE record_type=? AND record_id=?", (int(pinned), 3 if pinned else 2, now, clock["effective_seconds"], clock["turn_count"], kind, rid))
            self._audit(conn, state, now, "pin" if pinned else "unpin", "explicit_user_protection", 3 if pinned else 2)
            return True

    def correct_fact(self, *, record_id: str, scope: str, expected_hash: str, replacement: str, now: int, confirmed: bool) -> bool:
        """Explicit current-fact correction, never a semantic model write.

        Original conversation logs remain historical. Old current-fact hashes
        are revoked so backup restoration cannot silently reinstate old truth.
        """
        if not confirmed or not scope or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise ValueError("correction_requires_scope_hash_confirmation")
        kind, rid = self._identity(record_id)
        replacement = str(replacement).strip()
        if kind != "fact" or not replacement or len(replacement) > 6000 or sensitive(replacement):
            raise ValueError("invalid_or_sensitive_fact_correction")
        invalidated = {record_id}
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._record(conn, kind, rid)
            if row is None or row["status"] != "active" or row["scope"] != scope or digest(row["text"]) != expected_hash:
                return False
            if digest(replacement) == expected_hash:
                return True
            if self.revoked(conn, scope=scope, source_id="", text=replacement):
                raise ValueError("replacement_previously_revoked")
            self.revoke(conn, scope=scope, source_id="", text=row["text"], now=now)
            metadata = self.store._loads(row["metadata_json"])
            if metadata.get("confirmation_kind") == "explicit_memory_request":
                # Confirmation history must never copy the revoked old text.
                metadata.update({"corrected_at": now, "supersedes_hash": expected_hash,
                                 "correction_confirmation": "explicit", "correction_count":
                                 min(1000000, int(metadata.get("correction_count", 0)) + 1)})
                if metadata.get("owner_origin_record_id"):
                    # An explicitly corrected copy is now an independently
                    # confirmed scoped fact, not a derivative of the old value.
                    metadata["corrected_origin_record_id"] = metadata.pop("owner_origin_record_id")
                    metadata["corrected_origin_version"] = metadata.pop("owner_origin_version", "")
                    metadata["corrected_origin_scope"] = metadata.pop("owner_origin_scope", "")
            conn.execute("UPDATE memory_facts SET text=?,source_id=?,updated_at=?,metadata_json=? WHERE fact_id=?",
                         (replacement, f"correction:{rid}:{digest(replacement)[:16]}", now,
                          self.store._json(metadata), rid))
            self._drop_vectors(conn, kind, rid)
            self._enroll(conn, kind, self._record(conn, kind, rid), now)
            # Remove derived views transitively, but keep old conversation
            # turns as history rather than rewriting what somebody once said.
            summaries = conn.execute("SELECT * FROM memory_events WHERE scope=? AND status='active'", (scope,)).fetchall()
            while True:
                previous = len(invalidated)
                for summary in summaries:
                    current_id = f"event:{summary['event_id']}"
                    if current_id in invalidated:
                        continue
                    meta = self.store._loads(summary["metadata_json"])
                    parents = meta.get("source_record_ids", [])
                    if isinstance(parents, list) and invalidated.intersection(str(p) for p in parents):
                        conn.execute("UPDATE memory_events SET status='forgotten' WHERE event_id=?", (summary["event_id"],))
                        self._drop_vectors(conn, "event", summary["event_id"])
                        self.revoke(conn, scope=scope, source_id=summary["source_id"], text=summary["text"], now=now)
                        invalidated.add(current_id)
                if previous == len(invalidated):
                    break
        with self.store._lock:
            self.store._clear_derived_caches(scope=scope, record_ids=invalidated)
        return True

    def status(self) -> dict[str, Any]:
        with self.store._lock, self.store._connect() as conn:
            tiers = {str(row["tier"]): row["n"] for row in conn.execute("SELECT tier,count(*) n FROM memory_lifecycle GROUP BY tier")}
            count = conn.execute("SELECT count(*) FROM memory_lifecycle_evidence").fetchone()[0]
            revoked = conn.execute("SELECT count(*) FROM memory_revocations").fetchone()[0]
            event_bytes = self._protected_bytes(conn, int(self.wall_time()))
            fact_bytes = int(conn.execute("SELECT COALESCE(sum(length(CAST(text AS BLOB))),0) FROM memory_facts WHERE status='active'").fetchone()[0])
        return {"mode": self.policy.mode, "policy": asdict(self.policy), "tiers": tiers, "retained_evidence": count, "revocations": revoked, "worker_running": bool(self._worker and self._worker.is_alive()), "last_error": self._last_error, "physical_reclamation_automatic": False,
            "protected_event_payload_bytes": event_bytes, "explicit_fact_payload_bytes": fact_bytes,
            "protected_budget_warning": event_bytes + fact_bytes > self.policy.auto_raw_budget_bytes}

    def inspect(self, *, scope: str, limit: int = 20) -> dict[str, Any]:
        if not scope:
            raise ValueError("inspection_scope_required")
        cap = max(1, min(100, int(limit)))
        with self.store._lock, self.store._connect() as conn:
            rows = conn.execute("SELECT * FROM memory_lifecycle WHERE scope=? ORDER BY last_reinforced_at DESC,record_id LIMIT ?", (scope, cap)).fetchall()
            audit = conn.execute("SELECT * FROM memory_lifecycle_audit WHERE scope=? ORDER BY sequence DESC LIMIT ?", (scope, cap)).fetchall()
        return {"scope": scope, "records": [{key: row[key] for key in row.keys() if key != "summary_text"} for row in rows], "audit": [dict(row) for row in audit]}

    def export_revocations(self, *, scope: str) -> dict[str, Any]:
        if not scope:
            raise ValueError("revocation_export_scope_required")
        with self.store._lock, self.store._connect() as conn:
            rows = conn.execute("SELECT * FROM memory_revocations WHERE scope=? ORDER BY revoked_at,source_hash,content_hash", (scope,)).fetchall()
        return {"version": 1, "scope": scope, "revocations": [dict(row) for row in rows]}

    def import_revocations(self, payload: dict[str, Any], *, approved_scope: str, confirmed: bool) -> dict[str, Any]:
        """Reapply post-backup revocations BEFORE a restored index is served.

        Only compact hashes are transferred. This disables source/derived
        records; it does not claim media erasure or third-party deletion.
        """
        if not isinstance(payload, dict) or not confirmed or not approved_scope or payload.get("version") != 1 or payload.get("scope") != approved_scope:
            raise ValueError("revocation_import_requires_exact_scope_confirmation")
        records = payload.get("revocations")
        if not isinstance(records, list) or len(records) > 200000:
            raise ValueError("invalid_revocation_import_records")
        for row in records:
            if (not isinstance(row, dict) or set(row) != {"scope", "source_hash", "content_hash", "revoked_at"}
                or row.get("scope") != approved_scope
                or not re.fullmatch(r"[a-f0-9]{64}|", str(row.get("source_hash", "?")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("content_hash", "")))
                or isinstance(row.get("revoked_at"), bool) or not isinstance(row.get("revoked_at"), int) or row["revoked_at"] < 0):
                raise ValueError("invalid_revocation_import_row")
        disabled = 0
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany("INSERT OR IGNORE INTO memory_revocations VALUES (?,?,?,?)", [(row["scope"], row["source_hash"], row["content_hash"], row["revoked_at"]) for row in records])
            revoked_ids: set[str] = set()
            event_rows = conn.execute("SELECT * FROM memory_events WHERE scope=?", (approved_scope,)).fetchall()
            for kind, table, pk in (("event", "memory_events", "event_id"), ("fact", "memory_facts", "fact_id")):
                rows = event_rows if kind == "event" else conn.execute("SELECT * FROM memory_facts WHERE scope=?", (approved_scope,)).fetchall()
                for row in rows:
                    if not self.revoked(conn, scope=approved_scope, source_id=row["source_id"], text=row["text"]):
                        continue
                    rid = str(row[pk])
                    disabled += int(row["status"] == "active")
                    conn.execute(f"UPDATE {table} SET status='forgotten' WHERE {pk}=?", (rid,))
                    conn.execute("DELETE FROM memory_embeddings WHERE record_type=? AND record_id=?", (kind, rid))
                    conn.execute("DELETE FROM memory_lifecycle WHERE record_type=? AND record_id=?", (kind, rid))
                    conn.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type=? AND record_id=?", (kind, rid))
                    revoked_ids.add(f"{kind}:{rid}")
            while True:
                previous = set(revoked_ids)
                for row in event_rows:
                    current_id = f"event:{row['event_id']}"
                    if current_id in revoked_ids:
                        continue
                    meta = self.store._loads(row["metadata_json"])
                    parents = meta.get("source_record_ids", [])
                    linked = isinstance(parents, list) and bool(revoked_ids.intersection(str(item) for item in parents))
                    legacy_untracked = bool(revoked_ids) and meta.get("source") == "local_daily_rollup" and not parents
                    if linked or legacy_untracked:
                        disabled += int(row["status"] == "active")
                        conn.execute("UPDATE memory_events SET status='forgotten' WHERE event_id=?", (row["event_id"],))
                        conn.execute("DELETE FROM memory_embeddings WHERE record_type='event' AND record_id=?", (row["event_id"],))
                        conn.execute("DELETE FROM memory_lifecycle WHERE record_type='event' AND record_id=?", (row["event_id"],))
                        conn.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type='event' AND record_id=?", (row["event_id"],))
                        self.revoke(conn, scope=approved_scope, source_id=row["source_id"], text=row["text"], now=max((item["revoked_at"] for item in records), default=0))
                        revoked_ids.add(current_id)
                if revoked_ids == previous:
                    break
            # SQL listeners use separate connections. Commit source revocations
            # before invoking them, while keeping the store lock through cleanup.
            conn.commit()
            self.store._clear_derived_caches(scope=approved_scope, record_ids=revoked_ids)
        return {"scope": approved_scope, "revocations_received": len(records), "records_disabled": disabled, "physical_erasure": False}

    def start_worker(self) -> None:
        if not self.policy.enabled or (self._worker and self._worker.is_alive()):
            return
        self._stop.clear()
        def run():
            while not self._stop.is_set():
                try:
                    self.scan(now=int(self.wall_time()))
                    self._last_error = ""
                except Exception as exc:
                    self._last_error = type(exc).__name__  # No sensitive bodies.
                self._stop.wait(self.policy.scan_interval_seconds)
        self._worker = threading.Thread(target=run, name="shinku-memory-lifecycle", daemon=True)
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
