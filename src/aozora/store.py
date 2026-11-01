"""独立 SQLite 状态所有者：事件 -> 尝试 -> 原子决策 -> 送达回执。

每个方法使用短连接与事务，网络调用在事务外。同一数据库只持有一个运行中尝试；
崩溃后不按时间偷取占用，宿主先显式取消并核对未知外部效果，再决定是否重试。
旧尝试结果由尝试号挡住。模型仅更新投影，原始事件和历史决策保留用于溯源。
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time

from .contracts import Decision, Event, Identity, KernelError, digest, encode, identifier, text


class Store:
    def __init__(self, path: str | Path, identity: Identity):
        if str(path) == ":memory:":
            raise KernelError("persistent_database_required")
        self.path = Path(path)
        self.identity = identity.payload()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # 新数据库从创建瞬间即为私有，避免 SQLite 默认 umask 导致对话可被其他用户读取。
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with self._transaction() as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = db.execute("PRAGMA user_version").fetchone()[0]
            expected = {"aozora_meta", "events", "attempts", "entries", "outbox"}
            if names and (names != expected or version != 1):
                raise KernelError("not_an_aozora_database")
            if not names and version != 0:
                raise KernelError("unsupported_database_version")
            statements = [
                "CREATE TABLE IF NOT EXISTS aozora_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, scope TEXT NOT NULL, payload TEXT NOT NULL, fingerprint TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS attempts (event_id TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL, context TEXT NOT NULL, result TEXT, error TEXT, started REAL NOT NULL, ended REAL, PRIMARY KEY(event_id, attempt))",
                "CREATE TABLE IF NOT EXISTS entries (scope TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(scope, kind, key))",
                "CREATE TABLE IF NOT EXISTS outbox (delivery_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, attempt INTEGER NOT NULL, scope TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL, channel TEXT, receipt TEXT)",
            ]
            for statement in statements:
                db.execute(statement)
            row = db.execute("SELECT value FROM aozora_meta WHERE key='identity'").fetchone()
            if row is not None and json.loads(row[0]) != self.identity:
                raise KernelError("identity_mismatch")
            db.execute("INSERT OR IGNORE INTO aozora_meta VALUES ('identity', ?)", (encode(self.identity),))
            db.execute("PRAGMA user_version=1")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _latest(self, db, event_id: str, scope: str):
        return db.execute(
            "SELECT a.* FROM attempts a JOIN events e USING(event_id) WHERE a.event_id=? AND e.scope=? ORDER BY attempt DESC LIMIT 1",
            (event_id, scope),
        ).fetchone()

    def _view(self, db, row) -> dict:
        if row is None:
            raise KernelError("event_not_found")
        outbox = db.execute("SELECT * FROM outbox WHERE event_id=? AND attempt=?", (row["event_id"], row["attempt"])).fetchone()
        return {"event_id": row["event_id"], "attempt": row["attempt"], "status": row["status"],
                "error": row["error"], "decision": json.loads(row["result"]) if row["result"] else None,
                "delivery": dict(outbox) if outbox else None}

    def status(self, event_id: str, scope: str) -> dict:
        with self._transaction() as db:
            return self._view(db, self._latest(db, event_id, scope))

    def _snapshot(self, db, scope: str) -> dict:
        events = []
        for row in reversed(db.execute("SELECT payload FROM events WHERE scope=? ORDER BY seq DESC LIMIT 8", (scope,)).fetchall()):
            event = json.loads(row[0])
            event["source"] = {"message": "reported", "observation": "host_observation", "tick": "host_tick"}[event["kind"]]
            events.append(event)
        entries = {"attention": [], "understandings": []}
        for row in db.execute("SELECT kind, payload FROM entries WHERE scope=? ORDER BY kind, key", (scope,)):
            entry = json.loads(row["payload"])
            if row["kind"] == "attention" and entry["status"] != "open":
                continue
            entries[row["kind"]].append(entry)
        # 往日理解/关注保留其来源；上下文回填有界，缺失来源绝不靠模型补写。
        known = {event["event_id"] for event in events}
        refs = {ref for rows in entries.values() for row in rows for ref in row["evidence"]}
        sources = []
        for ref in sorted(refs - known):
            row = db.execute("SELECT payload FROM events WHERE event_id=? AND scope=?", (ref, scope)).fetchone()
            if row is None:
                raise KernelError("source_missing")
            event = json.loads(row[0])
            event["source"] = {"message": "reported", "observation": "host_observation", "tick": "host_tick"}[event["kind"]]
            sources.append(event)
        deliveries = [dict(row) for row in db.execute(
            "SELECT delivery_id, event_id, text, status, channel, receipt FROM outbox WHERE scope=? ORDER BY rowid DESC LIMIT 8", (scope,))]
        snapshot = {"identity": self.identity, "scope": scope, "events": events,
                    "sources": sources, **entries, "deliveries": list(reversed(deliveries))}
        if len(encode(snapshot).encode()) > 256 * 1024:
            raise KernelError("context_capacity_exceeded")
        return snapshot

    def snapshot(self, scope: str) -> dict:
        identifier(scope)
        with self._transaction() as db:
            return json.loads(encode(self._snapshot(db, scope)))

    def begin(self, event: Event, *, retry: bool = False) -> tuple[dict, dict | None]:
        payload = event.payload()
        with self._transaction() as db:
            existing = db.execute("SELECT fingerprint FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if existing and existing[0] != digest(payload):
                raise KernelError("event_id_conflict")
            old = self._latest(db, event.event_id, event.scope)
            if old and (not retry or old["status"] in {"running", "completed"}):
                return self._view(db, old), None
            if db.execute("SELECT 1 FROM attempts WHERE status='running' LIMIT 1").fetchone():
                raise KernelError("decision_busy")
            db.execute("INSERT OR IGNORE INTO events(event_id,scope,payload,fingerprint) VALUES (?,?,?,?)",
                       (event.event_id, event.scope, encode(payload), digest(payload)))
            context = self._snapshot(db, event.scope)
            context["current_event_id"] = event.event_id
            # 很久以前失败的事件可以显式重试，不能因最近八条投影而丢失当前输入。
            if event.event_id not in {item["event_id"] for item in context["events"]}:
                context["events"].append({**payload, "source": {
                    "message": "reported", "observation": "host_observation", "tick": "host_tick"}[event.kind]})
            if len(encode(context).encode()) > 256 * 1024:
                raise KernelError("context_capacity_exceeded")
            attempt = old["attempt"] + 1 if old else 1
            db.execute("INSERT INTO attempts(event_id,attempt,status,context,started) VALUES (?,?,'running',?,?)",
                       (event.event_id, attempt, encode(context), time.time()))
            return self._view(db, self._latest(db, event.event_id, event.scope)), context

    def finish(self, event: Event, attempt: int, decision: Decision) -> dict:
        # 在事务外也校验一次，随后仍从落库上下文验证引用，绝不信任模型回传的上下文。
        decision = Decision.parse(decision.payload())
        with self._transaction() as db:
            row = self._latest(db, event.event_id, event.scope)
            if row is None or row["attempt"] != attempt or row["status"] != "running":
                raise KernelError("stale_attempt")
            context = json.loads(row["context"])
            # 发送端可在模型思考期间确认旧回复；变化后的送达事实不能沿旧投影继续提交。
            for prior in context["deliveries"]:
                current = db.execute("SELECT status, channel, receipt FROM outbox WHERE delivery_id=? AND scope=?",
                                     (prior["delivery_id"], event.scope)).fetchone()
                if current is None or any(current[key] != prior[key] for key in ("status", "channel", "receipt")):
                    raise KernelError("delivery_context_changed")
            evidence = {item["event_id"] for item in context["events"] + context["sources"]}
            for kind, changes in (("attention", decision.attention), ("understandings", decision.understandings)):
                visible = {item["key"]: item for item in context[kind]}
                for change in changes:
                    if not set(change["evidence"]) <= evidence:
                        raise KernelError("unseen_evidence")
                    old = db.execute("SELECT payload FROM entries WHERE scope=? AND kind=? AND key=?",
                                     (event.scope, kind, change["key"])).fetchone()
                    if old:
                        if change["key"] not in visible or change["version"] != visible[change["key"]]["version"]:
                            raise KernelError("state_version_conflict")
                    elif change["version"] != 0 or (kind == "attention" and change["status"] != "open"):
                        raise KernelError("invalid_new_state")
                    updated = {**change, "version": change["version"] + 1,
                               "source": "model_interpretation", "updated_by": event.event_id}
                    db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?)", (event.scope, kind, change["key"], encode(updated)))
                rows = db.execute("SELECT payload FROM entries WHERE scope=? AND kind=?", (event.scope, kind)).fetchall()
                if sum(json.loads(item[0])["status"] in {"open", "tentative"} for item in rows) > 16:
                    raise KernelError("state_capacity_exceeded")
            # 新状态必须仍可投影；超限时连同上述所有更新回滚，保留已接纳事件和失败尝试。
            self._snapshot(db, event.scope)
            if decision.action != "wait":
                delivery_id = digest({"event": event.event_id, "attempt": attempt})[:32]
                db.execute("INSERT INTO outbox(delivery_id,event_id,attempt,scope,text,status) VALUES (?,?,?,?,?,'prepared')",
                           (delivery_id, event.event_id, attempt, event.scope, decision.reply))
            db.execute("UPDATE attempts SET status='completed', result=?, ended=? WHERE event_id=? AND attempt=?",
                       (encode(decision.payload()), time.time(), event.event_id, attempt))
            return self._view(db, self._latest(db, event.event_id, event.scope))

    def fail(self, event: Event, attempt: int, code: str) -> None:
        with self._transaction() as db:
            db.execute("UPDATE attempts SET status='failed', error=?, ended=? WHERE event_id=? AND attempt=? AND status='running'",
                       (text(code, 128), time.time(), event.event_id, attempt))

    def cancel(self, event_id: str, scope: str, *, expected_attempt: int | None = None) -> dict:
        """取消状态提交资格；不能中断已发出的 HTTP，也不表示调用免费或已撤销。"""
        with self._transaction() as db:
            row = self._latest(db, event_id, scope)
            if row is None:
                raise KernelError("event_not_found")
            if row["status"] == "running" and (expected_attempt is None or row["attempt"] == expected_attempt):
                db.execute("UPDATE attempts SET status='cancelled', ended=? WHERE event_id=? AND attempt=?",
                           (time.time(), event_id, row["attempt"]))
            return self._view(db, self._latest(db, event_id, scope))

    def claim_delivery(self, delivery_id: str, scope: str, channel: str) -> dict:
        """发送前持久占用。unknown 包含发送中/崩溃窗口，只有一次 claim 会返回正文。"""
        identifier(channel)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM outbox WHERE delivery_id=? AND scope=?", (delivery_id, scope)).fetchone()
            if row is None:
                raise KernelError("delivery_not_found")
            if row["status"] != "prepared":
                raise KernelError("delivery_not_prepared")
            db.execute("UPDATE outbox SET status='unknown', channel=? WHERE delivery_id=?", (channel, delivery_id))
            return {**dict(row), "status": "unknown", "channel": channel}

    def resolve_delivery(self, delivery_id: str, scope: str, *, delivered: bool, receipt: str) -> None:
        """仅供受信发送端/人工核验调用；模型没有此端口。失败也不自动重发。"""
        text(receipt, 512)
        if type(delivered) is not bool:
            raise KernelError("invalid_receipt")
        status = "delivered" if delivered else "failed"
        with self._transaction() as db:
            row = db.execute("SELECT * FROM outbox WHERE delivery_id=? AND scope=?", (delivery_id, scope)).fetchone()
            if row is None:
                raise KernelError("delivery_not_found")
            if row["status"] == status and row["receipt"] == receipt:
                return
            if row["status"] != "unknown":
                raise KernelError("delivery_not_claimed")
            db.execute("UPDATE outbox SET status=?, receipt=? WHERE delivery_id=?", (status, receipt, delivery_id))
