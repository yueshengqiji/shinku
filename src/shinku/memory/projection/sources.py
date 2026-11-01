"""Deadline-bound read transactions reuse the canonical source validator."""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import re

from ..source_reader import MemorySourceReader
from ..context_sources import SourceReadBatch


# A cue reserves context; it never decides what is true or authorizes an action.
_CLARIFICATION_CUE = re.compile(
    r"澄清|纠正|不是.{0,12}(?:玩笑|开玩笑|编的|那个意思)|"
    r"(?:说的|其实|确实).{0,6}(?:是真的|真实)|认真|这下信|当时给你拍"
)


@contextmanager
def connection(store, deadline, *, write=False):
    deadline.check()
    conn = sqlite3.connect(store.path, timeout=min(.1, deadline.remaining))
    conn.row_factory = sqlite3.Row
    conn.set_progress_handler(lambda: int(not deadline.remaining), 300)
    try:
        if not write:
            conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield conn
        if write:
            deadline.check()
            conn.commit()
    finally:
        conn.rollback()
        conn.close()


class ProjectionSources(MemorySourceReader):
    @staticmethod
    def turn_position(conn, request):
        """统一来源选择和复核的消息边界；处理时钟仅用于实时失效检查。

        队列等待期间可能写入上一轮回复，不能把它当作到达前已存在的上下文。
        无持久消息的兼容调用返回 None，仍由原调用方使用当前时间。
        """
        source = f"turn:{request.scope.split(':', 1)[-1]}:{request.message_id}"
        return conn.execute("""SELECT rowid,created_at FROM memory_events
            WHERE scope=? AND source_id=? ORDER BY rowid LIMIT 1""",
            (request.scope, source)).fetchone() if request.message_id else None

    def _snapshot(self, conn, record_id, scope, **kwargs):
        snapshot = super()._snapshot(conn, record_id, scope, **kwargs)
        if snapshot is None:
            return None
        record = snapshot[0]
        meta = record.metadata
        origin_scope = str(meta.get("owner_source_scope") or "")
        if meta.get("owner_memory") and origin_scope and origin_scope != scope and record.source_id.startswith("owner:"):
            # An owner copy is a capability to an attributed origin, not an
            # independent version that may survive a correction of that origin.
            kind = "fact" if record.layer == "fact" else "event"
            table, pk = ("memory_facts","fact_id") if kind=="fact" else ("memory_events","event_id")
            origin = conn.execute(f"SELECT {pk} FROM {table} WHERE scope=? AND source_id=?",
                                  (origin_scope,record.source_id.removeprefix("owner:"))).fetchone()
            if origin is None:
                return self._reject(kwargs.get("_rejections"), "owner_origin_unavailable")
            passed = {**kwargs, "_path":(*kwargs.get("_path", ()), record_id)}
            original = self._snapshot(conn,kind+":"+origin[0],origin_scope,**passed)
            if (original is None or original[0].text != record.text
                    or original[0].metadata.get("sender_id") != meta.get("sender_id")):
                return self._reject(kwargs.get("_rejections"), "owner_origin_changed")
            if original[0].created_at != record.created_at:
                # 群聊被动观察使用平台时间，正式回合可能几秒后才保存共享副本。
                # 只允许同源正式 user 回合证明第二个时间；正文、作者及撤销仍依赖原事件。
                turn = conn.execute("SELECT * FROM memory_turns WHERE scope=? AND source_id=? AND role='user'",
                    (origin_scope, record.source_id.removeprefix("owner:"))).fetchone()
                detail = json.loads(turn["metadata_json"]) if turn is not None else {}
                if (original[0].metadata.get("source") != "qq_passive_observation"
                        or meta.get("source") != "qq_turn" or meta.get("role") != "user"
                        or turn is None or turn["content"] != record.text
                        or turn["created_at"] != record.created_at
                        or detail.get("sender_id") != meta.get("sender_id")):
                    return self._reject(kwargs.get("_rejections"), "owner_origin_changed")
        return snapshot

    def read(self, handles, request, deadline):
        return self.read_checked(handles, request, deadline).records

    def read_checked(self, handles, request, deadline):
        """复用同一次来源读取，按输入槽位反馈固定原因码；不额外检索或查库猜因。"""
        if len(handles) > 256:
            raise ValueError("source_batch_unbounded")
        result, reasons = [], []
        with connection(self.store, deadline) as conn:
            current = self.turn_position(conn, request)
            for rid, scope, version in handles:
                deadline.check()
                if scope not in request.allowed_scopes:
                    result.append(None)
                    reasons.append("scope_not_allowed")
                    continue
                rejected = []
                snapshot = self._snapshot(conn, rid, scope, _now=request.now, _rejections=rejected)
                if snapshot is not None and current is not None:
                    record = snapshot[0]
                    position = conn.execute("SELECT rowid FROM memory_events WHERE event_id=?",
                        (rid.removeprefix("event:"),)).fetchone() if rid.startswith("event:") else None
                    if (record.created_at > current["created_at"] or
                            (record.created_at == current["created_at"] and position is not None
                             and position[0] >= current["rowid"] and scope == request.scope)):
                        snapshot = None
                        rejected.append("source_after_current_turn")
                if snapshot is not None and version and version != snapshot[1]:
                    snapshot = None
                    rejected.append("source_version_changed")
                if snapshot is not None and snapshot[0].metadata.get("owner_participant_id"):
                    # 来源有效不等于本轮仍获准读该参与者；发布前复核也使用当前配置。
                    if (scope != "private:master" or not request.scope.startswith("private:")
                            or snapshot[0].metadata["owner_participant_id"] not in request.participant_ids):
                        snapshot = None
                        rejected.append("scope_not_allowed")
                result.append(self._reading_source(snapshot) if snapshot is not None else None)
                reasons.append("" if snapshot is not None else (rejected[0] if rejected else "source_unavailable"))
        return SourceReadBatch(tuple(result), tuple(reasons))

    def dialogue(self, request, deadline, *, pairs=3, seconds=1800, recent_limit=12,
                 preserve_addressed=False):
        """Reserve attributed adjacent exchanges, not a semantic topic verdict.

        Only this requester's turns in the current scope qualify. The current
        event's position bounds discovery so a later queued correction cannot
        replace the context of an already running turn. All handles still pass
        the normal source/version/delivery validator in ``read``.
        """
        if request.start_at is not None or request.end_at is not None:
            return ()
        handles = []
        with connection(self.store, deadline) as conn:
            source = f"turn:{request.scope.split(':', 1)[-1]}:{request.message_id}"
            current = self.turn_position(conn, request)
            cutoff = min(request.now, current["created_at"]) if current else request.now
            position = current["rowid"] if current and current["created_at"] == cutoff else 2**63 - 1
            tail = set()
            if request.scope.startswith("group:"):
                tail = {r[0] for r in conn.execute("""SELECT event_id FROM memory_events
                    WHERE scope=? AND status='active' AND layer IN ('episodic','observation')
                      AND (created_at<? OR (created_at=? AND rowid<?))
                    ORDER BY created_at DESC,rowid DESC LIMIT ?""",
                    (request.scope, cutoff, cutoff, position, min(64, recent_limit))).fetchall()}
            rows = conn.execute("""SELECT event_id,source_id,created_at,text FROM memory_events
                WHERE scope=? AND status='active' AND layer IN ('episodic','observation')
                  AND source_id LIKE 'turn:%'
                  AND COALESCE(json_extract(metadata_json,'$.role'),'') IN ('','user')
                  AND json_extract(metadata_json,'$.sender_id')=?
                  AND created_at>=? AND (created_at<? OR (created_at=? AND rowid<?))
                  AND source_id<>?
                ORDER BY created_at DESC,rowid DESC LIMIT ?""",
                (request.scope, request.requester, cutoff - min(seconds, 1800),
                 cutoff, cutoff, position, source, min(12, max(3, pairs * 4)))).fetchall()
            selected_pairs = 0
            for row in rows:
                reply = conn.execute("""SELECT event_id FROM memory_events
                    WHERE scope=? AND source_id=? AND status='active'
                      AND created_at<=? AND json_extract(metadata_json,'$.role')='assistant'
                    ORDER BY created_at DESC LIMIT 1""",
                    (request.scope, row["source_id"] + ":assistant", cutoff)).fetchone()
                if request.scope.startswith("group:"):
                    # Same-author group traffic is not an addressed exchange.
                    # Only an independently validated bot delivery links it.
                    from .view import admissible
                    snapshot = self._snapshot(conn, "event:" + reply["event_id"], request.scope,
                                              _now=request.now) if reply else None
                    if (snapshot is None or snapshot[0].metadata.get("requester_id") != request.requester
                            or not admissible(self._reading_source(snapshot), request)):
                        continue
                    # Ordinary addressed group dialogue still ages out with
                    # group traffic. A clarification cue may recover its own
                    # adjacent exchanges, but cannot recover a foreign author.
                    if (not preserve_addressed and row["event_id"] not in tail and not _CLARIFICATION_CUE.search(
                            request.query + "\n" + row["text"])):
                        continue
                handles.append(("event:" + row["event_id"], request.scope, ""))
                if reply:
                    handles.append(("event:" + reply["event_id"], request.scope, ""))
                selected_pairs += 1
                if selected_pairs >= max(1, min(3, pairs)):
                    break
        return tuple(handles)

    def recent(self, request, deadline, limit, seconds):
        with connection(self.store, deadline) as conn:
            rows = conn.execute("""SELECT event_id,scope FROM memory_events
                WHERE scope=? AND status='active' AND created_at BETWEEN ? AND ?
                  AND layer IN ('episodic','observation')
                  AND source_id NOT LIKE 'owner:agent-task-%'
                ORDER BY created_at DESC,rowid DESC LIMIT ?""",
                (request.scope, request.now-seconds, request.now, limit)).fetchall()
        return tuple(("event:"+r["event_id"],r["scope"],"") for r in rows)

    def linked(self, records, request, deadline, limit):
        """Follow explicit message/parent edges; never infer group adjacency."""
        handles, missing = [], []
        with connection(self.store, deadline) as conn:
            # A cold-start reply handle is useful even when there are no recent
            # records from which to begin traversing.
            for mid in request.reply_message_ids[:8]:
                rows = conn.execute("""SELECT event_id FROM memory_events WHERE scope=? AND status='active'
                    AND (json_extract(metadata_json,'$.message_id')=? OR source_id=?
                    OR EXISTS (SELECT 1 FROM json_each(memory_events.metadata_json,'$.delivery_message_ids') WHERE value=?))
                    ORDER BY created_at DESC LIMIT 2""",
                    (request.scope, mid, f"turn:{request.scope.split(':',1)[-1]}:{mid}",mid)).fetchall()
                if not rows:
                    missing.append({"reason":"missing_predecessor", "message_id":str(mid)[:80]})
                handles.extend(("event:"+r[0],request.scope,"") for r in rows)
            for record in records:
                deadline.check()
                meta = record.metadata
                ids = list(meta.get("reply_message_ids") or ())[:8]
                if meta.get("reply_to_message_id"):
                    ids.append(str(meta["reply_to_message_id"]))
                for mid in dict.fromkeys(str(x) for x in ids):
                    rows = conn.execute("""SELECT event_id FROM memory_events WHERE scope=?
                        AND status='active' AND (json_extract(metadata_json,'$.message_id')=?
                        OR source_id=?) ORDER BY created_at DESC LIMIT 2""",
                        (record.scope, mid, f"turn:{record.scope.split(':',1)[-1]}:{mid}")).fetchall()
                    if not rows:
                        missing.append({"reason":"missing_predecessor","source_id":record.record_id})
                    handles.extend(("event:"+r[0],record.scope,"") for r in rows)
                for rid in list(meta.get("source_record_ids") or ())[:8]:
                    handles.append((str(rid), record.scope, ""))
                source = record.source_id
                if source.startswith(("turn:", "owner:turn:")):
                    target = source.removesuffix(":assistant") if source.endswith(":assistant") else source+":assistant"
                    row=conn.execute("SELECT event_id FROM memory_events WHERE scope=? AND source_id=? AND status='active'",
                                     (record.scope,target)).fetchone()
                    if row:
                        handles.append(("event:"+row[0],record.scope,""))
                    elif source.endswith(":assistant"):
                        missing.append({"reason":"missing_reply_context", "source_id":record.record_id})
                if len(handles) >= limit:
                    break
        return tuple(dict.fromkeys(handles))[:limit], missing
