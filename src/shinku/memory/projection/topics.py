"""Small handle-only continuation cache. Source text remains in canonical tables."""
from __future__ import annotations

import json
import re

from .sources import connection


class TopicState:
    def __init__(self, store, policy):
        self.store, self.policy = store, policy
        with store._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_projection_topics (
                scope TEXT NOT NULL, requester TEXT NOT NULL, topic_id TEXT NOT NULL,
                handles_json TEXT NOT NULL, updated_at INTEGER NOT NULL, policy TEXT NOT NULL,
                PRIMARY KEY(scope,requester))""")
            # 模型实际读取的来源另存多组定位符；不复用旧的关键词续聊或单主题判定。
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_projection_read_anchors (
                scope TEXT NOT NULL, requester TEXT NOT NULL, self_id TEXT NOT NULL,
                anchor_id TEXT NOT NULL, message_id TEXT NOT NULL, handles_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL, policy TEXT NOT NULL,
                PRIMARY KEY(scope,requester,self_id,anchor_id))""")

    def remember_read(self, request, records, deadline):
        """只保存宿主已读取并展示的原件定位符；不缓存正文、答案或模型推断。"""
        from .contracts import digest
        if not request.self_id or not request.message_id or not records:
            return
        handles = [(r.record_id, r.scope, r.metadata["source_version"]) for r in records
                   if r.scope in request.allowed_scopes][:8]
        anchor = digest([request.message_id, handles])
        with connection(self.store, deadline, write=True) as conn:
            conn.execute("INSERT OR REPLACE INTO memory_projection_read_anchors VALUES(?,?,?,?,?,?,?,?)",
                (request.scope, request.requester, request.self_id, anchor, request.message_id,
                 json.dumps(handles), request.now, self.policy.version))
            conn.execute("DELETE FROM memory_projection_read_anchors WHERE updated_at<?",
                         (request.now - self.policy.cache_seconds,))
            conn.execute("""DELETE FROM memory_projection_read_anchors WHERE rowid IN
                (SELECT rowid FROM memory_projection_read_anchors
                 WHERE scope=? AND requester=? AND self_id=? ORDER BY updated_at DESC,rowid DESC LIMIT -1 OFFSET 4)""",
                (request.scope, request.requester, request.self_id))
            conn.execute("""DELETE FROM memory_projection_read_anchors WHERE rowid IN
                (SELECT rowid FROM memory_projection_read_anchors ORDER BY updated_at DESC,rowid DESC LIMIT -1 OFFSET ?)""",
                (self.policy.cache_entries,))

    def read_anchors(self, request, deadline):
        """同会话、同人物、同账号多组承接；每次仍由调用方重新核验版本和授权。"""
        if not request.self_id:
            return ()
        with connection(self.store, deadline) as conn:
            rows = conn.execute("""SELECT handles_json FROM memory_projection_read_anchors
                WHERE scope=? AND requester=? AND self_id=? AND policy=? AND updated_at BETWEEN ? AND ?
                  AND message_id<>? ORDER BY updated_at DESC,rowid DESC LIMIT 4""",
                (request.scope, request.requester, request.self_id, self.policy.version,
                 request.now-self.policy.cache_seconds, request.now, request.message_id)).fetchall()
        groups = [json.loads(row[0]) for row in rows]
        # 轮流保留各组候选，不让最近一次读取覆盖仍在讨论的另一件事。
        handles = []
        for index in range(8):
            for group in groups:
                if index < len(group):
                    handle = tuple(group[index])
                    if (len(handle) == 3 and all(isinstance(v, str) for v in handle)
                            and handle[1] in request.allowed_scopes and handle not in handles):
                        handles.append(handle)
            if len(handles) >= 8:
                break
        return tuple(handles[:8])

    def continuation(self, request, deadline):
        followup = bool(re.search(r"^(?:那|然后|后来|她|他|继续|接着|刚才|这个|那个)|之前那", request.query))
        if not followup or request.flow_reason in {"different_speaker", "different_visual_source", "topic_changed"}:
            return (), None
        with connection(self.store, deadline) as conn:
            row = conn.execute("SELECT * FROM memory_projection_topics WHERE scope=? AND requester=?",
                               (request.scope, request.requester)).fetchone()
        if (not row or row["policy"] != self.policy.version
                or not 0 <= request.now-row["updated_at"] <= self.policy.cache_seconds):
            return (), None
        handles = json.loads(row["handles_json"])
        if not isinstance(handles,list) or len(handles)>self.policy.source_limit:
            return (), None
        return tuple(tuple(h) for h in handles if isinstance(h,list) and len(h)==3
                     and all(isinstance(v,str) for v in h) and h[1] in request.allowed_scopes), row["topic_id"]

    def save(self, request, topic, records, deadline):
        handles = [(r.record_id, r.scope, r.metadata["source_version"]) for r in records][:self.policy.source_limit]
        with connection(self.store, deadline, write=True) as conn:
            conn.execute("""INSERT INTO memory_projection_topics VALUES(?,?,?,?,?,?)
                ON CONFLICT(scope,requester) DO UPDATE SET topic_id=excluded.topic_id,
                handles_json=excluded.handles_json,updated_at=excluded.updated_at,policy=excluded.policy""",
                (request.scope, request.requester, topic["topic_id"], json.dumps(handles), request.now, self.policy.version))
            conn.execute("""DELETE FROM memory_projection_topics WHERE rowid IN
                (SELECT rowid FROM memory_projection_topics ORDER BY updated_at DESC,rowid DESC LIMIT -1 OFFSET ?)""",
                (self.policy.cache_entries,))
