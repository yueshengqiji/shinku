"""独立 Agent 记忆库及本库 FTS 索引；没有 Bot 数据库连接或路径回退。

宿主只传专属目录，文件名固定。已有库必须带专用 application_id 与身份表；
软链接、硬链接、外来表或宽松文件权限拒绝，不尝试修复/迁移外来库。
索引和记忆同事务更新；忘记留下无正文墓碑，自动经验索引不能使其复活。
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

from shinku.agent_runtime.contracts import canonical_hash
from .retrieval import AgentMemoryPolicy, terms, match_query

APPLICATION_ID = 0x41474D31


class AgentMemoryStore:
    def __init__(self, root, *, policy=None, clock=time.time):
        self.root = Path(root).absolute()
        self.path = self.root / 'agent-memory.sqlite3'
        self.policy = policy or AgentMemoryPolicy()
        self.clock, self.lock = clock, threading.RLock()
        self._paths()
        if self.root.exists() and any(p.name not in {'agent-memory.sqlite3', 'agent-memory.sqlite3-journal'}
                                      for p in self.root.iterdir()):
            raise ValueError('agent_memory_directory_not_dedicated')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        fresh = False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
            fresh = True
        with self.lock, sqlite3.connect(self.path, timeout=1) as db:
            if fresh:
                db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                db.execute('CREATE TABLE agent_memory_identity (name TEXT PRIMARY KEY, version INTEGER NOT NULL)')
                db.execute("INSERT INTO agent_memory_identity VALUES ('angelos-agent-only',1)")
                db.execute('''CREATE TABLE memories (id TEXT PRIMARY KEY, principal TEXT NOT NULL,
                    workspace TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    version TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)''')
                db.execute('CREATE INDEX memory_scope ON memories(principal,workspace,deleted)')
                db.execute('''CREATE TABLE forgotten_sources (principal TEXT, workspace TEXT,
                    source_key TEXT, PRIMARY KEY(principal,workspace,source_key))''')
                db.execute('CREATE VIRTUAL TABLE memory_fts USING fts5(id UNINDEXED, terms)')
            self._identity(db)
        self._paths()

    def _paths(self):
        for path in (self.root, *self.root.parents, self.path):
            if path.is_symlink():
                raise ValueError('agent_memory_symlink_forbidden')
        if self.path.exists():
            st = self.path.stat()
            if not self.path.is_file() or st.st_nlink != 1 or st.st_mode & 0o077:
                raise ValueError('agent_memory_file_boundary_invalid')

    @staticmethod
    def _identity(db):
        try:
            row = db.execute('SELECT name,version FROM agent_memory_identity').fetchall()
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            allowed = {'agent_memory_identity', 'memories', 'forgotten_sources', 'memory_fts', 'memory_fts_data',
                       'memory_fts_idx', 'memory_fts_content', 'memory_fts_docsize', 'memory_fts_config'}
            if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
                    or row != [('angelos-agent-only', 1)] or tables != allowed):
                raise ValueError('agent_memory_database_identity_mismatch')
        except sqlite3.DatabaseError as exc:
            raise ValueError('agent_memory_database_identity_mismatch') from exc

    @contextmanager
    def transaction(self):
        with self.lock:
            self._paths()
            with sqlite3.connect(self.path.as_uri()+'?mode=rw', uri=True, timeout=1) as db:
                self._identity(db)
                db.execute('BEGIN IMMEDIATE')
                yield db

    def put(self, *, principal, workspace, key, kind, text, source):
        if (kind not in {'user_statement', 'task_experience'} or not all(
                isinstance(x, str) and 0 < len(x) <= 256 for x in (principal, workspace, key))
                or not isinstance(text, str) or not 0 < len(text) <= (12000 if kind == 'user_statement' else 4096)
                or not isinstance(source, dict)):
            raise ValueError('agent_memory_record_invalid')
        payload = {'kind': kind, 'text': text, 'source': source,
                   'is_current_state_proof': False, 'grants_permissions': False}
        version = canonical_hash(payload)
        identifier = canonical_hash([principal, workspace, key])
        with self.transaction() as db:
            source_key = self.source_key(source)
            if source_key and db.execute('SELECT 1 FROM forgotten_sources WHERE principal=? AND workspace=? AND source_key=?',
                                         (principal,workspace,source_key)).fetchone():
                return {'id':identifier,'status':'forgotten','version':canonical_hash(['forgotten',identifier])}
            old = db.execute('SELECT deleted,version FROM memories WHERE id=?', (identifier,)).fetchone()
            if old and old[0]:
                return {'id': identifier, 'status': 'forgotten', 'version': old[1]}
            if old and old[1] == version:
                return {'id': identifier, 'status': 'unchanged', 'version': version}
            if old is None and db.execute('SELECT COUNT(*) FROM memories').fetchone()[0] >= self.policy.max_records:
                raise ValueError('agent_memory_capacity_reached')
            db.execute('INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,0,?)',
                (identifier, principal, workspace, kind, json.dumps(payload, ensure_ascii=False), version, self.clock()))
            db.execute('DELETE FROM memory_fts WHERE id=?', (identifier,))
            db.execute('INSERT INTO memory_fts VALUES (?,?)', (identifier, ' '.join(terms(text))))
        return {'id': identifier, 'status': 'saved', 'version': version}

    def search(self, principal, workspace, query):
        match = match_query(query, self.policy)
        if not match:
            return []
        with self.transaction() as db:
            rows = db.execute('''SELECT m.id,m.version,m.payload FROM memory_fts f
                JOIN memories m ON m.id=f.id WHERE memory_fts MATCH ?
                AND m.principal=? AND m.workspace=? AND m.deleted=0
                ORDER BY bm25(memory_fts),m.id LIMIT ?''',
                (match, principal, workspace, self.policy.candidate_limit)).fetchall()
        result = []
        for identifier, version, raw in rows:
            payload = json.loads(raw)
            if canonical_hash(payload) != version:
                raise ValueError('agent_memory_record_integrity_error')
            result.append({'id': identifier, 'version': version, **payload})
        return result

    def get(self, principal, workspace, identifier):
        with self.transaction() as db:
            row = db.execute('SELECT version,payload FROM memories WHERE id=? AND principal=? AND workspace=? AND deleted=0',
                             (identifier, principal, workspace)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[1])
        if canonical_hash(payload) != row[0]:
            raise ValueError('agent_memory_record_integrity_error')
        return {'id': identifier, 'version': row[0], **payload}

    def forget(self, principal, workspace, identifier):
        with self.transaction() as db:
            row = db.execute('SELECT deleted,payload FROM memories WHERE id=? AND principal=? AND workspace=?',
                             (identifier, principal, workspace)).fetchone()
            if row is None:
                raise ValueError('agent_memory_not_found')
            source_key = self.source_key(json.loads(row[1]).get('source',{})) if not row[0] else None
            identifiers = {identifier}
            if source_key:
                db.execute('INSERT OR IGNORE INTO forgotten_sources VALUES (?,?,?)',(principal,workspace,source_key))
                # 同一原输入可能同时产生用户陈述与任务经验；一起撤销，防止
                # 忘掉陈述却从经验副本召回。墓碑也拦截随后到达的经验同步。
                for key,payload in db.execute('SELECT id,payload FROM memories WHERE principal=? AND workspace=? AND deleted=0',
                                               (principal,workspace)):
                    if self.source_key(json.loads(payload).get('source',{})) == source_key:
                        identifiers.add(key)
            for key in identifiers:
                db.execute("UPDATE memories SET deleted=1,payload='{}',version=?,updated=? WHERE id=?",
                           (canonical_hash(['forgotten', key]), self.clock(), key))
                db.execute('DELETE FROM memory_fts WHERE id=?', (key,))
        return {'id': identifier, 'status': 'forgotten'}

    def forget_session(self, principal, session):
        """删除会话时跨工作区撤回其全部来源，保留无正文墓碑防止经验复活。"""
        count = 0
        with self.transaction() as db:
            rows = db.execute('SELECT id,workspace,payload FROM memories WHERE principal=? AND deleted=0',
                              (principal,)).fetchall()
            for identifier, workspace, payload in rows:
                source = json.loads(payload).get('source', {})
                if source.get('session') != session:
                    continue
                key = self.source_key(source)
                if key:
                    db.execute('INSERT OR IGNORE INTO forgotten_sources VALUES (?,?,?)', (principal, workspace, key))
                db.execute("UPDATE memories SET deleted=1,payload='{}',version=?,updated=? WHERE id=?",
                           (canonical_hash(['forgotten', identifier]), self.clock(), identifier))
                db.execute('DELETE FROM memory_fts WHERE id=?', (identifier,))
                count += 1
        return count

    @staticmethod
    def source_key(source):
        if source.get('session') and source.get('task_id'):
            return canonical_hash([source['session'],source['task_id']])
        return None

    def source_forgotten(self, principal, workspace, session, task_id):
        key = self.source_key({'session':session,'task_id':task_id})
        with self.transaction() as db:
            return bool(db.execute('SELECT 1 FROM forgotten_sources WHERE principal=? AND workspace=? AND source_key=?',
                                   (principal,workspace,key)).fetchone())
