"""工作台记录的唯一写入者。

SQLite 保存工作区、会话及任务结果的展示副本；执行事实仍由 AgentPort 提供。
事务只覆盖本机记录，模型请求和工具执行永远在事务之外。文件不进入 Git。
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class WorkbenchStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS items (kind TEXT, id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            if not db.execute("SELECT 1 FROM items WHERE kind='workspace'").fetchone():
                self.put(db, 'workspace', {'id': 'default', 'name': '日常工作', 'path': '', 'description': ''})
        self.path.chmod(0o600)

    @contextmanager
    def transaction(self):
        with self.lock, sqlite3.connect(self.path, timeout=10) as db:
            db.execute('BEGIN IMMEDIATE')
            yield db

    @staticmethod
    def put(db, kind, value):
        db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?)',
                   (kind, value['id'], json.dumps(value, ensure_ascii=False)))

    @staticmethod
    def get(db, kind, key):
        row = db.execute('SELECT payload FROM items WHERE kind=? AND id=?', (kind, key)).fetchone()
        if not row:
            raise ValueError('record_not_found')
        return json.loads(row[0])

    def list(self, kind):
        with self.transaction() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM items WHERE kind=?', (kind,))]

    def read(self, kind, key):
        with self.transaction() as db:
            return self.get(db, kind, key)

    def change(self, kind, key, update):
        with self.transaction() as db:
            value = self.get(db, kind, key)
            update(value)
            value['updated'] = time.time()
            self.put(db, kind, value)
            return value

    def workspace(self, data, key=None):
        name = str(data.get('name', '')).strip()
        path = str(data.get('path', '')).strip()
        description = str(data.get('description', '')).strip()
        if not name or len(name) > 60 or len(path) > 500 or len(description) > 2000:
            raise ValueError('invalid_workspace')
        if path:
            directory = Path(path).expanduser()
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError('workspace_directory_missing')
            path = str(directory.resolve())
        with self.transaction() as db:
            if key:
                self.get(db, 'workspace', key)
            for row in db.execute("SELECT payload FROM items WHERE kind='workspace'"):
                existing = json.loads(row[0])
                if existing['name'] == name and existing['id'] != key:
                    raise ValueError('workspace_name_exists')
            value = dict(id=key or uuid.uuid4().hex, name=name, path=path, description=description)
            self.put(db, 'workspace', value)
            return value

    def conversation(self, workspace):
        with self.transaction() as db:
            self.get(db, 'workspace', workspace)
            value = dict(id=uuid.uuid4().hex, workspaceId=workspace, title='新会话', draft='',
                         turns=[], created=time.time(), updated=time.time())
            self.put(db, 'conversation', value)
            return value
