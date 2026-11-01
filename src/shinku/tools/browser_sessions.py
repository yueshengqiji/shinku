"""宿主浏览器资源与防重日志；不读取用户 Cookie，不自动恢复旧网页。

线程调度和 Chromium 创建由宿主注入。每个任务拥有独立 context，终态释放；
跨进程仅保留动作摘要。旧进程曾派发动作时先要求独立核对，不按模型文字重放。
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import uuid

from .workspace import scope_key


def task_key(context):
    scope = scope_key(context)
    return (*scope[:2], str(context.get('agent_work_id') or scope[2]))


class BrowserJournal:
    """只记录作用域摘要、参数摘要和派发状态；写入失败必须先于浏览器动作。"""
    def __init__(self, path):
        self.path = Path(path)
        self.generation = uuid.uuid4().hex

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(self.path)
        self.path.chmod(0o600)
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS browser_actions '
                       '(scope TEXT, request TEXT, arguments TEXT, generation TEXT, status TEXT, '
                       'PRIMARY KEY(scope, request))')
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def scope(context):
        return hashlib.sha256(json.dumps(task_key(context)).encode()).hexdigest()

    def begin(self, context, request_key, arguments):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            scope = self.scope(context)
            if db.execute('SELECT 1 FROM browser_actions WHERE scope=? AND '
                          '(generation!=? OR status=?) LIMIT 1', (scope, self.generation, 'uncertain')).fetchone():
                raise ValueError('browser_prior_effect_requires_host_resolution')
            request = hashlib.sha256(request_key.encode()).hexdigest()
            if db.execute('SELECT 1 FROM browser_actions WHERE scope=? AND request=?', (scope, request)).fetchone():
                raise ValueError('browser_request_already_dispatched')
            db.execute('INSERT INTO browser_actions VALUES (?,?,?,?,?)',
                       (scope, request, hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest(),
                        self.generation, 'uncertain'))

    def observed(self, context, request_key):
        with self._db() as db:
            db.execute('UPDATE browser_actions SET status=? WHERE scope=? AND request=? AND generation=?',
                       ('observed', self.scope(context), hashlib.sha256(request_key.encode()).hexdigest(), self.generation))


class BrowserSessions:
    """任务 -> 独立 context。调用者必须通过同一个 run 所属线程访问。"""
    def __init__(self, create_context, *, max_sessions=16, max_pages=16):
        self.create_context = create_context
        self.max_sessions, self.max_pages = max_sessions, max_pages
        self.sessions = {}

    def _session(self, context):
        key = task_key(context)
        if key not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                raise ValueError('browser_session_limit')
            browser, isolated = self.create_context(context)
            self.sessions[key] = (browser, isolated)
        return self.sessions[key][1]

    def page(self, context):
        isolated = self._session(context)
        pages = [p for p in isolated.pages if not p.is_closed()]
        return pages[0] if pages else isolated.new_page()

    def pages(self, context):
        return [p for p in self._session(context).pages if not p.is_closed()]

    def new_page(self, context):
        if len(self.pages(context)) >= self.max_pages:
            raise ValueError('browser_page_limit')
        return self._session(context).new_page()

    def release(self, context):
        key = task_key(context)
        pair = self.sessions.get(key)
        if pair is None:
            return
        browser, isolated = pair
        # 关闭失败不丢失所有权；宿主可再次尝试回收，不能伪报已经停止。
        isolated.close()
        browser.close()
        self.sessions.pop(key, None)
