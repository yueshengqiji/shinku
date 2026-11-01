"""宿主持有的故障队列；SQLite 唯一键负责去重，不以模型答复宣布修复成功。"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time


@dataclass(frozen=True)
class FailureEvent:
    source: str
    event_id: str
    code: str
    revision: str
    observed_at: float
    maintenance: bool = False
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('source', 'event_id', 'code', 'revision'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError('invalid_failure_event')
        if (not isinstance(self.observed_at, (int, float)) or isinstance(self.observed_at, bool)
                or not 0 < self.observed_at < float('inf') or type(self.maintenance) is not bool
                or not isinstance(self.diagnostics, dict)
                or len(json.dumps(self.diagnostics, allow_nan=False).encode()) > 16384):
            raise ValueError('invalid_failure_event')


def classify_failure(event):
    """只分类宿主错误码；未知送达与外部故障不授权源码修复或业务重发。"""
    if event.maintenance or event.source == 'maintenance':
        return 'ignored'
    if event.diagnostics.get('delivery_status') in {'uncertain', 'unknown', 'partial', 'failed', 'error'}:
        return 'delivery_review'
    code = event.code.casefold()
    if any(x in code for x in ('timeout', 'deadline', 'quota', 'budget', 'permission',
                               'authorization', 'login', 'rate_limit', 'call_failed', 'source_changed')):
        return 'diagnosis_only'
    if code in {'conversation_output_invalid', 'conversation_output_empty', 'model_output_invalid',
                'model_output_empty', 'invalid_decision'} or re.fullmatch(
                    r'(bridge_error|planner_error|executor_error):(typeerror|attributeerror|valueerror|runtimeerror)', code):
        return 'repair'
    return 'diagnosis_only'


class RepairQueue:
    """每个故障/版本仅执行一次。进程崩溃留下 interrupted，不能自动重复副作用。"""
    def __init__(self, path, *, project, clock=time.time):
        self.path, self.project, self.clock = Path(path), str(project), clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.path.chmod(0o600)
        self.db.execute('''CREATE TABLE IF NOT EXISTS maintenance_jobs
            (id TEXT PRIMARY KEY, status TEXT NOT NULL, event TEXT NOT NULL,
             result TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL)''')
        self.db.commit()

    def enqueue(self, event):
        if not isinstance(event, FailureEvent):
            raise TypeError('host_failure_event_required')
        category = classify_failure(event)
        if category == 'ignored':
            return None
        # 同一类别错误在同一项目/运行版本只产生一个任务，不随消息 ID 反复收费。
        key = hashlib.sha256(json.dumps([self.project, event.code, event.revision],
                                      separators=(',', ':')).encode()).hexdigest()
        now = self.clock()
        with self.lock, self.db:
            self.db.execute('INSERT OR IGNORE INTO maintenance_jobs VALUES (?,?,?,?,?,?)',
                (key, 'queued' if category == 'repair' else category,
                 json.dumps(asdict(event), ensure_ascii=False), '{}', now, now))
        return key

    def interrupted(self):
        # 必须在宿主取得单实例锁后调用；第二个活实例不能把当前任务标为中断。
        with self.lock, self.db:
            self.db.execute("UPDATE maintenance_jobs SET status='interrupted',updated=? WHERE status='running'", (self.clock(),))

    def claim(self, *, max_per_day=2):
        with self.lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            count = self.db.execute("SELECT count(*) FROM maintenance_jobs WHERE status IN "
                "('running','candidate_ready','failed','stale','revoked','interrupted') AND updated>=?",
                (self.clock() - 86400,)).fetchone()[0]
            if count >= max_per_day:
                self.db.execute("UPDATE maintenance_jobs SET status='budget_blocked',result=?,updated=? WHERE status='queued'",
                    (json.dumps({'reason': 'daily_maintenance_limit'}), self.clock()))
                return None
            row = self.db.execute("SELECT id,event FROM maintenance_jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE maintenance_jobs SET status='running',updated=? WHERE id=?", (self.clock(), row[0]))
            return row[0], FailureEvent(**json.loads(row[1]))

    def finish(self, key, result, *, status):
        if status not in {'candidate_ready', 'failed', 'interrupted', 'stale', 'revoked'}:
            raise ValueError('invalid_maintenance_status')
        with self.lock, self.db:
            changed = self.db.execute("UPDATE maintenance_jobs SET status=?,result=?,updated=? WHERE id=? AND status='running'",
                (status, json.dumps(result, ensure_ascii=False), self.clock(), key)).rowcount
            if changed != 1:
                raise ValueError('maintenance_job_not_owned')

    def snapshot(self, *, limit=30):
        with self.lock:
            rows = self.db.execute('SELECT id,status,result,created,updated FROM maintenance_jobs ORDER BY created DESC LIMIT ?',
                                   (max(1, min(100, limit)),)).fetchall()
        return [dict(id=k, status=s, result=json.loads(r), created=c, updated=u) for k,s,r,c,u in rows]

    def close(self):
        with self.lock:
            self.db.close()
