"""可信故障采集与维护调度。正常任务循环不反向依赖本模块。"""
from dataclasses import asdict, replace
import json
import os
import threading
import time

from .queue import FailureEvent

def task_failures(rows, *, revision, not_before):
    """输入必须由可信宿主按允许的属主/作用域读取；不调用 get 触发旧库清理。"""
    for task in rows:
        if (task.get('requester') == 'host-maintenance' or str(task.get('session', '')).startswith('maintenance:')
                or task.get('status') != 'failed' or task.get('updated', 0) < not_before):
            continue
        yield FailureEvent('angelos', task['task_id'], task.get('stop_reason') or 'execution_failed',
            revision, task['updated'], diagnostics=dict(task_id=task['task_id'],
                protocol_failure=task.get('progress', {}).get('conversation_protocol_failure', ''),
                tools=[dict(name=call.get('name'), status=call.get('status')) for call in task.get('tool_calls', [])][-64:],
                budget_reason=task.get('budget', {}).get('reason', '')))


def trace_failures(rows, *, revision, not_before):
    """只读已终结的宿主 trace；不把用户原文、工具参数或模型猜测放入授权。"""
    for row in rows:
        finished = row.get('finished_at')
        if (not isinstance(finished, (int, float)) or finished < not_before
                or row.get('status') not in {'failed', 'finished', 'completed'}):
            continue
        delivery = row.get('delivery_status', '')
        if row.get('status') != 'failed' and delivery not in {'failed', 'error', 'partial', 'uncertain', 'unknown'}:
            continue
        yield FailureEvent('qq', row['trace_id'], str(row.get('reason') or 'delivery_failed'), revision, finished,
            diagnostics=dict(delivery_status=delivery,
                task_ids=[stage['task_id'] for stage in row.get('stages', [])
                          if stage.get('name') == 'agent_delegation' and stage.get('task_id')],
                error_types=[stage.get('error_type', '') for stage in row.get('stages', [])
                             if stage.get('error_type')],
                failed_phase=next((stage.get('phase', '') for stage in row.get('stages', [])
                                  if stage.get('status') == 'failed'), '')))


class MaintenanceHost:
    """一个可选宿主实例拥有锁、队列消费和线程；修复失败只落状态，不再入队。"""
    def __init__(self, queue, policy, runner, *, collect, evidence_reader=None, interval=5, clock=time.time):
        if (not callable(collect) or not callable(runner) or interval <= 0
                or evidence_reader is not None and not callable(evidence_reader)):
            raise ValueError('invalid_maintenance_host')
        self.queue, self.policy, self.runner, self.collect = queue, policy, runner, collect
        self.interval, self.clock = interval, clock
        self.evidence_reader = evidence_reader
        self.stopping = threading.Event()
        self.thread = None
        self.lockfile = None
        self.error = ''
        # 首次装配不追溯历史付费故障；重启沿用同一监听起点和去重账本。
        with queue.lock, queue.db:
            queue.db.execute('CREATE TABLE IF NOT EXISTS maintenance_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            queue.db.execute('INSERT OR IGNORE INTO maintenance_meta VALUES (?,?)', ('not_before', str(clock())))
            self.not_before = float(queue.db.execute('SELECT value FROM maintenance_meta WHERE key=?', ('not_before',)).fetchone()[0])

    def _acquire(self):
        import fcntl
        if self.lockfile is not None:
            return
        self.policy.check()
        path = self.policy.state_root / 'maintenance.lock'
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stream = path.open('a')
        path.chmod(0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            stream.close()
            raise
        self.lockfile = stream
        self.queue.interrupted()

    def tick(self):
        self._acquire()
        if self.stopping.is_set() or not self.policy.authorized():
            return
        try:
            version = self.policy.version()
            events = self.collect(version, self.not_before)
            for index, event in enumerate(events):
                if index >= 100:
                    break
                if not isinstance(event, FailureEvent) or event.revision != version:
                    raise ValueError('maintenance_source_version_changed')
                self.queue.enqueue(event)
        except Exception as error:
            # 采集器异常不是业务程序缺陷，不递归创建维护任务；不记录异常正文。
            self.error = 'collection:' + type(error).__name__
        row = self.queue.claim(max_per_day=self.policy.max_jobs_per_day)
        if row is None:
            return
        key, event = row
        try:
            self.policy.check()
            if event.revision != self.policy.version() or self.clock() - event.observed_at > self.policy.event_ttl_seconds:
                self.queue.finish(key, dict(reason='source_version_changed'), status='stale')
                return
            if self.evidence_reader is not None:
                # 补充器只返回资料；不能替换故障身份、运行版本或维护授权。
                details = self.evidence_reader(event)
                if not isinstance(details, dict):
                    raise ValueError('maintenance_evidence_invalid')
                event = replace(event, diagnostics={**event.diagnostics, **details})
            # 每个故障保留独立资料；资料不能成为 runner 的路径/模型/授权配置。
            evidence = self.policy.state_root / 'failures'
            evidence.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(evidence / (key + '.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(asdict(event), stream, ensure_ascii=False)
            result = self.runner(key, event, self.stopping)
            self.policy.check()
            if not isinstance(result, dict) or result.get('deployed') is not False or not result.get('commit'):
                raise ValueError('maintenance_candidate_receipt_required')
            self.queue.finish(key, result, status='candidate_ready')
        except PermissionError:
            self.queue.finish(key, dict(reason='authorization_revoked'),
                              status='interrupted' if self.stopping.is_set() else 'revoked')
        except Exception as error:
            code = str(error)
            if not code.startswith('maintenance_') or not code.replace('_', '').isalnum() or len(code) > 120:
                code = type(error).__name__
            self.queue.finish(key, dict(reason=code), status='interrupted' if self.stopping.is_set() else 'failed')

    def _loop(self):
        try:
            self._acquire()
            while not self.stopping.is_set():
                if not self.policy.authorized():
                    self.error = 'authorization_revoked'
                    break
                self.tick()
                self.stopping.wait(self.interval)
        except Exception as error:
            self.error = type(error).__name__
        finally:
            if self.lockfile is not None:
                import fcntl
                fcntl.flock(self.lockfile, fcntl.LOCK_UN)
                self.lockfile.close()
                self.lockfile = None

    def start(self):
        if self.thread is not None or self.stopping.is_set():
            return
        try:
            self.policy.check()
        except PermissionError:
            # 可选维护未获授权不能阻止正常 Angelos 工作台启动。
            self.error = 'authorization_revoked'
            return
        self.thread = threading.Thread(target=self._loop, name='shinku-maintenance', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        elif self.lockfile is not None:
            import fcntl
            fcntl.flock(self.lockfile, fcntl.LOCK_UN)
            self.lockfile.close()
            self.lockfile = None

    def snapshot(self):
        return dict(configured=True, running=bool(self.thread and self.thread.is_alive()),
                    stopping=self.stopping.is_set(), error=self.error, jobs=self.queue.snapshot(),
                    limits=dict(jobs_per_day=self.policy.max_jobs_per_day,
                                requests_per_job=self.policy.max_requests, tokens_per_job=self.policy.max_tokens,
                                seconds_per_job=self.policy.max_seconds))
