"""Angelos 专属权限与一次性审批；不读写 Bot 的命令状态。

宿主显式安装后，原授权先校验身份/预算，再由本模块收紧候选工具。
审批绑定执行中的任务、尝试、权限版本与完整参数；浏览器只能批准已有请求，
不能提交工具调用。等待有界，取消/撤权/重启不会重放或自动批准。
"""
from dataclasses import replace
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.tools.execution import ExecutionPolicy

LEVELS = (
    {'id': 'zayin', 'label': 'ZAYIN', 'description': '仅对话 · 不调用工具'},
    {'id': 'he', 'label': 'HE', 'description': '只读'},
    {'id': 'waw', 'label': 'WAW', 'description': '执行前批准'},
    {'id': 'aleph', 'label': 'ALEPH', 'description': '工具全开 · 无需确认 · 无 root'},
)
INTERNAL = {'declare_task_contract', 'read_self_state', 'update_work_plan',
            'read_work_context', 'task_material_read', 'request_context_compaction'}


def require_unprivileged():
    """正式 Linux 宿主启动前设置继承到子进程的 no_new_privs，拒绝 root 启动。"""
    if not hasattr(os, 'geteuid') or os.geteuid() == 0:
        raise RuntimeError('angelos_unprivileged_host_required')
    if sys.platform == 'linux':
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
            raise RuntimeError('angelos_no_new_privileges_required')
    else:
        raise RuntimeError('angelos_privilege_guard_unsupported')


def read_only(handler):
    return (getattr(handler, 'agent_permission_class', '') == 'read' and not
        set(getattr(handler, 'agent_completion_effects', ()) or ()).intersection(
            {'write', 'delete', 'send', 'process_stop', 'command_run', 'memory_manage'}))


class PermissionControls:
    """唯一权限状态写入者；持久化失败不改变内存中的已生效档位。"""
    def __init__(self, path, *, default='he', approval_timeout=300):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.pending = {}
        self.audit = []
        self.timeout = approval_timeout
        self.level, self.revision = default, uuid.uuid4().hex
        self.store = None
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            self.level = data.get('level')
        if self.level not in {v['id'] for v in LEVELS}:
            raise ValueError('invalid_permission_level')

    def snapshot(self):
        with self.lock:
            return {'available': True, 'level': self.level, 'revision': self.revision,
                    'levels': list(LEVELS), 'root': False,
                    'pending': [dict(v['public']) for v in self.pending.values()]}

    def select(self, level, revision):
        if level not in {v['id'] for v in LEVELS}:
            raise ValueError('invalid_permission_level')
        with self.lock:
            if revision != self.revision:
                raise ValueError('permission_version_conflict')
            if level == self.level:
                return self.snapshot()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex)
            try:
                with temporary.open('x', encoding='utf-8') as f:
                    os.chmod(temporary, 0o600)
                    json.dump({'level': level}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self.level, self.revision = level, uuid.uuid4().hex
            for row in self.pending.values():
                row['decision'] = 'revoked'
                row['event'].set()
            return self.snapshot()

    def decide(self, identifier, revision, decision):
        with self.lock:
            row = self.pending.get(identifier)
            if (row is None or row['public']['revision'] != revision or revision != self.revision
                    or row['decision'] is not None):
                raise ValueError('approval_expired')
            if decision not in {'approve', 'deny'}:
                raise ValueError('invalid_approval_decision')
            row['decision'] = decision
            row['event'].set()
            return self.snapshot()

    def install(self, service, requester):
        """在专用 ServiceHost 完成记忆/诊断工具注册之后安装；共享 Bot 不调用此入口。"""
        original = service.authorize
        registry = service.executor.tool_host.registry
        controls = self
        self.registry = registry
        for name, handler in list(registry.view().items()):
            registry.register(ControlledTool(handler, self), name=name, replace=True)

        def authorize(scope, task, operation):
            grant = original(scope, task, operation)
            if grant is None or scope.requester != requester or not scope.session.startswith('angelos:'):
                return grant
            return controls.restrict(grant)
        service.authorize = authorize

    def restrict(self, grant, *, respect_host_approval=False):
        """与来源宿主的工具集合取交集；跨进程宿主审批也由真实调用参数绑定。"""
        with self.lock:
            level, revision = self.level, self.revision
        registry = self.registry
        names = {n for n in grant.allowed_tools if grant.policy.accepts(n)}
        host_approvals = {n for n in names if grant.policy.requires_approval(n)} if respect_host_approval else set()
        if level == 'zayin':
            names.clear()
        elif level == 'he':
            names = {n for n in names if read_only(registry.get(n))}
        blocked = set(grant.policy.blocked) | (set(registry.names()) - names)
        if level == 'zayin':
            blocked |= INTERNAL
        allowed = names | ({n for n in INTERNAL if grant.policy.accepts(n)} if level != 'zayin' else set())
        policy = ExecutionPolicy(allowed=frozenset(allowed), blocked=frozenset(blocked),
            risk_by_tool=grant.policy.risk_by_tool, approval_required_by_risk=frozenset())
        return replace(grant, allowed_tools=frozenset(names), policy=policy,
            # 跨宿主不能把来源full展示成ultra；WAW仍在真实调用处逐次审批。
            permission_mode=('readonly' if level in {'he', 'zayin'} else
                             grant.permission_mode if respect_host_approval else 'ultra'),
            authorization_ref=grant.authorization_ref + ':angelos:' + revision,
            still_authorized=lambda: grant.still_authorized() and self.revision == revision,
            tool_context={**grant.tool_context, 'angelos_permission_revision': revision,
                          'angelos_permission_level': level,
                          'delegation_approval_tools': sorted(host_approvals)})

    def check(self, handler, call, context):
        revision = context.get('angelos_permission_revision')
        with self.lock:
            if revision != self.revision:
                raise AgentBoundaryError('agent_permission_changed')
            level = self.level
        read = read_only(handler)
        if level == 'zayin' or level == 'he' and not read:
            raise AgentBoundaryError('host_authorization_denied')
        host_approval = handler.tool_type in context.get('delegation_approval_tools', ())
        if not host_approval and (level != 'waw' or read):
            return
        task = context.get('agent_task_state', {})
        if not task.get('task_id') or (not str(task.get('session', '')).startswith('angelos:')
                and not context.get('agent_delegation_host')):
            raise AgentBoundaryError('approval_task_binding_required')
        identifier = uuid.uuid4().hex
        serialized = json.dumps(call, ensure_ascii=False, sort_keys=True)
        public = {'id': identifier, 'revision': revision, 'tool': handler.tool_type,
            'taskId': task['task_id'], 'conversationId': task['session'][8:] if task['session'].startswith('angelos:') else '',
            'originHost': context.get('agent_delegation_host', ''),
            'attempt': task.get('execution_attempt'), 'arguments': call,
            'digest': hashlib.sha256(serialized.encode()).hexdigest(), 'created': time.time()}
        rejected = context.setdefault('_angelos_denied_calls', set())
        if public['digest'] in rejected:
            raise AgentBoundaryError('approval_denied')
        row = {'public': public, 'event': threading.Event(), 'decision': None}
        with self.lock:
            if revision != self.revision:
                raise AgentBoundaryError('agent_permission_changed')
            self.pending[identifier] = row
        def interrupted():
            # 等待人的时间也在原任务预算内；迟到批准不能复活超时/取消的任务。
            from shinku.agent.budget import current_task_budget
            budget = current_task_budget()
            interrupt = context.get('agent_task_interrupt')
            return (revision != self.revision or bool(budget and budget.stop_reason())
                    or bool(callable(interrupt) and interrupt()))
        try:
            deadline = time.monotonic() + self.timeout
            while not row['event'].wait(.2):
                if interrupted():
                    raise AgentBoundaryError('approval_canceled')
                if time.monotonic() >= deadline:
                    raise AgentBoundaryError('approval_timed_out')
            if interrupted():
                raise AgentBoundaryError('approval_canceled')
            if row['decision'] != 'approve':
                rejected.add(public['digest'])
                raise AgentBoundaryError('approval_denied')
        finally:
            with self.lock:
                self.pending.pop(identifier, None)
                record = {**public, 'decision': row['decision'] or 'expired', 'finished': time.time()}
                self.audit.append(record)
                self.audit = self.audit[-100:]
                if self.store is not None:
                    with self.store.transaction() as db:
                        self.store.put(db, 'tool_approval', record)


class ControlledTool:
    """保留原工具的验证器/观察凭据，仅在实际副作用入口增加审批。"""
    def __init__(self, handler, controls):
        self.handler, self.controls = handler, controls

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def execute(self, *, call, context):
        try:
            self.controls.check(self.handler, call, context)
        except AgentBoundaryError as exc:
            return {"status": "blocked", "code": exc.code, "followup_context": "操作未执行：" + exc.code,
                    "state_updates": {"executed": False, "information_gap": exc.code}}
        return self.handler.execute(call=call, context=context)
