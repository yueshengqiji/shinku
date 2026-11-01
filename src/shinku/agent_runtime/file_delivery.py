"""执行侧文件交付请求：原任务库持有快照/状态，可信聊天宿主持有真正发送权。

阅读顺序：FileDeliveryTool -> request（持久化并等待）-> pending/payload（宿主取件）
-> acknowledge（宿主回执）-> 工具返回send证据。模型永远拿不到文件正文、凭据或收件人。
本模块只由启动器显式安装；读取路径由原文件权限解析器决定，不能由模型放宽。
"""
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from types import SimpleNamespace

from .contracts import AgentBoundaryError, AgentScope, canonical_hash

TOOL = 'private_qq_send_file'
MAX_FILE_BYTES = 16 * 1024 * 1024


def requested_file_send(requests, path):
    """发送的必要前提来自原始用户要求；契约/模型建议不能自行增加外发。

    复用现有否定与讨论投影。此检查只收紧新发送工具，不选择路由或授予权限；
    后续仍需路径权限、完整契约和真实回执。续接中最新明确撤销优先。
    """
    from shinku.agent.request_intent import assertion_text, execution_request_text
    from shinku.agent.verification import SmartCompletionVerifier
    from shinku.agent.verification_targets import bind_literal_paths, file_delivery_requested
    texts = [str(r.get('text', '')) for r in requests]
    if texts and re.search(r'(?:不要|不用|禁止|不得|别再)\s*(?:发送(?:文件|附件)?|发给我|传给我)', assertion_text(texts[-1])):
        return False
    for text in texts:
        source = assertion_text(execution_request_text(text)).replace(path, ' ')
        if re.search(r'如果|假设|假如|举例|例如|我会|我自己|我打算', source):
            continue
        projected = bind_literal_paths(execution_request_text(text), [path])
        roles = SmartCompletionVerifier._explicit_target_roles([projected])
        send_targets = [target for target, values in roles.items() if 'send' in values]
        if send_targets:
            # A的发送指令不能借给B；明确绝对路径采用原有严格匹配，
            # 只有用户只给文件名时，才允许匹配读取权限内的同名绝对路径。
            if any(SmartCompletionVerifier._targets_match(target, path) for target in send_targets):
                return True
            continue
        if file_delivery_requested([text], path):
            # 跨分句的“把文件发到当前会话”仍绑定原文提及的文件。
            # 未写路径的“把报告发给我”留给当前任务的产物/读取权限核验。
            targets = [target for target in SmartCompletionVerifier._extract_targets(projected) if '://' not in target]
            if targets and not any(SmartCompletionVerifier._targets_match(target, path) for target in targets):
                continue
            return True
    return False


def read_snapshot(path):
    """逐层no-follow打开绝对路径，只读取普通文件，避免校验后路径替换和特殊文件阻塞。"""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise AgentBoundaryError('file_delivery_absolute_path_required')
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        handle = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(handle, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
                raise AgentBoundaryError('file_delivery_type_or_size_invalid')
            data = stream.read(MAX_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (len(data) > MAX_FILE_BYTES or len(data) != before.st_size
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                   (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise AgentBoundaryError('file_delivery_source_changed')
        return data
    finally:
        os.close(fd)


class TaskFileDelivery:
    """以scope/task/attempt/path防重；重启时未完成项转unknown，不恢复外部发送。"""
    def __init__(self, service, resolver, *, wait_seconds=120):
        if not callable(resolver) or not 0 < wait_seconds <= 120:
            raise ValueError('file_delivery_host_resolver_required')
        self.service, self.resolver, self.wait_seconds = service, resolver, wait_seconds
        self.changed = threading.Condition()
        with service.task_store._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS agent_file_deliveries
                (id TEXT PRIMARY KEY, scope TEXT NOT NULL, task TEXT NOT NULL, attempt INTEGER NOT NULL,
                 authority TEXT NOT NULL, metadata TEXT NOT NULL, body BLOB,
                 status TEXT NOT NULL, receipt TEXT NOT NULL)''')
            db.execute("UPDATE agent_file_deliveries SET status='uncertain' WHERE status IN ('pending','claimed')")
        registry = service.executor.tool_host.registry
        if registry.get(TOOL) is not None:
            raise ValueError('file_delivery_tool_conflict')
        registry.register(FileDeliveryTool(self))

    def current(self, scope, task_id, attempt=None, authority=None):
        task = self.service._get(scope, task_id)
        grant = self.service._grant(scope, task, 'inspect')
        if (task.get('cancel_requested') or task.get('status') == 'canceled'
                or attempt is not None and task.get('execution_attempt') != attempt
                or authority is not None and grant.authorization_ref != authority
                or TOOL not in grant.allowed_tools or not grant.policy.accepts(TOOL)
                or not grant.still_authorized()):
            raise AgentBoundaryError('file_delivery_authorization_changed')
        return task, grant

    def row(self, scope, task_id, identifier):
        with self.service.task_store._transaction() as db:
            raw = db.execute('SELECT * FROM agent_file_deliveries WHERE id=? AND scope=? AND task=?',
                (identifier, canonical_hash(asdict(scope)), task_id)).fetchone()
        if raw is None:
            raise AgentBoundaryError('file_delivery_not_found')
        # 使用显式列顺序，兼容任务库连接的Row/tuple设置。
        keys = ('id', 'scope', 'task', 'attempt', 'authority', 'metadata', 'body', 'status', 'receipt')
        row = dict(zip(keys, raw))
        task, _ = self.current(scope, task_id, row['attempt'], row['authority'])
        if json.loads(row['metadata'])['request_hash'] != canonical_hash(task.get('requests', [])):
            raise AgentBoundaryError('file_delivery_request_changed')
        return row

    def request(self, call, context):
        bound = context.get('agent_request_scope')
        state = context.get('agent_task_state', {})
        if not context.get('agent_delegation_host') or bound is None:
            raise AgentBoundaryError('file_delivery_qq_host_required')
        scope = AgentScope(bound.requester, bound.session)
        if not scope.session.startswith(('private:', 'group:')):
            raise AgentBoundaryError('file_delivery_qq_scope_required')
        task_id, attempt = state.get('task_id'), state.get('execution_attempt')
        authority = context.get('agent_authority_revision')
        task, _ = self.current(scope, task_id, attempt, authority)
        if not requested_file_send(task.get('requests', []), call['path']):
            raise AgentBoundaryError('file_delivery_not_requested')
        source = Path(call['path'])
        if not source.is_absolute() or '..' in source.parts:
            raise AgentBoundaryError('file_delivery_absolute_path_required')
        path = Path(self.resolver(str(source), context))
        data = read_snapshot(path)
        digest = hashlib.sha256(data).hexdigest()
        identifier = 'file:' + canonical_hash([asdict(scope), task_id, attempt, str(path)])
        metadata = {'id': identifier, 'task_id': task_id, 'attempt': attempt,
                    'request_hash': canonical_hash(task.get('requests', [])),
                    'path': str(path), 'name': path.name, 'sha256': digest, 'bytes': len(data)}
        with self.service.task_store._transaction() as db:
            if db.execute("SELECT 1 FROM agent_file_deliveries WHERE task=? AND status='uncertain'",
                          (task_id,)).fetchone():
                raise AgentBoundaryError('file_delivery_previous_effect_unknown')
            # 显式续跑也不能把已经确认的同一文件再发一遍。原回执是事实，
            # 当前读取/发送权限仍须在上方重新取得；新请求应创建新任务。
            for prior in db.execute("SELECT metadata,receipt FROM agent_file_deliveries WHERE task=? AND scope=? AND status='sent'",
                                    (task_id, canonical_hash(asdict(scope)))).fetchall():
                previous = json.loads(prior[0])
                if (previous['path'], previous['sha256']) == (str(path), digest):
                    return self.result(previous, json.loads(prior[1]))
            old = db.execute('SELECT metadata FROM agent_file_deliveries WHERE id=?', (identifier,)).fetchone()
            if old and json.loads(old[0]) != metadata:
                raise AgentBoundaryError('file_delivery_input_changed')
            if not old:
                total = db.execute('SELECT COALESCE(sum(length(body)),0) FROM agent_file_deliveries').fetchone()[0]
                count = db.execute('SELECT count(*) FROM agent_file_deliveries WHERE task=? AND attempt=?',
                                   (task_id, attempt)).fetchone()[0]
                if total + len(data) > 64 * 1024 * 1024 or count >= 16:
                    raise AgentBoundaryError('file_delivery_capacity_reached')
                db.execute('INSERT INTO agent_file_deliveries VALUES (?,?,?,?,?,?,?,?,?)',
                    (identifier, canonical_hash(asdict(scope)), task_id, attempt, authority,
                     json.dumps(metadata, ensure_ascii=False), data, 'pending', '{}'))
        deadline = time.monotonic() + self.wait_seconds
        while True:
            row = self.row(scope, task_id, identifier)
            if row['status'] == 'sent':
                receipt = json.loads(row['receipt'])
                return self.result(metadata, receipt)
            if row['status'] not in {'pending', 'claimed'}:
                raise AgentBoundaryError('file_delivery_' + row['status'])
            interrupt = context.get('agent_task_interrupt')
            if callable(interrupt) and interrupt() or time.monotonic() >= deadline:
                with self.service.task_store._transaction() as db:
                    db.execute("UPDATE agent_file_deliveries SET status='uncertain' WHERE id=? AND status IN ('pending','claimed')", (identifier,))
                raise AgentBoundaryError('file_delivery_unconfirmed')
            with self.changed:
                self.changed.wait(.05)

    @staticmethod
    def result(metadata, receipt):
        return {'tool_type': TOOL, 'followup_context': '文件已由原QQ会话的宿主发送，并取得消息回执；相同请求不会重复发送。',
            'state_updates': {'verified_effects': ['send'], 'completion_targets': [metadata['path']],
                'delivery_receipt': receipt['message_ids'][0], 'delivery_id': metadata['id'],
                'sha256': metadata['sha256'], 'bytes': metadata['bytes'], 'path': metadata['path']}}

    def pending(self, scope, task_id):
        task, _ = self.current(scope, task_id)
        with self.service.task_store._transaction() as db:
            rows = db.execute("SELECT id FROM agent_file_deliveries WHERE scope=? AND task=? AND attempt=? AND status IN ('pending','claimed') LIMIT 16",
                (canonical_hash(asdict(scope)), task_id, task['execution_attempt'])).fetchall()
        return [json.loads(self.row(scope, task_id, r[0])['metadata']) for r in rows]

    def payload(self, scope, task_id, identifier):
        row = self.row(scope, task_id, identifier)
        if row['status'] not in {'pending', 'claimed'}:
            raise AgentBoundaryError('file_delivery_not_pending')
        metadata = json.loads(row['metadata'])
        data = bytes(row['body'])
        if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
            raise AgentBoundaryError('file_delivery_snapshot_changed')
        # 出队前再核对来源；发送器随后使用这份字节快照，不能转交可变本机路径。
        if hashlib.sha256(read_snapshot(metadata['path'])).hexdigest() != metadata['sha256']:
            raise AgentBoundaryError('file_delivery_source_changed')
        with self.service.task_store._transaction() as db:
            db.execute("UPDATE agent_file_deliveries SET status='claimed' WHERE id=? AND status='pending'", (identifier,))
        return {**metadata, 'data': base64.b64encode(data).decode('ascii')}

    def acknowledge(self, scope, task_id, identifier, receipt):
        row = self.row(scope, task_id, identifier)
        metadata = json.loads(row['metadata'])
        if (not isinstance(receipt, dict) or set(receipt) != {'status', 'message_ids', 'sha256'}
                or receipt['status'] not in {'sent', 'failed', 'uncertain'}
                or receipt['sha256'] != metadata['sha256']
                or not isinstance(receipt['message_ids'], list) or len(receipt['message_ids']) > 16
                or any(not isinstance(v, str) or not 1 <= len(v) <= 256 for v in receipt['message_ids'])
                or receipt['status'] == 'sent' and not receipt['message_ids']):
            raise AgentBoundaryError('file_delivery_receipt_invalid')
        with self.service.task_store._transaction() as db:
            current = db.execute('SELECT status,receipt FROM agent_file_deliveries WHERE id=?', (identifier,)).fetchone()
            if receipt['status'] == 'sent' and current[0] == 'pending':
                raise AgentBoundaryError('file_delivery_not_claimed')
            if current[0] not in {'pending', 'claimed'}:
                if current[0] == receipt['status'] and json.loads(current[1]) == receipt:
                    return {'recorded': True}
                raise AgentBoundaryError('file_delivery_receipt_conflict')
            # 未确认/失败快照作为审查证据保留；总容量门禁不会淘汰这些记录来重发。
            db.execute('UPDATE agent_file_deliveries SET status=?,receipt=?,body=CASE WHEN ?=\'sent\' THEN NULL ELSE body END WHERE id=?',
                (receipt['status'], json.dumps(receipt), receipt['status'], identifier))
        with self.changed:
            self.changed.notify_all()
        return {'recorded': True}


class FileDeliveryTool:
    """只声明路径；QQ目标、实际网络调用及消息回执均由原聊天宿主拥有。"""
    tool_type = TOOL
    agent_permission_class = 'change'
    agent_progress_class = 'change'
    agent_completion_effects = ('send',)
    agent_completion_target_arguments = ('path',)
    agent_auto_confirm_in_full_access = True

    def __init__(self, broker):
        self.broker = broker

    def tool_metadata(self):
        return SimpleNamespace(description='将已有文件或刚生成的文件作为普通文件附件发回原QQ会话。'
            '仅限当前宿主读取权限内的绝对路径、16MiB以内；工具等待实际消息回执。'
            '不能选择收件人，不能以创建文件代替发送；未知发送不重试。', input_schema={
                'type': 'object', 'properties': {'path': {'type': 'string'}, 'confirm': {'type': 'boolean'}},
                'required': ['path', 'confirm'], 'additionalProperties': False})

    def normalize_call(self, call):
        # 通用工具协议在handler调用前补type；它只能与已解析工具名一致。
        if (not isinstance(call, dict) or set(call) - {'path', 'confirm', 'type'}
                or not {'path', 'confirm'} <= set(call) or call.get('type', TOOL) != TOOL or call['confirm'] is not True
                or not isinstance(call['path'], str) or not 1 <= len(call['path']) <= 4096):
            return None
        return {'path': call['path'], 'confirm': True}

    def execute(self, *, call, context):
        args = self.normalize_call(call)
        if args is None:
            raise AgentBoundaryError('file_delivery_arguments_invalid')
        try:
            return self.broker.request(args, context)
        except (AgentBoundaryError, OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, AgentBoundaryError) else 'file_delivery_source_unavailable'
            return {'tool_type': TOOL, 'followup_context': '文件未取得已发送回执：' + code,
                    'state_updates': {'private_tool_error': True, 'code': code}}
