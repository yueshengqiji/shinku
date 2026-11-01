"""受限服务日志读取；执行和答案审查共用持久化的同一份查询快照。

宿主固定可读 user unit，模型只选择时间窗和条数，不接收命令、路径或 SQL。
快照归属于任务/请求/执行尝试及授权版本；复核只读原快照，不重新执行
journalctl，也不把后来追加的日志替换进旧证据。正文是日志观察，不是执行成功证明。
"""
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import selectors
import sqlite3
import subprocess
import time
from types import SimpleNamespace
import uuid

from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.information import safe_text
from shinku.tools.workspace import tool_result
from .task_records import TaskRecordsTool


def read_journal(query):
    """无 shell 的有界读取；超时/过大输出终止子进程，不能升级为完整读取。"""
    command = ['journalctl', '--user', '--no-pager', '--quiet', '--output=json',
               '--output-fields=__REALTIME_TIMESTAMP,__CURSOR,PRIORITY,MESSAGE,_PID,_SYSTEMD_USER_UNIT,_SYSTEMD_INVOCATION_ID',
               '--unit=' + query['unit'], '--since=@' + str(query['since']),
               '--until=@' + str(query['until']), '--reverse', '--lines=' + str(query['limit'] + 1)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        raw = bytearray()
        deadline = time.monotonic() + 10
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise ValueError('service_log_read_timeout')
                    chunk = process.stdout.read1(8192)
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if len(raw) > 256000:
                        raise ValueError('service_log_output_too_large')
                if process.wait(timeout=max(.01, deadline-time.monotonic())) != 0:
                    raise ValueError('service_log_read_failed')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    entries = []
    for row in rows[:query['limit']]:
        if not isinstance(row, dict) or not isinstance(row.get('MESSAGE'), str):
            raise ValueError('service_log_record_unreadable')
        entries.append({key: row.get(key) for key in ('__REALTIME_TIMESTAMP', '__CURSOR', 'PRIORITY', 'MESSAGE', '_PID', '_SYSTEMD_USER_UNIT', '_SYSTEMD_INVOCATION_ID')})
    return {'entries': list(reversed(entries)), 'more_in_window': len(rows) > query['limit']}


class ServiceLogsTool(TaskRecordsTool):
    """授权检查沿用工作台任务绑定；快照库只有此工具写入。"""
    tool_type = 'read_service_logs'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_local_answer_source = True
    agent_local_answer_reference_prefix = 'journal:'
    agent_completion_effects = ('read',)
    agent_completion_target_arguments = ()
    agent_observation_receipt_fields = ('path', 'sha256', 'query', 'read_complete')

    def __init__(self, service, requester, path, units, *, reader=read_journal, instances=None):
        super().__init__(service, requester)
        self.path, self.units, self.reader = Path(path), tuple(units), reader
        self.instances = instances
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.path.chmod(0o600)

    def build_prompt_instruction(self):
        return ('只读查询宿主允许的服务日志，优先用于排查服务/回复失败；不需要执行 Shell。'
                'unit 必须来自枚举；since/until 为 Unix 秒，省略则最近一小时；最多一天，limit为最近记录条数。'
                '返回原查询快照与哈希，正文会进入同一任务的答案证据。'
                '查询的服务是日志归属，不能把其他实例日志当目标故障；当前实例PID不覆盖历史条目PID。'
                'not_exhaustive=true 表示窗口还有较早记录，不能声称穷尽或没有其他错误。'
                '调查结论用 local_answer；不要把读取日志虚构成 command_run 条件。')

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': {'unit': {'type': 'string', 'enum': list(self.units)},
                           'since': {'type': 'number'}, 'until': {'type': 'number'},
                           'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200}},
            'required': ['unit']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not str(k).startswith('_tool_') and k != '_native_tool_call'}
        if set(args) - {'unit', 'since', 'until', 'limit'} or args.get('unit') not in self.units:
            return None
        if type(args.get('limit', 80)) is not int or not 1 <= args.get('limit', 80) <= 200:
            return None
        for key in ('since', 'until'):
            if key in args and (type(args[key]) not in (int, float) or not math.isfinite(args[key]) or args[key] < 0):
                return None
        return args

    def _owner(self, context):
        binding = self._authorized_task(context)
        task = context['agent_task_state']
        if tuple(task.get(k) for k in ('requester', 'session', 'task_id')) != binding:
            raise AgentBoundaryError('service_log_binding_changed')
        from shinku.agent.answer_protocol import request_binding
        return [*binding, task['execution_attempt'], request_binding(task), context['agent_authority_revision']]

    def execute(self, *, call, context):
        try:
            owner = self._owner(context)
            args = self.normalize_call(call)
            if args is None:
                raise ValueError('invalid_service_log_query')
            end = args.get('until', int(time.time()))
            start = args.get('since', max(0, end-3600))
            if not 0 < end-start <= 86400 or end > time.time()+5:
                raise ValueError('invalid_service_log_window')
            query = {'unit': args['unit'], 'since': start, 'until': end, 'limit': args.get('limit', 80)}
            observed = self.reader(query)
            if (not isinstance(observed, dict) or set(observed) != {'entries', 'more_in_window'}
                    or not isinstance(observed['entries'], list) or len(observed['entries']) > query['limit']
                    or type(observed['more_in_window']) is not bool):
                raise ValueError('invalid_service_log_result')
            # 只绑定登记的服务归属，不用当前PID重写历史日志，旧快照不迁移。
            instance_scope = self.instances.log_scope(query['unit']) if self.instances else {
                'registered_instances': [], 'attribution': 'unit_only_instance_unregistered'}
            text = safe_text(json.dumps({'instance_scope': instance_scope, 'query': query, 'scope': 'latest_records_in_selected_window',
                'not_exhaustive': observed['more_in_window'], 'entries': observed['entries']}, ensure_ascii=False))
            if len(text) > 60000:
                raise ValueError('service_log_output_too_large')
            if self._owner(context) != owner:
                raise ValueError('service_log_authorization_changed')
            identifier = uuid.uuid4().hex
            digest = hashlib.sha256(text.encode()).hexdigest()
            receipt = {'path': 'journal:' + identifier, 'sha256': digest, 'query': args,
                       'read_complete': True, 'verified_effects': ['read'],
                       'completion_targets': ['journal:' + identifier]}
            record = {'owner': owner, 'receipt': receipt, 'text': text}
            with sqlite3.connect(self.path) as db:
                db.execute('INSERT INTO snapshots VALUES (?,?)', (identifier, json.dumps(record, ensure_ascii=False)))
            return tool_result(self.tool_type, {**receipt, 'log_text': text})
        except (AgentBoundaryError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
            code = exc.code if isinstance(exc, AgentBoundaryError) else str(exc)
            if len(code) > 80 or ' ' in code:
                code = 'service_logs_unavailable'
            return tool_result(self.tool_type, {'code': code}, error=code)

    def agent_verify_result(self, *, receipt, context):
        """拒绝跨任务、旧请求、换尝试、撤权和篡改；重启后原绑定仍须完整重验。"""
        try:
            owner = self._owner(context)
            reference = receipt.get('path', '')
            if not isinstance(reference, str) or not reference.startswith('journal:'):
                return {'result_valid': False}
            with sqlite3.connect(self.path) as db:
                row = db.execute('SELECT payload FROM snapshots WHERE id=?', (reference[8:],)).fetchone()
            stored = json.loads(row[0]) if row else {}
            expected = stored.get('receipt', {})
            valid = (stored.get('owner') == owner and expected
                     and all(receipt.get(k) == v for k, v in expected.items())
                     and hashlib.sha256(stored['text'].encode()).hexdigest() == receipt.get('sha256')
                     and self._owner(context) == owner)
            return {'result_valid': bool(valid), 'sha256': expected.get('sha256'),
                    'file_text': stored['text'] if valid else ''}
        except (AgentBoundaryError, OSError, ValueError, KeyError, sqlite3.Error):
            return {'result_valid': False}


def install_service_logs(service, requester, path, units, *, instances=None):
    """显式安装后再由四档权限收紧；不修改 Bot 注册表或通用 Shell 权限。"""
    if service.executor.tool_host.registry.get(ServiceLogsTool.tool_type):
        raise AgentBoundaryError('service_log_tool_conflict')
    tool = ServiceLogsTool(service, requester, path, units, instances=instances)
    service.executor.tool_host.registry.register(tool)
    original = service.authorize
    def authorize(scope, task, operation):
        grant = original(scope, task, operation)
        if (grant is None or scope.requester != requester or not scope.session.startswith('angelos:')
                or not grant.policy.accepts(tool.tool_type)):
            return grant
        policy = replace(grant.policy, risk_by_tool={tool.tool_type: 'low', **grant.policy.risk_by_tool})
        return replace(grant, allowed_tools=grant.allowed_tools | {tool.tool_type}, policy=policy)
    service.authorize = authorize
    return tool
