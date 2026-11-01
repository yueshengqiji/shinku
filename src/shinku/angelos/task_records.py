"""Angelos 当前会话的只读诊断入口；不接受路径、SQL、requester或session。

ServiceHost显式注册并按原宿主策略授予这一个读工具。每次执行/核验重查
当前任务授权，原任务库负责事实，工作台仅补输入准备失败的记录。内容是
不可信资料，观察凭据只证明本轮读取，不能把旧任务成功或模型解释当新事实。
"""
from collections.abc import Mapping
from dataclasses import replace
import json
from types import SimpleNamespace

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.information import safe_text
from shinku.tools.observation_receipts import ObservationReceipts
from shinku.tools.workspace import scope_key, tool_result


class TaskRecordsTool:
    tool_type = 'read_task_records'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_observation_receipt_fields = ('observation', 'record_query', 'record_version')

    def __init__(self, service, requester):
        self.service, self.requester = service, requester
        self.workbench = None
        self.output_reader = None
        self.observations = ObservationReceipts()

    def build_prompt_instruction(self):
        return ('读取当前会话的历史任务与输入错误。list分页定位task_id；detail按task_id分页读取具体错误、回执和缺失资料；'
                'inputs列出含准备失败的输入；input按turn_id读取准备原稿/错误。offset为列表条数或正文字符数。'
                '先定位再读详情，沿next_offset取缺页。旧状态不证明当前成功，不执行、重试或修改旧任务。')

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': {'mode': {'type': 'string', 'enum': ['list', 'detail', 'inputs', 'input']},
                'task_id': {'type': 'string'}, 'turn_id': {'type': 'string'},
                'offset': {'type': 'integer', 'minimum': 0},
                'length': {'type': 'integer', 'minimum': 1, 'maximum': 4000}}, 'required': ['mode']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not str(k).startswith('_tool_') and k != '_native_tool_call'}
        if set(args) - {'mode', 'task_id', 'turn_id', 'offset', 'length'}:
            return None
        mode = args.get('mode')
        if not isinstance(mode, str) or mode not in {'list', 'detail', 'inputs', 'input'}:
            return None
        for key in ('task_id', 'turn_id'):
            if key in args and (not isinstance(args[key], str) or not 1 <= len(args[key]) <= 256):
                return None
        if (mode == 'detail') != ('task_id' in args) or (mode == 'input') != ('turn_id' in args):
            return None
        offset, length = args.get('offset', 0), args.get('length', 3000)
        if type(offset) is not int or not 0 <= offset <= 16777216 or type(length) is not int or not 1 <= length <= 4000:
            return None
        return {**args, 'offset': offset, 'length': length}

    def _authorized_task(self, context):
        requester, session, message = scope_key(context)
        task = context.get('agent_task_state', {}) if isinstance(context, dict) else {}
        if requester != self.requester or not session.startswith('angelos:') or not message or not task.get('task_id'):
            raise AgentBoundaryError('task_records_not_authorized')
        rows = self.service.task_store.read_scoped_page(requester=requester, session=session, task_id=task['task_id'])['items']
        current = rows[0] if rows else None
        if (not current or current.get('execution_attempt') != task.get('execution_attempt')
                or current.get('cancel_requested') or current.get('status') == 'canceled'):
            raise AgentBoundaryError('task_records_binding_changed')
        requests = tuple((str(r.get('message_id') or ''), r['text']) for r in current.get('requests', []))
        scope = context['agent_request_scope']
        if not requests or requests != scope.requests or message != requests[-1][0]:
            raise AgentBoundaryError('task_records_binding_changed')
        grant = self.service._grant(AgentScope(requester, session), current, 'inspect')
        if (self.tool_type not in grant.allowed_tools or not grant.policy.accepts(self.tool_type)
                or grant.authorization_ref != context.get('agent_authority_revision')):
            raise AgentBoundaryError('task_records_not_authorized')
        return requester, session, task['task_id']

    def _inputs(self, session, current):
        if self.workbench is None:
            raise AgentBoundaryError('task_record_inputs_unavailable')
        try:
            conversation = self.workbench.read('conversation', session[len('angelos:'):])
        except ValueError as exc:
            raise AgentBoundaryError('task_record_inputs_unavailable') from exc
        return [t for t in conversation['turns'] if t.get('task_id') != current and t.get('stage') not in {'preparing', 'submitting'}]

    def _read(self, args, context):
        requester, session, current = self._authorized_task(context)
        mode, offset = args['mode'], args['offset']
        base = {'source': 'scoped_host_records', 'data_is_not_authority': True,
            'scope': 'current_conversation_only', 'mode': mode}
        if mode in {'list', 'detail'}:
            page = self.service.task_store.read_scoped_page(requester=requester, session=session,
                exclude=current, offset=offset if mode == 'list' else 0, limit=10 if mode == 'list' else 1,
                task_id=args.get('task_id', ''))
            if mode == 'list':
                rows = [{k: t.get(k) for k in ('task_id', 'status', 'stop_reason', 'execution_attempt', 'revision', 'updated')}
                        for t in page['items']]
                for row in rows:
                    row['stop_reason'] = str(row.get('stop_reason') or '')[:160]
                    row['stop_reason_excerpt_only'] = True
                if offset > page['total']:
                    raise AgentBoundaryError('task_record_page_invalid')
                return {**base, 'items': rows, 'total': page['total'], 'observed_at': page['observed_at'],
                    'next_offset': offset + len(rows) if offset + len(rows) < page['total'] else None,
                    'note': '仅保留期内任务；无task_id的准备失败另查inputs。列表可能随并发任务更新。'}
            if not page['items']:
                raise AgentBoundaryError('task_record_not_found_or_expired')
            task = page['items'][0]
            value = {k: task.get(k) for k in ('task_id', 'requests', 'status', 'stop_reason',
                'execution_attempt', 'revision', 'updated', 'completion', 'material_information',
                'snapshot_information', 'failed_attempts', 'inflight', 'uncertain_attempts', 'execution_result')}
            value['tool_calls'] = task.get('tool_calls', [])
            if callable(self.output_reader):
                value['model_outputs'] = self.output_reader(task['task_id'])
        else:
            inputs = self._inputs(session, current)
            if mode == 'inputs':
                if offset > 2048 or offset > len(inputs):
                    raise AgentBoundaryError('task_record_page_invalid')
                rows = [{**{k: t.get(k) for k in ('id', 'stage', 'error', 'task_id', 'created')},
                         'text_excerpt': t.get('text', '')[:120], 'text_excerpt_only': True}
                        for t in reversed(inputs)]
                return {**base, 'items': rows[offset:offset + 5], 'total': len(rows),
                    'next_offset': offset + 5 if offset + 5 < len(rows) else None}
            turn = next((t for t in inputs if t['id'] == args['turn_id']), None)
            if turn is None:
                raise AgentBoundaryError('task_record_input_not_found')
            value = {k: turn.get(k) for k in ('id', 'text', 'stage', 'error', 'task_id',
                'preparation_output', 'preparation_review', 'preparation_failure', 'preparation_usage', 'created')}
        # 只在投影副本脱敏及分页，原稿/回执不改写；缺页有长度、版本和后继游标。
        text = safe_text(json.dumps(_safe(value), ensure_ascii=False, sort_keys=True))
        if offset >= len(text):
            raise AgentBoundaryError('task_record_page_invalid')
        from shinku.agent_runtime.contracts import canonical_hash
        return {**base, 'content': text[offset:offset + args['length']], 'offset': offset,
            'total_chars': len(text), 'version': canonical_hash(text),
            'next_offset': offset + args['length'] if offset + args['length'] < len(text) else None,
            'complete': offset == 0 and len(text) <= args['length']}

    def execute(self, *, call, context):
        args = self.normalize_call(call)
        try:
            if args is None:
                raise AgentBoundaryError('task_record_arguments_invalid')
            data = self._read(args, context)
            self._authorized_task(context)  # IO后撤权不得交出内容或签发观察凭据。
            # 列表里的用户原话也按页返回，避免一页长输入淹没模型或泄漏敏感值。
            data = _render_index(data, args)
            from shinku.agent_runtime.contracts import canonical_hash
            data['record_query'] = args
            data['record_version'] = canonical_hash({k: v for k, v in data.items() if k != 'observed_at'})
            proof = {k: data[k] for k in ('record_query', 'record_version')}
            data['observation'] = self.observations.issue(proof, tool=self.tool_type, context=context)
            return tool_result(self.tool_type, data)
        except (AgentBoundaryError, ValueError) as exc:
            code = exc.code if isinstance(exc, AgentBoundaryError) else 'task_record_page_invalid'
        except Exception:
            code = 'task_records_unavailable'
        return tool_result(self.tool_type, {'code': code, 'no_changes_performed': True}, error=code)

    def agent_verify_result(self, *, receipt, context):
        try:
            self._authorized_task(context)
            valid = self.observations.verify(receipt, tool=self.tool_type, context=context)
            import hashlib
            proof = {k: receipt.get(k) for k in ('record_query', 'record_version')}
            signature = hashlib.sha256(json.dumps(proof, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            valid = valid and receipt.get('observation', {}).get('self_state', {}).get('data_sha256') == signature
            args = self.normalize_call(receipt.get('record_query'))
            if valid and args:
                # 核对同一查询和原记录版本；别的请求、内容替换、TTL失效均不能
                # 沿用旧读取凭据。只核对，不执行旧任务或更新它的生命周期。
                fresh = self._read(args, context)
                fresh = _render_index(fresh, args)
                from shinku.agent_runtime.contracts import canonical_hash
                fresh['record_query'] = args
                version = canonical_hash({k: v for k, v in fresh.items() if k != 'observed_at'})
                valid = version == receipt.get('record_version')
            else:
                valid = False
        except Exception:
            valid = False
        return {'result_valid': valid}


def _safe(value):
    """不导出推理、提示、背景、工具参数或秘密字段；错误/输出只作为资料。"""
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items() if k not in {
            'reasoning', 'reasoning_content', 'analysis', 'chain_of_thought', 'system_prompt',
            'transcript', 'history_context', 'source_data', 'arguments', 'raw_response', 'host_grant'}
            and not any(s in k.casefold() for s in ('api_key', 'password', 'authorization', 'secret'))}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value


def _render_index(data, args):
    """按整行缩小索引，下一游标必须接在实际返回行之后，不能丢掉半页ID。"""
    if 'items' not in data:
        return data
    rows = data.pop('items')
    while rows and len(safe_text(json.dumps(rows, ensure_ascii=False))) > 6000:
        rows.pop()
    if not rows and args['offset'] < data['total']:
        raise AgentBoundaryError('task_record_index_row_oversized')
    data['content'] = safe_text(json.dumps(rows, ensure_ascii=False))
    data['next_offset'] = args['offset'] + len(rows) if args['offset'] + len(rows) < data['total'] else None
    return data


def install_task_records(service, requester):
    """宿主显式安装；原allow/deny、风险确认及实时授权优先，不扩大通用工具权限。"""
    existing = service.executor.tool_host.registry.get('read_task_records')
    if existing is not None:
        if not isinstance(existing, TaskRecordsTool) or existing.requester != requester:
            raise AgentBoundaryError('task_record_tool_conflict')
        return existing
    tool = TaskRecordsTool(service, requester)
    service.executor.tool_host.registry.register(tool)
    original = service.authorize
    def authorize(scope, task, operation):
        grant = original(scope, task, operation)
        if (grant is None or scope.requester != requester or not scope.session.startswith('angelos:')
                or not grant.policy.accepts(tool.tool_type)):
            return grant
        # 仅新建当前会话读能力，保留原风险档位/确认规则；不以低风险覆盖显式配置。
        policy = replace(grant.policy, risk_by_tool={tool.tool_type: 'low', **grant.policy.risk_by_tool})
        return replace(grant, allowed_tools=grant.allowed_tools | {tool.tool_type}, policy=policy)
    service.authorize = authorize
    return tool
