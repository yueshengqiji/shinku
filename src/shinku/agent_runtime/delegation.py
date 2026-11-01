"""独立 Agent 的可信宿主入口；复用任务库、协调器和执行器，不包含意图分类。

模型只提交目标。连接身份、调用者作用域、权限和短期授权租约由可信宿主
适配器提供；每次工具执行仍使用原执行授权。归档保存原资料，不连接 Bot 记忆。
"""
from dataclasses import asdict, dataclass, field, replace
import copy
import hmac
import threading
import time
import uuid

from .contracts import (AgentBoundaryError, AgentScope, AgentSupplement, AgentTurnRequest,
                        TaskHandle, canonical_hash)
from .execution import ExecutionGrant
from .materials import MaterialSource


@dataclass(frozen=True)
class DelegationPrincipal:
    """启动器提供的连接身份；authorize 不得从模型输出创建。"""
    secret: str = field(repr=False)
    authorize: object
    allowed_sessions: tuple[str, ...] = ('private:', 'group:', 'desktop:')

    def __post_init__(self):
        if (not isinstance(self.secret, str) or not 32 <= len(self.secret) <= 4096 or not callable(self.authorize)
                or not isinstance(self.allowed_sessions, tuple) or not self.allowed_sessions
                or any(not isinstance(s, str) or not s or s.startswith('angelos:') for s in self.allowed_sessions)):
            raise ValueError('invalid_delegation_principal')


def material_payload(source):
    """只跨进程传声明数据；活回调不能悄悄降级成永不过期的快照。"""
    if not isinstance(source, MaterialSource) or source.revalidate is not None:
        raise AgentBoundaryError('delegation_material_validator_unavailable')
    row = {k: copy.deepcopy(getattr(source, k)) for k in
           ('source_id', 'kind', 'content', 'provenance', 'required', 'version', 'complete')}
    row['local_path'] = str(source.local_path) if source.local_path else None
    return row


class DelegationGateway:
    """唯一宿主租约写入者。失联到期停止后续步骤；撤权不回滚已有副作用。"""
    def __init__(self, service, principals, *, lease_seconds=5, clock=time.time, grant_filter=None, file_delivery=None, summary_options=None):
        if not 0 < lease_seconds <= 30 or not principals:
            raise ValueError('invalid_delegation_gateway')
        self.service, self.principals = service, dict(principals)
        self.lease_seconds, self.clock = lease_seconds, clock
        self.grant_filter = grant_filter or (lambda grant: grant)
        self.file_delivery = file_delivery
        # 宿主显式配置预算与生成器；普通handoff从不启动摘要任务。
        from .summary_worker import SummaryWorker
        self.summaries = SummaryWorker(service.task_store, self._summary_original, **(summary_options or {}))
        self.lock = threading.RLock()
        self.leases = {}
        self.materials = {}
        from .handoff import TaskConversation
        self.conversation = TaskConversation(service.task_store)
        from .task_revision import TaskRevisions
        self.revisions = TaskRevisions(self)
        with service.task_store._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS agent_delegation_owners
                (scope TEXT PRIMARY KEY, host TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS agent_delegation_inputs
                (task_id TEXT PRIMARY KEY, digest TEXT NOT NULL)''')
        with service.task_store._transaction() as db:
            self.remote_scopes = {row[0] for row in db.execute('SELECT scope FROM agent_delegation_owners')}
        previous = service.authorize
        previous_conversation = service.conversation_authorizer
        def authorize(scope, task, operation):
            key = canonical_hash(asdict(scope))
            with self.lock:
                lease = self.leases.get((key, task.get('task_id')))
            if lease is None:
                return None if key in self.remote_scopes else previous(scope, task, operation)
            host, authority, deadline, generation = lease
            if deadline <= self.clock():
                return None
            from .delegation_http import intersect_authority
            grant = self.grant_filter(intersect_authority(self.principals[host].authorize(scope, task, operation), authority, readable_tools=self._readable()))
            if not isinstance(grant, ExecutionGrant):
                return None
            reference = authority.get('revision')
            revoked = [False]
            def current():
                with self.lock:
                    live = self.leases.get((key, task.get('task_id')))
                    valid = bool(not revoked[0] and live and live[0] == host and live[3] == generation
                                 and live[1].get('revision') == reference and live[2] > self.clock()
                                 and grant.still_authorized())
                    if not valid:
                        revoked[0] = True
                    return valid
            return replace(grant, still_authorized=current,
                material_sources=(*grant.material_sources, *self.materials.get((key, task.get('task_id')), ((), ()))[0]),
                material_gaps=(*grant.material_gaps, *self.materials.get((key, task.get('task_id')), ((), ()))[1]),
                tool_context={**grant.tool_context, 'agent_delegation_host': host})
        service.authorize = authorize
        service.conversation_authorizer = lambda scope: self._owns(scope) or bool(
            previous_conversation and previous_conversation(scope))

    def _summary_original(self, scope, task_id):
        """后台线程没有HTTP请求锁，须自行复验实时授权、当前修订及必需资料。"""
        with self.service._lock:
            task = self.service._get(scope, task_id)
            grant = self.service._grant(scope, task, 'inspect')
            result = self.service.bind(scope).result(task_id)
            if ((result['attempt'], result['version']) != (task['execution_attempt'], task['revision'])
                    or task.get('cancel_requested')):
                raise AgentBoundaryError('summary_result_changed')
            # 仅校验资料身份与授权，不重新分页读取审计日志或消耗模型材料额度。
            for source in result.get('information', {}).get('sources', []):
                if source.get('required') is not True:
                    continue
                if self.service.materials is None:
                    raise AgentBoundaryError('summary_required_material_unavailable')
                record = self.service.materials._record(source['reference'], task)
                self.service.materials.resolve(source['reference'], task=task,
                                               still_authorized=grant.still_authorized)
                if any(record.get(k) != source.get(k) for k in ('content_sha256', 'version')):
                    raise AgentBoundaryError('material_version_conflict')
            if not grant.still_authorized():
                raise AgentBoundaryError('host_authorization_denied')
            return result

    def _readable(self):
        return {name for name, handler in self.service.executor.tool_host.registry.view().items()
            if getattr(handler, 'agent_permission_class', '') == 'read' and not
            set(getattr(handler, 'agent_completion_effects', ()) or ()).intersection(
                {'write','delete','send','process_stop','command_run','memory_manage'})}

    def authenticate(self, host, credential):
        principal = self.principals.get(host)
        if principal is None or not isinstance(credential, str) or not hmac.compare_digest(principal.secret, credential):
            raise AgentBoundaryError('delegation_authentication_denied')
        return principal

    def _owns(self, scope):
        with self.lock:
            key = canonical_hash(asdict(scope))
            return any(k[0] == key and value[2] > self.clock() for k, value in self.leases.items())

    def dispatch(self, host, payload):
        """HTTP 适配器认证后调用；原文与宿主授权位于不同字段，拒绝未知字段。"""
        if not isinstance(payload, dict) or set(payload) != {'scope', 'authority', 'operation', 'arguments', 'nonce'}:
            raise AgentBoundaryError('delegation_protocol_invalid')
        expected = {
            'capabilities': set(), 'renew': {'task_id'}, 'status': {'task_id'}, 'result': {'task_id'},
            'context': {'task_id'},
            'revoke': {'task_id', 'attempt'}, 'submit': {'request', 'materials', 'material_gaps'},
            'presentation': {'task_id', 'result_hash'},
            'validate_presentation': {'task_id', 'result_hash', 'projection'},
            'handoff': {'task_id'}, 'details': {'task_id', 'result_hash'},
            'messages': {'task_id', 'after'},
            'request_summary': {'task_id', 'result_hash'},
            'summary': {'task_id', 'result_hash'},
            'file_deliveries': {'task_id'},
            'file_payload': {'task_id', 'delivery_id'},
            'file_receipt': {'task_id', 'delivery_id', 'receipt'},
            'cancel': {'task_id', 'attempt', 'version'},
            'recovery': {'task_id', 'attempt', 'version'},
            'resume': {'task_id', 'attempt', 'version', 'request_key'},
            'supplement': {'task_id', 'attempt', 'version', 'request', 'materials', 'material_gaps'},
            'revise': {'task_id', 'attempt', 'version', 'request', 'materials', 'material_gaps'},
        }
        operation, arguments = payload['operation'], payload['arguments']
        if not isinstance(operation, str) or operation not in expected or not isinstance(arguments, dict) or set(arguments) != expected[operation]:
            raise AgentBoundaryError('delegation_protocol_invalid')
        principal = self.principals[host]
        scope = AgentScope(**payload['scope'])
        if scope.requester == 'local-angelos' or not scope.session.startswith(principal.allowed_sessions):
            raise AgentBoundaryError('delegation_scope_denied')
        authority, arguments, operation = payload['authority'], payload['arguments'], payload['operation']
        if not isinstance(authority, dict) or not isinstance(arguments, dict):
            raise AgentBoundaryError('delegation_protocol_invalid')
        key = canonical_hash(asdict(scope))
        # 连接身份拥有作用域，另一连接不能覆盖同名用户/会话的租约。
        with self.service._lock, self.lock:
            from .delegation_http import intersect_authority
            probe = self.grant_filter(intersect_authority(principal.authorize(scope, {}, operation), authority, readable_tools=self._readable()))
            if not isinstance(probe, ExecutionGrant) or not probe.still_authorized():
                raise AgentBoundaryError('host_authorization_denied')
            probe.budget()
            with self.service.task_store._transaction() as db:
                owner = db.execute('SELECT host FROM agent_delegation_owners WHERE scope=?', (key,)).fetchone()
                if owner and owner[0] != host:
                    raise AgentBoundaryError('delegation_scope_owner_conflict')
                if not owner:
                    db.execute('INSERT INTO agent_delegation_owners VALUES (?,?)', (key, host))
            self.remote_scopes.add(key)
            target = arguments.get('task_id')
            if operation == 'submit':
                target = 'agent:' + canonical_hash([scope.requester, scope.session, arguments['request']['request_key']])[:32]
            if target:
                old = self.leases.get((key, target))
                generation = old[3] if (old and old[0] == host and old[2] > self.clock()
                    and old[1].get('revision') == authority.get('revision')) else uuid.uuid4().hex
                self.leases[(key, target)] = (host, copy.deepcopy(authority), self.clock() + self.lease_seconds, generation)
            port = self.service.bind(scope)
            if operation == 'capabilities':
                if arguments:
                    raise AgentBoundaryError('delegation_protocol_invalid')
                registry = self.service.executor.tool_host.registry
                names = {n for n in probe.allowed_tools if probe.policy.accepts(n)}
                return {'tools': registry.native_specs(allowed_tool_names=names) if names else [],
                        'host_file_delivery': self.file_delivery is not None and 'private_qq_send_file' in names,
                        'lease_seconds': self.lease_seconds, 'data_is_not_authority': True}
            if operation == 'revoke':
                task = self.service._get(scope, arguments['task_id'])
                if task.get('execution_attempt') != arguments.get('attempt'):
                    raise AgentBoundaryError('stale_execution_attempt')
                self.leases.pop((key, arguments['task_id']), None)
                return {'authorization_withdrawn': True}
            if operation == 'renew':
                task = self.service._get(scope, arguments['task_id'])
                self.service._grant(scope, task, 'inspect')
                return {'renewed': True}
            if operation == 'submit':
                if set(arguments) != {'request', 'materials', 'material_gaps'}:
                    raise AgentBoundaryError('delegation_protocol_invalid')
                rows = arguments['materials']
                material_sources, gaps = self._decode_materials(arguments)
                request = AgentTurnRequest(**arguments['request'])
                task_id = 'agent:' + canonical_hash([scope.requester, scope.session, request.request_key])[:32]
                digest = canonical_hash([rows, gaps])
                with self.service.task_store._transaction() as db:
                    existing = db.execute('SELECT digest FROM agent_delegation_inputs WHERE task_id=?', (task_id,)).fetchone()
                    if existing and existing[0] != digest:
                        raise AgentBoundaryError('delegation_material_conflict')
                    if not existing:
                        db.execute('INSERT INTO agent_delegation_inputs VALUES (?,?)', (task_id, digest))
                self.materials[(key, task_id)] = (material_sources, copy.deepcopy(gaps))
                handle = port.submit(request)
                self.conversation.append(self.service._get(scope, handle.task_id), 'submit', request.original_request,
                    key='submit:' + request.request_key, author='caller',
                    data={'history_context': request.history_context, 'sources': list(request.sources)})
                return asdict(handle)
            task_id = arguments.get('task_id')
            task = self.service._get(scope, task_id)
            self.service._grant(scope, task, 'inspect')
            if operation in {'file_deliveries', 'file_payload', 'file_receipt'}:
                if self.file_delivery is None:
                    raise AgentBoundaryError('file_delivery_unavailable')
                if operation == 'file_deliveries':
                    return self.file_delivery.pending(scope, task_id)
                if operation == 'file_payload':
                    return self.file_delivery.payload(scope, task_id, arguments['delivery_id'])
                return self.file_delivery.acknowledge(scope, task_id, arguments['delivery_id'], arguments['receipt'])
            if operation in {'request_summary', 'summary'}:
                action = self.summaries.request if operation == 'request_summary' else self.summaries.read
                return action(scope, task_id, arguments['result_hash'])
            if operation == 'revise':
                return self.revisions.accept(scope, task, arguments)
            if operation in {'status', 'result', 'handoff', 'messages', 'recovery', 'resume'}:
                self.revisions.drain(scope, task_id)
                task = self.service._get(scope, task_id)
            if operation == 'context':
                from .continuation_context import continuation_context
                return continuation_context(self.service, scope, task)
            if operation == 'messages':
                return self.conversation.read(task, after=arguments['after'])
            if operation in {'status', 'result', 'handoff', 'details'}:
                from .handoff import delivery_result
                if operation == 'details':
                    result = port.result(task_id)
                    view = port.status(task_id)
                    if ((result['attempt'], result['version']) != (view['attempt'], view['version'])
                            or result['result_hash'] != arguments['result_hash']):
                        raise AgentBoundaryError('delegation_result_changed')
                    return result
                if set(arguments) != {'task_id'}:
                    raise AgentBoundaryError('delegation_protocol_invalid')
                value = getattr(port, 'result' if operation == 'handoff' else operation)(task_id)
                kind = ('result' if value['execution']['result_available'] else 'progress') if operation != 'status' else 'progress'
                state = value['execution']['status'] if operation != 'status' else value['status']
                message_task = {**task, 'execution_attempt': value.get('attempt', task.get('execution_attempt')),
                                'revision': value.get('version', task.get('revision'))}
                self.conversation.append(message_task, 'clarification' if state == 'waiting_user' else kind,
                    value.get('execution_report', {}).get('text', '') or state,
                    key=kind + ':' + canonical_hash([state, message_task.get('execution_attempt'), message_task.get('revision'),
                                                     value.get('execution_report', {}).get('text', '')]), author='angelos',
                    data={'status': state, 'result_hash': value.get('result_hash', '') if kind == 'result' else ''})
                return delivery_result(value) if operation == 'handoff' else value
            if operation in {'presentation', 'validate_presentation'}:
                result = port.result(task_id)
                from .handoff import delivery_result
                projected = arguments.get('result_hash') == delivery_result(result)['result_hash']
                if result['result_hash'] != arguments.get('result_hash') and not projected:
                    raise AgentBoundaryError('delegation_result_changed')
                from .material_presentation import recover_presentation, validate_presentation
                current = lambda: self.service._grant(scope, self.service._get(scope, task_id), 'inspect').still_authorized()
                if operation == 'presentation':
                    projection = recover_presentation(self.service.materials, result, scope=scope,
                        still_authorized=current, delivery=projected)
                    if projected:
                        projection['source_result_hash'] = arguments['result_hash']
                    return projection
                projection = copy.deepcopy(arguments['projection'])
                if projected:
                    if projection.get('source_result_hash') != arguments['result_hash']:
                        raise AgentBoundaryError('presentation_source_conflict')
                    projection['source_result_hash'] = result['result_hash']
                return {'valid': validate_presentation(self.service.materials, result, projection,
                                                       scope=scope, still_authorized=current)}
            handle = TaskHandle(task_id, arguments['attempt'], arguments['version'])
            if operation == 'cancel':
                receipt = port.cancel(handle)
                self.conversation.append(task, 'cancel', '取消请求已登记，停止与否以回执为准。',
                    key=f'cancel:{handle.attempt}:{handle.version}', author='caller', data=receipt)
                return receipt
            if operation == 'recovery':
                return port.recovery(handle)
            if operation == 'resume':
                updated = port.resume(handle, request_key=arguments['request_key'])
                self.conversation.append(task, 'resume', '继续原任务。', key='resume:' + arguments['request_key'], author='caller')
                return asdict(updated)
            if operation == 'supplement':
                sources, gaps = self._decode_materials(arguments)
                # 输入只供本次补料事务；旧资料已在原任务材料库持久保留，不依赖内存重建。
                old = self.materials.pop((key, task_id), None)
                self.materials[(key, task_id)] = (sources, gaps)
                try:
                    request = AgentSupplement(**arguments['request'])
                    updated = port.supplement(handle, request)
                    self.conversation.append(self.service._get(scope, task_id), 'supplement', request.clarification,
                        key='supplement:' + request.request_key, author='caller', data={'constraints': list(request.constraints)})
                    return asdict(updated)
                finally:
                    self.materials.pop((key, task_id), None)
                    if old is not None:
                        self.materials[(key, task_id)] = old
            raise AgentBoundaryError('delegation_operation_invalid')

    @staticmethod
    def _decode_materials(arguments):
        rows, gaps = arguments['materials'], arguments['material_gaps']
        if not isinstance(rows, list) or len(rows) > 64 or not isinstance(gaps, list) or len(gaps) > 64:
            raise AgentBoundaryError('delegation_materials_invalid')
        sources = tuple(MaterialSource(**row) for row in rows)
        if any(source.kind == 'executor_result' or source.provenance.get('invocation_id') for source in sources):
            raise AgentBoundaryError('delegation_forged_execution_material')
        return sources, copy.deepcopy(gaps)
