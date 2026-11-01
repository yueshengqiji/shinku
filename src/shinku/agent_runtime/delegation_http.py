"""显式本机 HTTP 委托客户端；无自动重试、重定向或本地执行降级。

请求身份来自启动器，结果用连接密钥和本次随机数绑定。任务结果仍由原
AgentService 产生；网络成功、签名正确均不代表任务已经完成。
"""
from dataclasses import asdict
import hashlib
import hmac
import ipaddress
import json
import math
import secrets
import threading
from urllib.parse import urlsplit

import requests

from .contracts import AgentBoundaryError, AgentTurnRequest, TaskHandle

MAX_WIRE_BYTES = 24 * 1024 * 1024


def wire_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signature(secret, data):
    return hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()


def authority_payload(grant):
    """只接受宿主派生的 ExecutionGrant，不从聊天工具参数取授权字段。"""
    from .execution import ExecutionGrant
    if not isinstance(grant, ExecutionGrant) or not grant.still_authorized():
        raise AgentBoundaryError('host_authorization_denied')
    budget = grant.budget()
    # 现有委托协议只传显式有限预算；本机持续模式不能顺带扩散到另一宿主。
    if budget.metered_only:
        raise AgentBoundaryError('delegation_finite_budget_required')
    return {'revision': grant.authorization_ref, 'allowed_tools': sorted(grant.allowed_tools),
            'approval_tools': sorted(n for n in grant.allowed_tools if grant.policy.requires_approval(n)),
            'blocked_tools': sorted(grant.policy.blocked), 'permission_mode': grant.permission_mode,
            'budget_limits': dict(grant.budget_limits)}


def intersect_authority(base, authority, *, readable_tools=()):
    """服务端与来源宿主取权限/预算交集；这是授权校验，不是自然语言分流。"""
    from dataclasses import replace
    from .execution import ExecutionGrant
    from shinku.tools.execution import ExecutionPolicy
    if not isinstance(base, ExecutionGrant) or not isinstance(authority, dict) or set(authority) != {
            'revision', 'allowed_tools', 'approval_tools', 'blocked_tools', 'permission_mode', 'budget_limits'}:
        raise AgentBoundaryError('delegation_authority_invalid')
    if not isinstance(authority['revision'], str) or not 1 <= len(authority['revision']) <= 512:
        raise AgentBoundaryError('delegation_authority_invalid')
    for key in ('allowed_tools', 'approval_tools', 'blocked_tools'):
        if not isinstance(authority[key], list) or len(authority[key]) > 2048 or any(
                not isinstance(n, str) or not 1 <= len(n) <= 256 for n in authority[key]):
            raise AgentBoundaryError('delegation_authority_invalid')
    if authority['permission_mode'] not in {'read','readonly','ask','traditional','confirm','full','ultra'}:
        raise AgentBoundaryError('delegation_authority_invalid')
    limits = authority['budget_limits']
    if not isinstance(limits, dict) or set(limits) - {'max_requests','max_tokens','max_tools','max_active_seconds'}:
        raise AgentBoundaryError('delegation_authority_invalid')
    if any(type(v) is not int or v < 1 for k,v in limits.items() if k != 'max_active_seconds'):
        raise AgentBoundaryError('delegation_authority_invalid')
    active = limits.get('max_active_seconds', 1)
    if type(active) not in (int,float) or not math.isfinite(active) or active <= 0:
        raise AgentBoundaryError('delegation_authority_invalid')
    ExecutionGrant('validation', frozenset(), budget_limits=limits).budget()
    names = (set(base.allowed_tools) & set(authority['allowed_tools'])) - set(authority['blocked_tools'])
    names = {n for n in names if base.policy.accepts(n)}
    if authority['permission_mode'] in {'read','readonly'}:
        names &= set(readable_tools)
    approval = {n for n in names if base.policy.requires_approval(n)} | (names & set(authority['approval_tools']))
    # 契约声明/工作计划等是宿主内部控制，不扩大业务工具权限；ZAYIN的明确
    # blocked仍优先，模型不能靠声明契约取得文件或Shell能力。
    internal = {'declare_task_contract', 'read_self_state', 'update_work_plan', 'read_work_context', 'task_material_read', 'request_context_compaction'}
    internal = {n for n in internal if base.policy.accepts(n) and n not in authority['blocked_tools']}
    policy = ExecutionPolicy(allowed=frozenset(names | internal),
        blocked=base.policy.blocked | set(authority['blocked_tools']),
        risk_by_tool={n: 'high' if n in approval else 'low' for n in names},
        approval_required_by_risk=frozenset({'high'}))
    server_budget, caller_budget = base.budget(), ExecutionGrant('validation', frozenset(), budget_limits=limits).budget()
    bounded = {key: (getattr(caller_budget, key) if server_budget.metered_only else
                     min(getattr(server_budget, key), getattr(caller_budget, key))) for key in
               ('max_requests','max_tokens','max_tools','max_active_seconds')}
    # 来源档位与工具交集一并保留，不能用服务端默认ask或本机ALEPH掩盖来源full。
    # 这些字段由已认证宿主派生，仅解释权限；实际调用仍以names/policy为准。
    server_names = {n for n in base.allowed_tools if base.policy.accepts(n)}
    source_names = set(authority['allowed_tools']) - set(authority['blocked_tools'])
    if authority['permission_mode'] in {'read', 'readonly'}:
        source_names &= set(readable_tools)
    authority_view = {'source_mode': authority['permission_mode'],
        'source_excluded_tools': sorted(server_names - source_names),
        'server_excluded_tools': sorted(source_names - server_names)}
    return replace(base, authorization_ref=base.authorization_ref + ':' + authority['revision'],
                   allowed_tools=frozenset(names), policy=policy, budget_limits=bounded,
                   permission_mode=authority['permission_mode'],
                   tool_context={**base.tool_context, 'delegation_authority': authority_view})


class RemoteAgentClient:
    def __init__(self, endpoint, host_id, secret, *, authority, session=None, timeout=5):
        if (not isinstance(endpoint,str) or not isinstance(host_id,str) or not 1 <= len(host_id) <= 128
                or not isinstance(secret,str) or not 32 <= len(secret) <= 4096):
            raise ValueError('invalid_delegation_connection')
        parsed = urlsplit(endpoint)
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
            parsed.port
        except (ValueError, TypeError):
            local = False
        if (parsed.scheme != 'http' or not local or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/') or len(secret) < 32 or not callable(authority) or not 0 < timeout <= 30):
            raise ValueError('invalid_delegation_connection')
        self.endpoint, self.host_id, self.secret = endpoint.rstrip('/'), host_id, secret
        self.authority, self.timeout = authority, timeout
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.lock = threading.RLock()

    def call(self, scope, operation, arguments):
        # requests.Session不共享并发可变状态；续租和轮询串行使用同一连接池。
        with self.lock:
            return self._call(scope, operation, arguments)

    def _call(self, scope, operation, arguments):
        payload = {'scope': asdict(scope), 'operation': operation, 'arguments': arguments,
                   'authority': self.authority(scope, operation), 'nonce': secrets.token_hex(24)}
        try:
            body = wire_bytes(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AgentBoundaryError('delegation_input_invalid') from exc
        if len(body) > MAX_WIRE_BYTES:
            raise AgentBoundaryError('delegation_input_oversized')
        try:
            response = self.session.post(self.endpoint + '/angelos/host/v1', data=body,
                headers={'Authorization': 'Bearer ' + self.secret, 'X-Angelos-Host': self.host_id,
                         'Content-Type': 'application/json'}, timeout=self.timeout, allow_redirects=False, stream=True)
            with response:
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > MAX_WIRE_BYTES:
                        raise AgentBoundaryError('delegation_result_oversized')
                if not hmac.compare_digest(response.headers.get('X-Angelos-Signature', ''), signature(self.secret, raw)):
                    raise AgentBoundaryError('delegation_response_untrusted')
                value = json.loads(raw)
                if value.get('nonce') != payload['nonce']:
                    raise AgentBoundaryError('delegation_response_replayed')
                if not value.get('ok'):
                    raise AgentBoundaryError(value.get('reason', 'delegation_request_failed'))
                if response.status_code != 200:
                    raise AgentBoundaryError('delegation_transport_failed')
                return value['value']
        except AgentBoundaryError:
            raise
        except Exception as exc:
            # 异常中可能有连接凭据，不回显原文；提交响应丢失不自动重提或重执行。
            raise AgentBoundaryError('delegation_transport_unavailable') from exc

    def close(self):
        self.session.close()

    def port(self, scope):
        return RemoteAgentPort(self, scope)


class RemoteAgentPort:
    def __init__(self, client, scope):
        self.client, self.scope = client, scope

    def submit(self, request, *, materials=(), material_gaps=()):
        if not isinstance(request, AgentTurnRequest):
            raise AgentBoundaryError('typed_request_required')
        request.payload()
        from .delegation import material_payload
        return TaskHandle(**self.client.call(self.scope, 'submit', {'request': asdict(request),
            'materials': [material_payload(s) for s in materials], 'material_gaps': list(material_gaps)}))

    def status(self, task_id):
        return self.client.call(self.scope, 'status', {'task_id': task_id})

    def context(self, task_id):
        """只读接续背景；响应签名之外仍核对作用域与任务绑定。"""
        value = self.client.call(self.scope, 'context', {'task_id': task_id})
        if (not isinstance(value, dict) or value.get('scope') != asdict(self.scope)
                or value.get('task_id') != task_id or value.get('available') is not True
                or type(value.get('attempt')) is not int or type(value.get('version')) is not int
                or not isinstance(value.get('input_binding'), str)):
            raise AgentBoundaryError('delegation_context_binding_invalid')
        return value

    def result(self, task_id):
        return self._result('result', task_id)

    def handoff(self, task_id):
        """默认交付正文；完整机器回执在绑定版本的details读取。"""
        return self._result('handoff', task_id)

    def details(self, result):
        from .handoff import details_hash
        return self._result('details', result['task_id'], result_hash=details_hash(result))

    def request_summary(self, result):
        """仅宿主显式调用；普通handoff没有额外模型请求，未配置时回退。"""
        from .handoff import details_hash
        return self.client.call(self.scope, 'request_summary',
            {'task_id': result['task_id'], 'result_hash': details_hash(result)})

    def summary(self, result):
        from .handoff import details_hash
        return self.client.call(self.scope, 'summary',
            {'task_id': result['task_id'], 'result_hash': details_hash(result)})

    def messages(self, task_id, *, after=0):
        return self.client.call(self.scope, 'messages', {'task_id': task_id, 'after': after})

    def pending_files(self, task_id):
        return self.client.call(self.scope, 'file_deliveries', {'task_id': task_id})

    def file_payload(self, task_id, delivery_id):
        return self.client.call(self.scope, 'file_payload', {'task_id': task_id, 'delivery_id': delivery_id})

    def acknowledge_file(self, task_id, delivery_id, receipt):
        return self.client.call(self.scope, 'file_receipt', {
            'task_id': task_id, 'delivery_id': delivery_id, 'receipt': receipt})

    def _result(self, operation, task_id, **arguments):
        value = self.client.call(self.scope, operation, {'task_id': task_id, **arguments})
        from shinku.result_expression.facts import validate_result
        error = validate_result(value, self.scope)
        if error or value.get('task_id') != task_id:
            raise AgentBoundaryError(error or 'delegation_result_binding_invalid')
        return value

    def cancel(self, handle):
        return self.client.call(self.scope, 'cancel', asdict(handle))

    def recovery(self, handle):
        return self.client.call(self.scope, 'recovery', asdict(handle))

    def resume(self, handle, *, request_key):
        return TaskHandle(**self.client.call(self.scope, 'resume', {**asdict(handle), 'request_key': request_key}))

    def supplement(self, handle, request, *, materials=(), material_gaps=()):
        from .delegation import material_payload
        return TaskHandle(**self.client.call(self.scope, 'supplement', {**asdict(handle), 'request': asdict(request),
            'materials': [material_payload(s) for s in materials], 'material_gaps': list(material_gaps)}))

    def revise(self, handle, request, *, materials=(), material_gaps=()):
        """保存中途补充；pending表示仍等待停止，applied也不表示已续跑。"""
        from .delegation import material_payload
        return self.client.call(self.scope, 'revise', {**asdict(handle), 'request': asdict(request),
            'materials': [material_payload(s) for s in materials], 'material_gaps': list(material_gaps)})

    def presentation(self, result):
        return self.client.call(self.scope, 'presentation', {'task_id': result['task_id'], 'result_hash': result['result_hash']})

    def validate_presentation(self, result, projection):
        return self.client.call(self.scope, 'validate_presentation', {'task_id': result['task_id'],
            'result_hash': result['result_hash'], 'projection': projection})['valid'] is True
