"""依赖/服务/连接/登录/业务分层只读诊断，探测目录完全由宿主配置。

注册不是健康证明；上层探测依赖不满足时返回 blocked/unknown，不把它误报为
业务失败。不安装、启动、重启或登录，不回传响应正文/认证头/异常原文。
HTTP 的登录与业务语义必须由宿主显式 predicate 判定，HTTP 200 只证明连接。
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import re
import socket
import subprocess
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .workspace import scope_key, tool_result


@dataclass(frozen=True)
class DiagnosticTarget:
    name: str
    layer: str
    probe: object
    depends_on: tuple[str, ...] = ()
    configured: bool = True


def dependency_probe(module):
    def probe():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            present = False
        return ('ready', 'dependency_available') if present else ('unavailable', 'dependency_missing')
    return probe


def service_probe(unit, *, user=True):
    """固定 systemd 单元的只读检查；不接受模型拼接命令或 shell。"""
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.service', unit):
        raise ValueError('invalid_service_unit')
    def probe():
        try:
            result = subprocess.run(['systemctl', *(['--user'] if user else []), 'is-active', unit],
                                    capture_output=True, timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return 'unknown', 'service_probe_unavailable'
        state = result.stdout.strip()
        if result.returncode == 0 and state == b'active':
            return 'ready', 'service_running'
        if state in {b'inactive', b'failed', b'deactivating'}:
            return 'unavailable', 'service_not_running'
        return 'unknown', 'service_state_unknown'
    return probe


def tcp_probe(host, port):
    def probe():
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return 'ready', 'connection_available'
        except OSError:
            return 'unavailable', 'connection_failed'
    return probe


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_probe(url, *, headers=None, predicate=None, semantic_layer='connection'):
    """URL/凭据/predicate 仅宿主注入，不接收用户或模型任意端点。

    禁用环境代理和重定向，防止本地端点探测把认证转发给另一台主机。
    predicate 只接收有界解析结果，未知结构由 unknown 明确表达。
    """
    from urllib.parse import urlsplit
    from urllib.request import ProxyHandler
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('invalid_diagnostic_endpoint')
    if semantic_layer not in {'connection', 'login', 'business'} or semantic_layer != 'connection' and predicate is None:
        raise ValueError('semantic_probe_requires_predicate')
    def probe():
        try:
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(Request(url, headers=dict(headers or {})), timeout=2) as response:
                raw = response.read(65537)
                if len(raw) > 65536:
                    return 'unknown', 'response_over_limit'
                if response.status != 200:
                    return 'unavailable', 'endpoint_status_failed'
                if semantic_layer == 'connection':
                    return 'ready', 'http_connection_available'
                try:
                    verdict = predicate(json.loads(raw))
                except Exception:
                    return 'unknown', 'endpoint_schema_unknown'
                if verdict is None:
                    return 'unknown', 'endpoint_schema_unknown'
                return ('ready', semantic_layer + '_ready') if verdict is True else ('unavailable', semantic_layer + '_not_ready')
        except HTTPError as exc:
            code = 'authentication_required' if exc.code in {401, 403} else 'endpoint_status_failed'
            exc.close()
            return 'unavailable', code
        except (OSError, URLError, ValueError):
            return 'unavailable', 'connection_failed'
    return probe


class DiagnosticTool:
    tool_type = 'diagnostic_check'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_observation_receipt_fields = ('observation',)

    def __init__(self, targets, *, authorize):
        self.targets = {}
        self.authorize = authorize
        from .observation_receipts import ObservationReceipts
        self.observations = ObservationReceipts()
        for target in targets:
            if (target.name in self.targets or not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,80}', target.name)
                    or target.layer not in {'dependency', 'service', 'connection', 'login', 'business'}
                    or not callable(target.probe) or any(n not in self.targets for n in target.depends_on)):
                raise ValueError('invalid_diagnostic_catalog')
            self.targets[target.name] = target

    def build_prompt_instruction(self):
        return '按宿主目录只读诊断依赖、服务、连接、登录和业务，返回时间/层级/状态码。未探测为未知，不安装、重启或登录。'

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type': 'object', 'properties': {
            'target': {'type': 'string', 'enum': sorted(self.targets)}}, 'required': ['target'], 'additionalProperties': False})

    def normalize_call(self, call):
        if not isinstance(call, dict):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not k.startswith('_tool_')}
        return args if set(args) == {'target'} and isinstance(args['target'], str) and args['target'] in self.targets else None

    def execute(self, *, call, context):
        args = self.normalize_call(call)
        if args is None or not all(scope_key(context)) or not self.authorize(context):
            return tool_result(self.tool_type, {'code': 'diagnostic_not_authorized'}, error='diagnostic_not_authorized')
        rows = {}
        def check(name):
            if name in rows:
                return rows[name]
            target = self.targets[name]
            dependencies = [check(n) for n in target.depends_on]
            if not self.authorize(context):
                state, code = 'unknown', 'authorization_changed'
            elif not target.configured:
                state, code = 'unconfigured', 'target_not_configured'
            elif any(r['state'] != 'ready' for r in dependencies):
                state, code = 'blocked', 'prerequisite_not_ready'
            else:
                try:
                    state, code = target.probe()
                    if state not in {'ready', 'unavailable', 'unknown'} or not re.fullmatch(r'[a-z_]{1,64}', code):
                        raise ValueError('invalid_probe_result')
                except Exception:
                    state, code = 'unknown', 'probe_failed'
            row = {'target': name, 'layer': target.layer, 'state': state, 'code': code,
                   'observed_at': time.time(), 'depends_on': list(target.depends_on)}
            rows[name] = row
            return row
        check(args['target'])
        data = {'checks': list(rows.values()), 'registration_is_not_health': True,
                'scope': 'configured_host_targets_only', 'no_changes_performed': True}
        data['observation'] = self.observations.issue(data, tool=self.tool_type, context=context)
        return tool_result(self.tool_type, data)

    def agent_verify_result(self, *, receipt, context):
        return {'result_valid': bool(self.authorize(context)) and self.observations.verify(receipt, tool=self.tool_type, context=context)}
