"""实例现场只读适配器：服务→端口归属→运行接口与磁盘投影→再次核对进程。

不调用LLM，不改服务或配置。独立来源分别返回；缺失/冲突为信息缺口而非任务
失败。运行接口仅证明服务报告值，不证明反代最终上游，也不代表每个会话模型。
"""
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit

from .instance_registry import read_json


def identifier(value):
    # 模型/供应商只输出短标识，不透传接口里的任意正文或认证字段。
    return value if isinstance(value, str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/+@-]{0,191}', value) else None


def service_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '--property=MainPID',
        '--property=ActiveState', '--property=InvocationID', '--property=ControlGroup'], capture_output=True, text=True, timeout=3, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return {'pid': int(values.get('MainPID', 0)), 'state': identifier(values.get('ActiveState')),
            'generation': identifier(values.get('InvocationID')), 'cgroup': values.get('ControlGroup', ''), 'source': 'systemd_user'}


def service_pids(process, *, cgroup_root=Path('/sys/fs/cgroup')):
    """仅检查目标unit同cgroup进程，不靠父子进程名称扩大到其他实例。"""
    pid = process.get('pid')
    if type(pid) is not int or pid <= 0 or process.get('state') != 'active':
        return ()
    group = process.get('cgroup', '')
    if not group:
        return (pid,)
    if not isinstance(group, str) or not group.startswith('/') or '..' in Path(group).parts or group == '/':
        return ()
    try:
        with (cgroup_root / group.lstrip('/') / 'cgroup.procs').open() as stream:
            raw = stream.read(8193)
        pids = tuple(int(v) for v in raw.split())
        if len(raw) > 8192 or len(pids) > 256 or pid not in pids:
            return ()
        return pids
    except (OSError, ValueError):
        return ()


def owns_endpoint(process, endpoint):
    """用监听socket inode关联同unit主/子进程；归属无法核验就返回未知。"""
    sockets = set()
    for pid in service_pids(process):
        try:
            for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
                try:
                    target = fd.readlink().as_posix()
                except OSError:
                    continue
                if target.startswith('socket:['):
                    sockets.add(target[8:-1])
        except OSError:
            continue
    port = urlsplit(endpoint).port
    # 只允许已登记的127.0.0.1；通配监听可以接收该地址，远端连接不是监听。
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        columns = line.split()
        address, raw_port = columns[1].split(':')
        if columns[3] == '0A' and int(raw_port, 16) == port and address in ('0100007F', '00000000') and columns[9] in sockets:
            return True
    return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def runtime_state(instance):
    route = '/debug/hub/state' if instance.adapter == 'console' else '/angelos/api/models'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + 3
    raw = bytearray()
    with opener.open(instance.endpoint.rstrip('/') + route, timeout=3) as response:
        while True:
            if time.monotonic() > deadline:
                raise ValueError('instance_response_timeout')
            chunk = response.read1(65536)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError('instance_response_too_large')
    data = json.loads(raw)
    if instance.adapter == 'console':
        models = data['agent']['live_models']
        return {'chat_model': identifier(models.get('chat')), 'aux_model': identifier(models.get('aux')),
                'model_scope': 'service_runtime', 'source': 'running_service_api'}
    models = data['models']['selected']
    return {'chat_model': identifier(models.get('modelId')), 'provider_id': identifier(models.get('providerId')),
            'model_scope': 'instance_default_not_session_selection', 'source': 'running_service_api'}


def saved_state(instance):
    data = read_json(instance.model_store)
    if instance.store_format == 'settings':
        return {'chat_model': identifier(data.get('SHINKU_CHAT_MODEL_NAME')),
                'aux_model': identifier(data.get('SHINKU_AUX_MODEL_NAME')), 'source': 'saved_settings'}
    if instance.store_format == 'angelos_preferences':
        selected = data['selection']
        return {'chat_model': identifier(selected.get('modelId')), 'provider_id': identifier(selected.get('providerId')),
                'source': 'saved_instance_default'}
    provider = data['activeProviderId']
    providers = data['providers']
    selected = providers[provider] if isinstance(providers, dict) else next(p for p in providers if p['id'] == provider)
    return {'chat_model': identifier(selected.get('chatModel')), 'provider_id': identifier(provider),
            'source': 'saved_model_store'}


class InstanceProbe:
    """每次调用独立读取；快照不是常驻缓存，历史回执不冒充实时事实。"""
    def __init__(self, *, process_reader=service_state, endpoint_owner=owns_endpoint,
                 runtime_reader=runtime_state, saved_reader=saved_state, clock=time.time):
        self.process_reader, self.endpoint_owner = process_reader, endpoint_owner
        self.runtime_reader, self.saved_reader, self.clock = runtime_reader, saved_reader, clock

    @staticmethod
    def _read(reader, *args):
        try:
            value = reader(*args)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration, subprocess.SubprocessError):
            return None  # 异常原文可能含URL/配置秘密，不放入模型上下文。

    def inspect(self, instance, sections):
        result = {**instance.identity(), 'observed_at': self.clock(), 'gaps': [],
                  'scope_note': '本次观察不证明执行成功或消息送达；请求模型不证明代理最终上游。'}
        before = self._read(self.process_reader, instance.service)
        result['process'] = before
        bound = False
        try:
            bound = bool(before and self.endpoint_owner(before, instance.endpoint))
        except (OSError, ValueError, KeyError):
            pass
        if not bound:
            result['gaps'].append('endpoint_owner_unconfirmed')
        if 'models' in sections:
            # 无法确认端口属于目标进程时不读取邻居服务的模型来填空。
            running = self._read(self.runtime_reader, instance) if bound else None
            saved = self._read(self.saved_reader, instance) if instance.model_store else None
            result.update(running=running, saved=saved, consistency='unknown', comparison_fields=[])
            if not running or not running.get('chat_model'):
                result['gaps'].append('running_model_unavailable')
            if not saved or not saved.get('chat_model'):
                result['gaps'].append('saved_model_unavailable')
            if running and saved and running.get('chat_model') and saved.get('chat_model'):
                fields = ['chat_model'] + [k for k in ('provider_id', 'aux_model') if running.get(k) and saved.get(k)]
                result['comparison_fields'] = fields
                result['consistency'] = 'matched' if all(running[k] == saved[k] for k in fields) else 'different'
        if 'locations' in sections:
            result['locations'] = {'model_store': instance.model_store, 'data_root': instance.data_root,
                                   'log_path': instance.log_path, 'source': 'host_registration_not_loaded_path_verification'}
        after = self._read(self.process_reader, instance.service)
        stable = bool(before and after and before.get('generation') and before == after)
        bound_after = False
        try:
            bound_after = bool(after and self.endpoint_owner(after, instance.endpoint))
        except (OSError, ValueError, KeyError):
            pass
        result['endpoint_binding'] = 'confirmed' if bound and bound_after and stable else 'unconfirmed'
        if bound and not bound_after:
            result['gaps'].append('endpoint_owner_changed')
            if 'models' in sections:
                result.update(running=None, consistency='unknown', comparison_fields=[])
        result['endpoint_binding_scope'] = 'unit_main_or_same_cgroup_process_socket'
        if not stable:
            # 查询期间进程换代，保留时间与缺口，不将采样拼接为新进程的事实。
            result['gaps'].append('process_generation_unconfirmed_or_changed')
            if 'models' in sections:
                result.update(running=None, consistency='unknown', comparison_fields=[])
        return result
