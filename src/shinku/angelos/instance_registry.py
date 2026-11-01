"""可信工厂登记的实例目录；不从进程名、角色文字或模型输出猜服务身份。

登记只声明连接位置及配置来源，不等于现场验证。对象启动后不可改写；模型
只能提交实例ID。凭据不属于本配置；工厂可显式调用 from_file，默认不启用。
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from urllib.parse import urlsplit


def read_json(path, limit=4 * 1024 * 1024):
    """有界读取登记指定的普通文件，拒绝末端符号链接及设备；不输出异常正文。"""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('instance_file_not_regular')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('instance_file_too_large')
    return json.loads(raw)


@dataclass(frozen=True)
class RuntimeInstance:
    id: str
    display_name: str
    service: str
    endpoint: str
    adapter: str
    model_store: str = ''
    store_format: str = 'model_services'
    data_root: str = ''
    log_path: str = ''

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', self.id):
            raise ValueError('invalid_instance_id')
        if not isinstance(self.display_name, str) or not 1 <= len(self.display_name) <= 64 or any(ord(c) < 32 for c in self.display_name):
            raise ValueError('invalid_instance_name')
        if not isinstance(self.service, str) or not re.fullmatch(r'[a-zA-Z0-9_.@-]{1,128}\.service', self.service):
            raise ValueError('invalid_instance_service')
        # 只连接本机IPv4字面地址，无代理、重定向、URL凭据或自定义请求路径。
        u = urlsplit(self.endpoint)
        if (u.scheme != 'http' or u.hostname != '127.0.0.1' or u.username or u.password
                or not u.port or not 1024 <= u.port <= 65535 or u.path not in ('', '/') or u.query or u.fragment):
            raise ValueError('invalid_instance_endpoint')
        if self.adapter not in ('console', 'angelos') or self.store_format not in ('model_services', 'angelos_preferences', 'settings'):
            raise ValueError('invalid_instance_adapter')
        for value in (self.model_store, self.data_root, self.log_path):
            if not isinstance(value, str) or len(value) > 2048 or '\x00' in value or (value and not Path(value).is_absolute()):
                raise ValueError('invalid_instance_location')

    def identity(self):
        return {'instance_id': self.id, 'display_name': self.display_name,
                'service': self.service, 'endpoint': self.endpoint.rstrip('/'),
                'identity_source': 'host_registration'}


class InstanceRegistry:
    """只保存实例元数据；不持有Bot/Agent记忆，也不共享会话模型选择。"""
    def __init__(self, entries):
        entries = tuple(entries)
        if not 1 <= len(entries) <= 16 or not all(isinstance(e, RuntimeInstance) for e in entries):
            raise ValueError('invalid_instance_registry')
        if len({e.id for e in entries}) != len(entries) or len({e.endpoint.rstrip('/') for e in entries}) != len(entries):
            raise ValueError('duplicate_instance_registration')
        self.entries = MappingProxyType({e.id: e for e in entries})

    @classmethod
    def from_file(cls, path):
        """只有可信启动代码调用；拒绝其他用户拥有或组/全局可写的登记文件。"""
        path = Path(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError('instance_registry_permissions_invalid')
        value = read_json(path, 65536)
        if not isinstance(value, dict) or set(value) != {'schema_version', 'instances'} or value['schema_version'] != 1 or not isinstance(value['instances'], list):
            raise ValueError('invalid_instance_registry')
        try:
            return cls(RuntimeInstance(**row) for row in value['instances'])
        except TypeError as exc:
            raise ValueError('invalid_instance_registry') from exc

    def get(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.entries:
            raise ValueError('instance_not_registered')
        return self.entries[identifier]

    def catalog(self):
        return {'source': 'host_registration', 'is_live_state': False,
                'instances': [e.identity() for e in self.entries.values()]}

    def log_scope(self, unit):
        return {'registered_instances': [e.identity() for e in self.entries.values() if e.service == unit],
                'attribution': 'registered_service_only',
                'note': '日志按服务和时间归属；历史条目PID不等于当前PID，不能转作其他实例的故障证据。'}
