"""启动器显式加载的宿主连接配置；网页和模型均不能开启、改写或读取密钥。"""
import json
import os
from pathlib import Path
import stat

from shinku.agent_runtime import AgentBoundaryError, ExecutionGrant
from shinku.agent_runtime.delegation import DelegationGateway, DelegationPrincipal
from shinku.tools.execution import ExecutionPolicy


def protected_text(path):
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError('delegation_config_permissions_invalid')
    return path.read_text(encoding='utf-8')


def install_from_file(host, path):
    """只限制已有执行能力；来源文件根目录仍由现有材料宿主配置拥有。"""
    if host is None or getattr(host, 'delegation', None) is not None:
        raise ValueError('delegation_host_required_or_already_installed')
    path = Path(path).resolve()
    data = json.loads(protected_text(path))
    if not isinstance(data, dict) or set(data) != {'schema_version', 'hosts'} or data['schema_version'] != 1:
        raise ValueError('delegation_config_invalid')
    rows = data['hosts']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError('delegation_config_invalid')
    principals = {}
    registry = host.service.executor.tool_host.registry
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
                'id', 'token_file', 'requesters', 'sessions', 'allowed_tools', 'budget_limits'}:
            raise ValueError('delegation_config_invalid')
        name = row['id']
        if not isinstance(name, str) or not 1 <= len(name) <= 64 or name in principals:
            raise ValueError('delegation_config_invalid')
        for key in ('requesters', 'sessions', 'allowed_tools'):
            if not isinstance(row[key], list) or not row[key] or len(row[key]) > 1024 or any(
                    not isinstance(v, str) or not v or len(v) > 256 for v in row[key]):
                raise ValueError('delegation_config_invalid')
        if set(row['sessions']) - {'private:', 'group:', 'desktop:'}:
            raise ValueError('delegation_config_invalid')
        names = frozenset(row['allowed_tools'])
        if not names.issubset(set(registry.names())):
            raise ValueError('delegation_config_tool_unavailable')
        token_path = Path(row['token_file'])
        secret = protected_text(token_path if token_path.is_absolute() else path.parent/token_path).strip()
        grant = ExecutionGrant('delegation:' + name, names,
            policy=ExecutionPolicy(allowed=names | {'declare_task_contract','read_self_state','update_work_plan','read_work_context','task_material_read','request_context_compaction'}, approval_required_by_risk=frozenset()),
            budget_limits=row['budget_limits'], still_authorized=host.ready, resume_safe=True)
        grant.budget()
        requesters, sessions = frozenset(row['requesters']), tuple(row['sessions'])
        def authorize(scope, task, operation, *, grant=grant, requesters=requesters, sessions=sessions):
            if scope.requester not in requesters or not scope.session.startswith(sessions):
                raise AgentBoundaryError('delegation_requester_denied')
            return grant
        principals[name] = DelegationPrincipal(secret, authorize, sessions)
    controls = getattr(host, 'permissions', None)
    # 没有逐次审批宿主时不把来源宿主要求批准的调用转换为免确认。
    host.delegation = DelegationGateway(host.service, principals,
        file_delivery=getattr(host, 'file_delivery', None),
        grant_filter=(lambda grant: controls.restrict(grant, respect_host_approval=True)) if controls else None)
    return host.delegation
