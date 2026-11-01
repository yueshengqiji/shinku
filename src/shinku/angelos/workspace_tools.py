"""把已有文件工具接到当前工作台任务，不让模型参数决定授权根目录。

工作台保存用户选定的目录及提交时快照；AgentService持有真实任务/授权。
本模块只在二者一致时分发到WorkspaceTool，预览按任务尝试隔离，所有读取、
写入与验收都会重新检查。四档权限仍由外层PermissionControls负责。
"""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.tools.workspace import WorkspaceAccess, WorkspaceTool, scope_key, tool_result


class WorkbenchTools:
    def __init__(self, service, requester):
        self.service, self.requester = service, requester
        self.store = None
        self.lock = RLock()
        self.adapters = OrderedDict()
        self.tools = []

    def bind(self, store, permissions):
        """工作台绑定后才注册；兼容无工作台宿主自行安装的固定目录适配器。"""
        with self.lock:
            if self.store is not None:
                if self.store.path != store.path:
                    raise AgentBoundaryError('workspace_store_changed')
                return
            from .permissions import ControlledTool
            registry = self.service.executor.tool_host.registry
            tools = [WorkspaceInfo(self), *(RoutedWorkspaceTool(self, op)
                for op in ('search', 'read_lines', 'preview_patch', 'apply_patch'))]
            # 不覆盖旧根目录/预览状态，也不给旧工具补回原宿主未授予的权限。
            self.tools = [tool for tool in tools if registry.get(tool.tool_type) is None]
            for tool in self.tools:
                registry.register(ControlledTool(tool, permissions) if permissions is not None else tool)
            # 注册与权限包装完成之后才允许授权，避免初始化中的裸工具被使用。
            self.store = store

    def binding(self, scope, task):
        """仅取工作台提交时快照；约束文字、模型路径或其他会话不能创建绑定。"""
        if self.store is None or scope.requester != self.requester or not scope.session.startswith('angelos:'):
            return None
        requests = task.get('requests') or []
        if not requests:
            return None
        try:
            with self.store.transaction() as db:
                c = self.store.get(db, 'conversation', scope.session[8:])
                w = self.store.get(db, 'workspace', c['workspaceId'])
            turn = next((t for t in c['turns'] if t['id'] == requests[0]['message_id']), None)
            snapshot = (turn or {}).get('workspace_snapshot', {})
            if (not turn or turn.get('task_id') != task.get('task_id')
                    or snapshot.get('id') != w['id'] or snapshot.get('path') != w['path']):
                return None
            path = w['path']
            if path and (not Path(path).is_dir() or str(Path(path).resolve()) != path):
                return None
            return {'id': w['id'], 'root': path}
        except (ValueError, OSError, KeyError):
            return None

    def current(self, context, name):
        """再次核验真实任务、尝试、消息绑定和最新授权；不相信旧工具上下文。"""
        requester, session, message = scope_key(context)
        old = context.get('agent_task_state', {})
        if requester != self.requester or not session.startswith('angelos:') or not message:
            raise AgentBoundaryError('workspace_not_authorized')
        rows = self.service.task_store.read_scoped_page(requester=requester, session=session,
            task_id=old.get('task_id', ''))['items']
        task = rows[0] if rows else None
        if (not task or task.get('execution_attempt') != old.get('execution_attempt')
                or task.get('cancel_requested') or task.get('status') == 'canceled'):
            raise AgentBoundaryError('workspace_task_changed')
        if tuple((r['message_id'], r['text']) for r in task['requests']) != context['agent_request_scope'].requests:
            raise AgentBoundaryError('workspace_task_changed')
        scope = AgentScope(requester, session)
        grant = self.service._grant(scope, task, 'inspect')
        if (name not in grant.allowed_tools or not grant.policy.accepts(name)
                or grant.authorization_ref != context.get('agent_authority_revision')):
            raise AgentBoundaryError('workspace_not_authorized')
        binding = self.binding(scope, task)
        if binding is None or binding != context.get('angelos_workspace_binding'):
            raise AgentBoundaryError('workspace_binding_changed')
        return task, binding

    def tool(self, context, operation):
        name = 'workspace_' + operation
        task, binding = self.current(context, name)
        if not binding['root']:
            raise AgentBoundaryError('workspace_directory_not_selected')
        key = (task['task_id'], task['execution_attempt'], binding['root'], context['agent_authority_revision'])
        with self.lock:
            if key not in self.adapters:
                def authorized(ctx, write):
                    self.current(ctx, 'workspace_apply_patch' if write else ctx['_angelos_workspace_operation'])
                    return True
                # 各操作共享同一任务的预览缓存，其他任务/尝试不能套用预览凭据。
                access = WorkspaceAccess(binding['root'], authorize=authorized)
                self.adapters[key] = {op: WorkspaceTool(access, op)
                    for op in ('search', 'read_lines', 'preview_patch', 'apply_patch')}
                while len(self.adapters) > 128:
                    self.adapters.popitem(last=False)
            self.adapters.move_to_end(key)
            return self.adapters[key][operation]


class RoutedWorkspaceTool:
    """元数据复用原工具；执行时按可信工作台绑定分发，绝不改全局工具根目录。"""
    def __init__(self, owner, operation):
        self.owner, self.operation = owner, operation
        self.template = WorkspaceTool(None, operation)

    def __getattr__(self, name):
        return getattr(self.template, name)

    def agent_workspace_root(self, context):
        return self.owner.current(context, self.tool_type)[1]['root']

    def execute(self, *, call, context):
        context = {**context, '_angelos_workspace_operation': self.tool_type}
        try:
            return self.owner.tool(context, self.operation).execute(call=call, context=context)
        except (AgentBoundaryError, OSError, ValueError):
            return tool_result(self.tool_type, {'code': 'workspace_unavailable',
                'message': '当前任务未绑定可用工作区或授权已变化。请核对所选目录，不能改用其他路径绕过限制。'},
                error='workspace_unavailable')

    def agent_verify_result(self, *, receipt, context):
        context = {**context, '_angelos_workspace_operation': self.tool_type}
        try:
            return self.owner.tool(context, self.operation).agent_verify_result(receipt=receipt, context=context)
        except (AgentBoundaryError, OSError, ValueError):
            return {'result_valid': False}


class WorkspaceInfo:
    tool_type = 'workspace_info'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_execution_evidence = False

    def __init__(self, owner):
        self.owner = owner

    def build_prompt_instruction(self):
        return ('读取当前会话选定的工作区目录及Angelos自身运行源码位置，不扫描磁盘。'
                '目录为空时没有工作区文件授权；运行源码位置仅作定位线索，不代表已读或其他服务的加载位置。')

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False})

    def normalize_call(self, call):
        return {} if isinstance(call, dict) and not {k for k in call if k != 'type' and not k.startswith('_tool_')} else None

    def execute(self, *, call, context):
        try:
            _, binding = self.owner.current(context, self.tool_type)
            return tool_result(self.tool_type, {'workspace': binding,
                'angelos_source_directory': str(Path(__file__).resolve().parents[1]),
                'process_working_directory': str(Path.cwd()), 'is_execution_evidence': False})
        except AgentBoundaryError:
            return tool_result(self.tool_type, {'code': 'workspace_unavailable'}, error='workspace_unavailable')


def install_workspace_tools(service, requester):
    """仅独立Angelos装配授权入口；工作台绑定才注册，原allow/deny与四档继续收紧。"""
    owner = WorkbenchTools(service, requester)
    original = service.authorize

    def authorize(scope, task, operation):
        grant = original(scope, task, operation)
        if grant is None or scope.requester != requester or not scope.session.startswith('angelos:'):
            return grant
        binding = owner.binding(scope, task)
        names = {t.tool_type for t in owner.tools if binding is not None
                 and (binding['root'] or t.tool_type == 'workspace_info') and grant.policy.accepts(t.tool_type)}
        if grant.permission_mode in {'read', 'readonly'}:
            names.discard('workspace_apply_patch')
        # 历史任务恢复仍沿用提交时能力上界；加载新版本不能给旧任务追加权限，
        # 否则既扩大旧授权，也会触发原恢复门的resume_permission_expansion。
        previous = task.get('host_grant')
        if isinstance(previous, dict):
            names.intersection_update(previous.get('allowed_tools', ()))
        risks = {t.tool_type: 'low' for t in owner.tools if t.tool_type != 'workspace_apply_patch'}
        policy = replace(grant.policy, risk_by_tool={**risks, **grant.policy.risk_by_tool})
        # 每次执行/恢复重读规则，放在单独宿主上下文，不变成必读附件或硬验收项。
        instructions = None
        if binding is not None:
            from .workspace_instructions import read_instructions
            instructions = read_instructions({'path': binding['root']})
        return replace(grant, allowed_tools=grant.allowed_tools | names, policy=policy,
            tool_context={**grant.tool_context, 'angelos_workspace_binding': binding,
                          'angelos_workspace_instructions': instructions,
                          'angelos_investigation_guidance': True})
    service.authorize = authorize
    return owner
