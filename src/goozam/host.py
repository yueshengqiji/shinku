"""Angelos2 独立组合根；运行时由实验调用方显式注入，不读取生产环境或工厂。

复用原服务、工作台、记忆、工具权限和规范结果。学习工具仅检索建议；提案与
评估由离线批次驱动，不注册模型可调用的晋级/改权限接口。
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_memory.service import checked_result
from shinku.agent_runtime import AgentExecutor, AgentService, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls, require_unprivileged
from shinku.angelos.task_continuity import workspace_binding, task_turn, turn_workspace
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.tools.workspace import build_workspace_tools

from shinku.plugins.contracts import PluginContext, PluginDefinition, PluginManifest
from shinku.plugins.manager import PluginManager
from .plugin import Controls, GoozmaPlugin
from .learning import LearningStore
from .sandbox import plain_tree


class TaskSources:
    """只回查当前实验宿主可见、未遗忘的同工作区规范结果，不连接其他库。"""
    def __init__(self, host):
        self.host = host

    def __call__(self, binding, ref):
        if not isinstance(ref, dict) or set(ref) != {'session', 'task_id'}:
            raise ValueError('goozam_invalid_reference')
        memory = self.host.memory
        task = memory.continuity.read_task(ref['session'], ref['task_id'])
        conversation, current = workspace_binding(memory.continuity.workbench, ref['session'])
        turn = task_turn(conversation, task)
        if (binding != current or turn is None or turn_workspace(turn) != binding
                or memory.store.source_forgotten(self.host.requester, binding, ref['session'], ref['task_id'])):
            raise ValueError('goozam_source_scope_or_revocation')
        result = checked_result(task)
        if result['execution']['status'] not in {'completed', 'partial', 'failed', 'blocked', 'canceled'}:
            raise ValueError('goozam_terminal_source_required')
        # 比较原任务的权限/工具上界与额度，不能把放宽权限或增加预算算作学习收益。
        grant = task.get('host_grant', {})
        conditions = [result['constraints'], self.host.model,
            {k: grant.get(k) for k in ('allowed_tools', 'permission_mode', 'policy')},
            {k: task.get('budget', {}).get(k) for k in
             ('max_requests', 'max_tokens', 'max_tools', 'max_active_seconds')}]
        # 原结果与其指纹保持兼容；提案材料另带可见原稿，既不改失败状态，也
        # 不把模型自述提升为工具证据。权限/工作区/遗忘校验必须先于原稿读取。
        drafts = self.host.reasoning_view.outputs(ref['task_id'])
        material = dict(result, model_drafts={
            'is_execution_evidence': False, 'grants_permissions': False,
            'records': [{k: row[k] for k in ('id', 'text', 'source', 'diagnostics') if k in row}
                        for row in drafts]})
        return {'scope': binding, 'fingerprint': canonical_hash(result),
                'input_hash': canonical_hash([r['text'] for r in task['requests']]),
                'conditions_hash': canonical_hash(conditions),
                # 原验证器也可接受如实报告的失败；学习晋级需要完成状态与验收同时成立。
                'accepted': result['execution']['status'] == 'completed'
                    and result['verification'].get('accepted') is True,
                'status': result['execution']['status'], 'result': result,
                'learning_material': material}


@contextmanager
def experiment(root, runtime, *, profile=None):
    """仅供受隔离启动器/测试宿主装配；调用此函数本身不创建OS沙箱。"""
    require_unprivileged()
    if profile is not None and (not isinstance(profile, ExecutionProfile) or profile.runtime is not runtime):
        raise ValueError('goozam_explicit_matching_profile_required')
    root = plain_tree(root)
    workspace = root / 'workspace'
    if not workspace.is_dir():
        raise ValueError('goozam_workspace_required')
    data = root / 'data' / 'angelos2'
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    requester = 'local-goozam-angelos2'
    def workspace_authorized(context, write):
        scope = context.get('agent_request_scope')
        return getattr(scope, 'requester', '') == requester and getattr(scope, 'session', '').startswith('angelos:')
    handlers = list(build_workspace_tools(workspace, authorize=workspace_authorized))
    registry = ToolRegistry({h.tool_type: h for h in handlers})
    policy = ExecutionPolicy(risk_by_tool={**{h.tool_type: 'low' for h in handlers}, 'read_goozam_experiences': 'low'})
    def authorize(scope, task, operation):
        if scope.requester != requester or not scope.session.startswith('angelos:'):
            return None
        return ExecutionGrant('goozam-offline-v1', frozenset([h.tool_type for h in handlers] + ['read_goozam_experiences']),
            policy=policy, permission_mode='readonly', resume_safe=False,
            # 未声明真实usage的离线Runtime按原执行器保守预留；容纳完整工具链，
            # 不是已批准的真实模型费用额度，网络仍由沙箱完全隔断。
            budget_limits={'max_requests': 12, 'max_tools': 8, 'max_tokens': 600000,
                           'max_active_seconds': 60})
    tasks = AgentTaskStore(data / 'tasks.sqlite3')
    coordinator = AgentTaskCoordinator(state_path=data / 'workers.sqlite3', max_workers=1)
    try:
        service = AgentService(executor=AgentExecutor(ToolHost(registry),
            profile or ExecutionProfile(runtime, profile_id='goozam-explicit-runtime')),
            authorize=authorize, task_store=tasks, coordinator=coordinator,
            material_store=TaskMaterialStore(tasks), recovery_enabled=False)
        host = ServiceHost(service, requester=requester, diagnostic_units=(),
            reasoning_path=data / 'reasoning.sqlite3', memory_root=data / 'memory',
            permissions=PermissionControls(data / 'permissions.json', default='he'))
        app = create_angelos_app(root / 'data', host=host)
        app.title = 'Angelos2 / goozam 离线实验'
        @app.middleware('http')
        async def experiment_label(request, call_next):
            response = await call_next(request)
            if request.url.path in {'/angelos', '/angelos/'} and response.status_code == 200:
                from starlette.responses import Response
                raw = b''.join([chunk async for chunk in response.body_iterator]).decode('utf-8')
                raw = raw.replace('<title>Angelos / 工作台</title>', '<title>Angelos2 / goozam 离线实验</title>')
                raw = raw.replace('class="wordmark">ANGELOS</span>', 'class="wordmark">ANGELOS2</span>')
                raw = raw.replace('<button id="settings-button">', '<a href="/angelos/goozma" style="color:#78b7ff;padding:6px;border:1px solid #666;text-decoration:none">Goozma 插件</a><button id="settings-button">')
                raw = raw.replace('AGENT<br>WORKSPACE', 'GOOZAM<br>离线实验')
                headers = {k: v for k, v in response.headers.items() if k != 'content-length'}
                response = Response(raw, headers=headers, media_type='text/html')
            response.headers['X-Goozam-Experiment'] = 'offline'
            return response
        learning = LearningStore(root / 'learning', TaskSources(host))
        controls = Controls(root / 'learning')
        plugin = GoozmaPlugin()
        manager = PluginManager(PluginContext(host_id='angelos2', instance_id=requester,
            state_root=root / 'learning', services={'goozam_learning': learning,
                'goozam_controls': controls, 'agent_memory': host.memory}),
            [PluginDefinition(PluginManifest('goozma', version='experimental-1', hosts=('angelos2',)),
                              factory=lambda: plugin)])
        manager.register_tools(registry)
        if manager.diagnostics()['plugins'][0]['status'] != 'active':
            raise RuntimeError('goozam_plugin_activation_failed')
        # 接在原权限收缩之后，只为首次提交附加可选建议资料；inspect/run/resume
        # 不重新检索，因此运行中开关不会改写任务输入，重试仍由原服务保证幂等。
        from .delivery import experience_materials
        original_authorize = service.authorize
        def with_experience(scope, task, operation):
            grant = original_authorize(scope, task, operation)
            if grant is None or operation != 'submit':
                return grant
            sources, gaps = experience_materials(host, plugin, scope, task, grant)
            return replace(grant, material_sources=(*grant.material_sources, *sources),
                           material_gaps=(*grant.material_gaps, *gaps))
        service.authorize = with_experience
        from .web import install
        install(app, controls)
        yield SimpleNamespace(app=app, host=host, learning=learning,
                              workbench=app.state.angelos, workspace=workspace,
                              plugin=plugin, controls=controls, plugins=manager)
    finally:
        if 'manager' in locals():
            manager.close()
        if 'app' in locals():
            app.state.angelos.close()
        coordinator.shutdown(wait=True)
        tasks.close()
