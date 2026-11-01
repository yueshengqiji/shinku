"""独立记忆合成装配；不加载私用工厂、不读生产记忆、不访问模型供应商。"""
from contextlib import contextmanager
from dataclasses import replace

from shinku.agent.task_state import new_task
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.conversation import ConversationTurn
from shinku.agent_runtime.demo import demo_service
from shinku.agent_runtime.results import export_task_result
from shinku.angelos.host import ServiceHost
from shinku.angelos.store import WorkbenchStore


@contextmanager
def memory_fixture(root, *, mode='full', policy=None):
    service, file_tool, runtime = demo_service(root)
    original = service.authorize
    service.authorize = lambda *a: replace(original(*a), permission_mode=mode)
    host = ServiceHost(service, requester='owner', memory_root=root/'memory', memory_policy=policy)
    workbench = WorkbenchStore(root/'workbench.sqlite3')
    host.bind_workbench(workbench)
    c = workbench.conversation('default')
    try:
        yield service, host, workbench, c, file_tool, runtime
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def seed(fixture, key, text='请记住：我的合成客户端叫白鸽，来自本地包。', *, cid=None, status='running', calls=(), report=''):
    service, host, wb, conversation, _, _ = fixture
    cid = cid or conversation['id']
    scope = AgentScope('owner', 'angelos:'+cid)
    task = new_task(task_id=key, requester=scope.requester, session=scope.session,
        request=text, message_id=key+'-input', contract=None)
    task.update(input_mode='conversation', execution_attempt=1, status=status, tool_calls=list(calls))
    task = service.task_store.save(task)
    if status != 'running':
        task['completion'] = {'accepted': True, 'status': status, 'reason': 'synthetic_condition_failed',
                              'missing': ['old_command'], 'satisfied': ['install']}
        task['execution_result'] = export_task_result(task, report=report)
        task = service.task_store.save(task)
    workspace = wb.read('workspace', wb.read('conversation', cid)['workspaceId'])
    turn = {'id': key+'-input', 'text': text, 'stage': 'executing' if status == 'running' else 'finished',
            'task_id': key, 'workspace_snapshot': workspace, 'result': task.get('execution_result')}
    wb.change('conversation', cid, lambda c: c['turns'].append(turn))
    grant = service._grant(scope, task, 'inspect')
    ctx = execution_context(task, grant, attempt=1, work_id=key)
    ctx['_agent_conversation_turn'] = ConversationTurn(service, scope, key, 1, grant)
    return task, turn, ctx


def invoke(host, context, name='read_agent_memory', **args):
    return host.service.executor.tool_host.registry.get(name).execute(call=args, context=context)


def allow_memory_contract(service, task):
    task = service.task_store.get(task['task_id'], requester=task['requester'], session=task['session'])
    task['contract'] = {'goal': '保存用户陈述', 'deliverables': ['本库保存结果'], 'scope': ['只改Agent记忆'],
        'requirements': [{'id': 'memory', 'description': '保存当前原话', 'effect': 'memory_manage', 'targets': []}]}
    return service.task_store.save(task)
