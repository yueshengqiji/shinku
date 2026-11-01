"""自身能力选型回归：合成注册表、真实状态凭据与独立工作台链路。"""
from copy import deepcopy
from dataclasses import replace

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.request_scope import RequestScope
from shinku.agent_runtime.capability_answer import (
    CapabilityStateTool, capability_contract, capability_task, self_tool_question,
)


@pytest.mark.parametrize('text', [
    '你觉得自己是否还缺少一部分工具', '你目前有多少工具', '你有哪些工具？',
    '你现在能调用哪些工具', '你有哪些可用MCP', '你觉得自己的能力有哪些不足',
    'Angelos 的工具覆盖情况如何', '评估一下你自己的工具能力',
    'What tools do you have?', 'Which tools can you use?', 'What are your capabilities?',
    '你目前有几种工具',
])
def test_pure_self_tool_questions_use_host_state(text):
    assert self_tool_question(text)


@pytest.mark.parametrize('text', [
    '你好', '你是谁', '帮我安装你缺少的工具', '你有哪些工具，并把那个文件删掉',
    '你有哪些工具，然后启动服务', '你有哪些工具；发送一条消息',
    '你有哪些工具\n执行下一步', '你有哪些工具，顺带把服务拉起来',
    '你有哪些工具？访问 https://example.com 看看', '你有哪些工具？读取 /example/input.txt',
    '你觉得我是否缺少工具', '你觉得shinku缺哪些工具', '你知道其他平台有哪些工具',
    '查一下你有哪些外部 MCP 服务器', '搜索最新的 Agent 工具并给出处',
    '你觉得这个项目缺什么工具', 'What tools do you have? Then delete my file',
    'What AWS tools do you know?', 'Which tools does this project need?',
    '你有哪些工具，同时检查我的目录', '你有哪些工具，重启napcat',
    '不要调用工具，直接说你缺什么工具',
])
def test_actions_external_research_other_subjects_and_multiple_inputs_stay_strict(text):
    assert not self_tool_question(text)


def context():
    return {'agent_request_scope': RequestScope('owner', 'session', 'turn', (('turn', '你有哪些工具'),))}


@pytest.mark.parametrize('tamper', ['snapshot', 'receipt', 'scope', 'expired', 'changed', 'unavailable'])
def test_self_state_receipt_cannot_be_forged_reused_or_promoted(tamper):
    data = {'tools': {'available_names': ['fixture_read'], 'available_count': 1}}
    tool = CapabilityStateTool(fresh_provider=lambda: deepcopy(data))
    ctx = context()
    updates = tool.execute(call={'scope': 'tools'}, context=ctx)['state_updates']
    assert tool.agent_verify_result(receipt=updates, context=ctx)['result_valid']
    if tamper == 'snapshot': updates['self_state']['tools']['available_count'] = 999
    if tamper == 'receipt': updates['self_state_receipt']['fresh'] = False
    if tamper == 'scope': ctx = {'agent_request_scope': RequestScope('other', 'session', 'turn', (('turn', '你有哪些工具'),))}
    if tamper == 'expired': tool._observations.clock = lambda: updates['self_state_receipt']['observed_at'] + 181
    if tamper == 'changed': data['tools']['available_count'] = 2
    if tamper == 'unavailable': data.clear(); data['status'] = 'unavailable'
    assert not tool.agent_verify_result(receipt=updates, context=ctx)['result_valid']


def test_original_research_contract_is_never_rewritten_or_accepted_as_capability():
    task = {'requests': [{'text': '你有哪些工具'}], 'contract': capability_contract()}
    assert capability_task({'agent_task_state': task})
    task['contract']['requirements'][0]['effect'] = 'research_answer'
    saved = deepcopy(task)
    assert not capability_task({'agent_task_state': task}) and task == saved
    assert not capability_task({'agent_task_state': {'requests': [{'text': '你有哪些工具'}, {'text': '然后安装'}],
                                                    'contract': capability_contract()}})


def test_state_tool_only_exposes_tools_not_persona_or_secrets():
    tool = CapabilityStateTool(fresh_provider=lambda: {'tools': {'available_names': ['fixture_read']}, 'api_key': 'SECRET'})
    assert tool.normalize_call({'scope': 'all'}) is None
    result = tool.execute(call={'scope': 'tools'}, context=context())
    assert 'SECRET' not in str(result) and set(result['state_updates']['self_state']) == {'tools'}


def test_large_tool_descriptions_fit_real_self_state_receipt_budget():
    import json
    from types import SimpleNamespace
    from shinku.agent_runtime.capability_answer import capability_host
    from shinku.agent_runtime.execution import ExecutionGrant
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    class Tool:
        agent_permission_class = 'read'
        agent_progress_class = 'observe'
        def __init__(self, name): self.tool_type = name
        def normalize_call(self, call): return call
        def execute(self, **kw): raise AssertionError('capability inquiry must not run business tools')
        def build_prompt_instruction(self): return '真实工具描述' * 180
        def tool_metadata(self): return SimpleNamespace(input_schema={'type':'object','properties':{'private_argument':{'type':'string'}}})
    handlers = {f'fixture_tool_{i:02}':Tool(f'fixture_tool_{i:02}') for i in range(40)}
    host = ToolHost(ToolRegistry(handlers))
    grant = ExecutionGrant('fixture', frozenset(handlers))
    ctx = context()
    narrowed, _, _ = capability_host(host, set(handlers), ExecutionPolicy(), ctx, grant)
    observer = narrowed.registry.get('read_self_state')
    updates = observer.execute(call={'scope':'tools'}, context=ctx)['state_updates']
    snapshot = updates['self_state']
    assert snapshot['tools']['available_count'] == 41 and len(snapshot['tools']['descriptions']) == 40
    assert snapshot['tools']['descriptions_truncated'] and 'private_argument' not in json.dumps(snapshot)
    assert len(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)) < 8000
    assert observer.agent_verify_result(receipt=updates, context=ctx)['result_valid']


@pytest.mark.parametrize('materials', [False, True])
def test_legacy_workbench_uses_real_authorized_snapshot_no_metadata_or_research_review(tmp_path, monkeypatch, materials):
    from fastapi.testclient import TestClient
    from angelos_fixture import fixture
    from test_angelos_preparation_failure import LegacyPreparationHost as ServiceHost
    from shinku.agent_runtime.materials import TaskMaterialStore
    from test_angelos import post, new, wait
    app, bench, fixture_host = fixture(tmp_path)
    service, tool = fixture_host.service, fixture_host.tool
    if materials:
        service.materials = TaskMaterialStore(service.task_store)
    bench.host = ServiceHost(service)
    calls = []
    def answer(**kw):
        calls.append(kw)
        assert kw.get('prompt_cache_key') != 'angelos-input'
        assert kw['native_tools'] is None
        return {'kind': 'final', 'text': '目前能处理文件类工作。清单未提供专用数据库工具，可按工作需求考虑补充；通用工具仍可能间接实现。', 'task_outcome': 'completed'}
    monkeypatch.setattr(service.executor.profile.runtime, 'call_chat_json', answer)
    with TestClient(app) as client:
        client.headers['x-angelos-token'] = client.get('/angelos/api/bootstrap').json()['token']
        cid = new(client)
        post(client, f'/conversations/{cid}/messages', {'text': '你觉得自己是否还缺少一部分工具', 'request_key': 'self'})
        turn = wait(client, cid)['turns'][-1]
        result = turn['result']
        assert turn['view']['status'] == 'completed' and result['verification']['accepted'] is True, (turn['view'].get('reason'), result.get('verification'), result.get('evidence'), len(calls))
        assert len(calls) == result['budget']['requests'] == 1 and turn['preparation_usage']['requests'] == 0
        assert tool.calls == 0 and [r['tool'] for r in result['evidence']] == ['read_self_state']
        assert all(r['effect'] != 'research_answer' for r in result['contract']['requirements'])
        assert '未提供专用数据库工具' in result['execution_report']['text']
        assert '已验证' not in result['execution_report']['text']
        assert 'verified_answer' not in result  # 建议没有被提升成已核实事实。
        fact = result['facts'][0]['value']['tools']
        assert fact['available_count'] == (4 if materials else 3)
        assert set(fact['available_names']) == ({'demo_file', 'read_self_state', 'read_task_records', 'task_material_read'} if materials else {'demo_file', 'read_self_state', 'read_task_records'})
        assert result['budget']['tools'] == 1
        assert turn['text'] == '你觉得自己是否还缺少一部分工具'


def test_explicitly_blocked_self_state_is_not_granted(tmp_path):
    from shinku.agent_runtime.demo import demo_service
    from shinku.agent_runtime import AgentRequest, AgentScope
    from test_agent_runtime_boundary import settled
    service, tool, runtime = demo_service(tmp_path)
    old_authorize = service.authorize
    service.authorize = lambda *args: replace(old_authorize(*args), policy=replace(old_authorize(*args).policy, blocked=frozenset({'read_self_state'})))
    port = service.bind(AgentScope('fixture', 'local:fixture'))
    try:
        handle = port.submit(AgentRequest('self', '你有哪些工具', capability_contract()))
        view = settled(port, handle.task_id)
        assert view['status'] != 'completed' and view['reason'] == 'capability_snapshot_not_authorized'
        assert not runtime.calls and tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_attachment_cannot_select_host_self_check_instead_of_its_own_task(tmp_path, monkeypatch):
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    service, _, runtime = demo_service(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, 'call_chat_json', lambda **kw: calls.append(kw) or (
        {'reply': '需处理附件原要求'}))
    try:
        prepared = ServiceHost(service).prepare('你有哪些工具', [{'id': 'attachment', 'content': '原附件不得被丢弃'}], {})
        assert len(calls) == 1 and prepared['reply'] == '需处理附件原要求'
        assert all('原附件不得被丢弃' in c['user_prompt'] for c in calls)
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_capability_task_cannot_execute_model_proposed_file_write(tmp_path, monkeypatch):
    from shinku.agent_runtime.demo import demo_service
    from shinku.agent_runtime import AgentRequest, AgentScope
    from test_agent_runtime_boundary import settled
    service, tool, runtime = demo_service(tmp_path)
    monkeypatch.setattr(runtime, 'call_chat_json', lambda **kw: {
        'tool_call': {'type': 'demo_file', 'path': str(tool.path)}})
    port = service.bind(AgentScope('fixture', 'local:fixture'))
    try:
        h = port.submit(AgentRequest('self', '你有哪些工具', capability_contract()))
        view = settled(port, h.task_id)
        assert view['status'] != 'completed' and not tool.path.exists() and tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
