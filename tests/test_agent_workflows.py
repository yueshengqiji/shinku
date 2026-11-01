"""工作参考由同一模型按需选择；不分类、不发请求、不替代权限与执行回执。"""
from dataclasses import replace
import json

import pytest

from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.agent_runtime.workflows import GUIDES
from shinku.angelos.host import ServiceHost
from test_agent_conversation_loop import turn_env, native_sdk, observed_contract
from test_agent_runtime_boundary import settled
from test_agent_work_planning import saved_task


def read_guide(name):
    return {'tool_call': {'type': 'read_work_context', 'view': 'workflow', 'workflow': name}}


@pytest.mark.parametrize('name', list(GUIDES))
def test_reference_is_optional_currently_scoped_and_not_business_success(tmp_path, name):
    def respond(kw, count, service, tool):
        if count == 1:
            assert GUIDES[name]['steps'][0] not in kw['system_prompt'] + kw['user_prompt']
            return read_guide(name)
        assert GUIDES[name]['steps'][0] in kw['user_prompt']
        assert 'current_tools' in kw['user_prompt'] and 'adaptable_reference' in kw['user_prompt']
        return {'kind': 'final', 'text': '按当前情况参考这些步骤，具体执行还需当前授权与实际证据。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest(name, '讨论适合当前工作的处理方式'))
        view = settled(port, h.task_id); task = saved_task(service, h.task_id)
        assert view['status'] == 'completed' and not view['verification']['verified']
        assert len(calls) == task['budget']['requests'] == 2 and task['budget']['tools'] == 1
        assert task['progress']['stalled'] == 1 and task.get('work_state') is None and tool.calls == 0
        receipt = task['tool_calls'][0]['data']['state_updates']
        assert receipt['workflow'] == name and receipt['is_execution_evidence'] is False
        assert 'guide' not in receipt


def test_tools_and_concrete_missing_groups_follow_current_grant(tmp_path):
    observed = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        turn = context['_agent_conversation_turn']
        guide = turn.work_context.execute(call={'view': 'workflow', 'workflow': 'modify_validate'}, context=context)
        observed.append(guide['state_updates'])
        return {'kind': 'final', 'text': '这里只讨论工作方法。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        original = service.authorize
        service.authorize = lambda *a: replace(original(*a), allowed_tools=frozenset())
        h = port.submit(AgentTurnRequest('missing', '仅解释工具清单，不执行操作。'))
        assert settled(port, h.task_id)['status'] == 'completed'
        data = observed[0]
        assert not data['current_tools']['write'] and not data['current_tools']['verify']
        assert data['missing_preferred_groups'] == ['write', 'verify']
        assert '通用工具' in data['capability_boundary'] and tool.calls == 0


def test_reference_cannot_replace_contract_or_real_write_receipt(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return read_guide('modify_validate')
        if count == 2:
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': observed_contract('read_work_context')}}
        if count == 3:
            assert 'internal_plan_is_not_execution_evidence' in kw['user_prompt']
            return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}, 'task_contract': demo_request(tool).contract}
        return {'kind': 'final', 'text': '合成文件已创建并核验。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('real-result', '创建 ' + str(tool.path)))
        assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
        assert tool.calls == 1 and port.result(h.task_id)['verification']['satisfied'] == ['file']
        assert len(calls) == 4


@pytest.mark.parametrize('boundary', ['policy', 'revoke', 'bad_arguments'])
def test_reference_obeys_policy_and_rejects_model_authority_fields(tmp_path, boundary):
    active = [True]
    def respond(kw, count, service, tool):
        if count == 1:
            if boundary == 'revoke': active[0] = False
            value = read_guide('diagnosis')
            if boundary == 'bad_arguments': value['tool_call']['permissions'] = 'full'
            return value
        return {'kind': 'wait', 'text': '当前没有取得可用参考。'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        original = service.authorize
        def grant(*args):
            base = original(*args)
            if boundary == 'policy': return replace(base, policy=replace(base.policy, blocked=frozenset({'read_work_context'})))
            return replace(base, still_authorized=lambda: active[0])
        service.authorize = grant
        h = port.submit(AgentTurnRequest(boundary, '讨论排查方法'))
        view = settled(port, h.task_id); task = saved_task(service, h.task_id)
        assert view['status'] != 'completed' and not task.get('work_state') and tool.calls == 0
        assert all('guide_sha256' not in r.get('data', {}).get('state_updates', {}) for r in task['tool_calls'])


def test_real_native_sdk_reads_only_selected_guide_without_extra_planner(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    def respond(payload, count):
        if count == 1:
            assert GUIDES['diagnosis']['steps'][0] not in json.dumps(payload, ensure_ascii=False)
            return {'role': 'assistant', 'content': '', 'reasoning_content': 'SYNTHETIC_REASONING', 'tool_calls': [
                {'id': 'guide_1', 'type': 'function', 'function': {'name': 'read_work_context',
                    'arguments': json.dumps({'view': 'workflow', 'workflow': 'diagnosis'})}}]}, 'tool_calls'
        rows = [m for m in payload['messages'] if m['role'] == 'tool']
        assert len(rows) == 1 and rows[0]['tool_call_id'] == 'guide_1'
        assert GUIDES['diagnosis']['steps'][0] in rows[0]['content']
        assert GUIDES['project_analysis']['steps'][0] not in rows[0]['content']
        return {'role': 'assistant', 'content': json.dumps({'kind': 'final', 'text': '先定位原失败记录，再验证关键假设。',
            'task_outcome': 'completed'})}, 'stop'
    try:
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
            port = ServiceHost(service, requester='owner').port('one')
            h = port.submit(AgentTurnRequest('native-guide', '讨论故障诊断方法'))
            view = settled(port, h.task_id)
            assert view['status'] == 'completed', port.result(h.task_id)
            assert len(calls) == view['budget']['requests'] == 2 and view['budget']['reported_tokens'] == 32
            assert tool.calls == 0 and not view['verification']['verified']
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_effectful_tool_is_never_suggested_as_readonly_even_with_read_label():
    from types import SimpleNamespace
    from shinku.tools.execution import ExecutionPolicy
    from shinku.agent_runtime.workflows import workflow_reference
    mixed = SimpleNamespace(agent_permission_class='read', agent_completion_effects=('write',))
    grant = SimpleNamespace(allowed_tools={'mixed'}, policy=ExecutionPolicy())
    guide = workflow_reference('diagnosis', {'mixed': mixed}, grant)
    assert not guide['current_tools']['observe'] and not guide['current_tools']['read']
    assert guide['current_tools']['write'] == ['mixed'] and guide['grants_permissions'] is False
