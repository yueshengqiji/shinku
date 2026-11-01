"""诊断任务同一循环的来源、分层交付与传输失败回归；模型均为确定性替身。"""
import json
from types import SimpleNamespace

import pytest

from shinku.agent.analysis_acceptance import ordinary_analysis
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls
from test_agent_runtime_boundary import settled
from test_analysis_acceptance_layers import case, REPORT
from test_service_log_evidence import contract


@pytest.mark.parametrize('user_text', [
    '查一下示例服务最近一次回复失败的原因', '排查服务报错原因', '定位最近超时的原因',
    '核实之前失败的具体原因', '为什么没有回复，查一下原因', 'Diagnose the latest service failure',
])
def test_readonly_diagnosis_uses_analysis_delivery(case, user_text):
    state, gate, _, _, task = case
    task['requests'][0]['text'] = user_text
    state.context = execution_context(task, SimpleNamespace(tool_context={}, permission_mode='read',
                                     authorization_ref='fixture-grant'), attempt=1)
    assert ordinary_analysis(state)
    verdict = gate.evaluate(state, REPORT, outcome='completed')
    assert verdict.accepted and verdict.layers['answer_evidence'] == 'unverified'


@pytest.mark.parametrize('user_text', [
    '查一下失败原因并修改配置', '排查服务故障，然后重启服务', '排查失败原因并运行测试',
    '精确统计失败次数', '排查原因，每个结论逐条提供证据',
])
def test_actions_and_precise_verification_remain_strict(case, user_text):
    state, gate, _, _, task = case
    task['requests'][0]['text'] = user_text
    state.context = execution_context(task, SimpleNamespace(tool_context={}, permission_mode='read',
                                     authorization_ref='fixture-grant'), attempt=1)
    assert not ordinary_analysis(state)
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted


@pytest.mark.parametrize('mode', ['ordinary', 'strict_timeout', 'strict_connection', 'strict_verified'])
def test_log_evidence_and_failure_draft_through_real_service(tmp_path, mode):
    service, file_tool, base = demo_service(tmp_path)
    calls, queries = [], []
    answer = ('失败原因：上游容量不足。' if mode == 'strict_verified' else
              '失败原因：服务日志显示上游容量不足。建议稍后再试；未修改配置或重启。')
    def runtime(**kw):
        calls.append(kw)
        if len(calls) == 1:
            # 提前建立分析条件，实际成稿会进入质量审查；后声明条件的恢复另有覆盖。
            return {'tool_call': {'type': 'read_service_logs', 'unit': 'fixture.service', 'since': 100, 'until': 200},
                    'task_contract': contract()}
        if len(calls) == 2:
            return {'kind': 'final', 'text': answer, 'task_outcome': 'completed', 'task_contract': contract()}
        if len(calls) == 3:
            payload = json.loads(kw['user_prompt'])
            assert len(payload['read_sources']) == 1
            assert '上游容量不足' in payload['read_sources'][0]['text']
            if mode == 'strict_verified':
                return {'verdict': 'supported', 'claims': [{'answer_fragment': answer,
                    'evidence': [{'source_id': payload['read_sources'][0]['id'], 'quote': '上游容量不足'}]}],
                    'coverage': [{'field': '失败原因', 'supported': True, 'claim_indices': [0]}],
                    'missing_fields': []}
            return {'verdict': 'unsupported', 'claims': [], 'coverage': [], 'missing_fields': ['失败原因']}
        from shinku.llm.response_protocol import note_response, note_call_failure
        failure = ConnectionError('synthetic_connection_error') if mode == 'strict_connection' else TimeoutError('synthetic_write_timeout')
        note_response('transport_error'); note_call_failure('service_call', failure)
        return {'kind': 'final', 'text': '暂时无法完成这一步。', 'task_outcome': 'failed'}
    base.call_chat_json = runtime
    host = ServiceHost(service, requester='owner', reasoning_path=tmp_path/'reasoning.sqlite3',
                       diagnostic_units=('fixture.service',))
    if host.service_logs is None:
        service.coordinator.shutdown(wait=True);service.task_store.close()
        pytest.skip('Linux user journal adapter')
    def reader(query):
        queries.append(query)
        return {'entries': [{'MESSAGE': '上游容量不足'}], 'more_in_window': False}
    host.service_logs.reader = reader
    try:
        port = host.port('one')
        handle = port.submit(AgentTurnRequest('diagnose', '查一下最近失败的原因' +
            ('，每个结论逐条提供证据' if mode != 'ordinary' else '')))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert file_tool.calls == 0 and len(queries) == 1
        if mode in {'strict_timeout', 'strict_connection'}:
            reason = 'conversation_call_failed' if mode == 'strict_connection' else 'conversation_call_timeout'
            assert view['status'] == 'failed' and result['execution']['reason'] == reason, result
            assert not result['verification']['accepted']
            assert result['answer_draft']['text'] == answer and not result['answer_draft']['validated']
            assert '未核实的模型原稿' in result['execution_report']['text']
            assert reason in result['execution_report']['text'] and 'verified_answer' not in result
            assert len(calls) == 4
        elif mode == 'strict_verified':
            assert view['status'] == 'completed' and result['verification']['accepted'], result
            assert result['verification']['satisfied'] == ['answer']
            assert result['execution_report']['text'] == answer and len(calls) == 3
            saved = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
            assert saved['progress']['local_answer_summary']['verdict'] == 'supported'
            assert saved['progress']['local_answer_summary']['bound_claims'] == 1
        else:
            assert view['status'] == 'completed', result
            assert result['verification']['layers']['answer_evidence'] == 'unverified'
            assert result['execution_report']['text'] == answer and len(calls) == 2
            assert result['verification']['layers']['quality_review'] == 'not_requested'
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


@pytest.mark.parametrize('level,allowed', [('zayin', False), ('he', True), ('waw', True), ('aleph', True)])
def test_log_read_obeys_all_four_permission_levels(tmp_path, level, allowed):
    from shinku.agent_runtime import AgentScope
    from shinku.agent.task_state import new_task
    service, _, _ = demo_service(tmp_path)
    controls = PermissionControls(tmp_path/'permission.json', default=level)
    host = ServiceHost(service, requester='owner', reasoning_path=tmp_path/'reasoning.sqlite3',
                       permissions=controls, diagnostic_units=('fixture.service',))
    try:
        if host.service_logs is None: pytest.skip('Linux user journal adapter')
        task = new_task(task_id='current', requester='owner', session='angelos:one',
                        request='查一下失败原因', message_id='message', contract=contract())
        task.update(status='running', execution_attempt=1);service.task_store.save(task)
        grant = service._grant(AgentScope('owner', 'angelos:one'), task, 'inspect')
        assert ('read_service_logs' in grant.allowed_tools) is allowed
        ctx = execution_context(task, grant, attempt=1)
        queries = []
        host.service_logs.reader = lambda q: (queries.append(q) or {'entries': [], 'more_in_window': False})
        handler = service.executor.tool_host.registry.get('read_service_logs')
        result = handler.execute(call={'unit': 'fixture.service', 'since': 100, 'until': 200}, context=ctx)
        assert bool(queries) is allowed
        assert (result.get('status') != 'blocked') is allowed
        assert not controls.pending
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_negative_review_is_not_misreported_as_broken_json(case):
    state, *_ = case
    state.local_answer.apply_answer_review(REPORT, {'verdict': 'unsupported', 'claims': [], 'coverage': [],
        'missing_fields': ['项目结构']}, state.context)
    check = state.local_answer.answer_check
    assert check['binding_error'] == 'answer_not_supported' and not check['valid']
    assert check['missing_fields'] == ['项目结构']
