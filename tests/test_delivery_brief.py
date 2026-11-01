"""合成同轮交接：摘要可选，原稿/宿主状态/完整交付物不能被摘要覆盖。"""
import copy
from types import SimpleNamespace

import pytest

from shinku.agent.answer_protocol import retain_answer_draft
from shinku.agent.loop import AgentDecision
from shinku.agent_runtime.delivery_brief import summary_context
from shinku.agent_runtime.handoff import delivery_result
from shinku.agent_runtime.results import export_task_result, seal_result
from shinku.result_expression import ExpressionArchive, ExpressionService
from shinku.result_expression.natural import paragraph_blocks
from test_agent_delegation import remote, SCOPE, settled
from shinku.agent_runtime import AgentTurnRequest
from test_expression_persona_body import profile
from test_expression_semantic_coverage import semantic_sdk
from test_result_expression import host


REPORT = '检查了参数入口。\n' + '执行过程的完整说明。' * 150 + '\n已添加参数检查，测试未运行，部署未进行。'
BRIEF = '已添加参数检查；测试未运行，部署未进行，其他架构改进尚未完成。'


def sample(handoff=None):
    task = {'task_id': 'agent:synthetic', 'requester': SCOPE.requester, 'session': SCOPE.session,
        'requests': [{'text': '改进结构'}, {'text': '只改参数检查，先不部署'}],
        'confirmed_constraints': ['不得部署'], 'execution_attempt': 2, 'revision': 8,
        'status': 'partial', 'completion': {'accepted': False, 'missing': ['测试未运行']},
        'contract': {'requirements': [{'id': 'write', 'effect': 'write', 'target': '/fixture/a.py'}]},
        'progress': {}, 'material_information': {'required_status': 'complete', 'status': 'complete'}}
    state = SimpleNamespace(context={'agent_task_state': task}, progress=task['progress'],
                            local_answer=SimpleNamespace(enabled=True))
    value = {'kind': 'final', 'text': REPORT, 'task_outcome': 'partial'}
    if handoff is not None:
        value['handoff'] = handoff
    retain_answer_draft(state, value)
    return task, value


def result_with_brief():
    task, _ = sample({'mode': 'summary', 'text': BRIEF})
    return export_task_result(task, report=REPORT, reason='partial_completion')


def test_summary_is_same_turn_optional_and_canonical_report_is_preserved():
    original = result_with_brief(); before = copy.deepcopy(original)
    delivered = delivery_result(original)
    assert delivered['execution_report']['text'] == BRIEF
    assert original == before and original['execution_report']['text'] == REPORT
    assert delivered['presentation']['details_result_hash'] == original['result_hash']
    for key in ('original_requests', 'constraints', 'execution', 'remaining', 'uncertainties', 'artifacts'):
        assert delivered[key] == original[key]
    assert delivered['execution']['status'] == 'partial' and not delivered['verification']['accepted']
    blocks = paragraph_blocks(delivered, detail_policy_version=2, presentation_version=7)
    assert next(b for b in blocks if b['id'] == 'execution_body')['text'] == BRIEF


@pytest.mark.parametrize('handoff', [None, {}, 'broken', {'mode': 'summary'},
    {'mode': 'summary', 'text': ''}, {'mode': 'summary', 'text': 'a'*8193},
    {'mode': 'summary', 'text': BRIEF, 'permissions': 'ultra'}, {'mode': 'full', 'text': '导读'}])
def test_missing_invalid_or_full_brief_never_breaks_final_or_shortens_report(handoff):
    task, value = sample(handoff)
    assert AgentDecision.from_value(value).text == REPORT
    original = export_task_result(task, report=REPORT)
    assert delivery_result(original)['execution_report']['text'] == REPORT
    assert task['progress']['answer_drafts'][0]['text'] == REPORT


@pytest.mark.parametrize('mutation', ['request', 'attempt', 'scope', 'report'])
def test_stale_brief_cannot_follow_new_request_attempt_or_report(mutation):
    task, _ = sample({'mode': 'summary', 'text': BRIEF})
    report = REPORT
    if mutation == 'request': task['requests'].append({'text': '现在给完整原稿'})
    if mutation == 'attempt': task['execution_attempt'] += 1
    if mutation == 'scope': task['session'] += ':different'
    if mutation == 'report': report += '\n新结论'
    result = export_task_result(task, report=report)
    assert 'delivery_brief' not in result
    assert delivery_result(result)['execution_report']['text'] == report


@pytest.mark.parametrize('handoff', [None, {}, {'mode': 'full', 'text': '请看完整报告'}])
def test_new_final_with_same_report_does_not_reuse_previous_brief(handoff):
    task, _ = sample({'mode': 'summary', 'text': BRIEF})
    state = SimpleNamespace(context={'agent_task_state': task}, progress=task['progress'],
                            local_answer=SimpleNamespace(enabled=True))
    value = {'kind': 'final', 'text': REPORT, 'task_outcome': 'partial'}
    if handoff is not None:
        value['handoff'] = handoff
    retain_answer_draft(state, value)
    assert delivery_result(export_task_result(task, report=REPORT))['execution_report']['text'] == REPORT
    assert task['progress']['delivery_briefs'][0]['text'] == BRIEF  # 原摘要只保留，不复活。


@pytest.mark.parametrize('mutation', ['version', 'attempt', 'report', 'request', 'scope', 'failed', 'verified', 'answer_contract'])
def test_exported_summary_binding_and_full_answer_guard(mutation):
    result = result_with_brief(); result.pop('result_hash')
    if mutation == 'version': result['version'] += 1
    if mutation == 'attempt': result['attempt'] += 1
    if mutation == 'report': result['execution_report']['text'] += '新内容'
    if mutation == 'request': result['original_requests'].append({'text': '新要求'})
    if mutation == 'scope': result['scope']['session'] += ':new'
    if mutation == 'failed': result['execution']['status'] = 'failed'
    if mutation == 'verified': result['verified_answer'] = {'text': REPORT, 'validated': True}
    if mutation == 'answer_contract': result['contract']['requirements'].append({'id': 'answer', 'effect': 'local_answer'})
    result = seal_result(result)
    assert delivery_result(result)['execution_report']['text'] == result['execution_report']['text']


def test_summary_context_is_original_goal_corrections_host_state_and_original_report():
    original = result_with_brief()
    context = summary_context(original)
    assert context['original_requests'][-1]['text'] == '只改参数检查，先不部署'
    assert context['execution_report']['text'] == REPORT
    assert context['remaining'] == ['测试未运行'] and context['constraints'] == ['不得部署']
    assert 'delivery_brief' not in context and context['source_result_hash'] == original['result_hash']
    with pytest.raises(ValueError, match='requires_original'):
        summary_context(delivery_result(original))


def test_real_http_final_metadata_survives_loop_persistence_and_details(remote):
    service, tool, runtime, gateway, client, _ = remote
    old = runtime.call_chat_json
    def with_handoff(**kwargs):
        value = old(**kwargs)
        if value.get('kind') == 'final':
            value['handoff'] = {'mode': 'summary', 'text': '已创建并验证示例文件，尚未部署。'}
        return value
    runtime.call_chat_json = with_handoff
    handle = client.port(SCOPE).submit(AgentTurnRequest('brief', '创建示例文件并验证，不部署。'))
    result = settled(client, handle.task_id)
    assert result['execution']['status'] == 'completed', result
    calls = len(runtime.calls)
    delivered = client.port(SCOPE).handoff(handle.task_id)
    assert delivered['execution_report']['text'] == '已创建并验证示例文件，尚未部署。'
    assert client.port(SCOPE).details(delivered) == result
    assert len(runtime.calls) == calls and tool.calls == 1
    assert any('交接正文' in str(c) for c in runtime.calls)


def test_expression_rewrites_brief_and_preserves_host_limits_without_extra_summary_call(host):
    original = result_with_brief()
    source = delivery_result(original)
    def rewrite(payload):
        return {'schema_version': 2, 'binding_hash': payload['binding_hash'], 'paragraphs': [
            {'text': '这次的情况是：' + row['text'], 'fact_ids': [row['id']]}
            for row in payload['fact_blocks'] if row.get('required', True)]}
    # profile fixture scope与Agent fixture相同与否不作假设，用结果自己的scope。
    from shinku.agent_runtime import AgentScope
    scope = AgentScope(**source['scope'])
    with semantic_sdk(build=rewrite) as (model, checker, calls):
        expression = ExpressionService(ExpressionArchive(host), profile(model, checker, version=7))
        record = expression.prepare(source, scope=scope, check_current=lambda: '')
        assert record['status'] == 'ready', record['fallback_reason']
        assert BRIEF in record['text'] and REPORT not in record['text']
        assert '测试未运行' in record['text'] and '不得部署' in record['text']
        assert [c[0] for c in calls] == ['expression', 'check']
        assert all(REPORT not in str(c) for c in calls)
        assert record['canonical_result']['presentation']['details_result_hash'] == original['result_hash']
