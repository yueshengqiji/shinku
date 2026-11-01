"""讨论宾语、后续操作与真实执行验收的范围回归；仅使用合成输入。"""
from dataclasses import replace

import pytest

from shinku.agent.request_intent import execution_request_text
from shinku.agent.verification import SmartCompletionVerifier as Verifier
from shinku.agent.verification_rules import _WRITE_KEYWORDS, _SEND_KEYWORDS
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_request, DemoFileTool
from test_agent_conversation_loop import turn_env
from test_agent_intent_boundaries import state_for
from test_agent_runtime_boundary import settled


DISCUSSIONS = [
    '讨论修改验证流程',
    '请讨论修改验证流程',
    '帮我讨论修改验证流程',
    '探讨如何修改验收方式',
    '解释修改 config.json 的步骤',
    '请分析修改 settings.json 的风险',
    '我想知道如何保存 result.txt',
    '你觉得修改配置有哪些风险',
    '讨论删除 temp.txt 的风险',
    '讨论发送 report.txt 的流程',
    '讨论设置提醒的机制',
    '讨论停止进程的风险',
    '讨论“修改 a.txt，然后删除 b.txt”的含义',
    '讨论修改 a.txt。分析删除 b.txt 的风险。',
]


@pytest.mark.parametrize('text', DISCUSSIONS)
def test_discussion_operand_is_not_an_execution_requirement(text):
    assert not Verifier.request_requires_action(text)
    verdict = Verifier().evaluate(state_for(text), '这里只讨论工作方法。', outcome='completed')
    assert verdict.accepted and verdict.status == 'completed', verdict


@pytest.mark.parametrize('text', [
    '修改验证流程',
    '帮我修改 config.json',
    '请对 config.json 进行“修改”',
    '能不能帮我修改 config.json',
    '讨论方案，然后保存 result.txt',
    '讨论方案，请修改 config.json',
    '讨论方案并修改 config.json',
    '讨论方案并把 config.json 改为新配置',
    '讨论后对 config.json 进行修改',
    '讨论完直接修改 config.json',
    '讨论方案后请在 config.json 中修改设置',
    '分析项目并需要你修改 config.json',
    '讨论方案需要你保存 result.txt',
    '讨论方案我想修改 config.json',
    '分析修改方案；发送 report.txt',
    '讨论方案，删除 temp.txt',
    '讨论方案，同时设置提醒',
    '讨论方案，但停止进程',
    '先修改 config.json，再讨论验证流程',
    '不要修改 config.json，但讨论方案后保存 result.txt',
    '不能不修改 config.json，讨论如何验证',
])
def test_independent_operations_still_need_real_receipts(text):
    assert Verifier.request_requires_action(text)
    verdict = Verifier().evaluate(state_for(text), '这里只讨论工作方法。', outcome='completed')
    assert not verdict.accepted, verdict


@pytest.mark.parametrize('text,write,send', [
    ('讨论修改 a.txt，然后发送 b.txt', False, True),
    ('讨论发送 a.txt，然后修改 b.txt', True, False),
    ('分析怎么保存 a.txt，禁止发送 b.txt', False, False),
    ('修改 a.txt，讨论发送 b.txt 的方案', True, False),
    ('分析修改 a.txt 的方案并需要你发送 b.txt', False, True),
])
def test_effect_is_bound_to_its_own_clause(text, write, send):
    assert Verifier._user_requested_action(text, _WRITE_KEYWORDS) is write
    assert Verifier._user_requested_action(text, _SEND_KEYWORDS) is send


def test_projection_preserves_offsets_original_request_and_read_roles():
    text = '讨论修改 a.txt，然后保存 b.txt。读取 c.txt，不修改 d.txt。'
    projected = execution_request_text(text)
    assert len(projected) == len(text)
    assert projected.index('b.txt') == text.index('b.txt')
    assert 'a.txt' not in projected and 'd.txt' not in projected
    assert text == '讨论修改 a.txt，然后保存 b.txt。读取 c.txt，不修改 d.txt。'
    roles = Verifier._explicit_target_roles([text])
    assert 'write' not in roles.get('a.txt', set())
    assert roles['b.txt'] == {'write'} and roles['c.txt'] == {'read'}
    assert 'write' not in roles.get('d.txt', set())


@pytest.mark.parametrize('answer', ['已修改 a.txt。', '已发送 report.txt。', '已删除 temp.txt。'])
def test_success_claim_cannot_hide_inside_a_discussion_task(answer):
    verdict = Verifier().evaluate(state_for('讨论修改验证流程'), answer, outcome='completed')
    assert not verdict.accepted


def test_discussion_cannot_remove_an_existing_write_contract():
    state = state_for('讨论修改验证流程')
    state.context['agent_completion_contract'] = {'requirements': [
        {'id': 'saved', 'effect': 'write', 'targets': ['result.txt'], 'checks': {}}]}
    verdict = Verifier().evaluate(state, '这里只讨论工作方法。', outcome='completed')
    assert not verdict.accepted and 'saved' in verdict.missing


def test_success_claim_requires_its_own_target_even_after_real_unrelated_write(tmp_path):
    tool = DemoFileTool(tmp_path)
    result = tool.execute(call={'path': str(tool.path)}, context={})
    state = state_for('讨论修改 a.txt')
    state.tool_calls = [{'name': tool.tool_type, 'status': 'ok',
        'arguments': {'path': str(tool.path)}, 'data': result}]
    verifier = Verifier(tool_capabilities={tool.tool_type: {
        'effects': ['write'], 'target_arguments': ['path']}}, result_handlers={tool.tool_type: tool})
    verdict = verifier.evaluate(state, '已修改 a.txt。', outcome='completed')
    assert not verdict.accepted and tool.calls == 1
    assert tool.path.read_text() == 'standalone agent verified'


@pytest.mark.parametrize('text', ['讨论修改验证流程', '帮我讨论修改验证流程'])
def test_actual_conversation_finishes_once_without_business_execution(tmp_path, text):
    def respond(*args):
        return {'kind': 'final', 'text': '这里只讨论工作方法。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('discussion', text))
        view = settled(port, handle.task_id)
        result = port.result(handle.task_id)
        assert view['status'] == 'completed', result
        assert len(calls) == result['budget']['requests'] == 1
        assert result['budget']['tools'] == tool.calls == 0 and not tool.path.exists()
        assert result['contract'] is None and result['verification']['layers']['policy'] == 'conversation_reply'
        assert not result['verification']['evidence'][-1]['business_goal']
        saved = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
        assert saved['requests'][0]['text'] == text


@pytest.mark.parametrize('mode', ['missing', 'success', 'denied'])
def test_discussion_followed_by_operation_uses_original_contract_and_permission_gate(tmp_path, mode):
    def respond(kw, count, service, tool):
        if mode == 'missing':
            return {'kind': 'final', 'text': '这里只讨论工作方法。', 'task_outcome': 'completed'}
        if count == 1:
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        if count == 2:
            return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        return {'kind': 'final', 'text': '合成文件已创建并核验。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        if mode == 'denied':
            original = service.authorize
            service.authorize = lambda *args: replace(original(*args), policy=replace(
                original(*args).policy, blocked=frozenset({'demo_file'})))
        text = '讨论修改验证流程，然后创建 ' + str(tool.path)
        handle = port.submit(AgentTurnRequest('mixed', text))
        view = settled(port, handle.task_id)
        result = port.result(handle.task_id)
        if mode == 'success':
            assert view['status'] == 'completed' and result['verification']['satisfied'] == ['file'], result
            assert tool.calls == 1 and tool.path.read_text() == 'standalone agent verified'
            assert len(calls) == 3
        else:
            assert view['status'] != 'completed', result
            assert tool.calls == 0 and not tool.path.exists()
