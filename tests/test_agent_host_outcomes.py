"""宿主状态回归：历史失败措辞不裁决本轮结果，实际执行证据单独检查。"""
from dataclasses import replace

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.verification import SmartCompletionVerifier
from test_agent_conversation_loop import turn_env
from test_agent_reply_delivery import final, result_for


@pytest.mark.parametrize('outcome', ['', 'completed'])
@pytest.mark.parametrize('text', [
    '先前那次连接失败，记录仍然保留；这里说明我处理工作的方式。',
    '抱歉，我刚才理解偏了，现在直接回答你的观点。',
    '不能只凭一句话下结论，需要结合具体条件。',
    '没有找到唯一答案，并不意味着这个问题没有讨论价值。',
    'The previous connection failed; this is an explanation of my approach.',
])
def test_reply_wording_cannot_select_failure_status(text, outcome):
    # 故意不命中旧“解释/如何”问句白名单，不能靠扩充词表修复此故障。
    state = AgentState(transcript=[{'role': 'user', 'content': '换个角度聊聊你的工作方式。'}])
    verdict = SmartCompletionVerifier().evaluate(state, text, outcome=outcome)
    assert verdict.accepted and verdict.status == 'completed'
    assert verdict.reason != 'reported_failure'


@pytest.mark.parametrize('outcome', ['failed', 'blocked'])
def test_explicit_failure_does_not_need_keyword_detection(outcome):
    state = AgentState(transcript=[{'role': 'user', 'content': '创建 report.txt'}])
    verdict = SmartCompletionVerifier().evaluate(state, '当前停在这里。', outcome=outcome)
    assert verdict.accepted and verdict.status == outcome
    assert verdict.evidence[-1]['business_goal'] is False


@pytest.mark.parametrize('text', ['先前读取失败。', '已经写入 report.txt。'])
def test_missing_real_effect_is_rejected_regardless_of_wording(text):
    state = AgentState(context={'agent_completion_contract': {'requirements': [
        {'id': 'output', 'effect': 'write', 'targets': ['report.txt']},
    ]}})
    verdict = SmartCompletionVerifier().evaluate(state, text, outcome='completed')
    assert not verdict.accepted and 'output' in verdict.missing
    assert verdict.reason != 'reported_failure'


@pytest.mark.parametrize('reply_delivery', [False, True])
def test_service_stores_normal_reply_without_failure_from_its_text(tmp_path, reply_delivery):
    text = '先前的连接失败有记录；本轮只谈处理问题的方法，不能据此推断当前状态。'
    with turn_env(tmp_path, lambda *args: final(text), reply_delivery=reply_delivery) as (_, tool, _, port, calls):
        from shinku.agent_runtime import AgentTurnRequest
        from test_agent_runtime_boundary import settled
        h = port.submit(AgentTurnRequest('wording', '换个角度聊聊你的工作方式。'))
        view = settled(port, h.task_id)
        result = port.result(h.task_id)
        assert view['status'] == 'completed'
        assert result['execution_report']['text'] == text
        assert view['verification']['verified'] is False
        assert tool.calls == 0 and len(calls) == result['budget']['requests'] == 1


def test_written_file_and_untested_result_do_not_need_answer_reviewer(tmp_path, monkeypatch):
    """真实文件回执与未测试说明并存，引用审查器不得接管交付。"""
    from shinku.agent.planner import LLMPlanner
    from shinku.agent_runtime.stages import StagePolicy

    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary reply must not enter semantic review')

    for method in ('_review_research_candidates', '_review_research_answer', '_review_local_answer'):
        monkeypatch.setattr(LLMPlanner, method, forbidden)
    # 原任务写入后没有测试能力，不能因辅助测试缺失把正文销毁或假称全目标通过。
    text = '文件已写入。当前未运行测试；之前的连接失败不影响这次写入。'
    def respond(kw, count, service, tool):
        if count == 1:
            return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        assert count == 2
        return final(text, task_outcome='partial')

    with turn_env(tmp_path, respond, reply_delivery=True) as (service, tool, _, port, calls):
        service.executor.profile = replace(service.executor.profile,
            stage_policy=StagePolicy(lambda task: True))
        view, result = result_for(port, '创建合成文件，说明测试情况')
        assert view['status'] == 'partial'
        assert tool.path.read_text() == 'standalone agent verified'
        assert result['execution_report']['text'] == text
        assert result['verification']['evidence'][0]['receipt_valid'] is True
        assert result['verification']['layers']['answer_evidence'] == 'not_reviewed'
        assert len(calls) == result['budget']['requests'] == 2
        assert tool.calls == 1
