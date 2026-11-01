"""直接交付正文中的资料/示例对象不是任务指令；通用结构化消费者保持原协议。"""
import json
import pytest
from test_contract_c2_5 import _make_runtime, FakeResponse
from shinku.llm.response_protocol import observe_response

FALLBACK = {'kind': 'final', 'text': '', 'task_outcome': 'failed', '_agent_reply_delivery': True}


def call(text, fallback=None, finish='stop'):
    response = FakeResponse(text)
    response.choices[0].finish_reason = finish
    runtime = _make_runtime(scripts=[('response', response)])
    with observe_response(allow_plain_reply=True, allow_final_text_json=True):
        return runtime.call_chat_json(system_prompt='s', user_prompt='u',
                                     fallback=FALLBACK if fallback is None else fallback)


@pytest.mark.parametrize('embedded', [
    {'fixture': True, 'loaded_version': 'old', 'source_version': 'new'},
    {'tool_call': {'type': 'workspace_apply_patch', 'patch': 'example'}},
    {'kind': 'final', 'text': '仅示例', 'task_outcome': 'completed'},
    {'ok': False, 'reason': 'workspace_scan_limit'},
])
def test_embedded_data_stays_complete_prose(embedded):
    text = '观察到版本不一致。\n```json\n' + json.dumps(embedded) + '\n```\n仅静态分析，尚未重启。'
    result = call(text)
    assert result == {'kind': 'final', 'text': text, 'task_outcome': 'partial'}
    assert 'tool_call' not in result


def test_structured_failure_and_other_consumers_unchanged():
    value = {'kind': 'final', 'text': '实际失败', 'task_outcome': 'failed'}
    assert call(json.dumps(value)) == value
    assert call('资料：{"fixture":true}', fallback={'decision':'invalid'}) == {'fixture': True}


@pytest.mark.parametrize('finish', ['length', 'content_filter'])
def test_incomplete_not_recovered(finish):
    assert call('未完的正文 {"fixture":true}', finish=finish) == FALLBACK


def test_real_native_call_precedes_prose_recovery():
    from types import SimpleNamespace
    from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD
    response = FakeResponse('正在读取：{"fixture":true}')
    response.choices[0].message.tool_calls = [SimpleNamespace(id='call-1',
        function=SimpleNamespace(name='workspace_info', arguments='{}'))]
    runtime = _make_runtime(scripts=[('response', response)])
    parsed = runtime.call_chat_json(system_prompt='s', user_prompt='u', fallback=FALLBACK,
        native_tools=[{'type':'function','function':{'name':'workspace_info','parameters':{}}}])
    assert parsed[NATIVE_TOOL_CALL_FIELD]['tool_calls'][0]['name']=='workspace_info'
    assert 'text' not in parsed


def test_diagnostic_prose_survives_real_service_delivery(tmp_path):
    """同一SDK/规划/交付链保留含错误对象的答复，不把局部证据判成任务决策。"""
    from dataclasses import replace
    from shinku.agent_runtime import AgentTurnRequest
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    from test_agent_conversation_loop import native_sdk
    from test_agent_runtime_boundary import settled
    text = '项目工作区读取成功，但原默认工作区仍失败：{"ok":false,"reason":"workspace_scan_limit"}。尚未修复。'
    service, tool, _ = demo_service(tmp_path)
    try:
        with native_sdk(lambda payload, count: ({'role': 'assistant', 'content': text}, 'stop')) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            port = ServiceHost(service, requester='owner', reply_delivery=True).port('one')
            handle = port.submit(AgentTurnRequest('diagnose', '解释为什么原工作区仍打不开文件。'))
            assert settled(port, handle.task_id)['status'] == 'partial'
            result = port.result(handle.task_id)
            assert result['execution_report']['text'] == text
            assert result['execution']['reason'] == 'reply_delivered'
            assert len(calls) == 1 and tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
