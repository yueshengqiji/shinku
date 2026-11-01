"""A plain repair uses actual SDK output and independent current file receipts."""
from dataclasses import replace
from contextlib import contextmanager
import json

import httpx
import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.demo import demo_request, demo_service
from test_agent_runtime_boundary import settled
from test_task_call_failure_routing import actual_responses, close


@contextmanager
def execution_sdk(answers):
    with actual_responses(answers) as (runtime, calls):
        # FixtureRuntime skips ordinary runtime construction. The actual mock
        # SDK below supports native calls; give the host this fixture capability.
        runtime.chat_supports_native_tools = lambda: True
        yield runtime, calls


@pytest.mark.parametrize('original_outcome,expected', [('completed', 'completed'),
    ('partial', 'partial'), ('failed', 'failed'), ('blocked', 'blocked')])
def test_plain_final_repair_is_verified_without_inheriting_transport_failure(tmp_path, original_outcome, expected):
    service, tool, _ = demo_service(tmp_path)
    request = demo_request(tool)
    report = '目标文件已写入并通过核验，内容是 standalone agent verified。未部署、未发送。'
    with execution_sdk([{'name': tool.tool_type, 'arguments': {'path': str(tool.path)}},
            json.dumps({'task_outcome': original_outcome}), report]) as (runtime, calls):
        service.executor.profile = replace(service.executor.profile, runtime=runtime, max_rounds=2,
                                           runtime_meters_budget=True)
        client = service.bind(AgentScope('fixture', 'local:fixture'))
        try:
            handle = client.submit(request)
            view = settled(client, handle.task_id)
            result = client.result(handle.task_id)
            assert view['status'] == expected
            assert result['execution']['status'] == expected
            assert result['execution_report']['text'] == report
            assert tool.path.read_text(encoding='utf8') == 'standalone agent verified'
            assert tool.calls == 1 and len(calls) == 3
            assert not calls[2].get('tools') and not calls[2].get('tool_choice')
            assert result['budget']['requests'] == 3
            if expected == 'completed':
                assert result['verification']['accepted'] is True
                assert result['remaining'] == []
            else:
                assert result['verification']['status'] == expected
            before = tool.calls, len(calls)
            assert client.result(handle.task_id) == result
            assert client.submit(request).task_id == handle.task_id
            assert before == (tool.calls, len(calls))
        finally:
            close(service)


def test_plain_repaired_success_without_any_effect_cannot_complete(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    with execution_sdk([json.dumps({'task_outcome': 'completed'}),
            '目标文件已写入并通过核验。',
            json.dumps({'kind':'final','text':'未执行写入，本次任务没有完成。','task_outcome':'failed'})]) as (runtime, calls):
        service.executor.profile = replace(service.executor.profile, runtime=runtime, max_rounds=1,
                                           runtime_meters_budget=True)
        client = service.bind(AgentScope('fixture', 'local:fixture'))
        try:
            handle = client.submit(demo_request(tool))
            view = settled(client, handle.task_id)
            assert view['status'] != 'completed'
            assert client.result(handle.task_id)['verification']['status'] != 'completed'
            assert not tool.path.exists() and tool.calls == 0 and len(calls) == 3
            assert not calls[1].get('tools')
        finally:
            close(service)


@pytest.mark.parametrize('fault', [500, httpx.ReadTimeout('fixture-timeout'),
    ('unfinished response', 'length'), ('blocked response', 'content_filter')])
def test_failed_repair_cannot_upgrade_even_a_verified_prior_write(tmp_path, fault):
    service, tool, _ = demo_service(tmp_path)
    with execution_sdk([{'name': tool.tool_type, 'arguments': {'path': str(tool.path)}},
            json.dumps({'task_outcome': 'completed'}), fault]) as (runtime, calls):
        service.executor.profile = replace(service.executor.profile, runtime=runtime, max_rounds=2,
                                           runtime_meters_budget=True)
        client = service.bind(AgentScope('fixture', 'local:fixture'))
        try:
            handle = client.submit(demo_request(tool))
            view = settled(client, handle.task_id)
            assert view['status'] != 'completed'
            assert client.result(handle.task_id)['execution']['status'] != 'completed'
            assert tool.path.read_text(encoding='utf8') == 'standalone agent verified'
            assert tool.calls == 1 and len(calls) == 3
            assert not calls[2].get('tools')
        finally:
            close(service)


def test_repaired_plain_report_cannot_use_a_stale_file_receipt(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    with execution_sdk([{'name':tool.tool_type,'arguments':{'path':str(tool.path)}},
            json.dumps({'task_outcome':'completed'}), '目标文件已写入并通过核验。',
            json.dumps({'kind':'final','text':'当前文件内容与原要求不符。','task_outcome':'failed'})]) as (runtime, calls):
        actual = runtime.call_chat_json
        def changed_before_repair(**kwargs):
            if len(calls)==2:
                tool.path.write_text('changed outside the recorded invocation',encoding='utf8')
            return actual(**kwargs)
        runtime.call_chat_json = changed_before_repair
        service.executor.profile = replace(service.executor.profile,runtime=runtime,max_rounds=2,
                                           runtime_meters_budget=True)
        client = service.bind(AgentScope('fixture','local:fixture'))
        try:
            handle = client.submit(demo_request(tool))
            view = settled(client,handle.task_id)
            assert view['status'] != 'completed'
            result = client.result(handle.task_id)
            assert result['execution']['status'] != 'completed'
            assert 'file' in result['verification']['missing']
            assert tool.calls == 1 and len(calls) == 4
            assert tool.path.read_text(encoding='utf8')=='changed outside the recorded invocation'
            assert not calls[2].get('tools')
        finally:
            close(service)
