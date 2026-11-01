"""Concrete SDK failures retain safe first causes, counts, and no extra attempt."""
from contextlib import contextmanager
import json
import threading
import time

import httpx
from openai import OpenAI
import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.llm.runtime_types import ModelBundle
from shinku.llm.response_protocol import call_failure_metadata
from test_expression_runtime import FixtureRuntime
from test_result_expression import Model, SCOPE, canonical, host, prepare, service


@contextmanager
def sdk_expression(*, error=None, release=None):
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert not payload.get('tools') and not payload.get('tool_choice')
        if release is not None:
            assert release.wait(2)
        if error:
            raise error
        data = json.loads(payload['messages'][-1]['content'])
        candidate = dict(binding_hash=data['binding_hash'], layout=[
            {'fact': row['id']} for row in data['fact_blocks']])
        return httpx.Response(200, json=dict(id='offline-diagnostic', object='chat.completion',
            created=0, model='fixture', choices=[dict(index=0, finish_reason='stop',
                message=dict(role='assistant', content=json.dumps(candidate)))],
            usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16)))

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        with OpenAI(api_key='PRIVATE_DIAGNOSTIC_CREDENTIAL', base_url='https://offline.example.invalid/v1',
                    max_retries=0, http_client=http) as client:
            runtime = FixtureRuntime(client.chat.completions.create)
            runtime.chat = ModelBundle(client, 'fixture')
            runtime.aux = runtime.chat
            yield runtime, calls


@pytest.mark.parametrize('phase,method', [
    ('cache', '_record_cache'), ('response', '_read_content_text'),
    ('context_usage', '_record_context_budget_usage'),
])
def test_post_response_failure_keeps_actual_usage_and_safe_stage(host, monkeypatch, phase, method):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    with sdk_expression() as (runtime, calls):
        def fail(*args, **kwargs):
            raise PermissionError('PRIVATE_DIAGNOSTIC_BODY /secret/file token=secret')
        monkeypatch.setattr(runtime, method, fail)
        svc = service(host, RuntimeExpressionAdapter(runtime))
        record = prepare(svc)
        assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_model_error'
        channel = record['usage']['expression']
        assert channel['known'] is True and channel['total_tokens'] == 16
        diagnostic = channel['failure']
        assert diagnostic['phase'] == phase and diagnostic['exception_types'] == ['PermissionError']
        assert diagnostic['response_received'] is True and diagnostic['finish_reason'] == 'stop'
        assert record['canonical_result'] == canonical() and record['candidate'] is None
        assert record['delivery']['status'] == 'not_sent'
        assert not any(s in json.dumps(record) for s in ('PRIVATE_DIAGNOSTIC', '/secret/file', 'token=secret'))
        assert svc.archive.get(record['expression_id'], scope=SCOPE) == record
        assert prepare(svc) == record and len(calls) == 1


def test_dispatch_timeout_is_not_replaced_by_cleanup_failure(host, monkeypatch):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    with sdk_expression(error=httpx.ReadTimeout('PRIVATE_DIAGNOSTIC_BODY')) as (runtime, calls):
        def cleanup(*args, **kwargs):
            raise PermissionError('PRIVATE_DIAGNOSTIC_CLEANUP')
        monkeypatch.setattr(runtime, '_record_context_budget_usage', cleanup)
        svc = service(host, RuntimeExpressionAdapter(runtime))
        record = prepare(svc)
        assert record['fallback_reason'] == 'expression_transport_timeout'
        channel = record['usage']['expression']
        assert channel['known'] is False
        diagnostic = channel['failure']
        assert diagnostic['phase'] == 'dispatch' and diagnostic['response_received'] is False
        assert 'ReadTimeout' in diagnostic['exception_types']
        assert diagnostic['secondary_failures'] == [{'phase': 'context_usage', 'exception_types': ['PermissionError']}]
        assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(record)
        prepare(svc)
        assert len(calls) == 1


@pytest.mark.parametrize('channel', ['expression', 'check'])
def test_unknown_exception_names_and_messages_do_not_enter_archive(host, channel):
    error = type('PRIVATE_DIAGNOSTIC_CLASS', (Exception,), {})
    class Fails:
        calls = 0
        def call_expression_json(self, **options):
            self.calls += 1
            raise error('PRIVATE_DIAGNOSTIC_BODY')
    failed = Fails()
    model = failed if channel == 'expression' else Model(lambda v, _: {**v, 'layout': [*v['layout'], {'text': '慢慢看。'}]})
    svc = service(host, model, **({'checker': failed} if channel == 'check' else {}))
    record = prepare(svc)
    assert record['usage'][channel]['failure']['phase'] == 'service_call'
    assert record['usage'][channel]['failure']['exception_types'] == ['OtherError']
    assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(record)
    assert record['check']['accepted'] is False and record['candidate'] is None
    prepare(svc)
    assert failed.calls == 1


def test_late_sdk_success_adds_usage_without_erasing_host_timeout_or_reopening(host, monkeypatch):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    release = threading.Event()
    with sdk_expression(release=release) as (runtime, calls):
        svc = service(host, RuntimeExpressionAdapter(runtime), timeout_seconds=.03)
        try:
            record = prepare(svc)
            assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_timeout'
            assert record['usage']['expression']['failure']['phase'] == 'host_wait'
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while svc.active_calls and time.monotonic() < deadline:
            time.sleep(.005)
        assert svc.active_calls == 0
        final = svc.archive.get(record['expression_id'], scope=SCOPE)
        assert final['status'] == 'fallback' and final['text'] == record['text'] and final['candidate'] is None
        assert final['usage']['expression']['total_tokens'] == 16
        assert final['usage']['expression']['failure']['phase'] == 'host_wait'
        assert final['canonical_result'] == canonical() and len(calls) == 1
        assert prepare(svc) == final and len(calls) == 1


def test_diagnostic_failure_does_not_expand_execution_budget(host, monkeypatch):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    budget = TaskBudget(max_requests=1, max_tokens=10000, max_tools=1, max_active_seconds=30)
    budget.requests = 1
    before = budget.snapshot()
    with sdk_expression(error=httpx.ReadTimeout('PRIVATE_DIAGNOSTIC_BODY')) as (runtime, calls), bind_task_budget(budget):
        record = prepare(service(host, RuntimeExpressionAdapter(runtime)))
        assert record['usage']['expression']['failure']['phase'] == 'dispatch'
        assert budget.snapshot()['requests'] == before['requests']
        assert budget.tokens == before['tokens'] and budget.tools == before['tools'] and len(calls) == 1


def test_diagnostic_projection_does_not_copy_arbitrary_names_finish_fields_or_usage():
    error = type('PRIVATE_DIAGNOSTIC_CLASS', (Exception,), {})('PRIVATE_DIAGNOSTIC_BODY')
    value = call_failure_metadata(['PRIVATE_DIAGNOSTIC_PHASE'], error,
        response_received='PRIVATE_DIAGNOSTIC_RESPONSE', finish_reason=['PRIVATE_DIAGNOSTIC_FINISH'],
        usage={'total_tokens': 16, 'input_tokens': True, 'cost': 'PRIVATE_DIAGNOSTIC_COST',
               'PRIVATE_DIAGNOSTIC_FIELD': 'PRIVATE_DIAGNOSTIC_VALUE'})
    assert value['phase'] == 'service_call' and value['exception_types'] == ['OtherError']
    assert value['response_received'] is False and value['finish_reason'] == 'unknown'
    assert value['reported_usage'] == {'total_tokens': 16} and value['diagnostic_only'] is True
    assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(value)
