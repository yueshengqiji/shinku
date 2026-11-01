"""GF03: first failures and boolean fact checks survive without provider prose."""
import json
from types import SimpleNamespace

import httpx
import pytest

from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.result_expression import ModelReply
from test_expression_runtime import FixtureRuntime, response
from test_result_expression import SCOPE, Model, canonical, host, prepare, service


@pytest.mark.parametrize('content,finish,reason', [
    ('{"binding_hash":"fixture","layout":[]}', 'length', 'expression_output_truncated'),
    ('{bad-json}', 'stop', 'expression_json_invalid'),
    ('```json\n{}\n', 'stop', 'expression_wrapping_invalid'),
    ('[]', 'stop', 'expression_protocol_invalid'),
    ('{}', 'content_filter', 'expression_output_rejected'),
])
def test_concrete_transport_returns_fixed_first_reason_without_retry(monkeypatch, content, finish, reason):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    answer = response(content)
    answer.choices[0].finish_reason = finish
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), answer)[1])
    reply = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt='JSON', user_prompt='data')
    assert reply.value is None and getattr(reply, 'failure_reason', '') == reason
    assert len(calls) == 1 and reply.usage['total_tokens'] == 16


def test_native_tool_failure_has_its_own_code_and_never_executes(monkeypatch):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    answer = response(tools=[SimpleNamespace(id='fixture', type='function',
        function=SimpleNamespace(name='shell', arguments='{}'))])
    runtime = FixtureRuntime(lambda **_: answer)
    reply = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt='JSON', user_prompt='data')
    assert reply.value is None and getattr(reply, 'failure_reason', '') == 'expression_native_tool_call_forbidden'


def test_host_preserves_length_reason_usage_and_canonical_fallback(host, monkeypatch):
    monkeypatch.setenv('SHINKU_CONTEXT_BUDGET_MODE', 'off')
    answer = response('{"binding_hash":"unused","layout":[]}')
    answer.choices[0].finish_reason = 'length'
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), answer)[1])
    svc = service(host, RuntimeExpressionAdapter(runtime))
    record = prepare(svc)
    assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_output_truncated'
    assert record['usage']['expression']['total_tokens'] == 16
    assert record['canonical_result'] == canonical() and '12.5 ms' in record['text']
    assert prepare(svc) == record and len(calls) == 1


@pytest.mark.parametrize('exception,expected', [
    (httpx.ReadTimeout('PRIVATE_EXCEPTION_BODY'), 'expression_transport_timeout'),
    (httpx.ReadError('PRIVATE_EXCEPTION_BODY'), 'expression_transport_error'),
])
def test_runtime_transport_exception_chain_is_classified_without_storing_text(host, exception, expected):
    class Failed:
        calls = 0
        def call_expression_json(self, **_):
            self.calls += 1
            try:
                raise exception
            except Exception as cause:
                raise RuntimeError('PRIVATE_WRAPPER_BODY') from cause
    model = Failed()
    svc = service(host, model)
    record = prepare(svc)
    assert record['fallback_reason'] == expected and model.calls == 1
    assert 'PRIVATE_EXCEPTION_BODY' not in json.dumps(record) and 'PRIVATE_WRAPPER_BODY' not in json.dumps(record)
    assert record['usage']['expression']['known'] is False


@pytest.mark.parametrize('field,value,failed_key', [
    ('binding_hash', 'PRIVATE_FORGED_HASH', 'binding_matches'),
    ('supported', False, 'supported_true'),
    ('covered_fact_ids', ['PRIVATE_FORGED_ID'], 'coverage_exact'),
    ('covered_fact_ids', [42], 'covered_ids_valid'),
    ('additional_claims', ['PRIVATE_CLAIM'], 'additional_claims_empty'),
    ('contradictions', ['PRIVATE_CLAIM'], 'contradictions_empty'),
    ('omitted_limitations', ['PRIVATE_CLAIM'], 'omitted_limitations_empty'),
    ('extra', 'PRIVATE_EXTRA', 'schema_exact'),
])
def test_neutral_rejection_persists_only_fixed_boolean_checks(host, field, value, failed_key):
    model = Model(lambda c, _: {**c, 'layout': [*c['layout'], {'text': '慢慢看吧。'}]})
    calls = []
    class Checker:
        def call_expression_json(self, **options):
            calls.append(options)
            payload = json.loads(options['user_prompt'])
            candidate = {'binding_hash': payload['binding_hash'], 'supported': True,
                'covered_fact_ids': [b['id'] for b in payload['fact_blocks']],
                'additional_claims': [], 'contradictions': [], 'omitted_limitations': []}
            candidate[field] = value
            return ModelReply(candidate)
    svc = service(host, model, checker=Checker())
    record = prepare(svc)
    assert record['fallback_reason'] == 'expression_neutral_check_rejected'
    checks = record['check'].get('neutral_checks', {})
    assert len(checks) == 8 and all(type(v) is bool for v in checks.values()) and checks[failed_key] is False
    assert all(marker not in json.dumps(record) for marker in
               ('PRIVATE_FORGED_HASH', 'PRIVATE_FORGED_ID', 'PRIVATE_CLAIM', 'PRIVATE_EXTRA'))
    assert len(model.calls) == len(calls) == 1
    assert record['canonical_result'] == canonical()
    prepare(svc)
    assert len(model.calls) == len(calls) == 1


def test_protocol_fields_do_not_hide_as_json_syntax_error(host):
    record = prepare(service(host, Model(lambda c, _: {**c, 'unexpected': 'PRIVATE_INVALID_FIELD'})))
    assert record['fallback_reason'] == 'expression_protocol_invalid'
    assert 'PRIVATE_INVALID_FIELD' not in json.dumps(record)
