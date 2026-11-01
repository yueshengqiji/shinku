"""Long source text is rendered by the host and reviewed as complete prose.

Actual SDK adapters use isolated HTTP responses. Character counts measure
avoided transcription, not provider speed, persona quality or real cost.
"""
import copy
import json
import threading
import time

import httpx
import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ExpressionArchive, ExpressionService
from shinku.result_expression.facts import identity, template
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import STYLE, paragraphs, sdk_pair, service, source
from test_result_expression import SCOPE, host


TAIL = 'TAIL_LIMIT: 尚缺完整验收；引用不构成执行授权。'
LONG_TEXT = '资料段。' * 1800 + TAIL
CONDITION = '完整回执摘录，不代表已验收'
REFERENCE = 'https://source.example/retained-material'


def material_source():
    value = source()
    value.pop('result_hash')
    value['facts'].append(dict(label='必要原文', value=LONG_TEXT, condition=CONDITION,
                              reference=REFERENCE, source='executor_receipt'))
    value['presentation']['material_recovery'] = True
    result = seal_result(value)
    # The normalizer retains bounded canonical data. The scoped material
    # resolver supplies the full original, as in the existing material path.
    projection = dict(schema_version=1, source_result_hash=result['result_hash'],
                      status='complete', gaps=[], facts=copy.deepcopy(value['facts']), sources=[])
    assert TAIL not in result['facts'][1]['value']
    return result, projection


def references(payload):
    candidate = paragraphs(payload)
    blocks = {b['id']: b for b in payload['fact_blocks']}
    for row in candidate['paragraphs']:
        if row['fact_ids'] == ['information']:
            row['text'] = '这里还需要保留原交接与补读之间的区别。' + blocks['information']['text']
    covered = {v for row in candidate['paragraphs'] for v in row['fact_ids']}
    for key in payload['required_fact_ids']:
        if key not in covered and key != 'fact_1':
            candidate['paragraphs'].append(dict(text='请连同这项记录理解资料范围。' + blocks[key]['text'], fact_ids=[key]))
    candidate['paragraphs'].append(dict(fact='fact_1'))
    return candidate


def material_service(host, model, checker, projection, **options):
    svc = service(host, model, checker, max_input_chars=32768, max_output_tokens=8192, **options)
    svc.material_resolver = lambda result, **kwargs: copy.deepcopy(projection)
    return svc


def test_full_original_tail_conditions_sources_reach_both_sdk_and_delivered_body_without_transcription(host):
    value, projection = material_source()
    before = copy.deepcopy((value, projection))
    with sdk_pair(references) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and record['check']['accepted']
        assert [role for role, _, _ in calls] == ['expression', 'check']
        expr, checked = calls[0][2], calls[1][2]
        assert expr['host_reference_fact_ids'] == ['fact_1']
        long_block = next(b for b in expr['fact_blocks'] if b['id'] == 'fact_1')
        expected = '必要原文：' + LONG_TEXT + '；条件：' + CONDITION + '；来源：' + REFERENCE
        assert long_block['text'] == expected
        assert LONG_TEXT in checked['candidate_text'] and checked['candidate_text'] == record['text']
        assert next(b for b in checked['fact_blocks'] if b['id'] == 'fact_1') == long_block
        assert LONG_TEXT in record['text'] and TAIL in record['text']
        assert record['text'].count(LONG_TEXT) == 1
        assert CONDITION in record['text'] and REFERENCE in record['text']
        assert LONG_TEXT not in json.dumps(record['candidate'], ensure_ascii=False)
        assert record['check']['structure_checks']['host_reference_count'] == 1
        assert record['check']['structure_checks']['missing_fact_ids'] == []
        assert STYLE not in json.dumps(calls[1], ensure_ascii=False)
        assert all(wire['max_tokens'] == 8192 and not wire.get('tools') for _, wire, _ in calls)
        assert record['canonical_result'] == value and (value, projection) == before
        assert record['delivery']['status'] == 'not_sent'
        assert svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        svc.record_delivery(record, scope=SCOPE, receipt=dict(status='sent', sent=True, message_ids=['offline-reference']))
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert len(calls) == 2
        assert TAIL in json.dumps(svc.read_details(record['expression_id'], scope=SCOPE,
                                                 check_current=lambda: ''), ensure_ascii=False)


@pytest.mark.parametrize('row,reason', [
    ({'fact': 'unknown-private-id'}, 'expression_fact_or_phrase_unknown'),
    ({'fact': 'fact_0'}, 'expression_host_reference_forbidden'),
    ({'fact': 'status'}, 'expression_host_reference_forbidden'),
    ({'fact': 'delivery'}, 'expression_host_reference_forbidden'),
    ({'fact': 'fact_1', 'text': '替换原文'}, 'expression_layout_invalid'),
    ({'fact': 'fact_1', 'fact_ids': ['fact_1']}, 'expression_layout_invalid'),
    ({'fact': 1}, 'expression_layout_invalid'),
    ({'fact': ['fact_1']}, 'expression_layout_invalid'),
])
def test_model_cannot_choose_unknown_short_semantic_or_replacement_host_reference(host, row, reason):
    def build(payload):
        candidate = references(payload)
        candidate['paragraphs'][-1] = row
        return candidate
    value, projection = material_source()
    with sdk_pair(build) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == reason and record['candidate'] is None
        assert len(calls) == 1 and TAIL in record['text']
        assert 'unknown-private-id' not in json.dumps(record, ensure_ascii=False)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 1


@pytest.mark.parametrize('change,reason', [
    (lambda c: c['paragraphs'].pop(), 'expression_required_facts_missing_or_duplicated'),
    (lambda c: c['paragraphs'].append(dict(fact='fact_1')), 'expression_required_facts_missing_or_duplicated'),
    (lambda c: c['paragraphs'][1].update(text=c['paragraphs'][1]['text'].replace('12.5 ms', '12.6 ms')),
     'expression_protected_value_missing'),
    (lambda c: c.update(binding_hash='forged-old-binding'), 'expression_binding_conflict'),
])
def test_reference_preserves_required_coverage_measurement_and_source_binding(host, change, reason):
    def build(payload):
        candidate = references(payload)
        change(candidate)
        return candidate
    value, projection = material_source()
    with sdk_pair(build) as (model, checker, calls):
        record = material_service(host, model, checker, projection).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == reason and len(calls) == 1
        assert '12.5 ms' in record['text'] and LONG_TEXT in record['text']
        assert record['candidate'] is None and record['canonical_result'] == value


def test_catalogue_with_long_reference_still_requires_natural_explanation(host):
    def build(payload):
        rows = [b for b in payload['fact_blocks'] if b.get('required', True) and b['id'] != 'fact_1']
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[
            dict(text='\n'.join(b['text'] for b in rows), fact_ids=[b['id'] for b in rows]), dict(fact='fact_1')])
    value, projection = material_source()
    with sdk_pair(build) as (model, checker, calls):
        record = material_service(host, model, checker, projection).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_prose_required' and len(calls) == 1
        assert record['candidate'] is None and LONG_TEXT in record['text']


@pytest.mark.parametrize('input_limit,calls_expected,reason', [
    (512, 0, 'expression_input_oversized'),
    (15000, 1, 'expression_check_input_oversized'),
])
def test_reference_does_not_hide_original_input_or_bypass_full_body_checker_budget(host, input_limit,
                                                                                 calls_expected, reason):
    from dataclasses import replace
    value, projection = material_source()
    with sdk_pair(references) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        svc.profile = replace(svc.profile, max_input_chars=input_limit)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == reason and len(calls) == calls_expected
        assert record['candidate'] is None and LONG_TEXT in record['text'] and record['canonical_result'] == value
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == calls_expected


def test_privacy_check_includes_free_prose_beside_long_reference_before_neutral_request(host):
    def build(payload):
        candidate = references(payload)
        candidate['paragraphs'].append(dict(text=STYLE, fact_ids=[]))
        return candidate
    value, projection = material_source()
    with sdk_pair(build) as (model, checker, calls):
        record = material_service(host, model, checker, projection).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_private_context_leak' and len(calls) == 1
        assert record['candidate'] is None and STYLE not in json.dumps(record, ensure_ascii=False)
        assert LONG_TEXT in record['text']


@pytest.mark.parametrize('verdict', [
    dict(supported=False, contradictions=['引用不证明完整验收']),
    dict(supported=False, additional_claims=['无依据新事实']),
    dict(supported=False, omitted_limitations=['尾部限制未说明']),
    dict(covered_fact_ids=['status']),
])
def test_independent_rejection_of_final_host_rendered_body_is_not_bypassed_by_reference(host, verdict):
    value, projection = material_source()
    with sdk_pair(references, verdict=verdict) as (model, checker, calls):
        record = material_service(host, model, checker, projection).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_neutral_check_rejected'
        assert len(calls) == 2 and LONG_TEXT in calls[1][2]['candidate_text']
        assert record['candidate'] is None and record['canonical_result'] == value


@pytest.mark.parametrize('phase', ['expression', 'check'])
def test_permission_revocation_during_model_wait_discards_reference_and_blocks_publication(host, phase):
    value, projection = material_source()
    state = ['']
    options = {phase + '_hook': lambda: state.__setitem__(0, 'task_canceled')}
    with sdk_pair(references, **options) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: state[0])
        assert record['status'] == 'discarded' and record['text'] == '' and record['candidate'] is None
        assert record['fallback_reason'] == 'task_canceled'
        assert len(calls) == (1 if phase == 'expression' else 2)
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: state[0])
        assert record['canonical_result'] == value


def test_transport_timeout_keeps_first_cause_full_fallback_and_consumed_attempt(host):
    def fail():
        raise httpx.ReadTimeout('isolated fixture')
    value, projection = material_source()
    with sdk_pair(references, expression_hook=fail) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_transport_timeout' and LONG_TEXT in record['text']
        assert record['usage']['expression']['calls'] == 1 and not record['usage']['expression']['known']
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 1


def test_late_long_reference_reply_never_revives_timeout_or_starts_checker(host):
    entered, release = threading.Event(), threading.Event()
    def paused():
        entered.set()
        assert release.wait(2)
    value, projection = material_source()
    with sdk_pair(references, expression_hook=paused) as (model, checker, calls):
        svc = material_service(host, model, checker, projection, timeout_seconds=.04)
        try:
            record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
            assert entered.is_set() and record['fallback_reason'] == 'expression_timeout'
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while svc.active_calls and time.monotonic() < deadline:
            time.sleep(.001)
        assert not svc.active_calls
        cached = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert cached['status'] == 'fallback' and cached['candidate'] is None
        assert cached['fallback_reason'] == 'expression_timeout' and LONG_TEXT in cached['text']
        assert len(calls) == 1


def test_protocol_upgrade_retains_old_cache_and_details_without_buying_another_attempt(host):
    value, projection = material_source()
    with sdk_pair(references) as (model, checker, calls):
        svc = material_service(host, model, checker, projection)
        profile = svc.profile
        old_config = canonical_hash(dict(id=profile.configuration_id, style=profile.style_prompt,
            runtime=model.identity, checker=checker.identity, timeout=profile.timeout_seconds,
            check_timeout=profile.checker_timeout_seconds, input=profile.max_input_chars,
            output=profile.max_output_tokens, format=2))
        old_binding = canonical_hash({**identity(value), 'result_hash': value['result_hash']})
        key = canonical_hash(identity(value))
        old, owned = svc.archive.claim(key, scope=SCOPE, result=value, config_hash=old_config,
                                      binding_hash=old_binding, format_version=2)
        assert owned and old_config != profile.config_hash
        retained = svc.archive.finish(key, scope=SCOPE, status='fallback',
            text=template(paragraph_blocks(value)), fallback_reason='expression_timeout')
        current = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert current['status'] == 'discarded' and current['fallback_reason'] == 'expression_configuration_changed'
        assert not calls and svc.archive.get(key, scope=SCOPE) == retained
        assert svc.read_details(key, scope=SCOPE, check_current=lambda: '')['status'] == 'available'
        assert not calls


def test_fresh_service_replays_reference_cache_and_material_revocation_blocks_delivery(host):
    value, projection = material_source()
    with sdk_pair(references) as (model, checker, calls):
        original = material_service(host, model, checker, projection)
        record = original.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and len(calls) == 2
        restarted = ExpressionService(ExpressionArchive(host), original.profile,
            material_resolver=lambda *args, **kwargs: pytest.fail('cache must not reread material'),
            material_validator=lambda *args, **kwargs: False)
        assert restarted.prepare(value, scope=SCOPE, check_current=lambda: '') == record
        assert not restarted.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert restarted.read_details(record['expression_id'], scope=SCOPE,
                                      check_current=lambda: '')['status'] == 'discarded'
        assert restarted.read_details(record['expression_id'], scope=AgentScope('other', SCOPE.session),
                                      check_current=lambda: '')['status'] == 'discarded'
        assert len(calls) == 2 and original.archive.get(record['expression_id'], scope=SCOPE) == record


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('delivery', ['sent', 'uncertain'])
def test_actual_qq_task_long_read_fact_expression_and_transport_are_once_only(tmp_path, monkeypatch,
                                                                            asynchronous, delivery):
    from shinku.result_expression import ExpressionProfile
    from test_qq_independent_agent import turn
    from test_qq_result_expression import make_host, wait_work
    material_path = tmp_path / 'read-material.txt'
    material_path.write_text('资料段。' * 450 + TAIL, encoding='utf-8')
    original_text = material_path.read_text(encoding='utf-8')

    def build(payload):
        required = [b for b in payload['fact_blocks'] if b.get('required', True)]
        prose = [b for b in required if b['id'] not in payload['host_reference_fact_ids']]
        assert payload['host_reference_fact_ids'] == ['fact_0']
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[
            dict(text='执行回执已经核对，请同时保留这些范围和限制。\n' + '\n'.join(b['text'] for b in prose),
                 fact_ids=[b['id'] for b in prose]), dict(fact='fact_0')])

    with sdk_pair(build) as (model, checker, calls):
        runtime, tool, executor, request, bridge, chat, sent = make_host(tmp_path, asynchronous=asynchronous,
                                                                      delivery=delivery)
        actual_execute = tool.execute
        def read_and_execute(**kwargs):
            result = actual_execute(**kwargs)
            result['state_updates']['result_facts'] = [dict(label='实际读取原文',
                value=material_path.read_text(encoding='utf-8'), condition='只引用读取时的本地资料',
                reference=str(material_path))]
            return result
        monkeypatch.setattr(tool, 'execute', read_and_execute)
        bridge.result_expression.profile = ExpressionProfile(model, checker=checker, format_version=2,
            configuration_id='reference-qq-fixture', style_prompt=STYLE, max_input_chars=32768)
        try:
            reply = bridge(turn(request.original_request))
            if asynchronous:
                assert wait_work(runtime, reply.background_task_id)['status'] == 'completed'
            else:
                assert reply.agent_status == 'completed'
            task = runtime.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
            result = task['execution_result']
            record = bridge.result_expression.archive.get(canonical_hash(identity(result)), scope=SCOPE)
            assert record['status'] == 'ready' and record['delivery']['status'] == delivery
            assert record['canonical_result'] == result and result['verification']['accepted'] is True
            assert record['check']['structure_checks']['host_reference_count'] == 1
            assert len(calls) == 2 and tool.calls == 1
            actual = '\n'.join(part.text for msg in sent for part in msg.segments if part.kind == 'text')
            assert actual == record['text'] and original_text in actual and TAIL in actual
            assert calls[1][2]['candidate_text'] == actual
            counts = (len(sent), len(chat.calls), len(executor.calls), tool.calls, len(calls))
            assert bridge.result_expression.prepare(result, scope=SCOPE, check_current=lambda: '') == record
            assert not bridge.result_expression.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
            assert counts == (len(sent), len(chat.calls), len(executor.calls), tool.calls, len(calls))
        finally:
            runtime.coordinator.shutdown(wait=True)
            runtime.task_store.close()
