"""Natural paragraph contract through real SDK adapters, without live IO.

Explicit source values and adversarial verdicts are independent expectations.
Scripted prose/verdicts prove enforcement, never real persona quality.
"""
import copy
from contextlib import contextmanager
from dataclasses import replace
import json
import threading

import httpx
from openai import OpenAI
import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.results import seal_result
from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from test_expression_runtime import FixtureRuntime
from test_result_expression import SCOPE, canonical, host


STYLE = 'FULL_STYLE_PRIVATE_SENTINEL; 使用完整上下文写自然段落。'


def source():
    value = canonical()
    value.pop('result_hash')
    value['presentation'] = dict(schema_version=1, optional_success_receipts=True,
        policy_source='host_export', required='goals_constraints_facts_conditions_artifacts_limits')
    value['information'] = dict(status='complete', required_status='complete',
        snapshot=dict(gap_count=0, source_sha256='b' * 64))
    return seal_result(value)


def paragraphs(payload):
    return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[
        dict(text='你要的是「测量响应时间并生成报告，不部署」。测量和报告已完成，测量这个条件已满足，整项目标验收通过。',
             fact_ids=['goal', 'status', 'verification']),
        dict(text='响应时间记录为12.5 ms，确认约束是「条件为离线隔离环境」，来源是https://example.invalid/source。这个结果需要连同测量条件一起看。',
             fact_ids=['fact_0', 'constraints']),
        dict(text='产物回执对应fixture-report.txt，核验状态为task_passed；内容身份以核验时为准。', fact_ids=['artifact_0']),
        dict(text='未完成项：验收记录未列出未完成项。不过没有在线样本，所以不能把离线结果当成线上性能。'
                  '部署记录：未部署或缺少上线回执，不能确认已上线。'
                  '送达未知；本次新表达尚未取得完整送达回执。',
             fact_ids=['remaining', 'uncertainties', 'deployment', 'delivery']),
        dict(text='资料交接完整性标记为完整，所需资料也标记为完整；摘要或哈希不能证明模型已理解全量内容。',
             fact_ids=['information'])])


@contextmanager
def sdk_pair(build=paragraphs, *, verdict=None, check_hook=None, expression_hook=None, check_error=None):
    calls = []
    def transport(role):
        def handle(request):
            wire = json.loads(request.content)
            payload = json.loads(wire['messages'][-1]['content'])
            calls.append((role, wire, payload))
            assert not wire.get('tools') and not wire.get('tool_choice')
            if role == 'expression':
                if expression_hook:
                    expression_hook()
                value = build(payload)
            else:
                if check_hook:
                    check_hook()
                if check_error:
                    raise check_error
                value = dict(binding_hash=payload['binding_hash'], supported=True,
                    covered_fact_ids=[b['id'] for b in payload['fact_blocks']],
                    additional_claims=[], contradictions=[], omitted_limitations=[])
                value.update(verdict or {})
            return httpx.Response(200, json=dict(id='offline', object='chat.completion', created=0, model='offline',
                choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant', content=json.dumps(value, ensure_ascii=False)))],
                usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16)))
        return httpx.MockTransport(handle)
    with httpx.Client(transport=transport('expression'), trust_env=False) as expr_http, \
            httpx.Client(transport=transport('check'), trust_env=False) as check_http:
        with OpenAI(api_key='offline', base_url='https://paragraph.invalid/v1', max_retries=0, http_client=expr_http) as expr, \
                OpenAI(api_key='offline', base_url='https://checker.invalid/v1', max_retries=0, http_client=check_http) as check:
            yield RuntimeExpressionAdapter(FixtureRuntime(expr.chat.completions.create)), RuntimeExpressionAdapter(FixtureRuntime(check.chat.completions.create)), calls


def service(host, model, checker, **options):
    return ExpressionService(ExpressionArchive(host), ExpressionProfile(model, checker=checker,
        format_version=2, configuration_id='paragraph-fixture', style_prompt=STYLE, **options))


def test_sdk_natural_paragraphs_have_exact_values_semantic_review_and_separate_retained_audit(host):
    value = source(); before = copy.deepcopy(value)
    with sdk_pair() as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and record['check']['mode'] == 'protected_values_and_neutral'
        assert [role for role, _, _ in calls] == ['expression', 'check']
        assert 'phrases' not in calls[0][2] and 'layout_example' not in calls[0][2]
        assert calls[0][1]['messages'][0]['content'].endswith(STYLE)
        measurement = next(b for b in calls[0][2]['fact_blocks'] if b['id'] == 'fact_0')
        spans = {measurement['text'][a['start']:a['end']] for a in measurement['anchors']}
        assert spans == {'响应时间', '12.5 ms', '离线隔离环境', 'https://example.invalid/source'}
        assert STYLE not in json.dumps(calls[1])
        assert calls[1][2]['candidate_text'] == record['text']
        assert all(not b['id'].endswith('_audit') and b['id'] != 'evidence_0' for b in calls[1][2]['fact_blocks'])
        assert '实际记录（引用）' not in record['text'] and '"condition"' not in record['text']
        assert '12.5 ms' in record['text'] and 'https://example.invalid/source' in record['text']
        assert 'a' * 64 not in record['text'] and 'b' * 64 not in record['text']
        assert value == record['canonical_result'] == before and record['delivery']['status'] == 'not_sent'
        details = svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert details['status'] == 'available' and details['result_hash'] == value['result_hash']
        assert 'a' * 64 in json.dumps(details) and 'b' * 64 in json.dumps(details)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2
        assert svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        svc.record_delivery(record, scope=SCOPE, receipt=dict(status='sent', sent=True, message_ids=['test-out']))
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert len(calls) == 2 and value == before


@pytest.mark.parametrize('catalog', ['v1', 'v2'])
def test_observed_catalog_only_candidate_is_not_a_natural_expression_pass(host, catalog):
    def build(payload):
        if catalog == 'v1':
            return dict(binding_hash=payload['binding_hash'], layout=[dict(phrase='intro'), *[dict(fact=b['id']) for b in payload['fact_blocks']]])
        required = [b for b in payload['fact_blocks'] if b.get('required', True)]
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(text='\n'.join(b['text'] for b in required),
            fact_ids=[b['id'] for b in required])])
    with sdk_pair(build) as (model, checker, calls):
        svc = service(host, model, checker)
        value = source()
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and record['candidate'] is None
        assert record['fallback_reason'] == ('expression_protocol_invalid' if catalog == 'v1' else 'expression_prose_required')
        assert len(calls) == 1 and '12.5 ms' in record['text'] and '没有在线样本' in record['text']
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 1


@pytest.mark.parametrize('old,new', [
    ('12.5 ms', '12.6 ms'), ('12.5 ms', '912.5 ms'), ('12.5 ms', '12.5 msec'), ('12.5 ms', '12.5 ms²'),
    ('离线隔离环境', '任意环境'), ('https://example.invalid/source', 'https://example.invalid/forged'),
    ('https://example.invalid/source', 'https://example.invalid/source/forged'),
    ('fixture-report.txt', 'fixture-report.txt.exe'), ('没有在线样本', '已有在线样本'),
])
def test_protected_values_reject_changes_before_semantic_call_even_when_ids_are_present(host, old, new):
    def build(payload):
        candidate = paragraphs(payload)
        for paragraph in candidate['paragraphs']:
            paragraph['text'] = paragraph['text'].replace(old, new)
        return candidate
    with sdk_pair(build) as (model, checker, calls):
        svc = service(host, model, checker)
        value = source(); before = copy.deepcopy(value)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing'
        assert record['candidate'] is None and len(calls) == 1
        assert value == record['canonical_result'] == before and record['delivery']['status'] == 'not_sent'


@pytest.mark.parametrize('mutation,reason', [
    (lambda c: c.update(binding_hash='wrong'), 'expression_binding_conflict'),
    (lambda c: c.update(task_outcome='completed'), 'expression_protocol_invalid'),
    (lambda c: c.update(tool_call=dict(name='shell')), 'expression_protocol_invalid'),
    (lambda c: c['paragraphs'][0]['fact_ids'].remove('status'), 'expression_required_facts_missing_or_duplicated'),
    (lambda c: c['paragraphs'][0]['fact_ids'].append('status'), 'expression_required_facts_missing_or_duplicated'),
    (lambda c: c['paragraphs'][0]['fact_ids'].append('invented'), 'expression_fact_or_phrase_unknown'),
])
def test_attribution_and_binding_do_not_authorize_task_state_or_tools(host, mutation, reason):
    def build(payload):
        candidate = paragraphs(payload); mutation(candidate); return candidate
    with sdk_pair(build) as (model, checker, calls):
        record = service(host, model, checker).prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == reason and len(calls) == 1 and record['candidate'] is None


@pytest.mark.parametrize('claim,verdict', [
    ('现在已部署并送达。', dict(supported=False, contradictions=['deployment and delivery'])),
    ('这个结果适用于所有环境。', dict(supported=False, omitted_limitations=['offline only'])),
    ('另一个测量值是16 kg。', dict(supported=True, additional_claims=['unsupported measurement'])),
])
def test_semantic_gate_still_rejects_claims_even_with_all_exact_values(host, claim, verdict):
    def build(payload):
        value = paragraphs(payload); value['paragraphs'][0]['text'] += claim; return value
    with sdk_pair(build, verdict=verdict) as (model, checker, calls):
        record = service(host, model, checker).prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and len(calls) == 2
        assert claim not in record['text'] and record['candidate'] is None
        assert record['canonical_result']['deployment']['status'] == 'not_deployed'


def test_without_checker_version_two_has_no_transport_and_never_silently_uses_catalog(host):
    with sdk_pair() as (model, _, calls):
        svc = service(host, model, None)
        record = svc.prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_required' and calls == []


def test_style_leak_is_blocked_before_checker_without_persisting_rejected_prose(host):
    def build(payload):
        candidate = paragraphs(payload); candidate['paragraphs'][0]['text'] += STYLE; return candidate
    with sdk_pair(build) as (model, checker, calls):
        record = service(host, model, checker).prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_private_context_leak' and len(calls) == 1
        assert STYLE not in json.dumps(record) and record['candidate'] is None


@pytest.mark.parametrize('secret', ['api_key=synthetic-key', 'https://user:password@example.invalid',
                                  'https://example.invalid/?signature=synthetic-signature'])
def test_full_paragraph_sanitization_still_rejects_credentials_before_neutral_call(host, secret):
    def build(payload):
        candidate = paragraphs(payload); candidate['paragraphs'][0]['text'] += '普通说明。' * 600 + secret
        return candidate
    with sdk_pair(build) as (model, checker, calls):
        record = service(host, model, checker).prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_private_context_leak' and len(calls) == 1
        assert secret not in json.dumps(record) and record['candidate'] is None


@pytest.mark.parametrize('status,accepted,expected', [('partial', False, '尚未通过验收'),
    ('failed', False, '执行失败'), ('canceled', False, '取消不证明操作已撤销或进程已清理'),
    ('waiting_user', False, '等待用户确认'), ('completed', False, '尚未确认完成')])
def test_noncomplete_results_cannot_be_upgraded_by_natural_expression(host, status, accepted, expected):
    def build(payload):
        blocks = [b for b in payload['fact_blocks'] if b.get('required', True)]
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(
            text='需要先看清这份记录的边界。\n' + '\n'.join(b['text'] for b in blocks), fact_ids=[b['id'] for b in blocks])])
    with sdk_pair(build) as (model, checker, calls):
        svc = service(host, model, checker)
        value = canonical(status, accepted=accepted); before = copy.deepcopy(value)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and expected in record['text'] and len(calls) == 2
        assert value == record['canonical_result'] == before and not value['verification']['accepted']
        assert record['delivery']['status'] == 'not_sent'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_actual_qq_execution_to_paragraph_expression_and_delivery_is_once_only(tmp_path, asynchronous):
    from shinku.agent_runtime.contracts import canonical_hash
    from shinku.result_expression.facts import identity
    from test_qq_result_expression import make_host, wait_work
    from test_qq_independent_agent import turn
    def build(payload):
        blocks = [b for b in payload['fact_blocks'] if b.get('required', True)]
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(
            text='这份结果需要连同来源和限制一起看。\n' + '\n'.join(b['text'] for b in blocks[:2]),
            fact_ids=[b['id'] for b in blocks[:2]]), dict(
            text='再看记录中的具体结果和限制。\n' + '\n'.join(b['text'] for b in blocks[2:]),
            fact_ids=[b['id'] for b in blocks[2:]])])
    with sdk_pair(build) as (model, checker, calls):
        runtime, tool, executor, request, bridge, chat, sent = make_host(tmp_path, asynchronous=asynchronous)
        bridge.result_expression.profile = ExpressionProfile(model, checker=checker, format_version=2, style_prompt=STYLE)
        try:
            reply = bridge(turn(request.original_request))
            if asynchronous:
                assert wait_work(runtime, reply.background_task_id)['status'] == 'completed'
            else:
                assert reply.agent_status == 'completed'
            task = runtime.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
            result = task['execution_result']
            record = bridge.result_expression.archive.get(canonical_hash(identity(result)), scope=SCOPE)
            assert record['status'] == 'ready' and record['format_version'] == 2
            assert record['delivery']['status'] == 'sent' and tool.calls == 1
            assert record['delivered_content']['content_complete']
            actual = '\n'.join(part.text for msg in sent for part in msg.segments if part.kind == 'text')
            assert actual == record['text'] and '\n\n' in actual
            sent_before = len(sent)
            assert sent_before == len(record['delivery']['message_ids']) > 0
            assert len(calls) == 2 and record['canonical_result'] == result
            assert bridge.result_expression.prepare(result, scope=SCOPE, check_current=lambda: '') == record
            assert len(calls) == 2 and len(sent) == sent_before and tool.calls == 1
            assert bridge.result_expression.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')['status'] == 'available'
            assert len(calls) == 2 and len(sent) == sent_before and tool.calls == 1
        finally:
            runtime.coordinator.shutdown(wait=True)
            runtime.task_store.close()


def test_cancel_during_neutral_check_discards_and_detail_scope_has_no_model_or_delivery(host):
    state = ['']
    with sdk_pair(check_hook=lambda: state.__setitem__(0, 'task_canceled')) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(source(), scope=SCOPE, check_current=lambda: state[0])
        assert record['status'] == 'discarded' and record['text'] == '' and record['candidate'] is None
        assert svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: state[0])['status'] == 'discarded'
        assert svc.read_details(record['expression_id'], scope=AgentScope('other', SCOPE.session), check_current=lambda: '')['status'] == 'discarded'
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '') and len(calls) == 2


def test_checker_timeout_falls_back_once_and_retains_unknown_usage(host):
    with sdk_pair(check_error=httpx.ReadTimeout('offline')) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_transport_timeout' and len(calls) == 2
        assert record['usage']['check']['calls'] == 1 and record['usage']['check']['known'] is False
        assert svc.prepare(source(), scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2


def test_late_paragraph_reply_cannot_revive_timed_out_candidate(host):
    entered, release = threading.Event(), threading.Event()
    def paused():
        entered.set(); assert release.wait(2)
    with sdk_pair(expression_hook=paused) as (model, checker, calls):
        svc = service(host, model, checker, timeout_seconds=.02)
        value = source()
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert entered.is_set() and record['fallback_reason'] == 'expression_timeout'
        release.set()
        # _call owns a bounded worker; join its completion without a long wait.
        from time import monotonic, sleep
        deadline = monotonic() + 2
        while svc.active_calls and monotonic() < deadline:
            sleep(.005)
        repeated = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert repeated['status'] == 'fallback' and repeated['text'] == record['text']
        assert repeated['candidate'] is None and len(calls) == 1 and not svc.active_calls


def test_format_change_uses_new_cache_identity_without_buying_a_second_attempt(host):
    with sdk_pair() as (model, checker, calls):
        svc = service(host, model, checker)
        value = source()
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        old = replace(svc.profile, format_version=1)
        assert old.config_hash != svc.profile.config_hash
        discarded = ExpressionService(svc.archive, old).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert discarded['fallback_reason'] == 'expression_configuration_changed' and len(calls) == 2
        assert svc.archive.get(record['expression_id'], scope=SCOPE) == record


def test_expired_paragraph_claim_recovers_same_scoped_result_without_transport_or_late_revival(host):
    from shinku.agent_runtime.contracts import canonical_hash
    from shinku.result_expression.facts import identity
    clock = [1000]
    host.clock = lambda: clock[0]
    with sdk_pair() as (model, checker, calls):
        svc = service(host, model, checker)
        value = source(); before = copy.deepcopy(value)
        key = canonical_hash(identity(value))
        svc.archive.claim(key, scope=SCOPE, result=value, config_hash=svc.profile.config_hash, format_version=2,
            binding_hash=canonical_hash({**identity(value), 'result_hash': value['result_hash']}), processing_seconds=10)
        clock[0] += 11
        restarted = ExpressionService(ExpressionArchive(host), svc.profile)
        record = restarted.prepare(value, scope=SCOPE, check_current=lambda: '', recover_interrupted=True)
        assert record['fallback_reason'] == 'expression_interrupted' and record['format_version'] == 2
        assert '12.5 ms' in record['text'] and '没有在线样本' in record['text'] and not calls
        assert restarted.archive.finish(key, scope=SCOPE, status='ready', text='late result') == record
        assert record['candidate'] is None and record['canonical_result'] == value == before
        assert restarted.read_details(key, scope=SCOPE, check_current=lambda: '')['status'] == 'available'
        assert not calls and record['delivery']['status'] == 'not_sent'


def test_hash_reported_as_substantive_fact_cannot_be_hidden_as_optional_audit(host):
    value = source(); value.pop('result_hash')
    value['facts'].append(dict(label='内容哈希', value='a' * 64, source='executor_receipt'))
    value = seal_result(value)
    def build(payload):
        candidate = paragraphs(payload)
        candidate['paragraphs'].append(dict(text='内容哈希是' + 'a' * 64 + '，这里只对应原核验内容。', fact_ids=['fact_1']))
        return candidate
    with sdk_pair(build) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and 'a' * 64 in record['text'] and len(calls) == 2
        assert 'fact_1' in calls[0][2]['required_fact_ids']
        assert 'artifact_0_audit' in calls[0][2]['optional_fact_ids']


@pytest.mark.parametrize('drop_tail', [False, True])
def test_actual_material_recovery_keeps_full_condition_and_revoke_blocks_detail_and_publication(tmp_path, drop_tail):
    from shinku.agent_runtime.demo import demo_request
    from shinku.agent_runtime.material_presentation import recover_presentation, validate_presentation
    from test_agent_material_integration import receipt_fixture, wait_client, close
    runtime, tool, executor, materials = receipt_fixture(tmp_path)
    def build(payload):
        required = [b for b in payload['fact_blocks'] if b.get('required', True)]
        text = '这些记录需要连同原条件一起理解。\n' + '\n'.join(b['text'] for b in required)
        if drop_tail:
            text = text.replace('条件0：仅离线，不部署。', '')
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(text=text,
            fact_ids=[b['id'] for b in required])])
    try:
        handle = runtime.bind(SCOPE).submit(demo_request(tool)); wait_client(runtime, handle.task_id)
        value = runtime.bind(SCOPE).result(handle.task_id); before = copy.deepcopy(value)
        with sdk_pair(build) as (model, checker, calls):
            svc = ExpressionService(ExpressionArchive(runtime.task_store),
                ExpressionProfile(model, checker=checker, format_version=2, style_prompt=STYLE, max_input_chars=65536),
                material_resolver=lambda result, **kw: recover_presentation(materials, result, **kw),
                material_validator=lambda result, projection, **kw: validate_presentation(materials, result, projection, **kw))
            record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
            assert record['status'] == ('fallback' if drop_tail else 'ready'), record['fallback_reason']
            assert 'a' * 3000 in record['text'] and '条件0：仅离线，不部署。' in record['text']
            if drop_tail:
                assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1
            else:
                assert len(calls) == 2
            assert record['canonical_result'] == value == before and tool.calls == 1 and len(executor.calls) == 2
            task = runtime.task_store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
            limits = materials.limits(task)
            assert svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')['status'] == 'available'
            assert materials.limits(task) == limits
            materials.revoke(task['material_package']['items'][0]['reference'], task=task)
            assert svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')['status'] == 'discarded'
            assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
            assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record
            assert materials.limits(task) == limits and tool.calls == 1
    finally:
        close(runtime)
