"""Per-clause coverage through the real SDK, with independent bad verdicts."""
import copy
from contextlib import contextmanager
from dataclasses import replace
import json

import httpx
from openai import OpenAI
import pytest

from shinku.agent_runtime.contracts import canonical_hash
from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from test_expression_paragraphs import STYLE, sdk_pair, service, source
from test_expression_runtime import FixtureRuntime
from test_expression_verified_conditions import file_source
from test_result_expression import host, SCOPE


def full_body(payload):
    rows = [r for r in payload['fact_blocks'] if r.get('required', True)]
    return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[
        dict(text='本次记录如下。\n'+'\n'.join(r['text'] for r in rows),
             fact_ids=[r['id'] for r in rows])])


@contextmanager
def semantic_sdk(*, build=full_body, change=None):
    calls = []
    def transport(role):
        def respond(request):
            wire = json.loads(request.content)
            payload = json.loads(wire['messages'][-1]['content'])
            calls.append((role, wire, payload))
            assert not wire.get('tools') and not wire.get('tool_choice')
            if role == 'expression':
                value = build(payload)
            else:
                value = dict(binding_hash=payload['binding_hash'], supported=True,
                    covered_fact_ids=[b['id'] for b in payload['fact_blocks']],
                    additional_claims=[], contradictions=[], omitted_limitations=[],
                    semantic_coverage=[dict(id=r['id'], answer_fragment=(payload['candidate_text'][:2048] if r.get('coverage_scope') == 'whole_report' else r['text']), supported=True)
                                       for r in payload['semantic_requirements']])
                if change:
                    change(value, payload)
            return httpx.Response(200, json=dict(id='offline-semantic', object='chat.completion', created=0,
                model='offline', choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant',
                content=json.dumps(value, ensure_ascii=False)))],
                usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16)))
        return httpx.MockTransport(respond)
    with httpx.Client(transport=transport('expression'), trust_env=False) as expr_http, \
            httpx.Client(transport=transport('check'), trust_env=False) as check_http:
        with OpenAI(api_key='offline', base_url='https://semantic-expression.invalid/v1', max_retries=0,
                    http_client=expr_http) as expr, \
                OpenAI(api_key='offline', base_url='https://semantic-check.invalid/v1', max_retries=0,
                       http_client=check_http) as check:
            yield RuntimeExpressionAdapter(FixtureRuntime(expr.chat.completions.create)), \
                  RuntimeExpressionAdapter(FixtureRuntime(check.chat.completions.create)), calls


def semantic_service(host, model, checker, **options):
    return service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4,
                   semantic_review_version=1, **options)


def test_sdk_requires_each_clause_and_stores_only_safe_diagnostics_and_proof_hash(host):
    value = file_source(); before = copy.deepcopy(value)
    with semantic_sdk() as (model, checker, calls):
        svc = semantic_service(host, model, checker)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready', record['fallback_reason']
        assert record['canonical_result'] == before == value
        assert record['semantic_review_version'] == 1
        assert record['check']['mode'] == 'protected_values_neutral_and_semantic_coverage'
        assert [c[0] for c in calls] == ['expression', 'check']
        reqs = calls[0][2]['semantic_requirements']
        assert reqs == calls[1][2]['semantic_requirements']
        assert len([r for r in reqs if r['fact_id'] == 'verification']) >= 2
        assert all(r['fact_id'] not in {'goal', 'contract_context', 'information_audit'} for r in reqs)
        assert any('保持不变' in r['text'] for r in reqs)
        assert any('只写入一次' in r['text'] for r in reqs)
        assert STYLE not in json.dumps(calls[1])
        diagnostic = record['check']['semantic_checks']
        assert diagnostic['requirements_count'] == len(reqs)
        assert diagnostic['coverage_count'] == len(reqs)
        assert len(diagnostic['proof_hash']) == 64
        assert all(type(v) in (bool, int, str) for v in diagnostic.values())
        assert 'answer_fragment' not in json.dumps(record)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2
        assert svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
        assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')


def test_positive_whole_block_verdict_cannot_omit_one_required_clause(host):
    def remove_proof(v, payload):
        # The old six-field verdict still says supported=True and every fact ID.
        v['semantic_coverage'].pop()
    with semantic_sdk(change=remove_proof) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_semantic_coverage_invalid'
        assert record['check']['neutral_checks']['supported_true']
        assert record['check']['neutral_checks']['coverage_exact']
        assert not record['check']['semantic_checks']['coverage_exact']
        assert record['candidate'] is None and len(calls) == 2


def test_unsupported_omission_wins_even_with_positive_whole_block_verdict(host):
    def omitted(v, payload):
        v['semantic_coverage'][-1].update(supported=False, answer_fragment='')
    with semantic_sdk(change=omitted) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_semantic_coverage_rejected'
        assert record['check']['neutral_checks']['supported_true']
        assert not record['check']['semantic_checks']['supported_true']
        assert record['candidate'] is None and len(calls) == 2


@pytest.mark.parametrize('fault', ['wrong_id', 'duplicate', 'extra', 'non_substring', 'empty', 'punctuation',
    'too_long', 'bool_number', 'bool_string', 'extra_row_key', 'missing_row_key', 'not_list', 'not_dict'])
def test_bad_coverage_cannot_supply_proof(host, fault):
    def change(v, payload):
        rows = v['semantic_coverage']
        if fault == 'wrong_id': rows[0]['id'] = 'model-invented-id'
        if fault == 'duplicate': rows[1] = copy.deepcopy(rows[0])
        if fault == 'extra': rows.append(copy.deepcopy(rows[0]))
        if fault == 'non_substring': rows[0]['answer_fragment'] = 'UNSEEN_PRIVATE_VERDICT_FRAGMENT'
        if fault == 'empty': rows[0]['answer_fragment'] = ''
        if fault == 'punctuation': rows[0]['answer_fragment'] = '。'
        if fault == 'too_long': rows[0]['answer_fragment'] = '字'*2049
        if fault == 'bool_number': rows[0]['supported'] = 1
        if fault == 'bool_string': rows[0]['supported'] = 'true'
        if fault == 'extra_row_key': rows[0]['reasoning'] = 'UNSEEN_PRIVATE_VERDICT_FRAGMENT'
        if fault == 'missing_row_key': rows[0].pop('supported')
        if fault == 'not_list': v['semantic_coverage'] = {}
        if fault == 'not_dict': rows[0] = 'UNSEEN_PRIVATE_VERDICT_FRAGMENT'
    with semantic_sdk(change=change) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_semantic_coverage_invalid'
        assert record['candidate'] is None and len(calls) == 2
        assert 'UNSEEN_PRIVATE_VERDICT_FRAGMENT' not in json.dumps(record)


@pytest.mark.parametrize('field', ['additional_claims', 'contradictions', 'omitted_limitations',
                                  'binding_hash', 'supported', 'covered_fact_ids'])
def test_existing_neutral_gates_still_apply_with_complete_semantic_proof(host, field):
    def change(v, payload):
        v[field] = False if field == 'supported' else 'wrong' if field == 'binding_hash' else ['wrong']
    with semantic_sdk(change=change) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected'
        assert record['candidate'] is None and len(calls) == 2


@pytest.mark.parametrize('fault', ['legacy_verdict', 'unknown_key'])
def test_new_profile_requires_exact_extended_protocol(host, fault):
    def change(v, payload):
        if fault == 'legacy_verdict': v.pop('semantic_coverage')
        else: v['reasoning'] = 'UNSEEN_PRIVATE_VERDICT_FRAGMENT'
    with semantic_sdk(change=change) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_semantic_protocol_invalid'
        assert 'UNSEEN_PRIVATE_VERDICT_FRAGMENT' not in json.dumps(record) and len(calls) == 2


def test_value_failure_still_precedes_semantic_model_and_cannot_start_a_second_attempt(host):
    def change(payload):
        out = full_body(payload)
        out['paragraphs'][0]['text'] = out['paragraphs'][0]['text'].replace('12.5 ms', '13 ms')
        return out
    with semantic_sdk(build=change) as (model, checker, calls):
        svc = semantic_service(host, model, checker)
        record = svc.prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1
        assert svc.prepare(source(), scope=SCOPE, check_current=lambda: '') == record and len(calls) == 1


@pytest.mark.parametrize('fault,reason', [('checker_missing', 'expression_neutral_check_required'),
    ('input_oversized', 'expression_input_oversized'),
    ('requirements_oversized', 'expression_semantic_requirements_oversized')])
def test_missing_checker_and_input_limit_do_not_launch_model_calls(host, fault, reason):
    with semantic_sdk() as (model, checker, calls):
        options = dict(max_input_chars=512) if fault == 'input_oversized' else {}
        value = file_source(constraints=['条件。'*129]) if fault == 'requirements_oversized' else file_source()
        svc = semantic_service(host, model, None if fault == 'checker_missing' else checker, **options)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '')['fallback_reason'] == reason
        assert not calls


def test_old_profile_hash_payload_verdict_and_archive_are_exactly_unchanged(host):
    with sdk_pair(full_body) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4)
        expected = dict(id=svc.profile.configuration_id, style=STYLE, runtime=model.identity, checker=checker.identity,
            timeout=10, check_timeout=10, input=24000, output=2048, format=2,
            paragraph_reference_version=1, paragraph_presentation_version=4, detail_policy_version=2)
        assert svc.profile.config_hash == canonical_hash(expected)
        assert svc.profile.config_hash == replace(svc.profile, semantic_review_version=0).config_hash
        record = svc.prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and record['check']['mode'] == 'protected_values_and_neutral'
        from shinku.result_expression.facts import identity
        assert record['binding_hash'] == canonical_hash({**identity(file_source()),
            'result_hash': file_source()['result_hash'], 'paragraph_reference_version':1,
            'paragraph_presentation_version':4, 'detail_policy_version':2})
        assert 'semantic_review_version' not in record
        assert 'semantic_checks' not in record['check']
        assert all('semantic_requirements' not in c[2] and 'semantic_review_version' not in c[2] for c in calls)
        assert len(calls) == 2
        upgraded = ExpressionService(svc.archive, replace(svc.profile, semantic_review_version=1))
        rejected = upgraded.prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert rejected['fallback_reason'] == 'expression_configuration_changed' and len(calls) == 2
        assert svc.archive.get(record['expression_id'], scope=SCOPE) == record


@pytest.mark.parametrize('version', [True, False, -1, 2, '1', None])
def test_semantic_profile_version_must_be_an_explicit_supported_integer(version):
    with pytest.raises(ValueError, match='invalid_expression_semantic_review'):
        ExpressionProfile(format_version=2, semantic_review_version=version)


def test_semantic_profile_requires_paragraph_format():
    with pytest.raises(ValueError, match='invalid_expression_semantic_review'):
        ExpressionProfile(semantic_review_version=1)


def test_syntactic_requirements_keep_full_text_and_do_not_promote_optional_audit():
    from shinku.result_expression.semantic_coverage import semantic_requirements
    blocks = [dict(id='verification', semantic_review_required=True, text='条件一；限定二。\n条件三;英文限定。'),
              dict(id='audit', required=False, semantic_review_required=True, text='无需重述的审计原件。'),
              dict(id='literal', required=True, text='精确值仍由已有保护值检查。')]
    before = copy.deepcopy(blocks)
    reason, rows = semantic_requirements(blocks)
    assert not reason and blocks == before
    assert [r['text'] for r in rows] == ['条件一；', '限定二。', '条件三;', '英文限定。']
    assert [r['id'] for r in rows] == ['verification:1', 'verification:2', 'verification:3', 'verification:4']


def test_oversized_requirement_set_fails_closed_without_truncation():
    from shinku.result_expression.semantic_coverage import semantic_requirements
    reason, rows = semantic_requirements([dict(id='many', semantic_review_required=True, text='条件。'*129)])
    assert reason == 'expression_semantic_requirements_oversized' and rows == []


@pytest.mark.parametrize('limitation', ['输出不能有预先固定的文本期望值。', '不得臆造具体输出，contains/hash不预设。'])
def test_recorded_whole_block_false_positive_cannot_copy_missing_limitation_as_body_proof(host, limitation):
    value = file_source(); value.pop('result_hash')
    value['contract']['requirements'][1]['description'] += limitation
    from shinku.agent_runtime.results import seal_result
    value = seal_result(value)
    def omit(payload):
        out = full_body(payload)
        out['paragraphs'][0]['text'] = out['paragraphs'][0]['text'].replace(limitation, '')
        return out
    with semantic_sdk(build=omit) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_semantic_coverage_invalid'
        assert record['check']['neutral_checks']['supported_true']
        assert not record['check']['semantic_checks']['fragments_match']
        assert limitation in record['text'] and record['candidate'] is None and len(calls) == 2


def test_semantic_proof_allows_a_real_paraphrase_without_literal_wording_rules(host):
    value = file_source(); value.pop('result_hash')
    limitation = '输出不能有预先固定的文本期望值。'
    paraphrase = '结果的文本没有预设答案。'
    value['contract']['requirements'][1]['description'] += limitation
    from shinku.agent_runtime.results import seal_result
    value = seal_result(value)
    def build(payload):
        out = full_body(payload)
        out['paragraphs'][0]['text'] = out['paragraphs'][0]['text'].replace(limitation, paraphrase)
        return out
    def proof(v, payload):
        requirements = {r['id']: r for r in payload['semantic_requirements']}
        for row in v['semantic_coverage']:
            if requirements[row['id']]['text'].startswith(limitation):
                row['answer_fragment'] = paraphrase
    with semantic_sdk(build=build, change=proof) as (model, checker, calls):
        record = semantic_service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready', record['fallback_reason']
        assert paraphrase in record['text'] and limitation not in record['text'] and len(calls) == 2


@pytest.mark.parametrize('semantic_version', [0, 1])
def test_interrupted_profile_retains_metadata_and_only_recovers_template(host, semantic_version):
    value = file_source()
    with semantic_sdk() as (model, checker, calls):
        svc = semantic_service(host, model, checker)
        svc = ExpressionService(svc.archive, replace(svc.profile, semantic_review_version=semantic_version))
        from shinku.result_expression.facts import identity as source_identity
        identity = source_identity(value)
        binding = {**identity, 'result_hash':value['result_hash'], 'paragraph_reference_version':1,
                   'paragraph_presentation_version':4, 'detail_policy_version':2}
        if semantic_version: binding['semantic_review_version'] = 1
        record, owned = svc.archive.claim(canonical_hash(identity), scope=SCOPE, result=value,
            config_hash=svc.profile.config_hash, binding_hash=canonical_hash(binding), processing_seconds=-1,
            format_version=2, detail_policy_version=2, paragraph_presentation_version=4,
            semantic_review_version=semantic_version)
        assert owned
        recovered = svc.prepare(value, scope=SCOPE, check_current=lambda: '', recover_interrupted=True)
        assert recovered['fallback_reason'] == 'expression_interrupted' and not calls
        assert recovered.get('semantic_review_version', 0) == semantic_version
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '', recover_interrupted=True) == recovered
