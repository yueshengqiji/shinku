"""New prose context keeps original anchors, clauses, scope and old caches."""
import copy
from dataclasses import replace

import pytest

from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import STYLE, source
from test_expression_semantic_coverage import semantic_sdk
from test_expression_readable_context import query_source
from test_expression_verified_conditions import file_source
from test_result_expression import host, SCOPE


def profile(model, checker, version=5):
    return ExpressionProfile(model, checker=checker, format_version=2, configuration_id='connected-fixture',
        style_prompt=STYLE, detail_policy_version=2, paragraph_presentation_version=version,
        semantic_review_version=1)


@pytest.mark.parametrize('make', [source, file_source, query_source])
def test_new_context_preserves_every_required_anchor_and_whole_body_review(host, make):
    value = make(); before = copy.deepcopy(value)
    old = {b['id']: b for b in paragraph_blocks(value, detail_policy_version=2, presentation_version=4)}
    with semantic_sdk() as (model, checker, calls):
        service = ExpressionService(ExpressionArchive(host), profile(model, checker))
        result = service.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready', result.get('fallback_reason')
        assert result['canonical_result'] == before == value
        blocks = calls[0][2]['fact_blocks']
        assert calls[0][2]['required_fact_ids'] == [b['id'] for b in old.values() if b.get('required', True)]
        current = {b['id']: b for b in paragraph_blocks(value, detail_policy_version=2, presentation_version=5)}
        assert {k: b['anchors'] for k, b in current.items()} == {k: b['anchors'] for k, b in old.items()}
        assert [c[0] for c in calls] == ['expression', 'check']
        assert calls[0][2]['semantic_requirements'] == calls[1][2]['semantic_requirements']
        assert all(r['fact_id'] in calls[0][2]['required_fact_ids'] for r in calls[0][2]['semantic_requirements'])
        assert STYLE not in str(calls[1])
        assert result['check']['mode'] == 'protected_values_neutral_and_semantic_coverage'
        assert result['check']['semantic_checks']['supported_true'] is True
        assert not calls[0][1].get('tools') and not calls[1][1].get('tools')
        assert service.prepare(value, scope=SCOPE, check_current=lambda: '') == result
        assert len(calls) == 2


@pytest.mark.parametrize('information', [
    {'status':'complete','required_status':'complete','reason':'controlled_canonical_fixture_not_live_execution'},
    {'status':'incomplete','required_status':'complete','reason':'legacy_source_completeness_unknown','gaps':['原文尾部未提供']},
    {'status':'complete','required_status':'incomplete','gaps':['另一个输入尚未提供']},
    {'status':'unknown','required_status':'unknown','reason':'unknown_source_marker'},
])
def test_readable_handoff_keeps_complete_incomplete_unknown_and_original_gap(information):
    value = source(); value.pop('result_hash')
    from shinku.agent_runtime.results import seal_result
    value['information'] = information
    value = seal_result(value)
    rows = {r['id']:r for r in paragraph_blocks(value, detail_policy_version=2, presentation_version=5)}
    text = rows['information']['text']
    if information['status']=='complete': assert '已经交接的资料完整' in text
    if information['status']=='incomplete': assert '已经交接的资料有缺口' in text
    if information['required_status']=='incomplete': assert '所需资料仍有缺口' in text
    if information['status']=='unknown': assert '是否完整尚未确认' in text
    for gap in information.get('gaps', []): assert gap in text
    if information.get('reason')=='controlled_canonical_fixture_not_live_execution':
        assert '未执行线上操作' in text
    assert rows['information']['required'] and rows['information']['semantic_review_required']


@pytest.mark.parametrize('fault', ['omitted_clause','unsupported_clause','fabricated_fragment','unsupported_claim'])
def test_supported_overall_label_cannot_replace_the_original_semantic_or_fact_gate(host, fault):
    def change(value, payload):
        if fault=='omitted_clause': value['semantic_coverage'].pop()
        elif fault=='unsupported_clause': value['semantic_coverage'][0]['supported']=False
        elif fault=='fabricated_fragment': value['semantic_coverage'][0]['answer_fragment']='MODEL_ADDED_PROOF'
        else: value['additional_claims']=['The service was deployed.']
    with semantic_sdk(change=change) as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker)).prepare(
            file_source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'fallback' and result['check']['accepted'] is False
        assert len(calls) == 2 and result['delivery']['sent'] is False


def test_new_profile_is_separate_while_old_expression_and_cache_stay_bound(host):
    value = file_source()
    with semantic_sdk() as (model, checker, calls):
        old = ExpressionService(ExpressionArchive(host), profile(model, checker, 4))
        first = old.prepare(value, scope=SCOPE, check_current=lambda: '')
        new = ExpressionService(ExpressionArchive(host), profile(model, checker))
        second = new.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert first['status'] == 'ready'
        assert second['status'] == 'discarded' and second['fallback_reason'] == 'expression_configuration_changed'
        assert old.profile.config_hash != new.profile.config_hash
        assert first['canonical_result'] == value
        assert old.prepare(value, scope=SCOPE, check_current=lambda: '') == first
        assert new.prepare(value, scope=SCOPE, check_current=lambda: '') == second
        assert len(calls) == 2
        assert not first['delivery']['sent']


def test_new_context_requires_clause_review_and_cannot_be_enabled_implicitly():
    default = ExpressionProfile()
    assert default.paragraph_presentation_version == 1 and default.semantic_review_version == 0
    with pytest.raises(ValueError):
        replace(default, format_version=2, detail_policy_version=2, paragraph_presentation_version=5)
    valid = replace(default, format_version=2, detail_policy_version=2, paragraph_presentation_version=5,
                    semantic_review_version=1)
    assert valid.paragraph_presentation_version == 5


def test_connected_prompt_supplies_the_complete_paragraph_protocol(host):
    with semantic_sdk() as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker)).prepare(
            source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready'
        system = calls[0][1]['messages'][0]['content']
        assert '{"schema_version":2,"binding_hash":' in system
        assert '"paragraphs":[{"text":' in system
        assert '不能省略schema_version' in system


@pytest.mark.parametrize('version', [4, 5])
def test_missing_version_is_still_rejected_before_checker(host, version):
    from test_expression_semantic_coverage import full_body
    def missing_version(payload):
        value = full_body(payload)
        value.pop('schema_version')
        return value
    with semantic_sdk(build=missing_version) as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker, version)).prepare(
            source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'fallback'
        assert result['fallback_reason'] == 'expression_protocol_invalid'
        assert [c[0] for c in calls] == ['expression']


@pytest.mark.parametrize('version', [4, 5])
def test_only_connected_checker_explicitly_checks_added_causal_relationships(host, version):
    with semantic_sdk() as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker, version)).prepare(
            source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready'
        system = calls[1][1]['messages'][0]['content']
        assert ('同时出现不证明因果关系' in system) is (version == 5)
        assert ('真实来源明确给出的原因可以保留' in system) is (version == 5)
        assert ('不能只检查事实是否分别出现' in system) is (version == 5)


def test_checker_rule_change_cannot_reuse_an_old_connected_verdict(host, monkeypatch):
    from shinku.result_expression import natural
    value = source()
    with semantic_sdk() as (model, checker, calls):
        svc = ExpressionService(ExpressionArchive(host), profile(model, checker))
        first = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert first['status'] == 'ready'
        monkeypatch.setattr(natural, 'CONNECTED_PROSE_CHECK_RULES',
                            natural.CONNECTED_PROSE_CHECK_RULES + '新的事实关系核对规则。')
        current = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert current['status'] == 'discarded'
        assert current['fallback_reason'] == 'expression_configuration_changed'
        assert len(calls) == 2


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_connected_rule_change_keeps_earlier_profile_hashes(monkeypatch, version):
    from shinku.result_expression import natural
    config = profile(None, None, version)
    before = config.config_hash
    monkeypatch.setattr(natural, 'CONNECTED_PROSE_CHECK_RULES', '测试中的新规则。')
    assert config.config_hash == before
