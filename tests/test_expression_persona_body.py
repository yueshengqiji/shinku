"""执行报告改写的合成SDK回归；证明协议与边界，不充当真实人格质量评估。"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from shinku.result_expression.natural import paragraph_blocks, host_reference_ids
from test_expression_paragraphs import STYLE, source
from test_expression_semantic_coverage import semantic_sdk
from test_result_expression import host, SCOPE


REPORT = ('主管，测试已经结束。网页返回 HTTP 200，响应为 12.5 ms。'
          '来源 https://example.invalid/page ，报告位于 /workspace/项目 报告.md 。'
          '原文「仅限样本，不代表线上性能」。尚未登录，不能确认播放可用。\n'
          '```python\nvalue = 17\nprint(value)\n```')


def value(body=REPORT, *, verbatim=False):
    result = source()
    result.pop('result_hash')
    result['presentation']['handoff_version'] = 1
    result['execution_report'] = {'text': body, 'is_fact_evidence': False, 'source': 'execution_model'}
    result['original_requests'] = [{'text': '试试新浏览能力。' if not verbatim else '把执行报告逐字原样发给我。'}]
    return seal_result(result)


def profile(model, checker, version=6):
    return ExpressionProfile(model, checker=checker, format_version=2, detail_policy_version=2,
        paragraph_presentation_version=version, semantic_review_version=1, style_prompt=STYLE)


def rewritten(payload):
    rows = [r for r in payload['fact_blocks'] if r.get('required', True)]
    paragraphs = []
    for row in rows:
        text = row['text']
        if row['id'] == 'execution_body':
            text = ('我试过了：网页返回 HTTP 200，响应为 12.5 ms，来源是 https://example.invalid/page 。'
                    '报告放在 /workspace/项目 报告.md ，原文「仅限样本，不代表线上性能」。'
                    '这次尚未登录，所以不能确认播放可用。\n```python\nvalue = 17\nprint(value)\n```')
        paragraphs.append({'text': '本次记录如下。' + text, 'fact_ids': [row['id']]})
    return {'schema_version': 2, 'binding_hash': payload['binding_hash'], 'paragraphs': paragraphs}


def test_report_is_rewriteable_but_original_and_full_checker_context_are_retained(host):
    original = value(); before = copy.deepcopy(original)
    with semantic_sdk(build=rewritten) as (model, checker, calls):
        svc = ExpressionService(ExpressionArchive(host), profile(model, checker))
        result = svc.prepare(original, scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready', result['fallback_reason']
        assert '主管' not in result['text'] and '我试过了' in result['text']
        assert result['canonical_result'] == original == before
        assert result['canonical_result']['execution_report']['text'] == REPORT
        assert [r[0] for r in calls] == ['expression', 'check']
        for _, _, payload in calls:
            body = next(b for b in payload['fact_blocks'] if b['id'] == 'execution_body')
            assert body['text'] == REPORT and body['execution_report_projection']
            assert any(r.get('coverage_scope') == 'whole_report' for r in payload['semantic_requirements'])
        assert 'execution_body' not in calls[0][2]['host_reference_fact_ids']
        assert STYLE in calls[0][1]['messages'][0]['content']
        assert STYLE not in json.dumps(calls[1])
        assert '执行方的自称' in calls[0][1]['messages'][0]['content']
        assert svc.read_details(result['expression_id'], scope=SCOPE, check_current=lambda: '')['status'] == 'available'
        assert svc.prepare(original, scope=SCOPE, check_current=lambda: '') == result
        assert len(calls) == 2 and result['delivery']['status'] == 'not_sent'


@pytest.mark.parametrize('old,new', [('200', '1200'), ('12.5 ms', '12.6 ms'),
    ('https://example.invalid/page', 'https://example.invalid/other'),
    ('/workspace/项目 报告.md', '/workspace/另一个.md'),
    ('仅限样本，不代表线上性能', '代表线上性能'), ('value = 17', 'value = 18')])
def test_rewrite_cannot_change_exact_values_code_or_quotes(host, old, new):
    def changed(payload):
        answer = rewritten(payload)
        for row in answer['paragraphs']:
            if row['fact_ids'] == ['execution_body']:
                row['text'] = row['text'].replace(old, new)
        return answer
    with semantic_sdk(build=changed) as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker)).prepare(
            value(), scope=SCOPE, check_current=lambda: '')
        assert result['fallback_reason'] == 'expression_protected_value_missing'
        assert result['candidate'] is None and len(calls) == 1
        assert REPORT in result['text']


def test_long_executor_prose_cannot_bypass_rewrite_with_host_reference(host):
    original = value(REPORT + '\n普通执行叙述。' * 300)
    old = paragraph_blocks(original, detail_policy_version=2, presentation_version=5)
    new = paragraph_blocks(original, detail_policy_version=2, presentation_version=6)
    assert 'execution_body' in host_reference_ids(old)
    assert 'execution_body' not in host_reference_ids(new)
    def unchanged_reference(payload):
        rows = rewritten(payload)['paragraphs']
        rows[0:0] = [{'fact': 'execution_body'}]
        return {'schema_version': 2, 'binding_hash': payload['binding_hash'], 'paragraphs': rows}
    with semantic_sdk(build=unchanged_reference) as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker)).prepare(
            original, scope=SCOPE, check_current=lambda: '')
        assert result['fallback_reason'] == 'expression_host_reference_forbidden'
        assert original['execution_report']['text'] in result['text'] and len(calls) == 1


def test_explicit_verbatim_request_and_independent_long_quote_keep_exact_rule():
    original = value('原稿。' * 400, verbatim=True)
    blocks = paragraph_blocks(original, detail_policy_version=2, presentation_version=6)
    body = next(b for b in blocks if b['id'] == 'execution_body')
    assert not body.get('execution_report_projection')
    assert body['anchors'][0]['text'] == original['execution_report']['text']
    assert 'execution_body' in host_reference_ids(blocks)


def test_semantic_checker_still_rejects_omitted_report_limit(host):
    def reject(verdict, payload):
        for row in verdict['semantic_coverage']:
            if row['id'] == 'execution_body:meaning':
                row.update(supported=False, answer_fragment='')
    with semantic_sdk(build=rewritten, change=reject) as (model, checker, calls):
        result = ExpressionService(ExpressionArchive(host), profile(model, checker)).prepare(
            value(), scope=SCOPE, check_current=lambda: '')
        assert result['fallback_reason'] == 'expression_semantic_coverage_rejected'
        assert result['candidate'] is None and len(calls) == 2
        assert REPORT in result['text']


@pytest.mark.parametrize('required', [False, True])
def test_optional_gap_ids_are_audit_details_required_gap_stays_strict(required):
    original = value(); original.pop('result_hash')
    original['information'] = {'status': 'incomplete', 'required_status': 'complete',
        'reason': 'legacy_source_completeness_unknown',
        'gaps': [{'source_id': 'tool:SYNTHETIC', 'reference': 'tm:synthetic',
                  'reason': 'material_read_pending', 'remaining_chars': 213, 'required': required}]}
    original = seal_result(original)
    blocks = {r['id']: r for r in paragraph_blocks(original, detail_policy_version=2, presentation_version=6)}
    assert 'tm:synthetic' in blocks['information_audit']['text']
    if required:
        assert 'tm:synthetic' in blocks['information']['text']
    else:
        assert not blocks['information']['anchors'] and 'tm:synthetic' not in blocks['information']['text']
        assert '可选来源未完整读取' in blocks['information']['text']
        assert '上游资料是否完整仍未确认' in blocks['information']['text']


def test_profile_version_is_explicit_and_old_cache_is_not_rewritten(host):
    with pytest.raises(ValueError):
        ExpressionProfile(format_version=2, detail_policy_version=2, paragraph_presentation_version=6)
    with semantic_sdk(build=rewritten) as (model, checker, calls):
        old = ExpressionService(ExpressionArchive(host), profile(model, checker, 5))
        first = old.prepare(source(), scope=SCOPE, check_current=lambda: '')
        new = ExpressionService(old.archive, replace(old.profile, paragraph_presentation_version=6))
        second = new.prepare(source(), scope=SCOPE, check_current=lambda: '')
        assert second['fallback_reason'] == 'expression_configuration_changed'
        assert old.archive.get(first['expression_id'], scope=SCOPE) == first
        assert len(calls) == 2


@pytest.mark.parametrize('kind', ['unknown_reason', 'hidden_gap', 'export_projection', 'unknown_fields'])
def test_gap_projection_never_hides_unknown_or_additional_material_loss(kind):
    original = value(); original.pop('result_hash')
    gap = {'source_id': 'SYNTHETIC', 'reference': 'tm:synthetic', 'required': False,
           'reason': 'material_read_pending', 'remaining_chars': 213}
    original['information'] = {'status': 'incomplete', 'required_status': 'complete', 'gaps': [gap]}
    if kind == 'unknown_reason':
        gap['reason'] = 'SYNTHETIC_UNKNOWN_REASON'
    elif kind == 'hidden_gap':
        # seal_result会重算当前导出projection；历史快照的丢失记录应放snapshot，
        # 不能伪造一个随后会被正确覆盖的projection再声称测到了缺口。
        original['information']['snapshot'] = {'gap_count': 1, 'omitted_gap_records': 1}
    elif kind == 'export_projection':
        original['unverified_blob'] = 'x' * 10000  # 真实触发当前导出的有界投影丢失。
    else:
        gap['unknown_condition'] = '不得遗漏的限制'
    blocks = {b['id']: b for b in paragraph_blocks(seal_result(original), detail_policy_version=2,
                                                presentation_version=6)}
    assert 'tm:synthetic' in blocks['information']['text']
    assert blocks['information']['anchors']


def test_path_without_extension_cannot_stall_report_projection():
    # 子进程设置有界超时，旧正则回溯失控也不拖死整个回归或业务进程。
    # PYTHONPATH显式使用本轮检出，不能碰巧验证到宿主安装的旧模块。
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), SHINKU_ENV_FILE='/dev/null')
    result = subprocess.run([sys.executable, '-B', '-c',
        "from shinku.result_expression.persona_body import report_anchors; "
        "p='/segment'*100; a=report_anchors(p); assert a and a[0]['text']==p"],
        cwd=root, env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
