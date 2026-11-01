"""合成回放：权限来源、辅助材料额度及失败回复不得污染用户正文。"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.agent_runtime import ExecutionGrant
from shinku.agent_runtime.delegation_http import authority_payload, intersect_authority
from shinku.agent_runtime.handoff import delivery_result
from shinku.agent_runtime.results import seal_result
from shinku.agent_runtime.demo import DemoFileTool
from shinku.angelos.permissions import PermissionControls
from shinku.result_expression import ExpressionArchive, ExpressionService
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from test_expression_persona_body import profile, value, REPORT, rewritten
from test_expression_semantic_coverage import semantic_sdk
from test_result_expression import host, SCOPE


@pytest.mark.parametrize('source_mode,level,shell', [
    ('full', 'aleph', False), ('ultra', 'aleph', True), ('ultra', 'he', False),
    ('ultra', 'zayin', False), ('confirm', 'aleph', True), ('ultra', 'waw', True)])
def test_authority_intersection_keeps_origin_and_approval(tmp_path, source_mode, level, shell):
    names = frozenset({'file_read', 'shell'})
    base = ExecutionGrant('server', names, policy=ExecutionPolicy(allowed=names))
    caller = replace(base, authorization_ref='caller', permission_mode=source_mode,
        allowed_tools=names - {'shell'} if source_mode == 'full' else names,
        policy=ExecutionPolicy(allowed=names, approval_required_by_risk=frozenset({'low', 'medium', 'high'})
                               if source_mode == 'confirm' else frozenset()))
    controls = PermissionControls(tmp_path/'permission.json', default=level)
    reader, shell_tool = DemoFileTool(tmp_path), DemoFileTool(tmp_path)
    reader.agent_permission_class, reader.agent_completion_effects = 'read', ('read',)
    shell_tool.agent_permission_class, shell_tool.agent_completion_effects = 'change', ('command_run',)
    controls.registry = ToolRegistry({'file_read': reader, 'shell': shell_tool})
    grant = controls.restrict(intersect_authority(base, authority_payload(caller)), respect_host_approval=True)
    assert ('shell' in grant.allowed_tools) is shell
    assert grant.tool_context['delegation_authority']['source_mode'] == source_mode
    assert grant.permission_mode == ('readonly' if level in {'he', 'zayin'} else source_mode)
    if source_mode == 'full':
        assert grant.tool_context['delegation_authority']['source_excluded_tools'] == ['shell']
    if source_mode == 'confirm':
        assert set(grant.tool_context['delegation_approval_tools']) == names


def noisy_result():
    result = value(); result.pop('result_hash')
    result['execution'].update(status='partial', reason='partial_completion')
    result['verification'].update(accepted=False, missing=['未运行测试'])
    result['remaining'] = ['未运行测试']
    result['evidence'].append(dict(invocation_id='snapshot', tool='read_self_state',
        execution_status='ok', source='executor_receipt', receipt={}))
    result['facts'].append(dict(invocation_id='snapshot', source_index=0, source='executor_receipt',
        label='宿主状态快照（all）', value={'tools': {'registered_names': ['INTERNAL_SNAPSHOT'] * 100}}))
    result['information'] = dict(status='incomplete', required_status='complete', gaps=[],
        reason='legacy_source_completeness_unknown',
        snapshot=dict(gap_count=100, gaps=[dict(path='/evidence/0/receipt/tools/'+str(i),
            reason='depth_limit', kind='text', total=100, retained=0, omitted=100) for i in range(100)]))
    return seal_result(result)


@pytest.mark.parametrize('model_failure', [False, True])
def test_handoff_keeps_body_and_limits_but_not_snapshot_or_audit_spam(host, model_failure):
    original = noisy_result(); before = copy.deepcopy(original)
    delivered = delivery_result(original)
    assert all(r.get('invocation_id') != 'snapshot' for r in delivered['facts'])
    with semantic_sdk(build=(lambda p: {}) if model_failure else rewritten) as (model, checker, calls):
        svc = ExpressionService(ExpressionArchive(host), profile(model, checker, version=7))
        record = svc.prepare(delivered, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == ('fallback' if model_failure else 'ready'), record['fallback_reason']
        assert '未运行测试' in record['text'] and '12.5 ms' in record['text']
        assert '上游资料是否完整仍未确认' in record['text']
        assert all(s not in record['text'] for s in ['INTERNAL_SNAPSHOT', '/evidence/', '宿主状态快照', '资料投影审计'])
        assert len(record['text']) < 2500
        assert record['canonical_result'] == delivered and original == before
        assert record['canonical_result']['execution_report']['text'] == REPORT
        count = len(calls)
        assert svc.prepare(delivered, scope=SCOPE, check_current=lambda: '') == record
        assert len(calls) == count and record['delivery']['sent'] is False


def test_required_recovery_failure_stays_honest_and_does_not_invoke_model(host):
    original = delivery_result(noisy_result())
    projection = dict(schema_version=1, source_result_hash=original['result_hash'],
        status='incomplete', facts=original['facts'], sources=[],
        gaps=[{'reference': 'tm:synthetic', 'reason': 'material_version_conflict'}])
    original.pop('result_hash'); original['presentation']['material_recovery'] = True
    original = seal_result(original); projection['source_result_hash'] = original['result_hash']
    with semantic_sdk() as (model, checker, calls):
        svc = ExpressionService(ExpressionArchive(host), profile(model, checker, version=7),
            material_resolver=lambda *a, **kw: projection)
        record = svc.prepare(original, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_required_material_unavailable'
        assert not calls and '未能补读' in record['text'] and '未运行测试' in record['text']
        assert 'tm:synthetic' not in record['text'] and '/evidence/' not in record['text']
        assert record['presentation_projection']['gaps'] == projection['gaps']


def test_server_denial_is_not_misreported_as_source_denial():
    names = frozenset({'reader', 'shell'})
    base = ExecutionGrant('server', names, policy=ExecutionPolicy(blocked=frozenset({'shell'})))
    caller = replace(base, authorization_ref='caller', policy=ExecutionPolicy(), permission_mode='ultra')
    grant = intersect_authority(base, authority_payload(caller))
    assert grant.allowed_tools == {'reader'}
    assert grant.tool_context['delegation_authority']['source_excluded_tools'] == []
    assert grant.tool_context['delegation_authority']['server_excluded_tools'] == ['shell']


def test_full_does_not_open_ultra_tools_when_ordinary_allowlist_is_empty():
    from shinku.qq.agent_bridge import QQAgentBridge
    context = SimpleNamespace(policy=ExecutionPolicy(),
        _registry_handlers=lambda: {'shell': SimpleNamespace(agent_ultra_only=True)},
        _registered_tool_names=lambda: {'shell'})
    assert not QQAgentBridge._policy_for_permission_mode(context, 'full').accepts('shell')
    assert QQAgentBridge._policy_for_permission_mode(context, 'ultra').accepts('shell')
