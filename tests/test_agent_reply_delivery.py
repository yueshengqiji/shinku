"""直接交付链路的离线验收：真实文件、Service/SQLite、SDK隔离传输和权限负例。"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import threading

import pytest

from shinku.agent_runtime import AgentTurnRequest, AgentScope
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.agent_runtime.reply_delivery import POLICY, delivery_turn
from shinku.angelos.host import ServiceHost
from shinku.tools.workspace import build_workspace_tools
from test_agent_conversation_loop import turn_env, native_sdk
from test_agent_runtime_boundary import settled, handle_from


def final(text='主管，结果和限制如下。', **kw):
    return {'kind': 'final', 'text': text, 'task_outcome': 'completed', **kw}


@contextmanager
def direct_env(root, responder, **kw):
    with turn_env(root, responder, reply_delivery=True, **kw) as env:
        yield env


def result_for(port, text='按原要求分析并交付', key='direct'):
    handle = port.submit(AgentTurnRequest(key, text))
    view = settled(port, handle.task_id)
    result = port.result(handle.task_id)
    assert result['verification'].get('layers', {}).get('policy') == POLICY, json.dumps(result, ensure_ascii=False)
    assert view['verification']['verified'] is False
    assert result['verification']['layers']['goal_completion'] == 'not_verified'
    assert result['delivery']['sent'] is False
    assert 'verified_answer' not in result
    return view, result


@pytest.mark.parametrize('question', ['解释精确匹配算法', 'Explain exact matching', '分析项目结构', '调查引用来源'])
@pytest.mark.parametrize('size', [60, 11000])
def test_complete_answer_no_contract_no_semantic_retries(tmp_path, question, size):
    text = '主管，以下是分析和建议。' + '说明性正文。' * (size // 6)
    with direct_env(tmp_path, lambda *args: final(text)) as (_, tool, _, port, calls):
        view, result = result_for(port, question)
        assert view['status'] == 'completed'
        assert result['execution_report']['text'] == text
        assert len(calls) == result['budget']['requests'] == 1 and tool.calls == 0
        assert result['contract'] is None
        assert '先用declare_task_contract声明完整验收契约' not in calls[0]['system_prompt']


def test_default_service_host_real_write_without_declaration_and_no_false_goal_pass(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    try:
        port = ServiceHost(service, requester='owner').port('one')
        view, result = result_for(port, '创建合成文件 ' + str(tool.path))
        assert view['status'] == 'completed' and result['budget']['requests'] == 2
        assert tool.calls == 1 and tool.path.read_text() == 'standalone agent verified'
        assert result['verification']['evidence'][0]['receipt_valid'] is True
        assert result['artifacts'][0]['verification'] == 'unconfirmed'
        assert result['contract'] is None
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('tamper', ['changed', 'missing', 'exception'])
def test_failed_receipt_refresh_retains_body_and_never_verifies_goal(tmp_path, tamper):
    def respond(kw, count, service, tool):
        if count == 1:
            return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        if tamper == 'changed':
            tool.path.write_text('external update')
        elif tamper == 'missing':
            tool.path.unlink()
        else:
            tool.agent_verify_result = lambda **kw: (_ for _ in ()).throw(OSError('synthetic'))
        return final('原操作有回执，当前状态需另行核对。')
    with direct_env(tmp_path, respond) as (_, tool, _, port, calls):
        view, result = result_for(port)
        assert view['status'] == 'completed' and len(calls) == 2 and tool.calls == 1
        assert result['verification']['evidence'][0]['receipt_valid'] is False
        assert result['execution_report']['text'] == '原操作有回执，当前状态需另行核对。'
        assert result['artifacts'][0]['verification'] == 'unconfirmed'


def test_fabricated_execution_claim_cannot_become_host_verified_success(tmp_path):
    with direct_env(tmp_path, lambda *args: final('我已删除文件并发送报告，测试通过。')) as (_, tool, _, port, calls):
        _, result = result_for(port)
        # 不再用正则改写/驳回答案，也绝不把模型原文升级为执行或送达事实。
        assert result['evidence'] == result['artifacts'] == result['verification']['evidence'] == []
        assert result['execution_report']['is_fact_evidence'] is False
        assert tool.calls == 0 and len(calls) == 1


def test_mutable_optional_contract_does_not_create_hard_conditions(tmp_path):
    plans = []
    def respond(kw, count, service, tool):
        if count <= 2:
            proposal = deepcopy(demo_request(tool).contract)
            proposal['requirements'][0]['id'] = 'first' if count == 1 else 'replacement'
            proposal['requirements'][0]['effect'] = 'model_suggestion'
            plans.append(proposal)
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': proposal}}
        return final('调整计划后直接交付已有分析。')
    with direct_env(tmp_path, respond) as (service, tool, _, port, calls):
        _, result = result_for(port)
        task = service._get(AgentScope('owner', 'angelos:one'), result['task_id'])
        assert result['contract'] is None and result['remaining'] == []
        assert [r['proposal'] for r in task['progress']['advisory_plans']] == plans
        assert len(calls) == 3 and tool.calls == 0


def test_invalid_optional_metadata_does_not_destroy_answer(tmp_path):
    with direct_env(tmp_path, lambda *args: final(task_contract={'notes': 'x' * 34000})) as (service, _, _, port, calls):
        view, result = result_for(port)
        assert view['status'] == 'completed' and len(calls) == 1
        assert service._get(AgentScope('owner', 'angelos:one'), result['task_id'])['progress']['advisory_metadata_error'] == 'TaskContractError'


@pytest.mark.parametrize('boundary', ['revoke', 'cancel', 'approval', 'budget', 'deny'])
def test_no_contract_does_not_relax_execution_boundaries(tmp_path, boundary):
    active = [True]; entered = threading.Event(); release = threading.Event()
    def respond(kw, count, service, tool):
        if boundary == 'cancel':
            entered.set(); assert release.wait(3)
        if boundary == 'revoke': active[0] = False
        if boundary == 'budget': return {'tool_call': {'type': 'read_self_state', 'scope': 'all'}}
        return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
    with direct_env(tmp_path, respond) as (service, tool, _, port, calls):
        original = service.authorize
        def authorize(*args):
            grant = original(*args)
            if boundary == 'revoke': return replace(grant, still_authorized=lambda: active[0])
            if boundary == 'approval': return replace(grant, policy=replace(grant.policy, risk_by_tool={'demo_file': 'high'}))
            if boundary == 'budget': return replace(grant, budget_limits={**grant.budget_limits, 'max_requests': 1})
            if boundary == 'deny': return replace(grant, allowed_tools=frozenset())
            return grant
        service.authorize = authorize
        h = port.submit(AgentTurnRequest(boundary, '创建文件'))
        if boundary == 'cancel':
            assert entered.wait(3)
            port.cancel(handle_from(port.status(h.task_id))); release.set()
        view = settled(port, h.task_id)
        assert view['status'] != 'completed' and tool.calls == 0 and not tool.path.exists()
        if boundary == 'budget': assert view['reason'] == 'task_model_budget_exhausted' and len(calls) == 1
        if boundary == 'approval': assert view['reason'] == 'approval_gate_unavailable' and len(calls) == 1
        if boundary == 'cancel': assert view['status'] == 'canceled'


@pytest.mark.parametrize('path', ['module.py', '../outside.txt'])
def test_actual_workspace_reader_without_contract_preserves_scope_and_receipt(tmp_path, path):
    (tmp_path/'module.py').write_text('def calculate():\n    return 7\n')
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'workspace_read_lines', 'path': path}}
        return final('函数返回常量；越界资料不可用。')
    with direct_env(tmp_path, respond) as (service, _, _, port, calls):
        for handler in build_workspace_tools(tmp_path, authorize=lambda ctx, write: not write):
            service.executor.tool_host.registry.register(handler)
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), allowed_tools=frozenset({'workspace_read_lines'}),
            policy=replace(original(*args).policy, risk_by_tool={'workspace_read_lines': 'low'}))
        view, result = result_for(port)
        assert view['status'] == 'completed' and len(calls) == 2
        receipt = result['verification']['evidence'][0]
        assert receipt['execution_status'] == ('ok' if path == 'module.py' else 'error')
        assert receipt['receipt_valid'] is (True if path == 'module.py' else None)
        assert (tmp_path/'module.py').read_text() == 'def calculate():\n    return 7\n'


def test_real_sdk_native_tool_history_has_one_execution_and_one_answer(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    def respond(payload, count):
        if count == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'native_write', 'type': 'function',
                'function': {'name': 'demo_file', 'arguments': json.dumps({'path': str(tool.path)})}}]}, 'tool_calls'
        assert any(m.get('role') == 'tool' for m in payload['messages'])
        return {'role': 'assistant', 'content': json.dumps(final('合成文件已写入。'))}, 'stop'
    try:
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
            port = ServiceHost(service, requester='owner').port('one')
            view, result = result_for(port)
            assert view['status'] == 'completed' and len(calls) == 2 and tool.calls == 1
            assert result['budget']['requests'] == 2
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_wire_flags_cannot_enable_policy():
    assert delivery_turn({'reply_delivery': True, '_agent_conversation_turn': {'reply_delivery': True}}) is None


def test_structured_request_on_same_host_keeps_old_verifier(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    try:
        port = ServiceHost(service, requester='owner').port('one')
        h = port.submit(demo_request(tool))
        assert settled(port, h.task_id)['verification']['verified'] is True
        assert port.result(h.task_id)['verification'].get('layers', {}).get('policy') != POLICY
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_stale_local_source_and_old_analysis_limits_do_not_reject_answer(tmp_path):
    """读取十三份不同文件后仍可继续；旧的十二次分析阈值不接管正常会话。"""
    for i in range(14): (tmp_path/f'{i}.py').write_text(f'value = {i}\n')
    def respond(kw, count, service, tool):
        if count <= 14:
            assert any(s['function']['name'] == 'workspace_read_lines' for s in kw['native_tools'])
            if count == 2: (tmp_path/'0.py').write_text('changed after observation')
            return {'tool_call': {'type': 'workspace_read_lines', 'path': f'{count-1}.py'}}
        return final('主管，已得到各文件的结构。第一份文件后来改变，其当前内容未核实。')
    with direct_env(tmp_path, respond) as (service, _, _, port, calls):
        for handler in build_workspace_tools(tmp_path, authorize=lambda ctx, write: not write):
            service.executor.tool_host.registry.register(handler)
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), allowed_tools=frozenset({'workspace_read_lines'}),
            policy=replace(original(*args).policy, risk_by_tool={'workspace_read_lines': 'low'}),
            budget_limits={**original(*args).budget_limits, 'max_tools': 20, 'max_requests': 20, 'max_tokens': 2000000})
        view, result = result_for(port, '分析各文件')
        assert view['status'] == 'completed' and len(calls) == 15
        assert result['verification']['evidence'][0]['receipt_valid'] is False
        assert all(row['receipt_valid'] is True for row in result['verification']['evidence'][1:])
        assert len(result['evidence']) == 14


def test_legacy_contract_survives_resume_as_history_not_execution_gate(tmp_path):
    from work_state_fixture import plan
    def respond(kw, count, service, tool):
        if count == 1:
            return {'kind': 'wait', 'text': '等补充信息。', 'task_contract': demo_request(tool).contract}
        if count == 2:
            return {'tool_call': {'type': 'update_work_plan', 'expected_revision': 0, 'plan': plan()}}
        return final('补充信息已收到，交付当前分析。', task_contract={'goal': '辅助计划已调整'})
    with turn_env(tmp_path, respond) as (service, tool, _, port, calls):
        h = port.submit(AgentTurnRequest('resume', '说明工作方案'))
        first = settled(port, h.task_id)
        assert first['status'] == 'waiting_user'
        original = deepcopy(service._get(AgentScope('owner', 'angelos:one'), h.task_id)['contract'])
        service.conversation_reply_delivery = True  # 模拟宿主升级后的显式恢复。
        port.resume(handle_from(first), request_key='continue')
        view = settled(port, h.task_id); result = port.result(h.task_id)
        assert view['status'] == 'completed' and not view['verification']['verified'], result
        assert result['contract'] == original and result['remaining'] == []
        assert result['verification']['layers']['policy'] == POLICY and tool.calls == 0 and len(calls) == 3
        work = service.task_store.get_work_state(h.task_id, requester='owner', session='angelos:one')
        assert work['purpose'] == 'advisory' and work['plan']['steps'][0]['requirements'] == []


@pytest.mark.parametrize('required', [False, True])
def test_recovery_ignores_expired_optional_history_but_keeps_required_input_gate(tmp_path, required):
    from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialStore
    with direct_env(tmp_path, lambda *args: {'kind': 'wait', 'text': '等待补充。'}, recovery=True) as (service, _, _, port, _):
        now = [1000.0]; service.task_store.clock = lambda: now[0]
        service.materials = TaskMaterialStore(service.task_store, policy=MaterialPolicy(ttl_seconds=10))
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), material_sources=(
            MaterialSource('history', 'related_message', '合成资料', required=required),))
        h = port.submit(AgentTurnRequest('expiry', '先说明下一步'))
        first = settled(port, h.task_id)
        now[0] += 11
        review = port.recovery(handle_from(first))
        assert review['status'] == ('review_required' if required else 'continuable'), review
        if required:
            # 保留源墓碑与已清理引用的存储实现返回不同原因，但必需资料都必须阻止恢复。
            assert {'material_source_expired_or_revoked', 'material_reference_unavailable'} & set(review['reasons'])


def test_advisory_boundary_failure_is_not_swallowed_as_metadata_error(tmp_path, monkeypatch):
    from shinku.agent_runtime.conversation import ConversationTurn
    from shinku.agent_runtime import AgentBoundaryError
    def denied(*args): raise AgentBoundaryError('conversation_authorization_changed')
    monkeypatch.setattr(ConversationTurn, 'adopt', denied)
    with direct_env(tmp_path, lambda *args: final(task_contract={'goal': 'test'})) as (_, tool, _, port, calls):
        h = port.submit(AgentTurnRequest('denied', '说明方案'))
        view = settled(port, h.task_id)
        assert view['status'] != 'completed' and view['reason'] == 'conversation_authorization_changed'
        assert tool.calls == 0 and len(calls) == 1


@pytest.mark.parametrize('configured', [False, True])
def test_stage_and_research_review_models_are_not_called(tmp_path, configured):
    from shinku.agent_runtime.stages import StagePolicy
    class ForbiddenReview:
        def call_chat_json(self, **kw): raise AssertionError('extra semantic call')
    with direct_env(tmp_path, lambda *args: final()) as (service, _, _, port, calls):
        service.executor.profile = replace(service.executor.profile, stage_policy=StagePolicy(lambda task: True),
            research_review_runtime=ForbiddenReview() if configured else None,
            research_review_profile_id='forbidden-review' if configured else '')
        view, _ = result_for(port)
        assert view['status'] == 'completed' and len(calls) == 1


def test_expression_cannot_promote_invalid_current_artifact_or_hide_error(tmp_path):
    from shinku.result_expression.facts import fact_blocks
    from shinku.result_expression.verified_conditions import _current_artifact
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        tool.path.write_text('changed externally')
        return final('已取得写入记录，但文件之后被外部改变。')
    with direct_env(tmp_path, respond) as (_, _, _, port, _):
        _, result = result_for(port)
        assert _current_artifact(result, result['artifacts'][0]) is None
        blocks = {r['id']: r for r in fact_blocks(result)}
        assert '当前目标验收通过' not in blocks['status']['text']
        assert blocks['receipt_check_0']['required'] is True
        assert 'false' in blocks['receipt_check_0']['text']


def test_source_discovery_does_not_trigger_legacy_answer_closeout(tmp_path):
    from types import SimpleNamespace
    from shinku.tools.workspace import tool_result
    class Search:
        tool_type = 'fixture_search'
        agent_permission_class = 'read'
        agent_progress_class = 'observe'
        agent_research_candidate_field = agent_evidence_collection_field = 'sources'
        def normalize_call(self, call): return call
        def tool_metadata(self): return SimpleNamespace(input_schema={'type': 'object',
            'properties': {'query': {'type': 'string'}}, 'required': ['query']})
        def execute(self, *, call, context):
            return tool_result(self.tool_type, {'sources': [{'url': 'https://example.invalid/reference',
                'snippet': '独立合成观察 ' + call['query']}]})
    def respond(kw, count, service, tool):
        if count <= 4:
            assert kw['native_tools'], '旧研究状态不得提前禁止全部工具'
            return {'tool_call': {'type': 'fixture_search', 'query': str(count)}}
        return final('只取得候选线索，正文未核实，以下是下一步建议。')
    with direct_env(tmp_path, respond) as (service, _, _, port, calls):
        service.executor.tool_host.registry.register(Search())
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), allowed_tools=frozenset({'fixture_search'}),
            policy=replace(original(*args).policy, risk_by_tool={'fixture_search': 'low'}))
        view, result = result_for(port)
        assert view['status'] == 'completed' and len(calls) == 5
        assert len(result['evidence']) == 4
