"""持久计划的确定性验收；真实 Service/SQLite/文件适配器，不购买模型调用。"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from shinku.agent.task_state import AgentTaskStore, TaskContractError, new_task
from shinku.agent.work_state import digest, input_binding, validate_plan
from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentTurnRequest
from shinku.agent_runtime.demo import demo_request
from shinku.angelos.host import ServiceHost
from work_state_fixture import plan, record
from test_agent_conversation_loop import turn_env, native_sdk, observed_contract
from test_agent_runtime_boundary import settled, handle_from




def update(value=None, revision=0):
    return {'tool_call': {'type': 'update_work_plan', 'expected_revision': revision, 'plan': value or plan()}}


def saved_task(service, task_id):
    return service.task_store.get(task_id, requester='owner', session='angelos:one')


def history(service, task_id, revision=None):
    return service.task_store.get_work_state(task_id, requester='owner', session='angelos:one', revision=revision)


@pytest.mark.parametrize('materials', [False, True])
def test_optional_plan_is_same_loop_and_never_business_success(tmp_path, materials):
    def respond(kw, count, service, tool):
        if count == 1:
            assert 'update_work_plan' in str(kw['native_tools'])
            return update()
        assert 'Host-bound advisory work state' in kw['user_prompt']
        assert '核实原问题' in kw['user_prompt']
        from shinku.agent_runtime.conversation import conversation_turn
        context = next(iter(service._sessions.values())).context
        assert conversation_turn(context).work_plan.snapshot(context)['status'] == 'active'
        return {'kind': 'final', 'text': '先解释当前方案，具体原因还需核实。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, materials=materials) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('optional', '讨论一个后续排查方案'))
        view = settled(port, h.task_id); result = port.result(h.task_id)
        assert view['status'] == 'completed', result
        assert len(calls) == result['budget']['requests'] == 2
        assert not view['verification']['verified'] and result['contract'] is None
        stored = saved_task(service, h.task_id)
        assert stored['work_state']['revision'] == 1 and 'steps' not in stored['work_state']
        assert stored['progress']['stalled'] == 1
        assert stored['progress']['last_event']['progress'] is False
        assert stored['progress']['last_event']['verified_change'] is False
        assert 'plan' not in stored['tool_calls'][0]['data']['state_updates']
        assert 'plan' not in stored['tool_calls'][0]['arguments']
        assert stored['tool_calls'][0]['arguments']['plan_reference'] == stored['work_state']
        assert history(service, h.task_id)['plan'] == plan() and tool.calls == 0


def test_plan_plus_native_contract_real_write_and_receipt(tmp_path):
    from shinku.agent_runtime.demo import demo_service
    service, tool, _ = demo_service(tmp_path)
    def responder(payload, count):
        if count == 1:
            name, args = 'update_work_plan', {'expected_revision': 0, 'plan': plan()}
        elif count == 2:
            assert 'Host-bound advisory work state' in json.dumps(payload)
            name, args = 'declare_task_contract', {'task_contract': demo_request(tool).contract}
        elif count == 3:
            name, args = 'demo_file', {'path': str(tool.path)}
        else:
            assert tool.calls == 1
            return {'role': 'assistant', 'content': json.dumps({'kind': 'final', 'text': '合成文件已创建并核验。',
                'task_outcome': 'completed'})}, 'stop'
        return {'role': 'assistant', 'content': '', 'reasoning_content': 'SYNTHETIC_REASONING', 'tool_calls': [
            {'id': 'native_' + str(count), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
    try:
        with native_sdk(responder) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
            port = ServiceHost(service, requester='owner').port('one')
            h = port.submit(AgentTurnRequest('native-plan', '创建 ' + str(tool.path)))
            assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
            result = port.result(h.task_id)
            assert len(calls) == result['budget']['requests'] == 4 and result['budget']['reported_tokens'] == 64
            assert tool.calls == 1 and result['verification']['evidence'][-1]['receipt_valid'] is True
            assert result['verification']['layers']['goal_completion'] == 'not_verified'
            for i, payload in enumerate(calls):
                tools = [m for m in payload['messages'] if m['role'] == 'tool']
                assistants = [m for m in payload['messages'] if m['role'] == 'assistant']
                assert len(tools) == len(assistants) == i
                assert [m['tool_call_id'] for m in tools] == [m['tool_calls'][0]['id'] for m in assistants]
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('bad', ['unknown_field', 'cycle', 'unknown_current', 'missing_requirement', 'fake_reference', 'oversized', 'sensitive'])
def test_invalid_plan_cannot_weaken_contract_or_create_evidence(tmp_path, bad):
    def respond(kw, count, service, tool):
        if count == 1:
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        if count == 2:
            value = plan(['file'], status='done')
            if bad == 'unknown_field': value['permissions'] = 'full'
            if bad == 'cycle': value['steps'][0]['depends'] = ['inspect']
            if bad == 'unknown_current': value['current_step'] = 'absent'
            if bad == 'missing_requirement': value['steps'][0]['requirements'] = []
            if bad == 'fake_reference': value['steps'][0]['references'] = [{'kind': 'tool_call', 'id': 'other-session-call'}]
            if bad == 'oversized': value['steps'][0]['goal'] = 'x' * 1001
            if bad == 'sensitive': value['findings'] = ['api_key=synthetic_secret']
            return update(value)
        return {'kind': 'wait', 'text': '原条件尚未完成，不能宣称文件已创建。'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest(bad, '创建 ' + str(tool.path)))
        assert settled(port, h.task_id)['status'] != 'completed'
        stored = saved_task(service, h.task_id)
        assert stored['contract'] == demo_request(tool).contract and history(service, h.task_id) is None
        assert not tool.path.exists() and tool.calls == 0 and len(calls) == 3


def test_model_done_and_internal_tool_requirement_never_satisfy_real_execution(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return update(plan(status='done'))
        if count == 2:
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': observed_contract('update_work_plan')}}
        if count == 3:
            assert 'contract_declaration_is_not_execution_evidence' in kw['user_prompt']
            return {'kind': 'final', 'text': '文件已创建成功。', 'task_outcome': 'completed', 'task_contract': demo_request(tool).contract}
        return {'kind': 'wait', 'text': '没有文件写入回执，任务尚未完成。'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('false-done', '创建 ' + str(tool.path)))
        assert settled(port, h.task_id)['status'] != 'completed'
        assert not port.result(h.task_id)['verification']['accepted'] and tool.calls == 0
        assert history(service, h.task_id)['plan']['steps'][0]['status'] == 'done'


def test_plan_revisions_preserved_and_stale_update_rejected(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return update()
        if count == 2:
            value = plan(); value['reason'] = '阶段调整'; value['blocker'] = '具体来源缺失'
            return update(value, 1)
        if count == 3:
            value = plan(); value['reason'] = '旧版本不能覆盖'
            return update(value, 0)
        assert 'work_plan_revision_conflict' in kw['user_prompt']
        return {'kind': 'wait', 'text': '具体来源缺失，请补充对应输入。'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('versions', '讨论持续排查方案'))
        assert settled(port, h.task_id)['status'] == 'waiting_user'
        assert history(service, h.task_id, 1)['plan'] == plan()
        assert history(service, h.task_id)['revision'] == 2
        assert history(service, h.task_id)['plan']['blocker'] == '具体来源缺失'
        assert saved_task(service, h.task_id)['progress']['stalled'] == 3
        assert service.task_store.get_work_state(h.task_id, requester='other', session='angelos:one') is None
        assert service.task_store.get_work_state(h.task_id, requester='owner', session='other') is None








def test_restart_recovers_advisory_state_budget_originals_and_never_replays_write(tmp_path):
    from test_agent_recovery import service_at, Runtime, close
    service, _, tool, runtime = service_at(tmp_path)
    runtime.decisions = [update(),
        {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}},
        {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}, RuntimeError('synthetic_first_failure')]
    port = ServiceHost(service, requester='owner').port('one')
    h = port.submit(AgentTurnRequest('restart-plan', '创建 ' + str(tool.path)))
    first = settled(port, h.task_id); old = saved_task(service, h.task_id); revision = history(service, h.task_id)
    assert first['status'] == 'failed' and tool.calls == 1 and revision['revision'] == 1
    close(service)
    service, _, fresh_tool, runtime = service_at(tmp_path)
    def respond(**kw):
        runtime.calls.append(kw)
        assert 'requires_review' in kw['user_prompt'] and '核实原问题' in kw['user_prompt']
        return {'kind': 'final', 'text': '合成文件已按原回执核验，恢复没有重复写入。', 'task_outcome': 'completed'}
    runtime.call_chat_json = respond
    try:
        port = ServiceHost(service, requester='owner').port('one')
        current = handle_from(port.status(h.task_id))
        assert port.recovery(current)['status'] == 'continuable'
        port.resume(current, request_key='resume-once')
        view = settled(port, h.task_id); result = port.result(h.task_id)
        assert view['status'] == 'completed', result
        assert history(service, h.task_id) == revision and fresh_tool.calls == 0
        assert result['budget']['requests'] == first['budget']['requests'] + 1
        assert result['original_requests'] == old['requests'] and result['contract'] == old['contract']
        assert saved_task(service, h.task_id)['recovery_history'][0]['original_stop_reason']
    finally:
        close(service)


def test_user_supplement_invalidates_plan_view_but_preserves_all_revisions(tmp_path):
    from shinku.agent_runtime import AgentSupplement
    from test_agent_recovery import service_at, Runtime, close
    service, _, tool, runtime = service_at(tmp_path, Runtime([update(), {'kind': 'wait', 'text': '请补充来源。'}]), materials=True)
    try:
        port = ServiceHost(service, requester='owner').port('one')
        h = port.submit(AgentTurnRequest('supplement-plan', '讨论排查方案'))
        assert settled(port, h.task_id)['status'] == 'waiting_user'
        prior = history(service, h.task_id)
        h = port.supplement(handle_from(port.status(h.task_id)), AgentSupplement('extra', clarification='仅允许离线分析，原文件不能改动'))
        def respond(**kw):
            runtime.calls.append(kw)
            assert 'requires_review' in kw['user_prompt'] and '原文件不能改动' in kw['user_prompt']
            return {'kind': 'final', 'text': '按补充要求只讨论离线排查方案。', 'task_outcome': 'completed'}
        runtime.call_chat_json = respond
        port.resume(h, request_key='continue-supplement')
        assert settled(port, h.task_id)['status'] == 'completed'
        assert history(service, h.task_id) == prior
        assert saved_task(service, h.task_id)['requests'][-1]['text'] == '仅允许离线分析，原文件不能改动'
    finally:
        close(service)


@pytest.mark.parametrize('boundary', ['denied', 'revoke', 'cancel', 'budget'])
def test_plan_updates_obey_current_permission_cancel_and_original_budget(tmp_path, boundary):
    import threading
    active = [True]; entered = threading.Event(); release = threading.Event()
    def respond(kw, count, service, tool):
        if boundary == 'revoke': active[0] = False
        if boundary == 'cancel': entered.set(); assert release.wait(3)
        if boundary == 'budget' and count == 2:
            value = plan(); value['reason'] = '预算不能重置'
            return update(value, 1)
        if count == 1: return update()
        return {'kind': 'wait', 'text': '当前权限不允许继续更新。'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        original = service.authorize
        def authorize(*args):
            grant = original(*args)
            if boundary == 'denied': return replace(grant, policy=replace(grant.policy, blocked=frozenset({'update_work_plan'})))
            if boundary == 'revoke': return replace(grant, still_authorized=lambda: active[0])
            if boundary == 'budget': return replace(grant, budget_limits={**grant.budget_limits, 'max_requests': 1})
            return grant
        service.authorize = authorize
        h = port.submit(AgentTurnRequest(boundary, '讨论方案'))
        if boundary == 'cancel':
            assert entered.wait(3)
            port.cancel(handle_from(port.status(h.task_id))); release.set()
        view = settled(port, h.task_id)
        assert tool.calls == 0 and not tool.path.exists()
        if boundary == 'budget':
            assert view['budget']['requests'] == len(calls) == 1
            assert view['reason'] == 'task_model_budget_exhausted' and history(service, h.task_id)['revision'] == 1
        else:
            assert history(service, h.task_id) is None
        if boundary == 'cancel': assert view['status'] == 'canceled'
        if boundary == 'revoke': assert view['reason'] == 'agent_permission_changed'


def test_true_tool_reference_is_host_bound_and_old_receipt_is_not_new_execution(tmp_path):
    from test_angelos_initiative import seed
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}}
        if count == 2:
            task = next(iter(service._sessions.values())).context['agent_task_state']
            ref = {'kind': 'tool_call', 'id': task['tool_calls'][0]['id']}
            return update(plan(references=[ref]))
        assert 'requires_revalidation' in kw['user_prompt']
        return {'kind': 'final', 'text': '旧记录的具体错误是 synthetic_missing_input，未重跑旧任务。',
            'task_outcome': 'completed', 'task_contract': observed_contract('read_task_records')}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        old = seed(service, 'old')
        h = port.submit(AgentTurnRequest('ref', '查明之前任务的具体错误'))
        assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
        task = saved_task(service, h.task_id); ref = history(service, h.task_id)['references'][0]
        assert ref['receipt_sha256'] == digest(task['tool_calls'][0])
        assert ref['tool'] == 'read_task_records' and len(calls) == 3 and tool.calls == 0
        assert service.task_store.get('old', requester='owner', session='angelos:one') == old


@pytest.mark.parametrize('changed', [False, True])
def test_material_reference_uses_real_source_version_and_keeps_read_ledger(tmp_path, changed):
    from shinku.agent_runtime.materials import MaterialSource
    source = tmp_path / 'source.txt'; source.write_text('合成输入，原文必须保留。')
    def respond(kw, count, service, tool):
        if count == 1:
            task = next(iter(service._sessions.values())).context['agent_task_state']
            ref = task['material_package']['items'][0]['reference']
            if changed: source.write_text('外部修改后的合成输入。')
            return update(plan(references=[{'kind': 'material', 'id': ref}]))
        return {'kind': 'wait', 'text': '原文保留，具体处理待确认。'}
    with turn_env(tmp_path, respond, materials=True) as (service, tool, host, port, calls):
        service.materials.approved_roots = (tmp_path.resolve(),)
        original = service.authorize
        service.authorize = lambda *a: replace(original(*a), material_sources=(MaterialSource('source', 'attachment', local_path=source),))
        h = port.submit(AgentTurnRequest('material-ref', '讨论来源的处理方案'))
        assert settled(port, h.task_id)['status'] != 'completed'
        revision = history(service, h.task_id)
        if changed:
            assert revision is None
        else:
            task = saved_task(service, h.task_id); ref = revision['references'][0]
            assert ref['content_sha256'] == task['material_package']['items'][0]['content_sha256']
            assert service.materials.resolve(ref['id'], task=task, still_authorized=lambda: True) == '合成输入，原文必须保留。'
        assert tool.calls == 0


def test_plan_churn_cannot_reset_stall_or_buy_unbounded_tools(tmp_path):
    def respond(kw, count, service, tool):
        value = plan(); value['reason'] = '仍在改写计划' + str(count)
        return update(value, count - 1)
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('churn', '讨论方案'))
        view = settled(port, h.task_id); task = saved_task(service, h.task_id)
        assert view['status'] != 'completed' and len(calls) <= 8 and tool.calls == 0
        assert task['progress']['stalled'] > 1
        assert all(not c.get('data', {}).get('state_updates', {}).get('changed') for c in task['tool_calls'])


def test_shared_stage_dependency_validator_preserves_original_goal_checks():
    from shinku.agent_runtime.stages import StagePolicy, task_binding, validate_plan as legacy_validate
    from shinku.agent_runtime.demo import demo_request, DemoFileTool
    task = {'contract': {'requirements': [{'id': 'a'}]}}
    value = {'binding': task_binding(task), 'reason': '测试', 'stages': [
        {'id': 'first', 'goal': '', 'depends': [], 'requirements': ['a']}]}
    with pytest.raises(ValueError, match='invalid_stage_plan'):
        legacy_validate(value, task, StagePolicy(lambda _: True))


def test_finished_or_foreign_turn_cannot_update_task_or_invalidate_final_result(tmp_path):
    captured = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        captured.append((context['_agent_conversation_turn'], context))
        return {'kind': 'final', 'text': '这是普通解释。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('finished', '你好'))
        assert settled(port, h.task_id)['status'] == 'completed'
        old_result = port.result(h.task_id); turn, context = captured[0]
        with pytest.raises(AgentBoundaryError, match='work_plan_task_not_active'):
            turn.work_plan.update(0, plan(), context)
        with pytest.raises(AgentBoundaryError, match='conversation_authorization_changed'):
            turn.work_plan.update(0, plan(), {**context, '_agent_conversation_turn': object()})
        assert port.result(h.task_id) == old_result and history(service, h.task_id) is None


def test_old_attempt_token_cannot_change_recovered_plan(tmp_path):
    captured = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        if count == 1:
            captured.append((context['_agent_conversation_turn'], context))
            return update()
        if count == 2: return {'kind': 'wait', 'text': '等待进一步说明。'}
        old_turn, old_context = captured[0]
        with pytest.raises(AgentBoundaryError, match='conversation_binding_changed'):
            old_turn.work_plan.update(1, plan(), old_context)
        return {'kind': 'final', 'text': '继续给出解释，尚未执行操作。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, recovery=True) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('attempt', '讨论持续排查方案'))
        assert settled(port, h.task_id)['status'] == 'waiting_user'
        port.resume(handle_from(port.status(h.task_id)), request_key='resume')
        assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
        assert history(service, h.task_id)['revision'] == 1 and len(calls) == 3


def test_actual_failure_replans_in_same_loop_then_real_write_passes_original_gate(tmp_path):
    from shinku.agent_runtime.demo import DemoFileTool
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    class FaultOnce(DemoFileTool):
        failed = False
        def execute(self, *, call, context):
            if not self.failed:
                self.failed = True
                return {'status': 'error', 'tool_type': self.tool_type, 'error': 'synthetic_dependency_missing',
                    'followup_context': '合成前置条件尚未满足；没有写入文件。'}
            return super().execute(call=call, context=context)
    actual = FaultOnce(tmp_path)
    def respond(kw, count, service, tool):
        if count == 1:
            decision = update(plan(['file']))
            decision['task_contract'] = demo_request(tool).contract
            return decision
        if count in {2, 4}: return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        if count == 3:
            assert not actual.path.exists()
            task = next(iter(service._sessions.values())).context['agent_task_state']
            failure = task['tool_calls'][-1]
            value = plan(['file'], references=[{'kind': 'tool_call', 'id': failure['id']}])
            value['reason'] = '实际工具失败后调整执行顺序'
            return update(value, 1)
        return {'kind': 'final', 'text': '合成文件已创建并核验，首个失败回执和计划修订均保留。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handlers = service.executor.tool_host.registry.view()
        service.executor.tool_host = ToolHost(ToolRegistry({**handlers, 'demo_file': actual}))
        h = port.submit(AgentTurnRequest('failure-replan', '创建 ' + str(tool.path)))
        view = settled(port, h.task_id); result = port.result(h.task_id)
        assert view['status'] == 'completed', result
        assert actual.calls == 1 and len(calls) == result['budget']['requests'] == 5
        assert result['verification']['satisfied'] == ['file']
        assert history(service, h.task_id, 1)['plan']['reason'] == '按当前任务安排工作'
        ref = history(service, h.task_id)['references'][0]
        assert ref['execution_status'] == 'error' and ref['tool'] == 'demo_file'
        assert any(c['status'] == 'error' for c in saved_task(service, h.task_id)['tool_calls'])






def test_internal_plan_tool_is_absent_from_business_effect_catalog(tmp_path):
    from types import SimpleNamespace
    from shinku.qq.task_contracts import task_effect_catalog
    def respond(kw, count, service, tool):
        turn = next(iter(service._sessions.values())).context['_agent_conversation_turn']
        catalog = task_effect_catalog(deps=SimpleNamespace(registry_handlers=lambda: turn.handlers))
        assert 'update_work_plan' not in catalog and 'write' in catalog['demo_file']
        return {'kind': 'final', 'text': '这里只讨论方案。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('catalog', '讨论方案'))
        assert settled(port, h.task_id)['status'] == 'completed'
        assert len(calls) == 1 and tool.calls == 0
