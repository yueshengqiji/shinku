"""真实预算/账本/恢复链的离线回归；大计数不代表大规模模型调用。"""
from dataclasses import replace
import json

import pytest
from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget, metered_completion, output_reservation
from shinku.angelos.budget import LIMITS, local_task_limits
from shinku.agent_runtime.execution import ExecutionGrant
from shinku.agent_runtime.recovery import _budget_reason

UNLIMITED = dict.fromkeys(LIMITS)
FINITE = dict(zip(LIMITS, (800, 8000000, 512, 14400)))


def test_new_local_default_and_explicit_experiment_old_task_isolation():
    assert local_task_limits({}, FINITE) == UNLIMITED
    assert local_task_limits({'contract': {'budget_mode': 'metered'}}, FINITE, mode='bounded') == FINITE
    old = TaskBudget(**FINITE, tokens=8000001, reason='task_token_budget_exhausted').snapshot()
    assert local_task_limits({'budget': old}, FINITE) == FINITE
    budget = TaskBudget(**local_task_limits({'budget': old}, FINITE))
    budget.absorb_prior(old)
    assert budget.tokens == 8000001 and budget.reason == 'task_token_budget_exhausted'


def test_large_counters_unknown_usage_and_no_output_truncation():
    budget = TaskBudget(**UNLIMITED, requests=2000, tools=2000, tokens=30000000, active_seconds=90000, managed=True)
    sent = []
    def create(**payload):
        sent.append(payload)
        return {}  # 没有usage保留估计，不能因无法结算而占满/堵住下一次调用。
    with bind_task_budget(budget), output_reservation(0):
        for _ in range(3):
            metered_completion(create, {'messages': [{'role': 'user', 'content': '合成请求'}]})
            budget.use_tool()
    assert budget.requests == 2003 and budget.tools == 2003 and budget.tokens > 30000000
    assert budget.estimated_requests == 3 and budget.reported_tokens == 0
    assert not budget.stop_reason() and budget.stage() == 'working'
    assert all('max_tokens' not in p and 'max_completion_tokens' not in p for p in sent)
    with bind_task_budget(budget):
        metered_completion(create, {'messages': []})
    assert 'max_tokens' not in sent[-1]  # 无预留器的兼容入口也不偷偷注入32k。
    assert _budget_reason(json.loads(json.dumps(budget.snapshot(), allow_nan=False))) == ''


def test_stream_usage_and_failure_keep_accounting_without_latching_limit():
    budget = TaskBudget(**UNLIMITED)
    with bind_task_budget(budget):
        stream = metered_completion(lambda **_: iter([{'choices': []}, {'usage': {'total_tokens': 123}}]),
                                    {'messages': [], 'stream': True})
        list(stream)
        with pytest.raises(RuntimeError, match='offline'):
            metered_completion(lambda **_: (_ for _ in ()).throw(RuntimeError('offline')), {'messages': []})
    assert budget.reported_tokens == 123 and budget.estimated_requests == 1 and budget.requests == 2
    restored = TaskBudget(**UNLIMITED); restored.absorb_prior(json.loads(json.dumps(budget.snapshot())))
    assert restored.tokens == budget.tokens and restored.estimated_requests == 1 and not restored.reason


@pytest.mark.parametrize('field,reason', [('requests','task_model_budget_exhausted'), ('tokens','task_token_budget_exhausted'),
                                         ('tools','task_tool_budget_exhausted'), ('active_seconds','task_time_budget_exhausted')])
def test_finite_guards_still_stop(field, reason):
    budget = TaskBudget(**FINITE)
    setattr(budget, field, getattr(budget, 'max_'+field))
    with pytest.raises(TaskBudgetExceeded, match=reason):
        budget.use_tool() if field == 'tools' else budget.reserve_request({'messages': []})


def test_invalid_snapshot_and_stop_reason_are_not_erased():
    for snapshot in ({'max_requests': 5}, {'max_tokens': None}, {'limit_mode': 'metered', **UNLIMITED, 'tokens': -1}):
        b=TaskBudget(**UNLIMITED, managed=True); b.absorb_prior(snapshot)
        assert b.reason == 'task_budget_snapshot_invalid'
        assert b.snapshot()['reason'] == 'task_budget_snapshot_invalid'
    b=TaskBudget(**UNLIMITED);b.absorb_prior({**b.snapshot(),'reason':'task_canceled'})
    assert b.stop_reason() == 'task_canceled'
    assert _budget_reason({**b.snapshot(),'tokens':float('nan')}) == 'recovery_budget_invalid'
    with pytest.raises(ValueError): TaskBudget(max_tokens=None)
    with pytest.raises(ValueError): local_task_limits({},FINITE,mode='model_says_unlimited')


def test_remote_delegation_remains_finite():
    from shinku.agent_runtime.delegation_http import authority_payload, intersect_authority
    from shinku.agent_runtime.contracts import AgentBoundaryError
    local=ExecutionGrant('local',frozenset(),budget_limits=UNLIMITED)
    with pytest.raises(AgentBoundaryError,match='delegation_finite_budget_required'): authority_payload(local)
    caller=ExecutionGrant('remote',frozenset(),budget_limits=FINITE)
    intersected=intersect_authority(local,authority_payload(caller))
    assert dict(intersected.budget_limits)==FINITE and not intersected.budget().metered_only


def test_real_service_restart_preserves_usage_and_permissions(tmp_path):
    from test_agent_recovery import service_at, close, settle, handle, Runtime, SCOPE, FINAL, write
    from shinku.agent_runtime.demo import demo_request
    service,client,tool,runtime=service_at(tmp_path)
    original=service.authorize
    service.authorize=lambda *args:replace(original(*args),budget_limits=UNLIMITED)
    runtime.decisions=[write(tool), RuntimeError('fixture provider error')]
    try:
        h=client.submit(demo_request(tool));v=settle(client,h.task_id)
        assert v['status']=='failed' and tool.calls==1
        task=service.task_store.get(h.task_id,requester=SCOPE.requester,session=SCOPE.session)
        assert task['budget']['limit_mode']=='metered'
        task['budget'].update(requests=2000,tokens=30000000,tools=2000,active_seconds=90000)
        service.task_store.save(task)
    finally:close(service)
    service,client,newtool,runtime=service_at(tmp_path,Runtime([FINAL]))
    original=service.authorize
    service.authorize=lambda *args:replace(original(*args),budget_limits=UNLIMITED)
    try:
        h=handle(client.status(h.task_id));proof=client.recovery(h)
        assert proof['status']=='continuable'
        client.resume(h,request_key='resume-long')
        assert settle(client,h.task_id)['status']=='completed' and newtool.calls==0
        task=service.task_store.get(h.task_id,requester=SCOPE.requester,session=SCOPE.session)
        assert task['budget']['requests']>2000 and task['budget']['tokens']>=30000000
        assert task['budget']['max_tokens'] is None and tool.path.exists()
    finally:close(service)


def test_null_capacity_survives_redaction_but_secret_values_do_not():
    from shinku.agent.task_state import redact
    assert redact({'max_tokens': None}) == {'max_tokens': None}
    assert redact({'max_tokens': 'secret-fixture', 'access_token': 'secret-fixture'}) == {
        'max_tokens': '[redacted]', 'access_token': '[redacted]'}


def test_metered_slices_continue_and_repeat_guard_remains():
    from test_agent_slices import Handler
    from shinku.agent.loop import AgentLoop, AgentDecision
    from shinku.tools.invocation import ToolInvocation
    tool=Handler();budget=TaskBudget(**UNLIMITED,tools=2000,managed=True)
    def plan(*,state):
        return AgentDecision.tool(ToolInvocation('read',{'n':state.rounds})) if state.rounds <= 9 else AgentDecision.final('verified')
    loop=AgentLoop(planner=plan,handlers={'read':tool},max_rounds=2,sliced=True,
                   completion_gate=lambda state,text:len(state.tool_calls)==9)
    with bind_task_budget(budget):
        result=loop.run(task_id='long-sliced')
        while result.status=='yielded':result=loop.run(checkpoint=result.checkpoint)
    assert result.status=='completed' and tool.calls==list(range(1,10)) and budget.tools==2009
    assert any('持续执行模式' in row.get('content','') for row in result.transcript)
    tool=Handler();loop=AgentLoop(planner=lambda **_:AgentDecision.tool(ToolInvocation('read',{'n':1})),
                                handlers={'read':tool},max_rounds=1,sliced=True)
    with bind_task_budget(TaskBudget(**UNLIMITED)):
        result=loop.run()
        for _ in range(5):
            if result.status!='yielded':break
            result=loop.run(checkpoint=result.checkpoint)
    assert result.reason=='repeated_tool_call' and len(tool.calls)==1


@pytest.mark.parametrize('reason',['task_canceled','agent_permission_changed'])
def test_metered_session_still_cleans_up_on_cancel_or_revocation(reason):
    from test_agent_execution_lifecycle import SessionHandler,make_loop,start
    handler=SessionHandler()
    with bind_task_budget(TaskBudget(**UNLIMITED)):
        result=make_loop(handler,interruption_check=lambda _:reason if handler.polls>=2 else '').run(first_decision=start())
    assert result.reason==reason and not handler.running and handler.stops==[reason]
