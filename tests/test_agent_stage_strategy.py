"""P5b schema, metering, replan/restart and actual independent IO boundaries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget, observe_budget_reservations
from shinku.agent.loop import AgentState
from shinku.agent.planner import AgentPromptParts
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime.execution import _MeteredRuntime
from shinku.agent_runtime.stages import StageController, StagePolicy, StageStrategyStopped, task_binding, validate_plan


def task():
    return {'task_id': 'unit-task', 'requester': 'test', 'session': 'local:test',
        'requests': [{'text': 'Write both original files.'}], 'confirmed_constraints': ['No deployment.'],
        'contract': {'goal': 'Original goal', 'requirements': [
            {'id': 'R1', 'effect': 'write', 'targets': ['one.txt'], 'checks': {'contains': 'one'}},
            {'id': 'R2', 'effect': 'write', 'targets': ['two.txt'], 'checks': {'contains': 'two'}}]}}


def proposal(t):
    return {'binding': task_binding(t), 'reason': 'Read actual source; preserve the full contract.', 'stages': [
        {'id': 'S1', 'goal': 'Write first original output', 'depends': [], 'requirements': ['R1']},
        {'id': 'S2', 'goal': 'Write dependent output', 'depends': ['S1'], 'requirements': ['R2']}]}


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(binding='wrong'), lambda p: p.update(permission_mode='ultra'),
    lambda p: p.update(budget={'max_requests': 999}), lambda p: p.update(tool_call={'type': 'shell'}),
    lambda p: p['stages'].pop(), lambda p: p['stages'][0].update(depends=['S2']),
    lambda p: p['stages'][1].update(id='S1'), lambda p: p['stages'][1].update(requirements=['R1']),
    lambda p: p['stages'][0].update(requirements=['FORGED']),
    lambda p: p['stages'][0].update(status='completed'),
    lambda p: p['stages'][0].update(artifacts=['outside.txt']),
    lambda p: p['stages'][0].update(acceptance=[]), lambda p: p.update(reason='x'*1001),
    lambda p: p['stages'][1].update(depends=['S1','S1']),
    lambda p: p['stages'][0].update(requirements=['R1','R1']),
])
def test_proposals_cannot_remove_checks_forge_status_or_expand_authority(mutation):
    t = task()
    before = copy.deepcopy(t)
    value = proposal(t)
    mutation(value)
    with pytest.raises((ValueError, TypeError)):
        validate_plan(value, t, StagePolicy(lambda _: True))
    assert t == before


class Model:
    def __init__(self, t, *, invalid=False, interrupt=None):
        self.t, self.invalid, self.interrupt, self.calls = t, invalid, interrupt, []
    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.interrupt:
            self.interrupt()
        return {'tool_call': {'type': 'shell'}} if self.invalid else proposal(self.t)


class Evidence(SmartCompletionVerifier):
    def __init__(self):
        super().__init__()
        self.satisfied = []
    def evaluate(self, *args, **kwargs):
        return SimpleNamespace(satisfied=self.satisfied, accepted=True, status='partial')


def controller(t=None, *, enabled=True, invalid=False, interrupt=lambda _: '', max_replans=2):
    t = t or task()
    context = {'agent_task_state': t, 'agent_stage_read_tools': {'read'}}
    model, saved = Model(t, invalid=invalid), []
    c = StageController(context, StagePolicy(lambda _: enabled, max_replans=max_replans), _MeteredRuntime(model),
        profile_id='unit-executor', temperature=.2, interrupt=interrupt,
        save=lambda state, status: saved.append(copy.deepcopy(t['stage_strategy'])))
    c.gate = Evidence()
    s = AgentState(context=context)
    return c, s, model, saved


PROMPT = AgentPromptParts('Full original goal and both original checks.')


def test_disabled_or_not_yet_read_has_no_planning_call():
    for enabled in (True, False):
        c, s, m, _ = controller(enabled=enabled)
        with bind_task_budget(TaskBudget()):
            assert c.prepare(s, PROMPT) == PROMPT
        assert not m.calls
    assert 'stage_strategy' not in c.task


def test_plan_uses_same_meter_and_has_no_tool_schema_or_synthetic_acceptance():
    c,s,m,saved = controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    original = copy.deepcopy(c.task['contract'])
    b = TaskBudget(max_requests=5)
    with bind_task_budget(b):
        p = c.prepare(s, PROMPT)
        for _ in range(3):
            c.prepare(s, PROMPT)
    assert b.requests == len(m.calls) == 1 and b.tokens > 32768
    assert saved[0]['inflight'] and saved[0]['calls'] == 1
    assert saved[-1]['status'] == 'active' and not saved[-1]['inflight']
    assert m.calls[0]['native_tools'] is None
    assert PROMPT.user_prompt in m.calls[0]['user_prompt']
    assert p.user_prompt.startswith(PROMPT.user_prompt) and p.system_extra_blocks
    assert 'Write first original output' not in '\n'.join(p.system_extra_blocks)
    assert c.task['contract'] == original
    assert [r['status'] for r in c.record['stages']] == ['in_progress','pending']


def test_replans_are_triggered_by_new_feedback_and_bounded_in_same_budget():
    c,s,m,_ = controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    b = TaskBudget(max_requests=8)
    with bind_task_budget(b):
        c.prepare(s,PROMPT)
        for i in range(5):
            s.completion = {'accepted':False,'missing':[str(i)]}
            c.prepare(s,PROMPT)
            c.prepare(s,PROMPT)
    assert b.requests == len(m.calls) == c.record['calls'] == 3
    assert len(c.record['history']) == 3


def test_partial_report_acceptance_does_not_complete_unverified_stages_and_dependency_loss_replans():
    c,s,m,_ = controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    with bind_task_budget(TaskBudget()):
        c.prepare(s,PROMPT)
        c.gate.satisfied = ['R2']
        c.refresh(s)
        assert not any(r['status']=='completed' for r in c.record['stages'])
        c.gate.satisfied = ['R1','R2']
        c.refresh(s)
        assert all(r['status']=='completed' for r in c.record['stages'])
        c.gate.satisfied = ['R2']
        c.prepare(s,PROMPT)
    assert len(m.calls)==2 and m.calls[-1]['user_prompt'].endswith('}')
    assert 'artifact_or_dependency_invalidated' in c.record['history'][-1]['trigger']
    assert all(r['status']!='completed' for r in c.record['stages'])


def test_invalid_plan_falls_back_without_recursive_repair_or_a_tool_call():
    c,s,m,_=controller(invalid=True)
    s.tool_calls.append({'name':'read','status':'ok'})
    with bind_task_budget(TaskBudget()):
        assert c.prepare(s,PROMPT)==PROMPT
        assert c.prepare(s,PROMPT)==PROMPT
    assert len(m.calls)==1 and c.record['status']=='plan_rejected' and c.record['plan'] is None


def test_recovery_reuses_valid_hints_and_unknown_inflight_outcome_is_never_rebought():
    c,s,m,_=controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    with bind_task_budget(TaskBudget()):
        c.prepare(s,PROMPT)
    t = copy.deepcopy(c.task)
    resumed, state, model, _ = controller(t)
    state.tool_calls = copy.deepcopy(s.tool_calls)
    with bind_task_budget(TaskBudget()):
        resumed.prepare(state,PROMPT)
    assert not model.calls and resumed.record['calls']==1
    t['stage_strategy']['inflight']=True
    unknown,state,model,_=controller(t)
    state.tool_calls = copy.deepcopy(s.tool_calls)
    state.completion={'accepted':False,'missing':['changed']}
    with bind_task_budget(TaskBudget()):
        unknown.prepare(state,PROMPT)
    assert not model.calls and unknown.record['plan'] is None
    assert unknown.record['status']=='planning_outcome_unknown'


def test_permission_change_after_plan_prevents_execution_and_next_purchase():
    permission={'ok':True}
    c,s,m,_=controller(interrupt=lambda _: '' if permission['ok'] else 'agent_permission_changed')
    s.tool_calls.append({'name':'read','status':'ok'})
    m.interrupt=lambda: permission.update(ok=False)
    with bind_task_budget(TaskBudget()):
        with pytest.raises(StageStrategyStopped,match='agent_permission_changed'):
            c.prepare(s,PROMPT)
    assert len(m.calls)==1 and c.record['inflight']


def test_planning_can_exhaust_original_request_limit_without_expansion():
    c,s,m,_=controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    b=TaskBudget(max_requests=1)
    with bind_task_budget(b):
        c.prepare(s,PROMPT)
        s.completion={'accepted':False,'missing':['needs repair']}
        with pytest.raises(TaskBudgetExceeded,match='task_model_budget_exhausted'):
            c.prepare(s,PROMPT)
    assert b.requests==1 and len(m.calls)==1


def test_goal_constraint_change_discards_stale_hints_but_preserves_planning_count():
    c,s,m,_=controller()
    s.tool_calls.append({'name':'read','status':'ok'})
    with bind_task_budget(TaskBudget()):
        c.prepare(s,PROMPT)
        c.task['confirmed_constraints'].append('A new confirmed condition.')
        c.refresh(s)
    assert c.record['plan'] is None and c.record['calls']==1


def test_request_observer_saves_after_admission_and_before_provider_and_does_not_change_usage():
    b=TaskBudget(max_requests=3)
    calls=[]
    class Runtime:
        def call_chat_json(self,**kwargs):
            calls.append(('provider',b.requests,b.tokens))
            return {'usage':{'total_tokens':17}}
    with bind_task_budget(b), observe_budget_reservations(lambda budget: calls.append(('checkpoint',budget.requests,budget.tokens))):
        _MeteredRuntime(Runtime()).call_chat_json(user_prompt='original full request')
    assert [v[0] for v in calls]==['checkpoint','provider']
    assert calls[0][1:]==calls[1][1:] and calls[0][2]>32768
    assert b.requests==1 and b.tokens==b.reported_tokens==17 and b.estimated_requests==0
    with bind_task_budget(b):
        b.reserve_request({'messages':[]})
    assert len(calls)==2  # Context-local observer did not leak.


def test_checkpoint_failure_stops_provider_dispatch_and_keeps_conservative_charge():
    b=TaskBudget()
    model=Model(task())
    def unavailable(_):
        raise OSError('isolated ledger unavailable')
    with bind_task_budget(b), observe_budget_reservations(unavailable):
        with pytest.raises(OSError):
            _MeteredRuntime(model).call_chat_json(user_prompt='full request')
    assert model.calls==[] and b.requests==1 and b.tokens>32768


def test_bound_images_are_forwarded_to_the_same_planning_runtime():
    c,s,m,_=controller()
    c.context['agent_execution_images']=['data:image/png;base64,fixture']
    s.tool_calls=[{'name':'read','status':'ok'}]
    with bind_task_budget(TaskBudget()):
        c.prepare(s,PROMPT)
    assert m.calls[0]['user_images']==c.context['agent_execution_images']


def test_no_output_pending_execution_does_not_trigger_replanning():
    c,s,m,_=controller()
    s.tool_calls=[{'name':'read','status':'ok'}]
    with bind_task_budget(TaskBudget()):
        c.prepare(s,PROMPT)
        for _ in range(4):
            s.tool_calls.append({'name':'session_wait','status':'running','executed':True})
            c.prepare(s,PROMPT)
    assert len(m.calls)==1


@pytest.fixture(scope='module')
def bench():
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_agent_stages.py'
    spec=importlib.util.spec_from_file_location('p5b_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def paired(bench,tmp_path_factory):
    return bench.run_suite(tmp_path_factory.mktemp('p5b')/'paired',planning='autonomous')


def test_full_fixed_corpus_keeps_original_rules_and_all_batch_calls_count(paired):
    assert paired['all_pairs_present'] and len(paired['rows'])==30
    assert all(r['fixture_pass'] and r['contract_preserved'] and r['query_unchanged'] for r in paired['rows'])
    assert paired['batch_budget']['requests']==sum(r['model_calls'] for r in paired['rows'])
    assert sum(r['planning_calls'] for r in paired['rows'])==2
    assert paired['batch_budget']['requests']<=360
    assert all(r['planning_calls']==0 for r in paired['rows'] if r['kind']!='complex')
    assert paired['quality_conclusion']=='not_measured' and not paired['enable_production']


def test_real_io_cross_slice_retains_plan_and_current_host_artifact_status(paired):
    row=next(r for r in paired['rows'] if r['case']=='cross_slice_code' and r['arm']=='staged')
    assert row['planning_calls']==1 and row['model_calls']==row['budget']['requests']
    assert row['stage_checkpoint']['calls']==1
    assert len(row['stage_checkpoint']['stages'])==3
    assert all(r['status']=='completed' for r in row['stage_checkpoint']['stages'])
    assert row['publications']==1 and row['duplicate_effects']==row['duplicate_publications']==0


def test_independent_io_read_then_early_completion_replans_without_weakening_gate(bench,tmp_path):
    corpus=json.loads(bench.CORPUS.read_text(encoding='utf-8'))
    case=copy.deepcopy(next(r for r in corpus['cases'] if r['id']=='early_final_repair'))
    case['initial']={'spec.txt':'Both manifest and report are required; do not deploy.'}
    case['actions'].insert(0,{'op':'read','path':'spec.txt'})
    row=bench.run_case(case,'staged',tmp_path/'early',corpus['budget'],planning='autonomous')
    assert row['fixture_pass'] and row['planning_calls']==2
    assert row['stage_checkpoint']['history'][1]['trigger'].startswith('completion_rejected:')
    assert row['model_calls']==row['budget']['requests'] and row['publications']==1


def test_real_syntax_failure_replans_and_verifies_correction_without_repeating_effect(bench,tmp_path):
    corpus=json.loads(bench.CORPUS.read_text(encoding='utf-8'))
    case=copy.deepcopy(next(r for r in corpus['cases'] if r['id']=='syntax_correction'))
    case['initial']={'spec.txt':'Implement double(value); ensure valid Python syntax.'}
    case['actions'].insert(0,{'op':'read','path':'spec.txt'})
    row=bench.run_case(case,'staged',tmp_path/'syntax',corpus['budget'],planning='autonomous')
    assert row['fixture_pass'] and row['planning_calls']==2
    assert row['stage_checkpoint']['history'][1]['trigger'].startswith('artifact_check_failed:')
    assert len(row['writes'])==2 and row['duplicate_effects']==0
    assert len(row['stage_checkpoint']['stages'])==1
    assert all(r['status']=='completed' for r in row['stage_checkpoint']['stages'])


@pytest.mark.parametrize('kind', ['max_requests','max_tokens','max_tools','max_active_seconds'])
def test_entire_batch_cutoff_preserves_partial_report_and_never_calls_it_fair(bench,tmp_path,kind):
    limits={'max_requests':360,'max_tokens':18000000,'max_tools':240,'max_active_seconds':900}
    limits[kind]=.0001 if kind=='max_active_seconds' else 1
    result=bench.run_suite(tmp_path/kind,planning='autonomous',batch_limits=limits)
    assert not result['all_pairs_present'] and result['unrun_pairs']>0
    assert result['arms']=={} and result['batch_stop_reason']
    assert result['quality_conclusion']=='not_measured'
    if kind=='max_requests':
        assert sum(r['model_calls'] for r in result['rows'])<=1
    if kind=='max_tools':
        assert result['batch_budget']['tools']<=1


def test_expanded_batch_cap_is_rejected_before_workspace_io(bench,tmp_path):
    target=tmp_path/'forbidden'
    with pytest.raises(ValueError,match='batch_expands'):
        bench.run_suite(target,batch_limits={'max_requests':999,'max_tokens':18000000,'max_tools':240,'max_active_seconds':900})
    assert not target.exists()


@pytest.mark.parametrize('value', [float('nan'),float('inf'),1.5,True,'12'])
def test_malformed_batch_counter_limit_cannot_disable_admission(bench,tmp_path,value):
    target=tmp_path/'malformed'
    with pytest.raises(ValueError,match='batch_expands'):
        bench.run_suite(target,batch_limits={'max_requests':value,'max_tokens':18000000,'max_tools':240,'max_active_seconds':900})
    assert not target.exists()


def test_provider_dispatch_crash_keeps_original_budget_and_unknown_plan_checkpoint(bench,tmp_path):
    root=tmp_path/'crash'
    script=tmp_path/'kill_during_plan.py'
    script.write_text('''import importlib.util,json,os,pathlib,sys
path=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('crash_bench',path)
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
original=b.FixtureRuntime
class Crash(original):
    def call_chat_json(self,**kwargs):
        if str(kwargs.get('prompt_cache_key','')).endswith(':stage-plan'):
            from shinku.agent.budget import current_task_budget
            current_task_budget().reserve_request({'messages':[{'role':'user','content':kwargs['user_prompt']}]})
            os._exit(73)
        return super().call_chat_json(**kwargs)
b.FixtureRuntime=Crash
corpus=json.loads(b.CORPUS.read_text(encoding='utf-8'))
case=next(r for r in corpus['cases'] if r['id']=='two_modules')
b.run_case(case,'staged',sys.argv[2],corpus['budget'],planning='autonomous')
''',encoding='utf-8')
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
    run=subprocess.run([sys.executable,'-X','utf8','-B',str(script),str(Path(bench.__file__)),str(root)],
                       env=env,capture_output=True,text=True,timeout=20)
    assert run.returncode==73,(run.stdout,run.stderr)
    from shinku.agent.task_state import AgentTaskStore
    store=AgentTaskStore(root/'business.sqlite')
    try:
        rows=store.candidates(requester=bench.SCOPE.requester,session=bench.SCOPE.session)
        saved=rows[0]
    finally:
        store.close()
    assert saved['stage_strategy']['inflight']
    assert saved['stage_strategy']['calls']==1
    assert saved['budget']['requests']==2 and saved['budget']['tokens']>65536
    assert not (root/'math_ops.py').exists() and not (root/'client.py').exists()
    c,state,model,_=controller(saved)
    state.tool_calls=copy.deepcopy(saved['tool_calls'])
    with bind_task_budget(TaskBudget()):
        c.prepare(state,PROMPT)
    assert model.calls==[] and c.record['status']=='planning_outcome_unknown'
