"""阶段整理只给建议，不改变权限、预算和完成事实；旧原生前缀保持。"""
from types import SimpleNamespace

from shinku.agent.investigation import note_investigation, investigation_advisory


def state(enabled=True):
    return SimpleNamespace(context={'angelos_investigation_guidance':enabled,'agent_task_state':{'work_state':{'revision':0}}},
                           progress={},finalization_only=False)


def test_periodic_advice_does_not_disable_tools_or_claim_progress():
    value=state(); invocation=SimpleNamespace(name='workspace_search')
    for _ in range(7): note_investigation(value,invocation,{})
    assert not investigation_advisory(value)
    note_investigation(value,invocation,{})
    advice=investigation_advisory(value)
    assert '具体缺口' in advice and '不是调用次数上限或验收条件' in advice
    assert not value.finalization_only
    for _ in range(3): note_investigation(value,invocation,{})
    assert investigation_advisory(value)==advice
    value.context['agent_task_state']['work_state']['revision']=1
    for _ in range(7): note_investigation(value,invocation,{})
    assert investigation_advisory(value)==advice
    note_investigation(value,invocation,{})
    assert investigation_advisory(value)!=advice and not value.finalization_only


def test_waits_plan_tools_and_other_hosts_do_not_advance_counter():
    value=state()
    for _ in range(20):
        note_investigation(value,SimpleNamespace(name='private_shell'),{'waiting':True})
        note_investigation(value,SimpleNamespace(name='update_work_plan'),{})
    assert not value.progress
    value=state(False)
    for _ in range(20): note_investigation(value,SimpleNamespace(name='private_shell'),{})
    assert not value.progress


def test_repeated_receipt_does_not_count_as_new_investigation():
    value=state(); invocation=SimpleNamespace(name='private_shell')
    for _ in range(20):
        note_investigation(value,invocation,{'fingerprint':'same-terminal-receipt','waiting':False})
    assert value.progress['investigation_checkpoint']['calls']==1
    assert not investigation_advisory(value) and not value.finalization_only
    for index in range(7):
        note_investigation(value,invocation,{'fingerprint':f'new-observation-{index}'})
    assert value.progress['investigation_checkpoint']['calls']==8
    assert '对照是否确实改变关键条件' in investigation_advisory(value)
    assert not value.finalization_only


def test_receipt_dedup_survives_resume_without_disabling_tools():
    import copy
    first=state();call=SimpleNamespace(name='workspace_read_lines')
    note_investigation(first,call,{'fingerprint':'original'})
    resumed=state();resumed.progress=copy.deepcopy(first.progress)
    note_investigation(resumed,call,{'fingerprint':'original'})
    assert resumed.progress['investigation_checkpoint']['calls']==1
    assert first.progress==resumed.progress
