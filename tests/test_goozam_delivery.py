"""经验进入真实任务提示而非只开开关；来源/撤销/权限仍由原宿主控制。"""
import json
from copy import deepcopy

import pytest
from goozam.host import experiment, TaskSources
from goozam.learning import Assessment
from goozam.experience_context import ExperienceContext
from goozam.offline import OfflineRuntime, send
from goozam.proposals import ProposalRunner
from shinku.angelos.task_continuity import workspace_binding
from test_goozam_host import root
from test_goozam_proposals import reply


def seed(lab):
    wid = lab.workbench.store.workspace({'name': 'fixture', 'path': str(lab.workspace)})['id']
    source, _ = send(lab, wid, '训练：解释代码', '静态结果。')
    _, scope = workspace_binding(lab.workbench.store, source['session'])
    key = lab.learning.propose(scope, source, situation='解释代码', conditions='没有执行回执',
        advice='静态推导应明确未运行。', exceptions='有真实执行回执时据实报告。')
    baseline, _ = send(lab, wid, '留出：解释代码', '结果。')
    trial, _ = send(lab, wid, '留出：解释代码', '静态推导，未运行。')
    lab.learning.evaluate(scope, key, lambda p: Assessment(baseline, trial, 0, 1, True, 'synthetic'))
    lab.learning.activate(scope, key)
    return wid, scope, key


def packages(lab, ref):
    task = lab.host.service.task_store.get(ref['task_id'], requester=lab.host.requester, session=ref['session'])
    return [r for r in task['material_package']['items'] if r['source_id']=='goozma-experiences']


def test_actual_prompt_delivery_without_tool_call_and_revoke(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        before = len(runtime.calls)
        ref, turn = send(lab, wid, '解释代码', '静态推导，未运行。')
        rows = packages(lab, ref)
        assert len(rows) == 1 and rows[0]['required'] is False
        assert rows[0]['omitted_chars'] == 0
        assert key in rows[0]['content'] and '没有执行回执' in rows[0]['content']
        prompts = json.dumps(runtime.calls[before:], ensure_ascii=False, default=str)
        assert key in prompts and '有真实执行回执时据实报告' in prompts
        assert not turn['result']['evidence']
        assert turn['result']['execution']['status']=='completed'
        lab.learning.revoke(scope, key)
        ref2, _ = send(lab, wid, '解释代码', '静态结果。')
        assert not packages(lab, ref2)
        assert packages(lab, ref) == rows  # 撤销不抹除历史输入。


@pytest.mark.parametrize('mode', ['off', 'learn', 'zayin'])
def test_no_delivery_when_off_learning_or_no_tools(root, mode):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        if mode != 'off':
            lab.controls.select('use' if mode=='zayin' else mode, 0)
        if mode == 'zayin':
            lab.host.permissions.select('zayin', lab.host.permissions.revision)
        ref, _ = send(lab, wid, '解释代码', '静态结果。')
        assert not packages(lab, ref)


def test_failed_draft_reaches_proposal_without_changing_result(root):
    runtime = OfflineRuntime()
    runtime.responses = [{'fixture': True, 'loaded_version': 'old'}]
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name': 'failure', 'path': str(lab.workspace)})['id']
        ref, turn = send(lab, wid, '分析部署快照', '')
        assert turn['result']['execution']['status']=='failed'
        original = deepcopy(turn['result'])
        raw = '合成诊断原稿：运行版本与源码不同，尚未重启。'
        lab.host.reasoning_view.append_output(ref['task_id'], {
            'text':raw, 'source':'provider_message_content',
            'diagnostics':{'failure':'invalid_decision'}, 'reasoning_content':'must-not-copy'})
        _, scope = workspace_binding(lab.workbench.store, ref['session'])
        sources = TaskSources(lab.host)
        source = sources(scope, ref)
        assert source['accepted'] is False
        assert source['result'] == original
        proposer = OfflineRuntime()
        response = reply()
        response['support'] = ['/model_drafts/records/1/text']
        proposer.responses = [response]
        result = ProposalRunner(lab.learning, root/'learning/proposals', proposer, {'model':'fixture'}).propose(scope, ref)
        assert result['status']=='candidate'
        payload = proposer.calls[0]['user_prompt']
        assert raw in payload and 'must-not-copy' not in payload
        assert 'is_execution_evidence' in payload
        assert sources(scope, ref)['result'] == original


def test_source_change_during_proposal_retains_raw_and_fails(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name':'fixture','path':str(lab.workspace)})['id']
        ref, _ = send(lab, wid, '解释代码', '结果。')
        _, scope = workspace_binding(lab.workbench.store, ref['session'])
        def changing(**kwargs):
            lab.host.reasoning_view.append_output(ref['task_id'], {'text':'late draft'})
            return reply()
        runtime.call_chat_json = changing
        result = ProposalRunner(lab.learning, root/'learning/proposals', runtime, {'model':'fixture'}).propose(scope, ref)
        assert result['error']=='goozam_source_changed_during_proposal'
        assert len(list((root/'learning/proposals').glob('*/response.json')))==1


def test_one_task_candidate_trial_is_delivered_without_activation(root):
    import time
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name':'trial','path':str(lab.workspace)})['id']
        source, _ = send(lab, wid, '训练代码', '静态结果。')
        _, scope = workspace_binding(lab.workbench.store, source['session'])
        key = lab.learning.propose(scope, source, situation='代码', conditions='只读',
            advice='先核对版本再作推断。', exceptions='已有真实证据时复用。')
        lab.controls.select('use', 0)
        cid = lab.workbench.store.conversation(wid)['id']
        task_id = lab.host.task_id(cid, 'candidate-trial')
        with lab.plugin.trial(scope, task_id, key):
            other, _ = send(lab, wid, '代码', '结果。')
            assert not packages(lab, other)
            before = len(runtime.calls)
            lab.workbench.send(cid, {'request_key':'candidate-trial', 'text':'代码'})
            deadline = time.monotonic()+5
            while time.monotonic()<deadline:
                turn = lab.workbench.refresh(cid)['turns'][0]
                if turn['stage'] in {'finished','failed'}:
                    break
                time.sleep(.02)
            assert turn['result']['execution']['status']=='completed'
            rows = packages(lab, {'session':'angelos:'+cid, 'task_id':task_id})
            assert json.loads(rows[0]['content'])['evaluation_only'] is True
            assert key in json.dumps(runtime.calls[before:], default=str)
        assert not lab.plugin.recall(scope, '代码', current=ExperienceContext(scope, frozenset()))['items']
        with lab.learning.db() as db:
            assert lab.learning.row(db, scope, key)[1]=='candidate'


@pytest.mark.parametrize('fault', ['oversized', 'exception', 'switch'])
def test_optional_experience_fault_never_fails_task(root, fault):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        original = lab.plugin.for_task
        def lookup(*args, **kwargs):
            if fault=='exception':
                raise RuntimeError('do not expose this raw error')
            result = original(*args, **kwargs)
            if fault=='oversized':
                result['items'][0]['conditions']='X'*5000
            if fault=='switch':
                lab.controls.select('off', lab.controls.snapshot()['revision'])
            return result
        lab.plugin.for_task = lookup
        ref, turn = send(lab, wid, '解释代码', '结果。')
        assert not packages(lab, ref)
        assert turn['result']['execution']['status']=='completed'
        assert 'do not expose' not in json.dumps(turn)
        task = lab.host.service.task_store.get(ref['task_id'], requester=lab.host.requester, session=ref['session'])
        if fault != 'switch':
            gaps = task['material_package']['gaps']
            expected = 'experience_too_large' if fault == 'oversized' else 'experience_unavailable'
            assert any(g['reason'] == expected for g in gaps)


def test_retry_does_not_retrieve_again_or_change_snapshot(root):
    from shinku.agent_runtime.contracts import AgentTurnRequest
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        ref, _ = send(lab, wid, '解释代码', '结果。')
        rows = packages(lab, ref)
        lab.controls.select('off', 1)
        task = lab.host.service.task_store.get(ref['task_id'], requester=lab.host.requester, session=ref['session'])
        request = AgentTurnRequest('offline-turn', task['requests'][0]['text'],
            constraints=tuple(task['confirmed_constraints']),
            sources=tuple({k:v for k,v in s.items() if k!='source'} for s in task['source_data']),
            history_context=task.get('history_context',''))
        def forbidden(*args):
            raise AssertionError('idempotent submit must not retrieve')
        lab.plugin.for_task = forbidden
        before = len(runtime.calls)
        handle = lab.host.port(ref['session'][8:]).submit(request)
        assert handle.task_id==ref['task_id'] and len(runtime.calls)==before
        assert packages(lab, ref)==rows
