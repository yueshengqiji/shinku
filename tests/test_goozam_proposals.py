"""候选生成是有来源的建议，跳过/错误/重复不会改业务结论或自动启用。"""
from copy import deepcopy
import json

import pytest
from goozam.learning import LearningStore
from goozam.offline import OfflineRuntime
from goozam.proposals import ProposalRunner


def reply():
    return dict(decision='propose', reason='值得在新任务验证', support=['/execution'],
        proposal=dict(situation='部署分析', conditions='用户询问实际运行状态',
                      advice='先定位进程加载路径再读取源码', exceptions='只讨论设计时无需查进程'))


@pytest.fixture
def fixture(tmp_path):
    source = dict(scope='scope', fingerprint='v1', input_hash='input', conditions_hash='conditions',
                  accepted=False, result=dict(execution={'status':'partial'}, execution_report={'text':'尚未完成'}))
    def resolve(scope, ref):
        if scope != 'scope' or ref != {'id':'seed'}:
            raise ValueError('goozam_missing_source')
        return deepcopy(source)
    store = LearningStore(tmp_path/'learning', resolve)
    runtime = OfflineRuntime()
    runtime.responses = [reply()]
    runner = ProposalRunner(store, tmp_path/'runs', runtime, {'model':'fixture'})
    return runner, source, runtime


def test_real_candidate_is_not_activated_and_duplicate_does_not_call(fixture):
    runner, source, runtime = fixture
    before = deepcopy(source)
    result = runner.propose('scope', {'id':'seed'})
    assert result['status']=='candidate' and not result['activated']
    assert runner.propose('scope', {'id':'seed'}) == result
    assert len(runtime.calls)==1 and source==before
    assert not runner.learning.recall('scope', '部署')['items']
    with runner.learning.db() as db:
        assert db.execute('SELECT status FROM candidates').fetchone()[0]=='candidate'
    folder=runner.root/result['run_id']
    assert json.loads((folder/'source.json').read_text())['material']==source['result']
    assert json.loads((folder/'response.json').read_text())==reply()


def test_skip_is_persisted_without_candidate(fixture):
    runner, _, runtime = fixture
    runtime.responses=[dict(decision='skip',reason='单次外部余额故障无需操作经验')]
    result=runner.propose('scope',{'id':'seed'})
    assert result['status']=='skipped'
    assert runner.propose('scope',{'id':'seed'})==result and len(runtime.calls)==1
    with runner.learning.db() as db:
        assert db.execute('SELECT count(*) FROM candidates').fetchone()[0]==0


@pytest.mark.parametrize('change', ['invented_pointer','permission','bad_json','oversized'])
def test_invalid_or_unbounded_material_preserves_evidence(fixture,change):
    runner, source, runtime = fixture
    value=reply()
    if change=='invented_pointer': value['support']=['/evidence/99']
    if change=='permission': value['activate']=True
    if change=='bad_json': value={'text':'不符合协议'}
    if change=='oversized': source['result']['original_requests']='x'*128001
    runtime.responses=[value]
    result=runner.propose('scope',{'id':'seed'})
    assert result['status']=='failed'
    assert runner.propose('scope',{'id':'seed'})==result
    assert len(runtime.calls)==(0 if change=='oversized' else 1)
    with runner.learning.db() as db:
        assert db.execute('SELECT count(*) FROM candidates').fetchone()[0]==0


def test_source_changes_during_model_call_cannot_be_proposed(fixture):
    runner, source, runtime=fixture
    def changed(**kwargs):
        source['fingerprint']='v2'
        return reply()
    runtime.call_chat_json=changed
    result=runner.propose('scope',{'id':'seed'})
    assert result['status']=='failed' and 'changed_during' in result['error']


def test_unknown_interruption_never_reissues(fixture):
    runner, _, runtime=fixture
    def interrupted(**kwargs):
        raise KeyboardInterrupt()
    runtime.call_chat_json=interrupted
    with pytest.raises(KeyboardInterrupt): runner.propose('scope',{'id':'seed'})
    with pytest.raises(ValueError,match='requires_review'): runner.propose('scope',{'id':'seed'})


def test_batch_prechecks_all_scopes_before_any_model_call(fixture):
    runner, _, runtime=fixture
    with pytest.raises(ValueError): runner.batch('scope',[{'id':'seed'},{'id':'foreign'}])
    assert not runtime.calls


def test_actual_task_source_to_candidate_without_rewriting_result(tmp_path):
    from goozam.host import experiment
    from goozam.offline import send
    from shinku.angelos.task_continuity import workspace_binding
    for name in ('data','workspace','learning','evaluations','logs'): (tmp_path/name).mkdir()
    runtime=OfflineRuntime()
    with experiment(tmp_path,runtime) as lab:
        lab.runtime=runtime
        wid=lab.workbench.store.workspace({'name':'合成任务','path':str(lab.workspace)})['id']
        ref,turn=send(lab,wid,'解释合成程序，不运行','静态推导结果为2。未运行。')
        _,scope=workspace_binding(lab.workbench.store,ref['session'])
        before=lab.learning.resolve(scope,ref)
        runtime.responses=[reply()]
        runner=ProposalRunner(lab.learning,tmp_path/'learning/proposals',runtime,{'model':'fixture'})
        result=runner.propose(scope,ref)
        assert result['status']=='candidate'
        assert lab.learning.resolve(scope,ref)==before
        material=json.loads(runtime.calls[-1]['user_prompt'])['material']
        assert material['original_requests'][0]['text']=='解释合成程序，不运行'
        assert material['execution_report']['text']=='静态推导结果为2。未运行。'
        assert not lab.learning.recall(scope,'部署')['items']
