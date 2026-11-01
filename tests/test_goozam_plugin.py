"""可开关插件真实装配验收；不调用付费模型或修改正式实例。"""
from copy import deepcopy
import threading

from fastapi.testclient import TestClient
import pytest

from goozam.host import experiment
from goozam.learning import LearningStore
from goozam.experience_context import ExperienceContext
from goozam.offline import OfflineRuntime, send
from goozam.plugin import Controls, GoozmaPlugin
from shinku.plugins.contracts import PluginContext
from test_goozam_host import root
from test_goozam_proposals import reply


def test_controls_default_persistence_conflict_and_isolation(tmp_path):
    c = Controls(tmp_path / 'one')
    assert c.snapshot() == dict(mode='off', revision=0, enabled=False, use_experiences=False)
    assert c.select('learn', 0)['use_experiences'] is False
    assert Controls(tmp_path / 'one').snapshot()['mode'] == 'learn'
    with pytest.raises(ValueError, match='revision_conflict'):
        c.select('use', 0)
    assert Controls(tmp_path / 'two').snapshot()['mode'] == 'off'
    with pytest.raises(ValueError):
        c.select('on', 1)
    with pytest.raises(ValueError):
        c.select('use', True)


def make_plugin(tmp_path):
    sources = {}
    for i in range(3):
        sources[str(i)] = dict(scope='s', fingerprint=str(i), input_hash=str(i),
            conditions_hash='c', accepted=False, result=dict(execution={'status': 'partial'}))
    store = LearningStore(tmp_path / 'learning', lambda scope, ref: deepcopy(sources[ref['id']]))
    p = GoozmaPlugin()
    p.activate(PluginContext('angelos2', 'local-goozam-angelos2', services={
        'goozam_learning': store, 'goozam_controls': Controls(tmp_path / 'learning'), 'agent_memory': None}))
    return p


def test_off_no_requests_and_only_learning_does_not_activate(tmp_path):
    p = make_plugin(tmp_path)
    runtime = OfflineRuntime()
    runtime.responses = [reply()]
    refs = [{'id': '0'}]
    with pytest.raises(ValueError, match='paused'):
        p.propose('s', refs, runtime, {'model': 'fixture'})
    assert not runtime.calls
    p.controls.select('learn', 0)
    result = p.propose('s', refs, runtime, {'model': 'fixture'})
    assert result[0]['status'] == 'candidate'
    assert not p.recall('s', '部署')['items']
    assert p.propose('s', refs, runtime, {'model': 'fixture'}) == result
    assert len(runtime.calls) == 1
    with p.learning.db() as db:
        assert db.execute('SELECT status FROM candidates').fetchone()[0] == 'candidate'
    with pytest.raises(ValueError, match='evaluation_required'):
        p.learning.activate('s', result[0]['candidate_id'])


@pytest.mark.parametrize('reopen', [False, True])
def test_disable_during_model_keeps_original_and_stops_remaining(tmp_path, reopen):
    p = make_plugin(tmp_path)
    p.controls.select('learn', 0)
    started, finish = threading.Event(), threading.Event()
    runtime = OfflineRuntime()
    def call(**kwargs):
        started.set()
        assert finish.wait(5)
        return reply()
    runtime.call_chat_json = call
    output = []
    t = threading.Thread(target=lambda: output.extend(p.propose('s', [{'id':'0'}, {'id':'1'}], runtime, {'model':'fixture'})))
    t.start()
    assert started.wait(5)
    p.controls.select('off', 1)
    if reopen:
        p.controls.select('learn', 2)
    finish.set()
    t.join(5)
    assert not t.is_alive()
    assert [x['status'] for x in output] == ['candidate', 'paused']
    assert len(list((tmp_path / 'learning/proposals').glob('*/response.json'))) == 1
    assert len(list((tmp_path / 'learning/proposals').glob('*/source.json'))) == 1


def test_use_off_use_rechecks_sources_and_retains_records(tmp_path):
    p = make_plugin(tmp_path)
    p.controls.select('use', 0)
    # 检索内容与状态晋级由原LearningStore测试覆盖，这里验证入口每次都重查且不缓存正文。
    calls = []
    # 库只按作用域和来源检索；当前上下文由插件投影，不改变原排序。
    def recall(scope, query):
        calls.append((scope, query))
        return dict(items=[{'id':'fixture','version':7}], source_gaps=[], grants_permissions=False)
    p.learning.recall = recall
    assert p.recall('s', 'q', current=ExperienceContext('s', frozenset()))['items'][0]['version'] == 7
    p.controls.select('off', 1)
    assert not p.recall('s', 'q', current=ExperienceContext('s', frozenset()))['items'] and len(calls) == 1
    p.controls.select('use', 2)
    assert p.recall('s', 'q', current=ExperienceContext('s', frozenset()))['items'] and len(calls) == 2
    p.close()
    assert not p.recall('s', 'q', current=ExperienceContext('s', frozenset()))['items']


def test_switch_while_lookup_drops_new_material(tmp_path):
    p = make_plugin(tmp_path)
    p.controls.select('use', 0)
    def changed(scope, query):
        p.controls.select('off', 1)
        return dict(items=[{'id': 'old'}])
    p.learning.recall = changed
    assert not p.recall('s', 'q', current=ExperienceContext('s', frozenset()))['items']


def test_plugin_rejects_production_host():
    with pytest.raises(ValueError, match='experiment_host'):
        GoozmaPlugin().activate(PluginContext('shinku', 'production'))


def test_web_controls_guard_revision_and_no_model_calls(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        with TestClient(lab.app) as client:
            assert lab.plugins.diagnostics()['plugins'][0]['status'] == 'active'
            assert '/angelos/goozma' in client.get('/angelos').text
            assert 'Goozma / 实验插件' in client.get('/angelos/goozma').text
            value = client.get('/angelos/api/goozma').json()
            headers = {'x-goozma-token': value['token']}
            assert client.post('/angelos/api/goozma', json={'mode':'learn','revision':0}).status_code == 403
            assert client.post('/angelos/api/goozma', json={'mode':'learn','revision':0},
                headers={**headers, 'origin':'https://example.invalid'}).status_code == 403
            assert client.get('/angelos/api/goozma', headers={'sec-fetch-site':'cross-site'}).status_code == 403
            assert client.post('/angelos/api/goozma', json={'mode':'learn','revision':0}, headers=headers).json()['state']['mode']=='learn'
            assert client.post('/angelos/api/goozma', json={'mode':'off','revision':0}, headers=headers).status_code == 409
            assert client.post('/angelos/api/goozma', json={'mode':'use','revision':1,'activate':'x'}, headers=headers).status_code == 409
            assert client.post('/angelos/api/goozma', content='x'*1025, headers={**headers,'content-type':'application/json'}).status_code == 409
            assert not runtime.calls
        assert lab.controls.snapshot()['mode'] == 'learn'
    with experiment(root, OfflineRuntime()) as reopened:
        assert reopened.controls.snapshot()['mode'] == 'learn'


def test_toggle_does_not_cancel_running_agent(root):
    runtime = OfflineRuntime()
    entered, release = threading.Event(), threading.Event()
    def call(**kwargs):
        entered.set()
        assert release.wait(5)
        return {'kind':'final', 'text':'这是隔离测试结果。', 'task_outcome':'completed'}
    runtime.call_chat_json = call
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name':'fixture','path':str(lab.workspace)})['id']
        result = []
        t = threading.Thread(target=lambda: result.append(send(lab, wid, '解释测试概念', '')))
        t.start()
        assert entered.wait(5)
        lab.controls.select('learn', 0)
        lab.controls.select('off', 1)
        release.set()
        t.join(10)
        assert not t.is_alive()
        assert result[0][1]['result']['execution']['status'] == 'completed'
        assert lab.host.permissions.level == 'he'


def test_off_tool_is_successful_empty_and_does_not_change_authority(root):
    runtime = OfflineRuntime()
    runtime.responses = [
        {'tool_call': {'type':'read_goozam_experiences', 'query':'代码解释'}},
        {'kind':'final','text':'插件关闭，没有读取经验。','task_outcome':'completed'}]
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name':'off-tool','path':str(lab.workspace)})['id']
        _, turn = send(lab, wid, '读取当前工作区经验', '')
        assert any(e['tool']=='read_goozam_experiences' and e['execution_status']=='ok'
                   for e in turn['result']['evidence'])
        assert lab.controls.snapshot()['mode']=='off'
        assert not list((root/'learning').glob('proposals/*'))


def test_synthetic_exercise_restores_off(root):
    from goozam.offline import exercise
    assert exercise(root)['experience_read_via_agent_tool']
    assert Controls(root/'learning').snapshot()['mode']=='off'
