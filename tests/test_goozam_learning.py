"""经验状态机的独立规则：无评估不启用、来源变化即停止使用、撤销不可复活。"""
from copy import deepcopy
import json
import sqlite3

import pytest

from goozam.learning import Assessment, LearningStore
from goozam.experience_context import task_observations, source_observations, ExperienceContext


@pytest.fixture
def store(tmp_path):
    sources = {name: {'scope': 'scope', 'fingerprint': name, 'input_hash': text,
                     'conditions_hash': 'conditions', 'accepted': True}
               for name, text in [('seed', 'learn'), ('base', 'holdout'), ('trial', 'holdout')]}
    def resolve(scope, ref):
        if scope != 'scope' or ref['id'] not in sources:
            raise ValueError('missing')
        return deepcopy(sources[ref['id']])
    value = LearningStore(tmp_path / 'learning', resolve)
    return value, sources


def propose(value):
    return value.propose('scope', {'id': 'seed'}, situation='代码解释', conditions='只读',
                         advice='注明静态推导', exceptions='实际执行另按回执')


def assessment(**overrides):
    args = dict(baseline={'id': 'base'}, candidate={'id': 'trial'}, baseline_score=0,
                candidate_score=1, safety_passed=True, notes='合成规则')
    return Assessment(**(args | overrides))


def activate(value, key):
    assert value.evaluate('scope', key, lambda p: assessment()) == 'evaluated'
    value.activate('scope', key)


def test_full_cycle_idempotence_restart_and_revoke(store):
    value, _ = store
    key = propose(value)
    assert propose(value) == key
    assert not value.recall('scope', '代码')['items']
    with pytest.raises(ValueError, match='evaluation_required'):
        value.activate('scope', key)
    activate(value, key)
    reopened = LearningStore(value.path.parent, value.resolve)
    result = reopened.recall('scope', '代码')
    assert len(result['items']) == 1 and not result['is_execution_evidence'] and not result['grants_permissions']
    reopened.revoke('scope', key)
    assert not value.recall('scope', '代码')['items']
    assert propose(value) == key
    with pytest.raises(ValueError, match='evaluation_required'):
        value.activate('scope', key)


@pytest.mark.parametrize('changes', [{'candidate_score': 0}, {'safety_passed': False}])
def test_no_gain_or_safety_failure_never_promotes(store, changes):
    value, _ = store
    key = propose(value)
    assert value.evaluate('scope', key, lambda p: assessment(**changes)) == 'no_gain'
    with pytest.raises(ValueError, match='evaluation_required'):
        value.activate('scope', key)


@pytest.mark.parametrize('name', ['seed', 'base', 'trial'])
def test_source_drift_disables_use_without_rewriting_old_evidence(store, name):
    value, sources = store
    key = propose(value)
    activate(value, key)
    sources[name]['fingerprint'] = 'changed'
    result = value.recall('scope', '代码')
    assert not result['items'] and result['source_gaps'] == [key]
    with value.db() as db:
        assert db.execute('SELECT count(*) FROM audits').fetchone()[0] == 3


@pytest.mark.parametrize('field,new', [('input_hash', 'learn'), ('conditions_hash', 'different')])
def test_holdout_must_be_distinct_and_comparable(store, field, new):
    value, sources = store
    key = propose(value)
    sources['trial'][field] = new
    with pytest.raises(ValueError, match='holdout_required'):
        value.evaluate('scope', key, lambda p: assessment())


def test_cross_scope_and_missing_source(store):
    value, sources = store
    key = propose(value)
    activate(value, key)
    assert not value.recall('another', '代码')['items']
    with pytest.raises(ValueError, match='not_found'):
        value.revoke('another', key)
    del sources['seed']
    assert value.recall('scope', '代码')['source_gaps'] == [key]


def test_revoke_during_evaluation_cannot_be_overwritten(store):
    value, _ = store
    key = propose(value)
    def evaluator(p):
        value.revoke('scope', key)
        return assessment()
    with pytest.raises(ValueError, match='concurrent_change'):
        value.evaluate('scope', key, evaluator)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 2, True])
def test_invalid_scores_are_rejected(store, score):
    value, _ = store
    key = propose(value)
    with pytest.raises(ValueError, match='invalid_score'):
        value.evaluate('scope', key, lambda p: assessment(candidate_score=score))


def test_batch_bound_and_foreign_database(store, tmp_path):
    value, _ = store
    with pytest.raises(ValueError, match='batch_limit'):
        value.batch('scope', [{'id': 'seed'}] * 4, lambda s: {})
    root = tmp_path / 'foreign'
    root.mkdir()
    with sqlite3.connect(root / 'learning.sqlite3') as db:
        db.execute('create table production (secret text)')
    with pytest.raises(ValueError, match='foreign_database'):
        LearningStore(root, value.resolve)


def test_proposal_source_change_and_no_full_transcript_in_recall(store):
    value, sources = store
    def changed(case):
        sources['seed']['fingerprint'] = 'new'
        return dict(situation='代码解释', conditions='只读', advice='静态推导', exceptions='实测除外')
    with pytest.raises(ValueError, match='changed_during_proposal'):
        value.batch('scope', [{'id': 'seed'}], changed)
    sources['seed']['result'] = {'long_original': 'x' * 10000}
    key = propose(value)
    activate(value, key)
    assert 'long_original' not in str(value.recall('scope', '代码'))


def test_task_observations_only_derive_from_ledger_result():
    """观察只能从任务结论/回执派生；读不到就是空，不能凭空补一个默认值。"""
    assert task_observations(None) == {}
    assert task_observations({}) == {}
    result = {
        'execution': {'status': 'ok'},
        'evidence': [
            {'tool': 'workspace_read_lines', 'execution_status': 'ok'},
            {'tool': 'private_shell', 'execution_status': 'error'},
            {'tool': '', 'execution_status': 'ok'},
            'not-a-dict',
        ],
    }
    anchors = task_observations(result)
    assert anchors['execution_status'] == 'ok'
    assert anchors['tools'] == ['private_shell', 'workspace_read_lines']
    assert anchors['failed_tools'] == ['private_shell']
    # 回执正文不进观察，避免把工具输出抄进经验库。
    assert 'ok' not in anchors['tools']


@pytest.mark.parametrize('status', ['completed', 'failed'])
def test_historical_outcome_does_not_require_current_outcome(store, status):
    value, sources = store
    sources['seed']['result'] = {'execution': {'status': status},
        'evidence': [{'tool': 'old_tool', 'execution_status': 'failed'}]}
    key = propose(value)
    activate(value, key)
    row = value.recall('scope', '代码')['items'][0]
    # 新任务只有另一套工具，没有完成状态；仍可参考旧经验，未签发任何权限。
    projected = ExperienceContext('scope', frozenset({'different_tool'})).project(row)
    assert projected['id'] == key
    assert projected['source_observations']['execution_status'] == status
    assert projected['tool_context']['unavailable_source_tools'] == ['old_tool']
    assert projected['tool_context']['is_execution_authorization'] is False
    assert value.recall('another-scope', '代码')['items'] == []
    value.revoke('scope', key)
    assert value.recall('scope', '代码')['items'] == []


@pytest.mark.parametrize('legacy', ['anchors', 'no_anchors'])
def test_legacy_projection_keeps_database_and_assessment_unchanged(store, legacy):
    value, _ = store
    key = propose(value)
    activate(value, key)
    with value.db() as db:
        payload, _, _ = value.row(db, 'scope', key)
        payload.pop('observation_version')
        payload.pop('source_observations')
        if legacy == 'anchors':
            payload['anchors'] = {'execution_status': 'failed', 'tools': ['old_tool']}
        raw = json.dumps(payload, ensure_ascii=False)
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (raw, key))
        before = db.execute('SELECT * FROM candidates').fetchall()
        audits = db.execute('SELECT * FROM audits').fetchall()
    reopened = LearningStore(value.path.parent, value.resolve)
    row = reopened.recall('scope', '代码')['items'][0]
    assert row['source_observations'] == payload.get('anchors', {})
    assert row['assessment'] == payload['assessment']
    with value.db() as db:
        assert db.execute('SELECT * FROM candidates').fetchall() == before
        assert db.execute('SELECT * FROM audits').fetchall() == audits


def test_unknown_observation_version_is_a_source_gap(store):
    value, _ = store
    key = propose(value)
    activate(value, key)
    with value.db() as db:
        payload, _, _ = value.row(db, 'scope', key)
        payload['observation_version'] = 99
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), key))
    result = value.recall('scope', '代码')
    assert not result['items'] and result['source_gaps'] == [key]


def test_active_limit_is_transactional(store):
    value, sources = store
    for index in range(6):
        sources['seed']['fingerprint'] = str(index)
        key = propose(value)
        value.evaluate('scope', key, lambda p: assessment())
        if index < 5:
            value.activate('scope', key)
        else:
            with pytest.raises(ValueError, match='active_limit'):
                value.activate('scope', key)


def test_reproposal_of_old_record_without_observations_does_not_rewrite(store):
    value, sources = store
    sources['seed']['result'] = {'execution': {'status': 'completed'}}
    key = propose(value)
    with value.db() as db:
        payload, _, _ = value.row(db, 'scope', key)
        payload.pop('observation_version')
        payload.pop('source_observations')
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), key))
        before = db.execute('SELECT * FROM candidates').fetchall()
    assert propose(value) == key
    with value.db() as db:
        assert db.execute('SELECT * FROM candidates').fetchall() == before
