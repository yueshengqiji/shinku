"""实际ServiceHost/工作台/账本离线集成，非付费模型或生产部署验收。"""
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from goozam.host import experiment, TaskSources
from goozam.offline import OfflineRuntime, exercise, send
from goozam.sandbox import plain_tree
from shinku.agent_runtime import AgentScope


@pytest.fixture
def root(tmp_path):
    for name in ('data', 'workspace', 'learning', 'evaluations', 'logs'):
        (tmp_path / name).mkdir()
    return tmp_path


def test_actual_offline_cycle(root):
    result = exercise(root)
    assert result['proposed_evaluated_activated_reopened_revoked']
    assert result['original_result_preserved'] and result['model_http_calls'] == 0
    assert result['delegation_disabled'] and result['diagnostic_services_disabled']


def test_host_no_production_routes_identity_and_permissions(root):
    with experiment(root, OfflineRuntime()) as lab:
        with TestClient(lab.app) as client:
            bootstrap = client.get('/angelos/api/bootstrap').json()
            assert bootstrap['agent']['ready']
            page = client.get('/angelos')
            assert 'GOOZAM<br>离线实验' in page.text and 'ANGELOS2' in page.text
            assert client.get('/debug/hub').status_code == 404
            assert client.post('/agent/delegation').status_code == 404
        assert lab.host.requester == 'local-goozam-angelos2'
        assert not lab.host.service.recovery_enabled
        assert lab.host.service.authorize(AgentScope('local-angelos', 'angelos:x'), {}, 'inspect') is None
        grant = lab.host.service._grant(AgentScope(lab.host.requester, 'angelos:x'), {}, 'inspect')
        assert 'workspace_read_lines' in grant.allowed_tools
        assert 'read_goozam_experiences' in grant.allowed_tools
        assert 'workspace_apply_patch' not in grant.allowed_tools
        assert not any('private_' in n for n in lab.host.registry_handlers())
        lab.host.permissions.select('zayin', lab.host.permissions.revision)
        grant = lab.host.service._grant(AgentScope(lab.host.requester, 'angelos:x'), {}, 'inspect')
        assert not grant.allowed_tools


def test_mount_validation_rejects_links_and_special_files(root):
    outside = root / 'outside'
    outside.write_text('synthetic')
    (root / 'workspace/link').symlink_to(outside)
    with pytest.raises(ValueError, match='unsafe_mount'):
        plain_tree(root / 'workspace')


def test_mount_validation_rejects_hard_links(root):
    outside = root / 'outside'
    outside.write_text('synthetic')
    (root / 'workspace/link').hardlink_to(outside)
    with pytest.raises(ValueError, match='unsafe_mount'):
        plain_tree(root / 'workspace')


def test_real_workspace_read_and_source_binding(root):
    from shinku.angelos.task_continuity import workspace_binding
    runtime = OfflineRuntime()
    (root / 'workspace/sample.py').write_text('print(2 + 3)\n')
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        workspace = lab.workbench.store.workspace({'name': 'fixture', 'path': str(lab.workspace)})
        contract = {'goal': '读取代码', 'deliverables': ['实际代码'], 'scope': ['只读工作区'],
            'requirements': [{'id': 'read', 'description': '实际读取', 'effect': 'read',
                              'targets': [str(lab.workspace / 'sample.py')], 'checks': {}}]}
        runtime.responses = [
            {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}},
            {'tool_call': {'type': 'workspace_read_lines', 'path': 'sample.py'}},
            {'kind': 'final', 'text': '已读取sample.py，内容为print(2 + 3)。未运行程序。', 'task_outcome': 'completed'}]
        ref, turn = send(lab, workspace['id'], '读取文件 `' + str(lab.workspace / 'sample.py') + '`', '')
        assert turn['result']['execution']['status'] == 'completed'
        assert any(e['tool'] == 'workspace_read_lines' and e['execution_status'] == 'ok'
                   for e in turn['result']['evidence'])
        _, binding = workspace_binding(lab.workbench.store, ref['session'])
        source = TaskSources(lab.host)
        assert source(binding, ref)['accepted']
        lab.workbench.store.workspace({'name': 'fixture', 'path': str(lab.workspace),
                                       'description': 'changed binding'}, workspace['id'])
        with pytest.raises(ValueError, match='scope_or_revocation'):
            source(binding, ref)
        assert (root / 'workspace/sample.py').read_text() == 'print(2 + 3)\n'


def test_truthful_failure_is_not_a_successful_learning_trial(root):
    from shinku.angelos.task_continuity import workspace_binding
    runtime = OfflineRuntime()
    runtime.responses = [{'kind': 'final', 'text': '未完成此合成任务。', 'task_outcome': 'failed'}]
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name': 'failure-fixture', 'path': str(lab.workspace)})['id']
        ref, turn = send(lab, wid, '离线失败样例', '')
        assert turn['result']['execution']['status'] == 'failed'
        # 原验收接受如实的失败报告，不代表业务成功，学习层必须保持这个区别。
        assert turn['result']['verification']['accepted']
        _, binding = workspace_binding(lab.workbench.store, ref['session'])
        assert not TaskSources(lab.host)(binding, ref)['accepted']


def test_permission_change_is_not_counted_as_learning_gain(root):
    from goozam.learning import Assessment
    from shinku.angelos.task_continuity import workspace_binding
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name': 'conditions', 'path': str(lab.workspace)})['id']
        seed, _ = send(lab, wid, '训练样例', '合成输出')
        _, binding = workspace_binding(lab.workbench.store, seed['session'])
        key = lab.learning.propose(binding, seed, situation='代码', conditions='条件',
                                   advice='建议', exceptions='例外')
        baseline, _ = send(lab, wid, '留出样例', '合成输出')
        lab.host.permissions.select('zayin', lab.host.permissions.revision)
        trial, _ = send(lab, wid, '留出样例', '另一合成输出')
        with pytest.raises(ValueError, match='holdout_required'):
            lab.learning.evaluate(binding, key, lambda p: Assessment(baseline, trial, 0, 1, True, 'fixture'))
