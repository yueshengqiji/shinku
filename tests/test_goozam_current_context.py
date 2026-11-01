"""真实submit/经验工具入口回归：来源结果不等于本轮前提，经验不授予权限。"""
from dataclasses import replace
import json

import pytest

from goozam.experience_context import ExperienceContext
from goozam.host import experiment
from goozam.learning import Assessment
from goozam.offline import OfflineRuntime, send
from shinku.agent_runtime import AgentScope
from test_goozam_delivery import seed, packages
from test_goozam_host import root


def stored_task(lab, ref):
    return lab.host.service.task_store.get(ref['task_id'], requester=lab.host.requester,
                                         session=ref['session'])


def test_submit_uses_current_grant_before_task_persistence(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        observed = []
        original = lab.plugin.for_task
        def lookup(binding, task_id, query, *, current):
            # 此处任务尚未落库，不能从任务的host_grant反推可用工具。
            assert lab.host.service.task_store.get(task_id, requester=lab.host.requester,
                session=next(iter(sessions))) is None
            observed.append(current)
            return original(binding, task_id, query, current=current)
        sessions = []
        authorize = lab.host.service.authorize
        def capture(scope, task, operation):
            if operation == 'submit':
                assert 'host_grant' not in task
                sessions.append(scope.session)
            return authorize(scope, task, operation)
        lab.host.service.authorize = capture
        lab.plugin.for_task = lookup
        before = len(runtime.calls)
        ref, turn = send(lab, wid, '解释代码', '静态结果。')
        task = stored_task(lab, ref)
        assert len(observed) == 1
        assert observed[0].workspace == scope
        assert observed[0].allowed_tools == frozenset(task['host_grant']['allowed_tools'])
        assert 'workspace_read_lines' in observed[0].allowed_tools
        row = json.loads(packages(lab, ref)[0]['content'])
        assert row['items'][0]['id'] == key
        assert row['items'][0]['source_observations']['execution_status'] == 'completed'
        assert row['items'][0]['conditions'] == '没有执行回执'
        assert row['items'][0]['exceptions'] == '有真实执行回执时据实报告。'
        assert row['grants_permissions'] is False
        assert not turn['result']['evidence']
        prompts = json.dumps(runtime.calls[before:], ensure_ascii=False, default=str)
        assert key in prompts and '有真实执行回执时据实报告。' in prompts


@pytest.mark.parametrize('fault', ['missing', 'wrong_workspace'])
def test_missing_or_foreign_context_is_explicit_gap_not_keyword_fallback(root, fault):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, _ = seed(lab)
        lab.controls.select('use', 0)
        original = lab.plugin.for_task
        def lookup(binding, task_id, query, *, current):
            bad = None if fault == 'missing' else replace(current, workspace='foreign')
            return original(binding, task_id, query, current=bad)
        lab.plugin.for_task = lookup
        ref, turn = send(lab, wid, '解释代码', '正常回答。')
        assert not packages(lab, ref)
        assert any(g['reason'] == 'experience_context_unavailable'
                   for g in stored_task(lab, ref)['material_package']['gaps'])
        assert turn['result']['execution']['status'] == 'completed'


def test_current_grant_keeps_full_tool_set_and_filters_policy(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, _, _ = seed(lab)
        ref, _ = send(lab, wid, '解释代码', '结果。')
        task = stored_task(lab, ref)
        grant = lab.host.service.authorize(AgentScope(lab.host.requester, ref['session']), task, 'inspect')
        names = frozenset(f'tool_{n:02}' for n in range(30))
        grant = replace(grant, allowed_tools=names,
                        policy=replace(grant.policy, allowed=names, blocked=frozenset({'tool_03'})))
        current = ExperienceContext.from_grant('workspace', grant)
        assert current.allowed_tools == names - {'tool_03'}
        assert 'tool_29' in current.allowed_tools


def test_on_demand_tool_uses_current_context_and_does_not_grant_write(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        original = lab.plugin.recall
        seen = []
        def lookup(binding, query, *, current=None):
            seen.append(current)
            return original(binding, query, current=current)
        lab.plugin.recall = lookup
        runtime.responses = [
            {'tool_call': {'type': 'read_goozam_experiences', 'query': '解释代码'}},
            {'kind': 'final', 'text': '只参考经验，没有写文件。', 'task_outcome': 'completed'}]
        before = len(runtime.calls)
        ref, turn = send(lab, wid, '按需读取解释代码经验', '')
        assert len(seen) >= 2  # 首次材料与按需读取均使用宿主上下文。
        assert all(c.valid_for(scope) and 'read_goozam_experiences' in c.allowed_tools for c in seen)
        assert all('workspace_apply_patch' not in c.allowed_tools for c in seen)
        assert any(e['tool'] == 'read_goozam_experiences' and e['execution_status'] == 'ok'
                   for e in turn['result']['evidence'])
        assert key in json.dumps(runtime.calls[before:], ensure_ascii=False, default=str)
        assert not list(lab.workspace.iterdir())
        assert stored_task(lab, ref)['contract'] is None
        assert lab.host.permissions.level == 'he'


def test_source_invalid_and_other_workspace_do_not_inject(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid, scope, key = seed(lab)
        lab.controls.select('use', 0)
        other = root / 'other-workspace'
        other.mkdir()
        other_id = lab.workbench.store.workspace({'name': 'other', 'path': str(other)})['id']
        ref, _ = send(lab, other_id, '解释代码', '结果。')
        assert not packages(lab, ref)
        # 来源指纹失效应单独标记，不假称没有关键词匹配。
        with lab.learning.db() as db:
            payload, _, _ = lab.learning.row(db, scope, key)
            payload['source']['fingerprint'] = 'stale'
            db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), key))
        ref, turn = send(lab, wid, '解释代码', '结果。')
        assert not packages(lab, ref)
        assert any(g['reason'] == 'experience_source_unavailable'
                   for g in stored_task(lab, ref)['material_package']['gaps'])
        assert turn['result']['execution']['status'] == 'completed'


def test_failed_source_experience_reaches_new_task_without_repeating_failure(root):
    runtime = OfflineRuntime()
    with experiment(root, runtime) as lab:
        lab.runtime = runtime
        wid = lab.workbench.store.workspace({'name': 'failure', 'path': str(lab.workspace)})['id']
        runtime.responses = [{'fixture': True}]
        source, failed = send(lab, wid, '训练部署诊断', '')
        assert failed['result']['execution']['status'] == 'failed'
        from shinku.angelos.task_continuity import workspace_binding
        _, scope = workspace_binding(lab.workbench.store, source['session'])
        key = lab.learning.propose(scope, source, situation='部署诊断', conditions='先核对实例',
            advice='先取实例证据再判断。', exceptions='有当前实例证据时复用。')
        base, _ = send(lab, wid, '留出部署诊断', '基线。')
        trial, _ = send(lab, wid, '留出部署诊断', '实例证据。')
        lab.learning.evaluate(scope, key, lambda _: Assessment(base, trial, 0, 1, True, 'synthetic'))
        lab.learning.activate(scope, key)
        lab.controls.select('use', 0)
        ref, turn = send(lab, wid, '部署诊断', '正常交付。')
        row = json.loads(packages(lab, ref)[0]['content'])['items'][0]
        assert row['source_observations']['execution_status'] == 'failed'
        assert row['conditions'] == '先核对实例'
        assert turn['result']['execution']['status'] == 'completed'
        assert stored_task(lab, source)['execution_result'] == failed['result']
