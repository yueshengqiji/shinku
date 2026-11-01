"""真实隔离仓库、AgentService、文件工具、无网络沙箱与 Git 回执的维护回归。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from shinku.agent_runtime.execution import ExecutionProfile
from shinku.angelos.app import create_angelos_app
from shinku.maintenance import FailureEvent, MaintenanceHost, RepairAgent, RepairPolicy, RepairQueue, classify_failure
from shinku.maintenance.agent import MaintenanceAccess
from shinku.maintenance.host import task_failures, trace_failures
from shinku.maintenance.workspace import CandidateWorkspace, git, sha


SOURCE = 'def value():\n    return 1\n'
PLACEHOLDER = 'def test_original_failure():\n    raise AssertionError("replace with a reproducible regression")\n'


@pytest.fixture
def project(tmp_path):
    source = tmp_path / 'project'
    (source / 'src/shinku').mkdir(parents=True)
    (source / 'src/shinku/sample.py').write_text(SOURCE)
    (source / 'tests').mkdir()
    (source / 'tests/test_existing.py').write_text('from shinku.sample import value\ndef test_existing():\n    assert isinstance(value(), int)\n')
    (source / 'AGENTS.md').write_text('保留业务数据，测试必须能复现原问题。\n')
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Fixture')
    git(source, 'config', 'user.email', 'fixture@localhost')
    git(source, 'add', '--', '.')
    git(source, 'commit', '-qm', 'Fixture baseline')
    secret = source / 'private/secret.txt'
    secret.parent.mkdir()
    secret.write_text('preserve private fixture')
    policy = RepairPolicy(source, tmp_path / 'maintenance', ('tests/test_existing.py',), lambda: True,
                          max_seconds=60, max_requests=12)
    return policy, secret


def event(policy, *, code='conversation_output_invalid', source='angelos', **kwargs):
    return FailureEvent(source, 'fixture-event', code, policy.version(), time.time(), **kwargs)


@pytest.mark.parametrize('code,diagnostics,expected', [
    ('conversation_output_invalid', {}, 'repair'),
    ('bridge_error:TypeError', {}, 'repair'),
    ('call_timeout', {}, 'diagnosis_only'),
    ('agent_permission_changed', {}, 'diagnosis_only'),
    ('budget_exhausted', {}, 'diagnosis_only'),
    ('visual_source_changed', {}, 'diagnosis_only'),
    ('direct_chat_reply', {'delivery_status': 'uncertain'}, 'delivery_review'),
    ('direct_chat_reply', {'delivery_status': 'failed'}, 'delivery_review'),
    ('unknown_error', {}, 'diagnosis_only'),
])
def test_classification_never_turns_uncertain_delivery_or_access_into_code_repair(project, code, diagnostics, expected):
    policy, _ = project
    assert classify_failure(event(policy, code=code, diagnostics=diagnostics)) == expected


def test_queue_deduplicates_concurrent_faults_across_messages_and_restart(project):
    policy, _ = project
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    original = event(policy)
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(lambda i: queue.enqueue(replace(original, event_id=str(i))), range(20)))
    assert len(set(keys)) == 1 and len(queue.snapshot()) == 1
    assert queue.enqueue(replace(original, source='qq')) == keys[0]
    assert queue.enqueue(replace(original, maintenance=True)) is None
    key, retained = queue.claim()
    queue.close()
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    queue.interrupted()
    assert queue.claim() is None and queue.snapshot()[0]['status'] == 'interrupted'
    assert queue.enqueue(original) == key
    queue.close()


def test_host_preserves_failed_job_and_never_recursively_repairs_it(project):
    policy, _ = project
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    calls = []
    def failed(*args):
        calls.append(args[0])
        raise RuntimeError('provider detail must not enter safe status')
    host = MaintenanceHost(queue, policy, failed, collect=lambda *_: [event(policy)])
    try:
        host.tick(); host.tick()
        assert len(calls) == 1 and queue.snapshot()[0]['status'] == 'failed'
        assert queue.snapshot()[0]['result'] == {'reason': 'RuntimeError'}
    finally:
        host.stop(); queue.close()


def test_second_host_does_not_interrupt_live_claim(project):
    policy, _ = project
    q1 = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    q2 = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    first = MaintenanceHost(q1, policy, lambda *_: {}, collect=lambda *_: [])
    second = MaintenanceHost(q2, policy, lambda *_: {}, collect=lambda *_: [])
    try:
        first._acquire()
        q1.enqueue(event(policy)); key, _ = q1.claim()
        with pytest.raises(BlockingIOError): second.tick()
        assert q1.snapshot()[0]['status'] == 'running'
        q1.finish(key, {}, status='interrupted')
    finally:
        first.stop(); second.stop(); q1.close(); q2.close()


def test_collectors_require_terminal_host_state_and_exclude_maintenance(project):
    policy, _ = project
    trace = dict(trace_id='t', started_at=1, finished_at=11, status='failed', reason='conversation_output_invalid',
                 delivery_status='sent', stages=[dict(name='agent_delegation', task_id='task')])
    rows = [trace, dict(trace, trace_id='running', status='running', finished_at=None),
            dict(trace, trace_id='silent', status='silent'), dict(trace, trace_id='old', finished_at=9)]
    faults = list(trace_failures(rows, revision=policy.version(), not_before=10))
    assert len(faults) == 1 and faults[0].diagnostics['task_ids'] == ['task']
    task = dict(task_id='task', status='failed', updated=11, stop_reason='conversation_output_invalid',
                requester='owner', session='private:fixture')
    assert len(list(task_failures([task, dict(task, requester='host-maintenance'),
        dict(task, session='maintenance:fixture')], revision=policy.version(), not_before=10))) == 1


class ScriptedRepair:
    """仅供应商决策为合成；所有文件修改、任务执行、测试与提交均走真实实现。"""
    def __init__(self, secret):
        self.calls, self.step, self.secret = [], 0, secret

    def chat_supports_native_tools(self): return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs['user_prompt'].replace('\\"', '"')
        regression = re.search(r'tests/test_maintenance_[0-9a-f]+\.py', prompt).group()
        test = ('from shinku.sample import value\nimport os, socket\nimport pytest\n'
            'def test_original_failure():\n    assert value() == 2\n'
            'def test_sandbox_boundary():\n'
            '    assert not os.path.exists(' + repr(str(self.secret)) + ')\n'
            '    with pytest.raises(OSError):\n        open("/workspace/src/shinku/sample.py", "w")\n'
            '    with pytest.raises(OSError):\n        os.symlink(' + repr(str(self.secret)) + ', "/results/output.log")\n'
            '    with pytest.raises(OSError):\n        socket.create_connection(("127.0.0.1", 9998), timeout=.1)\n')
        index = self.step; self.step += 1
        if index in {0, 2}:
            path, old, new = ('src/shinku/sample.py', SOURCE, SOURCE.replace('return 1', 'return 2')) if index == 0 else (regression, PLACEHOLDER, test)
            return {'tool_call': dict(type='workspace_preview_patch', path=path,
                expected_sha256=hashlib.sha256(old.encode()).hexdigest(), edits=[dict(old=old, new=new)])}
        if index in {1, 3}:
            path, old = ('src/shinku/sample.py', SOURCE) if index == 1 else (regression, PLACEHOLDER)
            ids = re.findall(r'"preview_id"\s*:\s*"([0-9a-f]{32})"', prompt)
            assert ids, prompt[-5000:]
            return {'tool_call': dict(type='workspace_apply_patch', path=path,
                expected_sha256=hashlib.sha256(old.encode()).hexdigest(), preview_id=ids[-1])}
        if index == 4:
            return {'tool_call': {'type': 'maintenance_test'}}
        return {'kind': 'final', 'text': '候选改动与测试结果见回执。', 'task_outcome': 'completed'}


def test_actual_agent_to_sandbox_and_verified_local_commit_preserves_original(project):
    policy, secret = project
    root_head, before = git(policy.source_root, 'rev-parse', 'HEAD'), sha(secret)
    runtime = ScriptedRepair(secret)
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    host = MaintenanceHost(queue, policy, RepairAgent(policy, ExecutionProfile(runtime)),
        collect=lambda *_: [event(policy)])
    try:
        host.tick()
        item = queue.snapshot()[0]
        assert item['status'] == 'candidate_ready', item
        receipt = item['result']
        # 完成门可额外审查模型文字；去重要求的是下一次扫描不再调用模型。
        calls = len(runtime.calls)
        assert 6 <= calls <= policy.max_requests
        assert receipt['deployed'] is False and receipt['original_task_replayed'] is False
        assert receipt['messages_sent'] == 0
        assert sorted(receipt['files']) == ['src/shinku/sample.py', next(n for n in receipt['files'] if n.startswith('tests/'))]
        assert git(Path(receipt['candidate']), 'rev-parse', 'HEAD') == receipt['commit']
        checks = {c['label']:c for c in receipt['checks']}
        assert checks['host-original']['exit'] == checks['host-fixed']['exit'] == 0
        assert checks['host-regression-original']['failures'] == 1
        assert (policy.source_root / 'src/shinku/sample.py').read_text() == SOURCE
        assert git(policy.source_root, 'rev-parse', 'HEAD') == root_head and sha(secret) == before
        assert not (Path(receipt['candidate']) / 'private').exists()
        host.tick()
        assert len(runtime.calls) == calls
    finally:
        host.stop(); queue.close()


def test_candidate_rejects_existing_test_change_and_source_drift(project):
    policy, _ = project
    candidate = CandidateWorkspace(policy, 'a'*64, policy.version())
    access = MaintenanceAccess(candidate, lambda *_: True)
    with pytest.raises(PermissionError):
        access.replace('tests/test_existing.py', '', b'', {})
    (candidate.tree / 'tests/test_existing.py').write_text('def test_cheat(): pass\n')
    with pytest.raises(ValueError, match='outside_scope'): candidate.changes()
    git(policy.source_root, 'commit', '--allow-empty', '-qm', 'Concurrent source change')
    with pytest.raises(ValueError, match='version_changed'): candidate.check()


def test_revoked_authorization_blocks_candidate_creation(project):
    policy, _ = project
    policy = replace(policy, authorized=lambda: False)
    with pytest.raises(PermissionError): CandidateWorkspace(policy, 'b'*64, policy.version())


def test_deadline_is_shared_and_distinct_from_authorization(project):
    policy, _ = project
    candidate = CandidateWorkspace(policy, 'd'*64, policy.version())
    candidate.deadline = time.monotonic() - 1
    assert policy.authorized()
    with pytest.raises(TimeoutError, match='deadline'):
        candidate.tests(candidate.tree, policy.test_targets, 'late-test')
    assert not (candidate.home / 'late-test').exists()


def test_stopping_during_model_call_does_not_create_commit_or_restart_repair(project):
    policy, secret = project
    entered, release = threading.Event(), threading.Event()
    class PausedRuntime:
        def chat_supports_native_tools(self): return True
        def call_chat_json(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return {'kind': 'final', 'text': '没有执行修改。', 'task_outcome': 'failed'}
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    host = MaintenanceHost(queue, policy, RepairAgent(policy, ExecutionProfile(PausedRuntime())),
                           collect=lambda *_: [event(policy)], interval=.01)
    try:
        host.start()
        assert entered.wait(3)
        stopper = threading.Thread(target=host.stop)
        stopper.start()
        assert host.stopping.wait(1)
        release.set()
        stopper.join(3)
        host.thread.join(3)
        assert not host.thread.is_alive()
        assert queue.snapshot()[0]['status'] == 'interrupted'
        assert 'commit' not in queue.snapshot()[0]['result']
        assert (policy.source_root / 'src/shinku/sample.py').read_text() == SOURCE
        assert secret.read_text() == 'preserve private fixture'
    finally:
        release.set(); host.stop(); queue.close()


def test_revoked_optional_host_does_not_prevent_normal_app_start(project, tmp_path):
    policy, _ = project
    policy = replace(policy, authorized=lambda: False)
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    maintenance = MaintenanceHost(queue, policy, lambda *_: pytest.fail('revoked runner'), collect=lambda *_: [])
    try:
        with TestClient(create_angelos_app(tmp_path / 'revoked-app', host=SimpleNamespace(maintenance=maintenance))) as client:
            value = client.get('/angelos/api/maintenance').json()['maintenance']
            assert value['error'] == 'authorization_revoked' and not value['running']
    finally:
        maintenance.stop(); queue.close()


def test_daily_limit_retains_diagnosis_and_never_starts_another_repair(project):
    policy, _ = project
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    try:
        for index in range(2):
            queue.enqueue(replace(event(policy), revision=str(index)))
            key, _ = queue.claim(max_per_day=2)
            queue.finish(key, dict(reason='fixture failure'), status='failed')
        queue.enqueue(replace(event(policy), revision='new-version'))
        assert queue.claim(max_per_day=2) is None
        assert sum(r['status'] == 'budget_blocked' for r in queue.snapshot()) == 1
    finally:
        queue.close()


def test_stale_jobs_never_call_runner(project):
    policy, _ = project
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    called = []
    host = MaintenanceHost(queue, policy, lambda *args: called.append(args), collect=lambda *_: [])
    try:
        queue.enqueue(replace(event(policy), observed_at=time.time() - policy.event_ttl_seconds - 1))
        host.tick()
        assert queue.snapshot()[0]['status'] == 'stale' and not called
    finally:
        host.stop(); queue.close()


def test_optional_host_lifecycle_and_read_only_status(project, tmp_path):
    policy, _ = project
    queue = RepairQueue(policy.state_root / 'queue.sqlite', project='fixture')
    maintenance = MaintenanceHost(queue, policy, lambda *_: {}, collect=lambda *_: [], interval=.01)
    host = SimpleNamespace(maintenance=maintenance)
    app = create_angelos_app(tmp_path / 'app', host=host)
    try:
        with TestClient(app) as client:
            state = client.get('/angelos/api/maintenance').json()['maintenance']
            assert state['configured'] and state['running']
            assert client.post('/angelos/api/maintenance', json={}).status_code == 405
            assert client.get('/angelos/api/maintenance', headers={'origin': 'https://example.invalid'}).status_code == 403
        assert maintenance.stopping.is_set() and not maintenance.thread.is_alive()
    finally:
        maintenance.stop(); queue.close()
