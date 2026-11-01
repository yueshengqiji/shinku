"""进度只读投影与真实执行联测：等待不增加请求，不改变权限、预算或验收。"""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shinku.agent.work_state import input_binding
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls
from shinku.angelos.progress_view import project
from shinku.angelos.runtime_view import AngelosRuntime, ReasoningView


def sample():
    return {'task_id': 'a', 'session': 'angelos:c', 'requester': 'owner',
        'execution_attempt': 2, 'revision': 5, 'updated': 100, 'tool_calls': [],
        'requests': [], 'contract': None}, {'available': True, 'status': 'running',
        'worker_active': True, 'version': 5, 'execution': {}}


def receipt(name='private_file_read', attempt=2, **extra):
    return {'id': 'call1', 'name': name, 'arguments': {'path': '/workspace/sample.py'},
        'status': 'ok', 'executed': True, 'execution_binding': {
            'task_id': 'a', 'session': 'angelos:c', 'attempt': attempt}, **extra}


def test_wait_buckets_do_not_create_events_or_mutate_task():
    task, view = sample(); original = copy.deepcopy(task)
    for now, seconds in [(119, 0), (120, 20), (139, 20), (140, 40)]:
        p = project(task, view, activity={'started': 100}, now=now)
        assert p['phase'] == 'model' and p['waitSeconds'] == seconds
        assert p['events'] == [] and p['is_fact_evidence'] is False
    assert task == original


def test_tools_attempt_scope_and_receipt_are_distinct_from_completion():
    task, view = sample()
    wrong = receipt(); wrong['execution_binding']['session'] = 'angelos:other'
    task['tool_calls'] = [receipt(attempt=1), wrong, receipt(), receipt('private_file_write', status='error')]
    task['inflight'] = [{'name': 'private_shell', 'arguments': {'command': 'SECRET COMMAND'}}]
    p = project(task, view, now=145)
    assert len(p['events']) == 2 and p['phase'] == 'tool'
    assert 'SECRET COMMAND' not in str(p) and '完成' not in str(p)
    assert '/workspace/' not in str(p) and 'sample.py' in str(p)


@pytest.mark.parametrize('status', ['failed', 'canceled', 'completed', 'unverified', 'unknown'])
def test_terminal_and_unknown_stop_waiting_even_with_late_activity(status):
    task, view = sample(); view.update(status=status, worker_active=False)
    p = project(task, view, activity={'started': 1}, now=100)
    assert not p['active'] and p['waitSeconds'] == 0 and p['phase'] == status
    assert '已等待' not in p['text']


def test_approval_cancellation_and_stale_plan():
    task, view = sample(); task['inflight'] = [receipt()]
    pending = [{'taskId': 'a', 'conversationId': 'c', 'attempt': 2, 'created': 100}]
    assert project(task, view, pending=pending, now=145)['phase'] == 'approval'
    view['cancel_requested'] = True
    assert project(task, view, pending=pending, now=145)['phase'] == 'canceling'
    task['work_state'] = {'attempt': 2, 'binding': input_binding(task), 'plan': {
        'current_step': 'read', 'steps': [{'id': 'read', 'goal': '检查入口'}], 'next_action': '核对依赖'}}
    assert project(task, view)['plan']['current'] == '检查入口'
    assert project(task, view)['plan']['verified'] is False
    task['work_state']['attempt'] = 1
    assert not project(task, view)['plan']


def test_runtime_call_activity_is_bound_and_cleared_after_exception():
    entered, release = threading.Event(), threading.Event()
    def call(**kwargs):
        entered.set(); assert release.wait(3)
        raise ValueError('synthetic provider failure')
    view = ReasoningView()
    runtime = AngelosRuntime(SimpleNamespace(call_chat_json=call), view)
    def run():
        with runtime.bind_task('task'):
            with pytest.raises(ValueError): runtime.call_chat_json(system_prompt='')
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(run)
        assert entered.wait(3) and view.activity('task') and view.activity('other') is None
        release.set(); pending.result()
    assert view.activity('task') is None


def test_http_live_progress_tracks_actual_model_tool_and_completion(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    host = ServiceHost(service, permissions=PermissionControls(tmp_path/'permission.json', default='aleph'))
    entered, release = threading.Event(), threading.Event()
    tool_entered, tool_release = threading.Event(), threading.Event()
    original = tool.execute
    def execute(**kwargs):
        tool_entered.set(); assert tool_release.wait(5)
        return original(**kwargs)
    tool.execute = execute
    model_calls = []
    def reply(**kwargs):
        model_calls.append(kwargs)
        if len(model_calls) == 1:
            entered.set(); assert release.wait(5)
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        if not tool.calls:
            return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        return {'kind': 'final', 'text': '文件内容核验通过。', 'task_outcome': 'completed'}
    runtime.call_chat_json = reply
    app = create_angelos_app(tmp_path/'ui', host=host)
    try:
        with TestClient(app) as client:
            token = client.get('/angelos/api/bootstrap').json()['token']
            headers = {'x-angelos-token': token}
            cid = client.post('/angelos/api/conversations', headers=headers, json={}).json()['conversation']['id']
            endpoint = '/angelos/api/conversations/' + cid
            client.post(endpoint+'/messages', headers=headers, json={'text': '生成隔离文件', 'request_key': 'live'})
            assert entered.wait(3)
            for _ in range(3):
                turn = client.get(endpoint).json()['conversation']['turns'][0]
                assert turn['live_progress']['phase'] == 'model'
                live = client.get(endpoint+'/tools/'+turn['id']).json()['activity']['progress']
                assert live['phase'] == 'model' and live['version'] == turn['live_progress']['version']
            assert len(model_calls) == 1 and tool.calls == 0
            assert 'live_progress' not in app.state.angelos.store.read('conversation', cid)['turns'][0]
            assert host.read_progress('other', turn)['available'] is False
            assert client.get(endpoint, headers={'origin': 'https://example.invalid'}).status_code == 403
            release.set(); assert tool_entered.wait(3)
            assert client.get(endpoint).json()['conversation']['turns'][0]['live_progress']['phase'] == 'tool'
            live = client.get(endpoint+'/tools/'+turn['id']).json()['activity']['progress']
            assert live['phase'] == 'tool' and live['active']
            assert not tool.path.exists()
            tool_release.set()
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                turn = client.get(endpoint).json()['conversation']['turns'][0]
                if turn['stage'] == 'finished': break
                time.sleep(.01)
            assert turn['view']['verification']['verified'] is False
            assert turn['result']['verification']['layers']['policy'] == 'agent_reply_delivery'
            assert any(r['receipt_valid'] is True for r in turn['result']['verification']['evidence'])
            assert turn['live_progress']['phase'] == 'completed'
            assert len(turn['live_progress']['events']) == 2
            assert len(model_calls) == 3 and tool.calls == 1
            assert tool.path.read_text() == 'standalone agent verified'
    finally:
        release.set(); tool_release.set(); service.coordinator.shutdown(wait=True)


def test_progress_distinguishes_search_and_background_process_receipts():
    task, view = sample()
    task['tool_calls'] = [receipt('workspace_search'), receipt('private_shell', data={'state_updates': {
        'execution_session': {'status': 'running', 'returncode': None}}})]
    p = project(task, view)
    assert '搜索项目文件' in p['events'][0]['text']
    assert '命令仍在运行' in p['events'][1]['text']
    assert p['version'] == task['revision']
    task['tool_calls'][-1]['data']['state_updates']['execution_session'] = {'status': 'completed', 'returncode': 1}
    p = project(task, view)
    assert '退出码 1' in p['events'][-1]['text'] and '成功' not in p['events'][-1]['text']
