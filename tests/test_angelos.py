"""工作台到真实 Agent 适配器的隔离回归；只执行合成文件操作。"""
import base64
import copy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from angelos_fixture import fixture, FixtureHost
from shinku.angelos.store import WorkbenchStore
from shinku.angelos.workbench import Workbench


@pytest.fixture
def env(tmp_path):
    app, workbench, host = fixture(tmp_path)
    with TestClient(app) as client:
        token = client.get('/angelos/api/bootstrap').json()['token']
        client.headers['x-angelos-token'] = token
        yield client, workbench, host


def post(client, path, data):
    result = client.post('/angelos/api' + path, json=data)
    assert result.status_code == 200, result.text
    return result.json()


def new(client):
    return post(client, '/conversations', {'workspaceId': 'default'})['conversation']['id']


def wait(client, cid, stages=('replied', 'finished', 'failed', 'canceled')):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        c = client.get('/angelos/api/conversations/' + cid).json()['conversation']
        if c['turns'][-1]['stage'] in stages:
            return c
        time.sleep(.02)
    pytest.fail(str(c))


def test_page_assets_and_origin_boundary(env):
    client, _, _ = env
    assert '开始新的一天' in client.get('/angelos').text
    assert '我们开始工作吧，主管' in client.get('/angelos').text
    for name in ('app.js', 'app.css'):
        assert client.get('/angelos/assets/' + name).status_code == 200
    assert client.get('/angelos/assets/missing').status_code == 404
    assert client.get('/angelos/api/bootstrap', headers={'host': 'attacker.invalid'}).status_code == 403
    assert client.get('/angelos/api/bootstrap', headers={'sec-fetch-site': 'cross-site'}).status_code == 403
    assert client.post('/angelos/api/conversations', json={}, headers={'origin': 'https://attacker.invalid'}).status_code == 403
    assert client.post('/angelos/api/conversations', json={}, headers={'x-angelos-token': ''}).status_code == 403
    assert client.post('/angelos/api/conversations', content='{}', headers={'content-type': 'text/plain'}).status_code == 403


def test_workspace_draft_rename_restart(env, tmp_path):
    client, bench, _ = env
    workspace = post(client, '/workspaces', {'name': '项目', 'path': str(tmp_path), 'description': '保留来源'})['workspace']
    cid = post(client, '/conversations', {'workspaceId': workspace['id']})['conversation']['id']
    post(client, '/conversations/' + cid, {'title': '<标题>', 'draft': '未发送的文字'})
    store = WorkbenchStore(bench.store.path)
    assert store.read('conversation', cid)['draft'] == '未发送的文字'
    assert store.read('workspace', workspace['id'])['path'] == str(tmp_path)
    assert client.post('/angelos/api/workspaces', json={'name': '项目'}).status_code == 409
    assert client.post('/angelos/api/workspaces', json={'name': '坏目录', 'path': '/missing-directory-fixture'}).status_code == 409
    assert not bench.store.list('conversation')[0]['turns']


def test_chat_idempotency_and_attachment_scope(env):
    client, bench, host = env
    cid, other = new(client), new(client)
    attachment = post(client, '/conversations/' + cid + '/attachments', {'name': '资料.txt', 'data': base64.b64encode('实际内容'.encode()).decode()})['attachment']
    payload = {'text': '看看资料', 'request_key': 'same', 'attachments': [attachment['id']]}
    post(client, '/conversations/' + cid + '/messages', payload)
    c = wait(client, cid)
    assert '实际内容' in c['turns'][0]['reply']
    post(client, '/conversations/' + cid + '/messages', payload)
    assert host.preparations == 1
    conflict = client.post('/angelos/api/conversations/' + cid + '/messages', json={**payload, 'text': '改掉原话'})
    assert conflict.json()['reason'] == 'idempotency_key_conflict'
    cross = client.post('/angelos/api/conversations/' + other + '/messages', json=payload)
    assert cross.json()['reason'] == 'attachment_scope_mismatch'
    assert not bench.store.read('conversation', other)['turns']


def test_real_agent_file_receipt_and_archived_result(env):
    client, bench, host = env
    cid = new(client)
    post(client, '/conversations/' + cid + '/messages', {'text': '生成合成文件并核验', 'request_key': 'write'})
    c = wait(client, cid)
    t = c['turns'][0]
    assert t['stage'] == 'finished', t
    assert t['view']['verification']['verified'] is True
    assert t['view']['status'] == 'completed'
    assert host.tool.calls == 1
    url = f'/angelos/api/conversations/{cid}/artifacts/write/0'
    assert client.get(url).content == b'standalone agent verified'
    # 归档在工作台数据库中，刷新不会重新执行；其他会话不能凭路径下载。
    assert WorkbenchStore(bench.store.path).read('conversation', cid)['turns'][0]['result']['artifacts']
    assert client.get(f'/angelos/api/conversations/{new(client)}/artifacts/write/0').status_code == 409
    host.tool.path.write_text('被替换的内容')
    assert client.get(url).json()['reason'] == 'artifact_changed_or_oversized'
    assert host.tool.calls == 1


def test_cancel_preparation_discards_late_reply(env):
    client, bench, host = env
    entered, release = threading.Event(), threading.Event()
    def prepare(*args):
        entered.set()
        release.wait(3)
        return {'reply': '迟到的回复'}
    host.prepare = prepare
    cid = new(client)
    post(client, '/conversations/' + cid + '/messages', {'text': '等待取消', 'request_key': 'cancel-me'})
    assert entered.wait(2)
    post(client, f'/conversations/{cid}/turns/cancel-me/cancel', {})
    release.set()
    c = wait(client, cid)
    assert c['turns'][0]['stage'] == 'canceled'
    assert not c['turns'][0].get('reply')
    assert host.tool.calls == 0


def test_unavailable_is_not_demo_and_recovery_never_autoreplays(tmp_path):
    store = WorkbenchStore(tmp_path / 'workbench.sqlite3')
    c = store.conversation('default')
    store.change('conversation', c['id'], lambda c: c['turns'].append({'id': 'pending', 'text': '原话', 'stage': 'preparing'}))
    bench = Workbench(store)
    try:
        assert store.read('conversation', c['id'])['turns'][0]['stage'] == 'interrupted'
        with pytest.raises(ValueError, match='agent_unavailable'):
            bench.send(c['id'], {'text': '新请求', 'request_key': 'new'})
        assert len(store.read('conversation', c['id'])['turns']) == 1
    finally:
        bench.close()


def test_invalid_upload_and_payload_limits(env):
    client, _, _ = env
    cid = new(client)
    for raw in (b'\x89PNG\0\xff', b'\0abc', b'a' * 12001):
        result = client.post('/angelos/api/conversations/' + cid + '/attachments', json={'name': 'data.txt', 'data': base64.b64encode(raw).decode()})
        assert result.status_code == 409
    assert client.post('/angelos/api/conversations/' + cid, json={'requester': 'forged'}).status_code == 409
    assert client.post('/angelos/api/conversations/' + cid + '/messages', json={'text': 'x' * 96001, 'request_key': 'huge'}).status_code == 409


def test_standalone_routes_do_not_mount_qq_or_hub(tmp_path):
    from shinku.angelos.app import create_angelos_app
    with TestClient(create_angelos_app(tmp_path)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/debug/hub').status_code == 404
        assert client.post('/events/napcat', json={}).status_code == 404
        assert client.get('/angelos/api/bootstrap').json()['agent']['available'] is False


def test_service_host_binds_independent_conversations(env):
    from shinku.angelos.host import ServiceHost
    _, _, fixture_host = env
    host = ServiceHost(fixture_host.service)
    assert host.ready()
    assert host.task_id('one', 'request') != host.task_id('two', 'request')
    assert host.effect_catalog()['demo_file'] == ['write']


def test_control_version_prevents_replaying_completed_work(env):
    client, _, host = env
    cid = new(client)
    post(client, '/conversations/' + cid + '/messages', {'text': '生成文件', 'request_key': 'complete'})
    c = wait(client, cid)
    result = client.post(f'/angelos/api/conversations/{cid}/turns/complete/resume', json={
        'attempt': 1, 'version': 0, 'request_key': 'bad-version'})
    assert result.json()['reason'] == 'task_version_conflict'
    assert host.tool.calls == 1


@pytest.mark.parametrize('invalid', ['missing-target', 'write-as-observation', 'bad-shape'])
def test_service_host_rejects_incomplete_contract_before_execution(env, monkeypatch, invalid):
    from test_angelos_preparation_failure import LegacyPreparationHost as ServiceHost
    from shinku.agent_runtime.demo import demo_request
    client, bench, fixture_host = env
    host = ServiceHost(fixture_host.service)
    contract = copy.deepcopy(demo_request(fixture_host.tool).contract)
    if invalid == 'missing-target':
        contract['requirements'][0]['targets'] = [str(fixture_host.tool.path.parent / 'wrong.txt')]
    elif invalid == 'write-as-observation':
        contract['requirements'][0]['effect'] = 'tool:demo_file'
    else:
        contract['requirements'] = []
    monkeypatch.setattr(host.profile.runtime, 'call_chat_json', lambda **kwargs: {'task_contract': contract})
    bench.host = host
    cid = new(client)
    post(client, f'/conversations/{cid}/messages', {
        'text': '创建文件 ' + str(fixture_host.tool.path), 'request_key': 'invalid'})
    turn = wait(client, cid)['turns'][0]
    assert turn['stage'] == 'failed'
    assert turn['error'] == 'contract_preparation_rejected'
    assert not turn.get('task_id')
    assert fixture_host.tool.calls == 0


def test_service_host_same_loop_real_execution_and_plain_reply(env, monkeypatch):
    from shinku.angelos.host import ServiceHost
    from shinku.agent_runtime.demo import demo_request
    client, bench, fixture_host = env
    host = ServiceHost(fixture_host.service)
    runtime = host.profile.runtime
    execute = runtime.call_chat_json
    calls = []
    def prepare_or_execute(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {'tool_call': {'type': 'declare_task_contract',
                                 'task_contract': demo_request(fixture_host.tool).contract}}
        return execute(**kwargs)
    monkeypatch.setattr(runtime, 'call_chat_json', prepare_or_execute)
    bench.host = host
    cid = new(client)
    post(client, f'/conversations/{cid}/messages', {
        'text': '创建文件 ' + str(fixture_host.tool.path), 'request_key': 'valid'})
    turn = wait(client, cid)['turns'][0]
    assert turn['stage'] == 'finished', turn
    assert turn['view']['verification']['verified'] is False
    assert turn['result']['verification']['layers']['policy'] == 'agent_reply_delivery'
    assert any(r['receipt_valid'] is True for r in turn['result']['verification']['evidence'])
    assert fixture_host.tool.calls == 1
    assert len(calls) == turn['result']['budget']['requests'] == 3
    assert turn['preparation_usage']['requests'] == 0
    monkeypatch.setattr(runtime, 'call_chat_json', lambda **kwargs:
        {'kind': 'final', 'text': '我们开始工作吧。', 'task_outcome': 'completed'})
    other = new(client)
    post(client, f'/conversations/{other}/messages', {'text': '你好', 'request_key': 'chat'})
    reply = wait(client, other)['turns'][0]
    assert reply['stage'] == 'finished'
    assert reply['result']['execution_report']['text'] == '我们开始工作吧。'
    assert reply['view']['verification']['verified'] is False
    assert reply['result']['budget']['requests'] == 1
    assert fixture_host.tool.calls == 1
