"""迁移日常会话不重写历史授权；使用真实本机数据库和隔离 AgentService。"""
from copy import deepcopy

import pytest

from test_angelos import env, new, post, wait


def test_move_changes_future_constraints_and_preserves_old_result(env, tmp_path):
    client, bench, host = env
    old = post(client, '/workspaces', {'name': '合成测试', 'path': str(tmp_path), 'description': '仅操作合成文件'})['workspace']
    cid = post(client, '/conversations', {'workspaceId': old['id']})['conversation']['id']
    post(client, f'/conversations/{cid}/messages', {'text': '生成合成文件', 'request_key': 'before'})
    before = deepcopy(wait(client, cid)['turns'][0])
    moved = post(client, f'/conversations/{cid}', {'workspaceId': 'default'})['conversation']
    assert moved['turns'][0] == before and moved['workspaceId'] == 'default'
    assert bench.store.read('workspace', old['id']) == old
    post(client, f'/conversations/{cid}/messages', {'text': '生成合成文件', 'request_key': 'after'})
    after = wait(client, cid)['turns'][-1]
    assert after['result']['constraints'] == []
    assert before['result']['constraints'] == ['工作区目录：' + str(tmp_path), '仅操作合成文件']
    assert after['workspace_snapshot']['id'] == 'default'


@pytest.mark.parametrize('action', ['resume', 'supplement'])
def test_old_task_cannot_resume_under_moved_workspace(env, tmp_path, action):
    client, bench, host = env
    old = post(client, '/workspaces', {'name': '旧作用域', 'path': str(tmp_path), 'description': '旧范围'})['workspace']
    cid = post(client, '/conversations', {'workspaceId': old['id']})['conversation']['id']
    post(client, f'/conversations/{cid}/messages', {'text': '生成合成文件', 'request_key': 'before'})
    before = deepcopy(wait(client, cid)['turns'][0])
    post(client, f'/conversations/{cid}', {'workspaceId': 'default'})
    calls = host.tool.calls
    r = client.post(f'/angelos/api/conversations/{cid}/turns/before/{action}', json={
        'attempt': before['view']['attempt'], 'version': before['view']['version'],
        'request_key': 'continue', 'text': '继续'})
    assert r.status_code == 409 and r.json()['reason'] == 'task_workspace_changed'
    assert host.tool.calls == calls and bench.store.read('conversation', cid)['turns'][0] == before


@pytest.mark.parametrize('stage', ['preparing', 'submitting', 'executing'])
def test_migration_cannot_swap_active_input_scope(env, tmp_path, stage):
    client, bench, _ = env
    cid = new(client)
    target = post(client, '/workspaces', {'name': '目标', 'path': str(tmp_path)})['workspace']
    bench.store.change('conversation', cid, lambda c: c['turns'].append({'id': 'running', 'stage': stage}))
    r = client.post(f'/angelos/api/conversations/{cid}', json={'workspaceId': target['id']})
    assert r.json()['reason'] == 'conversation_busy'
    assert bench.store.read('conversation', cid)['workspaceId'] == 'default'


def test_invalid_destination_and_metadata_cannot_change_scope(env):
    client, bench, _ = env
    cid = new(client)
    for payload in ({'workspaceId': 'missing'}, {'workspaceId': None},
                    {'workspaceId': 'default', 'requester': 'forged'}):
        assert client.post(f'/angelos/api/conversations/{cid}', json=payload).status_code == 409
        assert bench.store.read('conversation', cid)['workspaceId'] == 'default'
