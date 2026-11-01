"""删除工作区登记的真实API/浏览器回归；仅用隔离库和合成项目。"""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from shinku.angelos.store import WorkbenchStore
from test_angelos import env, post, wait
from test_angelos_coding_view import coding


def remove(client, wid, **kwargs):
    return client.post('/angelos/api/workspaces/'+wid+'/delete', json={'confirm': True}, **kwargs)


def test_delete_preserves_conversations_drafts_attachments_receipts_files_and_scope(env, tmp_path):
    client, bench, host = env
    marker = tmp_path/'keep.txt'; marker.write_text('磁盘原文')
    w = post(client, '/workspaces', {'name': '历史项目', 'path': str(tmp_path), 'description': '原约束'})['workspace']
    cid = post(client, '/conversations', {'workspaceId': w['id']})['conversation']['id']
    post(client, f'/conversations/{cid}/messages', {'text': '生成合成文件', 'request_key': 'before'})
    turn = deepcopy(wait(client, cid)['turns'][0])
    post(client, f'/conversations/{cid}', {'draft': '未发送草稿', 'title': '原会话标题'})
    post(client, f'/conversations/{cid}/attachments', {'name': '资料.txt', 'data': 'cHJlc2VydmU='})
    attachments = bench.store.list('attachment'); calls = host.tool.calls; preparations = host.preparations
    result = remove(client, w['id'])
    assert result.status_code == 200 and result.json()['removed']['movedConversations'] == 1
    c = WorkbenchStore(bench.store.path).read('conversation', cid)
    assert c['workspaceId'] == 'default' and c['draft'] == '未发送草稿' and c['title'] == '原会话标题'
    assert c['turns'] == [turn] and bench.store.list('attachment') == attachments
    assert marker.read_text() == '磁盘原文' and host.tool.path.read_text() == 'standalone agent verified'
    assert all(x['id'] != w['id'] for x in WorkbenchStore(bench.store.path).list('workspace'))
    # 迁移列表归属不授权旧任务换目录恢复，既有约束检查仍应拒绝。
    resumed = client.post(f'/angelos/api/conversations/{cid}/turns/before/resume', json={
        'attempt': turn['view']['attempt'], 'version': turn['view']['version'], 'request_key': 'resume'})
    assert resumed.status_code == 409 and resumed.json()['reason'] == 'task_workspace_changed'
    assert host.tool.calls == calls and host.preparations == preparations


@pytest.mark.parametrize('stage', ['preparing', 'submitting', 'executing'])
def test_active_workspace_delete_is_atomic_and_does_not_cancel(env, stage):
    client, bench, host = env
    w = post(client, '/workspaces', {'name': '运行项目'})['workspace']
    cid = post(client, '/conversations', {'workspaceId': w['id']})['conversation']['id']
    bench.store.change('conversation', cid, lambda c: c['turns'].append({'id': 'active', 'stage': stage}))
    # 同一工作区的空闲会话也不能先迁走，必须整体拒绝。
    post(client, '/conversations', {'workspaceId': w['id']})
    before = bench.store.list('conversation')
    assert remove(client, w['id']).json()['reason'] == 'workspace_busy'
    assert bench.store.list('conversation') == before and bench.store.read('workspace', w['id']) == w
    assert host.preparations == host.tool.calls == 0


def test_delete_empty_workspace_keeps_other_running_workspace_and_default(env):
    client, bench, host = env
    cid = post(client, '/conversations', {})['conversation']['id']
    bench.store.change('conversation', cid, lambda c: c['turns'].append({'id': 'other', 'stage': 'executing'}))
    before = bench.store.read('conversation', cid)
    w = post(client, '/workspaces', {'name': '空项目'})['workspace']
    assert remove(client, w['id']).json()['removed']['movedConversations'] == 0
    assert bench.store.read('conversation', cid) == before and host.tool.calls == 0
    assert remove(client, 'default').json()['reason'] == 'workspace_default_required'
    assert remove(client, 'missing').json()['reason'] == 'record_not_found'
    assert bench.store.read('workspace', 'default')


@pytest.mark.parametrize('payload', [{}, {'confirm': False}, {'confirm': 1}, {'confirm': True, 'delete_files': True}])
def test_delete_requires_explicit_exact_confirmation(env, payload):
    client, bench, _ = env
    w = post(client, '/workspaces', {'name': '保留'})['workspace']
    response = client.post('/angelos/api/workspaces/'+w['id']+'/delete', json=payload)
    assert response.status_code == 409 and response.json()['reason'] == 'workspace_delete_confirmation_required'
    assert bench.store.read('workspace', w['id']) == w


def test_delete_reuses_write_origin_and_token_protection(env):
    client, bench, _ = env
    w = post(client, '/workspaces', {'name': '保留'})['workspace']
    for headers in ({'x-angelos-token': ''}, {'origin': 'https://foreign.invalid'}, {'sec-fetch-site': 'cross-site'}):
        assert remove(client, w['id'], headers=headers).status_code == 403
        assert bench.store.read('workspace', w['id']) == w


def test_browser_delete_current_workspace_cancel_and_preserve_draft(coding):
    c = coding; page = c.page
    page.locator('#view-chat').click()
    page.locator('#message-input').fill('保留当前未发送草稿')
    page.locator('[data-workspace]').first.click()
    delete = page.locator('[data-delete-workspace="'+c.workspace['id']+'"]')
    assert page.locator('[data-delete-workspace="default"]').count() == 0
    delete.click(); page.locator('#workspace-delete-dialog').wait_for(state='visible')
    assert '1 个会话' in page.locator('#workspace-delete-note').inner_text()
    assert '不删除磁盘' in page.locator('#workspace-delete-dialog').inner_text()
    evidence = os.environ.get('ANGELOS_UI_EVIDENCE_DIR')
    if evidence:
        dest = Path(evidence); dest.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(dest/'workspace-delete-wide.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(dest/'workspace-delete-narrow.png'))
    page.locator('#cancel-workspace-delete').click()
    assert c.bench.store.read('workspace', c.workspace['id']) == c.workspace
    page.set_viewport_size({'width': 1440, 'height': 1000})
    delete.click(); page.locator('#confirm-workspace-delete').click()
    delete.wait_for(state='detached')
    page.locator('#workspace-dialog [data-close]').last.click()
    assert page.locator('#message-input').input_value() == '保留当前未发送草稿'
    assert c.bench.store.read('conversation', c.conversation['id'])['workspaceId'] == 'default'
    assert (c.project/'sample.py').read_text() == c.code and c.tool.calls == 0 and not c.calls
    page.reload()
    page.locator('button[data-conversation="'+c.conversation['id']+'"][aria-current="true"]').wait_for()
    assert page.locator('#message-input').input_value() == '保留当前未发送草稿'


def test_browser_delete_refuses_running_task_without_interrupting(coding):
    c = coding; page = c.page
    page.locator('#view-chat').click(); page.locator('#message-input').fill('HOLD_TASK')
    page.locator('#send-message').click(); assert c.entered.wait(8)
    page.locator('[data-workspace]').first.click()
    page.locator('[data-delete-workspace="'+c.workspace['id']+'"]').click()
    page.locator('#confirm-workspace-delete').click()
    page.locator('#workspace-delete-error').filter(has_text='任务正在准备或执行').wait_for()
    assert c.bench.store.read('workspace', c.workspace['id']) == c.workspace
    assert len(c.calls) == 1 and c.tool.calls == 0
    c.release.set()
