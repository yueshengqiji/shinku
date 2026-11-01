"""工作卡只解释当前尝试；固定旧成功、失败副作用及正文缺失的展示边界。"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.progress_view import project


def sample():
    task = {'task_id': 't', 'requester': 'owner', 'session': 'angelos:c',
            'execution_attempt': 2, 'revision': 8, 'updated': 100, 'requests': [],
            'contract': {'requirements': [{'id': 'write', 'description': '保存文件'},
                                         {'id': 'check', 'description': '验证内容'}]},
            'tool_calls': []}
    view = {'available': True, 'status': 'partial', 'version': 8, 'worker_active': False,
            'verification': {'current_attempt': True, 'verified': False, 'accepted': True,
                             'satisfied': ['write'], 'missing': ['check']}}
    return task, view


def call(task, *, attempt=2, status='ok', executed=True, valid=True):
    return {'id': 'file', 'name': 'private_file_write', 'status': status, 'executed': executed,
            'execution_binding': {'task_id': task['task_id'], 'session': task['session'], 'attempt': attempt},
            'data': {'state_updates': {'artifact_receipts': [
                {'path': '/synthetic/project/example.py', 'result_valid': valid}]}}}


def overview(task, view):
    return project(task, view, now=101)['overview']


def test_partial_keeps_successful_file_and_missing_goal_without_claiming_complete():
    task, view = sample(); task['tool_calls'] = [call(task)]
    before = copy.deepcopy((task, view))
    data = overview(task, view)
    assert data['verification'] == 'partial'
    assert [r['state'] for r in data['checks']] == ['satisfied', 'missing']
    assert data['files'] == [{'name': 'example.py', 'state': 'receipt_valid'}]
    assert data['report'] == 'not_recorded'
    assert '/synthetic/' not in json.dumps(data)
    assert (task, view) == before


@pytest.mark.parametrize('status', ['failed', 'canceled', 'blocked', 'unverified', 'unknown', 'conflict'])
def test_successful_call_is_not_task_success(status):
    task, view = sample(); view['status'] = status; task['tool_calls'] = [call(task)]
    assert overview(task, view)['verification'] == 'not_verified'


def test_restored_attempt_excludes_old_and_cross_session_files_and_verification():
    task, view = sample(); view['status'] = 'running'
    view['verification'].update(current_attempt=False, verified=True, accepted=True)
    foreign = call(task); foreign['execution_binding']['session'] = 'angelos:other'
    task['tool_calls'] = [call(task, attempt=1), foreign]
    data = overview(task, view)
    assert data['verification'] == 'pending' and data['files'] == []
    assert all(r['state'] == 'pending' for r in data['checks'])


@pytest.mark.parametrize('status,executed', [('error', True), ('blocked', False), ('ok', False)])
def test_rejected_or_failed_tool_does_not_create_file_result(status, executed):
    task, view = sample(); task['tool_calls'] = [call(task, status=status, executed=executed)]
    assert overview(task, view)['files'] == []


@pytest.mark.parametrize('field,value', [('attempt', 1), ('task_id', 'foreign'),
                                      ('scope', {'requester': 'other', 'session': 'angelos:c'})])
def test_old_or_foreign_body_is_not_current_report(field, value):
    task, view = sample()
    task['execution_result'] = {'task_id': 't', 'attempt': 2,
        'scope': {'requester': 'owner', 'session': 'angelos:c'}, 'execution_report': {'text': '已保存的原文'}}
    assert overview(task, view)['report'] == 'available'
    task['execution_result'][field] = value
    assert overview(task, view)['report'] == 'not_recorded'


def test_analysis_and_conversation_do_not_claim_action_verification():
    task, view = sample(); view['status'] = 'completed'
    view['verification'].update(verified=True, layers={'policy': 'ordinary_readonly_analysis',
                                                      'answer_evidence': 'unconfirmed'})
    assert overview(task, view)['verification'] == 'analysis_delivered'
    view['verification']['layers']['policy'] = 'conversation_reply'
    assert overview(task, view) is None


def test_large_task_has_explicit_display_omissions():
    task, view = sample()
    task['contract']['requirements'] *= 20
    task['tool_calls'] = [call(task) for _ in range(15)]
    data = overview(task, view)
    assert len(data['checks']) == 24 and data['omittedChecks'] == 16
    assert len(data['files']) == 12 and data['omittedFiles'] == 3


def test_http_work_card_uses_real_file_receipt_and_read_only_refresh(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    host = ServiceHost(service)
    original = runtime.call_chat_json
    def reply(**kwargs):
        if not runtime.calls:
            runtime.calls.append(kwargs)
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        return original(**kwargs)
    runtime.call_chat_json = reply
    app = create_angelos_app(tmp_path/'ui', host=host)
    try:
        with TestClient(app) as client:
            headers = {'x-angelos-token': client.get('/angelos/api/bootstrap').json()['token']}
            cid = client.post('/angelos/api/conversations', headers=headers, json={}).json()['conversation']['id']
            url = '/angelos/api/conversations/'+cid
            assert client.post(url+'/messages', headers=headers,
                               json={'text': '生成并验证合成文件', 'request_key': 'test'}).status_code == 200
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                turn = client.get(url).json()['conversation']['turns'][0]
                if turn['stage'] == 'finished':
                    break
                time.sleep(.01)
            data = turn['live_progress']['overview']
            assert data['verification'] == 'reply_delivered', turn
            assert data['receiptChecks'][-1]['receipt_valid'] is True
            assert turn['view']['verification']['verified'] is False
            assert data['report'] == 'available' and data['files'][0]['name'] == tool.path.name
            assert tool.path.read_text() == 'standalone agent verified'
            model_count = len(runtime.calls)
            for _ in range(3):
                assert client.get(url).json()['conversation']['turns'][0]['live_progress']['overview'] == data
            assert len(runtime.calls) == model_count and tool.calls == 1
            assert 'live_progress' not in app.state.angelos.store.read('conversation', cid)['turns'][0]
            assert host.read_progress('other', turn)['available'] is False
            assert client.get(url, headers={'origin': 'https://example.invalid'}).status_code == 403
    finally:
        service.coordinator.shutdown(wait=True)


def test_frontend_work_card_escapes_content_and_retains_boundaries(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for frontend rendering checks')
    source = Path(__file__).resolve().parents[1]/'frontend/angelos/app.js'
    script = tmp_path/'render.cjs'
    script.write_text(r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const raw=fs.readFileSync(process.argv[2],'utf8');
const code=raw.slice(raw.indexOf('function taskOverviewHTML('),raw.indexOf('function selectedReasoning('));
const ctx={online:true,activityDisplay:undefined,states:{failed:'失败',completed:'完成'},messageError:x=>x,
 esc:x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))};
vm.createContext(ctx);vm.runInContext(code,ctx);
const turn={id:'test',live_progress:{overview:{status:'failed',verification:'not_verified',attempt:2,
 receiptCount:1,report:'not_recorded',checks:[{state:'missing',description:'<img onerror=alert(1)>'}],
 files:[{name:'<script>x</script>',state:'receipt_valid'}]}}};
let html=ctx.taskOverviewHTML(turn);
assert(html.includes('尚未通过整项验收')&&html.includes('尚无当前尝试的归档正文'));
assert(!html.includes('<img')&&!html.includes('<script>'));
assert(html.includes('不是当前磁盘差异')&&!html.includes('data-control='));
turn.live_progress.overview.cancelRequested=true;turn.live_progress.overview.workerActive=true;
assert(ctx.taskOverviewHTML(turn).includes('仍在等待操作退出'));
ctx.online=false;assert(ctx.taskOverviewHTML(turn).includes('上次读取的执行状态'));
assert.equal(ctx.taskOverviewHTML({}), '');
Object.assign(turn.live_progress.overview,{status:'completed',verification:'reply_delivered',
 receiptChecks:[{tool:'<script>read</script>',execution_status:'ok',receipt_valid:false}],checks:[]});
html=ctx.taskOverviewHTML(turn);
assert(html.includes('答复已交付')&&html.includes('回执核验未通过'));
assert(!html.includes('执行验收通过')&&!html.includes('<script>'));

''', encoding='utf-8')
    result = subprocess.run([node, str(script), str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
