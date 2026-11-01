"""只读SSE、工具账本隔离与真实浏览器选区回归；全部使用合成数据和离线SDK。"""
import asyncio
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import uvicorn

from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.live_activity import events, read_tools, turn_snapshot
from test_angelos_json_stream import stream_sdk, delta, USAGE


def fake_host(task, *, allow=True):
    def get(scope, task_id):
        if (scope.requester, scope.session, task_id) != (task['requester'], task['session'], task['task_id']):
            raise ValueError('scope rejected')
        return copy.deepcopy(task)
    def grant(*args):
        if not allow: raise PermissionError('denied')
    return SimpleNamespace(requester='owner', service=SimpleNamespace(_lock=threading.RLock(), _get=get, _grant=grant))


def task_fixture():
    binding = {'task_id': 't', 'requester': 'owner', 'session': 'angelos:c', 'attempt': 2}
    call = {'id': 'one', 'name': 'read', 'status': 'ok', 'arguments': {'path': 'safe.py', 'api_key': 'SECRET'},
            'execution_binding': binding, 'data': {'state_updates': {'read_complete': True}}}
    task = {'task_id': 't', 'requester': 'owner', 'session': 'angelos:c', 'execution_attempt': 2,
            'tool_calls': [call], 'inflight': [{'id': 'two', 'name': 'write', 'arguments': {'path': 'next.py'}}]}
    return task


def test_projection_uses_current_bound_receipts_without_changing_task():
    task = task_fixture(); old = copy.deepcopy(task['tool_calls'][0]); old['execution_binding']['attempt'] = 1
    foreign = copy.deepcopy(old); foreign['execution_binding']['session'] = 'angelos:elsewhere'
    task['tool_calls'] += [old, foreign]; before = copy.deepcopy(task)
    data = read_tools(fake_host(task), 'c', {'task_id': 't'})
    assert data['available'] and data['total'] == 1 and len(data['rows']) == 2
    assert data['rows'][0]['status'] == 'ok' and data['rows'][1]['status'] == 'pending_receipt'
    assert 'SECRET' not in json.dumps(data) and data['rows'][0]['arguments']['api_key'] == '[redacted]'
    assert task == before and data['is_fact_evidence'] is False
    assert not read_tools(fake_host(task), 'foreign', {'task_id': 't'})['available']
    assert not read_tools(fake_host(task, allow=False), 'c', {'task_id': 't'})['available']


def test_large_parameters_have_explicit_display_gaps_and_keep_original():
    task = task_fixture(); task['tool_calls'][0]['arguments']['body'] = 'x'*10000
    before = copy.deepcopy(task)
    data = read_tools(fake_host(task), 'c', {'task_id': 't'})
    assert data['information']['status'] == 'truncated'
    assert data['rows'][0]['arguments']['body']['omitted'] == 'text_limit' and task == before


def test_configured_secret_scrubber_applies_to_tool_display_without_rewriting_receipts():
    task = task_fixture(); task['tool_calls'][0]['arguments']['note'] = 'BARE_SECRET'
    host = fake_host(task)
    host.profile = SimpleNamespace(runtime=SimpleNamespace(_redact_secrets=lambda text, max_chars: text.replace('BARE_SECRET', '[redacted]')))
    assert 'BARE_SECRET' not in json.dumps(read_tools(host, 'c', {'task_id': 't'}))
    assert task['tool_calls'][0]['arguments']['note'] == 'BARE_SECRET'


def test_stream_disconnect_and_planned_rotation_have_no_task_actions():
    async def collect(disconnected, duration):
        request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=disconnected))
        return [row async for row in events(None, 'c', request, {'active': True}, duration=duration)]
    assert asyncio.run(collect(True, 2)) == []
    assert asyncio.run(collect(False, 0)) == ['event: rotate\ndata: {}\n\n']


def test_events_send_only_changed_snapshots_and_disconnect_is_readonly(monkeypatch):
    import shinku.angelos.live_activity as activity
    snapshots = [{'conversationId': 'c', 'turnId': 'turn', 'active': True, 'reasoning': []},
                 {'conversationId': 'c', 'turnId': 'turn', 'active': False, 'reasoning': [{'text': '合成增量'}]}]
    monkeypatch.setattr(activity, 'turn_snapshot', lambda *a: snapshots.pop(0))
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
    async def collect():
        initial = {'conversationId': 'c', 'turnId': 'turn', 'active': True, 'reasoning': []}
        return [part async for part in events(None, 'c', request, initial, duration=2, interval=.001)]
    rows = asyncio.run(collect())
    assert len(rows) == 2 and '合成增量' in rows[-1]


def test_progress_changes_stream_without_reasoning_or_tool_changes(monkeypatch):
    import shinku.angelos.live_activity as activity
    initial = {'active': True, 'reasoning': [], 'tools': {}, 'progress': {'phase': 'model', 'waitSeconds': 0}}
    waiting = {**initial, 'progress': {'phase': 'model', 'waitSeconds': 20}}
    stopped = {**waiting, 'active': False, 'progress': {'phase': 'failed'}}
    snapshots = [initial, waiting, waiting, stopped]
    monkeypatch.setattr(activity, 'turn_snapshot', lambda *a: snapshots.pop(0))
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
    async def collect():
        return [row async for row in events(None, 'c', request, initial, interval=.001)]
    rows = asyncio.run(collect())
    assert len(rows) == 3 and '20' in rows[1] and 'failed' in rows[-1]


def test_http_stream_and_tools_preserve_task_scope_and_do_not_write_or_call_model(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    host = ServiceHost(service); app = create_angelos_app(tmp_path/'ui', host=host)
    bench = app.state.angelos
    try:
        with TestClient(app) as client:
            boot = client.get('/angelos/api/bootstrap').json(); client.headers['x-angelos-token'] = boot['token']
            cid = client.post('/angelos/api/conversations', json={}).json()['conversation']['id']
            c = bench.store.read('conversation', cid)
            c['turns'] = [{'id': 'turn', 'stage': 'replied', 'text': '合成消息', 'reply': '合成回答', 'attachments': []}]
            with bench.store.transaction() as db: bench.store.put(db, 'conversation', c)
            host.reasoning_view.append('prepare:'+cid+':turn', '合成推理', model='fixture')
            before = bench.store.read('conversation', cid); requests = len(runtime.calls)
            endpoint = '/angelos/api/conversations/'+cid
            result = client.get(endpoint+'/activity')
            assert result.status_code == 200 and result.headers['content-type'].startswith('text/event-stream')
            assert result.headers['x-accel-buffering'] == 'no' and '合成推理' in result.text
            assert client.get(endpoint+'/tools/turn').json()['activity']['tools']['rows'] == []
            assert client.get(endpoint+'/activity', headers={'origin': 'https://elsewhere.invalid'}).status_code == 403
            assert client.get(endpoint+'/tools/turn', headers={'sec-fetch-site': 'cross-site'}).status_code == 403
            assert client.get(endpoint+'/tools/missing').status_code == 409
            assert client.get('/angelos/assets/activity.js').status_code == 200
            assert bench.store.read('conversation', cid) == before and len(runtime.calls) == requests and tool.calls == 0
            # 存储中被错误指向其他会话的任务也不能读出其推理；工具inspect先拒绝。
            foreign = client.post('/angelos/api/conversations', json={}).json()['conversation']['id']
            handle = host.port(foreign).submit(demo_request(tool))
            host.reasoning_view.append(handle.task_id, 'FOREIGN_PRIVATE', model='fixture')
            c['turns'][0]['task_id'] = handle.task_id
            with bench.store.transaction() as db: bench.store.put(db, 'conversation', c)
            snap = client.get(endpoint+'/tools/turn').json()['activity']
            assert not snap['tools']['available'] and snap['reasoning'] == []
            assert snap['progress'] is None
            assert 'FOREIGN_PRIVATE' not in json.dumps(snap)
    finally:
        service.coordinator.shutdown(wait=True)


def test_browser_stream_selection_tools_and_mode_switch_do_not_interrupt_task(tmp_path, monkeypatch):
    pw = pytest.importorskip('playwright.sync_api')
    (tmp_path/'project').mkdir()
    service, tool, _ = demo_service(tmp_path/'project')
    first, second, release_first, release_second = [threading.Event() for _ in range(4)]
    step = [0]; contract = demo_request(tool).contract
    long_reason = '\n'.join('合成推理第'+str(i)+'行：保留真实回执。' for i in range(120))+'\n增量末尾。'
    def script():
        step[0] += 1
        if step[0] == 1:
            yield delta({'reasoning_content': '合成推理第一段。\n'})
            first.set(); assert release_first.wait(20), 'first stream gate timed out'
            yield delta({'reasoning_content': long_reason})
            second.set(); assert release_second.wait(20), 'second stream gate timed out'
            answer = {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}}
        elif step[0] == 2:
            answer = {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        else:
            answer = {'kind': 'final', 'text': '已核验合成文件。', 'task_outcome': 'completed'}
        yield delta({'content': json.dumps(answer, ensure_ascii=False)}, finish='stop'); yield USAGE
    server = None; thread = None; sock = None
    try:
        with stream_sdk(script, monkeypatch) as (base, calls, _):
            service.executor.profile = replace(service.executor.profile, runtime=base, runtime_meters_budget=True)
            host = ServiceHost(service); app = create_angelos_app(tmp_path/'ui', host=host); bench = app.state.angelos
            sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
            thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True); thread.start()
            deadline = time.monotonic()+8
            while not server.started and time.monotonic() < deadline: time.sleep(.01)
            assert server.started
            with pw.sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000}); page.set_default_timeout(8000)
                errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
                page.add_init_script('''window.activityFrames=[];window.activityStreams=[];
                  const ES=window.EventSource;window.EventSource=class extends ES{constructor(url){super(url);
                    activityStreams.push(this);this.addEventListener('message',e=>activityFrames.push(JSON.parse(e.data)))}};''')
                page.goto(f'http://127.0.0.1:{port}/angelos')
                page.locator('button.conversation-item[aria-current="true"]').wait_for()
                assert not errors, errors
                page.locator('#message-input').fill('创建并核验合成文件'); page.locator('#send-message').click()
                entered = first.wait(8)
                if not entered:
                    snapshots = bench.store.list('conversation')
                    dest = Path(os.environ.get('ANGELOS_UI_EVIDENCE_DIR', str(tmp_path/'screenshots'))); dest.mkdir(parents=True, exist_ok=True)
                    (dest/'first-stream-failure.json').write_text(json.dumps(snapshots, ensure_ascii=False, indent=2))
                assert entered, {'calls': len(calls), 'errors': errors,
                    'turns': [(t.get('stage'), t.get('error'), t.get('view', {}).get('reason')) for c in bench.store.list('conversation') for t in c['turns']]}
                page.locator('.reasoning-content pre').filter(has_text='合成推理第一段。').wait_for()
                assert len(calls) == 1 and tool.calls == 0
                before = page.locator('.reasoning-content').inner_text()
                page.evaluate('''()=>{const n=document.querySelector('.reasoning-content pre').firstChild;
                  const r=document.createRange();r.setStart(n,0);r.setEnd(n,6);getSelection().removeAllRanges();getSelection().addRange(r)}''')
                selected = page.evaluate('getSelection().toString()')
                page.evaluate("activityStreams.at(-1).dispatchEvent(new Event('rotate'))")
                page.wait_for_function('()=>activityStreams.length>=2')
                assert len(calls) == 1 and tool.calls == 0
                release_first.set(); assert second.wait(8)
                page.wait_for_function("()=>activityFrames.some(f=>f.reasoning.some(r=>r.text.includes('增量末尾。')))")
                assert page.locator('.reasoning-content').inner_text() == before
                assert page.evaluate('getSelection().toString()') == selected
                page.evaluate('getSelection().removeAllRanges()')
                page.wait_for_function("()=>document.querySelector('.reasoning-content').innerText.includes('增量末尾。')")
                assert page.locator('.reasoning-content').evaluate('(n)=>n.scrollHeight-n.scrollTop-n.clientHeight<3')
                page.locator('#tools-button').click(); page.locator('#tools-panel').wait_for(state='visible')
                page.locator('#view-code').click(); page.locator('#tools-panel').wait_for(state='hidden')
                assert len(calls) == 1 and tool.calls == 0
                page.locator('#view-chat').click(); release_second.set()
                try:
                    page.get_by_text('已核验合成文件。', exact=True).wait_for()
                except Exception:
                    dest = Path(os.environ.get('ANGELOS_UI_EVIDENCE_DIR', str(tmp_path/'screenshots'))); dest.mkdir(parents=True, exist_ok=True)
                    (dest/'final-stream-failure.json').write_text(json.dumps(bench.store.list('conversation'), ensure_ascii=False, indent=2))
                    raise
                page.locator('#tools-button').click()
                page.locator('.tool-call-head strong').filter(has_text='demo_file').wait_for()
                assert tool.calls == 1 and tool.path.read_text() == 'standalone agent verified' and len(calls) == 3
                assert page.locator('#tools-panel').inner_text().count('已取得回执') >= 1
                file_card = page.locator('.tool-call').filter(has=page.locator('.tool-call-head strong').filter(has_text='demo_file'))
                file_card.locator('[data-field=arguments] summary').click()
                file_card.locator('[data-field=result] summary').click()
                assert str(tool.path) in file_card.inner_text() and 'result_valid' in file_card.inner_text()
                evidence = Path(os.environ.get('ANGELOS_UI_EVIDENCE_DIR', str(tmp_path/'screenshots'))); evidence.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(evidence/'live-tools-wide.png'))
                page.set_viewport_size({'width': 390, 'height': 844})
                page.wait_for_function("()=>document.querySelector('#tools-panel').getBoundingClientRect().top>=document.querySelector('.topbar').getBoundingClientRect().bottom")
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.screenshot(path=str(evidence/'live-tools-narrow.png'))
                page.set_viewport_size({'width': 1440, 'height': 1000})
                page.evaluate('''()=>{const n=document.querySelector('.tool-call pre').firstChild;
                  const r=document.createRange();r.selectNodeContents(n);getSelection().removeAllRanges();getSelection().addRange(r)}''')
                page.locator('#new-conversation').click()
                page.wait_for_function("()=>document.querySelector('#chat-document').hidden")
                page.evaluate("activityStreams[0].onmessage({data:JSON.stringify(activityFrames[0])})")
                assert page.locator('.reasoning-content').count() == 0
                assert page.locator('.tool-call').count() == 0
                assert page.locator('#tools-note').inner_text() == '选择一轮对话查看工具调用。'
                assert not errors, errors
                browser.close()
    finally:
        release_first.set(); release_second.set()
        if server: server.should_exit = True
        if thread: thread.join(8)
        if sock: sock.close()
        service.coordinator.shutdown(wait=True)
