"""真实 Chromium 的隔离页面联调，不使用用户标签页、登录态或外部站点。"""
import hashlib
from pathlib import Path

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.tools.browser import build_browser_tools
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation


HTML = '''<html><head><title>Isolated fixture</title></head><body>
<input placeholder="Title"><select aria-label="Choice"><option value="a">A</option><option value="b">B</option></select>
<input type="file" aria-label="Upload"><input type="password" placeholder="Password">
<button onclick="window.count=(window.count||0)+1;document.querySelector('output').textContent='Saved '+window.count">Save</button>
<a download="fixture.txt" href="data:text/plain,isolated-download">Download</a><output>Waiting</output></body></html>'''


def context(session='session', mode='full'):
    return {'agent_request_scope': RequestScope('owner', session, 'turn', (('turn', 'isolated form'),)),
            'agent_permission_mode': mode}


@pytest.fixture
def browser(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as p:
        if not Path(p.chromium.executable_path).is_file():
            pytest.skip('isolated Chromium unavailable; no auto-install')
        runtime = p.chromium.launch(headless=True)
        ctx = runtime.new_context(accept_downloads=True)
        page = ctx.new_page(); page.set_content(HTML)
        tools = build_browser_tools(page_provider=lambda context: page, run=lambda f: f(),
            authorize=lambda ctx, action, url: ctx.get('agent_permission_mode') == 'full' or action == 'observe',
            download_root=tmp_path / 'downloads',
            upload_reader=lambda ctx, path: {'name': 'fixture.txt', 'mimeType': 'text/plain', 'buffer': b'isolated-upload'})
        try:
            yield page, {t.tool_type: t for t in tools}
        finally:
            ctx.close(); runtime.close()


def invoke(tools, name, args=None, ctx=None, policy=None):
    return execute_invocation(ToolInvocation(name, args or {}), handlers=tools,
                              context=ctx or context(), policy=policy)[1]


def observe(tools):
    result = invoke(tools, 'browser_observe')
    assert result.status == 'ok', result.model_feedback
    return result.data['state_updates']['browser_observation']


def arguments(state, action, label, key, **extra):
    return {'action': action, 'ref': next(t['ref'] for t in state['targets'] if t['label'] == label),
            'state_id': state['state_id'], 'request_key': key, **extra}


def apply(tools, args):
    result = invoke(tools, 'browser_interact', args)
    assert result.status == 'ok', result.model_feedback
    return result.data['state_updates']


def test_real_fill_select_upload_submit_wait_and_download(browser):
    page, tools = browser
    state = observe(tools)
    assert 'password' not in state['text'] and state['targets_complete']
    state = apply(tools, arguments(state, 'fill', 'Title', 'fill', value='合成任务'))['browser_observation']
    assert page.locator('input').first.input_value() == '合成任务'
    state = apply(tools, arguments(state, 'select', 'Choice', 'select', value='b'))['browser_observation']
    assert page.locator('select').input_value() == 'b'
    state = apply(tools, arguments(state, 'upload', 'Upload', 'upload', path='fixture.txt'))['browser_observation']
    assert page.locator('input[type=file]').evaluate('(e)=>e.files[0].name') == 'fixture.txt'
    args = arguments(state, 'click', 'Save', 'save')
    result = apply(tools, args)
    assert 'Saved 1' in result['browser_observation']['text']
    assert apply(tools, args) == result and page.evaluate('window.count') == 1
    state = result['browser_observation']
    state = apply(tools, arguments(state, 'wait', 'Save', 'wait', value='Save'))['browser_observation']
    receipt = apply(tools, arguments(state, 'download', 'Download', 'download'))['action_receipt']['download']
    assert Path(receipt['path']).read_bytes() == b'isolated-download'
    assert receipt['sha256'] == hashlib.sha256(b'isolated-download').hexdigest()


def test_stale_state_scope_and_policy_never_submit(browser):
    page, tools = browser
    state = observe(tools)
    args = arguments(state, 'click', 'Save', 'save')
    assert invoke(tools, 'browser_interact', args, ctx=context(mode='read')).status == 'error'
    assert invoke(tools, 'browser_interact', args, policy=ExecutionPolicy(blocked=frozenset({'browser_interact'}))).executed is False
    assert invoke(tools, 'browser_observe', ctx=context('other')).status == 'error'
    page.locator('output').evaluate('(e)=>e.textContent="changed by outside"')
    assert invoke(tools, 'browser_interact', args).status == 'error'
    assert page.evaluate('window.count||0') == 0
    fresh = observe(tools)
    apply(tools, arguments(fresh, 'click', 'Save', 'save-fresh'))
    assert page.evaluate('window.count') == 1


def test_uncertain_effect_is_recorded_and_not_replayed(browser, monkeypatch):
    page, tools = browser
    state = observe(tools)
    args = arguments(state, 'click', 'Save', 'one-action')
    access = tools['browser_interact'].access
    original_observe = access.observe
    monkeypatch.setattr(access, 'observe', lambda *a: (_ for _ in ()).throw(RuntimeError('after dispatch lost')))
    first = invoke(tools, 'browser_interact', args)
    second = invoke(tools, 'browser_interact', args)
    assert first.status == second.status == 'error' and page.evaluate('window.count') == 1
    assert 'browser_effect_uncertain' in first.model_feedback
    monkeypatch.setattr(access, 'observe', original_observe)
    fresh = observe(tools)
    third = invoke(tools, 'browser_interact', arguments(fresh,'click','Save','different-key'))
    assert third.status == 'error' and page.evaluate('window.count') == 1


def test_navigation_and_target_changes_require_observation(browser):
    page, tools = browser
    state = observe(tools)
    args = {'action': 'navigate', 'url': 'data:text/html,<button>New</button>', 'state_id': state['state_id'], 'request_key': 'nav'}
    result = apply(tools, args)
    assert result['browser_observation']['targets'][0]['label'] == 'New'
    assert invoke(tools, 'browser_interact', arguments(state, 'click', 'Save', 'old-target')).status == 'error'


def test_wait_observes_later_text_and_receipts_reject_forgery(browser):
    from copy import deepcopy
    page, tools = browser
    state = observe(tools)
    # 先取得稳定初始观察，再触发待等待的变化；计时器不能与初始多步采样竞跑。
    # 保留真实DOM条件等待和所有回执/伪造断言，不用延长睡眠掩盖身份变化。
    page.evaluate('setTimeout(()=>document.querySelector("button").textContent="Ready",200)')
    result = apply(tools, arguments(state, 'wait', 'Save', 'wait-later', value='Ready'))
    handler = tools['browser_interact']
    assert 'Ready' in result['browser_observation']['text']
    assert handler.agent_verify_result(receipt=result, context=context())['result_valid']
    forged = deepcopy(result); forged['observation']['self_state']['data_sha256'] = 'forged'
    assert not handler.agent_verify_result(receipt=forged, context=context())['result_valid']
    assert not handler.agent_verify_result(receipt=result, context=context('other'))['result_valid']
