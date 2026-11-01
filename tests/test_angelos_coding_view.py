"""真实浏览器/编辑器到附件和任务入口的离线回归；不连接供应商或正式服务。"""
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from shinku.agent_runtime.demo import demo_service
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls


@pytest.fixture
def coding(tmp_path, request):
    playwright = pytest.importorskip('playwright.sync_api')
    project = tmp_path/'project'
    project.mkdir()
    code = 'class Example:\n    def greet(self):\n        return "你好，主管"\n'
    (project/'sample.py').write_text(code)
    service, tool, runtime = demo_service(project)
    calls, errors = [], []
    entered, release = threading.Event(), threading.Event()
    def respond(**kwargs):
        calls.append(kwargs)
        if 'HOLD_TASK' in kwargs.get('user_prompt', ''):
            entered.set()
            assert release.wait(30), 'offline task was not released'
        return {'kind': 'final', 'text': '离线合成回答', 'task_outcome': 'completed'}
    runtime.call_chat_json = respond
    permissions = PermissionControls(tmp_path/'permissions.json', default='waw')
    host = ServiceHost(service, permissions=permissions)
    app = create_angelos_app(tmp_path/'data', host=host)
    bench = app.state.angelos
    workspace = bench.store.workspace({'name': '代码测试', 'path': str(project)})
    conversation = bench.store.conversation(workspace['id'])
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic()+8
    while not server.started and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.started
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.set_default_timeout(8000)
            page.add_init_script('localStorage.setItem("angelos:selection", '+json.dumps(json.dumps(
                {'workspace': workspace['id'], 'conversation': conversation['id']}))+');')
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{port}/angelos')
            page.locator(f'button[data-conversation="{conversation["id"]}"]').wait_for()
            page.locator(f'button[data-conversation="{conversation["id"]}"]').click()
            page.locator('#view-code').click()
            page.locator('#open-code-path').fill('sample.py')
            page.locator('#open-code-form button').filter(has_text='打开文件').click()
            page.locator('#code-editor .cm-content').last.wait_for()
            yield SimpleNamespace(page=page, bench=bench, workspace=workspace, conversation=conversation,
                                  project=project, code=code, calls=calls, tool=tool, entered=entered, release=release)
            evidence = os.environ.get("ANGELOS_UI_EVIDENCE_DIR")
            if evidence:
                dest = Path(evidence)
                dest.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(dest/(request.node.name+".png")), full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        release.set()
        server.should_exit = True
        thread.join(8)
        sock.close()
        service.coordinator.shutdown(wait=True)


def select_code(page, *, original=False):
    """使用键盘选区，覆盖只读/编辑模式真实 CodeMirror 事件，而非直接调用回调。"""
    view = page.locator('#code-editor .cm-content').first if original else page.locator('#code-editor .cm-content').last
    view.click()
    page.keyboard.press('Control+Home')
    page.keyboard.press('Control+a')
    page.locator('#ask-code:enabled').wait_for()


def test_code_selection_roundtrip_and_switch_preserves_unsaved_draft(coding):
    c, page = coding, coding.page
    page.locator('#code-mode').select_option('edit')
    view = page.locator('#code-editor .cm-content').last
    view.click()
    page.keyboard.press('Control+End')
    page.keyboard.insert_text('# 未保存\n')
    page.locator('#message-input').fill('解释这个类')
    select_code(page)
    for _ in range(3):
        page.locator('#view-chat').click()
        assert page.locator('#changes-panel').is_hidden()
        page.locator('#view-code').click()
    assert '# 未保存' in view.inner_text()
    assert page.locator('#message-input').input_value() == '解释这个类'
    assert not c.calls and c.tool.calls == 0
    assert (c.project/'sample.py').read_text() == c.code
    page.locator('#ask-code').click()
    page.locator('#pending-files .pending-file').wait_for()
    attachment = c.bench.store.list('attachment')[0]
    body = json.loads(attachment['content'])
    assert body['contentState'] == 'unsaved_draft'
    assert body['code'] == c.code+'# 未保存\n'
    assert body['path'] == 'sample.py' and body['range']['startLine'] == 1
    assert body['range']['endLine'] == 5 and body['range']['endColumn'] == 1
    assert attachment['conversationId'] == c.conversation['id']
    page.locator('#send-message').click()
    page.locator('#messages').filter(has_text='离线合成回答').wait_for()
    assert len(c.calls) == 1 and '未保存' in c.calls[0]['user_prompt']
    assert 'diskVersionAtRead' in c.calls[0]['user_prompt']
    assert c.tool.calls == 0


def test_new_conversation_question_leaves_running_task_and_permissions_alone(coding):
    c, page = coding, coding.page
    page.locator('#message-input').fill('HOLD_TASK')
    page.locator('#send-message').click()
    assert c.entered.wait(5)
    # 模型开始等待早于浏览器处理提交响应；先等页面完成提交，避免其焦点归位抢走选区键盘输入。
    page.locator('#stop-response').wait_for()
    before = c.bench.store.read('conversation', c.conversation['id'])['turns'][0]
    task_id = before['task_id']
    revision = c.bench.host.permissions.revision
    # 对照栏与当前栏明确区分，不把对照版本标成最新文件。
    select_code(page, original=True)
    page.locator('#view-chat').click()
    page.locator('#view-code').click()
    page.locator('#ask-code-new').click()
    page.locator('#pending-files .pending-file').wait_for()
    attachment = c.bench.store.list('attachment')[0]
    assert attachment['conversationId'] != c.conversation['id']
    body = json.loads(attachment['content'])
    assert body['contentState'] == 'comparison_snapshot' and body['code'] == c.code
    old = c.bench.store.read('conversation', c.conversation['id'])['turns'][0]
    assert old['stage'] == 'executing' and old['task_id'] == task_id
    assert c.bench.host.permissions.revision == revision
    assert len(c.calls) == 1 and c.tool.calls == 0
    c.release.set()
    page.locator('#message-input').fill('解释选中的代码')
    page.locator('#send-message').click()
    page.locator('#messages').filter(has_text='离线合成回答').wait_for()
    assert len(c.calls) == 2
    old = c.bench.refresh(c.conversation['id'])['turns'][0]
    assert old['view']['status'] == 'completed'


def test_selection_scope_limits_and_narrow_layout(coding):
    c, page = coding, coding.page
    page.locator('#code-mode').select_option('edit')
    select_code(page)
    page.keyboard.insert_text('x'*12001)
    select_code(page)
    page.locator('#ask-code').click()
    page.locator('#connection-error').filter(has_text='选区过长').wait_for()
    assert not c.bench.store.list('attachment')
    # 切换工作区保留旧草稿，但禁止把它附到另一个工作区的会话。
    other = c.bench.store.workspace({'name': '其他项目', 'path': str(c.project.parent)})
    c.bench.store.conversation(other['id'])
    page.evaluate('refresh(true)')
    page.locator('[data-workspace]').first.click()
    page.locator(f'[data-select-workspace="{other["id"]}"]').click()
    page.locator('#ask-code:disabled').wait_for()
    assert '其他工作区' in page.locator('#code-selection-note').inner_text()
    page.set_viewport_size({'width': 720, 'height': 1000})
    page.wait_for_timeout(100)
    panel = page.locator('#changes-panel').bounding_box()
    composer = page.locator('#message-input').bounding_box()
    assert composer['y'] >= panel['y']+panel['height']
    assert composer['y']+composer['height'] <= 1000
    assert not c.calls


def test_delayed_selection_upload_stays_with_original_conversation(coding):
    c, page = coding, coding.page
    # 单行部分选区，验证行列定位和只读模式；上传延迟时用户切去新会话。
    view = page.locator('#code-editor .cm-content').last
    view.click()
    page.keyboard.press('Control+Home')
    page.keyboard.press('Shift+End')
    page.locator('#ask-code:enabled').wait_for()
    pending = []
    page.route('**/attachments', lambda route: pending.append(route))
    page.locator('#ask-code').click()
    deadline = time.monotonic()+5
    while not pending and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert len(pending) == 1
    # 空白会话现在会复用；先留下草稿，确保本用例确实切换会话再验证迟到附件归属。
    page.locator('#message-input').fill('继续解释所选代码')
    page.locator('#new-conversation').click()
    page.locator('button.conversation-item[aria-current=true]').filter(has_text='新会话').wait_for()
    # 新旧会话同名时进一步等待实际 id 改变。
    deadline = time.monotonic()+5
    while page.locator('button.conversation-item[aria-current=true]').get_attribute('data-conversation') == c.conversation['id'] and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert page.locator('button.conversation-item[aria-current=true]').get_attribute('data-conversation') != c.conversation['id']
    pending[0].continue_()
    page.locator('#toast').filter(has_text='目标会话草稿').wait_for()
    assert page.locator('#pending-files .pending-file').count() == 0
    attachment = c.bench.store.list('attachment')[0]
    body = json.loads(attachment['content'])
    assert attachment['conversationId'] == c.conversation['id']
    assert body['code'] == 'class Example:' and body['contentState'] == 'read_snapshot'
    assert body['range'] == {'startLine': 1, 'startColumn': 1, 'endLine': 1, 'endColumn': 15}
    page.locator(f'button[data-conversation="{c.conversation["id"]}"]').click()
    page.locator('#pending-files .pending-file').wait_for()
    assert not c.calls and c.tool.calls == 0


@pytest.mark.parametrize('style', ['absolute', 'quoted', 'relative'])
def test_path_input_opens_unicode_file_and_keeps_relative_attachment(coding, style):
    c, page = coding, coding.page
    target = c.project/'目录 空格'/'示例.py'
    target.parent.mkdir()
    target.write_text('answer = "你好"\n')
    value = str(target) if style == 'absolute' else '"'+str(target)+'"' if style == 'quoted' else './目录 空格/示例.py'
    page.locator('#open-code-path').fill(value)
    page.locator('#open-code-form button').filter(has_text='打开文件').click()
    page.locator('#code-path').filter(has_text='目录 空格/示例.py').wait_for()
    assert '你好' in page.locator('#code-editor .cm-content').last.inner_text()
    select_code(page)
    page.locator('#ask-code').click()
    page.locator('#pending-files .pending-file').wait_for()
    body = json.loads(c.bench.store.list('attachment')[0]['content'])
    assert body['path'] == '目录 空格/示例.py'
    assert body['workspace'] == str(c.project) and body['code'] == 'answer = "你好"\n'
    assert not c.calls


def test_absolute_path_rejects_outside_prefix_traversal_and_symlink(coding):
    c, page = coding, coding.page
    sibling = c.project.with_name(c.project.name+'-other')
    sibling.mkdir()
    secret = sibling/'outside.py'
    secret.write_text('OUTSIDE_CONTENT')
    (c.project/'link.py').symlink_to(secret)
    for path, expected in [(str(secret), '不在当前工作区'),
                           (str(c.project/'..'/'project-other'/'outside.py'), '不能包含 ..'),
                           (str(c.project), '需要文件路径'),
                           (str(c.project/'link.py'), '符号链接')]:
        page.locator('#open-code-path').fill(path)
        page.locator('#open-code-form button').filter(has_text='打开文件').click()
        page.locator('#code-status').filter(has_text=expected).wait_for()
        assert 'OUTSIDE_CONTENT' not in page.locator('#code-editor').inner_text()
        assert page.locator('#code-path').inner_text() == 'sample.py'
    page.locator('#open-code-form [data-workspace]').click()
    page.locator('#workspace-dialog').wait_for()
    assert not c.calls and c.tool.calls == 0


def test_native_picker_file_directory_and_cancel(coding):
    c, page = coding, coding.page
    selected = c.project/'选择 文件.py'
    selected.write_text('class Selected: pass\n')
    calls = []
    class Picker:
        available = True
        cancelled = False
        async def choose(self, kind, directory, disconnected):
            calls.append((kind, directory))
            return {'cancelled': self.cancelled, 'path': str(selected if kind == 'file' else c.project)}
    picker = Picker()
    c.bench.native_picker = picker
    page.reload()
    page.locator(f'button[data-conversation="{c.conversation["id"]}"]').click()
    page.locator('#view-code').click()
    page.locator('#browse-code').click()
    page.wait_for_function("() => document.querySelector('#open-code-path').value.endsWith('选择 文件.py')")
    page.locator('#code-editor .cm-content').filter(has_text='class Selected').wait_for()
    before_content = page.locator('#code-editor .cm-content').last.inner_text()
    picker.cancelled = True
    page.locator('#browse-code').click()
    page.locator('#code-status').filter(has_text='已取消').wait_for()
    assert page.locator('#code-editor .cm-content').last.inner_text() == before_content
    before = c.bench.store.list('workspace')
    page.locator('#open-code-form [data-workspace]').click()
    page.locator('#show-create-workspace').click()
    page.locator('#workspace-name').fill('浏览创建')
    picker.cancelled = False
    page.locator('#browse-workspace').click()
    page.wait_for_function('() => document.querySelector("#workspace-path").value !== ""')
    assert page.locator('#workspace-path').input_value() == str(c.project)
    assert c.bench.store.list('workspace') == before
    page.locator('#save-workspace').click()
    page.wait_for_function('() => !document.querySelector("#workspace-dialog").open')
    assert any(w['name'] == '浏览创建' for w in c.bench.store.list('workspace'))
    assert c.calls == []


def test_late_native_directory_does_not_overwrite_new_form(coding):
    import asyncio
    c, page = coding, coding.page
    started, release = threading.Event(), threading.Event()
    class Picker:
        available = True
        async def choose(self, *args):
            started.set()
            while not release.is_set():
                await asyncio.sleep(.01)
            return {'cancelled': False, 'path': str(c.project)}
    c.bench.native_picker = Picker()
    page.reload()
    page.locator(f'button[data-conversation="{c.conversation["id"]}"]').click()
    page.locator('#view-code').click()
    page.locator('#open-code-form [data-workspace]').click()
    page.locator('#show-create-workspace').click()
    try:
        page.locator('#browse-workspace').click()
        assert started.wait(5)
        assert page.request.get(page.url.replace('/angelos', '/angelos/api/bootstrap')).status == 200
        page.locator('#cancel-workspace-form').click()
        page.locator('#show-create-workspace').click()
        page.locator('#workspace-path').fill('/manual/new')
    finally:
        release.set()
    page.locator('#browse-workspace:enabled').wait_for()
    assert page.locator('#workspace-path').input_value() == '/manual/new'
    assert c.calls == []
