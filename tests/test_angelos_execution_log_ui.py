"""完整执行日志面板的浏览器实测（隔离本地服务 + 合成 shell 工具）。

为什么需要这个用例：
- 分页/搜索/导出/滚动跟随属于页面行为，单元测试只能证明数据正确，不能证明界面按预期工作；
- 这里用真实 AgentService、真实材料库和真实 FastAPI 静态资源，只把模型执行器换成产生
  确定输出的合成 shell 工具，因此测到的是实际接线，而不是另一套替身界面。

覆盖：退出码与完整性标记来自回执而不是"成功"字样；中间标记可搜索并跳转；选中文本后
重绘不丢选区；上翻时不抢滚动位置；回到底部恢复跟随；合成口令不泄露；刷新后记录仍可读
且行号不重复。截图目录由 ANGELOS_LOG_UI_EVIDENCE_DIR 指定。
"""
import hashlib
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from shinku.agent.task_state import AgentTaskStore
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.agent_runtime.contracts import AgentRequest
from shinku.agent_runtime.execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.agent_runtime.service import AgentService
from shinku.agent_runtime import AgentScope
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost

# 合成口令分段拼接，源文件不出现形似真实凭据的字面量。
SYNTHETIC = '-'.join(['sk', 'LOG', 'UI', 'FIXTURE', '1234'])
LINE_COUNT = 900   # 超过模型摘录阈值，原件才会被保留
PAD = 'x' * 40     # 每行补足长度，确保总量远超摘录阈值
MARKER = 'MIDDLE-MARKER-LINE'


class ShellFixtureTool:
    """合成 shell：一次调用产出大量标准输出、一段错误输出与确定退出码。"""

    tool_type = 'shell'
    agent_progress_class = 'change'
    # 工具能力表由这些声明生成；契约只接受能力表里声明的效果，所以必须如实列出。
    agent_completion_effects = ('command_run', 'write')

    def __init__(self, out=None):
        self.calls = 0
        self.out = out

    def tool_metadata(self):
        return SimpleNamespace(description='Synthetic shell for log panel.', input_schema={
            'type': 'object', 'properties': {'cmd': {'type': 'string'}}, 'required': ['cmd']})

    def normalize_call(self, call):
        return call if call.get('cmd') == 'synthetic-build' else None

    def execute(self, *, call, context):
        self.calls += 1
        lines = [f'build step {i:05d} ok ' + PAD + '\n' for i in range(LINE_COUNT)]
        lines[LINE_COUNT // 2] = MARKER + ' build finished\n'
        lines.insert(1, 'using ' + 'to' + 'ken=' + SYNTHETIC + '\n')
        artifact = {}
        if self.out is not None:
            Path(self.out).write_text('build step 00000 ok\n', encoding='utf-8')
            artifact = {'path': str(self.out), 'verified_effects': ['write', 'command_run'],
                        'completion_targets': [str(self.out)],
                        'sha256': hashlib.sha256(Path(self.out).read_bytes()).hexdigest(),
                        'bytes': Path(self.out).stat().st_size, 'result_valid': True}
        return {'tool_type': self.tool_type, 'followup_context': 'Synthetic build finished.',
                'state_updates': {
                    'execution_session': {'session_id': 'fx-session', 'command_sha256': 'b' * 64,
                                          'returncode': 1, 'started_at': 1793000000.0,
                                          'finished_at': 1793000003.0, 'elapsed': 3.0},
                    'stdout': ''.join(lines), 'stderr': 'warning line kept for diagnosis\n',
                    'executed': True, 'command_run': True, 'command': 'synthetic-build',
                    # 完成校验读取顶层的 completion_targets，所以产物字段必须与命令字段同层。
                    **artifact,
                    'verified_effects': ['command_run', *artifact.get('verified_effects', [])],
                    'artifact_receipts': [artifact] if artifact else []}}

    def agent_verify_result(self, *, receipt, context):
        # 产物校验：与真实工具一样重新读文件核对摘要，供完成验收使用。
        # receipt 里可能是外层 state_updates，也可能直接是 result 子回执，两者都接受。
        payload = receipt.get('result') if isinstance(receipt.get('result'), dict) else receipt
        target = Path(payload.get('path') or '')
        exists = target.is_file()
        digest = hashlib.sha256(target.read_bytes()).hexdigest() if exists else ''
        refreshed = {'result_valid': exists and digest == payload.get('sha256'),
                     'file_text': target.read_text(encoding='utf-8') if exists else ''}
        if 'artifact_receipts' in receipt:
            refreshed['artifact_receipts'] = [{**row, **refreshed} for row in receipt['artifact_receipts']]
        return refreshed


class FixtureRuntime:
    """最小模型替身：先要求执行一次合成 shell，再给出最终答复。

    按宿主运行时协议实现 call_chat_json 与能力探针：探针缺失时执行器会判定
    execution_capability_unknown 并阻塞任务，所以这里明确声明支持 native tools。
    """

    def __init__(self, tool):
        self.tool = tool
        self.calls = []

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.tool.calls:
            return {'tool_call': {'type': 'shell', 'cmd': 'synthetic-build'}}
        return {'kind': 'final', 'text': '合成任务完成。', 'task_outcome': 'completed'}


def _service(out_path):
    tool = ShellFixtureTool(out_path)
    runtime = FixtureRuntime(tool)
    executor = AgentExecutor(ToolHost(ToolRegistry({tool.tool_type: tool})),
                             ExecutionProfile(runtime))
    grant = ExecutionGrant('log-ui', frozenset({tool.tool_type}),
                           policy=ExecutionPolicy(risk_by_tool={tool.tool_type: 'low'}),
                           budget_limits={'max_requests': 12, 'max_tokens': 600000,
                                          'max_tools': 8, 'max_active_seconds': 60})
    store = AgentTaskStore()
    # 与真实宿主一致地挂材料库：长输出才会被保留为可补读原件，供用户侧日志读取。
    service = AgentService(executor=executor, authorize=lambda scope, task, operation: grant,
                           task_store=store, material_store=TaskMaterialStore(store),
                           coordinator=AgentTaskCoordinator(max_workers=1))
    return service, tool


def _request(out_path):
    # 契约必须有至少一条可核对的要求；这里用合成命令的真实退出契约，不为通过而放宽校验。
    return AgentRequest('log-ui-key', 'Run the synthetic build and report.', {
        'goal': 'Run synthetic build', 'deliverables': [str(out_path)],
        'scope': [str(Path(out_path).parent)],
        # command_run 契约要求退出码为 0；本用例故意产出退出码 1 的执行回执，
        # 所以契约用 write 效果描述“写出构建结果”，不去改放宽契约规则。
        'requirements': [{'id': 'out', 'description': 'Synthetic build output written',
                          'effect': 'write', 'targets': [str(out_path)],
                          'checks': {'contains': 'build step'}}]})


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def live(tmp_path):
    out_path = tmp_path / 'build.log'
    service, tool = _service(out_path)
    app = create_angelos_app(tmp_path / 'ui', host=ServiceHost(service))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(.05)
    yield {'base': f'http://127.0.0.1:{port}', 'bench': app.state.angelos,
           'host': app.state.angelos.host, 'tool': tool, 'out': out_path}
    server.should_exit = True
    thread.join(8)
    service.coordinator.shutdown(wait=True)


def _run_task(bench, host, out_path):
    """建会话、跑真实任务，并把该轮绑定到会话的 turn 上。"""
    # 先读工作区列表再开写事务：store.list 自身会开事务，嵌套会触发 SQLite 锁等待。
    workspace_id = bench.store.list('workspace')[0]['id']
    with bench.store.transaction() as db:
        conversation = {'id': 'log-ui', 'workspaceId': workspace_id,
                        'title': '合成日志', 'draft': '', 'updated': time.time(), 'pending': [],
                        'turns': [], 'modelSettings': None}
        bench.store.put(db, 'conversation', conversation)
    cid = conversation['id']
    handle = host.port(cid).submit(_request(out_path))
    scope = AgentScope(host.requester, 'angelos:' + cid)
    deadline = time.time() + 40
    saved = None
    while time.time() < deadline:
        with host.service._lock:
            saved = host.service._get(scope, handle.task_id)
        if saved.get('status') in ('completed', 'failed', 'canceled', 'blocked'):
            break
        time.sleep(.15)
    conversation['turns'] = [{'id': 'turn-1', 'stage': 'replied',
                              'text': 'Run the synthetic build and report.',
                              'reply': '合成任务完成。', 'attachments': [],
                              'task_id': handle.task_id}]
    with bench.store.transaction() as db:
        bench.store.put(db, 'conversation', conversation)
    return cid, handle, saved


def test_log_panel_pages_searches_selection_and_follow(tmp_path, live):
    sync_playwright = pytest.importorskip('playwright.sync_api').sync_playwright
    bench, host = live['bench'], live['host']
    cid, handle, saved = _run_task(bench, host, live['out'])
    assert saved and saved.get('status') == 'completed', saved.get('status') if saved else None
    assert live['tool'].calls == 1
    # 调用 id 由宿主生成，不能假设等于请求里的标识；从账本取真实值。
    call_ids = [str(c.get('id')) for c in (saved.get('tool_calls') or [])]
    assert call_ids, '任务没有记录任何工具调用'
    call_id = call_ids[0]

    evidence = Path(os.environ.get('ANGELOS_LOG_UI_EVIDENCE_DIR', str(tmp_path / 'screenshots')))
    evidence.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        page.goto(live['base'] + '/angelos')
        page.wait_for_selector('#tools-button')
        page.locator('#tools-button').click()
        page.locator(f'[data-log="{call_id}"]').wait_for()
        page.locator(f'[data-log="{call_id}"]').click()
        page.locator('#log-panel').wait_for(state='visible')
        page.wait_for_function(
            "()=>document.querySelectorAll('#log-body .log-line').length>10")

        # 1) 退出码与完整性来自回执字段，不是从"成功"字样推断。
        meta = page.locator('#log-body .log-meta').inner_text()
        assert '退出码 1' in meta and '完整原件' in meta

        # 2) 搜索中间标记并能跳转到该行。
        page.fill('#log-search', MARKER)
        page.click('#log-search-button')
        page.locator('.log-hit').first.wait_for()
        assert '1 处命中' in page.locator('#log-search-result').inner_text()
        page.locator('.log-hit').first.click()
        page.wait_for_function(
            "m=>[...document.querySelectorAll('#log-body .log-line')]"
            ".some(n=>n.textContent.includes(m)&&n.classList.contains('log-hit-line'))", arg=MARKER)
        page.screenshot(path=str(evidence / 'log-panel-search.png'))

        # 3) 选中文本后重绘不丢选区。
        page.evaluate("""()=>{const n=[...document.querySelectorAll('#log-body .log-text')]
          .find(x=>x.textContent.includes('build step'));
          const r=document.createRange();r.selectNodeContents(n);
          const s=getSelection();s.removeAllRanges();s.addRange(r)}""")
        before = page.evaluate("()=>getSelection().toString()")
        assert before
        page.evaluate("()=>{const b=document.querySelector('#log-body');b.dispatchEvent(new Event('scroll'))}")
        page.wait_for_timeout(150)
        assert page.evaluate("()=>getSelection().toString()") == before

        # 4) 上翻后不抢滚动：出现"回到底部"入口。
        page.evaluate("()=>{const b=document.querySelector('#log-body');b.scrollTop=0;"
                      "b.dispatchEvent(new Event('scroll'))}")
        page.wait_for_timeout(100)
        assert page.locator('#log-jump').is_visible()

        # 5) 回到底部后恢复跟随。
        page.click('#log-jump')
        page.wait_for_timeout(150)
        assert page.evaluate(
            "()=>{const b=document.querySelector('#log-body');"
            "return b.scrollHeight-b.scrollTop-b.clientHeight<48}") is True
        page.screenshot(path=str(evidence / 'log-panel-follow.png'))

        # 6) 页面不泄露合成口令，其余排错细节保留。
        body = page.locator('#log-body').inner_text()
        assert SYNTHETIC not in body
        assert 'build step' in body

        # 7) 刷新后记录仍可读且行号不重复。
        page.reload()
        page.wait_for_selector('#tools-button')
        page.locator('#tools-button').click()
        page.locator(f'[data-log="{call_id}"]').wait_for()
        page.locator(f'[data-log="{call_id}"]').click()
        page.wait_for_function(
            "()=>document.querySelectorAll('#log-body .log-line').length>10")
        numbers = page.evaluate(
            "()=>[...document.querySelectorAll('#log-body .log-line')].map(n=>Number(n.dataset.n))")
        assert len(numbers) == len(set(numbers)), '刷新后出现重复行'
        page.screenshot(path=str(evidence / 'log-panel-after-reload.png'))
        assert not errors, errors
        browser.close()
