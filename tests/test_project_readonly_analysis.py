"""合成项目读取故障经过真实SDK、工作区适配器、读取账本和完成门回归。

HTTP响应是固定夹具，只验证执行链路；不据此声称真实模型质量或缓存收益。
"""
from contextlib import contextmanager
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.tools.workspace import build_workspace_tools
from test_agent_conversation_loop import native_sdk
from test_agent_runtime_boundary import settled


FILES = {'app.py': 'def build_service(config):\n    return {"mode":config["mode"], "timeout":config["timeout"]}\n',
         'config.json': '{"mode":"readonly","timeout":30}',
         'README.txt': '合成项目：app.py按配置构建服务状态，未实现网络监听或任务调度。只读分析，不修改、不安装、不发送。'}
ANSWER = ('程序入口：app.py 提供 build_service(config)，返回包含 mode 和 timeout 的字典；'
          '没有看到主程序启动代码。\n配置作用：config.json 给出 mode=readonly、timeout=30，'
          '函数读取这两个键并传入返回值；源码没有体现超时单位、计时或只读权限检查。'
          'README.txt 说明未实现网络监听或任务调度。\n维护建议：函数直接下标读取两个配置键，'
          '建议补充缺键检查与清晰的错误说明；这是建议，没有修改文件或执行程序。')


@contextmanager
def project_run(root, *, boundary='none', max_requests=12):
    workspace = root / 'workspace'; workspace.mkdir()
    for name, text in FILES.items():
        (workspace / name).write_text(text)
    original = {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest() for name in FILES}
    contract = {'goal': '分析合成项目', 'deliverables': ['入口、配置作用、维护建议'], 'scope': ['只读'],
        'requirements': [
            *[{'id': name, 'description': '完整读取且不改变原文', 'effect': 'read',
               'targets': [str(workspace / name)], 'checks': {'unchanged': True}} for name in FILES],
            {'id': 'answer', 'description': '分析正文', 'effect': 'local_answer', 'targets': [],
             'checks': {'required_fields': ['程序入口', '配置作用', '维护建议'], 'min_sources': 3}}]}
    active = [True]
    handlers = {h.tool_type: h for h in build_workspace_tools(workspace,
        authorize=lambda context, write: active[0] and not write) if h.agent_permission_class == 'read'}
    service, business_tool, _ = demo_service(root)
    service.executor.tool_host = ToolHost(ToolRegistry(handlers))
    authorize = service.authorize
    service.authorize = lambda *args: replace(authorize(*args), permission_mode='read',
        allowed_tools=frozenset(handlers), policy=ExecutionPolicy(default_risk='low'),
        budget_limits={'max_requests': max_requests, 'max_tokens': 500000, 'max_tools': 24, 'max_active_seconds': 30},
        still_authorized=lambda: active[0])

    def responder(payload, index):
        if boundary == 'changed' and index == 3:
            # 完成门必须先否决旧原稿，再允许模型按来源失效暂停；不能用传输异常代替此检查。
            assert '已用于答复的来源变更或授权失效' in str(payload['messages'])
            return {'role': 'assistant', 'content': json.dumps({'kind': 'wait',
                'text': '配置来源已经变化，暂停使用旧内容交付完整分析。', 'reason': 'source_changed'}, ensure_ascii=False)}, 'stop'
        assert index <= 2, '已完整读取三个文件，不应重复读取或追加审查请求'
        if index == 1:
            calls = [('declare_task_contract', {'task_contract': contract})]
            # 与故障相同：声明和三份绝对路径读取在同一条原生响应中。
            for name in FILES:
                if boundary == 'missing' and name == 'README.txt':
                    continue
                calls.append(('workspace_read_lines', {'path': str(workspace / name), 'start_line': 1, 'line_count': 400}))
            return {'role': 'assistant', 'content': '', 'reasoning_content': 'SYNTHETIC_REASONING',
                    'tool_calls': [{'id': f'call_{i}', 'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(calls)]}, 'tool_calls'
        if boundary == 'changed':
            (workspace / 'config.json').write_text('{"mode":"readonly","timeout":60}')
        if boundary == 'revoked':
            active[0] = False
        return {'role': 'assistant', 'content': json.dumps(
            {'kind': 'final', 'text': ANSWER, 'task_outcome': 'completed'}, ensure_ascii=False)}, 'stop'

    try:
        with native_sdk(responder) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            host = ServiceHost(service, requester='owner', diagnostic_units=())
            port = host.port('one')
            request = '完整读取合成工作区的三个文件，说明入口、配置作用和有依据的维护建议。不修改文件。工作区路径：' + str(workspace)
            handle = port.submit(AgentTurnRequest('project', request))
            view = settled(port, handle.task_id)
            result = port.result(handle.task_id)
            task = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
            yield view, result, task, calls
            assert business_tool.calls == 0
            if boundary != 'changed':
                assert original == {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest() for name in FILES}
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_absolute_three_file_analysis_reaches_delivery_without_path_retry(tmp_path):
    with project_run(tmp_path) as (view, result, task, calls):
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert result['verification']['accepted'] and result['execution_report']['text'] == ANSWER
        assert len(calls) == result['budget']['requests'] == 2
        assert result['budget']['reported_tokens'] == 32 and result['budget']['estimated_requests'] == 0
        reads = [c for c in task['tool_calls'] if c['name'] == 'workspace_read_lines']
        assert len(reads) == 3 and all(c['status'] == 'ok' for c in reads)
        assert {c['arguments']['path'] for c in reads} == {str(tmp_path / 'workspace' / n) for n in FILES}
        for call in reads:
            receipt = call['data']['state_updates']
            assert receipt['path'] in FILES and receipt['read_complete']
            assert receipt['sha256'] == hashlib.sha256(FILES[receipt['path']].encode()).hexdigest()
            # 紧凑回执存来源/版本，原文在原生配对消息中保留；二者都不能被摘要替代。
            assert any(m['role'] == 'tool' and m['tool_call_id'] == call['id']
                       and FILES[receipt['path']] in json.loads(m['content'].split('\n\n[host receipt]')[0])['text']
                       for m in calls[1]['messages'] if m['role'] == 'tool')
        assert all('max_tokens' not in p and 'max_completion_tokens' not in p for p in calls)
        assert len([m for m in calls[1]['messages'] if m['role'] == 'tool']) == 4


@pytest.mark.parametrize('boundary', ['missing', 'changed', 'revoked'])
def test_analysis_cannot_claim_complete_without_current_authorized_sources(tmp_path, boundary):
    with project_run(tmp_path, boundary=boundary) as (view, result, task, calls):
        assert view['status'] != 'completed'
        assert not any(row.get('business_goal') is True for row in result['verification'].get('evidence', []))
        assert len(calls) == (3 if boundary == 'changed' else 2)
        assert len([c for c in task['tool_calls'] if c['name'] == 'workspace_read_lines']) == (2 if boundary == 'missing' else 3)
        if boundary == 'revoked':
            assert view['reason'] == 'agent_permission_changed'
        elif boundary == 'missing':
            assert view['status'] == 'partial'
            assert result['verification']['layers']['plan_gaps'] == ['README.txt']
            assert 'README.txt' not in result['verification']['satisfied']
        else:
            # 来源失效已在下一次请求前明确否决；交付阶段的wait也不能冒充分析完成。
            assert view['status'] == 'failed' and view['reason'] == 'local_analysis_nonfinal_decision'
            assert not result['verification']['accepted']


def test_request_budget_stop_preserves_reads_and_does_not_report_revocation(tmp_path):
    with project_run(tmp_path, max_requests=1) as (view, result, task, calls):
        assert view['status'] == 'blocked' and view['reason'] == 'task_model_budget_exhausted'
        assert not result['verification'].get('accepted', False) and len(calls) == 1
        assert result['budget']['reported_tokens'] == 16
        assert len([c for c in task['tool_calls'] if c['name'] == 'workspace_read_lines' and c['status'] == 'ok']) == 3


@pytest.mark.parametrize('change', ['none', 'changed', 'revoked', 'foreign_target', 'tampered_source',
                                    'tampered_path', 'unbound_source', 'partial'])
def test_restored_absolute_read_records_revalidate_canonical_sources(tmp_path, change):
    from shinku.agent.local_answer import LocalAnswerLedger
    from shinku.agent.loop import _call_signature
    from shinku.agent_runtime.context import execution_context
    from shinku.tools.invocation import ToolInvocation
    with project_run(tmp_path) as (_, _, task, _):
        workspace = tmp_path / 'workspace'
        # 不回放任何模型或工具，重建处理器只负责独立读取当前文件。
        handlers = {h.tool_type: h for h in build_workspace_tools(workspace,
            authorize=lambda ctx, write: change != 'revoked' and not write)}
        grant = SimpleNamespace(tool_context={}, permission_mode='read', authorization_ref='restore-fixture')
        context = execution_context(task, grant, attempt=1)
        calls = deepcopy(task['tool_calls'])
        read = next(c for c in calls if c['name'] == 'workspace_read_lines')
        receipt = read['data']['state_updates']
        if change == 'changed':
            (workspace / 'app.py').write_text('外部变更')
        elif change == 'foreign_target':
            read['arguments']['path'] = str(workspace / 'another.py')
            read['call_fingerprint'] = hashlib.sha256(_call_signature(
                ToolInvocation(read['name'], read['arguments'], id=read['id'])).encode()).hexdigest()
        elif change == 'tampered_source':
            receipt['source_path'] = str(workspace / 'another.py')
            receipt['completion_targets'].append(receipt['source_path'])
        elif change == 'tampered_path':
            # source_path仍匹配原参数，完成处理器必须发现path与source_path矛盾。
            receipt['path'] = 'config.json'
        elif change == 'unbound_source':
            receipt['completion_targets'].remove(receipt['source_path'])
        elif change == 'partial':
            receipt['read_complete'] = False
        ledger = LocalAnswerLedger(); ledger.configure(context)
        ledger.restore_reads(calls, handlers, context)
        assert len(ledger.sources) == (3 if change == 'none' else 0 if change == 'revoked' else 2)
        assert all(row['valid'] and row['text'] == FILES[Path(row['path']).name]
                   for row in ledger.sources.values())
        ledger.restore_reads(calls, handlers, context)
        assert len(ledger.sources) == (3 if change == 'none' else 0 if change == 'revoked' else 2)
        if change == 'changed':
            # 外部写入属于反例，不让project_run把它误记成Agent修改。
            (workspace / 'app.py').write_text(FILES['app.py'])
