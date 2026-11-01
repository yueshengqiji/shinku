"""个性化建议的真实只读链路；固定模型响应只验证接线和边界，不证明模型主动性。"""
from dataclasses import replace
import json

import pytest

from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.tools.workspace import build_workspace_tools
from test_agent_conversation_loop import native_sdk
from test_agent_runtime_boundary import settled


@pytest.mark.parametrize('boundary', ['allowed', 'revoked', 'outside'])
def test_personalized_advice_reads_actual_configuration_in_same_loop(tmp_path, boundary):
    """原生SDK→宿主逐次授权→真实文件读取→直接答复，不引入准备器或语义验收。"""
    root = tmp_path / 'workspace'; root.mkdir()
    config = root / 'settings.ini'; raw = 'engine=synthetic-composer\nconfigured_layout=pinyin\n'
    config.write_text(raw)
    (tmp_path / 'outside.ini').write_text('OUTSIDE_PRIVATE_CONTENT')
    request = '针对这个工作区，优化现有输入组件和重做一个，你推荐哪个？另外怎么处理“星桥层”？'
    answer = ('配置显示使用 synthetic-composer 并配置了拼音布局，不能据此确认正在运行。'
              '建议先优化现有配置。“星桥层”的具体含义还需要你说明。') if boundary == 'allowed' else (
              '当前读取受权限或目录范围限制，尚不能确认实际配置。可以先说明选择依据。')
    service, write_tool, _ = demo_service(tmp_path / 'state')
    allowed = [True]
    for handler in build_workspace_tools(root, authorize=lambda ctx, write: allowed[0] and not write):
        service.executor.tool_host.registry.register(handler)
    original = service.authorize
    service.authorize = lambda *args: replace(original(*args), allowed_tools=frozenset({'workspace_read_lines'}),
        policy=replace(original(*args).policy, risk_by_tool={'workspace_read_lines': 'low'}))

    def respond(payload, count):
        if count == 1:
            system = '\n'.join(m['content'] for m in payload['messages'] if m['role'] == 'system')
            # 检查正式模型入口收到新指导；响应本身是合成桩，不能冒充行为质量复验。
            assert '个性化建议' in system and '可自行查明的事实不要转问用户' in system
            assert any(request in m.get('content', '') for m in payload['messages'])
            if boundary == 'revoked':
                # 能力投影之后撤销权限，实际适配器仍须重新检查，不能靠提示词放行。
                allowed[0] = False
            path = '../outside.ini' if boundary == 'outside' else 'settings.ini'
            return {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'inspect-config', 'type': 'function', 'function': {
                    'name': 'workspace_read_lines', 'arguments': json.dumps({'path': path})}}]}, 'tool_calls'
        receipts = [m for m in payload['messages'] if m['role'] == 'tool']
        assert count == 2 and len(receipts) == 1
        assert receipts[0]['tool_call_id'] == 'inspect-config'
        content = receipts[0]['content']
        if boundary == 'allowed':
            assert 'synthetic-composer' in content and 'pinyin' in content
        else:
            assert 'synthetic-composer' not in content and 'OUTSIDE_PRIVATE_CONTENT' not in content
            # 权限异常由真实适配器脱敏成access_failed，不向模型暴露底层权限信息。
            assert ('workspace_access_failed' if boundary == 'revoked' else 'workspace_path_outside_scope') in content
        return {'role': 'assistant', 'content': json.dumps({
            'kind': 'final', 'text': answer, 'task_outcome': 'completed'})}, 'stop'

    try:
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            host = ServiceHost(service, requester='owner')
            host.prepare = lambda *args: pytest.fail('must not call a separate preparer')
            port = host.port('one'); handle = port.submit(AgentTurnRequest('advice', request))
            view = settled(port, handle.task_id); result = port.result(handle.task_id)
            assert view['status'] == 'completed', result
            assert result['execution_report']['text'] == answer
            assert len(calls) == result['budget']['requests'] == 2
            assert result['budget']['tools'] == 1 and write_tool.calls == 0
            assert result['contract'] is None
            evidence = result['verification']['evidence'][0]
            assert evidence['execution_status'] == ('ok' if boundary == 'allowed' else 'error')
            assert evidence['receipt_valid'] is (True if boundary == 'allowed' else None)
            assert result['verification']['layers']['goal_completion'] == 'not_verified'
            assert config.read_text() == raw
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()
