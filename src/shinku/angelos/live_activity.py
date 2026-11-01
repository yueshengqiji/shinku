"""只读实时展示：会话绑定、工具原账本投影和SSE传输，不调模型、不推进任务。

工作台拥有会话选择，AgentService拥有执行事实，ReasoningView拥有脱敏推理副本。
这里没有执行/重试/恢复入口；连接断开只结束观察，不取消或重跑后台任务。
"""
import asyncio
import copy
import json
import time

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.information import bounded_projection, safe_source


def read_tools(host, cid, turn):
    if not turn.get('task_id'):
        return {'available': True, 'rows': [], 'total': 0, 'attempt': None}
    try:
        scope = AgentScope(host.requester, 'angelos:' + cid)
        with host.service._lock:
            task = host.service._get(scope, turn['task_id'])
            host.service._grant(scope, task, 'inspect')
            task = copy.deepcopy(task)
        attempt = task['execution_attempt']
        calls = [c for c in task.get('tool_calls', []) if all(
            c.get('execution_binding', {}).get(k) == v for k, v in
            (('task_id', task['task_id']), ('requester', scope.requester), ('session', scope.session), ('attempt', attempt)))]
        rows = []
        for c in calls[-64:]:
            executed = (c.get('data') or {}).get('state_updates', {}).get('executed', c.get('executed'))
            rows.append({'id': str(c.get('id', '')), 'name': str(c.get('name', '')),
                'status': 'not_executed' if executed is False else c.get('status', 'unknown'),
                'arguments': c.get('arguments', {}), 'result': {'status': c.get('status', 'unknown'),
                    'receipt': c.get('data', {}), 'material_reference': c.get('material_reference')},
                'error': c.get('error', ''), 'source': 'executor_receipt'})
        # inflight属于宿主当前task快照；只称“等待回执”，不把提交或审批当作已执行。
        for c in task.get('inflight', [])[:8]:
            if not any(r['id'] == c.get('id') for r in rows):
                rows.append({'id': str(c.get('id', '')), 'name': str(c.get('name', '')),
                    'status': 'pending_receipt', 'arguments': c.get('arguments', {}),
                    'result': {}, 'error': '', 'source': 'host_inflight'})
        projected, information = bounded_projection(safe_source(rows), text_limit=2400, list_limit=72, mapping_limit=40)
        redact = getattr(getattr(getattr(host, 'profile', None), 'runtime', None), '_redact_secrets', None)
        if callable(redact):
            # 字段通用脱敏之后再使用当前供应商的已配置密钥脱敏；不修改原调用或回执。
            def scrub(value):
                if isinstance(value, str): return redact(value, max_chars=12000)
                if isinstance(value, list): return [scrub(v) for v in value]
                if isinstance(value, dict): return {k: scrub(v) for k, v in value.items()}
                return value
            projected = scrub(projected)
        removed = 0
        while len(json.dumps(projected, ensure_ascii=False).encode()) > 262144:
            if len(projected) > 1:
                projected.pop(0); removed += 1
            else:
                projected[0].update(arguments={'omitted': 'display_budget'}, result={'omitted': 'display_budget'})
                information['status'] = 'truncated'
                break
        if removed:
            information['status'] = 'truncated'
        return {'available': True, 'rows': projected, 'total': len(calls),
                'omitted': max(0, len(calls)-64) + removed, 'attempt': attempt,
                'information': information, 'is_fact_evidence': False}
    except Exception:
        return {'available': False, 'rows': [], 'reason': 'tool_activity_unavailable'}


def turn_snapshot(workbench, cid, turn_id=None):
    """只接受存储中的会话/轮次，不允许浏览器指定task ID、scope或数据库路径。"""
    with workbench.lock:
        conversation = workbench.store.read('conversation', cid)
        turn = (next((t for t in conversation['turns'] if t['id'] == turn_id), None)
                if turn_id is not None else next(iter(conversation['turns'][-1:]), None))
        turn = copy.deepcopy(turn)
    if turn_id is not None and turn is None:
        raise ValueError('turn_not_found')
    if not turn:
        return {'conversationId': cid, 'turnId': None, 'active': False, 'reasoning': [],
                'tools': {'available': True, 'rows': [], 'total': 0}}
    host = workbench.host
    reader = getattr(host, 'read_reasoning', None)
    tools = read_tools(host, cid, turn)
    progress_reader = getattr(host, 'read_progress', None)
    progress = None
    if callable(progress_reader) and tools['available']:
        # 复用宿主现有只读进度投影；不依赖整页轮询，不把展示数据写回模型历史。
        try:
            progress = progress_reader(cid, turn)
        except Exception:
            progress = {'available': False, 'phase': 'unavailable', 'active': False,
                        'text': '暂时无法核对最新进度，正在等待下一次更新。'}
    return {'conversationId': cid, 'turnId': turn['id'],
            'active': turn['stage'] in ('preparing', 'submitting', 'executing'),
            'reasoning': reader(cid, turn) if callable(reader) and tools['available'] else [],
            'tools': tools, 'progress': progress}


async def events(workbench, cid, request, initial, *, duration=45, interval=.35):
    """有限连接定期读取快照，只在变化时发送；无状态重连从最新快照继续。"""
    started = heartbeat = time.monotonic(); previous = None; value = initial
    while time.monotonic() - started < duration:
        if await request.is_disconnected():
            return
        raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        if raw != previous:
            yield 'data: ' + raw + '\n\n'
            previous = raw; heartbeat = time.monotonic()
        if not value['active']:
            return
        if time.monotonic() - heartbeat >= 10:
            yield ': keepalive\n\n'; heartbeat = time.monotonic()
        await asyncio.sleep(interval)
        try:
            value = await asyncio.to_thread(turn_snapshot, workbench, cid)
        except Exception:
            yield 'event: unavailable\ndata: {"reason":"activity_unavailable"}\n\n'
            return
    yield 'event: rotate\ndata: {}\n\n'
