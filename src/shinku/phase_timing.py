"""宿主阶段观测：只记录计时与关联，不拥有执行状态、预算或完成判定。

QQ 追踪和 Angelos 用量库分别持有记录；本模块只管理当前调用的上下文。
阶段用单调时钟计时。父阶段包含子阶段，汇总不把各阶段相加冒充端到端耗时。
跨线程须显式携带 observation_context；不同任务重新绑定，不能继承另一任务身份。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
import time
import uuid

PHASES = {
    'control': '控制指令', 'preparation': '上下文准备', 'memory': '记忆准备',
    'approval': '审批处理', 'routing': '路由决策', 'execution': '执行切片',
    'planning': '执行规划', 'tool': '工具调用', 'verification': '目标核验',
    'answer_review': '答案审查', 'candidate_review': '候选资料审查',
    'compaction': '上下文压缩', 'expression': '结果表达',
    'expression_check': '表达核验', 'delivery': '交付处理',
    'delegation_wait': '委托等待与交付', 'unclassified': '未分类',
}
_observer = ContextVar('phase_observer', default=None)
_phase = ContextVar('observed_phase', default=('unclassified', ''))


def current_phase():
    return _phase.get()


def observation_context():
    """只复制观测绑定，不复制预算、权限或其他业务 ContextVar。"""
    return _observer.get(), _phase.get()


@contextmanager
def inherit_observation(snapshot):
    observer, phase = snapshot
    a, b = _observer.set(observer), _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(b)
        _observer.reset(a)


@contextmanager
def bind_observer(binding, sink, *, clock=time.monotonic):
    with inherit_observation(((binding, sink, clock) if binding and callable(sink) else None,
                              ('unclassified', ''))):
        yield


def _emit(observer, event):
    try:
        observer[1](event)
    except Exception:
        pass  # 接收器自己标记不完整；诊断故障不改变原调用、不触发重试。


@contextmanager
def phase_scope(name):
    name = name if name in PHASES else 'unclassified'
    observer = _observer.get()
    ident = uuid.uuid4().hex if observer else ''
    parent = _phase.get()[1]
    token = _phase.set((name, ident))
    started = observer[2]() if observer else 0
    event = {'binding': observer[0] if observer else '', 'span_id': ident,
             'parent_id': parent, 'phase': name, 'status': 'started', 'duration_ms': None}
    if observer:
        _emit(observer, event)
    status = 'completed'
    try:
        yield
    except BaseException:
        status = 'failed'
        raise
    finally:
        if observer:
            duration = (observer[2]() - started) * 1000
            _emit(observer, {**event, 'status': status,
                'duration_ms': round(duration, 3) if math.isfinite(duration) and duration >= 0 else None})
        _phase.reset(token)


def timed(name):
    """同步调用的薄包装；不解析参数/返回正文，正常返回也不表示业务成功。"""
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            with phase_scope(name):
                return fn(*args, **kwargs)
        return call
    return decorate


def clean_span(row):
    """宿主接收器入口白名单；不接收工具参数、异常正文或任意阶段标签。"""
    if (row.get('phase') not in PHASES or row.get('status') not in {'started', 'completed', 'failed'}
            or any(not isinstance(row.get(k), str) or not row[k] or len(row[k]) > 256
                   for k in ('binding', 'span_id'))):
        raise ValueError('invalid_phase_event')
    value = row.get('duration_ms')
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError('invalid_phase_duration')
    parent = row.get('parent_id', '')
    if not isinstance(parent, str) or len(parent) > 256:
        raise ValueError('invalid_phase_parent')
    return {k: row.get(k) for k in ('binding', 'span_id', 'parent_id', 'phase', 'status', 'duration_ms')}


def summarize_phases(spans, calls=(), *, truncated=False):
    """只汇总已记录样本。每阶段 p50/p95 为单次调用分位数，不是任务分位数。

    请求用量沿用实际 SDK 回执口径；父子阶段的模型请求只归给最内层阶段，避免重计。
    started 缺少终态时只能称“未结束或中断”，不从当前时间补造耗时。
    """
    from shinku.agent_runtime.evaluation import summarize_calls
    latest = {}
    for row in spans:
        key = (row.get('binding'), row.get('span_id'))
        if key not in latest or latest[key].get('status') == 'started':
            latest[key] = row
    requests = {}
    for row in calls:
        key = (row.get('binding'), row.get('request_id'))
        if key not in requests or requests[key].get('status') in {'started', 'unknown'}:
            requests[key] = row
    names = {r.get('phase', 'unclassified') for r in (*latest.values(), *requests.values())}
    result = []
    for name in PHASES:
        if name not in names:
            continue
        items = [r for r in latest.values() if r.get('phase') == name]
        durations = sorted(r['duration_ms'] for r in items
            if r.get('status') in {'completed', 'failed'} and r.get('duration_ms') is not None)
        phase_calls = [dict(r, status='unknown' if r['status'] == 'started' else r['status'])
                       for r in requests.values() if r.get('phase', 'unclassified') == name]
        # 数据库身份是 binding + request_id；不同绑定的旧适配器可能复用短 ID。
        # 只在统计副本上组合 ID，原请求和账本记录保持不变。
        usage = summarize_calls([dict(r, request_id=str((r.get('binding'), r['request_id']))) for r in phase_calls])
        times = [r['elapsed_seconds'] for r in phase_calls if r.get('elapsed_seconds') is not None]
        result.append({'phase': name, 'label': PHASES[name], 'samples': len(items),
            'measured': len(durations), 'failed': sum(r.get('status') == 'failed' for r in items),
            'unsettled': sum(r.get('status') == 'started' for r in items),
            'totalMs': round(sum(durations), 3) if durations else None,
            'p50Ms': durations[math.ceil(len(durations) * .5)-1] if durations else None,
            'p95Ms': durations[math.ceil(len(durations) * .95)-1] if durations else None,
            'requests': len(phase_calls), 'requestFailed': sum(r['status'] in {'failed', 'timeout', 'cancelled'} for r in phase_calls),
            'requestUnsettled': sum(r['status'] == 'unknown' for r in phase_calls),
            'requestMs': round(sum(times) * 1000, 3) if times else None,
            'incompleteRequestTiming': len(phase_calls) - len(times),
            'knownInputTokens': usage['known_subtotals']['input_tokens'] if any(r['usage'].get('input_tokens') is not None for r in phase_calls) else None,
            'knownOutputTokens': usage['known_subtotals']['output_tokens'] if any(r['usage'].get('output_tokens') is not None for r in phase_calls) else None,
            'incompleteUsage': sum(any(r['usage'].get(k) is None for k in ('input_tokens', 'output_tokens')) for r in phase_calls)})
    return {'available': bool(result), 'phases': result, 'truncated': bool(truncated),
            'scope': 'recorded_samples', 'durationMode': 'inclusive_not_additive', 'is_fact_evidence': False}
