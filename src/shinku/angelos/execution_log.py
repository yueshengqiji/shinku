"""用户侧完整执行日志：按任务/工具调用读取已采集的实际输出。

职责与上下游：
- 上游是宿主任务账本里的 tool_calls 与材料库（agent_task_materials）中的回执原件；
- 下游是 is_angelos.execution_log 的分页与搜索接口、前端右侧日志面板；
- 这里只读，不推进任务、不写回执、不调用模型，也不把日志当作完成判定证据。

与模型侧的区别（重要设计原因）：
- ``agent.tool_context`` 为了控制上下文体积，对长输出只给模型首尾摘录，并附带
  material_reference 让模型按需补读；那份摘录**不是**用户的完整日志。
- 用户在页面上要看到的是业务执行细节，因此本模块直接解析材料库中的原件，
  按行切分后分页返回，不受模型页大小限制，也不消耗模型的材料读取次数/字符预算
  （那份预算是"模型能读多少"的约束，用户查看不应挤占它，也不能借用户查看放宽它）。

归属与安全边界：
- 只接受 (会话 id, 轮次 id, 工具调用 id) 三元组；任务 id 由存储中的轮次解析，
  浏览器不能指定任意 task_id、scope 或数据库路径。
- 每次读取前用当前会话的 AgentScope 做一次 inspect 授权检查，跨会话/跨用户不可读。
- 原件的校验沿用材料库自身的版本/摘要校验；校验失败如实返回不可读，不回退成
  "大概是这样"的伪造内容。
- 输出在返回前先做字段级通用脱敏，再叠加当前供应商配置的密钥脱敏；显示、导出
  和模型输入都走同一套脱敏结果。脱敏不替代完整：其余排错细节保持原样。

缺失字段处理：早期调用可能没有耗时、时间戳或会话绑定，统一标为未知
（``None`` + ``unknown_fields``），不伪造顺序号、时间或执行状态。
"""
import copy
import json
import re

from shinku.agent_runtime import AgentScope

# 单页行数与字符上限：保证 20,000 行级别的输出可以分页读完，
# 同时避免一次响应把长时间构建日志整体塞进浏览器。
PAGE_LINES = 2000
PAGE_CHARS = 400000
# 搜索默认返回的命中条数上限；命中总数另行统计，便于页面显示"还有更多"。
SEARCH_LIMIT = 200


class LogUnavailable(Exception):
    """日志不可读：轮次/调用不存在、归属不符、原件已过期或校验失败。"""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _redactor(host):
    """取当前供应商配置的密钥脱敏函数；不可用时返回 None，由字段级脱敏兜底。"""
    return getattr(getattr(getattr(host, 'profile', None), 'runtime', None), '_redact_secrets', None)


def _scrub(value, redact):
    """递归脱敏；只改展示副本，不动账本原件与回执。"""
    if isinstance(value, str):
        return redact(value, max_chars=len(value) + 64) if callable(redact) else value
    if isinstance(value, list):
        return [_scrub(v, redact) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, redact) for k, v in value.items()}
    return value


def _secret_shapes(text):
    """字段级通用脱敏：识别常见"键=值"形式的凭据，避免没有供应商配置时整段泄露。

    只在能确定是凭据赋值的位置替换值，保留键名与结构，便于排错。
    """
    if not isinstance(text, str) or not text:
        return text
    # 键名左右都不加 \b：token=/key=/password: 出现在行首、行中或被空白包围时都要命中，
    # 而 \b 在"行首键名"等位置会漏判，正是最常见泄露位置。值取到空白或常见分隔符为止，
    # 保证只替换值、保留键名，其余排错信息原样保留。
    #
    # 顺序很关键：先处理 "Bearer <token>" 两段式，再做通用键值脱敏。
    # 先做键值脱敏时，"Authorization: Bearer xxx" 里的 Authorization 键先被替成占位符，
    # 原本的 Bearer 前缀随之消失，第二段就再也找不到锚点，剩下的值反而整段裸露。
    # 这就是旧实现在 'Authorization: Bearer "token"' 上仍泄露值的原因。
    text = re.sub(
        r'(?i)(\bbearer\s*[=:]?\s*)(?!\[redacted\])(["\']?)([^\s"\',;]{4,})(\2)',
        lambda m: m.group(1) + m.group(2) + '[redacted]' + m.group(4), text)
    # 值可被引号包裹（真实日志里常是 "token": "xxx" / token="xxx"），
    # 原写法遇冒号后的引号立即停止匹配，导致 JSON 与带引号形态整段保留。
    pattern = re.compile(
        r'(?i)(["\']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|secret|passwd'
        r'|password|authorization|bearer|token|key)["\']?\s*[=:]\s*)(["\']?)([^\s"\',;]{4,})(\2)')
    return pattern.sub(lambda m: m.group(1) + m.group(2) + '[redacted]' + m.group(4), text)


def _task_for(host, cid, turn):
    """按会话范围的 scope 读取轮次对应的任务快照（深拷贝）。"""
    if not turn.get('task_id'):
        return None, None
    scope = AgentScope(host.requester, 'angelos:' + cid)
    with host.service._lock:
        task = host.service._get(scope, turn['task_id'])
        host.service._grant(scope, task, 'inspect')
        task = copy.deepcopy(task)
    return scope, task


def _bound_calls(scope, task, *, include_history=False):
    """取归属本次任务的调用。

    默认只看当前 attempt；include_history=True 时放宽为历次 attempt。两者职责不同：
    - 默认：当前读取口径。重试后旧 attempt 的材料不再参与当前执行，直接暴露会让
      "这次能不能看"与"上一轮能不能看"混在一起，所以保留过滤。
    - 历史：用户侧面板显式带 history=1 时才走这条路。重试会把 execution_attempt +1，
      但不删 tool_calls，旧调用的回执仍在原处；不给这个入口，用户就只能看到
      call_not_found，而看不到自己刚刚那份输出。
    两种路径都严格限定 requester 与会话匹配，不跨用户/跨会话。
    """
    attempt = task.get('execution_attempt')
    calls = []
    for call in task.get('tool_calls', []):
        binding = call.get('execution_binding') or {}
        if binding.get('task_id') != task['task_id']:
            continue
        if binding.get('requester') != scope.requester or binding.get('session') != scope.session:
            continue
        if not include_history and binding.get('attempt') != attempt:
            continue
        calls.append(call)
    return calls


def _original(host, task, call):
    """从材料库取回执原件，返回 {'feedback','execution_status','state_updates'} 或 None。

    复用材料库的版本与摘要校验：只有与本次调用绑定、且内容未变的原件才返回。
    这里不消费模型的读取预算；用户查看与模型补读是两条独立通路。
    """
    reference = call.get('material_reference')
    if not isinstance(reference, dict) or not reference.get('reference'):
        return None
    # 材料库由 AgentService 持有（宿主只做投影）；只读解析，不写账本。
    store = getattr(getattr(host, 'service', None), 'materials', None)
    if store is None:
        return None
    # 优先级：用户侧只读展示入口 > 普通 resolve。（三者的差异见 materials.py 各自 docstring）
    # - resolve_for_read_only_view：允许任务已取消后回看（取消会推进 revision，普通路径会拒），
    #   但仍逐项核对授权、作用域、attempt、引用归属与正文摘要；
    # - resolve：旧实现/替身兼容（例如测试里的最小替身只实现了 resolve）。
    readers = [getattr(store, 'resolve_for_read_only_view', None), getattr(store, 'resolve', None)]
    for reader in readers:
        if not callable(reader):
            continue
        try:
            # resolve 接收的是引用 id 字符串；传整个 dict 会被当作不可哈希的键而失败。
            value = reader(reference['reference'], task=task, still_authorized=lambda: True)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _envelope_from_tail(feedback):
    """回执反馈尾部通常带 ``[host receipt]`` JSON；解析失败返回 None，不猜结构。"""
    if not isinstance(feedback, str) or '[host receipt]' not in feedback:
        return None
    raw = feedback.rsplit('[host receipt]', 1)[1].strip()
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _split_lines(text):
    """按行切分并保留行尾；空文本返回空列表，不制造一行空内容。"""
    if not isinstance(text, str) or text == '':
        return []
    lines = text.splitlines(keepends=True)
    return lines


def log_segments(host, cid, turn, call_id, *, include_history=False):
    """汇总一次工具调用可展示的全部实际输出，返回段落列表。

    段落结构：{stream, lines:[{n, text}], total_lines, complete}。
    - stream 为 stdout/stderr/result/error 之一；缺失的流不出现，而不是补空段。
    - complete=False 表示原件不可读或只拿到回执摘要，页面必须如实标注。
    """
    scope, task = _task_for(host, cid, turn)
    if task is None:
        raise LogUnavailable('task_not_recorded')
    calls = _bound_calls(scope, task)
    call = next((c for c in calls if str(c.get('id')) == str(call_id)), None)
    if call is None and include_history:
        # 只在调用方显式请求历史时，才回头历次 attempt 里查找；默认路径不变，
        # 不把旧 attempt 的调用当成当前调用。
        calls = _bound_calls(scope, task, include_history=True)
        call = next((c for c in calls if str(c.get('id')) == str(call_id)), None)
    if call is None:
        # 还在执行中的调用只有 inflight 记录，没有已采集输出。
        pending = next((c for c in task.get('inflight', []) if str(c.get('id')) == str(call_id)), None)
        if pending is None:
            raise LogUnavailable('call_not_found')
        return {'call': {'id': str(call_id), 'name': str(pending.get('name', '')),
                         'status': 'pending_receipt', 'started': None, 'finished': None},
                'segments': [], 'complete': False, 'unknown_fields': ['started', 'finished', 'exit_code']}
    redact = _redactor(host)
    original = _original(host, task, call)
    updates = {}
    feedback = ''
    if original:
        updates = dict(original.get('state_updates') or {})
        feedback = original.get('feedback') or ''
    else:
        # 原件不可读时才退回回执自带的短期输出；不把回执摘要冒充完整日志。
        updates = dict((call.get('data') or {}).get('state_updates') or {})
    # 完整性以材料引用自带的 complete 为准，而不是"原件对象能取到"：
    # 执行器把 20000 行截成 3276 行时，材料库仍会保留这份被截断的原件并标记 complete=False。
    # 只看 original 是否非空，就会把截断结果显示成"完整原件"，正好掩盖了最需要用户知道的缺失。
    reference = call.get('material_reference') if isinstance(call.get('material_reference'), dict) else {}
    reference_complete = reference.get('complete')
    truncated_note = ''
    if isinstance(updates.get('execution_session'), dict) and updates['execution_session'].get('output_truncated'):
        truncated_note = 'output_truncated'
    elif updates.get('output_truncated'):
        truncated_note = 'output_truncated'
    complete = bool(original) and reference_complete is not False and not truncated_note
    reasons = []
    if not original:
        reasons.append('original_unreadable')
    if reference_complete is False:
        reasons.append('reference_incomplete')
    if truncated_note:
        reasons.append(truncated_note)
    segments = []
    for stream in ('stdout', 'stderr'):
        text = updates.get(stream)
        if not isinstance(text, str) or text == '':
            continue
        text = _secret_shapes(_scrub(text, redact))
        lines = _split_lines(text)
        segments.append({'stream': stream, 'lines': [{'n': i + 1, 'text': line} for i, line in enumerate(lines)],
                         'total_lines': len(lines), 'complete': complete})
    error = call.get('error')
    if error:
        text = _secret_shapes(_scrub(str(error), redact))
        lines = _split_lines(text)
        segments.append({'stream': 'error', 'lines': [{'n': i + 1, 'text': line} for i, line in enumerate(lines)],
                         'total_lines': len(lines), 'complete': True})
    envelope = _envelope_from_tail(feedback) if feedback else None
    status_block = (envelope or {}).get('status') if isinstance(envelope, dict) else None
    data = (envelope or {}).get('data') if isinstance(envelope, dict) else None
    execution = ((data or {}).get('state_updates') or {}).get('execution_session') if isinstance(data, dict) else None
    execution = execution if isinstance(execution, dict) else {}
    # 回执摘要里没有执行会话时，退回调用记录自身的 state_updates；两处都没有才标未知，
    # 不能因为摘要缺字段就把真实的退出码说成"未知"。
    if 'returncode' not in execution:
        local = (call.get('data') or {}).get('state_updates') or {}
        local_session = local.get('execution_session') if isinstance(local, dict) else None
        if isinstance(local_session, dict) and 'returncode' in local_session:
            execution = {**local_session, **{k: v for k, v in execution.items() if v is not None}}
        elif isinstance(local, dict) and 'returncode' in local:
            execution = {**execution, 'returncode': local.get('returncode')}
    unknown = [k for k, v in (('exit_code', execution.get('returncode')), ('elapsed', execution.get('elapsed')),
                              ('started', execution.get('started_at')), ('finished', execution.get('finished_at')))
               if v is None]
    call_view = {
        'id': str(call.get('id', '')), 'name': str(call.get('name', '')),
        # 执行事实只取回执状态，不按输出里的"失败/error"字词改判。
        'status': 'not_executed' if (call.get('data') or {}).get('state_updates', {}).get('executed') is False
                  else call.get('status', 'unknown'),
        'started': execution.get('started_at'), 'finished': execution.get('finished_at'),
        'exit_code': execution.get('returncode'), 'elapsed': execution.get('elapsed'),
    }
    return {'call': call_view, 'segments': segments, 'complete': complete,
            'incomplete_reasons': reasons, 'unknown_fields': unknown}


def read_log(host, cid, turn, call_id, *, stream='stdout', offset=0, limit=PAGE_LINES, max_chars=PAGE_CHARS,
             include_history=False):
    """分页读取某次调用的单个段落（stream）。

    返回 {call, stream, offset, lines, total_lines, next_offset, has_more, complete, streams, ...}。
    重复拉取同一 offset 得到同一片段，页面据此去重；不会因重连而重复追加。
    该流没有内容（例如成功命令没有 stderr）时返回空页而不是报错，但完整性与状态字段
    仍按本次调用如实返回，空页不等于日志不完整。
    """
    data = log_segments(host, cid, turn, call_id, include_history=include_history)
    names = [s['stream'] for s in data['segments']]
    segment = next((s for s in data['segments'] if s['stream'] == stream), None)
    if segment is None:
        # 空页也必须带上完整性与原因：旧写法在这里只返回 complete，把
        # incomplete_reasons 丢了，页面就只能说“不完整”而说不出为什么。
        return {'call': data['call'], 'complete': data['complete'], 'unknown_fields': data['unknown_fields'],
                'incomplete_reasons': data.get('incomplete_reasons', []),
                'stream': stream, 'offset': 0, 'lines': [], 'total_lines': 0,
                'next_offset': None, 'has_more': False, 'streams': names}
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or PAGE_LINES), PAGE_LINES * 4))
    lines = segment['lines'][offset:offset + limit]
    used = 0
    kept = []
    for row in lines:
        used += len(row['text'])
        if kept and used > max_chars:
            break
        kept.append(row)
    next_offset = offset + len(kept)
    return {'call': data['call'], 'complete': data['complete'], 'unknown_fields': data['unknown_fields'],
            'incomplete_reasons': data.get('incomplete_reasons', []),
            'stream': stream, 'offset': offset, 'lines': kept, 'total_lines': segment['total_lines'],
            'next_offset': next_offset if next_offset < segment['total_lines'] else None,
            'has_more': next_offset < segment['total_lines'], 'streams': names}


def search_log(host, cid, turn, call_id, query, *, limit=SEARCH_LIMIT, max_lines=SEARCH_LIMIT,
               include_history=False):
    """在本次调用的全部已采集行内按字面搜索；不经过模型、不改判定。"""
    data = log_segments(host, cid, turn, call_id, include_history=include_history)
    raw = (query or '').strip()
    if not raw:
        return {**data, 'query': '', 'hits': [], 'total_hits': 0, 'truncated': False}
    # 用户可能直接粘贴一段带凭据的文本搜索；回显的查询词必须是脱敏后的版本，
    # 否则页面与导出会经由"搜索了什么"泄露原值。
    needle = _secret_shapes(_scrub(raw, _redactor(host)))
    lowered = needle.lower()
    hits = []
    total = 0
    for segment in data['segments']:
        for row in segment['lines']:
            if lowered in row['text'].lower():
                total += 1
                if len(hits) < min(limit, max_lines):
                    hits.append({'stream': segment['stream'], 'n': row['n'], 'text': row['text']})
    return {**data, 'query': needle, 'hits': hits, 'total_hits': total, 'truncated': total > len(hits)}
