"""工具回执的模型视图；只在首次追加原生消息时生成，不修改宿主原件。

当前仅处理带执行会话、stdout/stderr 及对应可读块的命令结果。短输出去重；
长输出只有在材料会话确认原件可补读后才展示首尾，避免破坏缓存前缀或丢失证据。
权限、执行效果及完成判定仍使用原 envelope，不能使用此处的摘录代替验收。
"""
from collections.abc import Mapping
import copy
import json


PREVIEW_CHARS = 6000


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _budgets(stdout, stderr):
    """共用展示额度，短的一路让出空间；不以成功/失败关键词筛选输出。"""
    small = min(len(stdout), len(stderr), PREVIEW_CHARS // 2)
    return (PREVIEW_CHARS - small, small) if len(stdout) >= len(stderr) else (small, PREVIEW_CHARS - small)


def _excerpt(text, limit):
    if len(text) <= limit:
        return text, None
    head = limit // 2
    stop = len(text) - (limit - head)
    return text[:head] + f'\n[省略 {stop-head} 字符；需要时按引用补读，不能据摘录断言不存在其他结果]\n' + text[stop:], (head, stop)


def model_tool_content(envelope, call, material_session=None):
    """生成一次性展示副本；非命令工具及材料补读回执保持原协议逐字呈现。"""
    data = envelope.data
    updates = data.get('state_updates') if isinstance(data, Mapping) else None
    session = updates.get('execution_session') if isinstance(updates, Mapping) else None
    feedback = envelope.model_feedback
    if (envelope.executed is False or not isinstance(session, Mapping)
            or not session.get('session_id') or not session.get('command_sha256')
            or not all(isinstance(updates.get(k), str) for k in ('stdout', 'stderr'))):
        return feedback + '\n\n[host receipt]\n' + _json({'status': envelope.status, 'data': data})
    stdout, stderr = updates['stdout'], updates['stderr']
    block = f'stdout:\n{stdout}\nstderr:\n{stderr}'
    # 通用 envelope 适配器会 strip 反馈；空 stderr 的结尾换行可能已被去掉。
    # 只接受整个输出块的对应变体，并在可读副本中还原原始空白，不做子串猜测。
    matched = block if block in feedback else block.rstrip()
    # 不按任意子串替换，避免短 stdout 意外吞掉错误说明或状态字段。
    if feedback.count(matched) != 1:
        return feedback + '\n\n[host receipt]\n' + _json({'status': envelope.status, 'data': data})
    feedback = feedback.replace(matched, block, 1)
    view = copy.deepcopy(data)
    for key in ('stdout', 'stderr'):
        del view['state_updates'][key]
    lossless = feedback + '\n\n[host receipt]\n' + _json({'status': envelope.status, 'data': view})
    retained = None
    lookup = getattr(material_session, 'retained_tool_output', None)
    if len(stdout) + len(stderr) > PREVIEW_CHARS and callable(lookup):
        # 原件保存/重验失败不阻断真实回执，不隐瞒结果，也不能重放命令。
        try:
            retained = lookup(envelope, call.get('material_reference'))
        except Exception:
            retained = None
    projection = None
    if retained:
        projection = {'mode': 'head_tail', 'material_reference': retained['reference'],
            'reader': 'task_material_read', 'page_chars': retained['page_chars'], 'omitted_ranges': {}}
        shown = []
        for key, text, limit in zip(('stdout', 'stderr'), (stdout, stderr), _budgets(stdout, stderr)):
            excerpt, omitted = _excerpt(text, limit)
            shown.append(excerpt)
            if omitted:
                # offset 定位材料库的规范 JSON 文本（含转义），与现有分页协议一致；
                # 不另开路径读取入口，不把引用存在当成模型已读过省略部分。
                encoded = _json(text)
                marker = _json(key) + ':' + encoded
                if retained['text'].count(marker) != 1:
                    return lossless
                start = retained['text'].index(marker) + len(_json(key)) + 2
                left = len(_json(text[:omitted[0]])[1:-1])
                right = len(_json(text[:omitted[1]])[1:-1])
                projection['omitted_ranges'][key] = {'offset': start + left, 'length': right-left,
                    'omitted_output_chars': omitted[1]-omitted[0]}
        # 不能只确认“还能读一次”：剩余额度必须足以补回本次所有省略段。
        # 后续其他读取仍消费原累计预算，此处不预占、不重置，也不放宽授权。
        spans = projection['omitted_ranges'].values()
        if (sum(r['length'] for r in spans) > retained['remaining_chars']
                or sum((r['length'] + retained['page_chars'] - 1) // retained['page_chars']
                       for r in spans) > retained['remaining_reads']):
            return lossless
        feedback = feedback.replace(block, f'stdout:\n{shown[0]}\nstderr:\n{shown[1]}', 1)
    # 去掉的只有另一份输出副本；退出码、错误、游标、产物和执行状态不裁剪。
    receipt = {'status': envelope.status, 'data': view}
    if projection:
        receipt['output_preview'] = projection
    return feedback + '\n\n[host receipt]\n' + _json(receipt)
