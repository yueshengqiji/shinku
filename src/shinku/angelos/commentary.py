"""从已归档的公开助手正文提取进展；不读取推理、不生成总结、不签发执行事实。

调用者先核验任务归属。只选伴随原生工具调用的正文，最终答复由结果区交付；
稳定输出ID让刷新/重连保持顺序，历史尝试也保留为过程记录。
"""
import json
from shinku.agent_runtime.information import safe_text


def read_updates(host, task_id):
    reader = getattr(host.reasoning_view, 'outputs', None)
    if not callable(reader):
        return {'rows': [], 'omitted': 0}
    redact = getattr(host.profile.runtime, '_redact_secrets', None)
    rows = []
    for row in reader(task_id):
        diagnostics = row.get('diagnostics') or {}
        if (row.get('source') != 'provider_message_content'
                or not any(e.get('stage') == 'native_call' for e in diagnostics.get('events', []))):
            continue
        text = safe_text(str(row.get('text') or '')).strip()
        if not text:
            continue
        try:
            json.loads(text)
        except (ValueError, TypeError):
            pass
        else:
            continue  # 结构化协议对象留在诊断栏，不能当面向用户的进展。
        if callable(redact):
            text = redact(text, max_chars=12000)
        if rows and rows[-1]['text'] == text:
            continue
        rows.append({'id': row['id'], 'text': text[:2400], 'created': row.get('created'),
            'omitted_chars': max(0, len(text)-2400), 'source': 'assistant_commentary',
            'is_fact_evidence': False})
    return {'rows': rows[-64:], 'omitted': max(0, len(rows)-64)}
