"""把同会话原始文字与历史任务展示记录分开，供后续问题理解背景。

只读取工作台已刷新且属于当前会话的副本，不扫描其他会话或读取私用记忆。
宿主状态字段来自 AgentPort 的展示结果，但进入新请求后仍只作为历史资料。
"""
import json


def conversation_source(turns, *, include_ids=False):
    messages, recent = [], []
    for index, turn in enumerate(turns):
        result, view = turn.get('result') or {}, turn.get('view') or {}
        messages.append({'user': turn['text'],
            'supplements': [s['text'] for s in turn.get('supplements', [])],
            'assistant': turn.get('reply') or result.get('execution_report', {}).get('text', '')})
        if include_ids:
            messages[-1]['turn_id'] = turn.get('id', '')
        if not turn.get('task_id'):
            continue
        if result.get('verification', {}).get('layers', {}).get('policy') == 'conversation_reply':
            continue  # 普通聊天留在完整messages中，不挤占实际任务的最近记录索引。
        # 不读取模型正文来猜任务成功；accepted 与 status 分别保留，避免把
        # “如实报告失败已获接受”误写成执行成功。历史副本也不声称当前仍有效。
        tools = sorted({str(e.get('tool', ''))[:64] + ':' + str(e.get('execution_status', 'unknown'))[:24]
                        for e in result.get('evidence', []) if isinstance(e, dict)})
        recent.append({'message_index': index, 'task_id': turn['task_id'], 'turn_id': turn.get('id', ''),
            'status': view.get('status', 'unknown'),
            'reason': str(view.get('reason', ''))[:160],
            'accepted': result.get('verification', {}).get('accepted'),
            'goal_verified': view.get('verification', {}).get('verified') is True,
            'delivery_policy': result.get('verification', {}).get('layers', {}).get('policy', ''),
            'tool_results': tools[:8], 'omitted_tool_results': max(0, len(tools) - 8)})
    return {'id': 'conversation-history', 'content': json.dumps({
        'schema_version': 1, 'messages': messages, 'recent_tasks': recent[-8:],
        'omitted_task_records': max(0, len(recent) - 8),
        'record_navigation': '如需调查原因，通过read_task_records读取本会话完整索引及任务/输入详情；摘要不能代替原回执。',
        # 这里只标明历史资料的可信度；纠错步骤由统一排错指导提供。
        'usage': '历史助手文字（包括实例、模型名、路径和推断）只作待核实线索；'
                 '历史任务记录不代表当前状态或本轮授权。',
    }, ensure_ascii=False)}
