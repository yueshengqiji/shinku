"""Version-four prose context; diagnostic codes never prove their cause.

The complete code remains available to the neutral checker and scoped audit.
User conditions, material gaps, measurements and execution truth are unchanged.
"""
from __future__ import annotations


RULES = (
    "宿主启用表达上下文版本4，沿用版本3的条件和回执核对。"
    "stop_observation只解释宿主记录的诊断标记：known=false时具体含义和原因没有确认，"
    "不能从代码猜测网络故障、已恢复、已完成或应当重试。完整代码保存在reason_audit，"
    "正文须保留原因未确认的边界，可以自由措辞，不必抄写诊断代码。"
    "同一结果的状态、缺口和来源范围可以在一个自然段中共同归属多个fact_ids；"
    "同一限制说明一次即可，仍须完整覆盖每个必呈现事实的含义。"
    "交接完整不证明已经理解资料、取得未读取来源或完成任务；"
    "不同条件、数值、来源、未完成项和不确定性不能当成重复说明省略。"
)


def apply_readable_context(result, blocks, *, query_receipts):
    """Normalize only presentation; retain the original bound facts for review."""
    reason = blocks.get('reason')
    if reason and reason.get('required', True) and 'reason_audit' not in blocks:
        blocks['reason_audit'] = {**reason, 'id': 'reason_audit', 'required': False}
        code = reason['text'].removeprefix('停止原因（宿主记录）：')
        known_query = (query_receipts and result.get('execution', {}).get('reason') == 'public_query_answer_unverified'
            and result.get('execution', {}).get('status') in {'partial', 'unverified'}
            and result.get('verification', {}).get('accepted') is not True)
        blocks['reason'] = {**reason, 'anchors': [], 'semantic_review_required': True,
            'text': ('当前问题的答案尚未通过证据核验。' if known_query else
                     '宿主记录了诊断标记，但现有记录未确认其具体含义和原因。'),
            'stop_observation': {'code': code, 'known': bool(known_query), 'source': 'executor',
                                 'only_proves_recorded_marker': True}}

    information = result.get('information')
    if not isinstance(information, dict) or 'information' not in blocks:
        return
    no_gaps = (not information.get('gaps') and not information.get('export_gaps')
        and not any(isinstance(information.get(k), dict) and information[k].get('gap_count')
                    for k in ('projection', 'snapshot')))
    if (query_receipts and information.get('query_scope') == 'current_operation_export' and no_gaps
            and information.get('reason') in {None, '', 'query_receipt_and_answer_scope'}):
        # This is the same bound query scope that versions one to three use.
        # Answer status and all missing conditions still have required blocks.
        blocks['information'] = {**blocks['information'], 'anchors': [], 'semantic_review_required': True,
            'text': '来源回执只覆盖本次查询实际读取的资料，未读取来源的完整性没有确认。'}
    elif (no_gaps and not information.get('reason')
            and information.get('status') == information.get('required_status') == 'complete'):
        blocks['information'] = {**blocks['information'], 'anchors': [], 'semantic_review_required': True,
            'text': '本次所需资料已完整交接；这只说明交接和实际读取的范围，不证明已经理解全量内容。'}
