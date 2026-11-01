"""版本六的执行报告表达投影；原稿与执行事实仍由原归档、执行器持有。

普通执行叙述允许人格层重写，精确资料仍用锚点保护；整篇含义使用原中性检查，
而非把执行者称呼、标题和逐字原稿当成用户要求。这里只改变投影，不写任务账本。
"""
from __future__ import annotations

import re


RULES = (
    '宿主启用表达版本6。execution_report_projection标记执行方的普通报告，'
    '它是供你理解的执行叙述，不是真红要逐字转发的用户引文，也不是执行事实凭据。'
    '请通读完整报告，按当前完整人格和原始用户需求重新组织正文；'
    '执行方的自称、对用户的称呼、开场白、标题与署名不是必须保留的事实。'
    '不要以原执行者身份说话，也不要只原样插入执行报告再追加一段说明。'
    '此报告的普通叙述按本条改写，优先于普通长原文块逐字插入的规则。'
    '报告的结论、依据、条件、否定、未覆盖范围与用户需要的具体细节仍须完整保留；'
    '只能更换叙述方式，不能把未验证的自述变成事实，不能提升宿主验收或送达状态。'
    'anchors标明的代码、引文、地址、数值、单位与校验值继续精确保留。'
    '机器编号等审计信息若已放在可选上下文，不必对用户宣读；对应实际缺口与限制仍须说明。'
)

CHECK_RULES = (
    'execution_report_projection表示执行方叙述，不是当前说话者的人格。'
    '核对原报告完整含义与候选正文：结论、依据、条件、否定、来源及未覆盖范围不能丢失，'
    '用户明确要求的细节不能被一条概括替代；未核实的执行者自述不能升级为事实。'
    '执行方的自称、称呼、标题和署名可以按当前说话身份改写或省略，不将其当作事实遗漏。'
    'whole_report要求结合完整fact_blocks判断整篇报告的语义，不只检查某一句；'
    'answer_fragment选取候选正文中实际承载报告结论及限制的片段，不抄原报告充当候选。'
    '这不放宽其他宿主条件逐条核对、精确保护值、权限、任务状态和送达回执检查。'
)

REPORT_REVIEW = '完整保留执行报告的结论、依据、条件、否定、未覆盖范围与用户要求的细节；执行者称呼、标题和版式允许改写。'

# 只识别语法上可定位的精确资料；普通叙述中的因果与否定由整篇中性核对负责。
_EXACT = re.compile(
    r'```[^\n]*\n[\s\S]*?(?:```|\Z)|~~~[^\n]*\n[\s\S]*?(?:~~~|\Z)'
    r'|`[^`\n]+`|「[^」]*」|『[^』]*』|“[^”]*”|"[^"\n]+"'
    r'|https?://[^\s<>"\u3000，。；！？、）」》]+'
    r'|/[^\n<>:"，。；！？]*?\.[A-Za-z0-9]{1,10}(?=$|[\s，。；！？、）」])'
    r'|(?:/[A-Za-z0-9_.\-\u4e00-\u9fff]+)+(?:\.[A-Za-z0-9]+)?'
    r'|[A-Za-z]:[\\/][^\s<>"，。；！？]+'
    r'|(?<![\w])[A-Za-z0-9_.-]+\.(?:txt|md|json|py|js|html|csv|pdf|zip|log)(?![\w])'
    r'|(?<![A-Za-z0-9])[a-fA-F0-9]{32,128}(?![A-Za-z0-9])'
    r'|(?P<number>(?<![A-Za-z0-9_.+-])[+-]?(?:[0-9]+(?:\.[0-9]+)?)(?:[eE][+-]?[0-9]+)?'
    r'(?:[ \t]*(?:KiB|MiB|GiB|KB|MB|GB|ms|seconds|tokens|字节|字符|毫秒|秒|分钟|小时|万元|元|次|项|个|条|页|天|%|℃|万|亿))?'
    r'(?![A-Za-z0-9_.]))'
)


def report_anchors(text):
    """从完整正文取精确片段，不截短原文；相交片段由最先匹配的完整资料覆盖。"""
    return [{'text': match.group(), 'kind': 'number' if match.lastgroup == 'number' else 'literal'}
            for match in _EXACT.finditer(text)]


def apply_persona_body(result, blocks):
    """仅处理宿主识别的执行报告；不能由模型将别的引文降为普通叙述。"""
    body = blocks.get('execution_body')
    from shinku.agent.request_intent import assertion_text, execution_request_text
    verbatim = any(re.search(r'逐字|原样|一字不改', execution_request_text(str(row.get('text', ''))))
                   or re.search(r'不要改写|不得改写', assertion_text(str(row.get('text', ''))))
                   for row in result.get('original_requests', []) if isinstance(row, dict))
    if body is not None and body.get('source') == 'execution_model_report' and not verbatim:
        blocks['execution_body'] = {**body, 'anchors': report_anchors(body['text']),
            'semantic_review_required': True, 'execution_report_projection': True}

    # 仅把显式可选、且字段形状已知的资料缺口编号移入原有审计投影。
    # 必需/未知缺口沿用旧精确规则；绝不从required=false推断任务已经通过验收。
    info = result.get('information', {})
    gaps = info.get('gaps', []) if isinstance(info, dict) else []
    allowed = {'source_id', 'reference', 'reason', 'remaining_chars', 'required'}
    if ('information' in blocks and gaps and isinstance(gaps, list)
            and all(isinstance(g, dict) and g.get('required') is False and set(g) <= allowed
                    and g.get('reason') == 'material_read_pending'
                    and type(g.get('remaining_chars')) is int and g['remaining_chars'] >= 0 for g in gaps)
            and not info.get('export_gaps') and info.get('required_status') == 'complete'
            and all(not isinstance(info.get(key), dict)
                    or not any(info[key].get(field) for field in ('gap_count', 'gaps', 'omitted_gap_records'))
                    for key in ('projection', 'snapshot'))
            and info.get('reason') in {None, '', 'legacy_source_completeness_unknown'}):
        text = ('当前任务所需资料已提供；已交接记录中另有可选来源未完整读取。'
                '这些记录只说明交接与实际读取范围，不证明已经理解全量资料或完成任务。')
        if info.get('reason') == 'legacy_source_completeness_unknown':
            text += '上游资料是否完整仍未确认。'
        blocks['information'] = {**blocks['information'], 'text': text, 'anchors': [],
            'semantic_review_required': True}
