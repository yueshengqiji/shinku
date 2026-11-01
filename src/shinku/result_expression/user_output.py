"""版本七的用户正文投影：过程审计留归档，事实、原报告和真实限制仍交付。

只由表达宿主选择；不改规范结果、材料权限、执行状态或送达回执。完整原件由
ExpressionArchive/read_details持有，不能从这里删减执行证据或声称任务成功。
"""
from collections.abc import Mapping

from shinku.agent_runtime.handoff import diagnostic_invocations


def apply_user_output(result, blocks, presentation):
    """按结构化来源分离诊断，不用用户文字关键词猜是否需要严格验收。"""
    if not result.get('presentation', {}).get('handoff_version') == 1:
        return
    diagnostic = diagnostic_invocations(result)
    facts = (presentation or {}).get('facts', result.get('facts', []))
    for index, fact in enumerate(facts):
        if fact.get('invocation_id') in diagnostic:
            blocks.pop('fact_' + str(index), None)
    # 这些字段的全部原文仍在canonical_result/presentation_projection；不再喂给
    # 润色模型，也不进入模型失败时的模板，避免可选审计再次被主动抄成正文。
    for key in list(blocks):
        if key.endswith('_audit'):
            blocks.pop(key)

    def replace(key, text, required=True):
        if key in blocks:
            blocks[key] = {'id': key, 'text': text, 'anchors': [],
                'required': required, 'source': 'host_user_output', 'semantic_review_required': True}

    information = result.get('information', {})
    if isinstance(information, Mapping):
        required = information.get('required_status') != 'complete'
        # 明确必需/未知的来源限制保留原块；只收起结构已知的投影审计。
        gaps = [g for g in information.get('gaps', [])
                if not isinstance(g, Mapping) or g.get('required') is not False]
        export_gaps = information.get('export_gaps', [])
        reason = information.get('reason')
        if not gaps and not export_gaps and reason in (None, '', 'legacy_source_completeness_unknown',
                                                      'controlled_canonical_fixture_not_live_execution'):
            losses = [gap for kind in ('snapshot', 'projection')
                      for gap in information.get(kind, {}).get('gaps', [])]
            substantive = any(not isinstance(gap, Mapping) or not str(gap.get('path', '')).startswith(
                ('/evidence/', '/budget/', '/verification/evidence/')) for gap in losses)
            text = ('当前任务所需资料仍有缺口，不能确认未读取的内容。' if required else
                    '部分原始资料未完整展开，未展示的内容不能据此补全。' if substantive else
                    '部分过程明细未在正文展开，原始记录已保留；这不代表额外执行或验证。')
            if reason == 'legacy_source_completeness_unknown':
                text += '上游资料是否完整仍未确认。'
            elif reason == 'controlled_canonical_fixture_not_live_execution':
                text += '这是隔离样本，未执行线上操作。'
            replace('information', text, required or substantive or bool(reason))
    if presentation and presentation.get('gaps'):
        replace('presentation_information', '部分结果细节未能补读；以下保留已取得的正文和明确限制，缺失内容不作推断。')
    else:
        replace('presentation_information', '结果正文已整理，过程明细保存在原记录中。', False)
    if result.get('execution', {}).get('reason') == 'partial_completion':
        replace('reason', '本次只完成了部分工作，未完成项仍须处理。', False)

    # 失败/未知副作用仍须说明；只将嵌套机器回执换成其真实调用状态。
    labels = {'error': '报错', 'failed': '失败', 'blocked': '被权限或宿主规则阻止',
              'unknown': '结果未知', 'canceled': '已取消', 'pending': '尚未完成'}
    for index, row in enumerate(result.get('evidence', [])):
        key = 'evidence_' + str(index)
        if key not in blocks:
            continue
        state = row.get('execution_status')
        if state != 'ok' and not blocks[key].get('receipt_observation'):
            tool = str(row.get('tool', '工具'))
            replace(key, f'{tool} 的调用{labels.get(state, "未确认成功")}；不能据此声称该操作完成。')
