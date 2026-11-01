"""同一执行循环的阶段整理提醒；计数只安排提醒，不判失败或关闭工具。

宿主只知道调用/笔记发生过，不把新日志当作目标进展，也不替模型断定证据
足够。状态随原任务progress保存；原生请求仅在阶段变化时追加，保持旧前缀。
"""
from collections.abc import Mapping


def note_investigation(state, invocation, event):
    context = state.context if isinstance(state.context, Mapping) else {}
    if context.get('angelos_investigation_guidance') is not True:
        return
    if event.get('waiting') or invocation.name in {
        'update_work_plan', 'read_work_context', 'request_context_compaction', 'declare_task_contract'}:
        return
    # 同一回执被重复poll/回读不构成新的调查动作，也不应提前触发整理提醒。
    # 只用宿主进度账本的指纹去重；不读命令关键词、不判断业务结论。
    fingerprint = event.get('fingerprint')
    if isinstance(fingerprint, str) and fingerprint:
        seen = state.progress.setdefault('investigation_seen', [])
        if fingerprint in seen:
            return
        seen.append(fingerprint)
        del seen[:-128]
    task = context.get('agent_task_state', {})
    revision = task.get('work_state', {}).get('revision', 0)
    record = state.progress.setdefault('investigation_checkpoint', {'calls': 0, 'since_note': 0, 'plan_revision': revision})
    if revision != record['plan_revision']:
        record.update(since_note=0, plan_revision=revision)
    record['calls'] += 1
    record['since_note'] += 1
    if record['since_note'] % 8:
        return
    state.progress['investigation_advisory'] = (
        f"【调查阶段整理建议，第{record['calls']}次工具返回后】"
        "先对照用户原问题整理已确认的结论、证据位置和仍影响结论的具体缺口。"
        "命令成功或新增日志不等于目标已经推进。核对上一建议是否引入当前故障，"
        "本次对照是否确实改变关键条件；新证据已排除的解释不再反复提出。已有足够依据就交付答案；"
        "还需调查时，用update_work_plan记录发现、已排除路线和下一项能消除缺口的核实，"
        "不要为了取回已读正文重跑命令，也不要继续扩展无关目录。"
        "此提醒不是调用次数上限或验收条件；确有未解问题可以继续使用获准工具，"
        "简单任务无需为了提醒补计划，未核实部分应如实说明。")


def investigation_advisory(state):
    context = state.context if isinstance(state.context, Mapping) else {}
    return state.progress.get('investigation_advisory', '') if context.get('angelos_investigation_guidance') is True else ''
