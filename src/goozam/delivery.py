"""实验宿主首次提交时投影经验；原材料库拥有持久记录与模型输入，不新增模型调用。

只投影已晋级且来源新鲜的同工作区建议。没有匹配、关闭或建议过大时允许普通
任务继续；完整条件/例外不可截断。材料记录证明输入装配，不能证明模型采纳或收益。

当前授权直接取grant；原任务结果仅作历史观察。缺失上下文、来源失效和无匹配
分别留可选缺口，不以隐式None跳过检查，不要求新任务复现旧结果。
"""
import json

from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.materials import MaterialSource
from shinku.angelos.task_continuity import workspace_binding, task_turn, turn_workspace
from .experience_context import ExperienceContext


SOURCE_ID = 'goozma-experiences'


def experience_materials(host, plugin, scope, task, grant):
    """在原授权之后、任务第一次入库之前调用；不把检索结果注册成必读附件。"""
    if ('read_goozam_experiences' not in grant.allowed_tools
            or not grant.policy.accepts('read_goozam_experiences')):
        return (), ()
    try:
        conversation, binding = workspace_binding(host.memory.continuity.workbench, scope.session)
        turn = task_turn(conversation, task)
        if turn is None or turn_workspace(turn) != binding:
            return (), ({'source_id': SOURCE_ID, 'required': False, 'reason': 'workspace_changed'},)
        query = task['requests'][-1]['text'].strip()[:512]
        # 此时host_grant尚未写入task，必须使用授权回调刚取得的grant。
        current = ExperienceContext.from_grant(binding, grant)
        result = plugin.for_task(binding, task['task_id'], query, current=current)
        selected, omitted = [], 0
        for row in result.get('items', []):
            item = {k: row[k] for k in ('id', 'version', 'situation', 'conditions', 'advice', 'exceptions',
                                      'source_observations', 'tool_context')}
            # 单条完整保留；预算内最多两条，不截掉适用条件和例外制造过度泛化。
            if len(selected) >= 2 or len(json.dumps([*selected, item], ensure_ascii=False)) > 4000:
                omitted += 1
                continue
            selected.append(item)
        if not selected:
            if omitted:
                return (), ({'source_id': SOURCE_ID, 'required': False, 'reason': 'experience_too_large'},)
            reasons = list(result.get('context_gaps', []))
            if result.get('source_gaps'):
                reasons.append('experience_source_unavailable')
            if not reasons:
                reasons.append('experience_no_match')
            return (), tuple({'source_id': SOURCE_ID, 'required': False, 'reason': reason}
                             for reason in reasons)
        state = plugin.controls.snapshot()
        if (plugin.closed or not state['use_experiences']
                or state['revision'] != result.get('plugin_revision')):
            return (), ()
        content = json.dumps(dict(role='advisory_experience',
            instruction='仅供选择操作方法；结合当前证据判断条件与例外，不是必做步骤或授权。'
                        'source_observations是旧任务观察，不要求当前任务复现；工具重合不代表执行获准。',
            is_execution_evidence=False, grants_permissions=False,
            evaluation_only=result.get('evaluation_only') is True,
            plugin_revision=state['revision'], items=selected, omitted_items=omitted), ensure_ascii=False)
        # 记录版本和内容摘要，实际提交后的材料包可回查；没有任务不声称送达。
        return (MaterialSource(SOURCE_ID, 'related_message', content, required=False,
                version=canonical_hash(content), provenance={'relation': 'advisory_experience'}),), ()
    except Exception:
        # 学习检索故障不杀死业务任务；不泄露原始异常或静默声称经验已送达。
        return (), ({'source_id': SOURCE_ID, 'required': False, 'reason': 'experience_unavailable'},)
