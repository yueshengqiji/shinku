"""当前步骤的资料投影与可回读导航；不拥有授权、原稿或完成状态。

顺序：ConversationTurn校验绑定 -> 已保存计划的当前步/依赖引用 -> 可回读
工具预览投影。原材料、读取账本、模型原生消息都不写回；不完整/过期计划
只展示待复核导航，不据它省略资料。相关性不等于证据有效或任务已验收。
"""
from collections.abc import Mapping
import copy
from types import SimpleNamespace

from shinku.agent.work_state import digest
from shinku.agent.task_state import TaskContractError
from shinku.tools.workspace import tool_result
from .contracts import AgentBoundaryError


def focused_steps(work):
    plan = work.get('plan') or {}
    rows = {s['id']: s for s in plan.get('steps', [])}
    selected = set()
    def visit(key):
        if key in rows and key not in selected:
            selected.add(key)
            for dep in rows[key]['depends']:
                visit(dep)
    visit(plan.get('current_step'))
    return selected


def navigation(work):
    """只缩短模型的导航声明；完整修订由read_work_context随时回读。"""
    view = copy.deepcopy(work)
    if not view.get('plan'):
        return view
    selected = focused_steps(work)
    plan = view['plan']
    for row in plan['steps']:
        if row['id'] not in selected:
            row['goal_omitted_chars'] = max(0, len(row['goal']) - 160)
            row['goal'] = row['goal'][:160]
    for key in ('findings', 'hypotheses', 'decisions', 'ruled_out'):
        if key not in plan:
            continue
        values = plan[key]
        plan[key] = [s[:240] for s in values[:4]]
        view[key + '_omitted_items'] = max(0, len(values) - 4)
        view[key + '_omitted_chars'] = sum(map(len, values)) - sum(map(len, plan[key]))
    view['references'] = [r for r in view.get('references', []) if r['step_id'] in selected]
    view['full_plan_retrieval'] = {'tool': 'read_work_context', 'view': 'plan', 'revision': work['revision']}
    return view


def focus_payload(payload, transcript, work, task):
    """只投影有真实保留引用的可选工具预览；最近反馈和无回读入口的原文保留。"""
    data, history = copy.deepcopy(payload), copy.deepcopy(transcript)
    if work.get('status') != 'active' or not focused_steps(work):
        return data, history
    package = data.get('task_materials') or {}
    items = {r['reference']: r for r in package.get('items', [])}
    selected, missing = set(), []
    calls = {r.get('id'): r for r in task.get('tool_calls', [])}
    invalid = {g.get('reference') for g in data.get('information', {}).get('gaps', [])
        if g.get('reason') not in {'material_read_pending', 'material_media_input_pending'}}
    for ref in work.get('references', []):
        if ref['step_id'] not in focused_steps(work):
            continue
        if ref['kind'] == 'material':
            row = items.get(ref['id'])
            valid = bool(row and row.get('version') == ref.get('version')
                and row.get('content_sha256') == ref.get('content_sha256'))
            target = ref['id']
        else:
            call = calls.get(ref['id'])
            valid = bool(call and digest(call) == ref.get('receipt_sha256'))
            original_ref = (call or {}).get('material_reference', {})
            target = original_ref.get('reference')
            row = items.get(target)
            valid = bool(valid and row and row.get('provenance', {}).get('invocation_id') == ref['id']
                and row.get('version') == original_ref.get('version')
                and row.get('content_sha256') == original_ref.get('content_sha256'))
        # 信息包由原材料门实时重验；过期/撤销/上游不完整只回退预览选择，
        # 不清除缺口，不把“计划引用过它”当作新的可用凭据。
        valid = bool(valid and target not in invalid and row.get('retrieval') == 'available'
            and row.get('complete') is True)
        if valid:
            selected.add(target)
        else:
            missing.append(ref['id'])
    # 计划未选到可回读来源时不猜相关性；引用有缺口则保留全部预览供复核。
    if not selected or missing:
        return data, history
    recent = {r.get('material_reference', {}).get('reference') for r in task.get('tool_calls', [])[-2:]}
    selected |= recent
    omitted = []
    for row in package.get('items', []):
        if (row.get('kind') != 'executor_result' or row.get('required') is not False
                or row.get('retrieval') != 'available' or row['reference'] in selected):
            continue
        if isinstance(row.get('content'), str):
            row['content'] = ''
            row['omitted_chars'] = row['content_chars']
            omitted.append(row['reference'])
    for row in history[:-2]:
        if not isinstance(row, Mapping) or row.get('role') != 'tool':
            continue
        ref = row.get('material_reference') or {}
        if ref.get('reference') in omitted:
            original = str(row.get('content') or '')
            row['content'] = '[当前步骤外的工具预览；原文可按 material_reference 回读]'
            row['feedback_information'] = {'full_source_retained': True, 'omitted_chars': len(original),
                'reason': 'outside_current_step', 'prior_projection': row.get('feedback_information')}
    data['step_focus'] = {'mode': 'host_bound_plan_navigation', 'revision': work['revision'],
        'current_step': work['plan']['current_step'], 'selected_references': sorted(r for r in selected if r),
        'omitted_previews': omitted, 'full_sources_retained': True, 'read_records_unchanged': True,
        'is_execution_evidence': False, 'native_protocol_messages_unchanged': True}
    # 索引会重复携带旧用户背景，原请求/补充/原始历史仍在原包，按需回读。
    data.pop('answer_context', None)
    return data, history


class WorkContextTool:
    """同一任务内的导航读取；不执行外部操作，读取不重置停滞计数。"""
    tool_type = 'read_work_context'
    agent_permission_class = 'read'
    agent_progress_class = 'change'
    agent_execution_evidence = False
    agent_observation_receipt_fields = ('view', 'revision', 'workflow', 'guide_sha256', 'is_execution_evidence')

    def __init__(self, turn):
        self.turn = turn

    def build_prompt_instruction(self):
        return ('按需读取完整计划（view=plan，revision取当前导航修订号）或可调整工作参考（view=workflow，workflow为'
            'diagnosis/project_analysis/modify_validate）。简单任务无需调用；参考不授予权限或证明完成。')

    def tool_metadata(self):
        from .workflows import GUIDES
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': {'view': {'type': 'string', 'enum': ['plan', 'workflow']},
                'revision': {'type': 'integer', 'minimum': 1},
                'workflow': {'type': 'string', 'enum': list(GUIDES)}}, 'required': ['view']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not str(k).startswith('_tool_') and k != '_native_tool_call'}
        if (args.get('view') == 'plan' and set(args) <= {'view', 'revision'}
                and ('revision' not in args or type(args['revision']) is int and args['revision'] >= 1)):
            return args
        from .workflows import GUIDES
        if (set(args) == {'view', 'workflow'} and args['view'] == 'workflow'
                and isinstance(args['workflow'], str) and args['workflow'] in GUIDES):
            return args
        return None

    def execute(self, *, call, context):
        try:
            _, fresh = self.turn.current(context)
            if not fresh.policy.accepts(self.tool_type):
                raise AgentBoundaryError('work_context_not_authorized')
            args = self.normalize_call(call)
            if args is None:
                raise AgentBoundaryError('work_context_arguments_invalid')
            if args['view'] == 'workflow':
                from .workflows import workflow_reference
                return tool_result(self.tool_type, workflow_reference(args['workflow'], self.turn.handlers, fresh))
            work = self.turn.work_plan.snapshot(context)
            # 修订号同时约束快照与调用签名；计划更新后可读取新修订，既不开放
            # 无界重复读取，也不把内部导航冒充业务进展来重置原停滞保护。
            if 'revision' in args and args['revision'] != work['revision']:
                raise AgentBoundaryError('work_context_revision_conflict')
            return tool_result(self.tool_type, {'view': 'plan', **work})
        except (AgentBoundaryError, TaskContractError) as exc:
            return tool_result(self.tool_type, {'code': str(exc)}, error=str(exc))
