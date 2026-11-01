"""同一循环的按需计划工具：不运行独立规划模型，不拥有任务或执行权限。

ConversationTurn 提供进程内凭据；每次更新在 Service 原锁内重读任务和授权，
再使用 TaskStore 原事务原子保存修订和任务版本。恢复仅提供待核实导航，
不能据此重放副作用、重置预算或代替 CompletionGate 的实时验收。
"""
from collections.abc import Mapping
import copy
from types import SimpleNamespace

from shinku.agent.task_state import TaskContractError
from shinku.agent.work_state import digest, input_binding, validate_plan
from shinku.tools.workspace import tool_result
from .contracts import AgentBoundaryError


PLAN_GUIDANCE = (
    '复杂或持续任务可按需调用update_work_plan保存步骤、依赖、当前步、下一步、阻塞、发现与假设；'
    '简单聊天和单步任务无需计划。计划在当前正常循环中更新，不要先请求另一模型制定计划。'
    '多文件修改含相互依赖、诊断有待验证假设或预计需要恢复时，优先保存可检验步骤；'
    '只需一次读取即可回答时直接处理。是否计划由实际复杂度决定，不按任务名字强制套模板。'
    '只在阶段完成、真实失败、依赖变化或用户补充时调整，不要每次工具后重写计划。'
    'expected_revision取当前work_state.revision，无计划时为0。步骤requirements只引用原契约id；'
    '契约未声明时可为空，声明后更新计划应覆盖全部条件。引用只用宿主工具调用id或材料reference。'
    'findings是待核实的模型判断，hypotheses是假设；done是进度声明，绝非验证成功。'
    '恢复或输入变化后先核对当前证据；不要按计划重放已成功的写入。计划不能授予权限、修改目标或降低验收。'
    '已有有效读取用references定位，别为填计划重读；失败先记录具体错误，再调整受影响步骤。'
    '当前上下文可能只展示当前步骤与有界导航，需要完整计划时用read_work_context(view=plan,revision=当前修订号)回读后再调整。'
    '遇到同名软件/对象身份冲突时，可用可选objects保存name、candidates、conflicts；'
    '候选与冲突不是已确认事实，结合步骤references核对来源，不把同名直接合并。'
    '可选decisions记录当前选定路线及理由，ruled_out记录已排除路线及依据；'
    '与findings/hypotheses、blocker/next_action一起构成工作笔记。原目标仍取用户要求，不能用笔记改写。'
    '笔记中的已读、已改、已测应指向步骤references；有回执不等于整个目标已验收。'
    '准备主动压缩时，已有复杂计划可先更新必要工作笔记；简单任务不为压缩强制建计划。'
)


class WorkPlanTool:
    tool_type = 'update_work_plan'
    # 只改内部导航；change 且没有 changed 回执，因此更新不能重置无进展计数。
    agent_permission_class = 'read'
    agent_progress_class = 'change'
    agent_execution_evidence = False
    agent_observation_receipt_fields = ('revision', 'status', 'sha256', 'is_execution_evidence', 'grants_permissions')

    def __init__(self, turn):
        self.turn = turn

    def build_prompt_instruction(self):
        if self.turn.reply_delivery:
            return PLAN_GUIDANCE.replace('步骤requirements只引用原契约id；契约未声明时可为空，声明后更新计划应覆盖全部条件。',
                '步骤requirements可留空；计划是可替换的辅助导航，不必覆盖旧模型契约。')
        return PLAN_GUIDANCE

    def tool_metadata(self):
        reference = {'type': 'object', 'additionalProperties': False, 'properties': {
            'kind': {'type': 'string', 'enum': ['tool_call', 'material']}, 'id': {'type': 'string'}},
            'required': ['kind', 'id']}
        step = {'type': 'object', 'additionalProperties': False, 'properties': {
            'id': {'type': 'string'}, 'goal': {'type': 'string'},
            'depends': {'type': 'array', 'items': {'type': 'string'}},
            'requirements': {'type': 'array', 'items': {'type': 'string'}},
            'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'done', 'blocked']},
            'references': {'type': 'array', 'items': reference}},
            'required': ['id', 'goal', 'depends', 'requirements', 'status', 'references']}
        fields = {name: {'type': 'string'} for name in ('reason', 'current_step', 'next_action', 'blocker')}
        fields.update({name: {'type': 'array', 'items': {'type': 'string'}} for name in ('findings', 'hypotheses')})
        fields['steps'] = {'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': step}
        required = list(fields)
        for key in ('decisions', 'ruled_out'):
            fields[key] = {'type': 'array', 'maxItems': 12,
                'items': {'type': 'string', 'maxLength': 1000}}
        fields['objects'] = {'type': 'array', 'maxItems': 8, 'items': {
            'type': 'object', 'additionalProperties': False, 'properties': {
                'name': {'type': 'string', 'maxLength': 120},
                'candidates': {'type': 'array', 'maxItems': 12, 'items': {'type': 'string'}},
                'conflicts': {'type': 'array', 'maxItems': 12, 'items': {'type': 'string'}}},
            'required': ['name', 'candidates', 'conflicts']}}
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': {'expected_revision': {'type': 'integer', 'minimum': 0},
                'plan': {'type': 'object', 'additionalProperties': False, 'properties': fields, 'required': required}},
            'required': ['expected_revision', 'plan']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not str(k).startswith('_tool_') and k != '_native_tool_call'}
        if (set(args) != {'expected_revision', 'plan'} or type(args['expected_revision']) is not int
                or args['expected_revision'] < 0 or not isinstance(args['plan'], dict)):
            return None
        return args

    def _references(self, task, plan, context):
        refs = []
        for step in plan['steps']:
            for ref in step['references']:
                bound = {**ref, 'step_id': step['id']}
                if ref['kind'] == 'tool_call':
                    matches = [c for c in task.get('tool_calls', []) if c.get('id') == ref['id']]
                    call = matches[0] if len(matches) == 1 else None
                    handler = self.turn.handlers.get(call.get('name')) if call else None
                    _, fresh = self.turn.current(context)
                    if (not call or call.get('status') not in {'ok', 'error', 'blocked'} or handler is None
                            or call['name'] not in fresh.allowed_tools or not fresh.policy.accepts(call['name'])
                            or getattr(handler, 'agent_execution_evidence', True) is False):
                        raise AgentBoundaryError('work_plan_reference_unavailable')
                    # 失败/未执行的回执也能作为诊断索引；保留其状态，绝不升级为成功。
                    bound.update(receipt_sha256=digest(call), tool=call['name'],
                        execution_status=call['status'], executed=call.get('executed'))
                else:
                    materials = self.turn.service.materials
                    if materials is None:
                        raise AgentBoundaryError('work_plan_material_unavailable')
                    materials.resolve(ref['id'], task=task, still_authorized=self.turn.grant.still_authorized)
                    material = next(m for m in task['material_package']['items'] if m['reference'] == ref['id'])
                    bound.update(content_sha256=material['content_sha256'], version=material['version'])
                refs.append(bound)
        return refs

    def update(self, expected, proposal, context):
        service = self.turn.service
        with service._lock:
            task, fresh = self.turn.current(context)
            if not fresh.policy.accepts(self.tool_type):
                raise AgentBoundaryError('work_plan_not_authorized')
            if task.get('status') not in {'queued', 'running', 'planning'} or task.get('execution_result'):
                raise AgentBoundaryError('work_plan_task_not_active')
            if type(expected) is not int or expected < 0:
                raise AgentBoundaryError('work_plan_arguments_invalid')
            revision = task.get('work_state', {}).get('revision', 0)
            if expected != revision:
                raise AgentBoundaryError('work_plan_revision_conflict')
            plan = validate_plan(proposal, task, advisory=self.turn.reply_delivery)
            record = {'schema_version': 1, 'task_id': task['task_id'], 'requester': task['requester'],
                'session': task['session'], 'attempt': self.turn.attempt, 'task_version': task['revision'] + 1,
                'revision': revision + 1, 'binding': input_binding(task), 'plan': plan,
                'references': self._references(task, plan, context), 'created': service.task_store.clock()}
            if self.turn.reply_delivery:
                record['purpose'] = 'advisory'
            saved = service.task_store.save(task, work_state=record)
            ledger = context['agent_task_state']
            ledger.clear(); ledger.update(saved)
            # 工具参数/原稿已保留，下一轮控制消息另带当前完整计划。回执只保留
            # 索引，避免三份计划重复进入原生历史并挤满任务快照。
            return {'revision': saved['work_state']['revision'], 'status': 'saved',
                'sha256': saved['work_state']['sha256'], 'is_execution_evidence': False,
                'grants_permissions': False}

    def snapshot(self, context):
        service = self.turn.service
        with service._lock:
            task, _ = self.turn.current(context)
            record = service.task_store.get_work_state(task['task_id'], requester=task['requester'], session=task['session'])
            latest, _ = self.turn.current(context)
            if latest['revision'] != task['revision']:
                raise AgentBoundaryError('work_plan_task_changed_during_read')
            if record is None:
                return {'revision': 0, 'status': 'absent', 'is_execution_evidence': False}
            changed = record['binding'] != input_binding(task) or record['attempt'] != task['execution_attempt']
            return {'revision': record['revision'], 'status': 'requires_review' if changed else 'active',
                'plan': copy.deepcopy(record['plan']), 'references': copy.deepcopy(record['references']),
                'recorded_attempt': record['attempt'], 'current_attempt': task['execution_attempt'],
                'step_status_source': 'model_declaration', 'findings_source': 'model_unverified',
                'object_identity_source': 'model_unverified',
                'decisions_source': 'model_navigation', 'ruled_out_source': 'model_navigation',
                'references_current_validity': 'requires_revalidation', 'is_execution_evidence': False,
                'grants_permissions': False}

    def execute(self, *, call, context):
        args = self.normalize_call(call)
        try:
            if args is None:
                raise AgentBoundaryError('work_plan_arguments_invalid')
            snapshot = self.update(args['expected_revision'], args['plan'], context)
            return tool_result(self.tool_type, snapshot)
        except (TaskContractError, AgentBoundaryError) as exc:
            return tool_result(self.tool_type, {'code': str(exc), 'is_execution_evidence': False}, error=str(exc))
