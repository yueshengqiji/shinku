"""原始会话的同一执行循环；不另设准备器、路由模型或第二次核对。

入口：ServiceHost -> AgentTurnRequest -> AgentService._queue -> AgentExecutor。
原Service持有唯一任务/预算/持久化状态，本模块只持有本次执行的宿主适配。
独立宿主默认直接交付，模型计划不是执行前置条件；旧宿主可保留契约验收。
权限、材料门及真实执行回执仍归原宿主。恢复时重建适配，不反序列化回调。
"""
from collections.abc import Mapping
from dataclasses import replace
import copy
from types import SimpleNamespace

from shinku.agent.completion import CompletionVerdict
from shinku.agent.self_state import SelfStateTool
from shinku.agent.task_state import TaskContractError, validate_contract
from shinku.agent.analysis_contract import merge_plan
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry
from shinku.tools.workspace import tool_result
from .contracts import AgentBoundaryError


CONVERSATION_GUIDANCE = (
    '【同一会话执行循环】直接处理原始用户输入；你就是执行Agent，没有另一个准备器替你决定是否工作。'
    '普通交流/解释可直接final，真实缺少目标时wait并具体澄清；不要为聊天虚构工具条件。'
    # 用原始目标约束模型计划，而不是按关键词分类或另调用一个模型审查。
    '代码解释和学习辅导优先依据用户选区及必要的只读上下文，解释语义、变量与推导结果；'
    '静态推导须标明是推导，不能声称实测。不要自行把实际运行程序、安装依赖、修改文件'
    '或整项目测试加入这种请求的必需交付物和验收条件。'
    '用户明确要求运行、实测或调试时才以相应执行结果为目标，并继续严格核验真实回执；'
    '确需进一步验证才能确定的细节说明限制，不把可选验证变成原问题无法回答的理由。'
    '用户已要求调查时主动使用当前获准的只读工具取证，不再问是否做同一件事。'
    '可以先只读了解环境，再按证据确定执行/验收条件。命令、写入、发送及任何未知副作用操作前，'
    '先用declare_task_contract声明完整验收契约；同一原生工具历史会继续，声明不产生新的权限。'
    '调查目标和必要条件明确后，优先在首份完整成稿前调用declare_task_contract；'
    '不需要增加独立准备模型或重做读取。也可在成稿JSON里附task_contract，'
    '明确已经核实的条件后才能报告任务完成。用户目标、宿主硬条件只能追加，不能降低。'
    '宿主标记为model_plan的辅助读取和回答组织可调整；重规划保留历史，不改变已执行结果。'
    '说明性请求优先说明自己的模块；不要自行扩大到其他系统的完整审计。'
    '自身工具能力使用read_self_state，不当成网页研究；历史故障使用read_task_records定位当前会话的具体记录。'
    '服务日志诊断优先使用已注册的read_service_logs；其限定时间窗快照同时供执行与答案核验。'
    '通用Shell的stdout只是执行观察，不自动具备完整文件或日志来源资格；不要反复改写答案来弥补缺失来源。'
    '只读调查使用local_answer及实际需要的读取条件，不为了列目录人为添加command_run条件；'
    '用户要求真正执行命令、修改、重启或测试时，仍声明并严格核验对应效果。'
    '当前工具目录由宿主注册表与本任务策略提供：已注册、策略允许、实际调用成功是不同事实。'
    'available_names仅表示当前策略候选；具体路径、读写权限及审批仍由每次工具调用的宿主门检查。'
    '不要因某次参数错误或只读权限就说整个能力不存在；未调用的业务功能只能说尚未验证。'
    '能力建议先依据当前目录和用户需求，不为盘点工具去通读无关文件或继续旧任务。'
    '研究、项目分析、精确统计和实际行动按真实目标选择契约效果，不为了格式穷举无关条件。'
    '契约字段是内部元数据，正文自然回答；状态、原文缺失及执行失败须如实说明。'
)


def conversation_turn(context):
    value = context.get('_agent_conversation_turn') if isinstance(context, dict) else None
    return value if isinstance(value, ConversationTurn) else None


class ConversationTurn:
    """Service创建的进程内会话凭据；模型字典不能创建或恢复它。"""
    def __init__(self, service, scope, task_id, attempt, grant):
        self.service, self.scope, self.task_id, self.attempt, self.grant = service, scope, task_id, attempt, grant
        self.reply_delivery = getattr(service, 'conversation_reply_delivery', False) is True
        self.state = None
        self.handlers = {}
        self.declaration = DeclareContractTool(self)
        self.self_state = ConversationStateTool(self)
        from .work_planning import WorkPlanTool
        self.work_plan = WorkPlanTool(self)
        from .work_context import WorkContextTool
        self.work_context = WorkContextTool(self)

    def current(self, context, *, read_only=False):
        if conversation_turn(context) is not self or not self.grant.still_authorized():
            raise AgentBoundaryError('conversation_authorization_changed')
        if read_only:
            # 记忆/诊断读取不能借get触发整个任务库的TTL清理。绑定、尝试和
            # 实时授权检查与普通执行完全相同，只替换原件的读取方式。
            rows = self.service.task_store.read_scoped_page(requester=self.scope.requester,
                session=self.scope.session, task_id=self.task_id, limit=1)['items']
            if not rows:
                raise AgentBoundaryError('conversation_binding_changed')
            task = rows[0]
        else:
            task = self.service._get(self.scope, self.task_id)
        if (task.get('input_mode') != 'conversation' or task['execution_attempt'] != self.attempt
                or task.get('cancel_requested') or task.get('status') == 'canceled'):
            raise AgentBoundaryError('conversation_binding_changed')
        requests = tuple((str(r.get('message_id') or ''), r['text']) for r in task['requests'])
        bound = context.get('agent_request_scope')
        if (getattr(bound, 'requester', '') != self.scope.requester or getattr(bound, 'session', '') != self.scope.session
                or getattr(bound, 'requests', ()) != requests):
            raise AgentBoundaryError('conversation_binding_changed')
        fresh = self.service._grant(self.scope, task, 'inspect')
        if fresh.authorization_ref != self.grant.authorization_ref:
            raise AgentBoundaryError('conversation_authorization_changed')
        return task, fresh

    def attach(self, state, handlers):
        self.current(state.context)
        self.state, self.handlers = state, handlers

    def adopt(self, proposal):
        """在Service锁内校验并落盘，再更新同一循环；无新的模型或预算。"""
        if self.state is None:
            raise AgentBoundaryError('conversation_state_unavailable')
        if self.reply_delivery:
            from .reply_delivery import save_advisory
            return save_advisory(self, proposal)
        context = self.state.context
        from shinku.qq.task_contracts import validate_task_effects, task_target_coverage_gaps, task_effect_catalog
        from shinku.agent.loop import AgentDecision
        with self.service._lock:
            current, fresh = self.current(context)
            if not fresh.policy.accepts(self.declaration.tool_type):
                raise AgentBoundaryError('conversation_contract_not_authorized')
            contract = validate_contract(proposal)
            if any(r['effect'] in {'tool:declare_task_contract', 'tool:update_work_plan', 'tool:request_context_compaction'} for r in contract['requirements']):
                raise TaskContractError('contract_declaration_is_not_execution_evidence')
            if current.get('contract') is not None:
                contract = merge_plan(current, contract)
            deps = SimpleNamespace(registry_handlers=lambda: self.handlers)
            deps.task_effect_catalog = lambda: task_effect_catalog(deps=deps)
            validate_task_effects(contract, deps=deps)
            gaps = task_target_coverage_gaps({**current, 'contract': contract}, AgentDecision.final(''), deps=deps)
            if gaps:
                # 将当前请求中实际缺失的目标反馈给同一循环；错误不是新增授权。
                error = TaskContractError('contract_explicit_targets_uncovered', field_path='requirements')
                error.uncovered_targets = gaps
                raise error
            if contract != current.get('contract'):
                # TaskStore 事务统一生成来源归属并保留修订历史，不采信模型自报来源。
                saved = self.service._write(self.scope, self.task_id, self.attempt, lambda t: t.update(contract=contract))
                ledger = context['agent_task_state']
                ledger.clear(); ledger.update(saved)
            context['agent_completion_contract'] = copy.deepcopy(contract)
            # 契约在循环内才建立；重启/后声明分析条件时，用原回执重建答案账本，
            # 不重放读取，也不清空审查次数或先前已有来源。
            self.state.research.configure(self.handlers, context)
            was_local = self.state.local_answer.enabled
            self.state.local_answer.configure(context)
            if self.state.local_answer.enabled and not was_local:
                self.state.local_answer.restore_reads(self.state.tool_calls, self.handlers, context)
            self.state.history_answer.configure(context)
        return contract

    def tool_host(self, original, allowed, policy):
        handlers = {n: ConversationHandler(h, self) for n, h in original.registry.view().items()}
        allowed = set(allowed)
        risks = dict(policy.risk_by_tool)
        internal = [self.declaration, self.self_state, self.work_plan, self.work_context]
        if getattr(self.service.executor.profile.runtime, 'context_awareness_enabled', False) is True:
            from .context_tools import ContextCompactionTool
            internal.append(ContextCompactionTool(self))
        for tool in internal:
            if tool.tool_type in handlers:
                raise AgentBoundaryError('conversation_tool_conflict')
            if self.grant.policy.accepts(tool.tool_type):
                handlers[tool.tool_type] = tool
                allowed.add(tool.tool_type)
                risks.setdefault(tool.tool_type, 'low')
        # 初始policy由执行器按原注册表派生；只为宿主提供的内部工具加白名单，
        # 显式allow/deny和风险确认仍由上面的原grant策略裁决。
        adjusted = replace(policy, allowed=frozenset(allowed), risk_by_tool=risks)
        return ToolHost(ToolRegistry(handlers), host_id=original.host_id), allowed, adjusted

    def gate(self, original):
        if self.reply_delivery:
            from .reply_delivery import ReplyDeliveryGate
            return ReplyDeliveryGate(self)
        turn = self
        class Gate:
            def __getattr__(self, name):
                return getattr(original, name)

            def evaluate(self, state, text, *, outcome=''):
                task, _ = turn.current(state.context)
                from shinku.agent.analysis_acceptance import _execution_claim_error
                claim = _execution_claim_error(self, state, text)
                if claim:
                    return CompletionVerdict(False, reason=claim)
                if task.get('contract') is None and any(c.get('status') == 'ok' and c.get('executed') is not False
                        and getattr(turn.handlers.get(c.get('name')), 'agent_execution_evidence', True) is not False
                        for c in state.tool_calls) and outcome not in {'failed', 'blocked'}:
                    return CompletionVerdict(False, reason='已有工具取证，请在成稿JSON附task_contract或调用declare_task_contract明确验收条件；不要重做已完成的读取')
                verdict = original.evaluate(state, text, outcome=outcome)
                if task.get('contract') is None and isinstance(verdict, CompletionVerdict) and verdict.accepted and verdict.status == 'completed':
                    # 聊天交付与行动验收分开，不能把一句答复升级为执行成功。
                    evidence = tuple({**e, 'business_goal': False} if 'business_goal' in e else e for e in verdict.evidence)
                    return replace(verdict, reason='conversation_reply', evidence=evidence,
                        layers={'policy': 'conversation_reply', 'reply_delivery': 'delivered', 'execution_results': 'not_verified'})
                return verdict
        return Gate()


class ConversationHandler:
    """每次调用重验宿主权限；仅兼容模式保留模型契约前置门。"""
    def __init__(self, handler, turn):
        self.handler, self.turn = handler, turn

    def __getattr__(self, key):
        return getattr(self.handler, key)

    def execute(self, *, call, context):
        try:
            task, fresh = self.turn.current(context)
            name = self.handler.tool_type
            if name not in fresh.allowed_tools or not fresh.policy.accepts(name):
                raise AgentBoundaryError('conversation_tool_not_authorized')
            from .recovery import repeatable_read
            if not self.turn.reply_delivery and task.get('contract') is None and not repeatable_read(self.handler):
                raise AgentBoundaryError('conversation_contract_required_before_effect')
        except AgentBoundaryError as exc:
            return {'status': 'blocked', 'code': exc.code, 'followup_context':
                '该操作未执行：' + exc.code + '。依据宿主错误处理；权限不足时说明具体限制。',
                'state_updates': {'executed': False, 'information_gap': exc.code}}
        return self.handler.execute(call=call, context=context)


class DeclareContractTool:
    tool_type = 'declare_task_contract'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_execution_evidence = False

    def __init__(self, turn):
        self.turn = turn

    def build_prompt_instruction(self):
        if self.turn.reply_delivery:
            return '可选的兼容计划记录。可替换辅助步骤，不是执行或答复的前置条件，不授予权限或完成标记。'
        return '声明任务条件或调整宿主标记的model_plan调查步骤。用户目标和硬条件不能删改；不执行外部操作，不授予权限。'

    def tool_metadata(self):
        from shinku.agent.contract_schema import contract_input_schema
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': {'task_contract': contract_input_schema()}, 'required': ['task_contract']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not str(k).startswith('_tool_') and k != '_native_tool_call'}
        return args if set(args) == {'task_contract'} and isinstance(args['task_contract'], Mapping) else None

    def execute(self, *, call, context):
        args = self.normalize_call(call)
        try:
            if args is None or conversation_turn(context) is not self.turn:
                raise AgentBoundaryError('conversation_contract_arguments_invalid')
            contract = self.turn.adopt(args['task_contract'])
            return tool_result(self.tool_type, {'contract': contract, 'source': 'model_advisory_plan' if self.turn.reply_delivery else 'host_validated_contract',
                'grants_permissions': False, 'is_execution_evidence': False})
        except (TaskContractError, AgentBoundaryError) as exc:
            return tool_result(self.tool_type, {'code': str(exc), 'field_path': getattr(exc, 'field_path', ''),
                'uncovered_targets': getattr(exc, 'uncovered_targets', [])}, error=str(exc))


class ConversationStateTool(SelfStateTool):
    """同一循环可随时读取宿主工具清单；不依赖关键词选择专用任务类型。"""
    agent_observation_required_fields = ('self_state_receipt',)
    agent_observation_receipt_fields = ('self_state', 'self_state_receipt', 'result_valid', 'result_facts')
    agent_repeatable_observation = True

    def __init__(self, turn):
        super().__init__()
        self.turn = turn

    def _read_snapshot(self, context, *, fresh=False):
        task, grant = self.turn.current(context)
        if not grant.policy.accepts(self.tool_type):
            raise AgentBoundaryError('conversation_state_not_authorized')
        registered = sorted(self.turn.handlers)
        names = sorted(n for n in registered if grant.policy.accepts(n)
            and (n in grant.allowed_tools or n in {'read_self_state', 'declare_task_contract', 'update_work_plan', 'read_work_context', 'task_material_read', 'request_context_compaction'}))
        # 仅投影宿主已记录的调用状态，不暴露参数/正文，不把调用成功升级为业务健康。
        calls = task.get('tool_calls', [])
        observations = [{'name': row.get('name'), 'status': row.get('status', 'unknown'),
            'executed': row.get('executed'),
            'result_valid': (row.get('data', {}).get('state_updates') or {}).get('result_valid')}
            for row in calls[-16:] if row.get('name') in names and row.get('name') != self.tool_type]
        # 来源限制只投影协议中的公开解释字段，不能把宿主扩展上下文整体交给模型。
        authority = grant.tool_context.get('delegation_authority', {})
        sources = {key: copy.deepcopy(authority[key]) for key in
            ('source_mode', 'source_excluded_tools', 'server_excluded_tools') if key in authority}
        from shinku.tools.workspace import WorkspaceTool
        workspaces = {}
        for name in names:
            handler = self.turn.handlers[name]
            handler = handler.handler if isinstance(handler, ConversationHandler) else handler
            root_reader = getattr(handler, 'agent_workspace_root', None)
            if callable(root_reader):
                try:
                    root = root_reader(context)
                    if root:
                        workspaces.setdefault(root, []).append(name)
                except (ValueError, PermissionError):
                    pass
                continue
            if not isinstance(handler, WorkspaceTool):
                continue
            # 根目录来自已注册适配器，用户输入中的路径不能伪造可访问工作区。
            # check只核对本次读/写资格；不读取目录，不缓存授权，不证明文件存在。
            try:
                handler.access.check(context, write=handler.operation == 'apply_patch')
            except PermissionError:
                continue
            workspaces.setdefault(str(handler.access.root), []).append(name)
        model = getattr(getattr(self.turn.service.executor.profile.runtime, 'chat', None), 'model', '')
        return {'runtime': {'model': str(model)}, 'agent': {'name': 'Angelos', 'mode': 'single_execution_loop'},
            'tools': {'available_names': names, 'available_count': len(names),
                'registered_names': registered, 'registered_count': len(registered),
                'not_allowed_names': sorted(set(registered) - set(names)),
                'permission_mode': grant.permission_mode, 'task_scope': 'current_task',
                'approval_required_names': [n for n in names if grant.policy.requires_approval(n)],
                'workspace_access': [{'root': root, 'tools': tools, 'path_checks_required': True}
                    for root, tools in sorted(workspaces.items())],
                'workspace_access_scope': 'registered_workspace_adapters_only',
                'permission_sources': {**sources,
                    'angelos_level': grant.tool_context.get('angelos_permission_level', ''),
                    'rule': '当前可用工具是来源宿主与执行宿主授权的交集；本机档位不会补回来源未授予的工具。'},
                'execution_requires_host_checks': True, 'business_health': 'not_verified',
                'recent_calls': observations, 'registration_is_not_business_health': True}}, 'fresh_provider'

    def prompt_snapshot(self, context):
        """稳定的小型能力视图进入受预算约束的输入增量；详细回执仍按需读取。"""
        snapshot, _ = self._read_snapshot(context)
        return {key: snapshot['tools'][key] for key in ('available_names', 'not_allowed_names',
            'permission_mode', 'permission_sources', 'approval_required_names', 'workspace_access', 'workspace_access_scope',
            'execution_requires_host_checks', 'business_health')}

    def agent_verify_result(self, *, receipt, context):
        try:
            scope = receipt.get('self_state_receipt', {}).get('scope')
            resolved = self.resolve_state_observation(updates=receipt, arguments={'scope': scope}, context=context)
            current, _ = self._read_snapshot(context, fresh=True)
            selected = current if scope == 'all' else {scope: current.get(scope, {})}
            observed = copy.deepcopy(receipt.get('self_state'))
            # 历史调用是读取时的观察，不要求以后永不变化；实时工具目录/策略仍须一致。
            for snapshot in (selected, observed):
                if isinstance(snapshot, dict) and isinstance(snapshot.get('tools'), dict):
                    snapshot['tools'].pop('recent_calls', None)
            return {'result_valid': bool(resolved and resolved['valid'] and observed == selected)}
        except Exception:
            return {'result_valid': False}
