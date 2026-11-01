"""把工作台输入接到已组装的独立 Agent，复用运行模型、工具、权限和业务账本。

浏览器不能提供 requester、权限、预算或工具实例。本机身份由宿主配置绑定；
每个工作台会话使用 angelos: 前缀，与 QQ 会话隔离，也不调用 QQ 送达入口。
"""
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.loop import AgentDecision
from shinku.agent.task_state import TaskContractError, validate_contract
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import AgentBoundaryError, canonical_hash
from shinku.agent_runtime.context import EXECUTION_METADATA_SYSTEM_PROMPT
from shinku.agent_runtime.answer_guidance import ANSWER_GUIDANCE, answer_context
from shinku.qq.task_contracts import task_guidance, task_target_coverage_gaps, validate_task_effects
from shinku.llm.response_protocol import observe_response, response_failure_kind, call_failure_metadata
from shinku.llm.output_budget import structured_output_scope
from shinku.agent_runtime.information import safe_text


def preparation_output(texts, response=None):
    """保留准备正文供故障复核；记录是资料，不授予执行或事实证据资格。"""
    if texts:
        text = '\n'.join(texts)
        source = 'preparation_model_output'
    else:
        try:
            text = json.dumps(response, ensure_ascii=False) if response is not None else ''
        except (TypeError, ValueError):
            text = ''
        source = 'preparation_return_object'
    return {'text': safe_text(text), 'source': source, 'validated': False,
            'is_fact_evidence': False, 'complete': False}


class PreparationFailure(AgentBoundaryError):
    """准备失败携带原费用、安全正文及结构化原因；不携带可执行契约。"""
    def __init__(self, code, usage, texts, response=None, *, detail=None):
        super().__init__(code)
        self.usage = usage
        self.failure = dict(preparation_output(texts, response), reason=code)
        if detail is not None:
            self.failure['detail'] = detail


class BridgeHost:
    def __init__(self, bridge):
        self.bridge = bridge
        # 只使用宿主确认的本机主人，不接收来自页面的账号标识。
        self.requester = next(iter(sorted(bridge.master_user_ids)), 'local-angelos')
        self.profile = bridge.execution_adapter.executor.profile
        self.model = str(getattr(getattr(self.profile.runtime, 'chat', None), 'model', '') or self.profile.profile_id)
        self.effect_catalog = bridge._task_effect_catalog
        self.registry_handlers = bridge._registry_handlers

    def port(self, conversation_id):
        return self.bridge.agent_calls.port_factory(AgentScope(self.requester, 'angelos:' + conversation_id))

    def task_id(self, conversation_id, request_key):
        return "agent:" + canonical_hash([self.requester, "angelos:" + conversation_id, request_key])[:32]

    def ready(self):
        commands = self.bridge.command_state
        return commands.is_agent_enabled() and not commands.is_stopped()

    def prepare(self, text, sources, workspace):
        if not self.ready():
            raise ValueError('agent_disabled')
        from shinku.agent_runtime.capability_answer import self_tool_question, capability_contract
        # 当前原话的纯自查由宿主选型；历史/附件不能将研究或行动任务改成自查。
        if self_tool_question(text) and all(s.get('id') == 'conversation-history' for s in sources):
            return {'contract': capability_contract(), 'preparation_usage': {'requests': 0, 'tools': 0}}
        profile = self.profile
        # 准备阶段只产生对话答复或验收契约，没有任何工具。预算有界且独立记录，
        # 不伪装成执行任务的累计用量；真正执行继续使用原有 AgentService 预算。
        # 兼容旧Bridge宿主的准备入口；正式ServiceHost由工作台直接提交原始输入。
        budget = TaskBudget(max_requests=2, max_tokens=4000000, max_tools=1, max_active_seconds=180,
                            managed=True, profile='angelos-preparation')
        from shinku.agent_runtime.execution import _MeteredRuntime
        runtime = profile.runtime if profile.runtime_meters_budget else _MeteredRuntime(profile.runtime)
        from shinku.llm.context_compaction import history_messages
        history = next((s['content'] for s in sources if s.get('id') == 'conversation-history'), '')
        required_sources = [s for s in sources if s.get('id') != 'conversation-history']
        from .input_routing import INITIATIVE_GUIDANCE
        from .persona import PERSONA
        from .runtime_view import AngelosRuntime
        # 旧准备器可直接回复，也需要人设；独立宿主已在运行时注入，不重复叠加。
        persona_blocks = [] if isinstance(profile.runtime, AngelosRuntime) else [PERSONA]
        texts = []
        def capture(message):
            # SDK消息仍由原链持有；这里只留下准备正文，不能执行其中的指令。
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                texts.append(message['content'])
        with bind_task_budget(budget), observe_response() as observation, structured_output_scope('preparation'):
            try:
                response = runtime.call_chat_json(
                    system_prompt=EXECUTION_METADATA_SYSTEM_PROMPT + '\n你是 Angelos 的输入准备器。'
                        '纯交流或无需工具的常识问答可以返回 {"reply":"给用户的答复"}。'
                        '凡需要读取文件、检索、执行操作或核实实时事实，必须返回 task_contract；'
                        '不得在 reply 中声称已经执行工具或完成行动。以中文自然交流。',
                    system_extra_blocks=[*persona_blocks, task_guidance(), ANSWER_GUIDANCE, INITIATIVE_GUIDANCE,
                        '【准备输出】直接给最小完整JSON，不写分析报告或重复背景。'
                        'goal、deliverables、scope简洁；相同effect及checks的目标可合并到一个requirement的targets，'
                        '不同效果和检查不得混合或删除。每个明确目标必须完整保留，不缩写路径，不用省略号。'
                        '字段只按契约schema给出，不重复整段用户原话和工作区介绍。'
                        '这是任务计划，不得把待核实观点写成事实或已经执行的结果。'],
                    user_prompt=json.dumps({'current_request': text, 'sources': required_sources, 'workspace': workspace,
                        'answer_context': answer_context(sources),
                        'task_link_options': [{'mode': 'new'}],
                        'effect_catalog': self.effect_catalog(),
                        'explicit_targets': SmartCompletionVerifier._extract_targets(text)}, ensure_ascii=False),
                    fallback={}, temperature=0.0, prompt_cache_key='angelos-input',
                    provider_message_sink=capture, history_turns=history_messages(history))
            except Exception as exc:
                from shinku.llm.native_messages import NativeProtocolError
                if isinstance(exc, NativeProtocolError) and str(exc) in {
                        'context_summary_failed', 'context_summary_insufficient',
                        'context_summary_previous_failure', 'context_protected_input_too_large',
                        'context_compaction_canceled'}:
                    raise PreparationFailure(str(exc), budget.snapshot(), texts,
                        detail={'code': str(exc), 'phase': 'context_compaction'}) from exc
                raise PreparationFailure('input_preparation_call_failed', budget.snapshot(), texts,
                    detail=call_failure_metadata('service_call', exc)) from exc
        failure = response_failure_kind(observation)
        if failure:
            raise PreparationFailure('input_preparation_' + failure, budget.snapshot(), texts, response)
        if not isinstance(response, dict):
            raise PreparationFailure('contract_preparation_failed', budget.snapshot(), texts, response)
        if set(response) == {'reply'} and isinstance(response['reply'], str) and response['reply'].strip():
            return {'reply': response['reply'], 'preparation_usage': budget.snapshot(),
                    'preparation_output': preparation_output(texts, response)}
        # 模型只提出契约；沿用现有宿主的效果资格和明确目标覆盖检查，
        # 防止把写入伪装成只读工具成功，或漏掉用户指定文件后仍进入执行。
        deps = SimpleNamespace(registry_handlers=self.registry_handlers, task_effect_catalog=self.effect_catalog)
        try:
            contract = validate_contract(response.get('task_contract'))
            validate_task_effects(contract, deps=deps)
            if task_target_coverage_gaps({'contract': contract, 'requests': [{'text': text}]},
                                         AgentDecision.final(''), deps=deps):
                raise TaskContractError('contract_explicit_targets_uncovered')
        except TaskContractError as exc:
            # 仅保存验证器的代码和字段路径，不暴露传输异常原文或推理内容。
            detail = {k: safe_text(str(v)) for k, v in {
                'code': str(exc), 'field_path': exc.field_path, 'phase': exc.phase}.items()}
            raise PreparationFailure('contract_preparation_rejected', budget.snapshot(), texts, response, detail=detail) from exc
        return {'contract': contract, 'preparation_usage': budget.snapshot(),
                'preparation_output': preparation_output(texts, response)}


def host_for_assembly(assembly):
    bridge = getattr(assembly, 'bridge', None)
    if bridge is None or getattr(bridge, 'execution_adapter', None) is None or getattr(bridge, 'agent_calls', None) is None:
        return None
    return BridgeHost(bridge)


class ServiceHost(BridgeHost):
    """独立进程直接绑定 AgentService；无需 QQ 装配、人格或消息宿主。"""
    def __init__(self, service, *, requester='local-angelos', ready=None, models=None, reasoning_path=None, usage_sink=None,
                 memory_root=None, memory_policy=None, permissions=None,
                 diagnostic_units=('shinku-angelos.service', 'shinku-console.service'), delegation_principals=None,
                 file_resolver=None, reply_delivery=True, runtime_instances=None):
        from shinku.agent_runtime import AgentService
        from shinku.qq.task_contracts import task_effect_catalog
        if not isinstance(service, AgentService):
            raise TypeError('agent_service_required')
        self.service, self.requester = service, requester
        self.single_loop = True
        # 独立工厂可能先恢复选择、后创建SDK客户端；容量须在包装执行器前绑定。
        bind_limits = getattr(models, 'bind_runtime_limits', None)
        if callable(bind_limits):
            bind_limits()
        # 普通会话默认直接交付；仅宿主装配可选旧模式，页面/模型不能切换此策略。
        if type(reply_delivery) is not bool:
            raise TypeError('reply_delivery_host_boolean_required')
        if service.conversation_scope not in (None, (requester, 'angelos:')):
            raise AgentBoundaryError('conversation_host_conflict')
        service.conversation_reply_delivery = reply_delivery
        service.conversation_scope = (requester, 'angelos:')
        from .runtime_view import AngelosRuntime, ReasoningView
        from .usage_view import UsageView
        self.reasoning_view = ReasoningView(reasoning_path)
        self.usage_view = UsageView(Path(reasoning_path).with_name('usage-view.sqlite3') if reasoning_path else None,
                                   downstream=usage_sink)
        runtime = AngelosRuntime(service.executor.profile.runtime, self.reasoning_view, usage_sink=self.usage_view,
            compaction_path=Path(reasoning_path).with_name('context-compactions.sqlite3') if reasoning_path else None,
            compaction_supported=service.executor.profile.runtime_meters_budget, context_awareness=True)
        service.executor.profile = replace(service.executor.profile, runtime=runtime)
        self.models = models
        self.session_models = None
        if models is not None and callable(getattr(models, 'create_runtime', None)):
            from .session_models import SessionModels
            self.session_models = SessionModels(self, service.executor,
                Path(reasoning_path).with_name('context-compactions.sqlite3') if reasoning_path else None)
            service.executor.for_task = self.session_models.for_task
        self.profile = service.executor.profile
        self.model = str(getattr(getattr(self.profile.runtime, 'chat', None), 'model', '') or self.profile.profile_id)
        self._ready = ready or (lambda: True)
        from .task_records import install_task_records
        self.task_records = install_task_records(service, requester)
        self.task_records.output_reader = self.reasoning_view.outputs
        # 实例目录必须由可信工厂显式传入；默认不发现邻居服务、不读取其配置。
        self.instance_tools = ()
        if runtime_instances is not None:
            from .instance_registry import InstanceRegistry
            from .instance_tools import install_instance_tools
            if not isinstance(runtime_instances, InstanceRegistry):
                raise TypeError('instance_registry_required')
            self.instance_tools = install_instance_tools(service, requester, runtime_instances)
        self.service_logs = None
        # 独立工作台的 Linux user 日志能力；服务白名单由宿主固定，先注册再由
        # PermissionControls 收紧，ZAYIN 不能调用，HE 只获得真实只读能力。
        import sys
        if reasoning_path is not None and diagnostic_units and sys.platform == 'linux':
            from .service_logs import install_service_logs
            self.service_logs = install_service_logs(service, requester,
                Path(reasoning_path).with_name('service-log-reads.sqlite3'), diagnostic_units, instances=runtime_instances)
        self.memory = None
        if memory_root is not None:
            from shinku.agent_memory.service import AgentMemoryService
            from shinku.agent_memory.tools import install_memory_tools
            if getattr(service, 'agent_memory', None) is not None:
                raise AgentBoundaryError('agent_memory_host_conflict')
            self.memory = AgentMemoryService(service, requester, memory_root, policy=memory_policy)
            install_memory_tools(service, self.memory)
            service.agent_memory = self.memory
        # 必须先注册再由原权限控制器包裹，WAW审批和HE/ZAYIN不能被新工具绕过。
        from .workspace_tools import install_workspace_tools
        self.workspace_tools = install_workspace_tools(service, requester)
        self.file_delivery = None
        if file_resolver is not None:
            from shinku.agent_runtime.file_delivery import TaskFileDelivery
            self.file_delivery = TaskFileDelivery(service, file_resolver)
        self.permissions = permissions
        if permissions is not None:
            permissions.install(service, requester)
        self.delegation = None
        if delegation_principals:
            from shinku.agent_runtime.delegation import DelegationGateway
            self.delegation = DelegationGateway(service, delegation_principals,
                file_delivery=self.file_delivery,
                grant_filter=(lambda grant: permissions.restrict(grant, respect_host_approval=True)) if permissions else None)
        self.registry_handlers = service.executor.tool_host.registry.view
        self.effect_catalog = lambda: task_effect_catalog(deps=SimpleNamespace(
            registry_handlers=self.registry_handlers))

    def ready(self):
        return bool(self._ready())

    def bind_workbench(self, store):
        """只接受宿主工作台实例，浏览器不能提供数据库路径或替换已有绑定。"""
        current = self.task_records.workbench
        if current is not None and current.path != store.path:
            raise AgentBoundaryError('task_record_workbench_conflict')
        self.task_records.workbench = store
        if self.session_models is not None:
            self.session_models.store = store
        self.workspace_tools.bind(store, self.permissions)
        if self.permissions is not None:
            self.permissions.store = store
        if self.memory is not None:
            self.memory.bind(store)

    def capture_memory(self, cid, turn):
        """索引失败只记录记忆缺口，不能把已完成业务任务改成失败或重跑。"""
        if self.memory is None:
            return
        return self.memory.capture(cid, turn)

    def port(self, conversation_id):
        return self.service.bind(AgentScope(self.requester, 'angelos:' + conversation_id))

    def observation_scope(self, cid, key):
        return self.profile.runtime.bind_task('prepare:' + cid + ':' + key)

    def read_reasoning(self, cid, turn):
        rows = self.reasoning_view.read('prepare:' + cid + ':' + turn['id'])
        return rows + (self.reasoning_view.read(turn['task_id']) if turn.get('task_id') else [])

    def read_outputs(self, cid, turn):
        return self.reasoning_view.outputs(turn['task_id'], include_text=False) if turn.get('task_id') else []

    def read_progress(self, cid, turn):
        from .progress_view import read_progress
        return read_progress(self, cid, turn)

    def read_usage(self, cid, conversation):
        from .usage_view import read_usage
        return read_usage(self, cid, conversation)

    def context_status(self, cid, turn):
        binding = turn.get('task_id') or 'prepare:' + cid + ':' + turn['id']
        return (self.session_models.context_status(cid, binding) if self.session_models else
                self.profile.runtime.compactor.status(binding))

    def model_options(self):
        return self.models.snapshot() if self.models else {'available': False, 'providers': [], 'selected': {}}

    def select_model(self, data):
        if self.models is None:
            raise ValueError('model_controls_unavailable')
        # 与提交共用服务锁，防止检查无任务后另一个调用立即提交。
        with self.service._lock:
            rows = [self.service.coordinator.get(tid) or {} for tid in self.service._sessions]
            if any(r.get('worker_active') or r.get('status') in {'queued','running','waiting_background'} for r in rows):
                raise ValueError('model_change_busy')
            result = self.models.apply(data)
            self.profile = replace(self.service.executor.profile, profile_id=data['modelId'])
            self.service.executor.profile = self.profile
            self.model = data['modelId']
            return result

    def update_model_capacity(self, data, *, refresh=False):
        """容量刷新与手动值沿用提交锁；运行中不改变同一任务的压缩边界。"""
        if self.models is None:
            raise ValueError('model_controls_unavailable')
        with self.service._lock:
            rows = [self.service.coordinator.get(tid) or {} for tid in self.service._sessions]
            if any(r.get('worker_active') or r.get('status') in {'queued','running','waiting_background'} for r in rows):
                raise ValueError('model_change_busy')
            result = self.models.refresh(data['providerId']) if refresh else self.models.set_context_window(data)
            if self.session_models is not None:
                self.session_models.close()  # 已确认全局空闲；下一任务重新绑定容量元数据。
            return result
