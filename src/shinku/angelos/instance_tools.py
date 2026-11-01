"""两个无模型调用的实例观察工具；复用当前任务授权和观察凭据。

ServiceHost在权限包装前安装。读取失败以缺口返回，不要求模型必须调用，也不
改变回答验收。凭据证明当时读取与实例归属，不能证明模型推断或未来仍在运行。
"""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.tools.workspace import tool_result
from .instance_probe import InstanceProbe
from .task_records import TaskRecordsTool

# 两个工具各说明自身接口，避免同一长段在工具目录里注入两遍。
# 通用调查方法由input_routing负责，历史可信度由conversation_context说明。
LIST_GUIDANCE = (
    '列出宿主登记的实例ID、名称、服务与连接位置；登记目录不证明在线。'
    '需要现场信息时，用返回的instance_id调用inspect_runtime_instance。'
)
INSPECT_GUIDANCE = (
    '按登记instance_id查询现场；sections可选models、locations，默认models。'
    'running是服务报告值，saved是磁盘保存值，不能互相填补未知；locations是登记路径。'
    '结果只描述observed_at的观察，comparison_fields明确比较范围，gaps说明缺口；'
    '实例默认模型不代表会话选择，请求模型不证明代理最终上游，模型目录和工具白名单也不证明当前选择。'
    '本工具并非必选门槛；可用其他已授权只读证据，不要为核实模型名主动发模型请求。'
)


class InstanceTool(TaskRecordsTool):
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_completion_effects = ('read',)
    agent_local_answer_source = True
    agent_local_answer_reference_prefix = 'instance:'
    agent_observation_receipt_fields = ('observation', 'instance_snapshot', 'path', 'sha256')

    def __init__(self, service, requester, registry, *, inspect=False, probe=None):
        super().__init__(service, requester)
        self.registry, self.probe, self.inspect = registry, probe or InstanceProbe(), inspect
        self.tool_type = 'inspect_runtime_instance' if inspect else 'list_runtime_instances'

    def build_prompt_instruction(self):
        return INSPECT_GUIDANCE if self.inspect else LIST_GUIDANCE

    def tool_metadata(self):
        props = {} if not self.inspect else {
            'instance_id': {'type': 'string', 'enum': list(self.registry.entries)},
            'sections': {'type': 'array', 'items': {'type': 'string', 'enum': ['models', 'locations']},
                         'minItems': 1, 'maxItems': 2, 'uniqueItems': True}}
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': props, 'required': ['instance_id'] if self.inspect else []})

    def normalize_call(self, call):
        if not isinstance(call, dict):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not k.startswith('_')}
        if not self.inspect:
            return {} if not args else None
        if set(args) - {'instance_id', 'sections'}:
            return None
        try:
            self.registry.get(args.get('instance_id'))
        except ValueError:
            return None
        sections = args.get('sections', ['models'])
        if not isinstance(sections, list) or not 1 <= len(sections) <= 2 or any(not isinstance(s, str) or s not in ('models', 'locations') for s in sections) or len(set(sections)) != len(sections):
            return None
        return {'instance_id': args['instance_id'], 'sections': sections}

    def execute(self, *, call, context):
        try:
            args = self.normalize_call(call)
            if args is None:
                raise ValueError('invalid_instance_query')
            self._authorized_task(context)
            snapshot = (self.probe.inspect(self.registry.get(args['instance_id']), args['sections'])
                        if self.inspect else self.registry.catalog())
            self._authorized_task(context)  # IO期间撤权，不能交出结果或签发凭据。
            observation = self.observations.issue(snapshot, tool=self.tool_type, context=context)
            text = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
            return tool_result(self.tool_type, {'instance_snapshot': snapshot, 'observation': observation,
                'path': 'instance:' + observation['self_state_receipt']['read_id'],
                'sha256': hashlib.sha256(text.encode()).hexdigest()})
        except AgentBoundaryError as exc:
            code = exc.code
        except (ValueError, OSError):
            code = 'invalid_instance_query'
        return tool_result(self.tool_type, {'code': code}, error=code)

    def agent_verify_result(self, *, receipt, context):
        try:
            self._authorized_task(context)
            valid = self.observations.verify(receipt, tool=self.tool_type, context=context)
            text = json.dumps(receipt['instance_snapshot'], sort_keys=True, ensure_ascii=False)
            signature = hashlib.sha256(text.encode()).hexdigest()
            observation = receipt['observation']
            valid = (valid and signature == observation['self_state']['data_sha256']
                     and receipt.get('sha256') == signature
                     and receipt.get('path') == 'instance:' + observation['self_state_receipt']['read_id'])
            # 不重新查询现场；已发生的观察按当时内容核验，时间在snapshot中明确。
            return {'result_valid': bool(valid), 'file_text': text if valid else ''}
        except (AgentBoundaryError, KeyError, TypeError, ValueError):
            return {'result_valid': False}


def install_instance_tools(service, requester, registry):
    tools = tuple(InstanceTool(service, requester, registry, inspect=detail) for detail in (False, True))
    handlers = service.executor.tool_host.registry
    if any(handlers.get(tool.tool_type) is not None for tool in tools):
        raise AgentBoundaryError('instance_tools_already_installed')
    for tool in tools:
        handlers.register(tool)
    original = service.authorize
    def authorize(scope, task, operation):
        grant = original(scope, task, operation)
        if grant is None or scope.requester != requester or not scope.session.startswith('angelos:'):
            return grant
        names = {t.tool_type for t in tools if grant.policy.accepts(t.tool_type)}
        policy = replace(grant.policy, risk_by_tool={**{n: 'low' for n in names}, **grant.policy.risk_by_tool})
        return replace(grant, allowed_tools=grant.allowed_tools | names, policy=policy)
    service.authorize = authorize
    return tools
