"""Agent 专用检索/管理工具；原权限、契约与观察凭据继续裁决。

工具参数不含库路径、用户或工作区。读凭据只认证某次来源读取，不能认证
旧任务当前成功；写凭据只认证本库写入，不产生业务效果或执行授权。
"""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from shinku.agent_runtime.contracts import AgentBoundaryError, canonical_hash
from shinku.tools.observation_receipts import ObservationReceipts
from shinku.tools.workspace import tool_result


class AgentMemoryTool:
    agent_progress_class = 'observe'
    agent_observation_receipt_fields = ('observation', 'memory_query', 'memory_version')

    def __init__(self, memory, *, write=False):
        self.memory, self.write = memory, write
        self.tool_type = 'manage_agent_memory' if write else 'read_agent_memory'
        self.agent_permission_class = 'write' if write else 'read'
        self.agent_progress_class = 'change' if write else 'observe'
        self.agent_completion_effects = ('memory_manage',) if write else ()
        self.agent_observation_receipt_fields = ('observation', 'memory_query', 'memory_version') + (
            ('id', 'status', 'version', 'source', 'is_execution_evidence', 'grants_permissions') if write else ())
        self.observations = ObservationReceipts()

    def build_prompt_instruction(self):
        if self.write:
            return ('仅管理 Agent 自己的记忆。用户要求保存或忘记时在本循环声明effect=memory_manage、targets=[]的契约后调用；'
                'remember只能提交当前用户原话中的连续片段和message_id，不能保存模型推断；'
                '片段只定位原输入，实际保存整条原话及其条件/否定，不把片段单独当结论；'
                '陈述保持原条件、否定和归属，不当作当前事实或长期授权。forget使用本库读取出的id。')
        return ('只查询 Agent 独立记忆库，不查询当前角色Bot。先用当前任务记忆核对刚才实际执行结果；'
            '跨会话的项目、软件身份或经验需要时再检索。来源失效需明确缺口；'
            '用户陈述、旧助手结论与真实效果分开，名字相同不代表同一软件。')

    def tool_metadata(self):
        props = ({'operation': {'type': 'string', 'enum': ['remember', 'forget']},
            'text': {'type': 'string', 'maxLength': 2000}, 'message_id': {'type': 'string', 'maxLength': 256},
            'id': {'type': 'string', 'maxLength': 64}} if self.write else
            {'query': {'type': 'string', 'maxLength': self.memory.store.policy.max_query_chars}})
        return SimpleNamespace(input_schema={'type': 'object', 'additionalProperties': False,
            'properties': props, 'required': ['operation'] if self.write else ['query']})

    def normalize_call(self, call):
        if not isinstance(call, dict):
            return None
        args = {k: v for k, v in call.items() if k != 'type' and not k.startswith('_tool_') and k != '_native_tool_call'}
        expected = ({'operation', 'text', 'message_id'} if args.get('operation') == 'remember'
                    else {'operation', 'id'}) if self.write else {'query'}
        if (set(args) != expected or any(not isinstance(v, str) or not v.strip() for v in args.values())
                or any(len(v) > (2000 if k == 'text' else self.memory.store.policy.max_query_chars if k == 'query' else 256)
                       for k, v in args.items())
                or self.write and args.get('operation') not in {'remember', 'forget'}):
            return None
        return args

    def _read(self, args, context):
        return self.memory.recall(context, args['query'])

    def execute(self, *, call, context):
        try:
            args = self.normalize_call(call)
            if args is None:
                raise AgentBoundaryError('agent_memory_arguments_invalid')
            data = self.memory.manage(context, args) if self.write else self._read(args, context)
            data['memory_query'] = args
            data['memory_version'] = canonical_hash(data)
            proof = {k: data[k] for k in ('memory_query', 'memory_version')}
            data['observation'] = self.observations.issue(proof, tool=self.tool_type, context=context)
            if self.write:
                data.update(verified_effects=['memory_manage'], completion_targets=[data['id']])
            return tool_result(self.tool_type, data)
        except (ValueError, KeyError) as exc:
            code = exc.code if isinstance(exc, AgentBoundaryError) else str(exc) if isinstance(exc, ValueError) else 'agent_memory_source_invalid'
        except Exception:
            code = 'agent_memory_unavailable'
        # 管理 IO 后可能撤权，不能承诺没有改动；不自动重试，按id回读确认。
        return tool_result(self.tool_type, {'code': code, 'changes_may_have_occurred': self.write}, error=code)

    def agent_verify_result(self, *, receipt, context):
        try:
            task, binding, _ = self.memory.current(context, tool=self.tool_type)
            proof = {k: receipt.get(k) for k in ('memory_query', 'memory_version')}
            signature = hashlib.sha256(json.dumps(proof, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            valid = (self.observations.verify(receipt, tool=self.tool_type, context=context)
                and receipt.get('observation', {}).get('self_state', {}).get('data_sha256') == signature)
            # 原账本只存读哈希，不再复制大正文；若传入含正文的完整回执，
            # 同时验证其内容，防止保留签名却替换展示结果。写回执始终有完整小正文。
            if self.write or 'items' in receipt:
                valid = valid and canonical_hash({k: v for k, v in receipt.items()
                    if k not in {'observation', 'memory_version', 'verified_effects', 'completion_targets'}}) == receipt.get('memory_version')
            args = self.normalize_call(receipt.get('memory_query'))
            if not valid or args is None:
                return {'result_valid': False}
            if self.write:
                valid = (receipt.get('verified_effects') == ['memory_manage']
                    and receipt.get('completion_targets') == [receipt.get('id')])
                row = self.memory.store.get(self.memory.requester, binding, receipt.get('id'))
                valid = valid and (row is None if receipt.get('status') == 'forgotten'
                    else row is not None and row['version'] == receipt.get('version'))
            else:
                fresh = {**self._read(args, context), 'memory_query': args}
                valid = canonical_hash(fresh) == receipt.get('memory_version')
            return {'result_valid': valid}
        except Exception:
            return {'result_valid': False}


def install_memory_tools(service, memory):
    """显式宿主安装；原策略先判断，写工具不能因新注册绕过只读模式。"""
    tools = [AgentMemoryTool(memory), AgentMemoryTool(memory, write=True)]
    registry = service.executor.tool_host.registry
    if any(registry.get(t.tool_type) is not None for t in tools):
        raise AgentBoundaryError('agent_memory_tool_conflict')
    for tool in tools:
        registry.register(tool)
    original = service.authorize
    def authorize(scope, task, operation):
        grant = original(scope, task, operation)
        if grant is None or scope.requester != memory.requester or not scope.session.startswith('angelos:'):
            return grant
        names = {t.tool_type for t in tools if grant.policy.accepts(t.tool_type)
                 and (not t.write or grant.permission_mode in {'full','ultra'})}
        return replace(grant, allowed_tools=grant.allowed_tools | names,
            policy=replace(grant.policy, risk_by_tool={t.tool_type: 'low' for t in tools} | grant.policy.risk_by_tool))
    service.authorize = authorize
