"""执行模型的主动整理请求；只写压缩器的进程内请求，不拥有资料或权限。"""
from collections.abc import Mapping
from types import SimpleNamespace

from shinku.llm.native_messages import NativeProtocolError
from shinku.tools.workspace import tool_result
from .contracts import AgentBoundaryError


class ContextCompactionTool:
    tool_type = 'request_context_compaction'
    agent_permission_class = 'read'
    agent_progress_class = 'observe'
    agent_execution_evidence = False

    def __init__(self, turn):
        self.turn = turn

    def build_prompt_instruction(self):
        from shinku.llm.context_awareness import GUIDANCE
        return GUIDANCE

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type':'object','additionalProperties':False,
            'properties':{'reason':{'type':'string','minLength':1,'maxLength':600}},'required':['reason']})

    def normalize_call(self, call):
        if not isinstance(call, Mapping):
            return None
        args = {k:v for k,v in call.items() if k!='type' and not str(k).startswith('_tool_') and k!='_native_tool_call'}
        return args if set(args)=={'reason'} and isinstance(args['reason'],str) and 0<len(args['reason'].strip())<=600 else None

    def execute(self, *, call, context):
        try:
            args = self.normalize_call(call)
            if args is None:
                raise AgentBoundaryError('context_compaction_arguments_invalid')
            task, grant = self.turn.current(context)
            if not grant.policy.accepts(self.tool_type):
                raise AgentBoundaryError('context_compaction_not_authorized')
            runtime = self.turn.service.executor.profile.runtime
            result = runtime.request_context_compaction(task['task_id'], args['reason'])
            return tool_result(self.tool_type, result)
        except (AgentBoundaryError, NativeProtocolError) as ex:
            return tool_result(self.tool_type, {'code':str(ex),'grants_permissions':False,
                'is_execution_evidence':False}, error=str(ex))
