"""有明确对象的记忆操作收尾，不能用记忆写入替代文件/命令效果。

只识别完整的窄声明，行为本身从不由此触发；必须回查当前宿主工具签名、
操作类型和本库版本。混入安装、文件路径、配置或其他行动的句子走原验收。
"""
import re


def verified_memory_clause(state, clause):
    match = re.fullmatch(r'\s*(?:我|本轮)?(?:已经|已)(?:成功)?(?P<op>保存|记录|记住|忘记|删除|移除)'
        r'(?:了)?(?:你的原话|这条原话|该条原话|这条记忆|该条记忆|到\s*Agent\s*记忆库)\s*', clause, re.I)
    if not match:
        return False
    from shinku.agent_runtime.conversation import conversation_turn
    turn = conversation_turn(state.context)
    memory = getattr(turn.service, 'agent_memory', None) if turn is not None else None
    if memory is None:
        return False
    handler = turn.handlers.get('manage_agent_memory')
    if handler is None:
        return False
    operation = 'forget' if match['op'] in {'忘记', '删除', '移除'} else 'remember'
    for call in reversed(state.tool_calls):
        if call.get('name') != 'manage_agent_memory' or call.get('status') != 'ok' or call.get('executed') is False:
            continue
        receipt = (call.get('data') or {}).get('state_updates') or {}
        if receipt.get('memory_query', {}).get('operation') != operation:
            continue
        return handler.agent_verify_result(receipt=receipt, context=state.context).get('result_valid') is True
    return False
