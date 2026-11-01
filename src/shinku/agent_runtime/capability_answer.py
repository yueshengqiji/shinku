"""独立 Agent 的纯工具能力自查：宿主选型、真实快照与建议分离。

准备入口只识别明确的自身工具问题，不从历史、附件或模型契约扩大权限。
执行入口复用 SelfStateTool 的回合绑定读取凭据及原 SmartCompletionVerifier；
自查仅开放状态读取，注册表不代表依赖已安装或外部服务可用。进程内观察器
由单个任务 context 持有；重启/新输入必须重读，不反序列化旧凭据。
"""
from __future__ import annotations

from dataclasses import replace
import re
import json
import unicodedata

from shinku.agent.self_state import SelfStateTool
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry
from shinku.tools.invocation import ToolInvocation
from .contracts import AgentBoundaryError


def self_tool_question(text):
    """保守识别纯自查；行动、外部研究、项目评估和多段输入不走快捷选型。"""
    if not isinstance(text, str) or not 1 <= len(text) <= 200:
        return False
    text = unicodedata.normalize('NFKC', text).strip().casefold()
    from shinku.agent.verification import SmartCompletionVerifier
    # “覆盖面/覆盖情况”在这里指能力范围，不是覆盖文件的操作。
    if SmartCompletionVerifier.request_requires_action(re.sub(r'覆盖(?:情况|面|范围)', '范围', text)):
        return False
    if re.search(r'你(?:觉得|认为)(?!(?:自己|你自己|自身))', text):
        return False
    if not re.search(r'自己|自身|angelos|安吉罗斯|你(?:目前|现在|当前|还|能|可|有|的|会|是否|缺|工具|能力)|your (?:available )?(?:tools|capabilities)|^(?:what|which|how many) tools (?:do|can) you|you.*(?:missing|lack)', text):
        return False
    if re.search(r'https?://|[/\\\n;；]|```|然后|顺便|顺带|帮我|再帮|不要|不用|禁止|不许|外部|服务器|项目|官方|联网|搜索|文档|最新|安装|卸载|启动|重启|拉起来|关闭|停止|执行|下载|修改配置|接入|连接到|删|把|并|创建文件|写入|发送|运行命令|同时|\b(?:without|and|then|install|execute|run|restart|delete|send|search|server|project|latest)\b', text):
        return False
    return bool(re.search(r'你|自身|自己|angelos|安吉罗斯|\b(?:you|your)\b', text)
                and re.search(r'工具|能力|mcp|\b(?:tools?|capabilities)\b', text)
                and re.search(r'多少|几(?:个|种|项)|哪些|什么|如何|是否|有无|有没有|缺|不足|能做|能用|可用|可调用|可以用|评估|覆盖|\b(?:what|which|how many|lack|missing|have|available)\b', text))


def capability_contract():
    """这是新任务的宿主契约，不改写任何已经保存的 research 契约。"""
    return {'goal': '依据当前宿主授权工具清单，说明工具覆盖与可能的能力缺口',
            'deliverables': ['当前工具覆盖说明与能力改进建议'],
            'scope': ['仅自查宿主工具注册与授权；建议不代表已验证服务健康或执行改动'],
            'requirements': [{'id': 'tool_snapshot', 'description': '取得本任务新鲜、完整且绑定当前请求的宿主工具快照',
                              'effect': 'tool:read_self_state', 'targets': [], 'checks': {}}]}


def capability_task(context):
    task = context.get('agent_task_state', {}) if isinstance(context, dict) else {}
    requests = task.get('requests', [])
    return (len(requests) == 1 and self_tool_question(requests[0].get('text'))
            and task.get('contract') == capability_contract())


class CapabilityStateTool(SelfStateTool):
    """只开放 tools 范围，并重验原回合凭据及当前授权，不能伪造成功读取。"""
    agent_observation_required_fields = ('self_state_receipt',)
    # 循环只持久化显式声明的观察字段，必须完整保留签发时的四项数据，
    # 否则完成门无法与 StateObservationStore 中的原回合凭据逐项核对。
    agent_observation_receipt_fields = ('self_state', 'self_state_receipt', 'result_valid', 'result_facts')

    def normalize_call(self, call):
        value = super().normalize_call(call)
        return value if value and value['scope'] == 'tools' else None

    def agent_verify_result(self, *, receipt, context):
        resolved = self.resolve_state_observation(updates=receipt, arguments={'scope': 'tools'}, context=context)
        current, source = self._read_snapshot(context, fresh=True)
        return {'result_valid': bool(resolved and resolved['valid'] and source != 'unavailable'
                                     and receipt.get('self_state') == current)}

    def observation(self, state):
        for call in reversed(state.tool_calls):
            if call.get('name') == self.tool_type and call.get('status') == 'ok' and call.get('executed') is True:
                updates = call.get('data', {}).get('state_updates', {})
                if self.agent_verify_result(receipt=updates, context=state.context)['result_valid']:
                    return updates.get('self_state', {}).get('tools')
                return None
        return None


def capability_observer(state):
    context = state.context if isinstance(state.context, dict) else {}
    observer = context.get('_capability_observer')
    return observer if capability_task(context) and isinstance(observer, CapabilityStateTool) else None


def capability_host(host, allowed, policy, context, grant):
    """宿主只为已选定的纯自查补入只读状态能力，绝不开放其他执行工具。"""
    name = 'read_self_state'
    if name in policy.blocked:
        raise AgentBoundaryError('capability_snapshot_not_authorized')
    observer = context.get('_capability_observer')
    if not isinstance(observer, CapabilityStateTool):
        if name in host.registry.names():
            raise AgentBoundaryError('capability_snapshot_tool_conflict')
        registered = set(host.registry.names()) | {name}
        available = {n for n in allowed if policy.accepts(n)} | {name}
        if context.get('agent_material_session') is not None and 'task_material_read' not in policy.blocked:
            registered.add('task_material_read')
            available.add('task_material_read')
        def snapshot():
            if not grant.still_authorized():
                return {'status': 'unavailable', 'reason': 'authorization_changed'}
            # 描述来自宿主已注册元数据，不能从工具名臆测参数或业务健康。
            # 不回传完整 schema，避免能力问答把全部工具参数再注入一遍。
            specs = host.registry.native_specs(allowed_tool_names=set(available))
            descriptions = {s['function']['name']: s['function']['description'] for s in specs}
            result = {'tools': {'available_count': len(available), 'available_names': sorted(available),
                'dependency_state': 'not_probed', 'business_test': 'not_run',
                'registered_names': sorted(registered),
                'approval_required_names': sorted(n for n in available if n != name and policy.requires_approval(n)),
                'scope': 'host_registry_and_current_authorization',
                'limits': '注册与授权不等于依赖已安装、外部MCP已连接或服务健康；本轮只读取状态，不执行其他工具。'}}
            # SelfStateTool 的真实凭据上限是 pretty JSON 8000 字符。可选描述不能
            # 挤掉完整工具名/授权快照，也不能因增补元数据破坏原完成验收。
            for limit in (160, 120, 80, 40, 0):
                result['tools']['descriptions'] = {n: descriptions[n][:limit] for n in sorted(descriptions)} if limit else {}
                result['tools']['description_chars_limit'] = limit
                result['tools']['descriptions_truncated'] = any(len(v) > limit for v in descriptions.values())
                if len(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)) <= 7800:
                    break
            return result
        observer = CapabilityStateTool(fresh_provider=snapshot)
        context['_capability_observer'] = observer
    # 材料包装在此之后添加，保留原资料守卫与完成门；底层执行工具仍在快照
    # 中供评估，但本自查的派发权限只有状态工具和宿主资料读取工具。
    tools = dict(host.registry.view())
    tools[name] = observer
    scoped_policy = replace(policy, allowed=frozenset({name}),
                            risk_by_tool={**policy.risk_by_tool, name: 'low'})
    return ToolHost(ToolRegistry(tools), host_id=host.host_id), {name}, scoped_policy


def capability_first_decision():
    from shinku.agent.loop import AgentDecision
    return AgentDecision.tool(ToolInvocation('read_self_state', {'scope': 'tools'}))


def capability_input_complete(session, kwargs):
    """预判本次实际输入是否包含最后缺页，只控制工具选择，不签发完成。

    仍由 mark_model_input 在真实模型请求返回后标记提供，并由原资料门禁
    再核验；调用异常、缺页、失效或撤权均不能靠这个预判取得完成回执。
    """
    info = session.information()
    if any(g.get('required', True) and g.get('reason') != 'material_read_pending' for g in info['gaps']):
        return False
    prompts = [kwargs['user_prompt'], *(r.get('content', '') for r in kwargs.get('native_history') or [])]
    receipts = session.store.read_receipts(session._bound_task())
    for item in session.prompt_package().get('items', []):
        if not item.get('required', True):
            continue
        ranges = [(0, item['content_chars'] - item['omitted_chars'])]
        for receipt in receipts:
            if receipt['reference'] != item['reference']:
                continue
            if receipt.get('provided', True) or any(json.dumps(receipt, ensure_ascii=False) in text for text in prompts):
                ranges.append((receipt['offset'], receipt['offset'] + receipt['returned_chars']))
        end = 0
        for start, stop in sorted(ranges):
            if start > end:
                break
            end = max(end, stop)
        if end < item['content_chars']:
            return False
    return True


def capability_report_prompt(state, snapshot):
    """取证结束后独立成稿，只输入原目标、用户背景索引及本轮工具事实。

    旧助手原文和原生消息留在任务与资料库，不作为新答案的草稿模板；
    本轮资料必须先经真实请求提供，不能用这份导航替代原文读取条件。
    """
    from .answer_guidance import answer_context, task_background_sources
    task = state.context.get('agent_task_state', {})
    return json.dumps({'requests': task.get('requests', []), 'contract': task.get('contract'),
        'confirmed_constraints': task.get('confirmed_constraints', []),
        'answer_context': answer_context(task_background_sources(task)),
        'current_tool_observation': snapshot}, ensure_ascii=False)
