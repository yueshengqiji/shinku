"""Angelos 专用记忆编排：独立存储/策略，原任务与工作台提供可重验来源。

没有 Bot MemoryService/Router 依赖，没有全局缓存，没有模型调用。宿主实例
持有连接边界；模型只能调用有界工具，不能传身份、工作区、路径或数据库。
经验同步幂等且可重试；忘记墓碑优先，历史展示不能让已忘记的记忆复活。
"""
import json

from shinku.agent_runtime.contracts import AgentBoundaryError, AgentScope, canonical_hash
from shinku.agent_runtime.information import safe_text
from shinku.agent_runtime.task_memory import task_memory
from shinku.angelos.task_continuity import TaskContinuity, workspace_binding, turn_workspace, task_turn
from .store import AgentMemoryStore

TERMINAL = {'completed', 'partial', 'failed', 'blocked', 'canceled'}


def checked_result(task):
    """复验 canonical seal 和生成版本，不把工作台展示副本当权威事实。"""
    from shinku.result_expression.facts import validate_result
    result = task.get('execution_result')
    error = validate_result(result, AgentScope(task['requester'], task['session']))
    if error or (result.get('task_id') != task['task_id']
            or result.get('attempt') != task.get('execution_attempt', 0)
            or not 1 <= result.get('version', 0) <= task.get('revision', 0)
            or result.get('version') != task.get('execution_result_version', result.get('version'))
            or result['execution']['status'] != task.get('status')):
        raise AgentBoundaryError('agent_memory_result_invalid')
    return result


class AgentMemoryService:
    def __init__(self, service, requester, root, *, policy=None):
        self.tasks, self.requester = service, requester
        self.store = AgentMemoryStore(root, policy=policy)
        self.continuity = None

    def bind(self, workbench):
        if self.continuity is not None and self.continuity.workbench.path != workbench.path:
            raise AgentBoundaryError('agent_memory_workbench_conflict')
        self.continuity = TaskContinuity(self.tasks, workbench, self.requester)

    def current(self, context, *, tool=None):
        from shinku.agent_runtime.conversation import conversation_turn
        turn = conversation_turn(context)
        if turn is None or turn.service is not self.tasks or turn.scope.requester != self.requester:
            raise AgentBoundaryError('agent_memory_host_binding_invalid')
        task, grant = turn.current(context, read_only=True)
        if tool and (tool not in grant.allowed_tools or not grant.policy.accepts(tool)):
            raise AgentBoundaryError('agent_memory_not_authorized')
        if self.continuity is None:
            raise AgentBoundaryError('agent_memory_workbench_unavailable')
        conversation, binding = workspace_binding(self.continuity.workbench, task['session'])
        source = task_turn(conversation, task)
        if source is None or turn_workspace(source) != binding:
            raise AgentBoundaryError('agent_memory_workspace_changed')
        return task, binding, grant

    def _source(self, task, binding, *, kind, message_id=None):
        value = {'session': task['session'], 'task_id': task['task_id'], 'workspace': binding,
                 'attempt': task.get('execution_attempt', 0)}
        if kind == 'user_statement':
            request = next((r for r in task['requests'] if r.get('message_id') == message_id), None)
            if request is None:
                raise AgentBoundaryError('agent_memory_user_source_missing')
            value.update(message_id=message_id, input_sha256=canonical_hash(request))
        else:
            value['result_sha256'] = canonical_hash(task.get('execution_result'))
        return value

    def _resolve(self, row, binding):
        source = row['source']
        if source.get('workspace') != binding or self.continuity is None:
            raise AgentBoundaryError('agent_memory_workspace_changed')
        conversation, latest_binding = workspace_binding(self.continuity.workbench, source['session'])
        turn = next((t for t in conversation['turns'] if t.get('task_id') == source['task_id']), None)
        if latest_binding != binding or turn is None or turn_workspace(turn) != binding:
            raise AgentBoundaryError('agent_memory_source_workspace_changed')
        if row['kind'] == 'user_statement':
            # 长期陈述回查工作台原输入，不依赖短期任务TTL；不能把源任务过期
            # 误当成用户已撤回。原输入删除/修改或工作区变更仍立即失效。
            inputs = [{'message_id': turn['id'], 'text': turn['text']}] + [
                {'message_id': s.get('id'), 'text': s.get('text')} for s in turn.get('supplements', [])]
            request = next((r for r in inputs if r.get('message_id') == source['message_id']), None)
            if request is None or canonical_hash(request) != source['input_sha256'] or row['text'] not in request['text']:
                raise AgentBoundaryError('agent_memory_source_changed')
            return None
        task = self.continuity.read_task(source['session'], source['task_id'])
        checked_result(task)
        if task.get('execution_attempt', 0) != source.get('attempt'):
            raise AgentBoundaryError('agent_memory_source_attempt_changed')
        if (task.get('status') not in TERMINAL or not task.get('execution_result')
                or canonical_hash(task['execution_result']) != source.get('result_sha256')):
            raise AgentBoundaryError('agent_memory_source_changed')
        return task

    def recall(self, context, query):
        task, binding, _ = self.current(context, tool='read_agent_memory')
        candidates = self.store.search(self.requester, binding, query)
        rows, gaps = [], []
        for row in candidates:
            try:
                self._resolve(row, binding)
                rows.append(row)
            except (AgentBoundaryError, KeyError, ValueError) as exc:
                gaps.append({'id': row['id'], 'reason': getattr(exc, 'code', 'agent_memory_source_invalid')})
            if len(rows) >= self.store.policy.max_results:
                break
        self.current(context, tool='read_agent_memory')
        return {'items': rows, 'gaps': gaps, 'candidate_limit': self.store.policy.candidate_limit,
                'candidate_count': len(candidates), 'coverage': 'bounded_text_retrieval_not_exhaustive',
                'source': 'agent_only_memory', 'is_execution_evidence': False, 'grants_permissions': False}

    def manage(self, context, arguments):
        with self.tasks._lock:
            return self._manage(context, arguments)

    def _manage(self, context, arguments):
        task, binding, grant = self.current(context, tool='manage_agent_memory')
        if task.get('contract') is None:
            raise AgentBoundaryError('agent_memory_contract_required')
        from .intent import explicit_management
        if not explicit_management(task['requests'][-1]['text'],arguments['operation']):
            raise AgentBoundaryError('agent_memory_explicit_request_required')
        # 不按“记住”等关键词触发；当前正常循环选择管理工具，宿主仍校验权限、
        # 契约与用户原文。存的是有归属的陈述，绝不认证陈述内容为外部事实。
        if arguments['operation'] == 'remember':
            text, message_id = arguments['text'], arguments['message_id']
            if message_id != task['requests'][-1].get('message_id'):
                raise AgentBoundaryError('agent_memory_current_user_quote_required')
            request = next((r for r in task['requests'] if r.get('message_id') == message_id), None)
            if request is None or text not in request['text'] or safe_text(request['text']) != request['text']:
                raise AgentBoundaryError('agent_memory_user_quote_required')
            source = self._source(task, binding, kind='user_statement', message_id=message_id)
            result = self.store.put(principal=self.requester, workspace=binding,
                key=canonical_hash(['user', task['task_id'], message_id]),
                # 模型片段只定位原话，实际保留整条用户输入，防止摘去否定、
                # 条件或引用归属。原输入超过上限明确拒绝，不截成永久偏好。
                kind='user_statement', text=request['text'], source=source)
        else:
            result = self.store.forget(self.requester, binding, arguments['id'])
        self.current(context, tool='manage_agent_memory')
        return {**result, 'source': 'agent_only_memory', 'is_execution_evidence': False,
                'grants_permissions': False}

    def capture(self, cid, turn):
        """只在工作台已取得 canonical result 后索引，失败不改任务状态/交付。"""
        if self.continuity is None or not turn.get('result') or not turn.get('task_id'):
            return
        session = 'angelos:' + cid
        _, binding = workspace_binding(self.continuity.workbench, session)
        if turn_workspace(turn) != binding:
            raise AgentBoundaryError('agent_memory_workspace_changed')
        task = self.continuity.read_task(session, turn['task_id'])
        if task.get('status') not in TERMINAL or not task.get('execution_result'):
            return
        checked_result(task)
        # 工作台旧副本不能覆盖新尝试；索引内容只从实时账本生成。
        if canonical_hash(turn['result']) != canonical_hash(task['execution_result']):
            raise AgentBoundaryError('agent_memory_result_changed')
        view = task_memory(task, max_calls=3)
        text = safe_text(json.dumps({k: view[k] for k in (
            'goal', 'status', 'reason', 'recorded_effects', 'verification', 'report_delivery')}, ensure_ascii=False))[:4096]
        fresh = self.tasks._grant(AgentScope(self.requester,session),task,'inspect')
        if 'read_agent_memory' not in fresh.allowed_tools or not fresh.policy.accepts('read_agent_memory'):
            raise AgentBoundaryError('agent_memory_not_authorized')
        return self.store.put(principal=self.requester, workspace=binding, key=task['task_id'],
            kind='task_experience', text=text, source=self._source(task, binding, kind='task_experience'))

    def snapshot(self, context):
        task, binding, grant = self.current(context)
        # 显式禁止读取工具时，自动上下文也不能绕过；当前任务的原账本仍由原入口提供。
        if 'read_agent_memory' not in grant.allowed_tools or not grant.policy.accepts('read_agent_memory'):
            return {'status': 'not_authorized', 'grants_permissions': False}
        continuity = self.continuity.recent(task)
        prior = continuity['candidates']
        continuity['candidates'] = [r for r in prior if not self.store.source_forgotten(
            self.requester,binding,task['session'],r['task_id'])]
        continuity['omitted_forgotten_sources'] = len(prior)-len(continuity['candidates'])
        self.current(context,tool='read_agent_memory')  # IO期间取消/换工作区/撤权不能交出迟到内容。
        return {'status': 'available', 'current_task': task_memory(task, max_calls=4),
            'continuity': continuity,
            'long_term': {'tool': 'read_agent_memory', 'scope': 'agent_principal_and_workspace',
                'automatic_retrieval': False},
            'rule': '先核对原账本；候选和历史不确定时澄清对象，不因同名合并产品，不重放旧副作用。',
            'grants_permissions': False, 'is_execution_evidence': False}

    def focused_history(self, context, raw):
        """忘记只停止自动供给；不删用户原会话/任务审计，明确回读仍走原权限。"""
        task,binding,_ = self.current(context)
        try:
            data = json.loads(raw)
        except (ValueError,TypeError):
            return self.continuity.focused_history(raw,self.store.policy)
        if not isinstance(data,dict) or not isinstance(data.get('messages'),list):
            return raw
        conversation,_ = workspace_binding(self.continuity.workbench,task['session'])
        originals = {t['id']:t for t in conversation['turns']}
        for row in data['messages']:
            original = originals.get(row.get('turn_id')) if isinstance(row,dict) else None
            if (original and row.get('user') == original.get('text') and original.get('task_id')
                    and self.store.source_forgotten(self.requester,binding,task['session'],original['task_id'])):
                row.clear();row.update(user='[已从Agent记忆撤销自动供给的历史输入；原会话审计保留]',
                                       assistant='',supplements=[],memory_withdrawn=True)
        return self.continuity.focused_history(json.dumps(data,ensure_ascii=False),self.store.policy)
