"""同会话连续性只读取原任务，不按关键词自动恢复任务或继承授权。

最近任务是候选，不是“用户一定指它”的判决。所有资料绑定原输入的工作区
快照；工作区变更后拒绝旧绑定。原任务过期时显式列缺口，不清理原件。
"""
import json

from shinku.agent_runtime.contracts import AgentScope, canonical_hash, AgentBoundaryError
from shinku.agent_runtime.task_memory import task_memory


def workspace_binding(store, session):
    if not session.startswith('angelos:'):
        raise AgentBoundaryError('agent_memory_scope_invalid')
    conversation = store.read('conversation', session[len('angelos:'):])
    workspace = store.read('workspace', conversation['workspaceId'])
    value = {k: workspace.get(k, '') for k in ('id', 'path', 'description')}
    return conversation, canonical_hash(value)


def turn_workspace(turn):
    snapshot = turn.get('workspace_snapshot')
    if not isinstance(snapshot, dict) or not snapshot.get('id'):
        return None
    return canonical_hash({k: snapshot.get(k, '') for k in ('id', 'path', 'description')})


def task_turn(conversation, task):
    """submit 可先启动 worker；用已落盘输入绑定，不能因 task_id 展示指针迟到丢首轮。"""
    requests = task.get('requests') or []
    if not requests:
        return None
    key = requests[0].get('message_id')
    expected = 'agent:' + canonical_hash([task['requester'], task['session'], key])[:32]
    return next((t for t in conversation['turns'] if t.get('task_id') == task['task_id'] or
        (t.get('task_id') is None and expected == task['task_id'] and t.get('id') == key
         and t.get('text') == requests[0]['text'])), None)


class TaskContinuity:
    def __init__(self, service, workbench, requester):
        self.service, self.workbench, self.requester = service, workbench, requester

    def read_task(self, session, task_id):
        page = self.service.task_store.read_scoped_page(
            requester=self.requester, session=session, task_id=task_id, limit=1)
        if not page['items']:
            raise AgentBoundaryError('agent_memory_source_unavailable')
        task = page['items'][0]
        self.service._grant(AgentScope(self.requester, session), task, 'inspect')
        return task

    def recent(self, current, *, limit=2):
        conversation, binding = workspace_binding(self.workbench, current['session'])
        current_turn = task_turn(conversation, current)
        if current_turn is None or turn_workspace(current_turn) != binding:
            raise AgentBoundaryError('agent_memory_workspace_changed')
        rows, gaps = [], []
        preceding = conversation['turns'][:conversation['turns'].index(current_turn)]
        query = current['requests'][-1]['text']
        explicit = [t for t in preceding if t.get('task_id') and t['task_id'] in query]
        # 明确ID优先，只选同会话绑定；字面出现ID只是读取导航，不是恢复授权。
        selected = list(reversed(explicit)) if explicit else list(reversed(preceding[-8:]))
        for turn in selected:
            if len(rows) + len(gaps) >= limit:
                break
            if not turn.get('task_id') or turn_workspace(turn) != binding:
                continue
            try:
                old = self.read_task(current['session'], turn['task_id'])
            except AgentBoundaryError as exc:
                gaps.append({'task_id': turn['task_id'], 'reason': exc.code})
                continue
            if (old.get('execution_result', {}).get('verification', {}).get('layers', {}).get('policy')
                    == 'conversation_reply' and not explicit):
                continue
            rows.append(task_memory(old, max_calls=4))
        return {'candidates': rows, 'gaps': gaps, 'selection': 'explicit_same_conversation_ids' if explicit
                    else 'recent_same_conversation_candidates_only',
                'scanned_turn_limit': None if explicit else 8,
                'automatic_resume': False, 'workspace_binding': binding}

    def focused_history(self, raw, policy):
        """只改变输入投影，不改原历史；语义不确定时保留最近轮次和原件回读导航。"""
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return raw  # 旧纯文本协议由原 history_messages 处理，不能猜测其结构。
        if not isinstance(value, dict) or not isinstance(value.get('messages'), list):
            return raw
        messages = value['messages']
        selected = messages[-policy.max_history_turns:]
        projected, shortened = [], 0
        for index, row in enumerate(selected):
            # 最近三轮保留原答复；更早轮次只给原文摘录及回读导航，避免旧解释
            # 占满输入。按位置投影不声称语义无关，不重写或删除原会话。
            if (index < len(selected) - 3 and isinstance(row, dict)
                    and isinstance(row.get('assistant'), str) and len(row['assistant']) > 320):
                row = {**row, 'assistant': row['assistant'][:320],
                       'assistant_omitted_chars': len(row['assistant']) - 320,
                       'assistant_excerpt_only': True}
                shortened += 1
            projected.append(row)
        if len(messages) <= policy.max_history_turns and not shortened:
            return raw
        return json.dumps({**value, 'messages': projected,
            'history_projection': {'mode': 'agent_recent_dialogue',
                'omitted_turns': max(0, len(messages)-policy.max_history_turns),
                'shortened_assistant_turns': shortened,
                'full_history_retained': True,
                'retrieval': {'tool': 'read_task_records', 'mode': 'inputs'},
                'note': '较早答复只作原文摘录，不能因省略判定不存在或否认旧结论；'
                        '需要旧答复细节时按turn_id用read_task_records(mode=input)回读。'
                        '最新状态取当前回执，历史助手判断不等于已验证事实。'}}, ensure_ascii=False)
