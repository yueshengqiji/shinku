"""执行方交付投影：正文与证据分层，不追加总结模型，也不改变执行判定。

完整规范结果留在原任务库；有效的同轮交接正文可替代执行报告的传输正文，
完整交付物及目标、限制和未完成项保留。详情引用绑定当前结果哈希，读取仍须授权。
"""
import copy

from .contracts import AgentBoundaryError, canonical_hash


def details_hash(result):
    return result.get('presentation', {}).get('details_result_hash') or result['result_hash']


def diagnostic_invocations(result):
    """已有执行正文时，自查快照留在详情；按真实工具来源识别，不匹配用户关键词。"""
    if not result.get('execution_report', {}).get('text'):
        return set()
    return {row.get('invocation_id') for row in result.get('evidence', [])
            if row.get('tool') == 'read_self_state' and row.get('source') == 'executor_receipt'
            and isinstance(row.get('invocation_id'), str) and row['invocation_id']}


def delivery_result(result):
    """生成独立校验的交付视图；不得把模型报告提升为已核验事实。"""
    source = copy.deepcopy(result)
    original_hash = source.pop('result_hash')
    if canonical_hash(source) != original_hash:
        raise AgentBoundaryError('delegation_result_binding_invalid')
    diagnostic = diagnostic_invocations(source)
    source['facts'] = [row for row in source.get('facts', []) if row.get('invocation_id') not in diagnostic]
    # 失败/未知回执和HTTP错误仍是必要限制，只有正常成功回执折叠到详情。
    source['evidence'] = [row for row in source.get('evidence', [])
        if row.get('execution_status') != 'ok' or
        (type(row.get('receipt', {}).get('http_status')) is int and row['receipt']['http_status'] >= 400)]
    source['budget'] = {}
    source['verification'].pop('evidence', None)
    source['presentation'] = {**source.get('presentation', {}),
        'handoff_version': 1, 'details_result_hash': original_hash,
        'optional_success_receipts': True}
    from .delivery_brief import select_brief
    brief = select_brief(result)
    if brief is not None:
        source['execution_report'] = {**source['execution_report'], 'text': brief['text']}
        source['presentation'].update(body_source='same_turn_handoff',
            original_report_sha256=brief['report_sha256'])
    # 源信息缺口继续保留，不能因移除机器回执就声称已理解完整资料。
    source['result_hash'] = canonical_hash(source)
    return source


def message(task, kind, body, *, key, author, data=None):
    """宿主操作生成标签，用户正文永远不能自行产生授权标签。"""
    row = {'schema_version': 1, 'task_id': task['task_id'],
        'attempt': task.get('execution_attempt', 0), 'version': task.get('revision', 0),
        'kind': kind, 'author': author, 'body': body, 'data': data or {}, 'key': key}
    row['message_hash'] = canonical_hash(row)
    return row


class TaskConversation:
    """与原任务共用事务库的只追加消息记录；不负责执行、检索记忆或权限。"""
    def __init__(self, store):
        self.store = store
        with store._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS agent_task_messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                message_key TEXT NOT NULL, payload TEXT NOT NULL,
                UNIQUE(task_id,message_key))''')

    def append(self, task, kind, body, *, key, author, data=None):
        import json
        row = message(task, kind, body, key=key, author=author, data=data)
        with self.store._transaction() as db:
            existing = db.execute('SELECT payload FROM agent_task_messages WHERE task_id=? AND message_key=?',
                                  (task['task_id'], key)).fetchone()
            if existing:
                # 重放控制操作不覆盖原版本记录；业务幂等仍由原任务入口核对。
                old = json.loads(existing[0])
                if canonical_hash({k: v for k, v in old.items() if k != 'message_hash'}) != old['message_hash']:
                    raise AgentBoundaryError('task_message_integrity_failed')
                if (old['kind'], old['body'], old['author'], old['data']) != (kind, body, author, data or {}):
                    raise AgentBoundaryError('task_message_key_conflict')
                return
            db.execute('INSERT INTO agent_task_messages(task_id,message_key,payload) VALUES (?,?,?)',
                       (task['task_id'], key, json.dumps(row, ensure_ascii=False)))

    def read(self, task, *, after=0):
        import json
        if type(after) is not int or after < 0:
            raise AgentBoundaryError('invalid_message_cursor')
        with self.store._transaction() as db:
            rows = db.execute('SELECT seq,payload FROM agent_task_messages WHERE task_id=? AND seq>? ORDER BY seq LIMIT 50',
                              (task['task_id'], after)).fetchall()
        result = []
        for seq, raw in rows:
            row = json.loads(raw)
            if canonical_hash({k: v for k, v in row.items() if k != 'message_hash'}) != row['message_hash']:
                raise AgentBoundaryError('task_message_integrity_failed')
            result.append({'cursor': seq, **row})
        return {'messages': result, 'next_cursor': rows[-1][0] if rows else after}
