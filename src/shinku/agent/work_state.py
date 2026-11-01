"""持久工作计划的数据边界；任务生命周期和验收仍归原 Service/TaskStore。

阅读顺序：validate_plan 校验模型提案，Service 绑定真实引用，TaskStore 在
原任务 CAS 事务中追加修订。计划、发现和步骤 done 都不是执行回执。
修订表只保存有界导航数据，原请求、资料和工具输出继续留在原账本。
"""
from __future__ import annotations

import copy
import hashlib
import json
import re


MAX_REVISIONS = 64
MAX_PLAN_BYTES = 24576


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def input_binding(task):
    """观察写盘不使计划过期；请求、约束、契约、材料版本或授权变化才使其待复核。"""
    value = {k: task.get(k) for k in ('task_id', 'requester', 'session',
        'requests', 'confirmed_constraints', 'contract', 'host_grant', 'source_data')}
    package = task.get('material_package') or {}
    # 执行回执不断追加到同一材料包，但不是新用户输入；否则每用一次工具都会
    # 被误标为输入变化，诱使模型反复重写计划。读取用量同样不参与版本绑定。
    value['inputs'] = [{k: row.get(k) for k in ('reference', 'source_id', 'kind',
        'version', 'content_sha256', 'complete', 'required')}
        for row in package.get('items', []) if row.get('kind') != 'executor_result']
    value['required_gaps'] = [row for row in package.get('gaps', []) if row.get('required', True)]
    return digest(value)


def valid_step_order(row, earlier):
    """与旧 StageController 共享有向无环依赖约束；不购买旧规划请求。"""
    return (isinstance(row.get('id'), str)
        and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}', row['id']) is not None
        and row['id'] not in earlier and isinstance(row.get('depends'), list)
        and all(isinstance(v, str) for v in row['depends'])
        and len(set(row['depends'])) == len(row['depends'])
        and set(row['depends']) <= earlier)


def validate_plan(value, task, *, advisory=False):
    from .task_state import TaskContractError, redact
    def fail(code='work_plan_invalid'):
        raise TaskContractError(code)
    def text(v, limit=1000, empty=False):
        if not isinstance(v, str) or len(v) > limit or (not empty and not v.strip()):
            fail()
    def texts(v):
        if not isinstance(v, list) or len(v) > 12:
            fail()
        for item in v:
            text(item)
    fields = {'reason', 'current_step', 'next_action', 'blocker', 'findings', 'hypotheses', 'steps'}
    if not isinstance(value, dict) or set(value) - {'objects', 'decisions', 'ruled_out'} != fields:
        fail()
    text(value['reason']); text(value['current_step'], 32, empty=True)
    text(value['next_action'], empty=True); text(value['blocker'], empty=True)
    texts(value['findings']); texts(value['hypotheses'])
    # 可选工作笔记只追加到现有修订，不改变旧提案；决定/排除是模型导航，
    # 不因字段名称就变成用户要求、已验证事实或额外验收条件。
    for key in ('decisions', 'ruled_out'):
        if key in value:
            texts(value[key])
    objects = value.get('objects', [])
    if not isinstance(objects, list) or len(objects) > 8:
        fail('work_plan_objects_invalid')
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {'name', 'candidates', 'conflicts'}:
            fail('work_plan_objects_invalid')
        text(obj['name'], 120)
        texts(obj['candidates']); texts(obj['conflicts'])
        # 身份候选只是模型导航，不能以一个 confirmed 字段制造“已确认事实”。
    if not isinstance(value['steps'], list) or not 1 <= len(value['steps']) <= 12:
        fail()
    expected = {r['id'] for r in (task.get('contract') or {}).get('requirements', [])}
    ids, covered, declared_done = set(), set(), set()
    for row in value['steps']:
        if (not isinstance(row, dict) or set(row) != {'id', 'goal', 'depends', 'requirements', 'status', 'references'}
                or not valid_step_order(row, ids) or not isinstance(row['status'], str)
                or row['status'] not in {'pending', 'in_progress', 'done', 'blocked'}):
            fail()
        if row['status'] in {'in_progress', 'done'} and not set(row['depends']) <= declared_done:
            fail('work_plan_dependency_not_ready')
        if row['status'] == 'done':
            declared_done.add(row['id'])
        ids.add(row['id']); text(row['goal'])
        reqs = row['requirements']
        if (not isinstance(reqs, list) or any(not isinstance(r, str) for r in reqs)
                or len(set(reqs)) != len(reqs) or not advisory and set(reqs) - expected):
            fail('work_plan_unknown_requirement')
        covered.update(reqs)
        refs = row['references']
        if not isinstance(refs, list) or len(refs) > 8:
            fail()
        for ref in refs:
            if (not isinstance(ref, dict) or set(ref) != {'kind', 'id'} or not isinstance(ref['kind'], str)
                    or ref['kind'] not in {'tool_call', 'material'}):
                fail('work_plan_reference_invalid')
            text(ref['id'], 256)
    if value['current_step'] and value['current_step'] not in ids:
        fail('work_plan_current_step_unknown')
    if not advisory and covered != expected:
        fail('work_plan_omits_requirement')
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_PLAN_BYTES or redact(value) != value:
        fail('work_plan_oversized_or_sensitive')
    return copy.deepcopy(value)


def init_table(db):
    # 增量表不改旧 payload 格式。旧版本可忽略导航索引，但不能删除历史修订。
    db.execute('''CREATE TABLE IF NOT EXISTS agent_task_work_revisions (
        task_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY(task_id, revision))''')


def persist(db, safe, previous, record):
    """只在原任务写事务内调用；指针和修订一起提交，冲突/失败一起回滚。"""
    from .task_state import TaskContractError, redact
    old = (previous or {}).get('work_state')
    if safe.get('work_state') != old:
        raise TaskContractError('work_state_pointer_changed')
    if record is None:
        return
    revision = (old or {}).get('revision', 0) + 1
    if revision > MAX_REVISIONS:
        raise TaskContractError('work_plan_revision_limit')
    if (previous is None or record.get('schema_version') != 1 or record.get('revision') != revision
            or record.get('task_id') != safe['task_id'] or record.get('requester') != safe['requester']
            or record.get('session') != safe['session'] or record.get('attempt') != safe.get('execution_attempt')
            or record.get('task_version') != safe['revision'] or record.get('binding') != input_binding(safe)
            or redact(record) != record):
        raise TaskContractError('work_state_revision_invalid')
    # 可选计划只放开模型条件的覆盖关系，修订CAS、作用域、依赖与引用格式不变。
    advisory = record.get('purpose') == 'advisory' and safe.get('input_mode') == 'conversation'
    validate_plan(record['plan'], safe, advisory=advisory)
    sealed = copy.deepcopy(record)
    sealed['sha256'] = digest(record)
    db.execute('INSERT INTO agent_task_work_revisions VALUES (?, ?, ?)',
        (safe['task_id'], revision, json.dumps(sealed, ensure_ascii=False, separators=(',', ':'))))
    safe['work_state'] = {'revision': revision, 'sha256': sealed['sha256']}


def read(db, task, revision=None):
    from .task_state import TaskContractError
    pointer = task.get('work_state')
    if pointer is None:
        return None
    latest = pointer.get('revision') if isinstance(pointer, dict) else None
    if type(latest) is not int or not 1 <= latest <= MAX_REVISIONS:
        raise TaskContractError('work_state_revision_invalid')
    revision = latest if revision is None else revision
    if type(revision) is not int or not 1 <= revision <= latest <= MAX_REVISIONS:
        raise TaskContractError('work_state_revision_invalid')
    row = db.execute('SELECT payload FROM agent_task_work_revisions WHERE task_id=? AND revision=?',
        (task['task_id'], revision)).fetchone()
    record = json.loads(row[0]) if row else {}
    if (record.get('revision') != revision or any(record.get(k) != task.get(k) for k in ('task_id', 'requester', 'session'))
            or digest({k: v for k, v in record.items() if k != 'sha256'}) != record.get('sha256')
            or revision == latest and record.get('sha256') != pointer.get('sha256')):
        raise TaskContractError('work_state_integrity_failed')
    return record


def compact_plan_calls(db, task):
    """任务快照仅存已落盘计划的可回读索引，原提案保留在修订表。

    只投影与真实成功修订逐字匹配的内部参数，不碰失败原稿、其他工具、
    进程内原生协议或原始请求；发现冲突就回滚本次任务写入。
    """
    from .task_state import TaskContractError
    for call in task.get('tool_calls', []):
        args = call.get('arguments', {})
        receipt = call.get('data', {}).get('state_updates', {})
        if (call.get('name') != 'update_work_plan' or call.get('status') != 'ok'
                or receipt.get('status') != 'saved' or not isinstance(args.get('plan'), dict)):
            continue
        record = read(db, task, receipt.get('revision'))
        if (not record or record['sha256'] != receipt.get('sha256')
                or args.get('expected_revision') != record['revision'] - 1 or args['plan'] != record['plan']):
            raise TaskContractError('work_plan_receipt_conflict')
        call['arguments'] = {'expected_revision': args['expected_revision'],
            'plan_reference': {'revision': record['revision'], 'sha256': record['sha256']}}
