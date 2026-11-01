"""恢复时重读宿主事实：不重放工具，不修改旧回执，不代表最终验收通过。

进程状态由执行器只读钩子认证；文件变化只在对话接续中降为重验提示。
严格契约、未知副作用、撤权和无真实读取结果的路径仍交给原恢复门拒绝。
"""
import copy
import re
from collections.abc import Mapping

from shinku.agent.execution_sessions import execution_event


def process_context(call, context):
    """按旧调用的执行尝试恢复身份，不能借用另一任务或未来尝试的会话。"""
    binding = call.get('execution_binding')
    task = context.get('agent_task_state', {})
    if not isinstance(binding, Mapping):
        return None
    if (any(binding.get(k) != task.get(k) for k in ('task_id', 'requester', 'session'))
            or type(binding.get('attempt')) is not int
            or not 0 < binding['attempt'] <= task.get('execution_attempt', 0)
            or (binding.get('work_id') and (type(binding.get('work_attempt')) is not int
                                          or binding['work_attempt'] < 1))):
        return None
    return {**context, 'agent_task_state': {**task, 'execution_attempt': binding['attempt']},
            'agent_work_id': binding.get('work_id', ''), 'agent_work_attempt': binding.get('work_attempt', 0)}


def observe_process(call, handler, context):
    """只接受已注册执行器认证的观测，模型参数本身不能补造缺失的终态回执。"""
    data = call.get('data') or {}
    old = data.get('state_updates', {}) if isinstance(data, Mapping) else {}
    session = old.get('execution_session') if isinstance(old, Mapping) else None
    if call.get('executed') is not True or not (
            isinstance(session, Mapping) and session.get('status') == 'running'
            or call.get('status') == 'error' and not session):
        return call, None
    hook = getattr(handler, 'agent_recovery_observation', None)
    bound = process_context(call, context)
    if not callable(hook) or bound is None:
        return call, None
    try:
        observed = hook(call=copy.deepcopy(call), context=bound)
    except Exception:
        return call, None
    if not isinstance(observed, Mapping) or observed.get('status') not in {'ok', 'error'}:
        return call, None
    receipt = observed.get('state_updates')
    if not isinstance(receipt, Mapping) or execution_event(handler, receipt, bound) is None:
        return call, None
    current = receipt['execution_session']
    if session and any(current.get(k) != session.get(k) for k in ('session_id', 'command_sha256')):
        return call, None
    # 无原回执仅允许执行器明确认证的只读观察失败；失败的启动不能改成成功。
    if not session and observed.get('observation_only') is not True:
        return call, None
    proof = {**call, 'status': observed['status'], 'data': {'state_updates': copy.deepcopy(receipt)}}
    audit = {'source': 'executor_recovery_observation', 'original_status': call.get('status'),
             'execution_session': copy.deepcopy(current), 'observation_only': not bool(session)}
    return proof, audit


def changed_file_observation(call, handler, context):
    """已成功写入但后来改变的文件允许接续重查；文件不存在或授权失败仍阻止恢复。"""
    from .reply_delivery import delivery_turn
    if delivery_turn(context) is None or call.get('status') != 'ok' or call.get('executed') is not True:
        return None
    receipt = (call.get('data') or {}).get('state_updates', {})
    if set(receipt.get('verified_effects', ())) != {'write'}:
        return None
    verify = getattr(handler, 'agent_verify_result', None)
    if not callable(verify):
        return None
    try:
        actual = verify(receipt=copy.deepcopy(receipt), context=context)
    except Exception:
        return None
    if not isinstance(actual, Mapping):
        return None
    old_rows = receipt.get('artifact_receipts') or [receipt]
    new_rows = actual.get('artifact_receipts') or [actual]
    if (not isinstance(old_rows, (list, tuple)) or not isinstance(new_rows, (list, tuple))
            or len(old_rows) != len(new_rows)):
        return None
    changes = []
    for old, new in zip(old_rows, new_rows):
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return None
        if (not old.get('path') or not re.fullmatch(r'[0-9a-f]{64}', str(old.get('sha256', '')))
                or not re.fullmatch(r'[0-9a-f]{64}', str(new.get('sha256', '')))
                or ('path' in new and new['path'] != old['path'])):
            return None
        if new['sha256'] != old['sha256']:
            changes.append({'path': old['path'], 'previous_sha256': old['sha256'], 'current_sha256': new['sha256']})
        elif new.get('result_valid') is not True:
            return None
    return {'source': 'executor_file_recheck', 'changes': changes,
            'requires_revalidation': True, 'completion_accepted': False} if changes else None
