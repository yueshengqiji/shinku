"""从 Agent 原账本派生工作记忆，不持有第二份任务状态。

执行回执、整体验收和模型报告分别展示。只做有界投影，不执行旧命令、
不刷新证据寿命；完整原文仍通过原任务工具读取。计划中的判断始终待核实。
"""
from collections.abc import Mapping

from .contracts import canonical_hash
from .information import safe_text


def task_memory(task, *, max_calls=8):
    """只信任宿主传入的任务；返回值是导航，不是当前环境成功证明。"""
    calls = task.get('tool_calls', [])
    effects, errors, uncertain = [], [], []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        data = call.get('data') or {}
        receipt = data.get('state_updates', {}) if isinstance(data, Mapping) else {}
        if not isinstance(receipt, Mapping):
            receipt = {}
        session = receipt.get('execution_session') or {}
        item = {'call_id': call.get('id'), 'tool': call.get('name'),
                'status': call.get('status'), 'executed': call.get('executed'),
                'source': 'executor_ledger', 'is_current_state_proof': False}
        if session:
            item['process'] = {k: session.get(k) for k in ('session_id', 'status', 'returncode', 'reason')}
        verified = receipt.get('verified_effects')
        if call.get('status') == 'ok' and call.get('executed') is not False and verified:
            effects.append({**item, 'effects': list(verified)[:8],
                            'targets': list(receipt.get('completion_targets') or [])[:8]})
        if call.get('status') in {'error', 'blocked'}:
            # 不复制 stderr/参数/推理；具体错误留在材料库，避免把密钥带入长期索引。
            errors.append({**item, 'code': str(data.get('code') or data.get('error') or '')[:160],
                           'detail_reference': call.get('material_reference')})
        if call.get('executed') is not False and (session.get('status') == 'running'
                or call.get('status') == 'error' and not receipt.get('verified_effects')):
            uncertain.append(item)
    completion = task.get('completion') or {}
    result = task.get('execution_result') or {}
    report = (result.get('execution_report') or {}).get('text', '')
    requests = task.get('requests') or []
    value = {'schema_version': 1, 'task_id': task['task_id'],
        'attempt': task.get('execution_attempt', 0), 'version': task.get('revision', 0),
        'goal': safe_text(requests[0]['text'])[:1600] if requests else '',
        'status': task.get('status', 'unknown'), 'reason': task.get('stop_reason', ''),
        'verification': {k: completion.get(k) for k in ('accepted', 'status', 'reason', 'satisfied', 'missing', 'layers')},
        'recorded_effects': effects[-max_calls:], 'errors': errors[-max_calls:],
        'uncertain_effects': uncertain[-max_calls:],
        'omitted': {'effects': max(0, len(effects)-max_calls), 'errors': max(0, len(errors)-max_calls),
                    'uncertain': max(0, len(uncertain)-max_calls)},
        'report_delivery': 'canonical_report_present' if report else 'no_canonical_report',
        'report_is_execution_proof': False,
        'original_retrieval': {'tool': 'read_task_records', 'mode': 'detail', 'task_id': task['task_id']},
        'grants_permissions': False, 'is_execution_evidence': False}
    value['source_sha256'] = canonical_hash({k: task.get(k) for k in (
        'task_id', 'requester', 'session', 'execution_attempt', 'revision', 'requests',
        'tool_calls', 'completion', 'status', 'stop_reason', 'execution_result')})
    return value
