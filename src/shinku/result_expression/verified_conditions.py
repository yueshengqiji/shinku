"""Readable condition and artifact context from the sealed host result.

Requested conditions alone never prove execution. A condition description is
marked satisfied only by the completion verifier; artifact effects also need
the matching successful invocation and current content identity.
"""
from __future__ import annotations

import re


RULES = (
    "宿主启用表达上下文版本3。condition_observations将验收条件编号绑定到原条件的含义和宿主验收状态；"
    "用条件的含义向用户说明结果，编号用于归属和审计，无需在正文重复。"
    "contract_context是原请求的条件数据，单独不能证明操作已发生；只有验收标记及匹配回执支持已完成主张。"
    "artifact_observation仅证明匹配的当前调用已实际写入或读取；read_complete=false不能称完整回读，"
    "回执未给出调用总数时不能推断只写一次。不同内容身份不能合并成同一次回读核验。"
    "将具体成果、必要条件和仍待确认的范围连成面向用户的自然说明；审计字段用于核对，"
    "不要把字段名或编号当作用户需要阅读的结论。保持原风格、人称及称呼，事实和限制仍完整保留。"
)

_ARTIFACT_FIELDS = ('path', 'url', 'sha256', 'bytes', 'result_valid', 'completion_targets',
                    'verified_effects', 'read_complete')


def _requirements(result):
    from shinku.agent_runtime.reply_delivery import is_reply_delivery
    if is_reply_delivery(result.get('verification', {})):
        return {}  # 旧模型契约是历史资料，不在表达侧恢复为已确认用户硬条件。
    contract = result.get('contract')
    rows = contract.get('requirements') if isinstance(contract, dict) else None
    if not isinstance(rows, list) or not rows:
        return {}
    known = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                or row['id'] in known or not isinstance(row.get('description'), str) or not row['description'].strip()
                or not isinstance(row.get('targets', []), list)
                or any(not isinstance(v, str) for v in row.get('targets', []))
                or not isinstance(row.get('checks', {}), dict)):
            return {}
        known[row['id']] = {k: row[k] for k in ('id', 'description', 'effect', 'targets', 'checks') if k in row}
    return known


def _current_artifact(result, artifact):
    from shinku.agent_runtime.reply_delivery import is_reply_delivery
    reply_only = is_reply_delivery(result.get('verification', {}))

    if (artifact.get('source') != 'executor_receipt' or artifact.get('result_valid') is not True
            or type(artifact.get('result_version')) is not int or artifact['result_version'] != result.get('version')
            or not isinstance(artifact.get('path'), str) or not artifact['path']
            or not isinstance(artifact.get('sha256'), str) or not re.fullmatch('[a-fA-F0-9]{64}', artifact['sha256'])
            or type(artifact.get('bytes')) is not int or artifact['bytes'] < 0
            or not isinstance(artifact.get('completion_targets'), list)
            or artifact['path'] not in artifact['completion_targets']
            or not isinstance(artifact.get('verified_effects'), list)
            or not artifact['verified_effects'] or any(v not in {'read', 'write'} for v in artifact['verified_effects'])
            or 'read_complete' in artifact and type(artifact['read_complete']) is not bool):
        return None
    identity = {k: artifact[k] for k in _ARTIFACT_FIELDS if k in artifact}
    for row in result.get('evidence', []):
        if (not isinstance(row, dict) or row.get('source') != 'executor_receipt'
                or row.get('execution_status') != 'ok' or not isinstance(row.get('receipt'), dict)):
            continue
        if reply_only and not any(check.get('invocation_id') == row.get('invocation_id')
                and check.get('receipt_valid') is True for check in result['verification'].get('evidence', [])):
            continue  # 直接交付时不能用旧文件回执绕过本轮独立重验的失败或未知。
        for receipt in row['receipt'].get('artifact_receipts', []):
            if isinstance(receipt, dict) and {k: receipt[k] for k in _ARTIFACT_FIELDS if k in receipt} == identity:
                return {'verified_effects': list(artifact['verified_effects']),
                        'read_complete': artifact.get('read_complete', False),
                        'path': artifact['path'], 'sha256': artifact['sha256'],
                        'result_version': artifact['result_version'], 'invocation_id': row.get('invocation_id', ''),
                        'scope': 'recorded_invocation', 'source': 'executor_receipt'}
    return None


def _readable_check_text(checks, human):
    labels = {
        'min_sources': '来源数量下限', 'min_bytes': '字节数量下限',
        'contains': '必须包含的原文', 'sha256': 'SHA256内容校验值',
    }
    return '；'.join(labels.get(key, key) + '：' + human(value) for key, value in checks.items())


def apply_verified_context(result, blocks, *, safe, human, anchors, readable_checks=False):
    """Extend only explicit version-three presentation, never the ledger."""
    requirements = _requirements(result)
    verification = result.get('verification', {})
    satisfied, missing = verification.get('satisfied', []), verification.get('missing', [])
    consistent = (verification.get('source') == 'completion_verifier' and isinstance(satisfied, list)
        and isinstance(missing, list) and all(isinstance(v, str) for v in [*satisfied, *missing])
        and len(set(satisfied)) == len(satisfied) and not set(satisfied).intersection(missing)
        and not (verification.get('accepted') is True and missing))
    if requirements:
        data = safe(list(requirements.values()))
        blocks['contract_context'] = {'id': 'contract_context', 'text': '原始验收条件（请求数据，不是执行凭据）：' + human(data),
            'anchors': anchors(data), 'required': False, 'source': 'confirmed_contract',
            'condition_definitions': data}
    if consistent and satisfied and all(v in requirements for v in satisfied):
        original = blocks['verification']
        blocks['verification_audit'] = {**original, 'id': 'verification_audit', 'required': False}
        observed = safe([{**requirements[v], 'status': 'satisfied', 'source': 'completion_verifier'} for v in satisfied])
        text = '验收已确认的条件：' + '；'.join(row['description'] for row in observed)
        protected = []
        for row in observed:
            targets = row.get('targets', [])
            if targets:
                text += '；对应路径：' + human(targets)
                protected.extend(targets)
            exact_checks = {k: v for k, v in row.get('checks', {}).items()
                if type(v) in (int, float) or isinstance(v, str) and (any(c.isdecimal() for c in v) or re.search(r'[\\/]', v))}
            if exact_checks:
                text += '；核验条件：' + (_readable_check_text(exact_checks, human) if readable_checks else human(exact_checks))
                protected.extend(exact_checks.values())
        blocks['verification'] = {**original, 'text': text, 'anchors': anchors(protected),
            'semantic_review_required': True, 'condition_observations': observed}
    remaining = list(dict.fromkeys(str(v) for v in [*result.get('remaining', []), *missing]))
    if consistent and remaining and all(v in requirements and v in missing for v in remaining):
        old = blocks['remaining']
        blocks['remaining_audit'] = {**old, 'id': 'remaining_audit', 'required': False}
        rows = safe([{**requirements[v], 'status': 'missing', 'source': 'completion_verifier'} for v in remaining])
        paths = [p for row in rows for p in row.get('targets', [])]
        text = '尚未核验完成的条件：' + '；'.join(row['description'] for row in rows)
        if paths:
            text += '；对应路径：' + human(paths)
        blocks['remaining'] = {**old, 'text': text, 'anchors': anchors(paths), 'required': True,
            'semantic_review_required': True, 'condition_observations': rows}
    for i, artifact in enumerate(result.get('artifacts', [])):
        key = f'artifact_{i}'
        if key not in blocks or not isinstance(artifact, dict):
            continue
        observation = _current_artifact(result, artifact)
        if observation is None:
            continue
        actions = []
        if 'write' in observation['verified_effects']:
            actions.append('匹配的成功调用包含实际写入回执')
        if 'read' in observation['verified_effects']:
            actions.append('匹配的成功调用包含完整回读回执' if observation['read_complete'] else
                           '匹配的成功调用包含读取回执，未确认完整读取')
        row = blocks[key]
        row['text'] += '；' + '；'.join(actions) + '，只说明所列调用及核验时的内容身份。'
        row['semantic_review_required'] = True
        row['artifact_observation'] = safe(observation)
