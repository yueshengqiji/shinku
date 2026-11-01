"""任务工作卡：只读拼接当前尝试的目标、验收和产物记录。

调用方先核对任务作用域和版本；本模块不读磁盘、不调用模型、不修改账本。
旧结构化任务展示契约核验；普通会话区分答复交付与逐工具回执。文件只展示
执行器关联，不冒充 Git 改动归因或当前磁盘状态。旧尝试的成功、正文和产物
不能混进恢复中的新尝试。
"""
from pathlib import PurePosixPath

from shinku.agent_runtime.information import safe_text


def _text(value, limit=180):
    return safe_text(value)[:limit] if isinstance(value, str) else ''


def project_overview(task, view, calls):
    """接收已经过滤绑定的工具回执，生成展示数据，不授予新的执行/完成权限。"""
    verification = view.get('verification') or {}
    current = verification.get('current_attempt') is True
    layers = verification.get('layers', {}) if current else {}
    policy = layers.get('policy', '')
    status = view.get('status', 'unknown')
    if policy == 'conversation_reply' or policy == 'agent_reply_delivery' and not calls:
        return None  # 日常回复不展示空的任务验收卡。
    verified = current and verification.get('verified') is True and status == 'completed'
    accepted = current and verification.get('accepted') is True
    verdict = ('analysis_delivered' if verified and policy in {'ordinary_readonly_analysis', 'ordinary_analysis'}
               else 'verified' if verified else 'partial' if accepted and status == 'partial'
               else 'not_verified' if current else 'pending')
    if policy == 'agent_reply_delivery':
        verdict = 'reply_delivered'
    requirements = [] if policy == 'agent_reply_delivery' else (task.get('contract') or {}).get('requirements', [])
    satisfied = set(verification.get('satisfied', [])) if current else set()
    missing = set(verification.get('missing', [])) if current else set()
    checks = [{'id': _text(row.get('id'), 80), 'description': _text(row.get('description')),
               'state': 'satisfied' if row.get('id') in satisfied else
                        'missing' if row.get('id') in missing else 'pending'}
              for row in requirements[:24]]
    files, file_count = [], 0
    for call in calls:
        updates = call.get('data', {}).get('state_updates', {})
        # 拒绝执行/报错中的路径不被包装成成功产物；不从工具参数猜文件改动。
        if call.get('status') != 'ok' or updates.get('executed', call.get('executed')) is False:
            continue
        for artifact in updates.get('artifact_receipts', []):
            if not isinstance(artifact, dict) or not isinstance(artifact.get('path'), str):
                continue
            file_count += 1
            if len(files) < 12:
                files.append({'name': _text(PurePosixPath(artifact['path'].replace('\\', '/')).name),
                    'state': 'receipt_valid' if artifact.get('result_valid') is True else 'unconfirmed'})
    result = task.get('execution_result') or {}
    bound = (result.get('task_id') == task['task_id'] and
             result.get('attempt') == task['execution_attempt'] and
             result.get('scope') == {'requester': task['requester'], 'session': task['session']})
    body = (result.get('execution_report') or {}).get('text') if bound else None
    # 正文存在只是已归档，不代表用户看到、外部发送成功或事实已核验。
    report = 'available' if isinstance(body, str) and body.strip() else 'not_recorded'
    return {'source': 'host_task_records', 'is_authority': False,
        'attempt': task['execution_attempt'], 'status': status, 'verification': verdict,
        'answerEvidence': _text(layers.get('answer_evidence'), 60), 'report': report,
        'checks': checks, 'omittedChecks': max(0, len(requirements)-len(checks)),
        'files': files, 'omittedFiles': file_count-len(files),
        'receiptCount': len(calls), 'reason': _text(view.get('reason'), 120),
        'receiptChecks': [dict(row) for row in task.get('completion', {}).get('evidence', [])
                          if policy == 'agent_reply_delivery' and isinstance(row, dict)],
        'cancelRequested': view.get('cancel_requested') is True,
        'workerActive': view.get('worker_active') is True}
