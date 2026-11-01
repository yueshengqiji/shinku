"""提供同作用域的任务接续背景，不持有任务状态，也不生成新摘要。

Gateway 在原服务锁和 inspect 授权下调用；原请求、工作计划、报告仍归原账本。
此处只投影小型导航，明确区分模型陈述、历史版本和宿主事实，不能授权或证明执行。
"""
from dataclasses import asdict
import json

from shinku.agent.task_state import redact
from shinku.agent.work_state import input_binding
from .contracts import AgentBoundaryError
from .work_context import navigation


def retain_resume_report(service, scope, task, grant):
    """重开前把本轮旧报告放入原资料库；回答选项不会随execution_result清空而消失。

    仅在Service已通过恢复授权/预算/副作用检查后调用。报告是原模型背景，
    不转成用户要求或执行证据；沿用材料的分页、过期和读取账本，不额外调用模型。
    无资料库的兼容宿主保存在source_data原件中，仍受既有上下文预算约束。
    """
    if not task.get('execution_result'):
        return
    service._grant(scope, task, 'inspect')
    result = task['execution_result']
    from shinku.result_expression.facts import validate_result
    # 这里只恢复历史正文，不查询“当前执行结果”：取消或补充之后状态可合法不同。
    # 哈希/范围/轮次仍严格绑定，不能为了复用旧问题把旧completed认作当前完成。
    error = validate_result(result, scope)
    if (error or result.get('task_id') != task['task_id']
            or result.get('version') != task.get('execution_result_version', result.get('version'))):
        raise AgentBoundaryError(error or 'result_binding_conflict')
    report = result.get('execution_report', {}).get('text')
    if not report or result.get('attempt') != task['execution_attempt']:
        return
    body = {'source': 'previous_agent_report', 'is_fact_evidence': False,
        'is_authority': False, 'task_id': task['task_id'], 'attempt': result['attempt'],
        'version': result['version'], 'result_hash': result['result_hash'], 'text': report,
        'note': '这是上一轮Agent答复/待答问题原文，仅用于理解本轮补充；执行、权限和当前事实须重新核对。'}
    source_id = 'prior-agent-report:' + result['result_hash']
    if service.materials is not None:
        from .materials import MaterialSource
        old = task.get('material_package') or {}
        task['material_package'] = service.materials.package(task,
            [MaterialSource(source_id, 'related_message', body, required=False)],
            authorization_ref=grant.authorization_ref, previous=old, gaps=old.get('gaps', []))
        if any(row.get('source_id') == source_id for row in task['material_package']['items']):
            return
    # 材料容量/大小限制不能导致清空execution_result时丢原稿；兼容副本仍是数据，
    # 不是绕过读取门的已读凭证。已有必需材料缺口、上下文总预算保持原样。
    if not any(row.get('id') == source_id for row in task.get('source_data', [])):
        task['source_data'] = [*task.get('source_data', []),
            {'id': source_id, 'content': body, 'reference': 'model_background_not_authority'}]


def _excerpt(row):
    text = str(row.get('text') or '')
    return {'text': text[:2000], 'omitted_chars': max(0, len(text) - 2000),
            'message_id': str(row.get('message_id') or '')}


def continuation_context(service, scope, task):
    """读取当前版本；旧报告保留为背景，超长选项整段省略而不截掉条件。"""
    task_id = task['task_id']
    view = service._status(scope, task_id)
    if not view.get('available') or (view['attempt'], view['version']) != (
            task['execution_attempt'], task['revision']):
        raise AgentBoundaryError('delegation_context_changed')
    result = task.get('execution_result') or {}
    report = str(result.get('execution_report', {}).get('text') or '')
    current = (result.get('attempt'), result.get('version')) == (view['attempt'], view['version'])
    requests = task.get('requests') or []
    selected = requests[:1] + requests[max(1, len(requests) - 3):]
    record = service.task_store.get_work_state(task_id, requester=scope.requester, session=scope.session)
    work = {'status': 'absent', 'is_execution_evidence': False}
    if record:
        work = navigation({'revision': record['revision'], 'plan': record['plan'],
            'references': record['references'], 'is_execution_evidence': False,
            'step_status_source': 'model_declaration', 'findings_source': 'model_unverified',
            'references_current_validity': 'requires_revalidation',
            'status': 'active' if record['binding'] == input_binding(task)
                and record['attempt'] == task['execution_attempt'] else 'requires_review'})
        if len(json.dumps(work, ensure_ascii=False)) > 6000:
            work = {'revision': record['revision'], 'status': 'omitted_for_size',
                    'is_execution_evidence': False, 'full_plan_retrieval': work['full_plan_retrieval']}
    latest = service._get(scope, task_id)
    if (latest['execution_attempt'], latest['revision']) != (view['attempt'], view['version']):
        raise AgentBoundaryError('delegation_context_changed')
    return redact({'schema_version': 1, 'available': True, 'scope': asdict(scope),
        'task_id': task_id, 'attempt': view['attempt'], 'version': view['version'],
        'input_binding': input_binding(task), 'status': view['status'], 'updated': view['updated'],
        'is_authority': False, 'source': 'scoped_task_ledger',
        'requests': [_excerpt(r) for r in selected], 'omitted_requests': len(requests) - len(selected),
        'report': {'text': report if len(report) <= 6000 else '', 'current': current,
            'attempt': result.get('attempt'), 'version': result.get('version'),
            'result_hash': result.get('result_hash'), 'source': 'execution_model',
            'is_fact_evidence': False, 'omitted_chars': len(report) if len(report) > 6000 else 0},
        'work': work, 'done': view['done'], 'remaining': view['pending'], 'blocker': view['blocker'],
        'information': view['information'],
        'retrieval': {'tool': 'agent_task_result', 'task_id': task_id, 'view': 'full'},
        'originals_retained': True})
