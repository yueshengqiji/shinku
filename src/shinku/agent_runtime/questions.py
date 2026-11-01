"""等待问题归宿主结果所有；回答复用资料补充和恢复门，不建立第二个执行器。

question_id绑定原结果哈希（含作用域/尝试/版本）；补充键和恢复键由宿主
确定，浏览器重试不产生新执行。service锁串行化回答、取消与恢复；两次持久化
之间崩溃时只允许重试同一答案。答案只是用户资料，绝不是工具授权。
"""
from .contracts import AgentBoundaryError, AgentSupplement, canonical_hash


def question_for(result):
    if result.get('execution', {}).get('status') != 'waiting_user':
        return None
    text = result.get('execution_report', {}).get('text', '')
    if not text or not result.get('result_hash'):
        return None
    return {'id': 'q:' + result['result_hash'], 'text': text,
            'attempt': result['attempt'], 'version': result['version'],
            'created': result.get('updated'), 'status': 'pending'}


def pending_question(service, scope, task_id):
    """只读投影：已接收但尚未获准恢复的答案也返回工作台，刷新不丢失。"""
    with service._lock:
        task = service._get(scope, task_id)
        if task['status'] != 'waiting_user' or task.get('cancel_requested'):
            return None
        question = question_for(service._result(scope, task_id))
        if question:
            message_id = 'supplement:answer:' + question['id']
            answer = next((r for r in task.get('requests', []) if r['message_id'] == message_id), None)
            if answer:
                question = dict(question, answer_received=True, answer_text=answer['text'])
        return question


def answer_question(service, scope, handle, question_id, text):
    """返回恢复句柄；已收答案但恢复被拒时保留答案，重试不重复补充。

    只有同问题、同答案的重试可越过旧UI版本；其他输入仍精确核对版本。
    恢复已经入队时仅返回当前句柄，不把网络超时视为重新执行许可。
    """
    if not isinstance(question_id, str) or not question_id.startswith('q:') or len(question_id) != 66:
        raise AgentBoundaryError('question_not_current')
    if not isinstance(text, str) or not text.strip():
        raise AgentBoundaryError('question_answer_required')
    request = AgentSupplement('answer:' + question_id, text)
    digest = canonical_hash(request.payload())
    resume_key = 'resume:' + question_id
    with service._lock:
        task = service._get(scope, handle.task_id)
        if task.get('cancel_requested') or task['status'] == 'canceled':
            raise AgentBoundaryError('question_canceled')
        prior = next((r for r in task.get('material_supplements', []) if r['key'] == request.request_key), None)
        if prior and prior['digest'] != digest:
            raise AgentBoundaryError('question_answer_conflict')
        if prior and task.get('resume_request', {}).get('key') == resume_key:
            return service._handle(task)
        question = question_for(service._result(scope, handle.task_id))
        if task['status'] != 'waiting_user' or not question or question['id'] != question_id:
            raise AgentBoundaryError('question_not_current')
        if prior:
            # supplement保存输入及材料检查点各一次；除此之外的修订必须重新审阅。
            if task['revision'] not in {prior['version'] + 1, prior['version'] + 2}:
                raise AgentBoundaryError('task_version_conflict')
        else:
            service._check_handle(task, handle)
        latest = service._supplement(scope, service._handle(task), request)
        return service._resume(scope, latest, resume_key)
