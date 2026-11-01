"""从已有任务账本派生可见进度，不调用模型、不写任务状态、不参与完成判定。

阅读顺序：read_progress 校验宿主会话与任务版本，project 组织当前阶段和回执。
等待时长只是活动提示；模型计划单独标记，不能替代真实工具或验收证据。
"""
from pathlib import PurePosixPath
import re
import time

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.information import safe_text
from shinku.agent.work_state import input_binding
from .task_overview import project_overview

ACTIVE = {'queued', 'running', 'planning', 'yielded', 'waiting_background'}
TOOL_LABELS = {
    'workspace_info': '定位当前工作区', 'workspace_search': '搜索项目文件',
    'workspace_read_lines': '读取代码', 'workspace_preview_patch': '检查修改差异',
    'workspace_apply_patch': '应用文件修改', 'browser_observe': '查看网页',
    'private_file_read': '读取文件', 'private_file_write': '写入文件',
    'private_file_edit': '修改文件', 'private_shell': '执行命令',
    'private_shell_execute': '执行命令', 'private_browser_open': '读取网页',
    'private_web_search': '检索资料', 'private_download': '下载文件',
    'declare_task_contract': '确认任务目标', 'read_self_state': '查看当前能力',
    'update_work_plan': '更新工作计划', 'read_work_context': '读取工作记录',
    'task_material_read': '读取已保留资料', 'read_task_records': '读取任务记录',
}


def clean(value, limit=180):
    return safe_text(str(value or ''))[:limit]


def tool_label(call):
    name = str(call.get('name') or '')
    label = TOOL_LABELS.get(name, '调用工具 ' + (name if re.fullmatch(r'[\w.-]{1,80}', name) else '未知工具'))
    if name == 'private_shell':
        label = {'start': '运行命令', 'poll': '查看命令进展', 'write': '向运行中命令输入',
                 'stop': '停止命令'}.get(call.get('arguments', {}).get('action'), label)
    # 参数可能包含命令、凭据或大段正文；这里只摘取路径末段，不输出完整参数。
    path = call.get('arguments', {}).get('path')
    if isinstance(path, str):
        label += ' · ' + clean(PurePosixPath(path.replace('\\', '/')).name, 80)
    return label


def project(task, view, *, activity=None, pending=(), now=None):
    now = time.time() if now is None else now
    attempt = task['execution_attempt']
    session = task['session']
    calls = [c for c in task.get('tool_calls', []) if
        c.get('execution_binding', {}).get('attempt') == attempt
        and c.get('execution_binding', {}).get('task_id') == task['task_id']
        and c.get('execution_binding', {}).get('session') == session]
    events = []
    for index, call in enumerate(calls[-24:]):
        state = call.get('status', 'unknown')
        executed = call.get('data', {}).get('state_updates', {}).get('executed', call.get('executed'))
        if executed is False:
            state = 'blocked'
        outcome = {'ok': '已取得回执', 'error': '执行报错', 'blocked': '未执行'}.get(state, '结果待核实')
        process = call.get('data', {}).get('state_updates', {}).get('execution_session') or {}
        # 进程仍在运行时不能只写“已取得回执”，更不能把退出码等同于整项成功。
        if state == 'ok' and executed is not False:
            if process.get('status') == 'running':
                outcome = '命令仍在运行，等待后续结果'
            elif process.get('status') == 'completed' and type(process.get('returncode')) is int:
                outcome = '命令已退出，退出码 ' + str(process['returncode'])
        events.append({'id': f"{attempt}:{call.get('id', index)}", 'state': state,
            'text': tool_label(call) + '：' + outcome, 'source': 'executor'})
    status = view['status']
    running = status in ACTIVE and view.get('worker_active') is True
    phase, text, since = status, '', None
    if view.get('cancel_requested') and view.get('worker_active'):
        phase, text = 'canceling', '已请求停止，正在等待当前操作退出。'
    elif not running:
        text = {
            'completed': '本轮结果已交付。', 'failed': '本轮执行失败，请查看具体错误。',
            'canceled': '本轮已停止，已发生的改动保留。',
            'blocked': '本轮遇到阻碍，等待处理。', 'waiting_user': '需要你的补充或决定。',
            'partial': '已交付部分结果，仍有未完成项。', 'unverified': '已有结果，尚未通过验收。',
            'queued': '任务已排队，等待工作线程。',
        }.get(status, '暂时无法核对最新执行状态。')
    else:
        approvals = [p for p in pending if p.get('taskId') == task['task_id']
            and p.get('conversationId') == session[8:] and p.get('attempt') == attempt]
        if approvals:
            phase, text, since = 'approval', '正在等待你批准工具执行，尚未执行该操作。', approvals[0]['created']
        elif task.get('inflight'):
            phase, text, since = 'tool', '正在' + tool_label(task['inflight'][0]) + '。', task['updated']
        elif activity:
            phase, text, since = 'model', '正在等待模型返回，收到结果后继续处理。', activity['started']
        elif view.get('execution', {}).get('finalization_pending'):
            phase, text = 'finalizing', '正在封存本轮结果和执行回执。'
        else:
            phase, text, since = 'working', '任务仍在运行，正在等待下一条执行记录。', task['updated']
    wait_seconds = max(0, int(now - since)) if since is not None else 0
    # 每20秒只更新同一条等待提示，既不刷屏，也不把心跳算成工作进展。
    wait_bucket = wait_seconds // 20 * 20
    if wait_bucket:
        text += f' 已等待 {wait_bucket} 秒。'
    plan = task.get('work_state') or {}
    current_plan = {}
    if plan.get('attempt') == attempt and plan.get('binding') == input_binding(task):
        proposal = plan.get('plan', {})
        step = next((s for s in proposal.get('steps', []) if s.get('id') == proposal.get('current_step')), {})
        current_plan = {'current': clean(step.get('goal')), 'next': clean(proposal.get('next_action')),
            'blocker': clean(proposal.get('blocker')), 'source': 'model_plan', 'verified': False}
    return {'available': True, 'taskId': task['task_id'], 'attempt': attempt, 'version': task['revision'], 'phase': phase,
        'active': running, 'text': text, 'waitSeconds': wait_bucket, 'events': events,
        'totalEvents': len(calls), 'plan': current_plan, 'overview': project_overview(task, view, calls),
        'source': 'host_execution_state',
        'is_fact_evidence': False}


def read_progress(host, cid, turn):
    """仅宿主绑定的当前会话可读；跨版本快照拒绝展示，旧尝试不能冒充当前执行。"""
    task_id = turn.get('task_id')
    if not task_id:
        if turn.get('stage') in {'preparing', 'submitting'}:
            return {'available': True, 'phase': 'preparing', 'active': True,
                'text': '正在接收要求并准备本轮执行。', 'events': [], 'totalEvents': 0}
        return None
    try:
        scope = AgentScope(host.requester, 'angelos:' + cid)
        with host.service._lock:
            task = host.service._get(scope, task_id)
            host.service._grant(scope, task, 'inspect')
            view = host.port(cid).status(task_id)
            fresh = host.service._get(scope, task_id)
            if (not view.get('available') or view.get('version') != task['revision']
                    or (fresh['revision'], fresh['execution_attempt']) != (task['revision'], task['execution_attempt'])):
                raise ValueError('progress_snapshot_changed')
            controls = getattr(host, 'permissions', None)
            pending = controls.snapshot()['pending'] if controls else []
            result = project(task, view, activity=host.reasoning_view.activity(task_id), pending=pending)
            from .commentary import read_updates
            result['updates'] = read_updates(host, task_id)
            return result
    except Exception:
        # 展示故障不改变任务，不触发恢复、工具调用或额外模型重试。
        return {'available': False, 'phase': 'unavailable', 'active': False,
            'text': '暂时无法核对最新进度；下次刷新会重新读取。', 'events': [], 'totalEvents': 0}
