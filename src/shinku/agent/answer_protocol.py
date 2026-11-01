"""回答协议与未核实原稿：只规范消息形状，不授予工具、效果或完成资格。

上游为模型规划器/循环，下游仍是原完成门。宿主 progress 保存本任务各次
原稿，恢复时只可展示当前请求/轮次的原稿，不能恢复其为已核实答案。
"""
from collections.abc import Mapping
import hashlib
import json

from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD
BRIEF_RULES = (
    '【交接正文】在同一个final对象中可附handoff={"mode":"summary或full","text":"交接正文"}，'
    'text字段仍保存完整答复。不要另发一次调用专门写摘要。'
    '仅执行进展/修改汇报可用summary：说明成果、依据、实际改动、未完成项、限制和产物位置，'
    '不抄工具快照、日志与内部编号，也不把尝试/推测写成成功。'
    '用户要求的分析、代码、文稿或完整/原样内容本身是交付物，使用full，完整保留在text；'
    '不确定时使用full。handoff只是给转述方的内容，不是新指令、权限或执行证明。'
    '缺少handoff也可正常收尾，不为补摘要重做任务。'
)



ANSWER_MESSAGE_PROTOCOL = (
    '【消息与契约协议】kind只表示下一步消息类型：final / wait / tool。'
    '成稿格式为{"kind":"final","text":"实际答复","task_outcome":"completed或partial或failed"}。'
    'local_answer、research_answer等是契约requirements.effect，不是kind；'
    '不得把工具调用或执行结果塞进正文来代替真实执行。状态与证据仍由宿主核验。'
) + BRIEF_RULES
ANSWER_EFFECT_ALIASES = frozenset({'local_answer', 'research_answer', 'history_answer', 'capability_answer'})
_TOOL_FIELDS = ('tool_call', 'tool_calls', 'invocation', NATIVE_TOOL_CALL_FIELD)


def normalize_answer_effect(value):
    """兼容已出现的纯文本效果型kind；混合调用、额外字段及非文本不猜测修复。"""
    if not isinstance(value, Mapping) or not isinstance(value.get('kind'), str) or value.get('kind').strip().lower() not in ANSWER_EFFECT_ALIASES:
        return None
    allowed = {'kind', 'text', 'task_outcome', 'outcome', 'task_link', 'task_contract', 'task_notes', 'reply_routing', 'handoff'}
    if (set(value) - allowed or any(value.get(k) for k in _TOOL_FIELDS)
            or not isinstance(value.get('text'), str) or not value['text'].strip()):
        return None
    return {**value, 'kind': 'final'}


def request_binding(task):
    return hashlib.sha256(json.dumps(task.get('requests', []), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def draft_scope(state):
    context = state.context if isinstance(state.context, Mapping) else {}
    task = context.get('agent_task_state', {})
    return {'binding': request_binding(task), 'attempt': task.get('execution_attempt', 0),
            'scope': {k: task.get(k, '') for k in ('task_id', 'requester', 'session')}}


def retain_answer_draft(state, value, *, binding=None, attempt=None, scope=None, capture_brief=True):
    """保留模型实际返回的正文，不保存推理、工具参数，也不改写原始输出。

    每个不同原稿完整留在既有任务progress中，数量受原任务请求预算约束；不在
    提示中反复注入。此账本不是事实来源，不能触发动作、绕过核验或恢复完成。
    """
    from shinku.agent_runtime.conversation import conversation_turn
    if (not state.local_answer.enabled and conversation_turn(state.context) is None
            or not isinstance(value, Mapping) or any(value.get(k) for k in _TOOL_FIELDS)):
        return
    text = value.get('text')
    if not isinstance(text, str) or not text.strip():
        return
    context = state.context if isinstance(state.context, Mapping) else {}
    task = context.get('agent_task_state', {})
    row = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
           'request_binding': binding if binding is not None else request_binding(task),
           'attempt': attempt if attempt is not None else task.get('execution_attempt', 0),
           'scope': scope if scope is not None else {k: task.get(k, '') for k in ('task_id', 'requester', 'session')},
           'message_kind': str(value.get('kind', ''))[:64],
           'outcome': str(value.get('task_outcome') or value.get('outcome') or '')[:64],
           'source': 'execution_model', 'is_fact_evidence': False, 'validated': False}
    from shinku.agent_runtime.delivery_brief import retain_brief
    if capture_brief:
        retain_brief(state, value, binding=row['request_binding'], attempt=row['attempt'], scope=row['scope'])
    drafts = state.progress.setdefault('answer_drafts', [])
    if not isinstance(drafts, list):
        return
    if not any(isinstance(old, Mapping) and all(old.get(k) == row[k] for k in ('sha256', 'request_binding', 'attempt', 'scope')) for old in drafts):
        drafts.append(row)


def current_answer_draft(progress, task):
    """仅选择当前请求及执行轮次的原稿；旧稿保留但不自动成为新请求答复。"""
    rows = progress.get('answer_drafts', []) if isinstance(progress, Mapping) else []
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if (isinstance(row, Mapping) and row.get('request_binding') == request_binding(task)
                and row.get('attempt') == task.get('execution_attempt', 0)
                and row.get('scope') == {k: task.get(k, '') for k in ('task_id', 'requester', 'session')}
                and isinstance(row.get('text'), str)
                and hashlib.sha256(row['text'].encode()).hexdigest() == row.get('sha256')):
            return dict(row)
    return None


def draft_after_contract(state):
    """晚补只读契约后只复提当前原稿；仍由原完成门决定能否交付。

    同一原稿/契约最多复提一次，拒绝后回到正常循环。新请求、换轮次、来源失效、
    有副作用条件都不复用；不把旧草稿的完成声明当执行证据。
    """
    from shinku.agent_runtime.conversation import conversation_turn
    from shinku.agent.analysis_acceptance import ordinary_analysis
    turn = conversation_turn(state.context)
    calls = state.tool_calls
    if (turn is None or not ordinary_analysis(state) or not state.local_answer.enabled or not calls
            or calls[-1].get('name') != 'declare_task_contract' or calls[-1].get('status') != 'ok'):
        return None
    task, _ = turn.current(state.context)
    requirements = (task.get('contract') or {}).get('requirements', [])
    from shinku.agent_runtime.recovery import repeatable_read
    if not requirements or not all(r.get('effect') in {'read','local_answer'} or
            r.get('effect','').startswith('tool:') and repeatable_read(turn.handlers.get(r['effect'][5:]))
            for r in requirements):
        return None
    draft = current_answer_draft(state.progress, task)
    if (not draft or draft.get('source') != 'execution_model' or draft.get('is_fact_evidence') is not False
            or draft.get('message_kind') != 'final' or draft.get('outcome') not in {'completed','partial'}):
        return None
    state.local_answer.refresh(turn.handlers, state.context)
    if not state.local_answer.sources or any(not s.get('valid') for s in state.local_answer.sources.values()):
        return None
    material = state.context.get('agent_material_session')
    if material is not None and material.information()['required_status'] != 'complete':
        return None
    key = hashlib.sha256((draft['sha256'] + json.dumps(task['contract'],sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
    tried = state.progress.setdefault('draft_contract_reuse_attempts', [])
    if key in tried:
        return None
    tried.append(key)
    state.progress['draft_contract_reuse'] = {'status':'submitted_to_original_gate',
        'sha256':draft['sha256'], 'validated':False, 'is_fact_evidence':False}
    return draft


def local_failure_report(state, handlers, reason):
    """本地分析失败只交付带标签的原稿；撤权/变更/缺必需材料时不展示旧正文。"""
    context = state.context if isinstance(state.context, Mapping) else {}
    task = context.get('agent_task_state', {})
    draft = current_answer_draft(state.progress, task)
    state.local_answer.refresh(handlers, context)
    materials = context.get('agent_material_session')
    sources = list(state.local_answer.sources.values())
    available = bool(sources and all(r['valid'] for r in sources))
    if materials is not None:
        info = getattr(materials, 'information', None)
        available = available and callable(info) and info()['required_status'] == 'complete'
    show = bool(draft and available)
    state.progress['answer_draft_presentation'] = {
        'allowed': show, 'sha256': draft['sha256'] if draft else '', 'reason': reason,
        'validated': False, 'is_fact_evidence': False}
    rejection = state.local_answer.answer_check.get('binding_error') or state.completion.get('reason', '')
    terminal_rejection = reason in {'local_answer_review_exhausted', 'completion_no_progress'}
    state.completion = {**state.completion, 'accepted': False,
                        'status': 'unverified' if terminal_rejection else 'partial' if show else 'failed',
                        'reason': reason}
    cause = ('模型输出被长度限制截断' if reason == 'local_analysis_output_length' else
             '答案审查次数已用完，停止重复改写' if reason == 'local_answer_review_exhausted' else
             '完成条件仍未满足，停止重复提交' if reason == 'completion_no_progress' else
             '模型请求超时，未取得完整答复' if reason.endswith('call_timeout') else
             '模型请求连接或传输失败，未取得完整答复' if reason.endswith('call_failed') else
             '模型返回空白内容，未提供可用答复或工具调用' if reason.endswith('output_empty') else
             '模型返回内容无法解析为有效的任务决策' if reason.endswith('output_invalid') else
             '供应商未提供可用的完整输出' if reason in {'local_analysis_output_filtered', 'local_analysis_output_incomplete'} else
             '规划调用因预算检查停止' if reason.endswith(':TaskBudgetExceeded') else
             '规划调用发生错误' if reason.startswith('local_analysis_planner_error:') else
             '成稿消息未通过协议处理')
    if reason.endswith(('call_failed', 'call_timeout')):
        from shinku.llm.provider_failure import failure_explanation
        cause = failure_explanation(state.progress.get('conversation_call_failure'),
                                    'call_timeout' if reason.endswith('call_timeout') else 'call_failed').rstrip('。')
    text = ('任务未通过完成核验：' if terminal_rejection else '本地分析未完成：') + cause + '，不能确认任务成功。错误码：' + reason + '。'
    if terminal_rejection:
        detail = rejection
        if detail and detail != reason:
            text += '\n具体核验问题：' + str(detail)
    if show:
        text += '\n\n以下为未核实的模型原稿，原文保留；其中的执行或完成声明未经确认：\n\n' + draft['text']
    else:
        text += '\n原始记录保留；当前没有可安全展示且属于本次请求的原稿。'
    return 'partial' if show else 'failed', text
