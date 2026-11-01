"""Angelos 会话交付策略：正文不经过语义裁判，执行事实由宿主另列。

ServiceHost 显式安装，ConversationTurn 绑定当前授权及尝试；模型字段不能开启。
不调用旧完成门或任何审查模型，不修改执行回执、权限及送达状态。accepted 仅表示
本轮答复可以归档，不证明整个用户目标已完成；消费者须读取 layers 区分两者。
"""
from collections.abc import Mapping
import copy
import json

from shinku.agent.completion import CompletionVerdict
from shinku.agent.task_state import TaskContractError


POLICY = 'agent_reply_delivery'
GUIDANCE = (
    '【直接执行与答复】依据原始用户目标、确认约束和当前宿主授权主动工作。'
    '简单问题直接回答；复杂任务按需使用update_work_plan，不需要先声明验收契约。'
    '计划、task_contract及步骤done只是可调整的模型导航，不能授予权限或证明成功。'
    '使用已取得的工具结果和资料形成实际答复，不为引用格式、字数或辅助计划反复重写、扩读。'
    '事实、推断与建议区分清楚；准确保留失败、未完成项和不确定性。'
    '只有真实工具回执支持时才能声称已经修改、执行、测试或送达；注册能力不等于业务可用。'
    '失败的命令不能称为测试通过，生成文件不能称为已发送；需要用户操作时说明具体步骤。'
    '自身能力可用read_self_state，旧任务与日志按需读取read_task_records/read_service_logs。'
    '工具参数仍必须符合协议，路径范围、审批、预算、取消与幂等均由宿主严格执行。'
    '直接交付用户需要的正文；当前工作已完成可用completed，确有缺项用partial，实际失败用failed。'
    'task_outcome是你的结果报告，未做独立语义验收本身不是失败，不必据此反复重做任务。'
)


def delivery_turn(context):
    """只接受进程内宿主凭据，不能以 notes、JSON 或历史记录伪造模式。"""
    from .conversation import conversation_turn
    turn = conversation_turn(context)
    return turn if turn is not None and turn.reply_delivery else None


def is_reply_delivery(completion):
    """供只读导出/展示使用；本函数不参与执行授权。"""
    return isinstance(completion, Mapping) and completion.get('layers', {}).get('policy') == POLICY


def save_advisory(turn, proposal):
    """可选模型计划只写有界导航，原契约/原稿和工具账本不改写。"""
    if not isinstance(proposal, Mapping):
        raise TaskContractError('advisory_plan_invalid')
    value = copy.deepcopy(dict(proposal))
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 32768:
        raise TaskContractError('advisory_plan_oversized')
    with turn.service._lock:
        task, _ = turn.current(turn.state.context)
        progress = copy.deepcopy(turn.state.progress)
        history = progress.get('advisory_plans', task.get('progress', {}).get('advisory_plans', []))
        # 原模型输出另有完整档案；这里只保留最近修订导航，不能无限扩大任务快照。
        progress['advisory_plans'] = [*history[-7:], {'proposal': value,
            'attempt': turn.attempt, 'source': 'model_plan', 'is_execution_evidence': False}]
        saved = turn.service._write(turn.scope, turn.task_id, turn.attempt,
            lambda row: row.update(progress=progress))
        turn.state.progress.update(saved['progress'])
        ledger = turn.state.context['agent_task_state']
        ledger.clear(); ledger.update(saved)
    return value


def prepare_state(state):
    """包括恢复切片：停用在线答案账本；原件仍由工具/材料/原生历史保存。"""
    if delivery_turn(state.context) is None:
        return
    for ledger in (state.research, state.local_answer, state.history_answer):
        ledger.enabled = False
    # 旧分析探索上限不应在新版恢复后继续强迫收尾；工具防循环/总预算不重置。
    if state.progress.get('local_analysis', {}).get('phase') == 'drafting' and not state.tool_closeout_reason:
        state.finalization_only = False
    state.progress.pop('local_analysis', None)


class ReplyDeliveryGate:
    """只核对当前凭据及真实调用回执；单条回执失效不销毁用户可读正文。"""
    supports_plain_text_completion_verification = False
    enabled = True

    def __init__(self, turn):
        self.turn = turn

    def evaluate(self, state, text, *, outcome=''):
        turn = self.turn
        turn.current(state.context)
        if not isinstance(text, str) or not text.strip():
            return CompletionVerdict(False, reason='empty_reply')
        rows = []
        for call in state.tool_calls[-128:]:
            _, grant = turn.current(state.context)
            name = call.get('name', '')
            handler = turn.handlers.get(name)
            if getattr(handler, 'agent_execution_evidence', True) is False:
                continue
            receipt = (call.get('data') or {}).get('state_updates') or {}
            executed = receipt.get('executed', call.get('executed'))
            row = {'invocation_id': call.get('id', ''), 'tool': name,
                'execution_status': call.get('status', 'unknown'), 'executed': executed,
                'receipt_valid': None, 'business_goal': False}
            refresh = getattr(handler, 'agent_verify_result', None)
            if (name not in grant.allowed_tools and name != turn.self_state.tool_type) or not grant.policy.accepts(name):
                row['reason'] = 'receipt_not_currently_authorized'
            elif call.get('status') == 'ok' and executed is not False and callable(refresh):
                try:
                    checked = refresh(receipt=copy.deepcopy(receipt), context=state.context)
                    valid = checked.get('result_valid') if isinstance(checked, Mapping) else None
                    if valid is None and isinstance(checked, Mapping) and checked.get('artifact_receipts'):
                        artifacts = checked['artifact_receipts']
                        if isinstance(artifacts, list) and all(isinstance(a, Mapping) for a in artifacts):
                            values = [a.get('result_valid') for a in artifacts]
                            valid = False if False in values else True if all(v is True for v in values) else None
                    row['receipt_valid'] = valid if type(valid) is bool else None
                    row['reason'] = 'receipt_checked' if valid is True else 'receipt_unconfirmed'
                except Exception:
                    row['receipt_valid'], row['reason'] = False, 'receipt_refresh_failed'
            else:
                row['reason'] = 'receipt_unconfirmed'
            # 进程退出码来自执行器认证的会话事件，不能由正文或普通result_valid替代。
            from shinku.agent.execution_sessions import execution_event
            if name in grant.allowed_tools and grant.policy.accepts(name):
                event = execution_event(handler, receipt, state.context)
                session = receipt.get('execution_session')
                if event is not None and isinstance(session, Mapping):
                    row['process'] = {k: session[k] for k in ('status', 'returncode') if k in session}
                    row['process']['receipt_authenticated'] = True
                    row['process']['command_succeeded'] = (session.get('status') == 'completed'
                        and type(session.get('returncode')) is int and session['returncode'] == 0)
                    if not receipt.get('artifact_receipts'):
                        # 失败退出也可以有真实有效的进程回执；有效不等于执行成功。
                        row['receipt_valid'], row['reason'] = True, 'process_receipt_authenticated'
            rows.append(row)
        turn.current(state.context)  # 核验期间撤权也不能继续交付为当前结果。
        return CompletionVerdict(True, status=outcome or 'completed', reason='reply_delivered',
            evidence=tuple(rows), layers={'policy': POLICY, 'reply_delivery': 'delivered',
                'goal_completion': 'not_verified', 'answer_evidence': 'not_reviewed',
                'execution_results': 'per_tool_receipts', 'model_reported_outcome': outcome or 'completed',
                'omitted_receipts': max(0, len(state.tool_calls) - 128)})
