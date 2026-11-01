"""宿主结构化调用的输出额度，供准备器/审查器到真实SDK的同步调用链使用。

总输出包含供应商计入的推理；不修改模型或思考档位。ContextVar隔离并发任务，
退出/异常恢复旧额度，嵌套只可收紧。额度不是授权或完成证明，也不能增加任务
请求数/token/时间及外部付费上限；SDK拦截器仍可进一步收紧。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OutputBudget:
    stage: str
    max_tokens: int | None


_current = ContextVar('structured_output_budget', default=None)
_PROFILES = {'preparation': ('SHINKU_PREPARATION_MAX_OUTPUT_TOKENS', None),
             'answer_review': ('SHINKU_ANSWER_REVIEW_MAX_OUTPUT_TOKENS', None)}


def current_output_budget():
    return _current.get()


@contextmanager
def structured_output_scope(stage, *, maximum=None):
    """仅宿主选择阶段；默认沿用供应商输出能力，显式配置错误仍拒绝。"""
    if stage not in _PROFILES:
        raise ValueError('unknown_structured_output_stage')
    key, default = _PROFILES[stage]
    raw = os.environ.get(key, default) if maximum is None else maximum
    # 默认不发送单次输出上限；显式隔离实验仍可按批准额度收紧。
    if raw is None:
        previous = _current.get()
        token = _current.set(OutputBudget(stage, previous.max_tokens if previous else None))
        try:
            yield _current.get()
        finally:
            _current.reset(token)
        return
    try:
        count = int(raw)
    except (TypeError, ValueError):
        raise ValueError('invalid_structured_output_limit') from None
    if isinstance(raw, bool) or str(count) != str(raw).strip() or not 256 <= count <= 32768:
        raise ValueError('invalid_structured_output_limit')
    previous = _current.get()
    budget = OutputBudget(stage, min(count, previous.max_tokens) if previous and previous.max_tokens is not None else count)
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def apply_output_budget(payload):
    """仅设置显式输出额度；默认省略参数，不抬高外部实验或供应商限制。"""
    budget = _current.get()
    if budget is None or budget.max_tokens is None:
        return payload
    result = dict(payload)
    key = 'max_completion_tokens' if 'max_completion_tokens' in result else 'max_tokens'
    existing = result.get(key)
    result[key] = min(budget.max_tokens, int(existing)) if existing is not None else budget.max_tokens
    return result
