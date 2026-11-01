"""上下文容量提示与增长预测；不持有任务权限，不修改供应商输出上限。

ContextCompactor持有校准和整理状态；本模块仅作无副作用估算。实际用量只说明
上一物理请求，下一请求始终标为估算。缓存命中量不从上下文占用中扣除。
"""
import copy
import json
import math

from .context_budget import ContextBudgetPolicy, measure_payload
from .model_limits import model_limits

ADVISORY = '[Host context budget advisory; historical readings are not requirements]\n'
GUIDANCE = (
    '宿主在消息尾部提供上下文容量估算，不是累计费用预算或新增授权。'
    '根据预计剩余空间、下一工具返回量和任务阶段决定是否整理；'
    '可在工具调用与回执完整配对后调用request_context_compaction，说明原因。'
    '不必每轮整理或重复读取；接近下一请求90%边界时宿主会先整理。'
    '整理不意味着任务完成，不重放已执行操作，不把旧摘要当成当前事实。'
)


def projected_payload(kwargs):
    """预览当前原生链；SDK仍会对最终payload作独立硬检查。"""
    messages = [{'role': 'system', 'content': '\n\n'.join(
        [kwargs.get('system_prompt', ''), *(kwargs.get('system_extra_blocks') or [])])}]
    messages += copy.deepcopy(kwargs.get('history_turns') or [])
    content = kwargs.get('user_prompt', '')
    images = kwargs.get('user_images') or []
    if images:
        content = [{'type': 'text', 'text': content}, *copy.deepcopy(images)]
    messages.append({'role': 'user', 'content': content})
    messages += copy.deepcopy(kwargs.get('native_history') or [])
    return {'messages': messages, 'tools': kwargs.get('native_tools') or []}


def capacity(payload, bundle, previous=None):
    """保守输出/增长预留只参与调度；不生成max_tokens/max_completion_tokens参数。"""
    previous = previous or {}
    limits = model_limits(bundle);window = limits['window_tokens']
    raw = measure_payload(payload, ContextBudgetPolicy())['total']
    multiplier = previous.get('estimate_multiplier', 1.0)
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier < 1:
        multiplier = 1.0
    used = math.ceil(raw * multiplier)
    if not window:
        return {'window_tokens': 0, 'estimated_tokens': used, 'measurement': 'estimate',
                'remaining_tokens': None, 'status': 'window_unknown', 'threshold': .9,
                'capacity_source': limits['source'], 'automatic_compaction': False}
    policy = ContextBudgetPolicy.from_environment()
    observed_output = previous.get('provider_output_tokens')
    observed_output = observed_output if type(observed_output) is int and observed_output >= 0 else 0
    observed_growth = previous.get('observed_growth_tokens')
    observed_growth = observed_growth if type(observed_growth) is int and observed_growth >= 0 else 0
    output = max(window // 50, min(policy.output_reserve_tokens, window // 10), observed_output * 2)
    if limits['max_output_tokens']:
        output = min(output, limits['max_output_tokens'])
    margin = max(window // 100, min(policy.safety_margin_tokens, max(1, window // 50)))
    growth = max(window // 100, min(window // 10, observed_growth))
    reserve = output + margin + growth
    return {'window_tokens': window, 'estimated_tokens': used, 'raw_estimated_tokens': raw,
            'capacity_source': limits['source'], 'automatic_compaction': True,
            'measurement': 'estimate', 'estimate_multiplier': multiplier,
            'provider_input_tokens': previous.get('provider_input_tokens'),
            'provider_output_tokens': previous.get('provider_output_tokens'),
            'output_reserve_tokens': output, 'safety_margin_tokens': margin,
            'next_growth_tokens': growth, 'remaining_tokens': max(0, window - used - reserve),
            'projected_tokens': used + reserve, 'ratio': used/window, 'threshold': .9,
            'status': 'compact_before_next_request' if used + reserve >= window*.9 else 'within_budget'}


def advice(snapshot):
    """只输出短元数据，固定键序；旧快照不是待保留的用户硬条件。"""
    fields = ('window_tokens', 'estimated_tokens', 'remaining_tokens', 'output_reserve_tokens',
              'next_growth_tokens', 'measurement', 'status')
    return ADVISORY + json.dumps({k: snapshot.get(k) for k in fields}, ensure_ascii=False)
