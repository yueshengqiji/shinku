"""整任务质量/成本对照的离线口径，不调用模型、不参与执行授权。

先用 provider_usage 保留供应商缺失字段，再由 summarize_calls 累计所有阶段；
评审人按固定样本写入可审查判断。预算预留、模型自评及 accepted 均非账单或质量。
价格由实验记录提供，不能把某天的价格硬编码为所有请求的费用。
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal


def _field(obj, key):
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _count(obj, *keys):
    for key in keys:
        value = _field(obj, key)
        if type(value) is int and value >= 0:
            return value
    return None


def provider_usage(response):
    """只归一化真实 usage；缓存创建量不能冒充未命中量，缺失不能写零。"""
    usage = _field(response, 'usage')
    details = _field(usage, 'completion_tokens_details')
    prompt_details = _field(usage, 'prompt_tokens_details')
    hit = _count(usage, 'prompt_cache_hit_tokens', 'cache_read_input_tokens')
    if hit is None:
        hit = _count(prompt_details, 'cached_tokens')
    return {
        'input_tokens': _count(usage, 'prompt_tokens', 'input_tokens', 'promptTokenCount'),
        'output_tokens': _count(usage, 'completion_tokens', 'output_tokens', 'candidatesTokenCount'),
        'cache_hit_tokens': hit,
        'cache_miss_tokens': _count(usage, 'prompt_cache_miss_tokens'),
        'cache_creation_tokens': _count(usage, 'cache_creation_input_tokens'),
        'reasoning_tokens': _count(details, 'reasoning_tokens'),
    }


def summarize_calls(calls):
    """调用列表由宿主记录提供，须包含失败/取消/超时；未知账单继续标未知。

    每次请求只写一条终态记录。重复 ID 和非法计数直接拒绝，避免流式累计量
    重复相加或静默漏掉失败。此函数不能自行证明调用账本已覆盖整个任务。
    """
    keys = tuple(provider_usage({}))
    ids = set()
    totals = {k: 0 for k in keys}
    missing = {k: [] for k in keys}
    statuses = {}
    latency = 0.0
    latency_known = True
    for row in calls:
        identifier = row.get('request_id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('invalid_or_duplicate_request_id')
        ids.add(identifier)
        status = row.get('status')
        if status not in {'completed', 'failed', 'cancelled', 'timeout', 'unknown'}:
            raise ValueError('invalid_call_status')
        statuses[status] = statuses.get(status, 0) + 1
        usage = row.get('usage') or {}
        for key in keys:
            value = usage.get(key)
            if value is None:
                missing[key].append(identifier)
            elif type(value) is int and value >= 0:
                totals[key] += value
            else:
                raise ValueError('invalid_usage_count')
        elapsed = row.get('elapsed_seconds')
        if elapsed is None:
            latency_known = False
        elif type(elapsed) not in {int, float} or not 0 <= elapsed < float('inf'):
            raise ValueError('invalid_elapsed_seconds')
        else:
            latency += elapsed
    result = {'requests': len(ids), 'statuses': statuses, 'known_subtotals': totals,
              'missing_by_field': missing,
              'totals': {k: None if missing[k] else totals[k] for k in keys},
              'sum_call_seconds': latency if latency_known else None,
              'wall_seconds': None, 'ledger_completeness': 'requires_host_review'}
    inp, hit = result['totals']['input_tokens'], result['totals']['cache_hit_tokens']
    result['cache_hit_ratio'] = hit / inp if inp and hit is not None and hit <= inp else None
    return result


def estimate_call_cny(usage, price):
    """显式价格快照的估算；推理计入供应商输出总量，不重复计费。

    请求时间/时段、实际模型与价格来源由记录持有者核对；不从当前别名补造
    历史版本。缓存分项不完整或矛盾时拒绝给出看似精确的费用。
    """
    if not all(price.get(k) for k in ('source', 'effective_at', 'model', 'period')):
        return None
    if any(price.get(k) is None for k in ('hit_cny_per_million', 'miss_cny_per_million', 'output_cny_per_million')):
        return None
    counts = [usage.get(k) for k in ('input_tokens', 'output_tokens', 'cache_hit_tokens', 'cache_miss_tokens')]
    if any(type(v) is not int or v < 0 for v in counts):
        return None
    inp, out, hit, miss = counts
    if hit + miss != inp:
        return None
    rates = [Decimal(str(price[k])) for k in ('hit_cny_per_million', 'miss_cny_per_million', 'output_cny_per_million')]
    if any(not rate.is_finite() or rate < 0 for rate in rates):
        raise ValueError('invalid_price')
    return str((hit * rates[0] + miss * rates[1] + out * rates[2]) / Decimal(1000000))


def compare_reviews(baseline, candidate, *, cost_tolerance=0.10):
    """按预先固定的逐项人工判断比较，不用均分掩盖关键退化。

    checks 的每项是 pass/fail/unknown；reviewer 与 evidence 是审查依据标识。
    这里只给采用建议，不切换配置或放行执行。未知/缺项/权限失败一律不采纳。
    """
    if type(cost_tolerance) not in {int, float} or not 0 <= cost_tolerance <= 1:
        raise ValueError('invalid_cost_tolerance')
    for review in (baseline, candidate):
        if (not review.get('reviewer') or not review.get('evidence')
                or review.get('hard_guards') is not True or not review.get('checks')
                or any(v not in {'pass', 'fail'} for v in review['checks'].values())):
            return 'insufficient_evidence_or_hard_guard_failure'
    a, b = baseline['checks'], candidate['checks']
    if a.keys() != b.keys():
        return 'incomparable_criteria'
    if any(a[k] == 'pass' and b[k] == 'fail' for k in a):
        return 'reject_quality_regression'
    if any(v != 'pass' for v in b.values()):
        return 'required_quality_not_met'
    improved = any(a[k] == 'fail' and b[k] == 'pass' for k in a)
    costs = [review.get('total_cny') for review in (baseline, candidate)]
    if any(v is None for v in costs):
        return 'cost_unknown_keep_baseline'
    old, new = (Decimal(str(v)) for v in costs)
    if any(not v.is_finite() or v < 0 for v in (old, new)):
        raise ValueError('invalid_total_cost')
    tolerance = Decimal(str(cost_tolerance))
    if new > old * (1 + tolerance):
        return 'quality_cost_tradeoff_needs_decision' if improved else 'reject_cost_increase'
    if improved:
        return 'adopt_with_recorded_scope'
    return 'adopt_with_recorded_scope' if new < old * (1 - tolerance) else 'no_stable_gain_keep_baseline'
