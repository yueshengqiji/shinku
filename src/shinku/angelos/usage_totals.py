"""本机 Angelos 用量库的累计投影；不连接 Bot 数据、不扫描对话或触发模型。

在专用 UsageView 上缓存15秒，SQL只聚合数值，不向前端暴露绑定、请求或正文。
缺失回执不补零；命中率只对有完整缓存分项的请求按输入token加权，并展示覆盖数。
"""
import copy
import time


FIELDS = ('input_tokens', 'output_tokens', 'cache_hit_tokens', 'cache_miss_tokens')


def read_instance_usage(view):
    """锁由原用量库持有；短聚合完成后释放，缓存不参与预算和任务验收。"""
    with view.lock:
        cached = getattr(view, '_instance_usage_cache', None)
        now = time.monotonic()
        if cached is not None and now - cached[0] < 15:
            return copy.deepcopy(cached[1])
        columns = ','.join("json_extract(payload,'$.usage.%s') AS %s" % (k, k) for k in FIELDS)
        valid = ('input_tokens IS NOT NULL AND cache_hit_tokens IS NOT NULL '
                 'AND cache_hit_tokens<=input_tokens '
                 'AND (cache_miss_tokens IS NULL OR cache_hit_tokens+cache_miss_tokens=input_tokens)')
        aggregates = ','.join('SUM(%s),COUNT(%s)' % (k, k) for k in FIELDS)
        sql = f'''WITH calls AS (SELECT status,{columns} FROM usage_calls)
            SELECT COUNT(*),{aggregates},
                SUM(CASE WHEN input_tokens IS NOT NULL AND output_tokens IS NOT NULL
                    THEN input_tokens+output_tokens END),
                SUM(CASE WHEN input_tokens IS NOT NULL AND output_tokens IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN status='started' THEN 1 ELSE 0 END),
                SUM(CASE WHEN {valid} THEN cache_hit_tokens END),
                SUM(CASE WHEN {valid} THEN input_tokens END),
                SUM(CASE WHEN {valid} THEN 1 ELSE 0 END)
            FROM calls'''
        row = view.db.execute(sql).fetchone()
        n = row[0]
        sums = dict(zip(FIELDS, row[1:9:2]))
        counts = dict(zip(FIELDS, row[2:9:2]))
        total, complete, unsettled, hit, inp, covered = row[9:]
        complete, covered = complete or 0, covered or 0
        summary = dict(requests=n, totals={k:sums[k] if n and counts[k]==n else None for k in FIELDS},
            known=sums, totalTokens=total if n and complete==n else None, knownTotalTokens=total,
            cacheHitRatio=hit/inp if inp else None, cacheCoverageRequests=covered,
            excludedCacheRequests=n-covered, incompleteRequests=n-complete, unsettledRequests=unsettled or 0)
        value = dict(available=True, scope='angelos_instance_recorded_requests',
                     sampledAt=time.time(), healthy=view.healthy, summary=summary)
        view._instance_usage_cache = (now, value)
        return copy.deepcopy(value)
