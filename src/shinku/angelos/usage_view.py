"""Angelos 本机用量展示：接收实际 SDK 请求事件，按会话投影，不参与预算或验收。

UsageView 保存每次物理请求的最新回执，started/终态按 ID 去重；摘要、重试和
失败均沿用 request_ledger 的同一入口。read_usage 从工作台绑定解析作用域，
不允许浏览器提交任务 ID 或数据库路径。旧任务没有采集记录时不补造零消耗。
"""
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.evaluation import provider_usage, summarize_calls
from shinku.phase_timing import PHASES, clean_span, summarize_phases
from shinku.llm.cache_prefix import clean_prefix


class UsageView:
    """独立展示库的唯一写入者；不保存提示词、模型正文、工具参数或凭据。

    写入失败仅标记统计不完整，原模型调用和已有外部账本继续运行。任务恢复
    使用同一绑定，新请求使用新 ID，故包含旧尝试的真实消耗而不重复累计。
    """
    def __init__(self, path=None, *, downstream=None):
        self.lock, self.healthy, self.downstream = threading.RLock(), True, downstream
        self.phase_healthy = True
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ':memory:', check_same_thread=False)
        self.db.execute('''CREATE TABLE IF NOT EXISTS usage_calls (
            binding TEXT NOT NULL, request_id TEXT NOT NULL, status TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(binding, request_id))''')
        # 仍使用本机用量库；阶段记录不写任务账本、不参与恢复或预算。
        self.db.execute('''CREATE TABLE IF NOT EXISTS phase_timings (
            binding TEXT NOT NULL, span_id TEXT NOT NULL, status TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(binding, span_id))''')
        self.db.commit()
        if path:
            path.chmod(0o600)

    def __call__(self, event):
        try:
            # 只投影规范化字段；不把接收器的任意额外数据带进 API。
            row = {k: event.get(k) for k in ('binding', 'request_id', 'status', 'elapsed_seconds')}
            row['phase'] = event.get('phase') if event.get('phase') in PHASES else 'unclassified'
            row['prefix'] = clean_prefix(event.get('prefix'))
            started = event.get('started_at')
            row['started_at'] = started if type(started) in (int, float) and math.isfinite(started) and started >= 0 else None
            row['usage'] = {k: (event.get('usage') or {}).get(k) for k in provider_usage({})}
            if not all(isinstance(row[k], str) and row[k] for k in ('binding', 'request_id')):
                raise ValueError('invalid_usage_binding')
            check = dict(row, status='unknown' if row['status'] == 'started' else row['status'])
            summarize_calls([check])
            with self.lock, self.db:
                self.db.execute('''INSERT INTO usage_calls VALUES (?,?,?,?)
                    ON CONFLICT(binding,request_id) DO UPDATE SET
                    status=excluded.status, payload=excluded.payload
                    WHERE usage_calls.status='started' ''',
                    (row['binding'], row['request_id'], row['status'], json.dumps(row)))
        except Exception:
            self.healthy = False
        finally:
            # 外部评测/审计账本仍收到原始事件且只收到一次；观测失败由原入口隔离。
            if callable(self.downstream):
                self.downstream(event)

    def observe_phase(self, event):
        """每个阶段只留最新回执，终态不被迟到的开始事件覆盖；容量满时显式降级。"""
        try:
            row = clean_span(event)
            with self.lock, self.db:
                exists = self.db.execute('SELECT 1 FROM phase_timings WHERE binding=? AND span_id=?',
                                        (row['binding'], row['span_id'])).fetchone()
                if not exists and (self.db.execute('SELECT count(*) FROM phase_timings WHERE binding=?',
                    (row['binding'],)).fetchone()[0] >= 512
                    or self.db.execute('SELECT count(*) FROM phase_timings').fetchone()[0] >= 32768):
                    self.phase_healthy = False
                    return
                self.db.execute('''INSERT INTO phase_timings VALUES (?,?,?,?)
                    ON CONFLICT(binding,span_id) DO UPDATE SET status=excluded.status,payload=excluded.payload
                    WHERE phase_timings.status='started' ''',
                    (row['binding'], row['span_id'], row['status'], json.dumps(row)))
        except Exception:
            self.phase_healthy = False

    def read_phases(self, bindings):
        with self.lock:
            return [json.loads(row[0]) for binding in dict.fromkeys(bindings)
                    for row in self.db.execute('SELECT payload FROM phase_timings WHERE binding=? LIMIT 512', (binding,))]

    def phases_complete(self):
        # 饱和状态跨重启仍可见；不自动删除失败证据来换取“健康”状态。
        with self.lock:
            return self.phase_healthy and self.db.execute('SELECT count(*) FROM phase_timings').fetchone()[0] < 32768

    def read(self, bindings):
        with self.lock:
            rows = []
            for binding in dict.fromkeys(bindings):
                rows.extend(json.loads(r[0]) for r in self.db.execute(
                    'SELECT payload FROM usage_calls WHERE binding=?', (binding,)))
        # started 可能因进程退出而永远没有终态，不能伪装已结算或重新派发。
        return [dict(r, status='unknown' if r['status'] == 'started' else r['status']) for r in rows]


def _summarize(rows):
    """仅对已记录请求给出小计；缺失字段保留 unknown，不以缓存量重复增加总量。"""
    result = summarize_calls(rows)
    fields = ('input_tokens', 'output_tokens', 'cache_hit_tokens', 'cache_miss_tokens')
    counts = {k: result['requests'] - len(result['missing_by_field'][k]) for k in fields}
    totals = {k: result['totals'][k] if counts[k] else None for k in fields}
    known = {k: result['known_subtotals'][k] if counts[k] else None for k in fields}
    complete_pairs = [r for r in rows if all(r['usage'].get(k) is not None
                                           for k in ('input_tokens', 'output_tokens'))]
    known_total = sum(r['usage']['input_tokens'] + r['usage']['output_tokens'] for r in complete_pairs)
    total = known_total if rows and len(complete_pairs) == len(rows) else None
    # 矛盾缓存回执不展示百分比；不得用其它请求的低命中抵消本次超过输入量的问题。
    invalid_cache = any(
        r['usage'].get('input_tokens') is not None and r['usage'].get('cache_hit_tokens') is not None
        and (r['usage']['cache_hit_tokens'] > r['usage']['input_tokens']
             or (r['usage'].get('cache_miss_tokens') is not None
                 and r['usage']['cache_hit_tokens'] + r['usage']['cache_miss_tokens'] != r['usage']['input_tokens']))
        for r in rows)
    return {'requests': len(rows), 'totals': totals, 'known': known, 'reported': counts,
            'totalTokens': total, 'knownTotalTokens': known_total if complete_pairs else None,
            'cacheHitRatio': None if invalid_cache else result['cache_hit_ratio'],
            'invalidCache': invalid_cache,
            'unsettledRequests': sum(r['status'] == 'unknown' for r in rows),
            'incompleteRequests': len(rows) - len(complete_pairs)}


def summarize(rows):
    """同一物理请求只属于一组；旧记录不能倒推成普通调用或压缩后首轮。

    累计仍用全部供应商用量；逐请求诊断最多展示末20条，不影响分组/总量，
    不持久保存正文。缺回执、矛盾回执与失败按原统计语义保留。
    """
    result = _summarize(rows)
    groups = {key: [] for key in ('ordinary', 'summary', 'post_compaction', 'unclassified')}
    requests = []
    # 主键为随机请求ID，数据库索引顺序不代表调用先后；旧记录没有时钟则排前，
    # 只标记未分类，不伪造历史执行顺序。
    for index, row in enumerate(sorted(rows, key=lambda r: r.get('started_at') or 0)):
        prefix = clean_prefix(row.get('prefix'))
        category = prefix['category'] if prefix['available'] else (
            'summary' if row.get('phase') == 'compaction' else 'unclassified')
        groups[category].append(row)
        if index >= len(rows) - 20:
            requests.append({'requestId': row['request_id'], 'category': category,
                             'prefix': prefix, **_summarize([row])})
    result['cacheGroups'] = {key: _summarize(values) for key, values in groups.items()}
    result['recentRequests'] = requests[-20:]
    result['omittedRequestDetails'] = max(0, len(rows) - 20)
    return result


def read_usage(host, cid, conversation):
    """已鉴权 API 传入存储原件；再次核对任务所属会话，失败不暴露其它会话用量。"""
    if conversation.get('id') != cid:
        raise ValueError('usage_conversation_mismatch')
    rows, turns, spans = [], {}, []
    missing_turns = 0
    for turn in conversation['turns']:
        bindings = ['prepare:' + cid + ':' + turn['id']]
        if turn.get('task_id'):
            try:
                scope = AgentScope(host.requester, 'angelos:' + cid)
                with host.service._lock:
                    task = host.service._get(scope, turn['task_id'])
                    host.service._grant(scope, task, 'inspect')
                bindings.append(turn['task_id'])
            except Exception:
                turns[turn['id']] = {'available': False}
                missing_turns += 1
                continue
        calls = host.usage_view.read(bindings)
        phases = host.usage_view.read_phases(bindings)
        complete = host.usage_view.phases_complete() and all(
            sum(r['binding'] == binding for r in phases) < 512 for binding in bindings)
        turns[turn['id']] = dict(summarize(calls), available=True,
            performance=summarize_phases(phases, calls, truncated=not complete))
        rows.extend(calls)
        spans.extend(phases)
        missing_turns += not bool(calls)
    # 防止同一任务被多个历史索引引用时把消耗再加一次。
    unique = {(r['binding'], r['request_id']): r for r in rows}
    result = summarize(list(unique.values()))
    result['performance'] = summarize_phases(spans, list(unique.values()),
        truncated=any(t.get('performance', {}).get('truncated') for t in turns.values()))
    return {'available': True, 'conversationId': cid, 'sampledAt': time.time(),
            'refreshSeconds': 15, 'healthy': host.usage_view.healthy,
            'unrecordedTurns': missing_turns, 'summary': result, 'turns': turns,
            'source': 'provider_usage', 'scope': 'recorded_requests', 'is_fact_evidence': False}
