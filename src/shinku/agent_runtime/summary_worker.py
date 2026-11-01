"""显式按需的后台摘要；默认无模型，不进入任务执行或正常交付的关键路径。

宿主注入读取/授权、预算准入和生成回调。本模块持有独立摘要表；任务、原稿、
权限及实际执行记录只读。一个结果版本只尝试一次，超时/失败回退原报告。
生成回调必须遵守deadline并负责请求/用量计量；本模块不把准入当作无限预算。
超时结果不提交，未返回的生成器继续占用有界槽位，禁止自动增加并发。
"""
import copy
from dataclasses import asdict
import json
import math
import threading
import time

from .contracts import AgentBoundaryError, canonical_hash
from .delivery_brief import parse_brief, select_brief, summary_context


class SummaryWorker:
    def __init__(self, store, load_result, *, generate=None, admit=None, timeout_seconds=30,
                 max_input_chars=96000, max_workers=2, clock=time.time):
        if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 120 or type(max_workers) is not int or max_workers not in (1, 2)
                or type(max_input_chars) is not int or max_input_chars < 1):
            raise ValueError('invalid_summary_limits')
        self.store, self.load_result, self.generate, self.admit = store, load_result, generate, admit
        self.timeout, self.max_input, self.clock = timeout_seconds, max_input_chars, clock
        self.slots = threading.BoundedSemaphore(max_workers)
        with store._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS agent_delivery_summaries '
                       '(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)')

    def _original(self, scope, task_id, result_hash):
        # 每次读取（包含已完成缓存）都重新检查当前授权及版本，不能靠后台旧快照放行。
        result = self.load_result(scope, task_id)
        if result.get('scope') != asdict(scope) or result.get('task_id') != task_id:
            raise AgentBoundaryError('summary_scope_mismatch')
        if result.get('result_hash') != result_hash:
            raise AgentBoundaryError('summary_result_changed')
        summary_context(result)  # 拒绝投影或伪造原件；数据不作为指令。
        if not result.get('execution', {}).get('result_available'):
            raise AgentBoundaryError('summary_result_unavailable')
        return result

    def _key(self, scope, task_id, result_hash):
        return canonical_hash(['summary-v1', asdict(scope), task_id, result_hash])

    @staticmethod
    def _dump(value):
        body = {k: v for k, v in value.items() if k != 'record_hash'}
        return json.dumps({**body, 'record_hash': canonical_hash(body)}, ensure_ascii=False)

    @staticmethod
    def _decode(raw, key):
        try:
            value = json.loads(raw)
            digest = value.pop('record_hash')
            if value.get('cache_key') == key and canonical_hash(value) == digest:
                return value
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
        # 损坏的摘要可弃用，原任务/报告不作失败或重执行处理。
        return {'status': 'fallback', 'reason': 'summary_cache_invalid'}

    def _read(self, key):
        with self.store._transaction() as db:
            row = db.execute('SELECT payload FROM agent_delivery_summaries WHERE cache_key=?', (key,)).fetchone()
            value = self._decode(row[0], key) if row else None
            if value and value['status'] == 'pending' and self.clock() >= value['deadline']:
                value.update(status='fallback', reason='summary_timeout')
                db.execute('UPDATE agent_delivery_summaries SET payload=? WHERE cache_key=?',
                           (self._dump(value), key))
        return value

    def read(self, scope, task_id, result_hash):
        original = self._original(scope, task_id, result_hash)
        brief = select_brief(original)
        if brief is not None:
            return self._same_turn(brief, task_id, result_hash)
        return self._read(self._key(scope, task_id, result_hash)) or {
            'status': 'not_requested', 'source_result_hash': result_hash}

    @staticmethod
    def _same_turn(brief, task_id, result_hash):
        return {'source_result_hash': result_hash, 'details_reference': {'task_id': task_id, 'result_hash': result_hash},
                'is_fact_evidence': False, 'status': 'ready', 'brief': {k: brief[k] for k in ('mode', 'text')},
                'source': 'execution_model'}

    def request(self, scope, task_id, result_hash):
        """明确请求才排队；读handoff/status不会触发。返回立即，不等待模型。"""
        original = self._original(scope, task_id, result_hash)
        key = self._key(scope, task_id, result_hash)
        cached = self._read(key)
        if cached:
            return cached
        base = {'source_result_hash': result_hash, 'details_reference': {'task_id': task_id, 'result_hash': result_hash},
                'is_fact_evidence': False}
        brief = select_brief(original)
        if brief is not None:
            return self._same_turn(brief, task_id, result_hash)
        existing = original.get('delivery_brief')
        if (isinstance(existing, dict) and existing.get('mode') == 'full') or original.get('verified_answer'):
            return {**base, 'status': 'fallback', 'reason': 'full_report_required'}
        if not callable(self.generate) or not callable(self.admit):
            return {**base, 'status': 'fallback', 'reason': 'summary_not_configured'}
        packet = summary_context(original)
        if len(json.dumps(packet, ensure_ascii=False)) > self.max_input:
            return {**base, 'status': 'fallback', 'reason': 'summary_context_oversized'}
        if not self.slots.acquire(blocking=False):
            return {**base, 'status': 'fallback', 'reason': 'summary_worker_busy'}
        row = {**base, 'cache_key': key, 'status': 'pending', 'deadline': self.clock() + self.timeout}
        try:
            with self.store._transaction() as db:
                existing = db.execute('SELECT payload FROM agent_delivery_summaries WHERE cache_key=?', (key,)).fetchone()
                if existing:
                    self.slots.release()
                    return self._decode(existing[0], key)
                db.execute('INSERT INTO agent_delivery_summaries VALUES (?,?)', (key, self._dump(row)))
            threading.Thread(target=self._run, args=(scope, task_id, result_hash, key, packet, row),
                             name='agent-delivery-summary', daemon=True).start()
        except Exception:
            self.slots.release()
            raise
        return row

    def _run(self, scope, task_id, result_hash, key, packet, row):
        """摘要异常只落本模块状态；禁止修改任务、自动重试或发送消息。"""
        def read_details():
            if self.clock() >= row['deadline']:
                raise AgentBoundaryError('summary_timeout')
            return copy.deepcopy(self._original(scope, task_id, result_hash))
        try:
            read_details()
            if self.admit(copy.deepcopy(packet), deadline=row['deadline']) is not True:
                raise AgentBoundaryError('summary_budget_denied')
            read_details()
            value = self.generate(copy.deepcopy(packet), deadline=row['deadline'], read_details=read_details)
            read_details()
            brief = parse_brief(value)
            if brief is None:
                raise AgentBoundaryError('summary_output_invalid')
            outcome = {**row, 'status': 'ready', 'brief': brief, 'source': 'background_summary_model'}
        except Exception as exc:
            outcome = {**row, 'status': 'fallback',
                       'reason': exc.code if isinstance(exc, AgentBoundaryError) else 'summary_generation_failed'}
        try:
            with self.store._transaction() as db:
                current = db.execute('SELECT payload FROM agent_delivery_summaries WHERE cache_key=?', (key,)).fetchone()
                if current and self._decode(current[0], key)['status'] == 'pending':
                    if self.clock() >= row['deadline']:
                        outcome = {**row, 'status': 'fallback', 'reason': 'summary_timeout'}
                    db.execute('UPDATE agent_delivery_summaries SET payload=? WHERE cache_key=?',
                               (self._dump(outcome), key))
        finally:
            self.slots.release()
