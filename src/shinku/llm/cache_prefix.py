"""SDK派发前的结构前缀诊断；只保存摘要和长度，不保存正文或推理。

Angelos宿主绑定观察器，request_ledger在预算通过后调用。SQLite持有每个绑定
最近一次实际请求的签名；它不是供应商tokenizer或缓存命中证明，也不参与授权。
摘要的payload覆盖只在同步内部调用中生效，退出后恢复，不修改原生历史。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json

_observer = ContextVar('cache_prefix_observer', default=None)
_summary_payload = ContextVar('cache_summary_payload', default=None)
ENVELOPE_FIELDS = ('model', 'tools', 'system_extra_blocks', 'thinking', 'reasoning_effort',
                   'extra_body', 'response_format', 'tool_choice', 'temperature',
                   'max_tokens', 'max_completion_tokens', 'prompt_cache_key', 'prompt_cache_retention')


def clean_prefix(value):
    """用量API只准出结构字段；未知/损坏诊断不伪装成功，也不回传任意字符串。"""
    if not isinstance(value, dict) or value.get('available') is not True:
        return {'available': False}
    category = value.get('category')
    integers = ('matchedMessages', 'matchedMessageBytes', 'messageCount')
    index = value.get('firstChangedMessage')
    changes = value.get('changedEnvelope')
    if (category not in ('ordinary', 'summary', 'post_compaction')
            or any(type(value.get(k)) is not int or value[k] < 0 for k in integers)
            or type(value.get('baseline')) is not bool
            or index is not None and (type(index) is not int or index < 0)
            or not isinstance(changes, list) or any(k not in ENVELOPE_FIELDS for k in changes)):
        return {'available': False}
    return {**{k: value[k] for k in (*integers, 'baseline')}, 'available': True,
            'category': category, 'firstChangedMessage': index, 'changedEnvelope': list(changes),
            'measurement': 'canonical_json_complete_messages', 'providerCacheEvidence': False}


@contextmanager
def audit_scope(observer, binding):
    token = _observer.set((observer, binding) if binding else None)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def summary_payload_scope(payload):
    token = _summary_payload.set(payload)
    try:
        yield
    finally:
        _summary_payload.reset(token)


def summary_payload(bundle):
    value = _summary_payload.get()
    if value is None:
        return None
    if value['model'] != bundle.model:
        from .native_messages import NativeProtocolError
        raise NativeProtocolError('context_summary_model_changed')
    return deepcopy(value)


def diagnose_request(payload):
    bound = _observer.get()
    if not bound:
        return None
    try:
        return bound[0].observe_request(bound[1], payload)
    except Exception:
        # 诊断异常不能重放请求；统计中明确不可用，不冒充零变化。
        return {'available': False}


def _signature(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()
    return [hashlib.sha256(raw).hexdigest(), len(raw)]


class PrefixAudit:
    """仅保存规范JSON的完整消息签名；不把字节长度冒充公共前缀token数。

    参数的顺序不影响这里的结构比较，但供应商序列化仍可能不同。一次物理重试
    也有独立记录；未通过预算的请求不会更新这个观察基线。锁不跨SDK调用。
    """
    def __init__(self, db, lock, status):
        self.db, self.lock, self.status = db, lock, status
        with lock, db:
            db.execute('CREATE TABLE IF NOT EXISTS context_request_prefix (binding TEXT PRIMARY KEY, payload TEXT)')

    def observe_request(self, binding, payload):
        from shinku.phase_timing import current_phase
        messages = payload.get('messages') or []
        if len(messages) > 4096:
            return {'available': False}
        # tools和生成配置也单列：消息没变不代表请求封装/路由条件完全相同。
        fields = ENVELOPE_FIELDS
        signature = {'envelope': {key: _signature(payload.get(key)) for key in fields},
                     'messages': [_signature(m) for m in messages]}
        summary = current_phase()[0] == 'compaction'
        generation = self.status(binding).get('generation', '')
        with self.lock, self.db:
            row = self.db.execute('SELECT payload FROM context_request_prefix WHERE binding=?', (binding,)).fetchone()
            previous = json.loads(row[0]) if row else {}
            category = 'summary' if summary else ('post_compaction' if generation and
                generation != previous.get('execution_generation') else 'ordinary')
            prior = previous.get('signature')
            changed = [key for key in fields if prior and signature['envelope'][key] != prior['envelope'][key]]
            matched, byte_count = 0, 0
            if prior:
                for old, new in zip(prior['messages'], signature['messages']):
                    if old != new:
                        break
                    matched += 1
                    byte_count += new[1]
            result = {'available': True, 'category': category, 'baseline': bool(prior),
                      'matchedMessages': matched, 'matchedMessageBytes': byte_count,
                      'messageCount': len(messages), 'changedEnvelope': changed,
                      'firstChangedMessage': matched if prior and
                          (matched < len(messages) or matched < len(prior['messages'])) else None,
                      'measurement': 'canonical_json_complete_messages', 'providerCacheEvidence': False}
            saved = {'signature': signature, 'execution_generation':
                     previous.get('execution_generation', '') if summary else generation}
            self.db.execute('INSERT OR REPLACE INTO context_request_prefix VALUES (?,?)',
                            (binding, json.dumps(saved)))
        return result
