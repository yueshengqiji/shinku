"""Angelos 的宿主身份与显式本地推理视图；不进入执行决策或事实账本。

用户要求展示供应商返回的推理字段。仅该独立工作台启用此适配，原生协议
仍收到原消息；展示副本脱敏、限量并按真实任务绑定，绝不回灌历史或授权。
"""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sqlite3
import threading
import time

from shinku.agent_runtime.information import safe_text, safe_source
from shinku.phase_timing import bind_observer, phase_scope
from .persona import PERSONA

# 人设与宿主事实边界分别维护；调查/纠错方法集中在input_routing，不在身份段重复。
IDENTITY = PERSONA + '\n' + (
    '工作区、当前会话与执行权限是不同概念：当前目录以本轮工作区/confirmed_constraints为准；'
    '未关联目录就明确未关联，不把服务进程目录猜成用户工作目录。历史消息帮助理解工作，不自动授权操作。'
    '工作区路径为空不等于工具没有文件访问权限，不能据此断言没有已授权文件范围；具体路径仍按宿主工具策略核验。'
    '回答身份时自然说明自己的名字与工作职责，不复述内部实现。'
    '能说明自身用途和当前工作背景，但具体模型、工具、服务状态与执行成功须依据当前宿主信息或真实回执。'
)


class ReasoningView:
    """本机展示/执行诊断的唯一写入者；推理视图限最近512个绑定、每绑定64Ki字符。

    执行正文单独追加保存，不受推理展示裁剪影响，不能作为任务事实或执行回执。

    记录不属于 AgentTaskStore，不参与证据与完成核验。浏览器只能通过已绑定
    会话取得相应任务的副本；原始协议消息仍留在原生运行时，不在这里改写。
    """
    def __init__(self, path=None):
        self.lock = threading.RLock()
        self.active_calls = {}
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ':memory:', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS reasoning (binding TEXT PRIMARY KEY, updated REAL, payload TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS model_outputs (id INTEGER PRIMARY KEY, binding TEXT NOT NULL, payload TEXT NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS model_outputs_binding ON model_outputs(binding,id)')
        self.db.commit()
        if path:
            path.chmod(0o600)

    def read(self, binding):
        with self.lock:
            row = self.db.execute('SELECT payload FROM reasoning WHERE binding=?', (binding,)).fetchone()
        return json.loads(row[0]) if row else []

    @contextmanager
    def track_call(self, binding):
        """只记录宿主正在等待模型；进程退出即失效，不把旧等待恢复成活跃调用。"""
        marker = object()
        if binding:
            with self.lock:
                self.active_calls[binding] = (marker, time.time())
        try:
            yield
        finally:
            with self.lock:
                if binding in self.active_calls and self.active_calls[binding][0] is marker:
                    self.active_calls.pop(binding)

    def activity(self, binding):
        with self.lock:
            row = self.active_calls.get(binding)
            return {'started': row[1]} if row else None

    def append(self, binding, text, *, model):
        if not binding:
            return
        with self.lock, self.db:
            rows = self.read(binding)
            available = max(0, 65536 - sum(len(r['text']) for r in rows))
            if len(rows) >= 32:
                return
            text = safe_text(text or '')
            rows.append({'model': str(model)[:256], 'text': text[:min(12000, available)],
                'omitted_chars': max(0, len(text) - min(12000, available)),
                'source': 'provider_reasoning_content', 'is_fact_evidence': False,
                'returned': bool(text), 'created': time.time()})
            self.db.execute('INSERT OR REPLACE INTO reasoning VALUES (?,?,?)',
                (binding, time.time(), json.dumps(rows, ensure_ascii=False)))
            self.db.execute('DELETE FROM reasoning WHERE binding NOT IN (SELECT binding FROM reasoning ORDER BY updated DESC LIMIT 512)')

    def append_output(self, binding, record):
        """执行正文/诊断单独保留，不加入推理视图或反馈事实；原文不随摘要改写。"""
        if not binding:
            return
        payload = json.dumps(record, ensure_ascii=False)
        with self.lock, self.db:
            self.db.execute('INSERT INTO model_outputs(binding,payload) VALUES (?,?)', (binding, payload))

    def update_stream(self, binding, snapshot):
        """同一物理流原位更新展示行；保持原32行/64Ki限制，不制造额外请求记录。"""
        if not binding:
            return
        with self.lock, self.db:
            rows = self.read(binding)
            row = next((r for r in rows if r.get('id') == snapshot['id']), None)
            if row is None:
                if len(rows) >= 32:
                    return
                row = {'id': snapshot['id'], 'created': time.time()}; rows.append(row)
            available = max(0, 65536 - sum(len(r.get('text', '')) for r in rows if r is not row))
            text = safe_text(snapshot['text'])[:min(12000, available)]
            values = {'model': str(snapshot['model'])[:256], 'text': text,
                'omitted_chars': max(0, snapshot['observed_chars'] - len(text)),
                'source': 'provider_reasoning_content', 'is_fact_evidence': False,
                'returned': bool(text), 'status': snapshot['status'], 'streaming': snapshot['status'] == 'streaming'}
            if all(row.get(k) == v for k, v in values.items()):
                return
            row.update(values, updated=time.time())
            self.db.execute('INSERT OR REPLACE INTO reasoning VALUES (?,?,?)',
                (binding, time.time(), json.dumps(rows, ensure_ascii=False)))
            self.db.execute('DELETE FROM reasoning WHERE binding NOT IN (SELECT binding FROM reasoning ORDER BY updated DESC LIMIT 512)')

    def outputs(self, binding, *, include_text=True):
        with self.lock:
            rows = self.db.execute('SELECT id,payload FROM model_outputs WHERE binding=? ORDER BY id', (binding,)).fetchall()
        values = [dict(json.loads(p), id=i) for i, p in rows]
        return values if include_text else [{k: v for k, v in r.items() if k != 'text'} for r in values[-16:]]


class AngelosRuntime:
    """转交原模型调用，串联原生消息接收器与工作台展示，不改变解析结果。"""
    def __init__(self, runtime, view, *, usage_sink=None, compaction_path=None, compaction_supported=True, context_awareness=False):
        self.runtime, self.view = runtime, view
        self.usage_sink = usage_sink
        from shinku.llm.context_compaction import ContextCompactor
        self.compactor = ContextCompactor(runtime, compaction_path, aware=context_awareness)
        # 额外摘要调用须由真实SDK逐请求计量；不对不透明自定义适配器冒称支持。
        self.compaction_supported = compaction_supported
        self.binding = ContextVar('angelos_reasoning_binding', default='')

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def context_window_tokens(self):
        from shinku.llm.model_limits import model_limits
        bundle = getattr(self.runtime, 'chat', None)
        return model_limits(bundle)['window_tokens'] if bundle is not None and self.compaction_supported else 0

    def context_budget_advisory(self, kwargs):
        """由同一规划循环索取尾部预算提示；无SDK调用、不改系统或根请求。"""
        bundle = getattr(self.runtime, 'chat', None)
        binding = self.binding.get()
        if not self.context_awareness_enabled or bundle is None or not binding:
            return ''
        snapshot = self.compactor.preview(kwargs, bundle, binding)
        if not snapshot.get('window_tokens'):
            return ''
        return self.compactor.advisory(snapshot, binding)

    def register_work_context(self, text):
        """仅当前执行绑定能登记宿主工作笔记；模型工具和外部输入没有此入口。"""
        binding = self.binding.get()
        if binding and not binding.startswith('prepare:') and self.compaction_supported:
            self.compactor.register_work_context(binding, text)

    @property
    def context_awareness_enabled(self):
        return self.compactor.aware and self.compaction_supported

    def request_context_compaction(self, binding, reason):
        """仅当前执行绑定可发出一次整理请求；最终时机/配对/容量由压缩器裁决。"""
        if not self.context_awareness_enabled or binding != self.binding.get():
            from shinku.llm.native_messages import NativeProtocolError
            raise NativeProtocolError('context_compaction_binding_changed')
        if not self.context_window_tokens():
            from shinku.llm.native_messages import NativeProtocolError
            raise NativeProtocolError('context_compaction_window_unknown')
        return self.compactor.request(binding, reason)

    @contextmanager
    def bind_task(self, binding):
        token = self.binding.set(binding)
        try:
            # 既有任务绑定覆盖整个执行切片，工具/核验也能计时；不扩大模型请求预算。
            with bind_observer(binding, getattr(self.usage_sink, 'observe_phase', None)), phase_scope(
                    'preparation' if binding.startswith('prepare:') else 'execution'):
                yield
        finally:
            self.binding.reset(token)

    def call_chat_json(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs['system_prompt'] = kwargs.get('system_prompt', '') + '\n' + IDENTITY
        model = str(getattr(getattr(self.runtime, 'chat', None), 'model', ''))
        if model:
            kwargs['system_prompt'] += '\n宿主当前执行模型：' + model
        sink = kwargs.get('provider_message_sink')
        binding = self.binding.get()
        received = False
        stream_seen = False
        bodies = []
        def incremental(snapshot):
            nonlocal stream_seen
            stream_seen = True
            snapshot = dict(snapshot)
            if snapshot.get('partial_content'):
                bodies.append(safe_text(snapshot['partial_content']))
            redact = getattr(self.runtime, '_redact_secrets', None)
            if callable(redact):
                snapshot['text'] = redact(snapshot['text'], max_chars=65536)
            self.view.update_stream(binding, snapshot)
        def observe(message):
            nonlocal received
            # 原生协议优先；展示不能替代或篡改协议回执，也不能读取其工具参数。
            if sink:
                sink(message)
            received = True
            if isinstance(message.get('content'), str):
                bodies.append(safe_text(message['content']))
            text = message.get('reasoning_content', '')
            redact = getattr(self.runtime, '_redact_secrets', None)
            if callable(redact):
                text = redact(text, max_chars=65536)
            try:
                if not stream_seen:
                    self.view.append(binding, text, model=model)
            except (sqlite3.Error, OSError):
                pass  # 展示写入失败不能把原生工具调用伪装为执行失败或触发重放。
        if binding:
            kwargs['provider_message_sink'] = observe
        from shinku.llm.request_ledger import bind_provider_calls
        # 准备/执行由既有绑定传入；可选账本接收器属于宿主，模型不能配置。
        from shinku.llm.context_compaction import compaction_scope
        from shinku.llm.cache_prefix import audit_scope
        from shinku.llm.json_stream import json_stream_scope
        stream_sink = incremental if binding and getattr(self.runtime, 'incremental_json_supported', False) is True else None
        def usage(event):
            # 展示/预算沿用原事件；校准失败只影响保守估算，不遮盖供应商回执。
            if callable(self.usage_sink):
                self.usage_sink(event)
            try:
                self.compactor.observe_usage(binding, event)
            except (ValueError, TypeError, sqlite3.Error, OSError):
                pass
        result = None
        raised = None
        try:
            with self.view.track_call(binding), bind_provider_calls(binding, usage), audit_scope(self.compactor.prefix_audit, binding), compaction_scope(self.compactor if self.compaction_supported else None, binding), json_stream_scope(stream_sink):
                result = self.runtime.call_chat_json(**kwargs)
        except Exception as exc:
            raised = exc
            raise
        finally:
            if binding:
                from shinku.llm.response_protocol import response_diagnostics
                diagnostics = response_diagnostics()
                from shinku.llm.native_messages import NativeProtocolError
                if isinstance(raised, NativeProtocolError):
                    # 协议异常只允许静态错误码进入页面诊断，不丢失断流原因或暴露原始异常正文。
                    code = str(raised)
                    if code and len(code) <= 120 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in code):
                        diagnostics['protocol_error'] = code
                if raised is not None and not diagnostics.get('call_failure'):
                    from shinku.llm.response_protocol import call_failure_metadata
                    diagnostics['call_failure'] = call_failure_metadata('service_call', raised)
                # 不返回原始消息的适配器只标记返回对象；绝不伪造供应商原文，
                # 不记录推理或原生工具参数。主错误仍由原调用栈处理，不自动重试。
                try:
                    text = '\n'.join(bodies) if bodies else (json.dumps(safe_source(result), ensure_ascii=False)
                        if result is not None and not received else '')
                except (ValueError, TypeError):
                    text = ''
                    diagnostics['archive_failure'] = 'adapter_output_not_safe_serializable'
                try:
                    self.view.append_output(binding, {'text': text,
                        'source': 'provider_message_content' if bodies else 'adapter_return_object' if text else 'no_content',
                        'diagnostics': diagnostics, 'is_fact_evidence': False, 'created': time.time()})
                except (sqlite3.Error, OSError):
                    pass  # 诊断存储异常不能覆盖主调用失败或触发工具/模型重放。
        if binding and not received and not stream_seen:
            try:
                self.view.append(binding, '', model=model)
            except (sqlite3.Error, OSError):
                pass
        return result
