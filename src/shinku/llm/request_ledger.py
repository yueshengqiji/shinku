"""可选的供应商请求账本钩子，位于预算通过后的实际 SDK create 边界。

宿主给出绑定和接收器；不绑定时原样转发。每次物理尝试先记录 started，结束
再以同 ID 更新，重试不会漏计；进程中断的 started 必须保持未知。流式只取最终
累计 usage，不把每个 chunk 相加。记录不含提示词、工具参数、URL 或异常正文。
接收器失败不改变执行结果，宿主必须自行记录账本不完整，不能据此宣称省钱。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import time
import uuid
import json
import os
from pathlib import Path
import threading

from shinku.agent_runtime.evaluation import provider_usage

_binding = ContextVar('shinku_provider_request_ledger', default=None)


def provider_call_context():
    """供宿主表达线程显式续接观测，不携带预算或执行权限。"""
    return _binding.get() or ('', None)


class JsonCallLedger:
    """受限本机原始事件账本；同一请求终态按 ID 汇总，不修改/删除旧行。

    路径仅由宿主决定，权限 0600；不自动清理或改变既有数据库 TTL。接收失败
    标为不完整；重启复核还必须与独立的预期调用数比对，不能凭空推断没有漏记。
    """
    def __init__(self, path):
        self.path, self.lock, self.healthy = Path(path), threading.RLock(), True

    def __call__(self, row):
        try:
            with self.lock:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, 'a', encoding='utf-8', closefd=False) as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        os.fsync(fd)
                finally:
                    os.close(fd)
        except Exception:
            self.healthy = False
            raise

    def snapshot(self, binding, *, expected_requests=None):
        from shinku.agent_runtime.evaluation import summarize_calls
        with self.lock:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    raw = stream.read(32 * 1024 * 1024 + 1)
            finally:
                os.close(fd)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError('ledger_read_over_limit')
        rows = {}
        for line in raw.splitlines():
            row = json.loads(line)
            if row.get('binding') == binding:
                rows[row['request_id']] = row
        for row in rows.values():
            if row['status'] == 'started':
                row['status'] = 'unknown'
        result = summarize_calls(list(rows.values()))
        result.update(calls=list(rows.values()), ledger_complete=self.healthy and type(expected_requests) is int
                      and expected_requests == len(rows), expected_requests=expected_requests)
        return result


@contextmanager
def bind_provider_calls(binding, sink):
    token = _binding.set((binding, sink) if binding and callable(sink) else None)
    try:
        yield
    finally:
        _binding.reset(token)


def observed_completion(create, payload):
    bound = _binding.get()
    if bound is None:
        return create(**payload)
    binding, sink = bound
    from shinku.phase_timing import current_phase
    phase, span_id = current_phase()
    started = time.monotonic()
    extra = payload.get('extra_body') or {}
    row = {'binding': binding, 'request_id': uuid.uuid4().hex, 'status': 'started',
           'requested_model': str(payload.get('model') or '')[:256], 'reported_model': None,
           'reasoning_effort': payload.get('reasoning_effort') or extra.get('reasoning_effort'),
           'started_at': time.time(), 'elapsed_seconds': None, 'usage': provider_usage({}),
           'stream': bool(payload.get('stream')), 'span_id': span_id,
           'phase': phase if phase != 'unclassified' else 'preparation' if binding.startswith('prepare:') else 'execution'}
    from .cache_prefix import diagnose_request
    diagnostic = diagnose_request(payload)
    if diagnostic is not None:
        row['prefix'] = diagnostic

    def report(status, response=None):
        current = deepcopy(row)
        current.update(status=status, elapsed_seconds=None if status == 'started' else time.monotonic() - started,
                       usage=provider_usage(response))
        value = response.get('model') if isinstance(response, dict) else getattr(response, 'model', None)
        current['reported_model'] = value[:256] if isinstance(value, str) else None
        try:
            sink(current)
        except Exception:
            pass  # 观测故障不能改变已执行效果；完整性由宿主接收器另行记录。
    report('started')
    try:
        response = create(**payload)
    except BaseException as exc:
        report('cancelled' if isinstance(exc, (GeneratorExit, KeyboardInterrupt)) else
               'timeout' if 'timeout' in type(exc).__name__.lower() else 'failed')
        raise
    if not payload.get('stream'):
        report('completed', response)
        return response

    def stream():
        latest, finished = None, False
        try:
            for chunk in response:
                usage = chunk.get('usage') if isinstance(chunk, dict) else getattr(chunk, 'usage', None)
                if usage is not None:
                    latest = chunk
                yield chunk
            finished = True
        except GeneratorExit:
            raise
        finally:
            try:
                close = getattr(response, 'close', None)
                if callable(close):
                    close()
            finally:
                report('completed' if finished else 'unknown', latest)
    return stream()
