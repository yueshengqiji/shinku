"""实验模型出口：固定上游、独立批次账本和保守额度，不暴露通用网络代理。

模型密钥只存在于宿主代理。沙箱仅看到一个 Unix socket；请求不能指定 URL、
密钥、价格或模型档位。started 先落盘再发 HTTP，原稿与未知用量保留，不自动重试。
真实计费由供应商决定；价格/能力上界须由操作者核实，不把本地估算称为账单。
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal
from http.server import BaseHTTPRequestHandler
import json
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socketserver
import threading
import time
from urllib.parse import urlsplit
import uuid

import httpx

from shinku.agent_runtime.evaluation import estimate_call_cny, provider_usage
from shinku.llm.request_ledger import JsonCallLedger
from .sandbox import plain_tree

MAX_BODY = 2 * 1024 * 1024
MAX_RESPONSE = 16 * 1024 * 1024


class GatewayStopped(RuntimeError):
    """只给调用方稳定错误码，供应商原稿另存在本批次账本。"""


@dataclass(frozen=True)
class ModelRoute:
    endpoint: str
    model: str
    context_tokens: int
    max_output_tokens: int
    price: dict
    thinking: str = 'disabled'
    effort: str = ''
    # 仅供本机假上游集成；正式命令行不接受此开关。
    fixture_http: bool = False
    # 由操作者显式选择的本机代理；不接受DNS别名、任意内网地址或重定向。
    local_proxy: bool = False
    reported_model: str = ''
    billing: str = 'metered'
    billing_evidence: str = ''

    def __post_init__(self):
        url = urlsplit(self.endpoint)
        local = (self.local_proxy is True and url.scheme == 'http'
                 and url.hostname in {'127.0.0.1', '::1'} and url.port is not None
                 and url.path == '/v1/chat/completions')
        if type(self.local_proxy) is not bool or (self.local_proxy and not local):
            raise ValueError('goozam_explicit_loopback_proxy_required')
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or not url.path.endswith('/chat/completions')
                or (url.scheme != 'https' and not (self.fixture_http is True
                    and url.scheme == 'http' and url.hostname == '127.0.0.1') and not local)):
            raise ValueError('goozam_fixed_https_endpoint_required')
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 128:
            raise ValueError('goozam_model_required')
        if any(type(v) is not int or not 1 <= v <= 2000000
               for v in (self.context_tokens, self.max_output_tokens)):
            raise ValueError('goozam_model_limits_required')
        if self.thinking not in {'disabled', 'enabled'} or self.effort not in {'', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('goozam_invalid_thinking')
        if self.thinking != 'enabled' and self.effort:
            raise ValueError('goozam_effort_requires_thinking')
        if not isinstance(self.reported_model, str) or len(self.reported_model) > 128:
            raise ValueError('goozam_invalid_reported_model')
        if self.billing not in {'metered', 'no_incremental_charge'}:
            raise ValueError('goozam_invalid_billing')
        if self.billing == 'no_incremental_charge':
            if (not self.local_proxy or not isinstance(self.billing_evidence, str)
                    or not self.billing_evidence.strip() or self.price):
                raise ValueError('goozam_explicit_billing_evidence_required')
            return
        sample = dict(input_tokens=1, output_tokens=1, cache_hit_tokens=0, cache_miss_tokens=1)
        if (not isinstance(self.price, dict) or self.price.get('model') != self.model
                or estimate_call_cny(sample, self.price) is None):
            raise ValueError('goozam_explicit_price_required')

    def public(self):
        """沙箱只需模型能力；上游地址、价格和密钥无需进入模型上下文。"""
        return dict(model=self.model, context_tokens=self.context_tokens,
                    max_output_tokens=self.max_output_tokens,
                    route_fingerprint=hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest())


@dataclass(frozen=True)
class BatchLimits:
    authorization_ref: str
    requests: int
    tokens: int
    seconds: float
    cny: str

    def __post_init__(self):
        if not isinstance(self.authorization_ref, str) or not self.authorization_ref.strip():
            raise ValueError('goozam_authorization_reference_required')
        if any(type(v) is not int or v <= 0 for v in (self.requests, self.tokens)):
            raise ValueError('goozam_invalid_batch_limits')
        if (type(self.seconds) not in (int, float) or not math.isfinite(self.seconds)
                or not 0 < self.seconds <= 3600):
            raise ValueError('goozam_invalid_batch_seconds')
        amount = Decimal(str(self.cny))
        if not amount.is_finite() or amount <= 0:
            raise ValueError('goozam_invalid_batch_cost')


class ModelGateway:
    """一个全新目录对应一次批次；锁覆盖预留、HTTP、结算，禁止并发超卖额度。"""
    def __init__(self, root, route, limits, api_key, *, transport=None):
        self.root = plain_tree(root)
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.route, self.limits = route, limits
        if not isinstance(api_key, str) or not api_key.strip() or '\n' in api_key or '\r' in api_key:
            raise ValueError('goozam_explicit_key_required')
        self._key = api_key
        self.transport = transport
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.requests, self.tokens, self.cost = 0, 0, Decimal(0)
        self.stopped = ''
        self.binding = uuid.uuid4().hex
        self.ledger = JsonCallLedger(self.root / 'calls.jsonl')
        self._save('batch.json', dict(binding=self.binding, route=asdict(route), limits=asdict(limits)))

    def _save(self, name, value):
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def stop(self, reason):
        if not self.stopped:
            self.stopped = reason
            self._save('stopped.json', dict(reason=reason, requests=self.requests,
                       known_tokens=self.tokens, known_estimated_cny=str(self.cost)))
        raise GatewayStopped(self.stopped)

    def _payload(self, payload):
        allowed = {'model', 'messages', 'tools', 'tool_choice', 'temperature', 'response_format'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError('goozam_unsupported_request_fields')
        if payload.get('model') != self.route.model:
            raise ValueError('goozam_model_override_denied')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages or len(messages) > 512:
            raise ValueError('goozam_messages_required')
        # 第一版只支持文本和函数工具。拒绝远程图片/文件 URL，避免模型出口代为抓取。
        for message in messages:
            if (not isinstance(message, dict) or set(message) - {
                    'role', 'content', 'tool_calls', 'tool_call_id', 'reasoning_content', 'name'}
                    or message.get('role') not in {'system', 'user', 'assistant', 'tool'}
                    or not isinstance(message.get('content', ''), (str, type(None)))):
                raise ValueError('goozam_text_messages_only')
        tools = payload.get('tools', [])
        if not isinstance(tools, list) or len(tools) > 64 or any(
                not isinstance(t, dict) or t.get('type') != 'function' for t in tools):
            raise ValueError('goozam_function_tools_only')
        value = json.loads(json.dumps(payload, allow_nan=False))
        value['thinking'] = {'type': self.route.thinking}
        if self.route.effort:
            value['reasoning_effort'] = self.route.effort
        size = len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
        if size > MAX_BODY:
            raise ValueError('goozam_request_too_large')
        # 字节估量偏保守；能力上界另作整次请求预留，不给 SDK 塞输出截断参数。
        if size + 4096 > self.route.context_tokens:
            raise ValueError('goozam_input_reservation_exceeded')
        return value

    def complete(self, payload):
        value = self._payload(payload)
        with self.lock:
            remaining = self.limits.seconds - (time.monotonic() - self.started)
            if self.stopped:
                raise GatewayStopped(self.stopped)
            if (self.root / 'revoked').exists():
                self.stop('goozam_batch_revoked')
            if remaining <= 0:
                self.stop('goozam_batch_deadline')
            reserve = self.route.context_tokens + self.route.max_output_tokens
            rates = ([Decimal(0)] if self.route.billing == 'no_incremental_charge' else
                     [Decimal(str(self.route.price[k])) for k in (
                         'hit_cny_per_million', 'miss_cny_per_million', 'output_cny_per_million')])
            reserve_cost = reserve * max(rates) / Decimal(1000000)
            if (self.requests >= self.limits.requests or self.tokens + reserve > self.limits.tokens
                    or self.cost + reserve_cost > Decimal(str(self.limits.cny))):
                self.stop('goozam_batch_reservation_exhausted')
            started = time.monotonic()
            row = dict(binding=self.binding, request_id=uuid.uuid4().hex, status='started',
                requested_model=self.route.model, reported_model=None, request=value,
                response_text=None, http_status=None, usage=provider_usage({}),
                elapsed_seconds=None, estimated_cny=None)
            # 账本失败即拒绝派发；绝不沿用正式统计的“观测失败仍继续”语义。
            self.ledger(row)
            self.requests += 1
            failure = ''
            raw = bytearray()
            try:
                with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                                  timeout=min(remaining, 120)) as client:
                    with client.stream('POST', self.route.endpoint, json=value,
                            headers={'Authorization': 'Bearer ' + self._key,
                                     'Accept-Encoding': 'identity'}) as response:
                        row['http_status'] = response.status_code
                        for chunk in response.iter_bytes():
                            raw.extend(chunk[:MAX_RESPONSE + 1 - len(raw)])
                            if len(raw) > MAX_RESPONSE:
                                raise ValueError('goozam_response_too_large')
                            if time.monotonic() - self.started >= self.limits.seconds:
                                raise TimeoutError('goozam_batch_deadline')
                        row['response_text'] = raw.decode('utf-8', errors='replace').replace(self._key, '[REDACTED]')
                        if response.status_code != 200:
                            raise ValueError('goozam_upstream_http_error')
                result = json.loads(row['response_text'])
                row['reported_model'] = result.get('model')
                row['usage'] = provider_usage(result)
                usage = row['usage']
                if row['reported_model'] != (self.route.reported_model or self.route.model):
                    raise ValueError('goozam_reported_model_changed')
                # 免按量增费声明只影响金额，不允许缺失token或伪造缓存命中。
                if any(usage[k] is None for k in ('input_tokens', 'output_tokens')):
                    raise ValueError('goozam_usage_or_cost_unknown')
                inp = usage['input_tokens']
                hit, miss = usage['cache_hit_tokens'], usage['cache_miss_tokens']
                if (any(v is not None and v > inp for v in (hit, miss))
                        or (hit is not None and miss is not None and hit + miss != inp)):
                    raise ValueError('goozam_inconsistent_cache_usage')
                price = ('0' if self.route.billing == 'no_incremental_charge'
                         else estimate_call_cny(usage, self.route.price))
                if price is None:
                    raise ValueError('goozam_usage_or_cost_unknown')
                if (usage['input_tokens'] > self.route.context_tokens
                        or usage['output_tokens'] > self.route.max_output_tokens):
                    raise ValueError('goozam_provider_exceeded_declared_limits')
                self.tokens += usage['input_tokens'] + usage['output_tokens']
                self.cost += Decimal(price)
                row['estimated_cny'] = price
                row['status'] = 'completed'
            except Exception as exc:
                failure = str(exc) if isinstance(exc, ValueError) and str(exc).startswith('goozam_') else 'goozam_upstream_' + type(exc).__name__
                row['status'] = 'timeout' if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else 'failed'
                row['error'] = failure
                row['response_text'] = raw.decode('utf-8', errors='replace').replace(self._key, '[REDACTED]')
            row['elapsed_seconds'] = time.monotonic() - started
            try:
                self.ledger(row)
            except Exception:
                self.stopped = 'goozam_ledger_failed'
                raise GatewayStopped(self.stopped) from None
            if failure:
                self.stop(failure)
            if (self.root / 'revoked').exists():
                self.stop('goozam_batch_revoked')
            return result


class _Server(socketserver.UnixStreamServer):
    pass


def _serve(root, route, limits, key, ready):
    """单独进程持有连接。硬期限终止慢响应；started 原件继续标未知，禁止恢复重发。"""
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, limits.seconds)
    gateway = ModelGateway(root, route, limits, key)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # 请求正文/密钥不能流入默认终端日志。

        def do_POST(self):
            self.connection.settimeout(min(5, limits.seconds))
            status = 200
            payload = None
            try:
                if self.path != '/v1/chat/completions' or self.headers.get('Transfer-Encoding'):
                    raise ValueError('goozam_route_denied')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('goozam_request_size_denied')
                payload = json.loads(self.rfile.read(length))
                result = gateway.complete(payload)
            except (ValueError, GatewayStopped) as exc:
                status = 400 if isinstance(exc, ValueError) else 503
                reason = str(exc) if str(exc).startswith('goozam_') else 'goozam_invalid_json'
                # 未派发的拒绝单独留证，不混进供应商请求量；原任务仍保留具体失败。
                JsonCallLedger(gateway.root / 'rejections.jsonl')(dict(
                    reason=reason, request=payload, created=time.time()))
                result = {'error': {'message': reason}}
            except Exception:
                status, result = 503, {'error': {'message': 'goozam_gateway_error'}}
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass  # 断开的客户端不触发任何重发，已发生请求仍在宿主账本中。

    # Linux sockaddr_un 路径很短；仅这个专属子进程切目录，避免长项目路径启动失败。
    os.chdir(root)
    with _Server('model.sock', Handler) as server:
        os.chmod(Path(root) / 'model.sock', 0o600)
        ready.send(True)
        server.timeout = .1
        # 控制管道仅在宿主父子进程之间；沙箱不能伪造已关闭/已结算标记。
        while not ready.poll():
            server.handle_request()
        if ready.recv() != 'close':
            raise ValueError('goozam_invalid_control')
        gateway._save('closed.json', dict(requests=gateway.requests, ledger_healthy=gateway.ledger.healthy,
            known_tokens=gateway.tokens, known_estimated_cny=str(gateway.cost), stopped=gateway.stopped))
        ready.close()


def summarize_batch(root):
    """进程退出后回查账本；没有宿主关闭计数时完整性未知，不能靠行数自证。"""
    root = Path(root)
    metadata = json.loads((root / 'batch.json').read_text())
    closed_path = root / 'closed.json'
    closed = json.loads(closed_path.read_text()) if closed_path.exists() else {}
    ledger = JsonCallLedger(root / 'calls.jsonl')
    if not ledger.path.exists():
        return dict(requests=0, ledger_complete=closed.get('requests') == 0,
                    estimated_cny=None, status='no_recorded_model_calls', calls=[])
    result = ledger.snapshot(metadata['binding'], expected_requests=closed.get('requests'))
    result['ledger_complete'] = result['ledger_complete'] and closed.get('ledger_healthy') is True
    costs = [row.get('estimated_cny') for row in result['calls']]
    result['known_estimated_cny'] = str(sum((Decimal(v) for v in costs if v is not None), Decimal(0)))
    result['estimated_cny'] = result['known_estimated_cny'] if result['ledger_complete'] and all(v is not None for v in costs) else None
    result['stopped'] = closed.get('stopped', 'closure_unknown')
    return result


@contextmanager
def serve_gateway(root, route, limits, key):
    """只在显式调用时启动；退出/异常必回收代理进程，保留原稿目录，不能重用批次。"""
    root = Path(root).absolute()
    if root.exists():
        raise ValueError('goozam_batch_already_exists')
    root = plain_tree(root)
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe()
    process = ctx.Process(target=_serve, args=(root, route, limits, key, send), daemon=True)
    process.start()
    send.close()
    try:
        try:
            started = receive.poll(min(10, limits.seconds)) and receive.recv()
        except EOFError:
            started = False
        if not started:
            raise RuntimeError('goozam_gateway_start_failed')
        yield root / 'model.sock'
    finally:
        if process.is_alive():
            try:
                receive.send('close')
            except (OSError, EOFError):
                pass
            process.join(2)
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(3)
