"""显式MCP宿主：官方SDK持有stdio协议，本模块持有线程、权限和版本快照。

阅读顺序：StdioConfig -> MCPTransport.describe -> admit -> bridge/call -> close。
发现的schema是外部资料，只有宿主逐项接纳后才可调用。构造与导入不启动服务，
SDK为可选依赖；断线、超时及取消不重连、不重试，可能已发生的效果交给业务账本核对。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
from types import MappingProxyType

from .external import ExternalToolBridge

SDK_VERSION = '1.30.0'
MAX_BYTES = 2 * 1024 * 1024


class MCPTransportError(ValueError):
    """仅暴露固定原因码；may_have_effect不能当作已成功或可安全重发。"""
    def __init__(self, code, *, may_have_effect=False):
        self.code, self.may_have_effect = code, may_have_effect
        super().__init__(code)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StdioConfig:
    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict, repr=False)
    timeout: float = 20
    stderr_path: str | None = None

    def __post_init__(self):
        if (not isinstance(self.command, str) or not Path(self.command).is_absolute()
                or not isinstance(self.args, tuple) or len(self.args) > 64
                or any(not isinstance(v, str) or len(v) > 4096 or '\0' in v for v in self.args)
                or type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or not 0 < self.timeout <= 60
                or self.cwd is not None and (not isinstance(self.cwd, str) or not Path(self.cwd).is_absolute())
                or not isinstance(self.env, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) or '\0' in k + v for k, v in self.env.items())):
            raise ValueError('mcp_stdio_configuration_invalid')
        object.__setattr__(self, 'env', MappingProxyType(dict(self.env)))


class MCPTransport:
    """单个后台线程/异步任务拥有SDK会话；Agent工作线程只提交有界请求。

    authorize由宿主注入，不从工具参数产生。接纳表按完整schema摘要冻结，
    每次工具调用前重新发现，防服务悄悄改schema或把新工具加入权限。
    """
    def __init__(self, config, *, authorize):
        if not isinstance(config, StdioConfig) or not callable(authorize):
            raise ValueError('mcp_host_binding_required')
        self.config, self.authorize = config, authorize
        self._thread, self._loop, self._queue = None, None, None
        self._ready, self._closing = threading.Event(), threading.Event()
        self._lock = threading.RLock()
        self._approved, self._error = {}, ''
        self._initialized = False

    def _start(self):
        if self._closing.is_set():
            raise MCPTransportError('mcp_closed')
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name='mcp-stdio', daemon=True)
            self._thread.start()
        if not self._ready.wait(self.config.timeout + 2) or not self._initialized:
            self.close()
            raise MCPTransportError(self._error or 'mcp_initialize_failed')

    def _worker(self):
        try:
            asyncio.run(self._serve())
        except Exception:
            self._error = self._error or 'mcp_connection_failed'
        finally:
            self._initialized = False
            self._ready.set()

    async def _serve(self):
        # 延迟导入，公共检出和未启用MCP的现有入口不需要安装SDK。
        try:
            if importlib.metadata.version('mcp') != SDK_VERSION:
                self._error = 'mcp_sdk_version_mismatch'
                return
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except (ImportError, importlib.metadata.PackageNotFoundError):
            self._error = 'mcp_sdk_unavailable'
            return
        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        path = self.config.stderr_path
        if path:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                os.close(fd)
                raise MCPTransportError('mcp_stderr_file_invalid')
            os.fchmod(fd, 0o600)
            log = os.fdopen(fd, 'w')
        else:
            log = open(os.devnull, 'w')
        params = StdioServerParameters(command=self.config.command, args=list(self.config.args),
            cwd=self.config.cwd, env=dict(self.config.env))
        with log:
            async with stdio_client(params, errlog=log) as streams:
                async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=self.config.timeout)) as session:
                    async with asyncio.timeout(self.config.timeout):
                        initialized = await session.initialize()
                    if initialized.capabilities.tools is None:
                        self._error = 'mcp_tools_capability_missing'
                        return
                    self._initialized = True
                    self._ready.set()
                    while not self._closing.is_set():
                        job = await self._queue.get()
                        if job is None:
                            break
                        method, arguments, guard, result = job
                        started = False
                        task = None
                        try:
                            guard()
                            started = method == 'call'
                            operation = session.call_tool(**arguments) if started else self._discover(session)
                            task = asyncio.create_task(operation)
                            deadline = time.monotonic() + self.config.timeout
                            while not task.done():
                                guard()
                                if self._closing.is_set():
                                    raise MCPTransportError('mcp_closed')
                                if time.monotonic() >= deadline:
                                    raise MCPTransportError('mcp_timeout')
                                if path and os.fstat(log.fileno()).st_size > MAX_BYTES:
                                    raise MCPTransportError('mcp_stderr_limit')
                                await asyncio.wait({task}, timeout=.03)
                            value = task.result()
                            guard()
                            value = value.model_dump(mode='json', exclude_none=True) if started else value
                            if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_BYTES:
                                raise MCPTransportError('mcp_result_oversized')
                            result.set_result(value)
                        except Exception as exc:
                            if task is not None and not task.done():
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                            code = exc.code if isinstance(exc, MCPTransportError) else 'mcp_request_failed'
                            result.set_exception(MCPTransportError(code, may_have_effect=started))
                    # 队列中的调用尚未发送；关闭只结束会话，不恢复调用。
                    while not self._queue.empty():
                        job = self._queue.get_nowait()
                        if job is not None:
                            job[-1].set_exception(MCPTransportError('mcp_closed'))

    async def _discover(self, session):
        result, cursor, seen = [], None, set()
        for _ in range(8):
            page = await session.list_tools(cursor=cursor)
            for tool in page.tools:
                if tool.name in seen or len(result) >= 128:
                    raise MCPTransportError('mcp_tool_catalog_invalid')
                seen.add(tool.name)
                value = tool.model_dump(mode='json', exclude_none=True)
                if len(json.dumps(value).encode()) > 65536:
                    raise MCPTransportError('mcp_tool_schema_oversized')
                result.append(value)
            if not page.nextCursor:
                return result
            if page.nextCursor == cursor:
                break
            cursor = page.nextCursor
        raise MCPTransportError('mcp_tool_catalog_incomplete')

    def _request(self, method, arguments, guard=lambda: None):
        guard()
        self._start()
        result = Future()
        asyncio.run_coroutine_threadsafe(self._queue.put((method, arguments, guard, result)), self._loop).result(2)
        try:
            return result.result(self.config.timeout + 4)
        except TimeoutError as exc:
            # 不能证明服务器未执行；调用者必须将写入保留为未知。
            self._closing.set()
            raise MCPTransportError('mcp_worker_unresponsive', may_have_effect=method == 'call') from exc

    def describe(self):
        """只供可信宿主发现；返回值仍未取得调用资格。"""
        with self._lock:
            return self._request('list', {})

    def admit(self, expected):
        """显式接纳逐工具摘要；远端更新不会自动改这份宿主白名单。"""
        with self._lock:
            current = {row['name']: row for row in self.describe()}
            if not expected or any(name not in current or fingerprint(current[name]) != digest for name, digest in expected.items()):
                raise MCPTransportError('mcp_tool_schema_changed')
            from jsonschema import Draft202012Validator
            for name in expected:
                def inspect(value):
                    if isinstance(value, dict):
                        for key in ('$ref', '$dynamicRef', '$recursiveRef'):
                            if key in value and not str(value[key]).startswith('#'):
                                raise MCPTransportError('mcp_external_schema_reference_denied')
                        for item in value.values(): inspect(item)
                    elif isinstance(value, list):
                        for item in value: inspect(item)
                # SDK也可能验证outputSchema；只检查入参会遗留通过输出schema
                # 远程解析引用的入口。两端均限本地引用，禁止隐式网络读取。
                for key in ('inputSchema', 'outputSchema'):
                    schema = current[name].get(key)
                    if schema is not None:
                        Draft202012Validator.check_schema(schema)
                        inspect(schema)
            self._approved = {name: current[name] for name in expected}

    def call(self, *, tool_name, arguments, context=None):
        from shinku.agent.budget import current_task_budget
        budget = current_task_budget()
        def guard():
            if not self.authorize(context, tool_name):
                raise MCPTransportError('mcp_authorization_denied')
            if isinstance(context, dict):
                for key in ('agent_host_still_authorized', 'agent_task_interrupt'):
                    check = context.get(key)
                    if callable(check) and (not check() if key.endswith('authorized') else check()):
                        raise MCPTransportError('mcp_authorization_changed')
            if budget is not None and budget.stop_reason():
                raise MCPTransportError('mcp_budget_exhausted')
        guard()
        with self._lock:
            row = self._approved.get(tool_name)
            if row is None:
                raise MCPTransportError('mcp_tool_not_admitted')
            from jsonschema import Draft202012Validator
            if not Draft202012Validator(row['inputSchema']).is_valid(arguments):
                raise MCPTransportError('mcp_arguments_invalid')
            current = {item['name']: item for item in self._request('list', {}, guard)}
            if tool_name not in current or fingerprint(current[tool_name]) != fingerprint(row):
                raise MCPTransportError('mcp_tool_schema_changed')
            return self._request('call', {'name': tool_name, 'arguments': arguments}, guard)

    def bridge(self, *, host_id, descriptors):
        descriptors = tuple(descriptors)
        if any(row.name not in self._approved for row in descriptors):
            raise MCPTransportError('mcp_tool_not_admitted')
        return ExternalToolBridge(host_id=host_id, transport=self, descriptors=descriptors)

    def health(self):
        return {'ready': self._initialized and not self._closing.is_set(),
                'admitted_tools': sorted(self._approved), 'reason': self._error,
                'business_success': 'not_verified'}

    def close(self):
        self._closing.set()
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(self.config.timeout + 4)
