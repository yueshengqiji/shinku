"""实例级插件生命周期：显式选择 → 准备 → 发布 → 停用 → 排空后关闭。

管理器拥有运行记录和注册项，插件拥有自身资源。工具包装器只检查插件生命周期，
不查看用户文字、不修改调用参数、不新增审批。停用不会强杀同进程调用，也不声称
撤销已发生的副作用；在途调用完成后才关闭插件资源，不自动重发或重试。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import importlib
import importlib.metadata
import threading
from typing import Any

from shinku.tools.registry import ToolRegistry
from .contracts import PluginContext, PluginContributions, PluginDefinition


class PluginUnavailable(RuntimeError):
    """只表示插件已停用；宿主按原有工具失败规则处理，不把它当成用户违规。"""


class _Lease:
    """只锁调用计数与停止标记；执行和资源关闭都在锁外，避免阻塞其他调用。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = 0
        self.stopping = False
        self.cleanup = None

    def enter(self):
        with self.lock:
            if self.stopping:
                raise PluginUnavailable("plugin_disabled")
            self.running += 1

    def leave(self):
        cleanup = None
        with self.lock:
            self.running -= 1
            if self.stopping and not self.running:
                cleanup, self.cleanup = self.cleanup, None
        if cleanup is not None:
            cleanup()

    def stop(self, cleanup):
        with self.lock:
            self.stopping = True
            self.cleanup = cleanup
            if self.running:
                return
            self.cleanup = None
        cleanup()


class _PluginTool:
    """保留 handler 全部元数据；过期 AgentLoop 即使持有旧 handler 也不能新执行。"""

    def __init__(self, handler, lease):
        self.handler, self.lease = handler, lease

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def normalize_call(self, call):
        return self.handler.normalize_call(call)

    def execute(self, *, call, context):
        self.lease.enter()
        try:
            return self.handler.execute(call=call, context=context)
        finally:
            self.lease.leave()


@dataclass
class _Record:
    definition: PluginDefinition
    status: str = "pending"
    reason: str = ""
    cleanup_error: str = ""
    instance: Any = None
    contributions: PluginContributions = field(default_factory=PluginContributions)
    lease: _Lease = field(default_factory=_Lease)
    registrations: list = field(default_factory=list)
    cleanup_ready: bool = False
    cleanup_started: bool = False
    cleanup_done: bool = False


class PluginManager:
    """由一个可信启动上下文持有；不使用跨实例全局缓存，不支持原地热替换。

    工厂返回提供 activate(context) 和可选 close() 的对象。activate 失败后仍尝试
    close，以清理部分创建的资源。异常只记录类型，避免将插件凭据写入诊断。
    """

    def __init__(self, context: PluginContext, definitions: Iterable[PluginDefinition] = ()):
        self.context = context
        self._records = [_Record(item) for item in definitions]
        self._by_id: dict[str, _Record] = {}
        self._lock = threading.RLock()
        self._started = self._closed = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def start(self):
        """仅校验实际接线所需信息；空 hosts 表示适用于当前宿主。"""
        with self._lock:
            if self._started or self._closed:
                return self
            self._started = True
        pending = []
        for record in self._records:
            item, manifest = record.definition, record.definition.manifest
            if not isinstance(manifest.id, str) or not manifest.id.strip():
                record.status, record.reason = "failed", "plugin_id_required"
            elif manifest.id in self._by_id:
                record.status, record.reason = "failed", "plugin_id_conflict"
            else:
                self._by_id[manifest.id] = record
                if not item.enabled:
                    record.status = "disabled"
                elif item.load_error:
                    record.status, record.reason = "failed", item.load_error
                elif manifest.hosts and "*" not in manifest.hosts and self.context.host_id not in manifest.hosts:
                    record.status, record.reason = "inactive", "different_host"
                elif manifest.api_version != 1:
                    record.status, record.reason = "failed", "plugin_api_incompatible"
                else:
                    pending.append(record)
        while pending:
            progressed = False
            for record in tuple(pending):
                dependencies = [self._by_id.get(name) for name in record.definition.manifest.depends_on]
                if any(dep is None or dep.status in {"failed", "disabled", "inactive"} for dep in dependencies):
                    record.status, record.reason = "failed", "plugin_dependency_unavailable"
                elif all(dep.status == "active" for dep in dependencies):
                    self._activate(record)
                else:
                    continue
                pending.remove(record)
                progressed = True
            if not progressed:
                for record in pending:
                    record.status, record.reason = "failed", "plugin_dependency_cycle"
                break
        return self

    @staticmethod
    def _factory(value):
        if callable(value):
            return value
        if value.startswith("entrypoint:"):
            name = value.partition(":")[2]
            entries = list(importlib.metadata.entry_points(group="shinku.plugins", name=name))
            if len(entries) != 1:
                raise ValueError("plugin_entrypoint_not_unique")
            return entries[0].load()
        module_name, separator, attribute = value.partition(":")
        factory = importlib.import_module(module_name)
        for part in (attribute if separator else "create").split("."):
            factory = getattr(factory, part)
        return factory

    def _activate(self, record):
        try:
            record.instance = self._factory(record.definition.factory)()
            context = replace(self.context, plugin_config=record.definition.config)
            value = record.instance.activate(context)
            if value is None:
                value = PluginContributions()
            if not isinstance(value, PluginContributions) or not isinstance(value.prompt_notes, str):
                raise TypeError("plugin_contributions_required")
            # 使用原注册边界检验 handler，避免另建更严格的参数或请求规则。
            ToolRegistry(value.tools)
            if value.chat_read_capabilities:
                from shinku.qq.interaction_tools import ChatReadCapability
                if any(not isinstance(cap, ChatReadCapability) or name != cap.name
                       for name, cap in value.chat_read_capabilities.items()):
                    raise TypeError("plugin_read_capability_required")
            record.contributions, record.status = value, "active"
        except Exception as exc:
            record.status, record.reason = "failed", f"plugin_activate_failed:{type(exc).__name__}"
            self._cleanup(record)

    def _active(self):
        return [record for record in self._records if record.status == "active"]

    def register_tools(self, registry: ToolRegistry):
        """在启动装配时发布；冲突不覆盖原工具，部分注册失败只撤回本插件的项。"""
        self.start()
        for record in self._active():
            if record.status != "active":
                continue
            if any(bound is registry for bound, _, _ in record.registrations):
                continue
            try:
                for name, handler in record.contributions.tools.items():
                    wrapped = _PluginTool(handler, record.lease)
                    registered = registry.register(wrapped, name=name)
                    record.registrations.append((registry, registered, wrapped))
            except Exception as exc:
                self._disable_record(record, reason=f"plugin_register_failed:{type(exc).__name__}", failed=True)
                self._disable_dependents(record.definition.manifest.id)

    def read_capabilities(self, base=None):
        """保留宿主/旧扩展已声明的只读项；重名的可选插件单独停用。"""
        self.start()
        result = dict(base or {})
        for record in self._active():
            if record.status != "active":
                continue
            reads = record.contributions.chat_read_capabilities
            if set(reads) & set(result):
                self._disable_record(record, reason="plugin_read_name_conflict", failed=True)
                self._disable_dependents(record.definition.manifest.id)
                continue
            result.update(reads)
        # 后出现的冲突可能同时停用前面已收集的依赖者；返回时按最终活跃集合重建。
        return {**dict(base or {}), **{
            name: capability for record in self._active()
            for name, capability in record.contributions.chat_read_capabilities.items()
        }}

    def prompt_notes(self):
        return tuple(record.contributions.prompt_notes for record in self._active()
                     if record.contributions.prompt_notes)

    def _cleanup(self, record):
        with self._lock:
            if record.cleanup_started:
                return
            record.cleanup_started = True
        try:
            close = getattr(record.instance, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            record.cleanup_error = f"plugin_close_failed:{type(exc).__name__}"
        if record.status == "draining":
            record.status = "disabled"
        record.cleanup_done = True

    def _ready_to_close(self, record):
        with self._lock:
            record.cleanup_ready = True
        self._drain_cleanups()

    def _drain_cleanups(self):
        # 依赖者还有在途调用时，被依赖插件也必须保留资源；关闭回调全在锁外。
        for record in reversed(self._records):
            with self._lock:
                waiting = any(
                    record.definition.manifest.id in other.definition.manifest.depends_on
                    and other.instance is not None and not other.cleanup_done
                    for other in self._records
                )
                ready = record.cleanup_ready and not record.cleanup_started and not waiting
            if ready:
                self._cleanup(record)
                self._drain_cleanups()

    def _disable_record(self, record, *, reason="", failed=False):
        with self._lock:
            if record.status != "active":
                return
            record.status, record.reason = ("failed" if failed else "draining"), reason
            registrations, record.registrations = record.registrations, []
        # 先封住旧 handler，移除只属于自己的注册项；不会删除并行替换的新 handler。
        record.lease.stop(lambda: self._ready_to_close(record))
        for registry, name, handler in reversed(registrations):
            registry.unregister(name, expected=handler)

    def _disable_dependents(self, plugin_id):
        for record in reversed(self._active()):
            if plugin_id in record.definition.manifest.depends_on:
                self._disable_dependents(record.definition.manifest.id)
                self._disable_record(record, reason="plugin_dependency_unavailable")

    def disable(self, plugin_id):
        """先停依赖者；允许在途调用结束，返回诊断中的 draining 状态。"""
        self._disable_dependents(plugin_id)
        record = self._by_id.get(plugin_id)
        if record is not None:
            self._disable_record(record)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # 反向拓扑关系由 disable 处理，声明顺序无需等于依赖顺序。
        for record in reversed(self._active()):
            self.disable(record.definition.manifest.id)

    def diagnostics(self):
        with self._lock:
            return {"closed": self._closed, "plugins": [
                {"id": record.definition.manifest.id, "version": record.definition.manifest.version,
                 "status": record.status, "reason": record.reason, "cleanup_error": record.cleanup_error,
                 "cleanup_pending": record.instance is not None and not record.cleanup_done and record.status != "active",
                 "tools": [name for _, name, _ in record.registrations]}
                for record in self._records]}

    def runtime_extension(self, *, legacy=None):
        """返回原 RuntimeExtension 接口；由持有本管理器的上下文负责 close。"""
        from .runtime import build_runtime_extension
        return build_runtime_extension(self, legacy=legacy)
