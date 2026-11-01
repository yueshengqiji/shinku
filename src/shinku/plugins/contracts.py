"""插件描述、实例输入和能力贡献；不读取环境、不加载模型、不创建状态目录。

描述中的权限等扩展元数据只供宿主展示，不自动产生执行授权。进程内插件属于
可信代码；services 是启动器显式注入的依赖，并不构成 Python/OS 安全沙箱。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class PluginManifest:
    """api_version 是 SDK 主版本；version 是展示标签，不强制语义版本格式。"""

    id: str
    version: str = ""
    api_version: int = 1
    hosts: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginDefinition:
    """宿主选择；factory 可以是无参工厂、module:callable 或 entrypoint:名称。

    未启用的定义不导入实现，未知的可选元数据保留在 manifest.metadata 中。
    单个定义不兼容只影响该插件及真正依赖它的插件。
    """

    manifest: PluginManifest
    factory: Callable[[], Any] | str
    enabled: bool = True
    config: Mapping[str, Any] = field(default_factory=dict)
    # 读取清单时形成的固定原因码；可选配置损坏也只形成一个故障项，不拖垮宿主。
    load_error: str = ""


@dataclass(frozen=True)
class PluginContext:
    """一个实例中的一个宿主拥有上下文和管理器，插件配置单独传入副本。

    state_root 只提供路径，不自动创建目录或迁移数据。config 保留宿主原配置；
    插件自身设置位于 plugin_config，工具权限仍由原执行宿主逐调用裁决。
    """

    host_id: str
    instance_id: str = ""
    config: Any = None
    state_root: Path | None = None
    services: Mapping[str, Any] = field(default_factory=dict)
    plugin_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))
        object.__setattr__(self, "plugin_config", MappingProxyType(dict(self.plugin_config)))


@dataclass(frozen=True)
class PluginContributions:
    """沿用已有 handler 和 ChatReadCapability，不另外定义调用/结果协议。

    prompt_notes 由显式选择的可信插件贡献，保持原文，不替换既有人设。资料来源
    和后台任务的专用接口后续按真实迁移需求添加，当前不开放任意回合拦截钩子。
    """

    tools: Mapping[str, Any] = field(default_factory=dict)
    chat_read_capabilities: Mapping[str, Any] = field(default_factory=dict)
    prompt_notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))
        object.__setattr__(self, "chat_read_capabilities", MappingProxyType(dict(self.chat_read_capabilities)))
