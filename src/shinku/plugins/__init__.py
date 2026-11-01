"""可信宿主显式装配的插件接口。

阅读顺序为 contracts → manager → runtime。管理器只管理插件接线和资源，
工具调用仍进入原有宿主的权限、执行、核验与送达链，不审查用户措辞。
"""

from .contracts import PluginContext, PluginContributions, PluginDefinition, PluginManifest
from .manager import PluginManager, PluginUnavailable

__all__ = [
    "PluginContext", "PluginContributions", "PluginDefinition", "PluginManifest",
    "PluginManager", "PluginUnavailable",
]
