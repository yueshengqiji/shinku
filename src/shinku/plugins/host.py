"""为已有可信 QQ 宿主工厂增加插件；沿用原模型、委托、权限与生命周期。

这里只做启动接线，不读生产配置、不选择模型。原工厂先创建依赖，插件随后创建；
退出时先关闭插件，再交回原工厂关闭其依赖，避免插件持有已关闭的模型/连接。
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from shinku.extensions import load_runtime_extension
from .config import load_plugin_definitions
from .contracts import PluginContext
from .manager import PluginManager


def with_plugins(bindings_factory, definitions, *, host_id="shinku", instance_id="", state_root=None, services=None):
    """返回可传给 CLI agent_bindings 的工厂；definitions 可为清单路径或定义序列。

    读取路径和启用列表由可信启动器选择，用户回合不能更换。配置文件整体不可读时
    产生明确的配置故障项；单插件启动失败由管理器隔离，既有宿主仍能工作。
    """
    # 配置序列只保存描述，避免生成器在第一个实例消费后让后续实例悄悄无插件。
    selected = definitions if isinstance(definitions, (str, Path)) else tuple(definitions)

    @contextmanager
    def factory(config):
        with bindings_factory(config) as bindings:
            entries = load_plugin_definitions(selected) if isinstance(selected, (str, Path)) else selected
            context = PluginContext(host_id, instance_id, config=config, state_root=state_root, services=services or {})
            with PluginManager(context, entries) as manager:
                legacy = bindings.runtime_extension or load_runtime_extension(config.runtime_extension_module)
                extension = manager.runtime_extension(legacy=legacy)
                yield replace(bindings, runtime_extension=extension)
    # CLI/启动器可能借此标记判断原宿主是否明确使用远端委托，不能包装后丢失。
    if hasattr(bindings_factory, "delegation_enabled"):
        factory.delegation_enabled = bindings_factory.delegation_enabled
    return factory
