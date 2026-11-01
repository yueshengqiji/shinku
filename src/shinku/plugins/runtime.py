"""将管理器适配为原扩展接口；旧人设/命令/运行时绑定保持原执行语义。

旧扩展由可信启动器显式传入，不隐式导入生产配置。管理器的所有权留在启动器
ExitStack 中；QQ 装配只借用同一个扩展对象，不重复创建插件或资源。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

from shinku.extensions import RuntimeExtension


@dataclass(frozen=True)
class ManagedRuntimeExtension(RuntimeExtension):
    manager: Any
    legacy: RuntimeExtension | None = None

    @property
    def owner_gate_required(self):
        # 添加公共插件本身不应使既有公共宿主新增主人限制；旧私用扩展仍保持原门禁。
        return self.legacy is not None and self.legacy.owner_gate_required

    def diagnostics(self):
        return {**super().diagnostics(), **self.manager.diagnostics()}


def build_runtime_extension(manager, *, legacy=None):
    manager.start()
    module = ModuleType("shinku.plugins.managed")

    def register_tools(*, registry, config):
        if legacy is not None:
            legacy.register_tools(registry=registry, config=config)
        manager.register_tools(registry)

    def reads(*, config):
        base = legacy.build_chat_read_capabilities(config=config) if legacy is not None else {}
        return manager.read_capabilities(base)

    def prompt(*, prompt, config):
        original = legacy.augment_system_prompt(prompt=prompt, config=config) if legacy is not None else prompt
        notes = manager.prompt_notes()
        return original + ("\n\n" + "\n\n".join(notes) if notes else "")

    module.register_tools = register_tools
    module.build_chat_read_capabilities = reads
    module.augment_system_prompt = prompt
    if legacy is not None:
        module.bind_runtime = legacy.bind_runtime
        module.build_command_interpreter = legacy.build_command_interpreter
    return ManagedRuntimeExtension("shinku.plugins.managed", module, manager, legacy)
