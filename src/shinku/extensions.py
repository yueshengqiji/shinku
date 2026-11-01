"""可选运行时扩展边界。

公共发行版不内置任何私用能力。只有调用方显式设置
``SHINKU_RUNTIME_EXTENSION_MODULE`` 时，启动器才会导入一个外部模块，
并让它向工具注册表增加本机能力。扩展模块不属于公共包，也不会被默认发现。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any

__all__ = ["RuntimeExtension", "RuntimeExtensionError", "load_runtime_extension"]


class RuntimeExtensionError(ValueError):
    """运行时扩展显式配置但无法安全加载时抛出。"""


@dataclass(frozen=True)
class RuntimeExtension:
    module_name: str
    module: ModuleType

    @property
    def owner_gate_required(self) -> bool:
        """原外部扩展沿用私用主人边界；公共插件适配器单独保留宿主原策略。"""
        return True

    def bind_runtime(self, **kwargs: Any) -> Any:
        """Optional late binding after live command state exists; no saved grants."""
        hook = getattr(self.module, "bind_runtime", None)
        if hook is None:
            return None
        if not callable(hook):
            raise RuntimeExtensionError("runtime_extension_bind_runtime_not_callable")
        try:
            return hook(**kwargs)
        except RuntimeExtensionError:
            raise
        except Exception as exc:
            raise RuntimeExtensionError(
                f"runtime_extension_bind_runtime_failed:{type(exc).__name__}"
            ) from exc

    def register_tools(self, *, registry: Any, config: Any) -> Any:
        hook = getattr(self.module, "register_tools", None)
        if hook is None:
            return None
        if not callable(hook):
            raise RuntimeExtensionError("runtime_extension_register_tools_not_callable")
        try:
            return hook(registry=registry, config=config)
        except RuntimeExtensionError:
            raise
        except Exception as exc:
            raise RuntimeExtensionError(
                f"runtime_extension_register_tools_failed:{type(exc).__name__}"
            ) from exc

    def build_chat_read_capabilities(self, *, config: Any) -> Any:
        """由显式扩展声明聊天只读目录，不从执行工具标记自动推导授权。"""
        hook = getattr(self.module, "build_chat_read_capabilities", None)
        if hook is None:
            return {}
        if not callable(hook):
            raise RuntimeExtensionError("runtime_extension_read_capabilities_not_callable")
        try:
            return hook(config=config)
        except Exception as exc:
            # 不把扩展异常中的路径、凭据等任意正文带进启动日志。
            raise RuntimeExtensionError(
                f"runtime_extension_read_capabilities_failed:{type(exc).__name__}"
            ) from exc

    def augment_system_prompt(self, *, prompt: str, config: Any) -> str:
        """Allow a private runtime to add its own role/context layer.

        The public build keeps this hook empty.  A configured extension may
        return a new prompt, but it must return text explicitly; silently
        accepting arbitrary objects would make a broken private extension
        indistinguishable from a valid persona.
        """

        hook = getattr(self.module, "augment_system_prompt", None)
        if hook is None:
            return str(prompt or "")
        if not callable(hook):
            raise RuntimeExtensionError("runtime_extension_augment_system_prompt_not_callable")
        try:
            value = hook(prompt=str(prompt or ""), config=config)
        except RuntimeExtensionError:
            raise
        except Exception as exc:
            raise RuntimeExtensionError(
                f"runtime_extension_augment_system_prompt_failed:{type(exc).__name__}"
            ) from exc
        if not isinstance(value, str):
            raise RuntimeExtensionError("runtime_extension_augment_system_prompt_must_return_text")
        return value

    def build_command_interpreter(self, **kwargs: Any) -> Any | None:
        """Optionally provide a private command interpreter.

        Slash-command behavior is not part of the public extension surface by
        default.  Only an explicitly configured module can opt into this hook.
        """

        hook = getattr(self.module, "build_command_interpreter", None)
        if hook is None:
            return None
        if not callable(hook):
            raise RuntimeExtensionError("runtime_extension_build_command_interpreter_not_callable")
        try:
            return hook(**kwargs)
        except RuntimeExtensionError:
            raise
        except Exception as exc:
            raise RuntimeExtensionError(
                f"runtime_extension_build_command_interpreter_failed:{type(exc).__name__}"
            ) from exc

    def diagnostics(self) -> dict[str, object]:
        return {
            "configured": True,
            "module": self.module_name,
            "register_tools": callable(getattr(self.module, "register_tools", None)),
        }


def load_runtime_extension(module_name: str = "") -> RuntimeExtension | None:
    name = str(module_name or "").strip()
    if not name:
        return None
    if any(character in name for character in "/\\:;\n\r"):
        raise RuntimeExtensionError("runtime_extension_module_invalid")
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeExtensionError(
            f"runtime_extension_import_failed:{type(exc).__name__}"
        ) from exc
    return RuntimeExtension(name, module)
