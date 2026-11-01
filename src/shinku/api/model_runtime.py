"""把模型服务注册表接到独立 QQ Agent 的运行时。

模型服务目录本身只负责校验、保存和探测；本模块负责把控制中心选中的设置
同步到 Shinku 的 ``SHINKU_*`` 运行环境，并请求现有 LLMRuntime 热重载。
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Mapping

from ..providers.catalog import (
    ModelServiceConfigStore,
    apply_model_service_settings,
    load_and_apply_saved_model_service,
)

__all__ = [
    "RuntimeModelServiceConfig",
    "RuntimeModelServiceController",
]


class RuntimeModelServiceConfig(SimpleNamespace):
    """给 ``providers.catalog`` 使用的可变配置视图。

    旧的供应商模块按属性名读配置，而独立 Agent 的运行时按环境变量读配置。
    这里保留两者之间的一层窄适配，不把供应商逻辑散落进 API 路由。
    """

    @classmethod
    def from_agent_config(cls, config: Any, environ: Mapping[str, str] | None = None) -> "RuntimeModelServiceConfig":
        env = os.environ if environ is None else environ
        chat_key = str(getattr(config, "chat_api_key", "") or "")
        chat_base = str(getattr(config, "chat_base_url", "") or "")
        chat_model = str(getattr(config, "chat_model_name", "") or "")
        chat_protocol = str(getattr(config, "chat_api_protocol", "auto") or "auto")
        values: dict[str, Any] = {
            "settings": SimpleNamespace(
                **{
                    key: str(env.get(f"SHINKU_{key}", "") or "")
                    for key in (
                        "GLM_API_KEY", "GLM_BASE_URL", "GLM_MODEL_NAME",
                        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL_NAME",
                        "PINAIC_API_KEY", "PINAIC_BASE_URL", "PINAIC_MODEL_NAME",
                    )
                },
                CHAT_API_KEY=chat_key,
                CHAT_BASE_URL=chat_base,
                CHAT_MODEL_NAME=chat_model,
                CHAT_API_PROTOCOL=chat_protocol,
            ),
            "ENABLE_NATIVE_TOOL_DECISION": str(env.get("SHINKU_ENABLE_NATIVE_TOOL_DECISION", "") or "").lower()
            in {"1", "true", "yes", "on"},
            "NATIVE_TOOL_PROVIDER_ALLOWLIST": str(env.get("SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST", "") or ""),
        }
        for prefix in ("TEXT", "AUX", "CHAT", "VISION"):
            values[f"{prefix}_API_KEY"] = chat_key if prefix != "VISION" else str(env.get("SHINKU_VISION_API_KEY", chat_key) or "")
            values[f"{prefix}_BASE_URL"] = chat_base if prefix != "VISION" else str(env.get("SHINKU_VISION_BASE_URL", chat_base) or "")
            values[f"{prefix}_MODEL_NAME"] = chat_model if prefix != "VISION" else str(env.get("SHINKU_VISION_MODEL_NAME", chat_model) or "")
            values[f"{prefix}_API_PROTOCOL"] = chat_protocol if prefix != "VISION" else str(env.get("SHINKU_VISION_API_PROTOCOL", chat_protocol) or "")
        return cls(**values)

    def sync_to_environment(self) -> None:
        """让热重载看到控制中心刚刚应用的设置。"""

        for prefix in ("TEXT", "AUX", "CHAT", "VISION"):
            for field in ("API_KEY", "BASE_URL", "MODEL_NAME", "API_PROTOCOL"):
                value = str(getattr(self, f"{prefix}_{field}", "") or "")
                os.environ[f"SHINKU_{prefix}_{field}"] = value
        os.environ["SHINKU_ENABLE_NATIVE_TOOL_DECISION"] = (
            "true" if bool(getattr(self, "ENABLE_NATIVE_TOOL_DECISION", False)) else "false"
        )
        os.environ["SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST"] = str(
            getattr(self, "NATIVE_TOOL_PROVIDER_ALLOWLIST", "") or ""
        )


class RuntimeModelServiceController:
    """模型服务注册表与当前 LLMRuntime 的热重载桥。"""

    def __init__(
        self,
        *,
        store: ModelServiceConfigStore,
        config_module: RuntimeModelServiceConfig,
        runtime: Any | None = None,
    ) -> None:
        self.store = store
        self.config_module = config_module
        self.runtime = runtime

    def load_saved(self) -> Any | None:
        """启动装配前加载已激活供应商；没有注册表时保持环境配置。"""

        settings = load_and_apply_saved_model_service(
            store=self.store,
            config_module=self.config_module,
            on_error=lambda exc: None,
        )
        if settings is not None:
            apply_model_service_settings(self.config_module, settings)
            self.config_module.sync_to_environment()
        return settings

    def reload_model_services(self) -> dict[str, Any]:
        self.config_module.sync_to_environment()
        runtime = self.runtime
        if runtime is None or not callable(getattr(runtime, "reload_from_config", None)):
            return {"status": "saved_but_runtime_not_attached"}
        result = runtime.reload_from_config()
        return result if isinstance(result, dict) else {"status": "reloaded"}

    def attach_runtime(self, runtime: Any) -> None:
        self.runtime = runtime
