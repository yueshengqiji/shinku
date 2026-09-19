"""真红侧的供应商目录，以及「把设置写回项目配置」这一件事的做法。

:mod:`shinku.providers.service` 是**与项目无关**的原语：它不知道本项目目录里有几家供应商。
本模块补上项目侧那半边：

- **目录**：在通用预置前面加上智谱 GLM，并给出别名与默认模型；
- **视图**：把注册表折成给控制中心看的一页（不含密钥）；
- **应用**：一套设置落到项目配置模块上时，改哪些通道、视觉通道怎么处理、
  原生工具开关怎么跟着走。

「应用」这一步的对象是**注入进来的配置模块**（鸭子类型，只认若干属性名），
所以本模块不 import 任何配置类型，也不参与配置模块的构造。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .config import (
    COMMON_PROVIDER_PRESETS,
    ProviderPreset,
    base_url_matches_provider,
    canonical_provider_id,
    infer_provider_id as shared_infer_provider_id,
    preset_index,
    provider_presets_payload as shared_presets_payload,
)
from .service import (
    DEFAULT_TIMEOUT_SECONDS,
    ModelServiceConfigStore as _SettingsRegistry,
    ModelServiceSettings,
    build_model_service_settings,
    build_public_model_service_snapshot,
    effective_model_service_settings,
    load_and_apply_saved_model_service as _apply_saved,
    probe_metadata,
    probe_model_ids as _probe_shared,
    public_provider_entry,
    test_model_service as _test_shared,
)

__all__ = [
    "MODEL_SERVICE_SCHEMA_VERSION",
    "PROVIDER_DEFAULT_MODELS",
    "PROVIDER_ID_ALIASES",
    "PROVIDER_PRESETS",
    "PRESET_BY_ID",
    "ModelProviderPreset",
    "ModelServiceConfigStore",
    "apply_model_service_settings",
    "effective_settings_from_config",
    "environment_settings_for_provider",
    "infer_provider_id",
    "load_and_apply_saved_model_service",
    "probe_model_ids",
    "provider_presets_payload",
    "public_model_service_snapshot",
    "public_model_services_snapshot",
    "settings_from_mapping",
    "test_model_service",
]

MODEL_SERVICE_SCHEMA_VERSION = 2

ModelProviderPreset = ProviderPreset

#: 目录顺序 = 控制中心里的展示顺序；GLM 排在最前是因为它是本项目的默认供应商。
PROVIDER_PRESETS: tuple[ModelProviderPreset, ...] = (
    ModelProviderPreset(
        id="glm",
        label="智谱 GLM",
        protocol="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_required=True,
        description="智谱 AI 的 OpenAI 兼容接口。",
        capabilities=("stream", "vision", "native_tools"),
    ),
    *COMMON_PROVIDER_PRESETS,
)
PRESET_BY_ID = preset_index(PROVIDER_PRESETS)

#: 选了某家但没填模型时的兜底模型。
PROVIDER_DEFAULT_MODELS = {
    "glm": "glm-4.5-air",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-3-5-sonnet-latest",
}

#: 用户/前端可能写出的别名（含错别字与中文）。
PROVIDER_ID_ALIASES = {
    "gml": "glm",
    "智谱": "glm",
    "智谱ai": "glm",
    "智谱glm": "glm",
    "zhipu": "glm",
    "zhipuai": "glm",
}

#: 有专属环境变量的供应商：``(密钥, 端点, 模型)`` 三个属性名。
_SPECIFIC_ENV_FIELDS = {
    "glm": ("GLM_API_KEY", "GLM_BASE_URL", "GLM_MODEL_NAME"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL_NAME"),
    "pinaic": ("PINAIC_API_KEY", "PINAIC_BASE_URL", "PINAIC_MODEL_NAME"),
}

#: 老部署把每家供应商塞在通道里而不是专属字段上，逐个通道认一遍。
_CHANNEL_PREFIXES = ("CHAT", "TEXT", "AUX")

#: 已知供应商的视觉模型前缀。换供应商后残留的旧视觉模型名靠它识别出来。
_VISION_MODEL_PREFIXES = {
    "glm": ("glm-",),
    "deepseek": ("deepseek-",),
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "gemini": ("gemini-",),
    "anthropic": ("claude-",),
}


class ModelServiceConfigStore(_SettingsRegistry):
    """本项目的注册表入口：目录、别名、默认模型都在这里定死。"""

    def __init__(self, path: Path) -> None:
        super().__init__(
            path,
            presets=PROVIDER_PRESETS,
            aliases=PROVIDER_ID_ALIASES,
            default_models=PROVIDER_DEFAULT_MODELS,
        )


def provider_presets_payload() -> list[dict[str, Any]]:
    """目录的载荷形态：通用预置那几项 + 一行「支持模型探测」标记。"""

    payload = shared_presets_payload(PROVIDER_PRESETS, include_capabilities=True)
    for item in payload:
        item["supportsModelDiscovery"] = True
    return payload


def settings_from_mapping(
    raw: dict[str, Any],
    *,
    existing_api_key: str = "",
    require_model: bool = True,
) -> ModelServiceSettings:
    """从一份载荷折设置。

    这里**不传默认模型**：本函数用于「校验用户提交的表单」，用户没填模型就该空着，
    由调用方决定要不要补；注册表的读写走 :class:`ModelServiceConfigStore`，那边才带默认模型。
    """

    return build_model_service_settings(
        raw,
        presets=PROVIDER_PRESETS,
        aliases=PROVIDER_ID_ALIASES,
        default_models={},
        existing_api_key=existing_api_key,
        default_timeout=DEFAULT_TIMEOUT_SECONDS,
        require_model=require_model,
    )


def effective_settings_from_config(config_module: Any) -> ModelServiceSettings:
    return effective_model_service_settings(
        config_module,
        infer_provider_id=infer_provider_id,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )


def infer_provider_id(*, protocol: str, base_url: str) -> str:
    """从协议与端点反推供应商；只在**本项目目录里有的 id** 之间挑。"""

    return shared_infer_provider_id(
        protocol=protocol,
        base_url=base_url,
        provider_ids=PRESET_BY_ID,
    )


def public_model_service_snapshot(
    settings: ModelServiceSettings,
    *,
    source: str,
    load_status: str = "ok",
) -> dict[str, Any]:
    return build_public_model_service_snapshot(
        settings,
        source=source,
        load_status=load_status,
        providers=provider_presets_payload(),
    )


def public_model_services_snapshot(
    store: ModelServiceConfigStore,
    config_module: Any,
    *,
    source: str = "local_file",
) -> dict[str, Any]:
    """控制中心那一页：目录里每一家的状态 + 当前激活的那一家，**不含密钥**。

    目录之外的条目（自定义兼容服务）会追加在后面，所以用户自己加过的服务不会丢。
    """

    registry = store.load_registry()
    records = registry.get("providers") if isinstance(registry.get("providers"), dict) else {}
    effective = effective_settings_from_config(config_module)
    active_id = _as_provider_id(registry.get("activeProviderId")) or effective.provider_id
    active_settings = store.load() or effective
    entries: list[dict[str, Any]] = []

    for preset in PROVIDER_PRESETS:
        raw = records.get(preset.id)
        settings = settings_from_mapping(raw, require_model=False) if isinstance(raw, dict) else None
        metadata = probe_metadata(raw if isinstance(raw, dict) else {})
        if settings is None:
            settings = environment_settings_for_provider(preset.id, config_module)
        entries.append(
            public_provider_entry(
                preset=preset,
                settings=settings,
                metadata=metadata,
                active=(preset.id == active_id),
            )
        )

    known_ids = {preset.id for preset in PROVIDER_PRESETS}
    for provider_id, raw in records.items():
        if provider_id in known_ids or not isinstance(raw, dict):
            continue
        entries.append(
            public_provider_entry(
                preset=None,
                settings=settings_from_mapping(raw, require_model=False),
                metadata=probe_metadata(raw),
                active=(provider_id == active_id),
            )
        )

    return {
        **public_model_service_snapshot(active_settings, source=source),
        "schemaVersion": MODEL_SERVICE_SCHEMA_VERSION,
        "activeProviderId": active_id,
        "activeModel": active_settings.chat_model,
        "providers": entries,
    }


def environment_settings_for_provider(
    provider_id: str,
    config_module: Any,
) -> ModelServiceSettings | None:
    """目录里「没在注册表里配过」的供应商，改从环境变量里认一遍。

    热切换会**改写**配置模块的 ``CHAT_*`` 全局量，只看 ``CHAT_*`` 会把「配了但当前没激活」
    的供应商显示成没配。所以这里优先读最初从 .env 载入的那个 ``settings`` 对象——
    它没有被热切换动过。
    """

    provider_id = _as_provider_id(provider_id)
    effective = effective_settings_from_config(config_module)
    if effective.provider_id == provider_id and effective.configured:
        return effective

    raw_settings = getattr(config_module, "settings", None)
    if raw_settings is None:
        return None

    fields = _SPECIFIC_ENV_FIELDS.get(provider_id)
    if fields:
        key = str(getattr(raw_settings, fields[0], "") or "").strip()
        base = str(getattr(raw_settings, fields[1], "") or "").strip()
        model = str(getattr(raw_settings, fields[2], "") or "").strip()
        if key or base or model:
            try:
                candidate = settings_from_mapping(
                    {
                        "providerId": provider_id,
                        "apiKey": key,
                        "baseUrl": base,
                        "chatModel": model or PROVIDER_DEFAULT_MODELS.get(provider_id, ""),
                    },
                    require_model=False,
                )
                if candidate.configured:
                    return candidate
            except Exception:
                pass

    for prefix in _CHANNEL_PREFIXES:
        base = str(getattr(raw_settings, f"{prefix}_BASE_URL", "") or "").strip()
        if not base_url_matches_provider(base, provider_id):
            continue
        try:
            candidate = settings_from_mapping(
                {
                    "providerId": provider_id,
                    "apiKey": str(getattr(raw_settings, f"{prefix}_API_KEY", "") or "").strip(),
                    "baseUrl": base,
                    "chatModel": str(
                        getattr(raw_settings, f"{prefix}_MODEL_NAME", "") or ""
                    ).strip()
                    or PROVIDER_DEFAULT_MODELS.get(provider_id, ""),
                    "protocol": str(getattr(raw_settings, f"{prefix}_API_PROTOCOL", "") or ""),
                },
                require_model=False,
            )
            if candidate.configured:
                return candidate
        except Exception:
            continue
    return None


def apply_model_service_settings(config_module: Any, settings: ModelServiceSettings) -> None:
    """把一套设置写回项目配置模块：三个文本通道 + 视觉通道 + 原生工具开关。"""

    for prefix in ("TEXT", "AUX", "CHAT"):
        setattr(config_module, f"{prefix}_API_KEY", settings.api_key)
        setattr(config_module, f"{prefix}_BASE_URL", settings.base_url)
        setattr(config_module, f"{prefix}_MODEL_NAME", settings.chat_model)
        setattr(config_module, f"{prefix}_API_PROTOCOL", settings.protocol)

    if settings.use_for_vision and _vision_model_fits_provider(settings):
        setattr(config_module, "VISION_API_KEY", settings.api_key)
        setattr(config_module, "VISION_BASE_URL", settings.base_url)
        setattr(config_module, "VISION_MODEL_NAME", settings.vision_model or settings.chat_model)
        setattr(config_module, "VISION_API_PROTOCOL", settings.protocol)
    elif settings.use_for_vision:
        # 换供应商时不能把别家的模型名直接丢给视觉端点：聊天切到 GLM 而 .env 里的
        # 视觉服务是 Gemini 兼容时最容易踩到。这里保留已载入的专用视觉配置，
        # 让图像链路继续用它，而不是报一个误导性的 model-not-found。
        import logging

        logging.getLogger("shinku.model_service").warning(
            "Ignoring incompatible vision model %s for provider %s; preserving existing vision service",
            settings.vision_model or settings.chat_model,
            settings.provider_id,
        )

    _sync_native_tool_policy(config_module, settings)


def _vision_model_fits_provider(settings: ModelServiceSettings) -> bool:
    """存着的视觉模型名是否还属于当前供应商。

    自定义与本地服务刻意接受任意模型 id（它们本来就可能代理多家），
    只有**已知供应商**才做前缀检查，让残留的旧 id 回落到专用视觉服务。
    """

    model = str(settings.vision_model or settings.chat_model or "").strip().casefold()
    provider = str(settings.provider_id or "").strip().casefold()
    if not model:
        return False
    prefixes = _VISION_MODEL_PREFIXES.get(provider)
    if prefixes is None:
        return True
    return model.startswith(prefixes)


def _sync_native_tool_policy(config_module: Any, settings: ModelServiceSettings) -> None:
    """原生工具的开关与白名单跟着当前供应商/模型走。

    配置模块没有对应属性就整段跳过——这两个开关是可选的，不能因为缺属性就写失败。
    """

    if not hasattr(config_module, "ENABLE_NATIVE_TOOL_DECISION"):
        return
    presets = getattr(config_module, "MODEL_PRESETS", {})
    preset = presets.get(settings.provider_id) if isinstance(presets, dict) else None
    native_tools = bool(isinstance(preset, dict) and preset.get("native_tools"))
    setattr(config_module, "ENABLE_NATIVE_TOOL_DECISION", native_tools)
    if not native_tools:
        if hasattr(config_module, "NATIVE_TOOL_PROVIDER_ALLOWLIST"):
            setattr(config_module, "NATIVE_TOOL_PROVIDER_ALLOWLIST", "")
        return

    template = str((preset or {}).get("native_tool_allowlist") or "").strip()
    host = urlsplit(settings.base_url).netloc or settings.base_url
    allowlist = template.format(model=settings.chat_model, host=host)
    if not allowlist:
        allowlist = f"{host}:{settings.chat_model}" if host and settings.chat_model else ""
    if hasattr(config_module, "NATIVE_TOOL_PROVIDER_ALLOWLIST"):
        setattr(config_module, "NATIVE_TOOL_PROVIDER_ALLOWLIST", allowlist)


def probe_model_ids(settings: ModelServiceSettings) -> list[str]:
    """本项目走 Ollama 原生清单端点（SDK 的 ``models.list()`` 拿不到本地标签名）。"""

    return _probe_shared(settings, use_ollama_tags=True)


def test_model_service(settings: ModelServiceSettings) -> str:
    return _test_shared(settings)


def load_and_apply_saved_model_service(
    *,
    store: ModelServiceConfigStore,
    config_module: Any,
    on_error: Callable[[Exception], None] | None = None,
) -> ModelServiceSettings | None:
    return _apply_saved(
        store=store,
        config_module=config_module,
        apply=apply_model_service_settings,
        on_error=on_error,
    )


def _as_provider_id(value: Any) -> str:
    return canonical_provider_id(value, aliases=PROVIDER_ID_ALIASES)
