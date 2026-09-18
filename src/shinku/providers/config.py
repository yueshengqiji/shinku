"""供应商词汇与端点归一化。

同一个供应商在配置里有很多种写法：协议字段可能写 ``auto``，base_url 可能带或不带 ``/v1``，
也可能被拼成 ``.../v1/chat/completions`` 这种完整路径。这个模块把归一化规则与常见供应商的
预置收在一处，让上层只跟一套词打交道。

三组「信号词」是分开的，因为它们回答的不是同一个问题（且**必须**保持分开）：

- :data:`_PROTOCOL_SIGNALS`：从 base_url 判**协议**，宽松到只要主机里出现 ``anthropic`` 就算；
- :data:`_INFER_SIGNALS`：从 base_url 反推**供应商 id**，只认官方主机名（``anthropic.com``）；
- :data:`_MATCH_SIGNALS`：判「某 base_url 是否属于某供应商」，用域名根（``deepseek.com``）。

把三张表合成一张会改变行为，例如 ``https://deepseek.com/x`` 在推断里不构成 deepseek（要
``api.deepseek.com``），在匹配里却算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

#: 支持的 API 协议。属线路事实。
SUPPORTED_PROTOCOLS = ("openai", "anthropic", "ollama")

#: 推不出具体供应商时的兜底 id。属线路事实。
DEFAULT_PROVIDER_ID = "openai_compatible"

#: Ollama 的本地默认端点（base_url 为空时补这个）。
_OLLAMA_BASE = "http://127.0.0.1:11434"

#: 配置里可能写成完整路径的尾部片段，归一化时剥掉（依次各检查一次）。
_ENDPOINT_SUFFIXES = ("/chat/completions", "/models")
#: Anthropic 的 Messages 路径尾缀。
_ANTHROPIC_MESSAGES = "/v1/messages"

#: 从 base_url 判协议的信号（宽松）。
_PROTOCOL_SIGNALS = {
    "anthropic": ("anthropic", "/claude"),
    "ollama": ("11434", "ollama"),
}

#: 从 base_url 反推供应商 id 的信号（严格，只认官方主机名），键顺序即优先级。
_INFER_SIGNALS = {
    "ollama": ("11434", "ollama"),
    "anthropic": ("anthropic.com", "/claude"),
    "glm": ("open.bigmodel.cn", "bigmodel.cn"),
    "openai": ("api.openai.com",),
    "deepseek": ("api.deepseek.com",),
    "pinaic": ("api.pinaic.com",),
    "gemini": ("generativelanguage.googleapis.com",),
}

#: 除 base_url 信号外，可凭 protocol 直接认定的供应商。
_PROTOCOL_PROVIDERS = {"ollama": "ollama", "anthropic": "anthropic"}

#: 判「base_url 是否属于某供应商」的信号（用域名根）。
_MATCH_SIGNALS = {
    "glm": ("bigmodel.cn",),
    "deepseek": ("deepseek.com",),
    "gemini": ("generativelanguage.googleapis.com",),
    "anthropic": ("anthropic.com", "/claude"),
    "ollama": ("11434", "ollama"),
    "openai": ("api.openai.com",),
    "pinaic": ("api.pinaic.com",),
}


@dataclass(frozen=True)
class ProviderPreset:
    """一个供应商预置：展示用的与连接用的信息各就各位。"""

    id: str
    label: str
    protocol: str
    base_url: str
    api_key_required: bool
    description: str
    capabilities: tuple[str, ...] = ("stream",)


#: 常见供应商预置。``description`` 是前端展示文案，``id``/``label``/``base_url`` 是线路事实。
COMMON_PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset("openai", "OpenAI", "openai", "https://api.openai.com/v1", True, "OpenAI 官方接口。", ("stream", "vision", "native_tools")),
    ProviderPreset("deepseek", "DeepSeek", "openai", "https://api.deepseek.com/v1", True, "DeepSeek 官方 OpenAI 兼容接口。", ("stream", "native_tools")),
    ProviderPreset("pinaic", "Pinaic", "openai", "https://api.pinaic.com/v1", True, "Pinaic 的 OpenAI 兼容接口。", ("stream",)),
    ProviderPreset("gemini", "Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai", True, "Google AI Studio 的 OpenAI 兼容接口。", ("stream", "vision")),
    ProviderPreset("anthropic", "Anthropic Claude", "anthropic", "https://api.anthropic.com", True, "Anthropic 官方 Messages API。", ("stream", "vision", "native_tools")),
    ProviderPreset("ollama", "Ollama 本地模型", "ollama", "http://127.0.0.1:11434", False, "本机 Ollama 的 OpenAI 兼容接口。", ("stream", "vision")),
    ProviderPreset("openai_compatible", "其他 OpenAI 兼容服务", "openai", "", True, "中转站、自部署网关或其他兼容 /v1 的服务。", ("stream", "vision", "native_tools")),
)


def preset_index(presets: Iterable[ProviderPreset]) -> dict[str, ProviderPreset]:
    """按 id 建索引；同 id 后者覆盖前者。"""

    table: dict[str, ProviderPreset] = {}
    for preset in presets:
        table[preset.id] = preset
    return table


def _has_signal(lowered_url: str, signals: Mapping[str, tuple[str, ...]], key: str) -> bool:
    return any(word in lowered_url for word in signals.get(key, ()))


def normalize_api_protocol(protocol: str = "", base_url: str = "") -> str:
    """定出协议名：显式值优先，其次按 base_url 猜，最后兜底 openai。"""

    explicit = str(protocol or "").strip().lower()
    if explicit in SUPPORTED_PROTOCOLS:
        return explicit
    lowered = str(base_url or "").strip().lower()
    for candidate in ("anthropic", "ollama"):
        if _has_signal(lowered, _PROTOCOL_SIGNALS, candidate):
            return candidate
    return "openai"


def normalize_base_url(*, protocol: str, base_url: str) -> str:
    """归一 base_url：去尾斜杠；ollama 补默认端点与 ``/v1``。"""

    value = str(base_url or "").strip().rstrip("/")
    if protocol != "ollama":
        return value
    value = value or _OLLAMA_BASE
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def normalize_configured_base_url(base_url: str, *, protocol: str) -> str:
    """把「配置里写成完整路径」的 base_url 收回主机根。"""

    value = str(base_url or "").strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
    if protocol == "anthropic" and value.endswith(_ANTHROPIC_MESSAGES):
        value = value[: -len("/messages")]
    return value


def canonical_provider_id(value: object, *, aliases: Mapping[str, str] | None = None) -> str:
    """把别名（如 ``claude``）折成规范 id；不认识就返回自身的规范化形式。"""

    normalized = str(value or "").strip().lower()
    return dict(aliases or {}).get(normalized, normalized)


def infer_provider_id(
    *,
    protocol: str,
    base_url: str,
    provider_ids: Iterable[str] | None = None,
) -> str:
    """从协议与 base_url 反推供应商 id；给了候选集就只在候选里挑。"""

    lowered = str(base_url or "").strip().lower()
    allowed = {str(item or "").strip().lower() for item in provider_ids or ()}
    for provider_id in _INFER_SIGNALS:
        if allowed and provider_id not in allowed:
            continue
        required_protocol = _PROTOCOL_PROVIDERS.get(provider_id)
        if required_protocol is not None and required_protocol == protocol:
            return provider_id
        if _has_signal(lowered, _INFER_SIGNALS, provider_id):
            return provider_id
    return DEFAULT_PROVIDER_ID


def base_url_matches_provider(base_url: str, provider_id: str) -> bool:
    """base_url 是否落在该供应商的域名根里；未知 id 一律 False。"""

    lowered = str(base_url or "").strip().lower()
    return _has_signal(lowered, _MATCH_SIGNALS, str(provider_id or "").strip().lower())


def provider_presets_payload(
    presets: Iterable[ProviderPreset],
    *,
    include_capabilities: bool = False,
) -> list[dict[str, object]]:
    """把预置转成前端用的 camelCase 载荷；键名属线路事实。"""

    rows: list[dict[str, object]] = []
    for preset in presets:
        row: dict[str, object] = {
            "id": preset.id,
            "label": preset.label,
            "protocol": preset.protocol,
            "baseUrl": preset.base_url,
            "apiKeyRequired": preset.api_key_required,
            "description": preset.description,
        }
        if include_capabilities:
            row["capabilities"] = list(preset.capabilities)
        rows.append(row)
    return rows


__all__ = [
    "SUPPORTED_PROTOCOLS",
    "DEFAULT_PROVIDER_ID",
    "ProviderPreset",
    "COMMON_PROVIDER_PRESETS",
    "preset_index",
    "normalize_api_protocol",
    "normalize_base_url",
    "normalize_configured_base_url",
    "canonical_provider_id",
    "infer_provider_id",
    "base_url_matches_provider",
    "provider_presets_payload",
]
