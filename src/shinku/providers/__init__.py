"""供应商域的公开面：预置目录与端点/协议归一化。

本包收「模型供应商」这一域的词汇：有哪些预置、每个预置用什么协议、base_url 该怎么归一、
一段 base_url 该判给哪个供应商。LLM client 与模型服务配置都从这里取词，避免各写一套字符串比较。

后续批次（C2）会把同域的 ``huggingface_provider`` 一并放进来。
"""

from __future__ import annotations

from .config import (
    COMMON_PROVIDER_PRESETS,
    DEFAULT_PROVIDER_ID,
    SUPPORTED_PROTOCOLS,
    ProviderPreset,
    base_url_matches_provider,
    canonical_provider_id,
    infer_provider_id,
    normalize_api_protocol,
    normalize_base_url,
    normalize_configured_base_url,
    preset_index,
    provider_presets_payload,
)

__all__ = [
    "COMMON_PROVIDER_PRESETS",
    "DEFAULT_PROVIDER_ID",
    "SUPPORTED_PROTOCOLS",
    "ProviderPreset",
    "base_url_matches_provider",
    "canonical_provider_id",
    "infer_provider_id",
    "normalize_api_protocol",
    "normalize_base_url",
    "normalize_configured_base_url",
    "preset_index",
    "provider_presets_payload",
]
