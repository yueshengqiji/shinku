"""供应商域的公开面：预置目录与端点/协议归一化。

本包收「模型供应商」这一域的词汇：有哪些预置、每个预置用什么协议、base_url 该怎么归一、
一段 base_url 该判给哪个供应商。LLM client 与模型服务配置都从这里取词，避免各写一套字符串比较。

C2-1 又把同域的两个模块放进本包：``embedding``（哈希词袋与 LRU 缓存）
与 ``huggingface``（可选的本地句向量后端）。两者**不进本包的 ``__all__``**——
``__all__`` 是从 C1-4 的扁平模块 ``provider_config`` 迁进来时留下的兼容面，
再往里塞东西会把这个聚合面变成"什么都有"的口袋；
调用方按全路径导入即可，做法同 :mod:`shinku.tools` 与 :mod:`shinku.guards`。
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
