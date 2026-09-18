"""LLM 调用链上的基础件：上游静默开关与传输客户端。

本包只收「发起一次上游请求」这条链路上的进程级状态与传输件。
供应商词汇表在 ``shinku.providers``，两者不互相包含。

C2-1 放熔断，C2-2 放传输客户端；``runtime``（LLM 运行时）由 C2 后续批次落地。
"""

from __future__ import annotations

from .circuit_breaker import LlmCircuitBreaker, get_llm_circuit_breaker
from .client import AnthropicCompatClient, build_llm_client

__all__ = [
    "AnthropicCompatClient",
    "LlmCircuitBreaker",
    "build_llm_client",
    "get_llm_circuit_breaker",
]
