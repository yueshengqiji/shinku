"""LLM 调用链上的基础件：上游静默开关与（后续批次）运行时。

本包只收「发起一次上游请求」这条链路上的进程级状态与传输件。
供应商词汇表在 ``shinku.providers``，两者不互相包含。

C2-1 只放熔断；``runtime``（LLM 运行时）与传输客户端由 C2 后续批次落地。
"""

from __future__ import annotations

from .circuit_breaker import LlmCircuitBreaker, get_llm_circuit_breaker

__all__ = [
    "LlmCircuitBreaker",
    "get_llm_circuit_breaker",
]
