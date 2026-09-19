"""LLM 调用引擎组合根（C2-5a）。

把三片合成公开的 ``LLMRuntime``：

- ``RuntimeCore``（``runtime_core.py``）—— 传输骨架与底层通道；
- ``RuntimeCapabilitiesMixin``（``runtime_capabilities.py``）—— 能力协商；
- ``RuntimeAuditMixin``（``runtime_audit.py``）—— 审计与用量度量。

调用方（后续批次的 engine / 记忆整理服务）只需要 ``from shinku.llm.runtime
import LLMRuntime``；模块级再导出结果类型与工具函数，旧测试面引用的
``StreamTap`` 也从这里进。
"""

from __future__ import annotations

from shinku.llm.runtime_audit import RuntimeAuditMixin
from shinku.llm.runtime_capabilities import RuntimeCapabilitiesMixin
from shinku.llm.runtime_core import (
    ChatJSONStreamResult,
    ModelBundle,
    NDJSONCallResult,
    ProviderToolProfile,
    RuntimeCore,
    StreamTap,
    normalize_reply_medium,
)

__all__ = [
    "LLMRuntime",
    "ChatJSONStreamResult",
    "ModelBundle",
    "NDJSONCallResult",
    "ProviderToolProfile",
    "StreamTap",
    "normalize_reply_medium",
]


class LLMRuntime(RuntimeCapabilitiesMixin, RuntimeAuditMixin, RuntimeCore):
    """完整调用引擎。

    三片混入的顺序即方法解析顺序：能力协商与审计优先于传输骨架。
    构造参数与公开面见 ``RuntimeCore``（``metrics_path`` 可选）。
    """
