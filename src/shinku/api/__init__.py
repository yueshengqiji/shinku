"""HTTP 层。C 阶段会把会话、上下文、能力、工具结果、任务状态和供应商协议的
路由逐个加进这里；B1 只有健康探针。"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
