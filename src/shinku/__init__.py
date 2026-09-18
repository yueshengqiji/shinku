"""Shinku 独立实现。

本仓库是洁净室重建产物：它**不包含** AkaneCompanionLab 的任何源码、文档或素材。
业务模块在 C 阶段逐个重写后加入；加入时更新 ``NOTICE`` 与 ``docs/SOURCE_RECORD.md``。

当前（B1）仓库里只有骨架：命名契约、配置加载、后端应用工厂和命令行入口。
"""

from __future__ import annotations

__all__ = ["__version__", "PROJECT_NAME"]

PROJECT_NAME = "shinku"
__version__ = "0.1.0"
