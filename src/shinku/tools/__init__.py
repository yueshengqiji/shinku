"""工具域。

C1-2 先把包建起来，放进第一个模块——provider 无关的工具调用协议
（:mod:`shinku.tools.invocation`）。工具编排、结果回传与失败恢复按 C3 的行为契约
重写后落在这里。

本包不在 ``__init__`` 里聚合子模块，理由同 :mod:`shinku.tasks`。
"""

from __future__ import annotations

__all__: list[str] = []
