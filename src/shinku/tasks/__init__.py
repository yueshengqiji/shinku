"""任务域。

C1-2 先把包建起来，放进第一个模块——产物视图（:mod:`shinku.tasks.artifacts`）。
任务状态、状态机与代理循环按 C3 的行为契约重写后陆续落在这里。

本包不在 ``__init__`` 里聚合子模块：任务域会持续长大，
把所有名字都挤进包级命名空间很快就会变成维护负担。导入请写明路径。
"""

from __future__ import annotations

__all__: list[str] = []
