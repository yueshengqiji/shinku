"""五个宿主进程的落点。

B1 阶段这里是空的——只有一个包标记。

为什么不先搬旧实现
------------------
严格分类下，旧项目 183 个运行时文件里只有 2 个具备 ``INDEPENDENT_KEEP`` 依据
（且依据是使用者的人工声明），其余是 106 个 ``REWRITE_REQUIRED`` + 36 个 ``UNCONFIRMED``。
按计划 B 的验收口径（"不存在 UNCONFIRMED 或 REWRITE_REQUIRED 文件进入独立发布包"），
**这四个宿主在 B1 一个文件都不能搬**。它们按 C 阶段的行为契约重写完再进来。

顺序（C 阶段）：response-host → time-manager → browser-host → backend 内的 Agent 编排接入。
"""

from __future__ import annotations

__all__: list[str] = []
