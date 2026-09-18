"""公开发言的控制面：进门前的准入限制，出门前的内容兜底。

一次公开发言在两处被确定性规则拦一下——**进**的时候看并发与当日额度够不够（
:mod:`shinku.guards.public_think`），**出**的时候把没有证据支撑的句子和拿供应方内部
当解释的句子削掉（:mod:`shinku.guards.evidence`）。两者都不改写内容，只做准入与减法。

本包不在 ``__init__`` 里聚合子模块，理由同 :mod:`shinku.tasks`。
"""

from __future__ import annotations

__all__: list[str] = []
