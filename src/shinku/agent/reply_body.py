"""「最终发出去的是哪段正文」这一件事的唯一判定处。

一条回合里正文会在好几条路径上被拼出来 —— 直聊分支、自主群聊的沉默分支、
公开查询快路径、感知快轨 —— 而它们各自都带着同一对小规则的手抄本：

* 什么算「模型没给出可用答复」；
* 什么算那个私有的沉默控制值。

一条规则有两份抄本，就等于迟早会分叉（此前拒答文案就在两个文件里分叉过一次），
所以两条规则现在都只住在这里。**树里任何其他地方都不许再把沉默标记写成字面量。**

只判空与标记，不碰业务分支 —— 调用方各自决定兜底文案是什么。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AUTONOMOUS_SILENCE_MARKER",
    "finalize_reply",
    "is_empty_reply",
    "is_silence_marker",
]

#: 线路层控制值，含义是「这个自主回合不说话」。
#: 它从来不是给人看的正文：模型被要求只输出这个 token，而运行时在投递前把它筛掉。
AUTONOMOUS_SILENCE_MARKER = "<SILENCE>"


def is_silence_marker(text: Any) -> bool:
    """判断 ``text`` 是不是那个沉默控制 token。

    比较时忽略首尾空白与大小写 —— 模型给这个 token 补空格、改大小写的次数
    多到足以让精确匹配误判。
    """
    return str(text or "").strip().casefold() == AUTONOMOUS_SILENCE_MARKER.casefold()


def is_empty_reply(text: Any) -> bool:
    """判断有没有可发的正文：空文本，或者沉默 token。"""
    return not str(text or "").strip() or is_silence_marker(text)


def finalize_reply(text: Any, fallback: str) -> str:
    """把模型原始输出收敛成将要投递的正文。

    ``fallback`` 代表「没有可用答复」。沉默 token 在这里也筛一道，
    于是即便某条路径忘了在自己那一层检查，控制值也漏不到用户面前。
    """
    body = str(text or "").strip()
    return fallback if is_empty_reply(body) else body
