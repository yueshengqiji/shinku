"""Lexical image-topic hints for candidate ranking and attention sampling.

Neither a match nor a miss resolves a reference or closes a topic. The chat
decision selects or clarifies; the host verifies scope, version and read budget.
"""

from __future__ import annotations

import re


VISUAL_REACTION_WINDOW_SECONDS = 60.0
VISUAL_QUESTION_WINDOW_SECONDS = 30.0
_ADDRESS_RE = re.compile(r"^(?:二阶堂真红|nikaidoushinku|shinku|真红|老婆)[，,：:、]*", re.IGNORECASE)
_REACTION_RE = re.compile(
    r"(?:"
    r"(?:(?:这|那)(?:个)?(?:也)?)?"
    r"(?:真的?|也太|太|很|好|挺|蛮|非常|特别|超级|确实|有点|还)?"
    r"(?:好看|可爱|漂亮|不错|棒|赞|好笑|搞笑|有趣|丑|奇怪|吓人|帅)"
    r"|(?:我)?(?:真的?|也太|太|很|好|挺|非常|特别|超级)?(?:喜欢|爱了)(?:这个|那个)?"
    r"|(?:你)?(?:给|送|发)(?:我)?的(?:真|太|很|挺)?(?:好|不错|好看|可爱)"
    r"|(?:那|这)?(?:我)?(?:要|选|拿)(?:大|小|左边|右边|左|右)(?:的|那个|这个)?"
    r"|哈{2,8}|笑死(?:我)?"
    r")(?:了|啊|呀|诶|哇|吧|呢|耶|哦|喔)*",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"(?:这|那)(?:个|些|两个|两种)?(?:要|应该|该)?"
    r"(?:怎么(?:分辨|区分|辨别|判断|认出)|如何(?:分辨|区分|辨别|判断)"
    r"|是(?:什么|谁)|有什么(?:区别|不同)|哪个(?:是|才是)(?:真的|假的|对的|正确的))"
    r"(?:呢|啊|呀|吗)?",
    re.IGNORECASE,
)


def is_short_visual_reaction(text: str, *, aliases=None) -> bool:
    value = re.sub(r"\s+", "", str(text or "")).strip("。，、！？!?；;：:…~～")
    if not value or len(value) > 28:
        return False
    pattern = _ADDRESS_RE if aliases is None else re.compile(
        "^(?:" + "|".join(re.escape(re.sub(r"\s+", "", name)) for name in sorted(aliases, key=len, reverse=True)) + ")[，,：:、]*", re.IGNORECASE)
    value = pattern.sub("", value, count=1)
    # Every clause must be a reaction; "好看，不过今晚吃什么" is a topic switch.
    clauses = re.split(r"[，,。！!？?；;]+", value)
    return bool(clauses) and all(_REACTION_RE.fullmatch(clause) for clause in clauses)


def is_short_visual_question(text: str, *, aliases=None) -> bool:
    """Recognize a deictic candidate hint, without confirming an image source."""
    value = re.sub(r"\s+", "", str(text or "")).strip("。，、！？!?；;：:…~～")
    if not value or len(value) > 28:
        return False
    pattern = _ADDRESS_RE if aliases is None else re.compile(
        "^(?:" + "|".join(re.escape(re.sub(r"\s+", "", name)) for name in sorted(aliases, key=len, reverse=True)) + ")[，,：:、]*", re.IGNORECASE)
    value = pattern.sub("", value, count=1)
    return bool(_QUESTION_RE.fullmatch(value))
