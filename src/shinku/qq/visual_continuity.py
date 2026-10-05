"""Conservative, short reactions to an already established image topic.

This is not image-reference resolution. Callers must separately enforce the
same sender/conversation, an open topic, a short time window and the existing
image budget. Full-clause matching keeps a new topic containing a compliment
from accidentally reviving an old picture.
"""

from __future__ import annotations

import re


VISUAL_REACTION_WINDOW_SECONDS = 60.0
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


def is_short_visual_reaction(text: str) -> bool:
    value = re.sub(r"\s+", "", str(text or "")).strip("。，、！？!?；;：:…~～")
    if not value or len(value) > 28:
        return False
    value = _ADDRESS_RE.sub("", value, count=1)
    # Every clause must be a reaction; "好看，不过今晚吃什么" is a topic switch.
    clauses = re.split(r"[，,。！!？?；;]+", value)
    return bool(clauses) and all(_REACTION_RE.fullmatch(clause) for clause in clauses)
