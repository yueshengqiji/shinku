"""中英混排文本切词：给检索侧做词袋用的最小实现。

汉字没有分词库可用，于是按「整串保留 + 再补所有 2 字与 3 字滑窗」展开，
让长词与它的片段都能被检索到；拉丁字母、数字与下划线按整串保留。
切分前先小写化，所以英文词一律以小写形态出现。
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")

STOPWORDS = {
    "的",
    "了",
    "呢",
    "啊",
    "呀",
    "吗",
    "吧",
    "哦",
    "喵",
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "然后",
    "就是",
    "这个",
    "那个",
    "现在",
    "一下",
}

#: 汉字片段的识别口径：整段都是汉字才算「汉字串」，混排的交给整串保留那一支。
_CJK_ONLY = re.compile(r"[\u4e00-\u9fff]+")

#: 滑窗宽度。3 字以内不补滑窗——短串本身就是它自己的片段。
_SLIDING_WIDTHS = (2, 3)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _expanded(word: str) -> list[str]:
    """汉字串展开成「整串 + 滑窗片段」；长度不超过最大窗宽时只给整串。"""
    if len(word) <= max(_SLIDING_WIDTHS):
        return [word]
    pieces = [word]
    for width in _SLIDING_WIDTHS:
        pieces.extend(
            word[start : start + width]
            for start in range(len(word) - width + 1)
        )
    return pieces


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    collected: list[str] = []
    for chunk in TOKEN_RE.findall(normalized.lower()):
        if _CJK_ONLY.fullmatch(chunk):
            collected.extend(_expanded(chunk))
        else:
            collected.append(chunk)
    return [piece for piece in collected if piece and piece not in STOPWORDS]


__all__ = ["STOPWORDS", "TOKEN_RE", "normalize_text", "tokenize"]
