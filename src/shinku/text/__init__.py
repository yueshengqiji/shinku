"""中英混排的切词与归一化，供检索侧做词袋。

没有引入分词库，汉字因此按「整串 + 2/3 字滑窗」展开，拉丁字母与数字按整串保留。
小写化在切分**之前**发生，所以英文词一律是小写形态。

C2-1 只落地检索侧需要的最小件（``TOKEN_RE``／``STOPWORDS``／``normalize_text``／``tokenize``）；
时间表达解析、话题抽取、聊天渲染属 C5，不进本包。
"""

from __future__ import annotations

from .tokenizer import STOPWORDS, TOKEN_RE, normalize_text, tokenize

__all__ = [
    "STOPWORDS",
    "TOKEN_RE",
    "normalize_text",
    "tokenize",
]
