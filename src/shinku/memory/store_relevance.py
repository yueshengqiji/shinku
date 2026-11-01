"""查询相关性特征：由 Store 在每次查询内构造，不写入记忆或改写编码输入。

词法覆盖反映正文及已登记参与者署名的匹配；精确通道识别明确引用或技术标识符。重要性、时间、
话题和发送者由原始来源评分单独提供。所有特征都是检索信号，不代表事实成立。
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_QUOTES = re.compile(r'“([^”\n]+)”|「([^」\n]+)」|『([^』\n]+)』|"([^"\n]+)"|`([^`\n]+)`')
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_./:\\-]{1,127}")
_BOUNDARY = r"[A-Za-z0-9_./:\\-]"
_RIGHT_BOUNDARY = r"(?:[A-Za-z0-9_/:\\-]|\.[A-Za-z0-9_])"
# 去掉询问动作，不删除否定词、人物或具体条件；未命中不会获得词法贡献。
_QUESTION_TERMS = frozenset({"之前", "提到", "什么", "情况", "记得", "说过", "我说", "你记",
                           "帮我", "一下", "请问", "哪个", "哪里", "怎么", "查询", "查找"})


@dataclass(frozen=True)
class ExactAnchor:
    text: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class QueryRelevance:
    terms: tuple[str, ...]
    anchors: tuple[ExactAnchor, ...]

    def scores(self, text: str) -> tuple[float, float]:
        """返回词法覆盖和精确覆盖；各自归一化，不累加重复出现次数。"""
        body = text.lower()
        lexical = sum(term in body for term in self.terms) / len(self.terms) if self.terms else 0.0
        exact = sum(bool(anchor.pattern.search(body)) for anchor in self.anchors) / len(self.anchors) if self.anchors else 0.0
        return lexical, exact


def query_relevance(query: str, terms: list[str]) -> QueryRelevance:
    """只从当前查询抽取有界锚点；不从答案或记忆正文反推检索词。

    引号内至少六个非空白字符才允许片段匹配。未引用的标识符须包含数字或
    分隔符，且匹配两端边界，防止 R7 命中 R70、v2.1 命中 v2.10。
    """
    lowered = query.lower()
    anchors: dict[str, ExactAnchor] = {}
    quoted_spans = []
    for match in _QUOTES.finditer(lowered):
        value = next(group for group in match.groups() if group is not None).strip()
        if 6 <= len(value.replace(" ", "")) <= 160:
            quoted_spans.append(match.span())
            anchors[value] = ExactAnchor(value, re.compile(re.escape(value)))
    for match in _IDENTIFIER.finditer(lowered):
        if any(start <= match.start() < end for start, end in quoted_spans):
            continue
        value = match.group().rstrip(".")
        if not any(char.isdigit() or char in "_./:\\-" for char in value):
            continue
        # 句末英文句号是标点；.bak/.10 等继续组成标识符，不能视作边界。
        anchors.setdefault(value, ExactAnchor(value, re.compile(r"(?<!" + _BOUNDARY + ")" + re.escape(value) + r"(?!" + _RIGHT_BOUNDARY + ")")))
    return QueryRelevance(tuple(term for term in terms if term not in _QUESTION_TERMS), tuple(anchors.values())[:8])


def attributed_search_text(text: str, metadata) -> str:
    """参与者经历按本机标签/署名检索，正文和证据版本仍保持原样。

    只扩展显式收录的经历；名称仅影响相关性，不决定身份、范围或权限。
    """
    if not metadata.get("owner_participant_id"):
        return text
    labels = [str(metadata.get(key) or "")[:64] for key in
              ("owner_participant_label", "owner_participant_id", "sender_name")]
    return " ".join((text, *labels))
