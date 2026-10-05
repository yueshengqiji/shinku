"""Split final QQ text at sentence boundaries without changing its meaning.

The model decides the content and response length. This boundary only shapes
delivery: each sentence/paragraph beat becomes its own message, configured
final periods are removed after splitting, and exceptionally long prose is
divided to fit the segment target. Links, inline code and indivisible code lines
may exceed that soft target: changing executable/link content is never allowed.
There is no message-count cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["ReplyHost", "ReplyShape", "clean_voice_text"]


_TECHNICAL_TERMS = (
    "代码", "脚本", "函数", "报错", "异常", "日志", "接口", "api", "端口",
    "数据库", "配置", "部署", "命令", "终端", "服务器", "进程", "模型",
    "token", "上下文", "缓存", "测试", "调试", "排查", "修复", "安装",
    "编程", "traceback", "exception", "stack trace",
)
_EXPLICIT_LONG_HINTS = (
    "详细", "展开说", "具体解释", "完整步骤", "一步一步", "长一点", "多说一点",
    "分析一下", "原理",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_URL_RE = re.compile(r"https?://[^\s<>'\"`，。！？；]+", re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]+`+")
_ATOMIC_RE = re.compile(_URL_RE.pattern + "|" + _INLINE_CODE_RE.pattern, re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"(```[\s\S]*?```)")
_PROTECTED_RE = re.compile(r"```[\s\S]*?```|" + _ATOMIC_RE.pattern, re.IGNORECASE)
_FINAL_PERIOD_RE = re.compile(r"[。．.]+$")
_DASH_RE = re.compile(r"[—―]{2,}")
_CHALLENGE_CLOSER_RE = re.compile(r"(?:，|。|\s)+(?:说吧|你说呢)[？?。！!]?$")
_CYNICAL_COAX_RE = re.compile(r"少拿这种话(?:来)?哄我[。！!]*")


def clean_voice_text(text: str) -> str:
    """平滑折叠辩手破折号、尾部挑战反问与市井防备腔，代码块内严格保留。"""
    raw = str(text or "")
    if not raw or (
        "——" not in raw
        and "――" not in raw
        and "说吧" not in raw
        and "你说呢" not in raw
        and "哄我" not in raw
    ):
        return raw
    cleaned_parts: list[str] = []
    position = 0
    for match in _PROTECTED_RE.finditer(raw):
        cleaned_parts.append(_clean_prose_voice(raw[position:match.start()]))
        cleaned_parts.append(match.group())
        position = match.end()
    cleaned_parts.append(_clean_prose_voice(raw[position:]))
    return "".join(cleaned_parts)


def _clean_prose_voice(text: str) -> str:
    value = _DASH_RE.sub("，", text)
    value = _CHALLENGE_CLOSER_RE.sub("。", value)
    return _CYNICAL_COAX_RE.sub("……头转过去。", value)


@dataclass(frozen=True)
class ReplyShape:
    messages: tuple[str, ...]
    expanded: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages": list(self.messages),
            "expanded": self.expanded,
            "reason": self.reason,
        }


class ReplyHost:
    """Turn one final text into sentence-sized QQ messages without a count cap."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        long_trigger_chars: int = 360,
        max_segment_chars: int = 600,
        strip_final_periods: bool = False,
    ) -> None:
        self._enabled = bool(enabled)
        self.long_trigger_chars = max(80, min(2400, int(long_trigger_chars or 360)))
        self.max_segment_chars = max(120, min(1800, int(max_segment_chars or 600)))
        self.strip_final_periods = bool(strip_final_periods)

    def enabled(self) -> bool:
        return self._enabled

    def should_expand(self, text: str, *, user_message: str = "") -> bool:
        combined = f"{text}\n{user_message}".lower()
        return (
            any(term.lower() in combined for term in _TECHNICAL_TERMS)
            or any(term in str(user_message or "") for term in _EXPLICIT_LONG_HINTS)
            or len(str(text or "").strip()) > self.long_trigger_chars
        )

    def shape_text(self, text: str, *, user_message: str = "") -> list[str]:
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            return []
        value = clean_voice_text(value).strip()
        if not value:
            return []
        if not self.enabled():
            return [value]

        pieces = [item for item in self._split_text(value) if item]
        if not self.strip_final_periods:
            return pieces
        cleaned: list[str] = []
        for item in pieces:
            value = self._strip_final_period(item)
            if value:
                cleaned.append(value)
        return cleaned

    def shape(self, text: str, *, user_message: str = "") -> ReplyShape:
        value = str(text or "").strip()
        expanded = self.should_expand(value, user_message=user_message) if value else False
        messages = tuple(self.shape_text(value, user_message=user_message))
        return ReplyShape(messages=messages, expanded=expanded, reason="" if not messages else (
            "long_or_technical" if expanded else "sentence_punctuation"
        ))

    def _split_text(self, text: str) -> list[str]:
        result: list[str] = []
        for part in _CODE_FENCE_RE.split(text):
            if not part.strip():
                continue
            if part.lstrip().startswith("```"):
                result.extend(self._split_code(part.strip()))
                continue
            for line in (item.strip() for item in part.splitlines() if item.strip()):
                for sentence in self._split_sentences(line):
                    result.extend(self._split_overlong(sentence))
        return result or [text]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Sentence punctuation inside ?query URLs and inline commands is data,
        # not a conversation boundary. Protect it BEFORE any sentence split.
        result: list[str] = []
        current = ""

        def append_prose(value: str) -> None:
            nonlocal current
            position = 0
            for boundary in _SENTENCE_SPLIT_RE.finditer(value):
                current += value[position:boundary.start()]
                if current.strip():
                    result.append(current.strip())
                current = ""
                position = boundary.end()
            current += value[position:]

        value = str(text or "")
        position = 0
        for match in _ATOMIC_RE.finditer(value):
            append_prose(value[position:match.start()])
            current += match.group()
            position = match.end()
        append_prose(value[position:])
        if current.strip():
            result.append(current.strip())
        return result

    def _split_overlong(self, text: str) -> list[str]:
        if len(text) <= self.max_segment_chars:
            return [text]
        # Work with explicit atomic spans rather than short placeholders whose
        # restored URL was previously cut into arbitrary character fragments.
        chunks: list[tuple[str, bool]] = []
        position = 0
        for match in _ATOMIC_RE.finditer(text):
            chunks.extend((item, False) for item in re.findall(r"\s+|\S+", text[position:match.start()]))
            chunks.append((match.group(), True))
            position = match.end()
        chunks.extend((item, False) for item in re.findall(r"\s+|\S+", text[position:]))
        result: list[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current.strip():
                result.append(current.strip())
            current = ""

        for chunk, atomic in chunks:
            if len(current) + len(chunk) <= self.max_segment_chars:
                current += chunk
                continue
            flush()
            if atomic:
                current = chunk  # An overlong link/command stays independently copyable.
                continue
            while len(chunk) > self.max_segment_chars:
                current, chunk = chunk[:self.max_segment_chars], chunk[self.max_segment_chars:]
                flush()
            current = chunk
        flush()
        return result

    def _split_code(self, text: str) -> list[str]:
        if len(text) <= self.max_segment_chars:
            return [text]
        lines = text.split("\n")
        if len(lines) < 3 or lines[-1].strip() != "```":
            return [text]
        opening, closing = lines[0], lines[-1]
        body_budget = max(1, self.max_segment_chars - len(opening) - len(closing) - 2)
        result: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines[1:-1]:
            extra = len(line) + (1 if current else 0)
            if current and length + extra > body_budget:
                result.append(opening + "\n" + "\n".join(current) + "\n" + closing)
                current, length = [], 0
                extra = len(line)
            current.append(line)
            length += extra
        if current:
            result.append(opening + "\n" + "\n".join(current) + "\n" + closing)
        return result or [text]

    @staticmethod
    def _strip_final_period(text: str) -> str:
        """Remove only sentence-final full stops, never punctuation in code."""

        value = str(text or "").strip()
        if not value or value.lstrip().startswith("```"):
            return value
        if any(match.end() == len(value) for match in _ATOMIC_RE.finditer(value)):
            return value
        return _FINAL_PERIOD_RE.sub("", value).rstrip()
