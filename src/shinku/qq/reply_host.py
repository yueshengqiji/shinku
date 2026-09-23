"""Shape final QQ text without changing what the model meant.

The LLM decides the content.  This small boundary only decides how a final
text reply is delivered: sentence boundaries stay intact, configured final
periods are removed after splitting, short replies are kept compact, and
long/technical replies may use a few more messages.  It deliberately knows
nothing about providers or old project configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["ReplyHost", "ReplyShape"]


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
_SOFT_SPLIT_RE = re.compile(r"(?<=[，,、：:])\s+|\s+")
_URL_RE = re.compile(r"https?://[^\s<>'\"，。！？；;]+", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"(```[\s\S]*?```)")
_FINAL_PERIOD_RE = re.compile(r"[。．.]+$")


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
    """Turn one final text into a bounded sequence of QQ text messages."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        long_trigger_chars: int = 360,
        max_segment_chars: int = 600,
        short_max_segments: int = 2,
        long_max_segments: int = 4,
        strip_final_periods: bool = False,
    ) -> None:
        self._enabled = bool(enabled)
        self.long_trigger_chars = max(80, min(2400, int(long_trigger_chars or 360)))
        self.max_segment_chars = max(120, min(1800, int(max_segment_chars or 600)))
        self.short_max_segments = max(1, min(6, int(short_max_segments or 2)))
        self.long_max_segments = max(
            self.short_max_segments,
            min(10, int(long_max_segments or 4)),
        )
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
        if not self.enabled():
            return [value]

        expanded = self.should_expand(value, user_message=user_message)
        if expanded:
            pieces = self._split_text(value)
            limit = self.long_max_segments
            reason = "long_or_technical"
        else:
            # Even ordinary replies regain the requested period-based bubble
            # behavior.  A one-sentence reply remains one message.  When a
            # compact reply already contains several complete sentence beats,
            # preserve those boundaries instead of hiding the tail inside the
            # second bubble; the configured long limit still bounds it.
            pieces = self._split_sentences(value)
            limit = self._compact_sentence_limit(value, pieces)
            reason = "sentence_punctuation"
        limited = self._limit([item for item in pieces if item], limit)
        if not self.strip_final_periods:
            return limited
        cleaned: list[str] = []
        for item in limited:
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

    def max_segments_for_message(self, *, user_message: str = "", expanded: bool = False) -> int:
        return self.long_max_segments if expanded or self.should_expand("", user_message=user_message) else self.short_max_segments

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
        return [item.strip() for item in _SENTENCE_SPLIT_RE.split(str(text or "")) if item.strip()]

    def _compact_sentence_limit(self, text: str, pieces: list[str]) -> int:
        """Choose a delivery cap without gluing short complete sentences.

        ``short_max_segments`` remains the compact default.  A reply that is
        short overall and consists of at most the long-message cap's worth of
        small sentence beats is already deliberately punctuated by the model,
        so keeping each beat is more faithful than merging the tail into the
        last bubble.  Longer or denser output keeps the original compact cap.
        """

        limit = self.short_max_segments
        if len(pieces) <= limit:
            return limit
        compact_piece_limit = min(self.max_segment_chars, 180)
        if (
            len(pieces) <= self.long_max_segments
            and len(str(text or "").strip()) <= self.long_trigger_chars
            and all(len(str(item or "").strip()) <= compact_piece_limit for item in pieces)
        ):
            return len(pieces)
        return limit

    def _split_overlong(self, text: str) -> list[str]:
        if len(text) <= self.max_segment_chars:
            return [text]
        urls: list[str] = []

        def protect(match: re.Match[str]) -> str:
            urls.append(match.group(0))
            return f" __SHINKU_URL_{len(urls) - 1}__ "

        protected = _URL_RE.sub(protect, text)
        chunks = [item.strip() for item in _SOFT_SPLIT_RE.split(protected) if item.strip()]
        result: list[str] = []
        current = ""
        for chunk in chunks:
            candidate = f"{current} {chunk}".strip()
            if current and len(candidate) > self.max_segment_chars:
                result.append(current)
                current = chunk
            else:
                current = candidate
        if current:
            result.append(current)
        if not result:
            result = [protected]

        restored: list[str] = []
        for item in result:
            for index, url in enumerate(urls):
                item = item.replace(f"__SHINKU_URL_{index}__", url)
            if len(item) <= self.max_segment_chars:
                restored.append(item.strip())
                continue
            restored.extend(
                item[index : index + self.max_segment_chars].strip()
                for index in range(0, len(item), self.max_segment_chars)
            )
        return [item for item in restored if item]

    def _split_code(self, text: str) -> list[str]:
        if len(text) <= self.max_segment_chars:
            return [text]
        lines = text.splitlines()
        result: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines:
            extra = len(line) + (1 if current else 0)
            if current and length + extra > self.max_segment_chars:
                result.append("\n".join(current).strip())
                current, length = [], 0
            current.append(line)
            length += extra
        if current:
            result.append("\n".join(current).strip())
        return [item for item in result if item]

    @staticmethod
    def _strip_final_period(text: str) -> str:
        """Remove only sentence-final full stops, never punctuation in code."""

        value = str(text or "").strip()
        if not value or value.lstrip().startswith("```"):
            return value
        return _FINAL_PERIOD_RE.sub("", value).rstrip()

    @staticmethod
    def _limit(messages: list[str], limit: int) -> list[str]:
        clean = [str(item or "").strip() for item in messages if str(item or "").strip()]
        limit = max(1, int(limit or 1))
        if len(clean) <= limit:
            return clean
        # Do not discard the tail of a technical answer; combine overflow in
        # the last bubble instead of silently truncating it.
        return [*clean[: limit - 1], "\n".join(clean[limit - 1:]).strip()]
