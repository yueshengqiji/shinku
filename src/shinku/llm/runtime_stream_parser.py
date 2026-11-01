"""Incremental top-level JSON field parser for streamed replies."""
from __future__ import annotations
import re
from typing import Any
from .runtime_reply_format import JSON_ESCAPE_MAP, normalize_reply_medium

class StreamTap:
    """流式顶层 JSON 的增量抽取器。

    逐字符扫过流式片段，在对象还没拼完整时就把 ``emotion`` / ``speech`` /
    ``reply_medium`` / ``speech_segments`` 抽出来变成 UI 事件，让不同宿主
    能边收边播。``feed()`` 返回本次新产生的事件列表。
    """

    def __init__(self):
        self.depth = 0
        self.in_string = False
        self.string_role = ""
        self.string_buffer: list[str] = []
        self.current_key = ""
        self.captured_value_key: str | None = None
        self.expecting_key = False
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False
        self.escape_pending = False
        self.unicode_buffer: list[str] | None = None
        self.latest_emotion = ""
        self.latest_speech = ""
        self.latest_reply_medium = ""
        self._ui_hinted = False
        self._medium_hinted = False
        self._segment_count = 0
        self._segment_cursor = 0
        self._segment_limit = 3
        self._segments_array_depth: int | None = None
        self._emitted_segment_keys: set[str] = set()
        self._inline_speech_seen = False
        self._speech_segments: list[str] = []

    def feed(self, text: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        speech_delta: list[str] = []
        for char in str(text or ""):
            if self.in_string:
                self._consume_char(char, events, speech_delta)
                continue

            if self.in_primitive:
                if self.depth == 1 and char == ",":
                    self._reset_pair()
                    continue
                if self.depth == 1 and char == "}":
                    self.in_primitive = False
                    self._clear_pair()
                    self.depth = max(0, self.depth - 1)
                    continue
                if char in "{[":
                    self.depth += 1
                elif char in "}]":
                    self.depth = max(0, self.depth - 1)
                continue

            if char in " \t\r\n":
                continue

            if self._segments_array_depth is not None:
                if char == '"' and self.depth == self._segments_array_depth:
                    self._begin_string("speech_segment")
                    continue
                if char == "]" and self.depth == self._segments_array_depth:
                    self._segments_array_depth = None
                    self.depth = max(0, self.depth - 1)
                    continue

            if char == "{":
                self.depth += 1
                if self.depth == 1:
                    self.expecting_key = True
                    self.expecting_colon = False
                    self.expecting_value = False
                    self.current_key = ""
                elif self.expecting_value:
                    self.expecting_value = False
                continue

            if char == "[":
                self.depth += 1
                if self.expecting_value:
                    if self.current_key == "speech_segments" and self.depth == 2:
                        self._segments_array_depth = self.depth
                    self.expecting_value = False
                continue

            if char == "}":
                if self.depth == 1:
                    self._clear_pair()
                self.depth = max(0, self.depth - 1)
                continue

            if char == "]":
                self.depth = max(0, self.depth - 1)
                continue

            if self.depth != 1:
                continue

            if char == ",":
                self._reset_pair()
                continue

            if self.expecting_key:
                if char == '"':
                    self._begin_string("key")
                continue

            if self.expecting_colon:
                if char == ":":
                    self.expecting_colon = False
                    self.expecting_value = True
                continue

            if self.expecting_value:
                if char == '"':
                    self._begin_string("value")
                    self.captured_value_key = self.current_key
                else:
                    self.expecting_value = False
                    self.in_primitive = True
                continue

        if speech_delta:
            delta_text = "".join(speech_delta)
            if delta_text:
                self.latest_speech += delta_text
                events.append({"type": "speech_chunk", "text": delta_text})
                if self._segment_count < self._segment_limit:
                    remaining = self.latest_speech[self._segment_cursor:]
                    match = re.search(r"[。！？!?\n]", remaining)
                    if match:
                        end = self._segment_cursor + match.end()
                        self._emit_segment(
                            events,
                            self.latest_speech[self._segment_cursor:end],
                        )
                        self._segment_cursor = end
        return events

    def _begin_string(self, role: str) -> None:
        self.in_string = True
        self.string_role = role
        self.string_buffer = []
        self.escape_pending = False
        self.unicode_buffer = None

    def _consume_char(
        self,
        char: str,
        events: list[dict[str, Any]],
        speech_delta: list[str],
    ) -> None:
        if self.unicode_buffer is not None:
            if char.lower() in "0123456789abcdef":
                self.unicode_buffer.append(char)
                if len(self.unicode_buffer) == 4:
                    decoded = ""
                    try:
                        decoded = chr(int("".join(self.unicode_buffer), 16))
                    except Exception:
                        decoded = ""
                    self.unicode_buffer = None
                    if decoded:
                        self._append_char(decoded, speech_delta)
                return

            self.unicode_buffer = None
            self._append_char("u", speech_delta)

        if self.escape_pending:
            self.escape_pending = False
            if char == "u":
                self.unicode_buffer = []
                return
            self._append_char(JSON_ESCAPE_MAP.get(char, char), speech_delta)
            return

        if char == "\\":
            self.escape_pending = True
            return

        if char == '"':
            self.in_string = False
            text = "".join(self.string_buffer)
            if self.string_role == "key":
                self.current_key = text
                self.expecting_key = False
                self.expecting_colon = True
            elif self.string_role == "speech_segment":
                segment_text = self._tidy_segment_text(text)
                if segment_text:
                    self._speech_segments.append(segment_text)
                    if not self._inline_speech_seen:
                        self.latest_speech = "\n".join(self._speech_segments)
                    self._emit_segment(events, segment_text)
            else:
                if self.captured_value_key == "emotion":
                    self.latest_emotion = text
                    if text and not self._ui_hinted:
                        self._ui_hinted = True
                        events.append({"type": "ui", "emotion": text})
                elif self.captured_value_key == "reply_medium":
                    medium = normalize_reply_medium(text)
                    if medium:
                        self.latest_reply_medium = medium
                        if not self._medium_hinted:
                            self._medium_hinted = True
                            events.append({"type": "delivery_hint", "medium": medium})
                self.expecting_value = False
                self.captured_value_key = None
            self.string_role = ""
            self.string_buffer = []
            return

        self._append_char(char, speech_delta)

    def _append_char(self, char: str, speech_delta: list[str]) -> None:
        self.string_buffer.append(char)
        if self.string_role == "value" and self.captured_value_key == "speech":
            # 顶层 speech 与 speech_segments 二选一：见到前者就清掉后者的缓冲。
            if not self._inline_speech_seen and self._speech_segments:
                self.latest_speech = ""
            self._inline_speech_seen = True
            speech_delta.append(char)

    def _emit_segment(self, events: list[dict[str, Any]], text: str) -> bool:
        if self._segment_count >= self._segment_limit:
            return False
        segment_text = self._tidy_segment_text(text)
        if not segment_text:
            return False
        key = re.sub(r"\s+", "", segment_text)
        if not key or key in self._emitted_segment_keys:
            return False
        self._emitted_segment_keys.add(key)
        self._segment_count += 1
        events.append({
            "type": "speech_segment",
            "index": self._segment_count - 1,
            "text": segment_text,
        })
        return True

    def _tidy_segment_text(self, text: Any) -> str:
        return " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()

    def _reset_pair(self) -> None:
        self.current_key = ""
        self.captured_value_key = None
        self.expecting_key = True
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False

    def _clear_pair(self) -> None:
        self.current_key = ""
        self.captured_value_key = None
        self.expecting_key = False
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False
