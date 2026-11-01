"""Reply-medium normalization and JSON escape decoding constants."""
from __future__ import annotations
from typing import Any

JSON_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


REPLY_MEDIUM_ALIASES = {
    "text": "text",
    "文字": "text",
    "文本": "text",
    "voice": "voice",
    "audio": "voice",
    "record": "voice",
    "语音": "voice",
    "both": "both",
    "all": "both",
    "text_voice": "both",
    "voice_text": "both",
    "文字语音": "both",
    "双发": "both",
}


def normalize_reply_medium(value: Any) -> str:
    """把用户输入的媒介词归一成 ``text`` / ``voice`` / ``both``，认不出给空串。"""

    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    return REPLY_MEDIUM_ALIASES.get(text, "")
