"""Conversation-depth and low-frequency voice-habit control for QQ turns.

This module does not provide reply templates and never rewrites generated text.
It keeps a small, in-memory conversation state so the model can distinguish a
new topic from a follow-up without adding another LLM call.  The stable system
guidance is intentionally qualitative; the existing ReplyHost remains the hard
delivery boundary for message count and sentence splitting.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .turns import QQTurn

__all__ = [
    "CHAT_FLOW_SYSTEM_GUIDANCE",
    "ChatFlowController",
    "ChatFlowDecision",
    "append_chat_flow_guidance",
    "is_address_only_message",
]


CHAT_FLOW_SYSTEM_GUIDANCE = """【聊天节奏与表达变化】
根据当前语境自主判断答复深度。新话题或首次回应时，先给解决眼前内容所需的最小充分答复，不为了显得健谈而补充无关背景。对方继续同一话题，并且提出了新的问题、补充、反驳或未解决点时，再自然展开；没有新的内容就简洁承接。
上一轮较长不代表这一轮必须更长。完整处理当前内容后回到普通节奏，不要让回复长度逐轮累加。
“嗯”“我在这里”以及类似的陪伴性表达不是固定开场。只有确实需要停顿、确认陪伴或安慰时才使用；连续几轮出现同类表达时，自然换成直接判断、具体回应或合适的停顿。不要机械禁用这些表达，也不要为了变化而刻意换词。
被单独点名、但没有问题、任务、图片或明确话题时，把它视为一次注意力召回：先接住当前尚未结束的话题；确实没有可承接内容时才做最小确认。不要把点名自动回答成同一套确认句，也不要为了继续聊天机械追问。"""


_DEFAULT_ADDRESS_ALIASES = (
    "二阶堂真红",
    "真红",
    "Nikaidou Shinku",
    "Shinku",
    "老婆",
)
_ADDRESS_PUNCTUATION_RE = re.compile(
    r"[\s\u3000\u200b\u200c\u200d。，、！？!?；;：:“”\"'‘’（）()【】\[\]<>《》…·~～—_-]+"
)
_ADDRESS_ONLY_FILLERS = frozenset(
    {
        "早",
        "早安",
        "早上好",
        "上午好",
        "午安",
        "中午好",
        "下午好",
        "晚安",
        "晚上好",
        "早啊",
        "晚安呀",
        "你好呀",
        "你好",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "hey",
        "回来了",
        "我回来了",
        "我来了",
        "来啦",
        "我来啦",
        "回来啦",
        "在吗",
        "在么",
        "在不在",
        "在不",
        "还在吗",
        "醒着吗",
        "听得到吗",
        "你在吗",
        "你还在吗",
        "啊",
        "呀",
        "呢",
        "喂",
    }
)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ACK_OPEN_RE = re.compile(r"^(?:嗯|嗯嗯)(?:[…….…，,、：:;；\s]|$)")
_PRESENCE_OPEN_RE = re.compile(r"^我(?:就)?在这里(?:[…….…，,、：:;；\s]|$)")
_ADDRESS_OPEN_RE = re.compile(r"^(?:你啊|你还真是)(?:[…….…，,、：:;；\s]|$)")


def append_chat_flow_guidance(system_prompt: str) -> str:
    """Append one stable guidance block without duplicating it.

    Keeping this block stable preserves the system-prefix cache.  Per-turn
    state is added to the user-side prompt tail by ``QQAgentBridge`` instead.
    """

    base = str(system_prompt or "").strip()
    if CHAT_FLOW_SYSTEM_GUIDANCE in base:
        return base
    if not base:
        return CHAT_FLOW_SYSTEM_GUIDANCE
    return f"{base}\n\n{CHAT_FLOW_SYSTEM_GUIDANCE}"


@dataclass(frozen=True)
class ChatFlowDecision:
    """A soft per-turn hint; it is not a reply template or a hard ban."""

    key: str
    stage: str
    same_topic: bool
    allow_expansion: bool
    reason: str
    guidance: str
    habit_guidance: str = ""
    address_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "same_topic": self.same_topic,
            "allow_expansion": self.allow_expansion,
            "reason": self.reason,
            "guidance": self.guidance,
            "habit_guidance": self.habit_guidance,
            "address_only": self.address_only,
        }


@dataclass
class _FlowState:
    last_user_text: str = ""
    last_sender_id: str = ""
    last_at: float = 0.0
    last_reply_ids: tuple[str, ...] = ()
    last_expanded: bool = False
    cooldown_remaining: int = 0
    recent_openings: tuple[str, ...] = ()


class ChatFlowController:
    """Track topic continuity without creating a sticky long-reply mode.

    The controller deliberately does not decide what the model should say.
    It only supplies a small context hint and records whether the previous
    answer expanded.  The model still makes the semantic decision under the
    fixed system guidance, while ReplyHost keeps existing hard limits intact.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_seconds: float = 600.0,
        cooldown_turns: int = 1,
        voice_window: int = 4,
        clock: Callable[[], float] | None = None,
        max_states: int = 256,
    ) -> None:
        self.enabled = bool(enabled)
        self.ttl_seconds = max(30.0, min(86400.0, float(ttl_seconds or 600.0)))
        self.cooldown_turns = max(0, min(3, int(cooldown_turns or 0)))
        self.voice_window = max(2, min(12, int(voice_window or 4)))
        self._clock = clock or time.time
        self._max_states = max(16, min(4096, int(max_states or 256)))
        self._states: dict[str, _FlowState] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> "ChatFlowController":
        env = os.environ if environ is None else environ

        def flag(key: str, default: bool) -> bool:
            raw = str(env.get(key, "") or "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def number(key: str, default: float, cast: Callable[[str], Any]) -> Any:
            raw = str(env.get(key, "") or "").strip()
            if not raw:
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=flag("SHINKU_QQ_CHAT_FLOW_ENABLED", True),
            ttl_seconds=number("SHINKU_QQ_CHAT_FLOW_TTL_SECONDS", 600.0, float),
            cooldown_turns=number("SHINKU_QQ_CHAT_FLOW_COOLDOWN_TURNS", 1, int),
            voice_window=number("SHINKU_QQ_CHAT_FLOW_VOICE_WINDOW", 4, int),
            clock=clock,
        )

    def begin_turn(self, turn: "QQTurn") -> ChatFlowDecision:
        """Classify the current turn as fresh, follow-up, or cooldown."""

        key = self._conversation_key(turn)
        address_only = _is_address_only_turn(turn)
        if not self.enabled:
            return ChatFlowDecision(
                key=key,
                stage="disabled",
                same_topic=False,
                allow_expansion=True,
                reason="chat_flow_disabled",
                guidance="",
                address_only=address_only,
            )

        now = float(self._clock())
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(key)
            habit_guidance = self._habit_guidance(state)
            if state is None or now - state.last_at > self.ttl_seconds:
                return ChatFlowDecision(
                    key=key,
                    stage="fresh",
                    same_topic=False,
                    allow_expansion=False,
                    reason="new_topic_or_expired",
                    guidance="本轮是当前话题的首次回应：先解决眼前内容，避免无关扩展。",
                    habit_guidance=habit_guidance,
                    address_only=address_only,
                )

            same_topic, reason = self._is_continuation(turn, state, now=now)
            if not same_topic:
                return ChatFlowDecision(
                    key=key,
                    stage="fresh",
                    same_topic=False,
                    allow_expansion=False,
                    reason=reason,
                    guidance="话题看起来已经切换：直接处理当前内容，不要把上一话题的长度或情绪带过来。",
                    habit_guidance=habit_guidance,
                    address_only=address_only,
                )

            if state.cooldown_remaining > 0:
                return ChatFlowDecision(
                    key=key,
                    stage="cooldown",
                    same_topic=True,
                    allow_expansion=False,
                    reason="same_topic_after_expansion",
                    guidance="本轮仍在延续同一话题，但上一轮已经展开过：只回答本轮新增内容；确实需要说明时再完整展开。",
                    habit_guidance=habit_guidance,
                    address_only=address_only,
                )
            return ChatFlowDecision(
                key=key,
                stage="followup",
                same_topic=True,
                allow_expansion=True,
                reason=reason,
                guidance="本轮延续了同一话题：先判断是否有新的未解决点，有才自然展开，没有就简洁承接。",
                habit_guidance=habit_guidance,
                address_only=address_only,
            )

    def finish_turn(
        self,
        turn: "QQTurn",
        *,
        expanded: bool,
        reply_message_ids: tuple[str, ...] = (),
        at: float | None = None,
        final_text: str = "",
    ) -> None:
        """Record one completed natural-language turn for the next decision."""

        if not self.enabled:
            return
        now = float(self._clock() if at is None else at)
        key = self._conversation_key(turn)
        sender_id = self._sender_id(turn)
        opening = _opening_marker(final_text)
        with self._lock:
            self._prune_locked(now)
            previous = self._states.get(key) or _FlowState()
            openings = list(previous.recent_openings)
            openings.append(opening)
            self._states[key] = _FlowState(
                last_user_text=str(getattr(turn, "combined_text", "") or "").strip(),
                last_sender_id=sender_id,
                last_at=now,
                last_reply_ids=tuple(str(item).strip() for item in reply_message_ids if str(item).strip()),
                last_expanded=bool(expanded),
                cooldown_remaining=(
                    self.cooldown_turns
                    if expanded
                    else max(0, int(previous.cooldown_remaining) - 1)
                ),
                recent_openings=tuple(openings[-self.voice_window :]),
            )
            self._prune_locked(now)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
                "cooldown_turns": self.cooldown_turns,
                "voice_window": self.voice_window,
                "active_states": len(self._states),
            }

    def _is_continuation(self, turn: "QQTurn", state: _FlowState, *, now: float) -> tuple[bool, str]:
        if now - state.last_at > self.ttl_seconds:
            return False, "topic_window_expired"
        current_sender = self._sender_id(turn)
        sender_matches = not state.last_sender_id or not current_sender or current_sender == state.last_sender_id
        if not sender_matches:
            return False, "different_speaker"
        reply_targets = _reply_targets(turn)
        if reply_targets and set(reply_targets).intersection(state.last_reply_ids):
            return True, "reply_chain_continuation"
        current_text = str(getattr(turn, "combined_text", "") or "").strip()
        overlap = _topic_similarity(state.last_user_text, current_text)
        if overlap >= 0.14:
            return True, f"topic_overlap:{overlap:.2f}"
        # Very short same-speaker follow-ups are often elliptical (for
        # example, a short objection or a request for the missing reason).
        # This is only a soft stage hint; the model still decides whether to
        # expand, so a short new topic cannot force a long answer.
        if len(current_text) <= 18 and now - state.last_at <= 120:
            return True, "short_followup_candidate"
        return False, "topic_changed"

    def _habit_guidance(self, state: _FlowState | None) -> str:
        if state is None:
            return ""
        markers = [item for item in state.recent_openings if item]
        # Only intervene when the same family was actually repeated.  Two
        # different natural openings are variation, not a problem, and should
        # not add a dynamic prompt block or spend extra input tokens.
        if len(markers) < 2 or markers[-1] != markers[-2]:
            return ""
        return (
            "最近几轮已经重复使用同类开场。本轮不要把它当作默认填充；"
            "只有确实需要停顿、确认陪伴或安慰时才保留，优先直接回应眼前内容，"
            "并自然换一种开头。"
        )

    def _prune_locked(self, now: float) -> None:
        expired = [key for key, state in self._states.items() if now - state.last_at > self.ttl_seconds]
        for key in expired:
            self._states.pop(key, None)
        if len(self._states) <= self._max_states:
            return
        ordered = sorted(self._states.items(), key=lambda item: item[1].last_at)
        for key, _ in ordered[: len(self._states) - self._max_states]:
            self._states.pop(key, None)

    @staticmethod
    def _conversation_key(turn: "QQTurn") -> str:
        conversation = str(getattr(turn, "conversation_id", "") or "unknown").strip() or "unknown"
        conversation_type = str(getattr(turn, "conversation_type", "") or "unknown").strip().casefold() or "unknown"
        if conversation_type == "group":
            sender = ChatFlowController._sender_id(turn) or "unknown"
            return f"{conversation_type}:{conversation}:sender:{sender}"
        return f"{conversation_type}:{conversation}"

    @staticmethod
    def _sender_id(turn: "QQTurn") -> str:
        messages = tuple(getattr(turn, "messages", ()) or ())
        for message in reversed(messages):
            sender_id = str(getattr(message, "sender_id", "") or "").strip()
            if sender_id:
                return sender_id
        return ""


def _opening_marker(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    first_line = value.splitlines()[0].strip()
    if _PRESENCE_OPEN_RE.match(first_line):
        return "presence"
    if _ACK_OPEN_RE.match(first_line):
        return "ack"
    if _ADDRESS_OPEN_RE.match(first_line):
        return "address"
    return ""


def is_address_only_message(
    text: str,
    *,
    aliases: Sequence[str] | None = None,
) -> bool:
    """Return whether ``text`` is only a call for attention.

    This is deliberately a small local signal, not a reply selector.  It
    recognizes a name/relationship address and presence fillers, but leaves
    any substantive remainder (a question, task, opinion, or description) to
    the model.  Keeping this decision local avoids another LLM call while the
    final wording remains model-generated.
    """

    raw = str(text or "").strip()
    if len(raw) > 80:
        return False
    value = _normalize_address_text(raw)
    if not value:
        return True

    configured = list(aliases or ())
    if not configured:
        configured.extend(_DEFAULT_ADDRESS_ALIASES)
        extra = os.environ.get("SHINKU_QQ_ADDRESS_ALIASES", "")
        configured.extend(item for item in re.split(r"[,，|]", extra) if item.strip())
    normalized_aliases = sorted(
        {_normalize_address_text(item) for item in configured if _normalize_address_text(item)},
        key=len,
        reverse=True,
    )
    # Strip one or more adjacent address terms, e.g. ``真红真红老婆``.
    changed = True
    while value and changed:
        changed = False
        for alias in normalized_aliases:
            if value.startswith(alias):
                value = value[len(alias) :]
                changed = True
                break
    return not value or value in _ADDRESS_ONLY_FILLERS


def _normalize_address_text(text: str) -> str:
    return _ADDRESS_PUNCTUATION_RE.sub("", str(text or "").casefold())


def _is_address_only_turn(turn: "QQTurn") -> bool:
    # A blank text turn carrying an image is an image request, not a bare
    # address.  The same distinction keeps the dynamic prompt honest for
    # inherited image context.
    if tuple(getattr(turn, "visual_inputs", ()) or ()):
        return False
    return is_address_only_message(getattr(turn, "combined_text", ""))


def _reply_targets(turn: "QQTurn") -> tuple[str, ...]:
    values: list[str] = []
    for message in tuple(getattr(turn, "messages", ()) or ()):
        for segment in tuple(getattr(message, "segments", ()) or ()):
            if str(getattr(segment, "kind", "") or "").casefold() != "reply":
                continue
            payload = getattr(segment, "payload", {})
            payload = payload if isinstance(payload, Mapping) else {}
            data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
            for source in (payload, data):
                for key in ("id", "message_id", "messageId"):
                    value = source.get(key)
                    if value is not None and str(value).strip():
                        values.append(str(value).strip())
    return tuple(dict.fromkeys(values))


def _topic_signature(text: str) -> set[str]:
    signature: set[str] = set()
    for token in _WORD_RE.findall(str(text or "").casefold()):
        if _CJK_RE.search(token):
            if len(token) == 1:
                signature.add(token)
            else:
                signature.update(token[index : index + 2] for index in range(len(token) - 1))
        elif len(token) >= 2:
            signature.add(token)
    return signature


def _topic_similarity(left: str, right: str) -> float:
    a = _topic_signature(left)
    b = _topic_signature(right)
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / max(1, len(a.union(b)))
