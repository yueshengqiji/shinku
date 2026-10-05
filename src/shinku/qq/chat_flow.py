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

from .visual_continuity import VISUAL_REACTION_WINDOW_SECONDS, is_short_visual_reaction

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
“嗯”“我在这里”以及类似的陪伴性表达不是固定开场或固定收尾。只有当前内容确实需要确认陪伴时才使用；已经表达过愿意陪伴，就不要换词重复同一个意思。若用户直接确认你是否在线，可以自然回答；否则优先回应他这轮新说的内容。不要机械禁用这些表达，也不要为了变化而刻意换词。
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
_ACK_OPEN_RE = re.compile(
    r"^(?:嗯嗯?|听到了|听见了|听清(?:了|楚了)?|收到了?|知道了|明白了|懂了|在听)"
    r"(?:[…….…。，,、：:;；\s]|$)"
)
_REPAIR_OPEN_RE = re.compile(
    r"^(?:刚才那句|刚才那句话|刚才没接好|我刚才没接好|你再说一遍|再说一遍)"
)
_PRESENCE_OPEN_RE = re.compile(
    r"^(?:(?:嗯嗯?)[…….…。，,、：:;；\s]*)?我(?:就)?在这里(?:[…….…。，,、：:;；\s]|$)"
)
_ADDRESS_OPEN_RE = re.compile(r"^(?:你啊|你还真是)(?:[…….…。，,、：:;；\s]|$)")


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
    voice_anchor: str = ""
    address_only: bool = False
    previous_exchange: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "same_topic": self.same_topic,
            "allow_expansion": self.allow_expansion,
            "reason": self.reason,
            "guidance": self.guidance,
            "habit_guidance": self.habit_guidance,
            "voice_anchor": self.voice_anchor,
            "address_only": self.address_only,
            "previous_exchange": self.previous_exchange,
        }


@dataclass
class _FlowState:
    last_user_text: str = ""
    last_assistant_text: str = ""
    last_sender_id: str = ""
    last_at: float = 0.0
    last_reply_ids: tuple[str, ...] = ()
    last_visual_sources: tuple[tuple[str, str], ...] = ()
    last_visual_attachment: bool = False
    last_expanded: bool = False
    cooldown_remaining: int = 0
    recent_openings: tuple[str, ...] = ()
    recent_companionship_modes: tuple[str, ...] = ()
    recent_affection_turns: tuple[bool, ...] = ()


class ChatFlowController:
    """Track topic continuity without creating a sticky long-reply mode.

    The controller deliberately does not decide what the model should say.
    It retains one short-lived exchange so clear follow-ups can resolve
    omitted references. Receipt-backed private pairs also survive short topic
    changes; that does not force the new topic into follow-up mode.
    The model still makes the semantic decision under the fixed system
    guidance, while ReplyHost only enforces per-message transport-size limits.
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
        voice_anchor = _detect_voice_anchor(turn)
        if not self.enabled:
            return ChatFlowDecision(
                key=key,
                stage="disabled",
                same_topic=False,
                allow_expansion=True,
                reason="chat_flow_disabled",
                guidance="",
                voice_anchor=voice_anchor,
                address_only=address_only,
            )

        now = float(self._clock())
        with self._lock:
            self._prune_locked(now)
            state = self._states.get(key)
            if state is None or now - state.last_at > self.ttl_seconds:
                return ChatFlowDecision(
                    key=key,
                    stage="fresh",
                    same_topic=False,
                    allow_expansion=False,
                    reason="new_topic_or_expired",
                    guidance="本轮是当前话题的首次回应：先解决眼前内容，避免无关扩展。",
                    habit_guidance="",
                    voice_anchor=voice_anchor,
                    address_only=address_only,
                )

            current_text = str(getattr(turn, "combined_text", "") or "")
            # One earlier affectionate turn is not yet a habit. Treat the
            # feeling as familiar after it has appeared at least twice in the
            # recent window; the current turn then makes it a repeated pattern.
            frequent_affection = _is_affection_turn(current_text) and (
                sum(state.recent_affection_turns[-self.voice_window :]) >= 2
            )
            voice_anchor = _detect_voice_anchor(
                turn,
                familiar_affection=frequent_affection,
            )
            same_topic, reason = self._is_continuation(turn, state, now=now)
            short_private = (
                str(getattr(turn, "conversation_type", "")).lower() == "private"
                and bool(self._sender_id(turn)) and self._sender_id(turn) == state.last_sender_id
                and 0 <= now - state.last_at <= 120 and bool(state.last_reply_ids)
                and reason not in {"different_visual_source", "different_speaker"}
            )
            previous_exchange = self._render_previous_exchange(
                state, current_text, reason, allow_short_private=short_private,
            ) if same_topic or short_private else ""
            if not same_topic:
                return ChatFlowDecision(
                    key=key,
                    stage="fresh",
                    same_topic=False,
                    allow_expansion=False,
                    reason=reason,
                    guidance="话题看起来已经切换：直接处理当前内容，不要把上一话题的长度或情绪带过来。",
                    # Style habits belong to the conversation, not only to one
                    # semantic topic. In particular, repeated generic
                    # acknowledgements can sound like delivery/attention bugs.
                    habit_guidance=self._opening_repeat_guidance(state),
                    voice_anchor=voice_anchor,
                    address_only=address_only,
                    previous_exchange=previous_exchange,
                )

            habit_guidance = self._habit_guidance(
                state,
                current_text=str(getattr(turn, "combined_text", "") or ""),
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
                    voice_anchor=voice_anchor,
                    address_only=address_only,
                    previous_exchange=previous_exchange,
                )
            return ChatFlowDecision(
                key=key,
                stage="followup",
                same_topic=True,
                allow_expansion=True,
                reason=reason,
                guidance="本轮延续了同一话题：先判断是否有新的未解决点，有才自然展开，没有就简洁承接。",
                habit_guidance=habit_guidance,
                voice_anchor=voice_anchor,
                address_only=address_only,
                previous_exchange=previous_exchange,
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
            continuing = bool(
                (previous.last_user_text or previous.last_visual_attachment)
                and previous.last_assistant_text
                and self._is_continuation(turn, previous, now=now)[0]
            )
            # Keep the tiny opening history across topic switches. Topic and
            # companionship guidance remain scoped to continuity below, while
            # generic acknowledgements are a conversation-wide voice habit.
            openings = list(previous.recent_openings)
            openings.append(opening)
            companionship_modes = (
                list(previous.recent_companionship_modes) if continuing else []
            )
            current_modes = _companionship_modes(final_text)
            companionship_modes.append(",".join(current_modes))
            affection_turns = list(previous.recent_affection_turns)
            affection_turns.append(
                _is_affection_turn(str(getattr(turn, "combined_text", "") or ""))
            )
            self._states[key] = _FlowState(
                last_user_text=str(getattr(turn, "combined_text", "") or "").strip(),
                last_assistant_text=(str(final_text or "").strip()[:1200] + "（节选）"
                    if len(str(final_text or "").strip()) > 1200 else str(final_text or "").strip()),
                last_sender_id=sender_id,
                last_at=now,
                last_reply_ids=tuple(str(item).strip() for item in reply_message_ids if str(item).strip()),
                last_visual_sources=_visual_sources(turn),
                last_visual_attachment=any(inputs for _, inputs in getattr(turn, "visual_inputs", ())),
                last_expanded=bool(expanded),
                cooldown_remaining=(self.cooldown_turns if expanded else (
                    max(0, int(previous.cooldown_remaining) - 1) if continuing else 0
                )),
                recent_openings=tuple(openings[-self.voice_window :]),
                recent_companionship_modes=tuple(
                    companionship_modes[-self.voice_window :]
                ),
                recent_affection_turns=tuple(affection_turns[-self.voice_window :]),
            )
            self._prune_locked(now)

    @staticmethod
    def _render_previous_exchange(
        state: _FlowState, current_text: str, reason: str, *, allow_short_private: bool = False,
    ) -> str:
        current = str(current_text or "").strip()
        has_topic_or_reply_link = reason in {
            "reply_chain_continuation", "visual_source_continuation", "visual_reaction_continuation",
        } or reason.startswith("topic_overlap:")
        if not has_topic_or_reply_link and not allow_short_private:
            current_signature = _topic_signature(current.replace("没有", "没"))
            answer_signature = _topic_signature(str(state.last_assistant_text or "").replace("没有", "没"))
            shared_bigrams = {
                token
                for token in current_signature.intersection(answer_signature)
                if len(token) == 2 and _CJK_RE.search(token)
            }
            normalized = _normalize_address_text(current)
            starts_with_reference = normalized.startswith(
                ("那", "这", "它", "他", "她", "刚才", "前面", "上面", "上句", "上条", "上一")
            )
            asks_about_previous = starts_with_reference and any(
                marker in normalized for marker in ("为什么", "怎么", "哪里", "哪儿", "什么", "吗", "呢")
            )
            asks_why = len(normalized) <= 18 and normalized.startswith(("为什么", "怎么会", "怎么就", "怎么不"))
            if len(shared_bigrams) < 2 and not asks_about_previous and not asks_why:
                return ""
        user_text = str(state.last_user_text or "").strip()
        user_text = user_text[:600] + "（节选）" if len(user_text) > 600 else user_text
        if not user_text and state.last_visual_attachment:
            user_text = "[图片附件；该轮没有文字正文]"
        assistant_text = str(state.last_assistant_text or "").strip()
        if not user_text or not assistant_text:
            return ""
        return f"上一轮用户：{user_text}\n上一轮真红：{assistant_text}"

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
        current_visual_sources = _visual_sources(turn)
        if state.last_visual_sources and current_visual_sources:
            if set(state.last_visual_sources).intersection(current_visual_sources):
                return True, "visual_source_continuation"
            return False, "different_visual_source"
        reply_targets = _reply_targets(turn)
        if reply_targets and set(reply_targets).intersection(state.last_reply_ids):
            return True, "reply_chain_continuation"
        if (state.last_visual_sources and current_sender and current_sender == state.last_sender_id
            and 0 <= now - state.last_at <= VISUAL_REACTION_WINDOW_SECONDS
            and is_short_visual_reaction(_primary_text(turn))):
            return True, "visual_reaction_continuation"
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

    def _habit_guidance(self, state: _FlowState | None, *, current_text: str = "") -> str:
        if state is None:
            return ""
        guidance: list[str] = []
        opening_guidance = self._opening_repeat_guidance(state)
        if opening_guidance:
            guidance.append(opening_guidance)

        recent_modes = {
            mode
            for item in state.recent_companionship_modes[-self.voice_window :]
            for mode in item.split(",")
            if mode
        }
        current = str(current_text or "").strip()
        directly_asks_for_presence = any(
            term in current for term in ("在吗", "在不在", "你在不在", "还在吗", "你在吗")
        )
        explicitly_requests_company = any(
            term in current for term in ("陪我", "陪着我", "陪我一会", "陪我睡", "留下来")
        )
        explicitly_asks_physical_limit = any(
            term in current for term in ("碰不到", "为什么不能", "能不能过来", "能来吗", "到不了", "现实中")
        )
        if "presence" in recent_modes and not (directly_asks_for_presence or explicitly_requests_company):
            guidance.append(
                "前几轮已经明确表达过陪伴。本轮若只是延续同一话题，不要再补“我在这里/陪着你”式的同义承诺，先回应新内容；"
                "对方重新明确请求陪伴时，仍可简短接住，不必解释距离或存在方式。"
            )
        if "physical_limit" in recent_modes and not explicitly_asks_physical_limit:
            guidance.append(
                "前几轮已经说明过现实动作或距离限制；本轮没有直接追问这项能力时，不要再重复限制说明，回到对方真正想表达的事。"
            )
        return "\n".join(guidance)

    @staticmethod
    def _opening_repeat_guidance(state: _FlowState | None) -> str:
        if state is None:
            return ""
        markers = state.recent_openings
        if not markers or not markers[-1]:
            return ""
        # These two families were showing up as false recovery messages. One
        # occurrence is enough to nudge the next turn; keep an exception for a
        # real acknowledgement/clarification request instead of banning them.
        if markers[-1] in {"ack", "repair"}:
            return (
                "上一轮已经用了“听到了/再说一遍”式确认。本轮除非对方确实在确认有没有听清、"
                "或当前消息本身缺失/无法理解，不要再次声称没接好或让对方重说；直接回应这条新消息。"
            )
        # For less intrusive, characterful openings, intervene only after the
        # same family has actually occurred twice in a row.
        if len(markers) >= 2 and markers[-1] == markers[-2]:
            return (
                "最近几轮已经重复使用同类开场。本轮不要把它当作默认填充；"
                "只有确实需要停顿、确认陪伴或安慰时才保留，优先直接回应眼前内容，"
                "并自然换一种开头。"
            )
        return ""

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
        primary_id = str(getattr(turn, "primary_message_id", "") or "")
        for message in messages:
            if str(getattr(message, "message_id", "") or "") == primary_id:
                return str(getattr(message, "sender_id", "") or "").strip()
        for message in reversed(messages):
            sender_id = str(getattr(message, "sender_id", "") or "").strip()
            if sender_id:
                return sender_id
        return ""


def _primary_text(turn: "QQTurn") -> str:
    primary_id = str(getattr(turn, "primary_message_id", "") or "")
    for message in getattr(turn, "messages", ()):
        if str(getattr(message, "message_id", "") or "") == primary_id:
            return str(getattr(message, "text_without_at_mentions", getattr(message, "text", "")) or "").strip()
    return str(getattr(turn, "combined_text", "") or "").strip()


def _visual_sources(turn: "QQTurn") -> tuple[tuple[str, str], ...]:
    sources = []
    for message_id, inputs in getattr(turn, "visual_inputs", ()):
        for item in inputs:
            source = turn.visual_source(message_id, item)
            source_id, sender_id = source["source_message_id"], source["source_sender_id"]
            # Unresolved quote authors remain unknown; never substitute the
            # current requester as an implicit image source.
            if source_id and sender_id and (source_id, sender_id) not in sources:
                sources.append((source_id, sender_id))
    return tuple(sources[:5])


_STATUS_UPDATE_TERMS = (
    "回来了", "我回来了", "刚忙完", "忙完了", "忙好了", "今天好累", "好累啊",
    "累死了", "终于下班", "刚下班", "到家了", "刚到家", "终于搞定", "搞定收工",
    "收工了", "下班啦", "打完卡",
)
_DEBATE_OPINION_TERMS = (
    "真要辩", "真要论", "胡说", "不对吧", "才不是", "凭什么", "怎么可能",
    "明明是", "漏洞在", "强词夺理", "说不通", "不讲理", "瞎说", "根本不是",
)
_PRAISE_AFFECTION_TERMS = (
    "可爱", "真可爱", "太可爱", "真棒", "夸你", "喜欢你", "最喜欢真红", "喜欢真红",
    "亲亲", "mua", "抱抱", "贴贴", "摸摸头", "真好看", "迷人", "爱你", "想你",
)
_AFFECTION_ADDRESS_TERMS = ("老婆", "亲爱的", "宝贝", "宝宝")
_AFFECTION_HISTORY_TERMS = (
    "老婆", "亲爱的", "宝贝", "宝宝", "喜欢你", "喜欢真红", "最喜欢真红",
    "爱你", "想你", "亲亲", "mua", "抱抱", "贴贴", "摸摸头",
)


def _is_affection_turn(text: str) -> bool:
    value = str(text or "").strip().lower()
    if any(term in value for term in _AFFECTION_HISTORY_TERMS):
        return True
    return bool(re.search(r"(?:你|真红).{0,5}(?:可爱|真棒|真好看|迷人)", value))


def _detect_voice_anchor(turn: "QQTurn", *, familiar_affection: bool = False) -> str:
    text = str(getattr(turn, "combined_text", "") or "").strip().lower()
    if not text:
        return ""
    if any(term in text for term in _STATUS_UPDATE_TERMS):
        return "对方在报近况或分享状态：两拍接住（接住对方 + 说自己当下在做什么），不催人睡觉、不主动叮嘱休息、不当照护员。"
    if any(term in text for term in _DEBATE_OPINION_TERMS):
        return "当前处于观点争论：亮明立场并给一条最关键的支撑理由，接住对方点名的那一句；严禁把对方的话拆解成前半句/后半段逐条反驳，禁止破折号插入语，禁止以‘说吧/真要辩’挑战式反问收尾。"
    if any(term in text for term in _AFFECTION_ADDRESS_TERMS):
        if familiar_affection:
            return "对方近期频繁用亲密称呼，这是你们熟悉的相处方式。自然接住称呼并优先回应后面的实际内容，不要每次都表现得意外或害羞，也不要复用固定反问。"
        return "亲密称呼本身是熟悉的称谓，不等于突然表白；先自然回应后面的实际内容。只有消息本身另有新鲜、真挚的夸奖或表白时，才轻轻流露半拍害羞。"
    if any(term in text for term in _PRAISE_AFFECTION_TERMS):
        if familiar_affection:
            return "对方近期已经多次表达亲近或赞美。把这份心意当作熟悉的日常，不必每次都害羞或惊讶；直接接住内容，偶尔自然轻快、从容或轻微吐槽即可。"
        return "对方这次表达了亲近或赞美：先诚实接住心意。若确实新鲜或出乎意料，可以有一瞬轻微难为情；不要套用固定的惊讶反问。"
    return ""


def _opening_marker(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    first_line = value.splitlines()[0].strip()
    first_line = re.sub(r"^[\s…\.。·,，:：;；、—-]+", "", first_line)
    if _PRESENCE_OPEN_RE.match(first_line):
        return "presence"
    if _ACK_OPEN_RE.match(first_line):
        return "ack"
    if _REPAIR_OPEN_RE.match(first_line):
        return "repair"
    if _ADDRESS_OPEN_RE.match(first_line):
        return "address"
    return ""


def _companionship_modes(text: str) -> tuple[str, ...]:
    """Remember recent comfort phrasing so a nearby follow-up can avoid echoes."""

    value = str(text or "").strip().casefold()
    if not value:
        return ()
    modes: list[str] = []
    presence_phrases = (
        "我在这里陪着你",
        "我在这里陪你",
        "我在这儿陪着你",
        "我会陪着你",
        "我陪你到睡着",
        "我陪你说说话",
        "我本来就在这里",
        "我在这里",
        "我在这儿",
        "我一直都在",
        "陪在你身边",
    )
    physical_limit_phrases = (
        "碰不到你",
        "碰不着你",
        "没法碰到你",
        "不能碰到你",
        "没办法过去",
        "没法过去",
        "过不去",
        "不能亲自到场",
        "无法亲自到场",
        "距离或媒介",
        "现实动作的限制",
        "只能像这样陪你",
    )
    if any(phrase in value for phrase in presence_phrases):
        modes.append("presence")
    if any(phrase in value for phrase in physical_limit_phrases):
        modes.append("physical_limit")
    return tuple(modes)


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
