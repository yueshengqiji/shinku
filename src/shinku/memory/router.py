"""Semantic memory routing without a keyword pile-up.

The router uses a small deterministic guard for explicit commands and obvious
chatter.  Ambiguous turns can be judged by the configured chat model, but the
model only decides *whether and what to retrieve*; it never writes facts or
chooses which record is true.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from .models import HISTORY_MODES, MemoryDecision

__all__ = ["MemoryJudge", "LLMMemoryJudge", "MemoryRouter"]


class MemoryJudge(Protocol):
    def __call__(
        self,
        *,
        user_message: str,
        recent_context: str,
        conversation_type: str,
    ) -> Mapping[str, Any]: ...


class LLMMemoryJudge:
    """Adapter for the existing structured chat runtime."""

    _SYSTEM = """你是记忆路由器，不是聊天助手。
只判断当前消息是否需要查询历史记忆，不回答用户问题，不生成事实。
只返回 JSON：
{
  \"need_memory\": true/false,
  \"history_mode\": \"none|recent|retrieve|summary\",
  \"memory_type\": \"event|identity|preference|relationship|technical|none\",
  \"scope\": \"conversation|private|group\",
  \"query\": \"适合检索的短查询\",
  \"time_range\": \"若有时间范围就填写，否则为空\",
  \"confidence\": 0到1之间的数字,
  \"reason\": \"一句很短的判断理由\"
}
history_mode 的优先级高于旧版 need_memory：
- none：本轮不需要历史；
- recent：当前回复链或连续话题，只带入最近几条；
- retrieve：查询较早的具体事件、人物或事实；
- summary：日期、较长时间段或“群里发生了什么”这类概览。
只有用户确实需要历史上下文时才选择 recent/retrieve/summary。
当前问题、普通闲聊、问候、技术请求本身不需要历史记忆时返回 false。
不要因为消息里出现“你”“之前”以外的普通代词就触发检索。"""

    def __init__(
        self,
        runtime: Any,
        *,
        prompt_cache_key: str = "shinku:memory-router:v2",
        recent_context_chars: int = 1800,
    ) -> None:
        if not callable(getattr(runtime, "call_chat_json", None)):
            raise TypeError("runtime must implement call_chat_json")
        self.runtime = runtime
        self.prompt_cache_key = str(prompt_cache_key or "shinku:memory-router:v1")
        self.recent_context_chars = max(100, min(10000, int(recent_context_chars or 1800)))

    def __call__(
        self,
        *,
        user_message: str,
        recent_context: str,
        conversation_type: str,
    ) -> Mapping[str, Any]:
        prompt = (
            f"会话类型：{conversation_type}\n"
            f"最近对话（仅用于判断承接关系，不是可检索记忆）：\n{recent_context[-self.recent_context_chars:]}\n\n"
            f"当前消息：\n{user_message}"
        )
        result = self.runtime.call_chat_json(
            system_prompt=self._SYSTEM,
            user_prompt=prompt,
            fallback={
                "need_memory": False,
                "history_mode": "none",
                "memory_type": "none",
                "scope": "conversation",
                "query": "",
                "confidence": 0.0,
                "reason": "memory_router_fallback",
            },
            temperature=0.0,
            prompt_cache_key=self.prompt_cache_key,
            history_turns=[],
        )
        return result if isinstance(result, Mapping) else {}


class MemoryRouter:
    """Choose the retrieval path for one turn."""

    _EXPLICIT_RECALL = re.compile(
        r"(?:记得|回忆|想起|之前聊过|上次|以前|哪天|几号|发生了什么|说过什么|提过什么|聊了什么|谈了什么|聊天记录|历史记录|查看(?:一下)?(?:记忆|记录|历史)|回看|回顾|当时)",
        re.IGNORECASE,
    )
    _SUMMARY_INTENT = re.compile(
        r"(?:昨天|前天|大前天|上周|上个月|前几天|最近几天|那天|哪天|几号|日期|时间段|这段时间|发生了什么|聊了什么|谈了什么|聊天记录|历史记录|群里.*(?:情况|发生|总结)|最后怎么定)",
        re.IGNORECASE,
    )
    _DATE_QUERY = re.compile(
        r"(?:(?:20\d{2}\s*[年/-]\s*)?\d{1,2}\s*[月/-]\s*\d{1,2}\s*(?:日|号)?|今天|今日|昨晚|昨夜|昨天|前天|大前天|上周|上个月|前几天)",
        re.IGNORECASE,
    )
    _CONTINUITY = re.compile(r"(?:那个|这件事|他|她|它|刚才|上一条|上面|继续|后来|然后呢|这问题|这个问题)")
    _TOPIC_SWITCH = re.compile(r"^(?:另外|换个话题|不说这个|顺便问|还有一个问题)")
    _EXPLICIT_FORGET = re.compile(r"(?:忘掉|忘记|别记|不要记|清除|删除).{0,24}", re.IGNORECASE)
    _SHORT_CHAT = re.compile(r"^(?:嗯|哦|好|好的|行|可以|谢谢|谢了|哈哈|早|晚安|在吗|？|\?)$")
    _CURRENT_TASK = re.compile(r"^(?:请|帮我|帮忙|能不能|可以|怎么|为什么|解决|修复|测试|检查|开始|继续|看看)")

    def __init__(
        self,
        judge: MemoryJudge | None = None,
        *,
        model_min_confidence: float = 0.55,
        deterministic_guards: bool = True,
        max_query_chars: int = 240,
    ) -> None:
        self.judge = judge
        self.model_min_confidence = max(0.0, min(1.0, float(model_min_confidence)))
        self.deterministic_guards = bool(deterministic_guards)
        self.max_query_chars = max(20, min(2000, int(max_query_chars or 240)))

    def decide(
        self,
        *,
        user_message: str,
        recent_context: str = "",
        conversation_type: str = "private",
        message_meta: Mapping[str, Any] | None = None,
        topic_state: Mapping[str, Any] | None = None,
    ) -> MemoryDecision:
        text = str(user_message or "").strip()
        if not text:
            return MemoryDecision(False, reason="empty_message")
        if self.deterministic_guards and self._EXPLICIT_FORGET.search(text):
            return MemoryDecision(False, memory_type="none", reason="forget_command")
        explicit_recall = bool(self._EXPLICIT_RECALL.search(text))
        date_recall = bool(self._DATE_QUERY.search(text)) and bool(
            re.search(r"(?:聊|谈|说|记录|历史|回忆|记得|发生|内容|哪件|什么)", text)
        )
        if self.deterministic_guards and (explicit_recall or date_recall):
            mode = "summary" if date_recall or self._SUMMARY_INTENT.search(text) else "retrieve"
            signals = ["explicit_history_intent", mode]
            if date_recall:
                signals.append("date_query")
            return MemoryDecision(
                True,
                memory_type="event",
                scope=self._scope_hint(conversation_type),
                query=text[: self.max_query_chars],
                confidence=0.98,
                reason="explicit_summary_request" if mode == "summary" else "explicit_recall_request",
                history_mode=mode,
                local_signals=tuple(signals),
            )
        if self.deterministic_guards and self._SHORT_CHAT.fullmatch(text):
            return MemoryDecision(False, reason="obvious_short_chat")
        if self.deterministic_guards and self._CURRENT_TASK.match(text) and len(text) < 80:
            return MemoryDecision(False, reason="current_task_request")

        meta = dict(message_meta or {})
        recent = bool(str(recent_context or "").strip())
        signals: list[str] = []
        if not self._TOPIC_SWITCH.match(text):
            if str(meta.get("reply_to_message_id") or "").strip():
                signals.append("reply_chain")
            if bool(meta.get("topic_continuation")):
                signals.append("topic_continuation")
            if bool(meta.get("same_topic")):
                signals.append("same_topic")
            if self._CONTINUITY.search(text):
                signals.append("deictic_reference")
        # This branch is entirely local.  It is the cheap path for the normal
        # group continuation case, and does not wake an LLM memory judge.
        if recent and signals and ("reply_chain" in signals or "topic_continuation" in signals or "same_topic" in signals):
            return MemoryDecision(
                False,
                memory_type="event",
                scope=self._scope_hint(conversation_type),
                confidence=0.92,
                reason="local_recent_topic",
                history_mode="recent",
                local_signals=tuple(signals),
            )
        if self.deterministic_guards and self._SUMMARY_INTENT.search(text):
            return MemoryDecision(
                True,
                memory_type="event",
                scope=self._scope_hint(conversation_type),
                query=text[: self.max_query_chars],
                confidence=0.94,
                reason="local_summary_intent",
                history_mode="summary",
                local_signals=("time_or_overview",),
            )

        # No model call for a plainly unrelated message.  Only an unresolved
        # reference plus a recent snapshot is ambiguous enough for the judge.
        ambiguous = recent and bool(self._CONTINUITY.search(text)) and not self._TOPIC_SWITCH.match(text)
        if not ambiguous and self.deterministic_guards:
            return MemoryDecision(False, reason="local_no_history_signal")
        if self.judge is None:
            return MemoryDecision(False, reason="no_memory_judge_configured")

        try:
            raw = self.judge(
                user_message=text,
                recent_context=str(recent_context or ""),
                conversation_type=conversation_type,
            )
        except Exception:
            return MemoryDecision(False, reason="memory_judge_error", used_model=True)
        return self._from_model(raw, text=text, conversation_type=conversation_type)

    def _from_model(self, raw: Mapping[str, Any], *, text: str, conversation_type: str) -> MemoryDecision:
        need = raw.get("need_memory")
        if isinstance(need, str):
            need = need.strip().lower() in {"1", "true", "yes", "on", "是"}
        need = bool(need)
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence_accepted = confidence >= self.model_min_confidence
        if not confidence_accepted:
            need = False
        mode = str(raw.get("history_mode") or "").strip().lower()
        if mode not in HISTORY_MODES:
            mode = "retrieve" if need else "none"
        if not confidence_accepted:
            mode = "none"
        elif mode in {"retrieve", "summary"}:
            need = True
        elif mode == "none":
            need = False
        memory_type = str(raw.get("memory_type") or ("event" if need else "none")).strip().lower()
        if memory_type not in {"event", "identity", "preference", "relationship", "technical", "none"}:
            memory_type = "event" if need else "none"
        scope = str(raw.get("scope") or "conversation").strip().lower()
        if scope not in {"conversation", "private", "group"}:
            scope = "conversation"
        return MemoryDecision(
            need_retrieval=need,
            memory_type=memory_type if need else "none",
            scope=self._scope_hint(conversation_type, requested=scope),
            query=str(raw.get("query") or text)[: self.max_query_chars] if need else "",
            time_range=str(raw.get("time_range") or "")[:80],
            confidence=confidence,
            reason=str(raw.get("reason") or "model_memory_judgement")[:120],
            used_model=True,
            history_mode=mode,
            local_signals=("model_judgement",),
        )

    @staticmethod
    def _scope_hint(conversation_type: str, *, requested: str = "conversation") -> str:
        requested = str(requested or "conversation").lower()
        kind = str(conversation_type or "private").lower()
        if requested == "private" or kind in {"private", "friend", "direct"}:
            return "private"
        if requested == "group" or "group" in kind:
            return "group"
        return "conversation"
