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

from .models import MemoryDecision

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
  \"memory_type\": \"event|identity|preference|relationship|technical|none\",
  \"scope\": \"conversation|private|group\",
  \"query\": \"适合检索的短查询\",
  \"time_range\": \"若有时间范围就填写，否则为空\",
  \"confidence\": 0到1之间的数字,
  \"reason\": \"一句很短的判断理由\"
}
只有用户确实在询问过去的信息、已确认事实或需要历史上下文时才返回 true。
当前问题、普通闲聊、问候、技术请求本身不需要历史记忆时返回 false。
不要因为消息里出现“你”“之前”以外的普通代词就触发检索。"""

    def __init__(
        self,
        runtime: Any,
        *,
        prompt_cache_key: str = "shinku:memory-router:v1",
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
        r"(?:记得|回忆|想起|之前聊过|上次|以前|哪天|几号|发生了什么|说过什么|提过什么|当时)",
        re.IGNORECASE,
    )
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
    ) -> MemoryDecision:
        text = str(user_message or "").strip()
        if not text:
            return MemoryDecision(False, reason="empty_message")
        if self.deterministic_guards and self._EXPLICIT_FORGET.search(text):
            return MemoryDecision(False, memory_type="none", reason="forget_command")
        if self.deterministic_guards and self._EXPLICIT_RECALL.search(text):
            return MemoryDecision(
                True,
                memory_type="event",
                scope=self._scope_hint(conversation_type),
                query=text[: self.max_query_chars],
                confidence=0.98,
                reason="explicit_recall_request",
            )
        if self.deterministic_guards and self._SHORT_CHAT.fullmatch(text):
            return MemoryDecision(False, reason="obvious_short_chat")
        if self.deterministic_guards and self._CURRENT_TASK.match(text) and len(text) < 80:
            return MemoryDecision(False, reason="current_task_request")
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
        if confidence < self.model_min_confidence:
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
