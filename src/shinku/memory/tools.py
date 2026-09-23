"""只读记忆工具。

这个 handler 是 Agent 访问记忆库的唯一小入口：它不修改、不删除记忆，
并且总是从当前 QQTurn 取会话作用域，避免模型把一个群的记录拿到另一个
私聊里。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from .service import MemoryService

__all__ = ["MemorySearchTool"]


class MemorySearchTool:
    """Expose explicit recent/retrieve/summary inspection to the Agent."""

    tool_type = "memory_search"
    _MODES = frozenset({"recent", "retrieve", "summary"})

    def __init__(self, service: MemoryService) -> None:
        if not isinstance(service, MemoryService):
            raise TypeError("memory search tool requires a MemoryService")
        self.service = service

    def tool_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            description=(
                "只读查看当前私聊或群聊的历史记忆。用户明确要求查看记忆，"
                "或询问以前、上次、某天发生了什么时使用；主人本人还可以"
                "查询跨私聊/群聊的主人记忆；不会修改或删除记忆。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查找的主题、人物、日期或事件；没有具体关键词时可省略。",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["recent", "retrieve", "summary"],
                        "description": "recent 看最近记录，retrieve 按主题检索，summary 查看较早日期摘要。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                        "description": "最多返回的记录数，默认 8。",
                    },
                },
                "additionalProperties": False,
            },
        )

    def build_prompt_instruction(self) -> str:
        return (
            "当用户明确让你查看、回忆或核对以前的聊天时调用 memory_search。"
            "主人本人发起时可以跨私聊和群聊查看主人记忆，其他成员只能看当前作用域；"
            "普通闲聊不要调用；这是只读工具，不要把历史记录当成人设或事实。"
        )

    def normalize_call(self, call: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(call, Mapping):
            return None
        query = str(call.get("query") or "").strip()[:240]
        raw_mode = str(call.get("mode") or "").strip().lower()
        mode = raw_mode or ("retrieve" if query else "recent")
        if mode not in self._MODES:
            return None
        raw_limit = call.get("limit", 8)
        if isinstance(raw_limit, bool):
            return None
        try:
            limit = max(1, min(12, int(raw_limit)))
        except (TypeError, ValueError):
            return None
        return {"query": query, "mode": mode, "limit": limit}

    def execute(self, *, call: dict[str, Any], context: Any) -> dict[str, Any]:
        arguments = self.normalize_call(call)
        if arguments is None:
            raise ValueError("memory_search_arguments_invalid")
        turn = context.get("qq_turn") if isinstance(context, Mapping) else getattr(context, "qq_turn", None)
        conversation_id = str(getattr(turn, "conversation_id", "") or "").strip()
        conversation_type = str(getattr(turn, "conversation_type", "private") or "private").strip()
        requester_id = str(getattr(turn, "sender_id", "") or "").strip()
        if not requester_id:
            for message in tuple(getattr(turn, "messages", ()) or ()):
                requester_id = str(getattr(message, "sender_id", "") or "").strip()
                if requester_id:
                    break
        if not conversation_id:
            inspection = {
                "error": "conversation_scope_missing",
                "mode": arguments["mode"],
                "query": arguments["query"],
                "count": 0,
                "records": [],
                "rendered": "当前没有绑定 QQ 会话，无法确定应该查看哪个作用域的记忆。",
            }
            return {
                "tool_type": self.tool_type,
                "followup_context": inspection["rendered"],
                "state_updates": {"memory_inspection": inspection},
            }
        try:
            result = self.service.inspect(
                conversation_id=conversation_id,
                conversation_type=conversation_type,
                requester_id=requester_id,
                query=arguments["query"],
                mode=arguments["mode"],
                limit=arguments["limit"],
            )
        except Exception as exc:
            inspection = {
                "error": "memory_inspection_failed",
                "error_type": type(exc).__name__,
                "mode": arguments["mode"],
                "query": arguments["query"],
                "count": 0,
                "records": [],
                "rendered": "查看记忆时遇到内部错误；不要根据猜测补写历史内容。",
            }
            return {
                "tool_type": self.tool_type,
                "followup_context": inspection["rendered"],
                "state_updates": {"memory_inspection": inspection},
            }
        return {
            "tool_type": self.tool_type,
            "followup_context": result["followup_context"],
            "state_updates": {"memory_inspection": result},
        }
