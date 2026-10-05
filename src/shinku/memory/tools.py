"""只读记忆工具。

这个 handler 是 Agent 访问记忆库的唯一小入口：它不修改、不删除记忆，
并且总是从当前 QQTurn 取会话作用域，避免模型把一个群的记录拿到另一个
私聊里。
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from types import SimpleNamespace
from typing import Any

from .service import MemoryService

__all__ = ["MemorySearchTool"]


class MemorySearchTool:
    """Expose explicit recent/retrieve/summary inspection to the Agent."""

    tool_type = "memory_search"
    agent_progress_class = "observe"
    agent_permission_class = "read"
    agent_fast_perception = True
    agent_history_answer_source = True
    agent_completion_effects = ("history_answer",)
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
                "查询跨私聊/群聊的主人记忆；若检索结果给出 Agent 任务档案 ID，"
                "主人私聊询问真红刚才在群里回复谁时，还可核对真红自己的已发送回复，"
                "不会开放其他群成员原话；没有结果不能当作没有发生；"
                "私聊不能指定任意群或用户读取其原话，scope/group_id/user_id 不是合法参数。"
                "历史问答的任务验收使用history_answer，不是需要网页URL的research_answer或本地文件read。"
                "可传 task_id 精确打开该任务卡片（目标、状态、结果和工具摘要，"
                "不是原始调用日志）；不会修改或删除记忆。"
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
                    "task_id": {
                        "type": "string",
                        "maxLength": 240,
                        "description": "只用于精确打开 memory_search 已返回的主人 Agent 任务档案 ID。",
                    },
                },
                "additionalProperties": False,
            },
        )

    def build_prompt_instruction(self) -> str:
        return (
            "当用户明确让你查看、回忆或核对以前的聊天时调用 memory_search。"
            "主人本人发起时可以跨私聊和群聊查看主人记忆，其他成员只能看当前作用域；"
            "主人在私聊问你刚才在群里和谁说话时，用原问题查询自己的发送证据，"
            "不要把当前私聊没有那条记录理解成群里没有发言；"
            "如果检索到主人 Agent 任务档案 ID，且用户追问该任务细节，再用 task_id 精确打开摘要卡片；"
            "普通闲聊不要调用；这是只读工具，不要把历史记录当成人设或事实。"
        )

    def normalize_call(self, call: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(call, Mapping):
            return None
        if any(key not in {"type", "query", "mode", "limit", "task_id"} and not str(key).startswith("_tool_") for key in call):
            return None
        if "query" in call and not isinstance(call["query"], str):
            return None
        query = str(call.get("query") or "").strip()[:240]
        task_id = str(call.get("task_id") or "").strip()
        if len(task_id) > 240:
            return None
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
        return {"query": query, "mode": mode, "limit": limit, "task_id": task_id}

    def execute(self, *, call: dict[str, Any], context: Any) -> dict[str, Any]:
        arguments = self.normalize_call(call)
        if arguments is None:
            raise ValueError("memory_search_arguments_invalid")
        turn = context.get("qq_turn") if isinstance(context, Mapping) else getattr(context, "qq_turn", None)
        conversation_id = str(getattr(turn, "conversation_id", "") or "").strip()
        conversation_type = str(getattr(turn, "conversation_type", "private") or "private").strip()
        requester_id = str(getattr(turn, "sender_id", "") or "").strip()
        primary_id = str(getattr(turn, "primary_message_id", "") or "")
        if not requester_id:
            messages = tuple(getattr(turn, "messages", ()) or ())
            selected = [m for m in messages if str(getattr(m, "message_id", "")) == primary_id] if primary_id else []
            if not selected and len(messages) == 1:
                selected = list(messages)
            for message in selected:
                requester_id = str(getattr(message, "sender_id", "") or "").strip()
                if requester_id:
                    break
        from shinku.agent.request_scope import RequestScope
        scope = context.get("agent_request_scope") if isinstance(context, Mapping) else None
        user_text = "\n".join(scope.user_messages()) if isinstance(scope, RequestScope) else str(getattr(turn, "combined_text", "") or "")
        if isinstance(scope, RequestScope):
            requester_id = scope.requester
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
                "status": "error", "ok": False,
                "tool_type": self.tool_type,
                "followup_context": inspection["rendered"],
                "state_updates": {"memory_inspection": inspection, "memory_lookup": {"status": "scope_missing", "query_completed": False}},
            }
        if arguments["task_id"]:
            try:
                result = self.service.inspect_agent_task(
                    requester_id=requester_id,
                    task_id=arguments["task_id"],
                )
            except Exception as exc:
                result = {"error": "memory_inspection_failed", "error_type": type(exc).__name__,
                          "rendered": "任务档案读取失败，尚未核对该任务。"}
            # Exact task inspection returns a card, not the search record list.
            # Preserve only the service-authorized records as history evidence.
            result = {**result, "records": [row for key in ("record", "detail_record")
                if isinstance(row := result.get(key), Mapping)], "scope": self.service.OWNER_SCOPE,
                "retrieval_scopes": [self.service.OWNER_SCOPE] if self.service.is_master_user(requester_id) else []}
            return {
                **({"status": "error", "ok": False} if result.get("error") else {}),
                "tool_type": self.tool_type,
                "followup_context": result["rendered"],
                "state_updates": {"memory_inspection": result, "memory_lookup": self._lookup_receipt(result, query=user_text)},
            }
        self_id = ""
        for message in tuple(getattr(turn, "messages", ()) or ()):
            event = getattr(message, "raw_event", {})
            if isinstance(event, Mapping):
                self_id = str(event.get("self_id") or event.get("selfId") or "").strip()
                if self_id:
                    break
        try:
            result = self.service.inspect(
                conversation_id=conversation_id,
                conversation_type=conversation_type,
                requester_id=requester_id,
                query=arguments["query"],
                mode=arguments["mode"],
                limit=arguments["limit"],
                message_id=str(getattr(turn, "primary_message_id", "") or ""),
                user_text=user_text,
                self_id=self_id,
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
                "status": "error", "ok": False,
                "tool_type": self.tool_type,
                "followup_context": inspection["rendered"],
                "state_updates": {"memory_inspection": inspection, "memory_lookup": {"status": "read_failed", "query_completed": False}},
            }
        # Model query terms may narrow/flag a scope, never grant it. Include both
        # texts so short follow-ups cannot hide a requested inaccessible group.
        lookup = self._lookup_receipt(result, query=user_text + "\n" + arguments["query"])
        return {
            "tool_type": self.tool_type,
            "followup_context": result["followup_context"],
            "state_updates": {"memory_inspection": result, "memory_lookup": lookup},
        }

    @staticmethod
    def _lookup_receipt(result: Mapping[str, Any], *, query: str) -> dict[str, Any]:
        scopes = result.get("retrieval_scopes", [result.get("scope", "")])
        scopes = [str(s) for s in scopes if isinstance(s, str)] if isinstance(scopes, (list, tuple)) else []
        matches = re.findall(r"(?<!\d)(\d{5,12})\s*群|群\s*[:：]?\s*(\d{5,12})(?!\d)", query)
        groups = list(dict.fromkeys("group:" + (first or second) for first, second in matches))
        records = result.get("records", [])
        origins = set(scopes)
        for row in records if isinstance(records, list) else []:
            if isinstance(row, Mapping):
                metadata = row.get("metadata", {})
                if isinstance(metadata, Mapping):
                    origins.add(str(metadata.get("owner_source_scope") or metadata.get("owner_origin_scope") or ""))
        supported = all(group in origins for group in groups)
        failed = bool(result.get("error"))
        return {"status": "read_failed" if failed else "scope_limited" if not supported else "candidates" if records else "not_found",
                "query_completed": not failed, "requested_scopes": groups, "retrieval_scopes": scopes,
                "requested_scope_supported": supported, "record_count": len(records) if isinstance(records, list) else 0}
