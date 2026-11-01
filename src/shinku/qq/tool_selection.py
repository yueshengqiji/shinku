"""Compact capability catalog and exact registered-tool selection."""

from __future__ import annotations

import json
import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from shinku.agent.loop import AgentDecision
from shinku.tools.invocation import ToolInvocation

COMPACT_TOOL_DISPATCH_NAME = "__shinku_tool_dispatch__"


def native_repair_tool_specs(tool_specs):
    """Offer real schemas on repair; route hints are host metadata, not tool args."""
    specs = copy.deepcopy(tool_specs)
    for spec in specs:
        schema = spec["function"]["parameters"]
        schema.setdefault("properties", {}).update({
            "_shinku_request_kind": {"type": "string", "enum": ["query", "public_query", "task", "continue"],
                "description": "完整请求意图，不仅是当前第一步；写入/研究/多步行动用task，简单公开查证用public_query，本地或历史只读问答用query，续接用continue。"},
            "_shinku_task_id": {"type": "string", "description": "continue时指定当前会话中的候选任务ID。"},
        })
        schema["required"] = [*schema.get("required", []), "_shinku_request_kind"]
    return specs


def resolve_native_repair_intent(decision, available_tool_names):
    """Strip only advertised route metadata, retaining all execution safeguards."""
    if decision.kind != "tool":
        return decision
    calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
    if len(calls) == 1 and calls[0].name == COMPACT_TOOL_DISPATCH_NAME:
        return decision  # Compatibility envelope still goes through its own validator.
    if not calls or any(call.name not in available_tool_names for call in calls):
        return AgentDecision.final("工具恢复返回了未登记的工具名，尚未执行操作。", outcome="failed")
    routes = [(call.arguments.get("_shinku_request_kind"), call.arguments.get("_shinku_task_id", "")) for call in calls]
    if all(kind is None and not task_id for kind, task_id in routes) and decision.task_link is not None:
        return decision  # Existing explicit JSON routes retain host validation.
    kind, task_id = routes[0]
    if (kind not in ("query", "public_query", "task", "continue")
            or not isinstance(task_id, str) or any(route != routes[0] for route in routes)):
        return AgentDecision.final("工具恢复缺少有效的完整请求意图，尚未执行操作。", outcome="failed")
    cleaned = tuple(replace(call, arguments={k: v for k, v in call.arguments.items()
                    if k not in {"_shinku_request_kind", "_shinku_task_id"}}) for call in calls)
    link = {"mode": "continue", "task_id": task_id} if kind == "continue" else {"mode": "auto", "intent": kind}
    if decision.task_link is not None and (not isinstance(decision.task_link, Mapping) or dict(decision.task_link) != link):
        return AgentDecision.final("工具恢复的任务分流字段冲突，尚未执行操作。", outcome="failed")
    return replace(decision, invocation=cleaned[0], invocations=cleaned, task_link=link)


def dispatch_diagnostics(decision, available_tool_names):
    """Describe a rejected envelope without persisting arguments or unknown names."""
    calls = (decision.invocations or ((decision.invocation,) if decision.invocation else ())) if decision else ()
    if not calls:
        return {"status": "rejected", "reason": "unparsed_tool_call", "count": 0}
    names = []
    reason = "invalid_native_envelope"
    for call in calls:
        if call.name == COMPACT_TOOL_DISPATCH_NAME:
            args = call.arguments
            name = args.get("tool_name") or args.get("target_tool") or args.get("name")
            if not isinstance(name, str) or name not in available_tool_names:
                reason = "unknown_target_tool"
            else:
                names.append(name)
                reason = "invalid_selector_arguments_or_intent"
        elif call.name in available_tool_names:
            names.append(call.name)
        else:
            reason = "unknown_native_tool"
    return {"status": "rejected", "reason": reason, "selected_tools": names, "count": len(calls)}


def compact_tool_dispatch_spec(
    tool_specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Build a compact model-selected dispatcher and capability index.

    The model remains the tool selector. Full native schemas are retained for
    AgentLoop follow-up calls and for a one-time repair when the compact
    selector returns an invalid name or argument envelope. Required structured
    parameters also retain their full schema in the first-turn catalog.
    """

    names: list[str] = []
    catalog_lines = [
        "【可用工具能力目录】",
        "按当前任务的真实需要自主判断是否选工具；目录只描述能力，不代表授权。普通聊天直接回答。一次先选一个工具；高风险操作仍走现有授权。",
    ]
    for item in tool_specs:
        function = item.get("function") if isinstance(item.get("function"), Mapping) else {}
        name = str(function.get("name") or "").strip()
        if not name or name == COMPACT_TOOL_DISPATCH_NAME:
            continue
        names.append(name)
        description = " ".join(str(function.get("description") or "").split())[:150]
        parameters = function.get("parameters") if isinstance(function.get("parameters"), Mapping) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
        raw_required = parameters.get("required", ())
        required = {
            str(value).strip()
            for value in raw_required
            if str(value).strip()
        } if isinstance(raw_required, (list, tuple)) else set()
        fields: list[str] = []
        for index, (key, raw_schema) in enumerate(properties.items()):
            # The optional-field preview limit must not hide required inputs.
            if index >= 12 and str(key) not in required:
                continue
            schema = raw_schema if isinstance(raw_schema, Mapping) else {}
            if str(key) in required and (
                schema.get("type") in ("object", "array")
                or any(part in schema for part in (
                    "properties", "items", "prefixItems", "allOf", "anyOf", "oneOf", "$ref",
                ))
            ):
                # A flat type label cannot describe nested required fields,
                # bounds, or cross-field meaning (for example saved dates).
                # Serialize the canonical subtree without truncating descriptions
                # or inventing defaults. Existing offer-size accounting sees it.
                fields.append(f"{key} (必填，完整参数定义):" + json.dumps(
                    dict(schema), ensure_ascii=False, separators=(",", ":")))
                continue
            field_type = str(schema.get("type") or "any")
            field = f"{key}:{field_type}{'!' if str(key) in required else ''}"
            enum = schema.get("enum")
            if isinstance(enum, (list, tuple)) and enum:
                field += "=" + "/".join(str(value)[:24] for value in enum[:5])
            detail = " ".join(str(schema.get("description") or "").split())[:52]
            if detail:
                field += f"({detail})"
            fields.append(field)
        extra = max(0, len(properties) - len(fields))
        signature = ", ".join(fields)
        if extra:
            signature += f", …另有{extra}项"
        line = f"- {name}: {description}"
        if signature:
            line += f" | 参数 {signature}"
        catalog_lines.append(line)

    router_spec = {
        "type": "function",
        "function": {
            "name": COMPACT_TOOL_DISPATCH_NAME,
            "description": "仅在确实需要工具时选择目录中的一个工具，并提供该工具的参数；不执行工具、不代表用户授权。",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_kind": {"type": "string", "enum": ["query", "public_query", "task", "continue"],
                        "description": "完整目标：系统/历史只读问答=query；公开事实查证=public_query；写入、研究或多步行动=task；明确承接旧目标=continue。宿主负责建任务与入口选择；省略时按正式任务处理。"},
                    "task_id": {"type": "string", "description": "续接目标ID；仅一个候选时可省略，多项候选须指明。"},
                    "tool_name": {
                        "type": "string",
                        "enum": names,
                        "description": "要调用的目标工具名称",
                    },
                    "arguments": {
                        "type": "string",
                        "description": "以 JSON 字符串形式提供的参数字典（例如 '{\"install_missing\": true}'）；若无需参数传 '{}'。",
                    },
                },
                "required": ["tool_name", "arguments"],
            },
        },
    }
    return router_spec, "\n".join(catalog_lines)


def resolve_compact_tool_dispatch(
    decision: AgentDecision,
    *,
    available_tool_names: set[str],
) -> tuple[AgentDecision, bool]:
    """Convert the model's compact selector call to its exact registered call."""

    calls = decision.invocations or ((decision.invocation,) if decision.invocation is not None else ())
    if len(calls) != 1 or calls[0].name != COMPACT_TOOL_DISPATCH_NAME:
        return decision, False
    selector = calls[0]
    tool_name = str(
        selector.arguments.get("tool_name")
        or selector.arguments.get("target_tool")
        or selector.arguments.get("name")
        or ""
    ).strip()
    arguments = selector.arguments.get("arguments")
    if arguments is None and "parameters" in selector.arguments:
        arguments = selector.arguments.get("parameters")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            try:
                import json_repair

                arguments = json_repair.repair_json(arguments, return_objects=True)
            except Exception:
                return decision, False
    if tool_name not in available_tool_names or not isinstance(arguments, Mapping):
        return decision, False
    metadata = {}
    if "request_kind" in selector.arguments:
        kind = selector.arguments["request_kind"]
        if kind not in ("query", "public_query", "task", "continue"):
            return decision, False
        metadata["task_link"] = ({"mode": "continue", "task_id": selector.arguments.get("task_id", "")}
            if kind == "continue" else {"mode": "auto", "intent": kind})
    return AgentDecision.tool(
        ToolInvocation(
            name=tool_name,
            arguments=dict(arguments),
            source=selector.source,
            id=selector.id,
        )
    ).with_task_metadata(metadata).with_task_metadata({
        key: selector.arguments[key] for key in ("task_contract", "task_link", "task_notes")
        if key in selector.arguments
    }).with_task_metadata({
        key: getattr(decision, key) for key in ("task_contract", "task_link", "task_notes")
        if getattr(decision, key) is not None
    }), True
