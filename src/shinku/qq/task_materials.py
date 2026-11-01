"""Host-selected QQ originals/media, shared by legacy and independent execution."""
from __future__ import annotations

from collections.abc import Mapping
import json
import re

from shinku.agent.loop import AgentRunResult
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.materials import MaterialSource, TaskMaterialSession
from .cards import card_context_complete


_CONTEXT_REFERENCE = re.compile(r"按(?:之前|先前|上次)|(?:之前|先前|上次)(?:那个)?方案|那个方案|不要改那个(?:文件)?", re.I)


def sources_for_turn(turn, task, context):
    """仅选择当前请求者提供的原件；卡片是外部引用，保留读取缺口而不授予权限。"""
    sources, gaps = [], []
    primary = next((m for m in turn.messages if str(getattr(m, "message_id", "")) == turn.primary_message_id), None)
    if primary is None:
        raise AgentBoundaryError("material_primary_message_missing")
    requester = str(getattr(primary, "sender_id", ""))
    if (requester != task["requester"] or task["session"] != f"{turn.conversation_type.casefold()}:{turn.conversation_id}"):
        raise AgentBoundaryError("material_scope_conflict")
    selected_ids = {turn.primary_message_id}
    for message in turn.messages:
        message_id = str(getattr(message, "message_id", ""))
        if str(getattr(message, "sender_id", "")) != requester:
            continue
        if message_id != turn.primary_message_id and str(getattr(message, "text", "")):
            selected_ids.add(message_id)
            sources.append(MaterialSource("message:" + message_id, "related_message", str(message.text),
                provenance={"message_id": message_id, "sender_id": requester, "relation": "same_request_batch"}))
        card = str(getattr(message, "card_context", "") or "")
        if card:
            sources.append(MaterialSource("qq-card:" + message_id, "related_message", card,
                provenance={"message_id": message_id, "sender_id": requester,
                            "relation": "external_card_or_forward"}, complete=card_context_complete(card)))
        quoted = getattr(message, "quoted_message", {})
        quote_ids = tuple(getattr(message, "reply_message_ids", ()) or ())
        quote_id = str(quote_ids[0]) if quote_ids else ""
        if isinstance(quoted, Mapping) and quoted:
            quote_id = str(quoted.get("message_id") or quote_id)
            same = (str(quoted.get("conversation_id") or turn.conversation_id) == turn.conversation_id
                    and str(quoted.get("conversation_type") or turn.conversation_type) == turn.conversation_type)
            if not same:
                gaps.append({"source_id": "quote:" + quote_id, "required": True, "reason": "material_quote_scope_conflict"})
                continue
            text = str(getattr(message, "quoted_text", "") or quoted.get("text") or "")
            if text and quote_id:
                selected_ids.add(quote_id)
                sources.append(MaterialSource("quote:" + quote_id, "related_message", text,
                    provenance={"message_id": quote_id, "sender_id": str(quoted.get("sender_id") or ""), "relation": "quoted"},
                    complete=not quoted.get("card_context") or card_context_complete(str(quoted["card_context"]))))
            elif quote_id and getattr(message, "quoted_attachments", ()):
                selected_ids.add(quote_id)
            elif quote_id:
                gaps.append({"source_id": "quote:" + quote_id, "required": True, "reason": "material_quote_body_unavailable"})
        elif quote_id:
            gaps.append({"source_id": "quote:" + quote_id, "required": True, "reason": "material_quote_body_unavailable"})
    if turn.visual_context_inherited and turn.visual_context_source_message_id:
        selected_ids.add(turn.visual_context_source_message_id)
        selected_ids.update(message_id for message_id, inputs in turn.visual_inputs
                            if any(item.reference_kind == "inherited" for item in inputs))
    for message_id, inputs in turn.visual_inputs:
        if message_id not in selected_ids:
            continue
        for index, item in enumerate(inputs):
            provenance = turn.visual_source(message_id, item)
            source_id = f"image:{provenance['source_message_id'] or message_id}:{index}"
            model_item = item.as_model_item()
            if provenance["source_message_id"] not in selected_ids:
                gaps.append({"source_id": source_id, "required": True, "reason": "material_quote_scope_conflict"})
            elif model_item is None or not provenance["source_message_id"] or not provenance["source_sender_id"]:
                gaps.append({"source_id": source_id, "required": True, "reason": "material_image_unavailable"})
            else:
                sources.append(MaterialSource(source_id, "image", {"data_url": model_item["data_url"]},
                    provenance={"message_id": provenance["source_message_id"], "sender_id": provenance["source_sender_id"],
                                "relation": provenance["reference_kind"]}))
    for message in turn.messages:
        if str(getattr(message, "sender_id", "")) != requester:
            continue
        message_id = str(getattr(message, "message_id", ""))
        for relation, attachments in (("direct", getattr(message, "attachments", ())),
                                      ("quoted", getattr(message, "quoted_attachments", ()))):
            quoted = getattr(message, "quoted_message", {})
            origin_id = str(quoted.get("message_id") or "") if relation == "quoted" else message_id
            origin_sender = str(quoted.get("sender_id") or "") if relation == "quoted" else requester
            if relation == "quoted" and origin_id not in selected_ids:
                continue
            available_images = sum(source.kind == "image" and source.provenance.get("message_id") == origin_id
                                   for source in sources)
            seen_images = 0
            for index, row in enumerate(attachments):
                if not isinstance(row, Mapping):
                    continue
                kind = str(row.get("kind") or "")
                if kind == "image":
                    seen_images += 1
                    if seen_images > available_images:
                        gaps.append({"source_id": f"image:{origin_id}:{index}", "required": True,
                                     "reason": "material_image_unavailable"})
                    continue
                source_id = f"attachment:{message_id}:{relation}:{index}"
                path = str(row.get("local_path") or "")
                if kind == "file" and path:
                    sources.append(MaterialSource(source_id, "attachment", local_path=path,
                        provenance={"message_id": origin_id, "sender_id": origin_sender, "relation": relation}))
                else:
                    gaps.append({"source_id": source_id, "required": True, "reason": "material_attachment_adapter_required"})
    explicit = context.get("agent_material_sources", ())
    if not isinstance(explicit, (tuple, list)) or any(not isinstance(row, MaterialSource) for row in explicit):
        raise AgentBoundaryError("host_material_source_required")
    sources.extend(explicit)
    if (_CONTEXT_REFERENCE.search(str(primary.text)) and not any(source.kind in {"related_message", "clarification"}
            for source in sources) and not any(row["kind"] in {"related_message", "clarification"}
                for row in task.get("material_package", {}).get("items", []))):
        gaps.append({"source_id": "related_context", "required": True, "reason": "material_related_context_missing"})
    return sources, gaps


def bind_turn_materials(bridge, context, *, checkpoint=None):
    store = bridge.material_store
    if store is None:
        return None
    task = context.get("agent_task_state")
    if not isinstance(task, dict):
        raise AgentBoundaryError("host_admitted_task_required")
    if checkpoint is not None:
        active = checkpoint.state.context.get("agent_material_session")
        if not isinstance(active, TaskMaterialSession) or active.store is not store:
            raise AgentBoundaryError("material_checkpoint_conflict")
        context["agent_material_session"] = active
        return active
    sources, gaps = sources_for_turn(context["qq_turn"], task, context)
    old = task.get("material_package", {})
    # Related retained originals remain available on a confirmed continuation;
    # newly missing originals do not become available from a model report.
    task["material_package"] = store.package(task, sources,
        authorization_ref=str(context.get("agent_authority_revision", "")), previous=old, gaps=gaps)
    saved = bridge.task_store.save(task)
    task.clear()
    task.update(saved)
    revision = context.get("agent_authority_revision", bridge.command_state.authority_revision())
    active = TaskMaterialSession(store, task, still_authorized=lambda: bridge.command_state.is_agent_enabled()
        and not bridge.command_state.is_stopped() and bridge.command_state.authority_revision() == revision)
    context["agent_material_session"] = active
    active.checkpoint()
    return active


def prepare_legacy_media(session, runtime, context):
    probe = getattr(runtime, "chat_supports_images", None)
    try:
        supported = probe() if callable(probe) else None
        context["user_images"] = session.media(supported=supported)
        reason = session.unavailable_reason()
        if len(json.dumps(session.prompt_package(), ensure_ascii=False).encode()) > 65536:
            reason = reason or "execution_context_oversized"
    except AgentBoundaryError as exc:
        reason = exc.code
    if reason:
        session.checkpoint()
        return AgentRunResult("waiting_user", 0, reason=reason,
            final_text="必要任务资料尚未完整交接，请补充可读取的原文、图片或附件后继续。",
            completion={"accepted": False, "reason": reason, "missing": [reason]})
    return None
