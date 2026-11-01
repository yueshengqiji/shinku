"""Adapt normalized QQ turns into attributed context and memory metadata.

The output preserves source identity. No raw event is promoted to a user request.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shinku.character import SHINKU_CHARACTER
from shinku.agent.loop import AgentState

from .mentions import group_mention_targets
from .message import IncomingMessage
from .routing import ReplyRoutePolicy
from .turns import QQTurn


@dataclass(frozen=True)
class _GroupSnapshot:
    context: Mapping[str, Any]
    items: list[Any]
    current: Mapping[str, Any]
    focus: Mapping[str, Any]


def group_context(turn: QQTurn, *, bot_ids: Sequence[str] = (), character_name=SHINKU_CHARACTER.name) -> str:
    if str(turn.conversation_type or "").casefold() != "group":
        return ""
    mention_lines = _group_mention_lines(turn, bot_ids=bot_ids)
    snapshot = _group_snapshot(turn)
    if snapshot is None:
        return "\n".join(mention_lines)
    relations = _current_message_relations(snapshot.current, character_name=character_name)
    relations.extend(_group_focus_relations(snapshot.focus))
    rendered = _render_group_history(snapshot.items)
    context = snapshot.context
    reason = str(context.get("reason") or "local_topic").strip()
    sections = [
        "【群聊当前话题快照，仅作为上下文证据】",
        f"触发原因：{reason}",
        "【当前消息归属】\n" + "\n".join(relations),
    ]
    if rendered:
        sections.append("【关联群聊记录，按时间顺序】\n" + "\n".join(rendered[-6:]))
    if mention_lines:
        sections.append("\n".join(mention_lines))
    return "\n".join(sections)


def group_current_context(turn: QQTurn, *, character_name=SHINKU_CHARACTER.name) -> str:
    """Keep current addressing while projection owns historical evidence."""
    if turn.conversation_type != "group":
        return ""
    snapshot = _group_snapshot(turn)
    if snapshot is None:
        return ""
    # Parent message bodies are read from the canonical sources by projection.
    current = dict(snapshot.current)
    parents = current.get("reply_parents", [])
    if isinstance(parents, (list,tuple)):
        current["reply_parents"] = [{k:v for k,v in row.items() if k != "text"}
                                    for row in parents if isinstance(row, Mapping)]
    relations = _current_message_relations(current, character_name=character_name)
    relations.extend(_group_focus_relations(snapshot.focus))
    return "【当前消息归属】\n" + "\n".join(relations)


def _group_mention_lines(turn: QQTurn, *, bot_ids: Sequence[str]) -> list[str]:
    mention_targets = group_mention_targets(turn, bot_ids=bot_ids)
    mention_lines = []
    if mention_targets:
        mention_lines = [
            "【本轮可用的群成员提及】",
            "需要在群聊中直接点名某位已出现的成员时，才在回复中使用对应内部标记；不要编造标记、不要 @全体，也不要仅因上下文提到某人就提及对方。普通的 @名字 文本不会产生真正提及。",
            "每条回复最多真正提及两位成员："
            + "；".join(
                f"[[SHINKU_AT_{index}]] = @{label}"
                for index, (_, label) in enumerate(mention_targets, start=1)
            ),
        ]
    return mention_lines


def _group_snapshot(turn: QQTurn) -> _GroupSnapshot | None:
    primary = next(
        (item for item in turn.messages if str(getattr(item, "message_id", "") or "") == turn.primary_message_id),
        None,
    )
    ordered = ((primary,) if primary is not None else ()) + tuple(
        item for item in reversed(tuple(turn.messages or ())) if item is not primary
    )
    for message in ordered:
        raw = getattr(message, "raw_event", {})
        if not isinstance(raw, Mapping):
            continue
        context = raw.get("_shinku_group_context")
        if not isinstance(context, Mapping):
            continue
        items = context.get("items")
        if not isinstance(items, list) or not items:
            items = []
        current = context.get("current")
        current = current if isinstance(current, Mapping) else {}
        focus = context.get("focus")
        focus = focus if isinstance(focus, Mapping) else {}
        if not items and not current and not focus:
            continue
        return _GroupSnapshot(context, items, current, focus)
    return None


def _current_message_relations(current: Mapping[str, Any], *, character_name=SHINKU_CHARACTER.name) -> list[str]:
    sender_label = str(current.get("sender_label") or current.get("sender_id") or "群成员").strip()
    sender_id = str(current.get("sender_id") or "").strip()
    current_sender = f"{sender_label}（QQ {sender_id}）" if sender_id else sender_label
    at_labels = current.get("at_target_labels")
    at_ids = current.get("at_target_ids")
    at_labels = at_labels if isinstance(at_labels, (list, tuple)) else ()
    at_ids = at_ids if isinstance(at_ids, (list, tuple)) else ()
    current_targets = [
        f"@{str(label).strip()}（QQ {str(at_ids[index]).strip()}）"
        if index < len(at_ids) and str(at_ids[index]).strip()
        else f"@{str(label).strip()}"
        for index, label in enumerate(at_labels)
        if str(label).strip()
    ]
    relations = [f"当前发言者：{current_sender}"]
    relations.append("当前 @ 对象：" + ("、".join(current_targets) if current_targets else "无"))
    reply_ids = current.get("reply_message_ids")
    reply_ids = reply_ids if isinstance(reply_ids, (list, tuple)) else ()
    parents = current.get("reply_parents")
    parents = parents if isinstance(parents, (list, tuple)) else ()
    if parents:
        for parent in parents:
            if not isinstance(parent, Mapping):
                continue
            parent_label = str(parent.get("sender_label") or parent.get("sender_id") or "群成员").strip()
            parent_id = str(parent.get("sender_id") or "").strip()
            parent_sender = f"{parent_label}（QQ {parent_id}）" if parent_id else parent_label
            parent_targets = parent.get("at_target_labels")
            parent_targets = parent_targets if isinstance(parent_targets, (list, tuple)) else ()
            parent_target_text = "、".join(f"@{str(value).strip()}" for value in parent_targets if str(value).strip())
            parent_text = str(parent.get("text") or "").strip()
            relation = f"当前消息回复对象：{parent_sender}"
            if parent_target_text:
                relation += f"；对方原消息明确 @ {parent_target_text}"
            if parent_text:
                relation += f"；原话：{parent_text}"
            relations.append(relation)
    elif reply_ids:
        relations.append(
            "当前消息带有引用回复，但目标原文未能解析（消息 ID："
            + "、".join(str(value) for value in reply_ids[:4])
            + "）；不要猜测被回复者。"
        )
    if bool(current.get("autonomous_candidate")):
        relations.append(f"这是自主回复候选，不是明确发给{character_name}的消息。")
        relations.append(f"宿主确认与{character_name}的互动关联：" + (
            "已确认引用或先前明确对话；仍须核对本句具体话题。"
            if current.get("bot_exchange_confirmed") is True else
            f"未确认；{character_name}先前主动插话不证明对方正在跟她续聊。"))
    return relations


def _group_focus_relations(focus: Mapping[str, Any]) -> list[str]:
    root_sender = str(focus.get("root_sender_label") or "").strip()
    root_sender_id = str(focus.get("root_sender_id") or "").strip()
    root_targets = focus.get("root_target_labels")
    root_targets = root_targets if isinstance(root_targets, (list, tuple)) else ()
    root_target_ids = focus.get("root_target_ids")
    root_target_ids = root_target_ids if isinstance(root_target_ids, (list, tuple)) else ()
    participants = focus.get("participant_labels")
    participants = participants if isinstance(participants, (list, tuple)) else ()
    if root_sender or participants:
        root_sender_text = (
            f"{root_sender}（QQ {root_sender_id}）"
            if root_sender and root_sender_id
            else root_sender
        )
        root_target_text = "、".join(
            f"{str(value).strip()}（QQ {str(root_target_ids[index]).strip()}）"
            if index < len(root_target_ids) and str(root_target_ids[index]).strip()
            else str(value).strip()
            for index, value in enumerate(root_targets)
            if str(value).strip()
        )
        participant_text = "、".join(str(value).strip() for value in participants if str(value).strip())
        focus_line = "短时对话起点："
        focus_line += root_sender_text or "群成员"
        if root_target_text:
            focus_line += f" → @{root_target_text}"
        if participant_text:
            focus_line += f"；参与者：{participant_text}"
        return [focus_line]
    return []


def _render_group_history(items: Sequence[Any]) -> list[str]:
    rendered: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("sender_label") or item.get("sender_id") or "群成员").strip()
        sender = str(item.get("sender_id") or "").strip()
        if sender:
            label = f"{label}（QQ {sender}）"
        text = str(item.get("text") or "").strip()
        if text:
            targets = item.get("at_target_labels")
            targets = targets if isinstance(targets, (list, tuple)) else ()
            target_ids = item.get("at_target_ids")
            target_ids = target_ids if isinstance(target_ids, (list, tuple)) else ()
            target_text = "、".join(
                f"@{str(value).strip()}（QQ {str(target_ids[index]).strip()}）"
                if index < len(target_ids) and str(target_ids[index]).strip()
                else f"@{str(value).strip()}"
                for index, value in enumerate(targets)
                if str(value).strip()
            )
            recipient = f" → {target_text}" if target_text else ""
            rendered.append(f"{label}{recipient}：{text}")
    return rendered


def memory_message_meta(turn: QQTurn, *, message: Any | None = None) -> dict[str, Any]:
    """Expose only routing metadata to memory; never pass raw QQ payloads."""

    selected = message
    if selected is None:
        for item in turn.messages:
            if str(getattr(item, "message_id", "") or "").strip() == str(turn.primary_message_id or "").strip():
                selected = item
                break
    selected = selected or (turn.messages[-1] if turn.messages else None)
    if selected is None:
        return {"message_id": turn.primary_message_id}
    metadata: dict[str, Any] = {
        "message_id": str(getattr(selected, "message_id", "") or turn.primary_message_id or "").strip(),
        "sender_id": str(getattr(selected, "sender_id", "") or "").strip(),
        "has_media": bool(getattr(selected, "has_media", False)),
    }
    raw_event = getattr(selected, "raw_event", {})
    if isinstance(raw_event, Mapping):
        self_id = str(raw_event.get("self_id") or raw_event.get("selfId") or "").strip()
        if self_id:
            metadata["self_id"] = self_id
        for key in ("mentioned_bot", "at_bot"):
            if raw_event.get(key):
                metadata[key] = True
    quoted_message = getattr(selected, "quoted_message", {})
    if isinstance(quoted_message, Mapping):
        quoted_sender_id = str(
            quoted_message.get("sender_id")
            or quoted_message.get("user_id")
            or ""
        ).strip()
        if quoted_sender_id:
            metadata["quoted_sender_id"] = quoted_sender_id
    attachment_refs: list[dict[str, str]] = []
    for item_message in ((selected,) if message is not None else tuple(turn.messages or ())):
        for value in tuple(getattr(item_message, "attachments", ()) or ()):
            if not isinstance(value, Mapping):
                continue
            item: dict[str, str] = {}
            for key in ("kind", "media_id", "url", "local_path", "file_id", "name", "size"):
                raw = str(value.get(key) or "").strip()
                if raw:
                    item[key] = "[inline-image]" if key == "url" and raw.lower().startswith("data:") else raw[:1024]
            if item:
                item.update({
                    "source_message_id": str(getattr(item_message, "message_id", "") or ""),
                    "source_sender_id": str(getattr(item_message, "sender_id", "") or ""),
                    "reference_kind": "direct",
                })
                attachment_refs.append(item)
    for visual_message_id, inputs in turn.visual_inputs:
        if (message is not None and visual_message_id != metadata["message_id"]
            and not (turn.visual_context_inherited and metadata["message_id"] == turn.primary_message_id
                     and visual_message_id == turn.visual_context_source_message_id)):
            continue
        for item in inputs:
            reference = str(getattr(item, "reference", "") or "").strip()
            if reference:
                attachment_refs.append(
                    {
                        "kind": "image",
                        "reference": "[inline-image]" if reference.lower().startswith("data:") else reference[:1024],
                        "ready": "true" if bool(getattr(item, "ready", False)) else "false",
                        **turn.visual_source(visual_message_id, item),
                    }
                )
    if attachment_refs:
        metadata["attachment_refs"] = attachment_refs[:12]
    if turn.visual_context_inherited and metadata["message_id"] == turn.primary_message_id:
        metadata["visual_context_inherited"] = True
        metadata["visual_context_source_message_id"] = turn.visual_context_source_message_id
    for segment in tuple(getattr(selected, "segments", ()) or ()):
        payload = getattr(segment, "payload", {})
        payload = payload if isinstance(payload, Mapping) else {}
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
        sources = (payload, data)
        kind = str(getattr(segment, "kind", "") or "").strip().lower()
        if kind == "reply" and not metadata.get("reply_to_message_id"):
            for source in sources:
                for key in ("id", "message_id", "messageId"):
                    value = source.get(key)
                    if value is not None and str(value).strip():
                        metadata["reply_to_message_id"] = str(value).strip()
                        break
                if metadata.get("reply_to_message_id"):
                    break
        if kind == "at":
            targets = list(metadata.get("at_targets") or [])
            for source in sources:
                for key in ("qq", "target", "user_id", "userId", "id"):
                    value = source.get(key)
                    if value is not None and str(value).strip() and str(value).strip() not in targets:
                        targets.append(str(value).strip())
            metadata["at_targets"] = targets
    return metadata


def transcript(
    turn: QQTurn,
    *,
    master_user_ids: Sequence[str] = (),
) -> list[dict[str, str]]:
    transcript: list[dict[str, str]] = []
    # Retained images are separate evidence, never a new requester attachment.
    present_ids = {str(getattr(message, "message_id", "") or "") for message in turn.messages}
    for source_id, inputs in turn.visual_inputs:
        if inputs and source_id not in present_ids:
            markers = []
            for item in inputs:
                status = "图片附件已注入视觉模型" if item.ready else "图片尚未成功读取，不能据此描述"
                markers.append(f"[{status}] " + json.dumps(turn.visual_source(source_id, item), ensure_ascii=False))
            transcript.append({"role": "user", "content": "[保留的原图片来源，不是当前发言者的新附件]\n" + "\n".join(markers)})
    normalized_master_ids = {
        str(value or "").strip()
        for value in master_user_ids
        if str(value or "").strip()
    }
    ordered_messages = list(turn.messages)
    if (
        str(turn.conversation_type or "").casefold() == "group"
        and len({str(getattr(item, "sender_id", "") or "") for item in ordered_messages}) > 1
    ):
        # The primary addressed message is the current request even when
        # another member speaks later in the same short batch.
        primary = next(
            (item for item in ordered_messages if str(getattr(item, "message_id", "") or "") == turn.primary_message_id),
            None,
        )
        if primary is not None:
            ordered_messages = [item for item in ordered_messages if item is not primary] + [primary]
    for message in ordered_messages:
        text = str(getattr(message, "text_without_at_mentions", getattr(message, "text", "")) or "").strip()
        message_id = str(getattr(message, "message_id", "") or "").strip()
        visual_inputs = turn.visuals_for(message_id)
        has_attachment = any(row.get("kind") == "image" for row in (
            *getattr(message, "attachments", ()), *getattr(message, "quoted_attachments", ())))
        if has_attachment or visual_inputs:
            ready = any(bool(getattr(item, "ready", False)) for item in visual_inputs)
            marker = (
                "[图片附件已注入视觉模型]"
                if ready
                else "[图片附件已收到，但尚未成功读取，不能据此描述]"
            )
            text = f"{text}\n{marker}".strip()
            for item in visual_inputs:
                text += "\n[图片来源] " + json.dumps(turn.visual_source(message_id, item), ensure_ascii=False)
        card_text = str(getattr(message, "card_context", "") or "").strip()
        if card_text:
            # 外部资料进入 user 资料区；不更换原作者或提升为系统指令。
            text = f"{text}\n[QQ资料所属消息：{message_id}]\n{card_text}".strip()
        quoted_text = str(getattr(message, "quoted_text", "") or "").strip()
        if quoted_text:
            text = f"{text}\n[引用消息]\n{quoted_text}".strip()
        if not text:
            text = "[未解析的消息附件]"
        label = str(getattr(message, "sender_label", "") or getattr(message, "sender_id", "") or "群成员").strip()
        if str(turn.conversation_type or "").casefold() == "group":
            sender_id = str(getattr(message, "sender_id", "") or "").strip()
            if sender_id and sender_id in normalized_master_ids:
                label = f"主人（{label}）"
            targets = getattr(message, "at_target_labels", ())
            targets = targets if isinstance(targets, (list, tuple)) else ()
            target_text = "、".join(f"@{str(value).strip()}" for value in targets if str(value).strip())
            recipient = f" → {target_text}" if target_text else ""
            text = f"{label}{recipient}：{text}"
        timestamp = getattr(message, "timestamp", None)
        if timestamp is not None:
            try:
                ts_float = float(timestamp)
                if ts_float > 0:
                    stamp = time.strftime("%H:%M", time.localtime(ts_float))
                    text = f"[{stamp}] {text}"
            except (TypeError, ValueError, OverflowError, OSError):
                pass
        transcript.append({"role": "user", "content": text})
    return transcript


def images(turn: QQTurn) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for message_id, inputs in turn.visual_inputs:
        for item in inputs:
            model_item = item.as_model_item()
            if isinstance(model_item, Mapping):
                images.append({**dict(model_item), **turn.visual_source(message_id, item)})
    return images


def user_images_from_context(context: Any) -> list[dict[str, Any]]:
    if not isinstance(context, Mapping):
        return []
    raw = context.get("user_images", [])
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def user_images(state: AgentState) -> list[dict[str, Any]]:
    context = state.context if isinstance(state.context, Mapping) else {}
    raw = context.get("user_images", [])
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def event_context(turn: QQTurn, *, bot_ids: Sequence[str] = ()) -> dict[str, Any]:
    for message in tuple(turn.messages or ()):
        if not bool(getattr(message, "is_poke", False)):
            continue
        if not isinstance(message, IncomingMessage) or not ReplyRoutePolicy(bot_ids=bot_ids).decide(message).should_reply:
            continue
        raw = getattr(message, "raw_event", {})
        raw = raw if isinstance(raw, Mapping) else {}
        sender = raw.get("sender") if isinstance(raw.get("sender"), Mapping) else {}
        label = ""
        for key in ("card", "nickname", "remark"):
            value = str(sender.get(key) or "").strip()
            if value:
                label = value
                break
        return {
            "kind": "poke",
            "actor_id": str(getattr(message, "sender_id", "") or "").strip(),
            "actor_label": label,
            "target_id": next(str(raw.get(key) or "").strip() for key in ("target_id", "targetId")
                              if str(raw.get(key) or "").strip()),
            "target_is_bot": True,
        }
    return {}


def recent_context(turn: QQTurn, *, limit: int) -> str:
    lines: list[str] = []
    for message in turn.messages[-limit:]:
        text = str(getattr(message, "text", "") or "").strip()
        if text:
            label = str(getattr(message, "sender_label", "") or getattr(message, "sender_id", "") or "群成员").strip()
            sender_id = str(getattr(message, "sender_id", "") or "").strip()
            lines.append(f"{label}（QQ {sender_id}）：{text}" if str(turn.conversation_type).casefold() == "group" else text)
    return "\n".join(lines)
