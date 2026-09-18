"""任务工作区的产物视图。

工具在流式执行过程中会吐出两类"可以留在工作区里的东西"：生成的文件，和
从附件收件箱取回来的素材。这个模块把它们统一成一种**产物字典**，
并给出合并与去重的规则。

全部是纯函数——不做 I/O、不持状态，输入输出都是普通字典与列表。
这样做的直接好处是：任务链路的其余部分可以在没有存储和进程的情况下被测试。
"""

from __future__ import annotations

from typing import Any

#: 透传字段的"视为未提供"判定集合。用 ``==`` 比较，因此 ``0`` 与 ``False`` 会被保留。
_ABSENT = (None, "", [], {})

_GENERATED_PASSTHROUGH = (
    "file_ext",
    "file_size",
    "created_by_tool",
    "version_of_generated_id",
    "version_no",
)

_ATTACHMENT_PASSTHROUGH = (
    "origin_name",
    "file_ext",
    "file_size",
    "mime_type",
)

#: 标题统一截断到这个长度。
_TITLE_CHARS = 120

#: ``stem_role`` 统一截断到这个长度。
_STEM_ROLE_CHARS = 40

#: 身份串判定的键序。顺序即优先级，不可调整。
_IDENTITY_KEYS = (
    "id",
    "generated_handle",
    "generated_id",
    "attachment_handle",
    "attachment_id",
)


def _first_present(*candidates: Any) -> Any:
    """返回第一个真值候选；全为假值时返回 ``None``。

    先选值、再做字符串加工，这个顺序不能反——``"   "`` 是真值但去空白后为空，
    如果先加工再选，就会错误地跳过它而落到下一个候选上。
    """

    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _text(value: Any) -> str:
    """``str(value or "").strip()``。"""

    return str(value or "").strip()


def _kind(primary: Any, secondary: Any) -> str:
    """产物类型：先取第一个真值，再小写化；仍为空则回落 ``"file"``。"""

    return str(_first_present(primary, secondary, "file")).strip().lower() or "file"


def _label(*candidates: Any) -> str:
    """标题：先取第一个真值，再去空白并截断。"""

    chosen = _first_present(*candidates)
    return str(chosen).strip()[:_TITLE_CHARS] if chosen is not None else ""


def _status(value: Any) -> str:
    """状态：去空白；为空则回落 ``"ready"``。"""

    return _text(value) or "ready"


def _carry_over(target: dict[str, Any], source: Any, keys: tuple[str, ...]) -> None:
    """把 ``keys`` 里存在且非空的值原样搬到 ``target``（不做类型转换）。"""

    for key in keys:
        value = source.get(key)
        if value not in _ABSENT:
            target[key] = value


def artifact_from_generated_file(
    *,
    generated: Any,
    tool_type: str,
    send_to_user: bool,
) -> dict[str, Any] | None:
    """把一条生成文件记录转成产物。

    身份取 ``generated_handle``，没有则退到 ``generated_id``；两者都空则
    这条记录不构成产物，返回 ``None``。
    """

    if not isinstance(generated, dict):
        return None

    handle = _text(generated.get("generated_handle"))
    generated_id = _text(generated.get("generated_id"))
    artifact_id = handle or generated_id
    if not artifact_id:
        return None

    card = generated.get("content_card") if isinstance(generated.get("content_card"), dict) else {}
    separation = card.get("separation") if isinstance(card.get("separation"), dict) else {}

    requested = bool(send_to_user)
    artifact: dict[str, Any] = {
        "id": artifact_id,
        "kind": _kind(generated.get("output_format"), generated.get("file_ext")),
        "title": _label(generated.get("output_title"), handle, "生成文件"),
        "status": _status(generated.get("status")),
        "source": "generated_file",
        "tool": tool_type,
        "send_to_user": requested,
        # 用户点名要的东西是"交付物"，顺手产生的只是"工作区素材"。
        "delivery_role": "requested_output" if requested else "workspace_material",
    }
    if generated_id:
        artifact["generated_id"] = generated_id
    if handle:
        artifact["generated_handle"] = handle

    stem_role = _text(separation.get("stem_role"))
    if stem_role:
        artifact["stem_role"] = stem_role[:_STEM_ROLE_CHARS]

    _carry_over(artifact, generated, _GENERATED_PASSTHROUGH)
    return artifact


def artifact_from_attachment_item(*, item: Any, tool_type: str) -> dict[str, Any] | None:
    """把一条附件收件箱条目转成产物。

    身份取 ``attachment_handle``，没有则退到 ``attachment_id``。
    附件一律是工作区素材，不会成为"用户点名要的交付物"。
    """

    if not isinstance(item, dict):
        return None

    handle = _text(item.get("attachment_handle"))
    attachment_id = _text(item.get("attachment_id"))
    artifact_id = handle or attachment_id
    if not artifact_id:
        return None

    artifact: dict[str, Any] = {
        "id": artifact_id,
        "kind": _kind(item.get("kind"), item.get("file_ext")),
        "title": _label(item.get("summary_title"), item.get("origin_name"), handle, "临时素材"),
        "status": _status(item.get("status")),
        "source": "attachment_inbox",
        "tool": tool_type,
        # 条目来源名是 ``source``，落到产物上叫 ``source_type``——产物自己的
        # ``source`` 已经被"附件收件箱"这个语义占用了。
        "source_type": _text(item.get("source")),
        "delivery_role": "workspace_material",
    }
    if attachment_id:
        artifact["attachment_id"] = attachment_id
    if handle:
        artifact["attachment_handle"] = handle

    _carry_over(artifact, item, _ATTACHMENT_PASSTHROUGH)
    return artifact


def extract_artifacts_from_tool_events(
    *,
    tool_type: str,
    stream_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从工具发出的流式事件里捞出所有产物，保持事件原有的先后顺序。

    只认两种事件类型，其余的（进度、日志、中间态）一律跳过。
    """

    found: list[dict[str, Any]] = []
    for event in stream_events:
        if not isinstance(event, dict):
            continue
        event_type = _text(event.get("type"))
        if event_type == "generated_file_ready":
            artifact = artifact_from_generated_file(
                generated=event.get("generated_file"),
                tool_type=tool_type,
                send_to_user=bool(event.get("send_to_user")),
            )
        elif event_type == "attachment_remote_media_ready":
            artifact = artifact_from_attachment_item(item=event.get("item"), tool_type=tool_type)
        else:
            artifact = None
        if artifact:
            found.append(artifact)
    return found


def artifact_identity(artifact: dict[str, Any]) -> str:
    """产物的去重身份串。

    优先用各类句柄/ID；都没有时退到 ``title:<kind>:<title>``，
    这样"同一个标题的同类产物"至少还能被认出来。都没有则为空串（等于"无法去重"）。
    """

    for key in _IDENTITY_KEYS:
        value = _text(artifact.get(key))
        if value:
            return value
    title = _text(artifact.get("title"))
    if not title:
        return ""
    return f"title:{_text(artifact.get('kind'))}:{title}"


def merge_artifacts(
    *,
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把新产物并入既有列表，按身份串去重。

    返回 ``(合并后的完整列表, 本次真正新增的部分)``。

    **两处刻意的行为：**

    - ``existing`` 里重复的条目**不会被清理**——它们已经写进过工作区，历史事实保留；
      去重只作用于新加进来的部分；
    - 同一个产物对象**同时出现在两个返回列表里**（不是各存一份拷贝）。
      调用方改其中一份，另一份同步可见，这是既有语义，消费方依赖它。
    """

    merged: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for item in existing:
        if not isinstance(item, dict):
            continue
        copy = dict(item)
        merged.append(copy)
        identity = artifact_identity(copy)
        if identity:
            index.setdefault(identity, copy)

    added: list[dict[str, Any]] = []
    for item in additions:
        if not isinstance(item, dict):
            continue
        identity = artifact_identity(item)
        if not identity or identity in index:
            continue
        copy = dict(item)
        merged.append(copy)
        added.append(copy)
        index[identity] = copy

    return merged, added


def compact_workspace_for_event(task: dict[str, Any]) -> dict[str, Any]:
    """任务工作区的事件用摘要——只留四个标量，产物本身只报数量。

    事件流里不适合塞完整产物列表，这个函数是那条边界。

    **这里刻意不做 ``strip``。** 前三个字段是任务自身的原值，摘要是"取一段"而不是"整理"；
    只有 ``normalized_goal`` 会被截到 200 字符。产物列表只报长度，不报内容。
    """

    goal = str(task.get("normalized_goal") or "")
    artifacts = task.get("artifacts") or []

    return {
        "task_id": str(task.get("task_id") or ""),
        "status": str(task.get("status") or ""),
        "normalized_goal": goal[:200],
        "artifact_count": len(list(artifacts)),
    }


__all__ = [
    "artifact_from_attachment_item",
    "artifact_from_generated_file",
    "artifact_identity",
    "compact_workspace_for_event",
    "extract_artifacts_from_tool_events",
    "merge_artifacts",
]
