"""Bot 历史问答的本地来源检查，不判断生成文字的语义，不派发模型请求。

工具执行后的快照只保存摘要；发布前重新按当前身份与作用域读取同一原件。
复用既有只读来源校验器，保留遗忘、到期、作者、派生来源和版本边界。
这不是另一个记忆库，亦不将读取成功认证为模型观点全部正确。
"""
from collections.abc import Mapping
from collections import Counter
from dataclasses import replace
import hashlib
import json

from shinku.memory.context_sources import (
    CONTEXT_SOURCE_KEY, ContextSourceCheck, ContextSourceGroup, PromptSourceDependencies,
)


def history_snapshot(turn, invocations, handlers, envelopes):
    """返回当前可读原件的内容摘要；任一原件不可验证时返回 None。"""
    from shinku.memory.projection.contracts import Deadline, ProjectionRequest
    from shinku.memory.projection.sources import ProjectionSources

    primary = next((m for m in turn.messages if m.message_id == turn.primary_message_id), None)
    requester = str(getattr(primary, "sender_id", ""))
    snapshots = []
    for invocation, envelope in zip(invocations, envelopes):
        handler = handlers.get(invocation.name)
        if not getattr(handler, "agent_history_answer_source", False):
            continue
        updates = envelope.data.get("state_updates", {}) if isinstance(envelope.data, Mapping) else {}
        inspection = updates.get("memory_inspection", {})
        rows = inspection.get("records", []) if isinstance(inspection, Mapping) else []
        if not rows:
            continue
        service = getattr(handler, "service", None)
        if service is None or not requester or len(rows) > 12:
            return None
        scope = service.scope_for(turn.conversation_id, turn.conversation_type)
        owner = service._is_master_request(conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type, message_meta={"sender_id": requester})
        mode = str(invocation.arguments.get("mode") or "retrieve")
        model_retrieval = (service.policy.projection.enabled and service.policy.projection.retrieval_strategy == "model")
        allowed = service._history_scopes(scope, owner_request=owner,
            mode="retrieve" if model_retrieval and owner and scope.startswith("private:") else mode)
        handles = []
        for row in rows:
            # recent 可返回本轮原话；它由当前 turn 持有，不是待回查的旧历史。
            current_source = f"turn:{turn.conversation_id}:{turn.primary_message_id}"
            if isinstance(row, Mapping) and row.get("source_id") in {current_source, "owner:" + current_source}:
                continue
            if not isinstance(row, Mapping) or row.get("scope") not in allowed:
                return None
            metadata = row.get("metadata", {})
            version = str(metadata.get("source_version") or "") if isinstance(metadata, Mapping) else ""
            handles.append((str(row.get("record_id") or ""), row["scope"], version))
        request = ProjectionRequest(scope, tuple(allowed), requester, "", "", turn.primary_message_id,
                                    int(service.clock()),
                                    participant_ids=service.policy.owner_participant_ids if owner else ())
        try:
            fresh = ProjectionSources(service.store).read(tuple(handles), request, Deadline(300))
        except Exception:
            return None
        retained = [row for row in rows if row.get("source_id") not in {
            f"turn:{turn.conversation_id}:{turn.primary_message_id}",
            f"owner:turn:{turn.conversation_id}:{turn.primary_message_id}"}]
        for original, current in zip(retained, fresh):
            text = original.get("text")
            if current is None or not isinstance(text, str) or not current.text.startswith(text):
                return None
            snapshots.append((current.record_id, current.scope, current.metadata["source_version"]))
    return hashlib.sha256(json.dumps(snapshots, ensure_ascii=False).encode()).hexdigest()


def negative_history_reply(invocations, handlers, envelopes):
    """失败、越界和未找到由真实回执直接表述，不再花一次生成来措辞。"""
    from shinku.agent.history_answer import HistoryAnswerLedger
    ledger = HistoryAnswerLedger(enabled=True)
    for invocation, envelope in zip(invocations, envelopes):
        ledger.observe(envelope, handlers.get(invocation.name))
    feedback = ledger.negative_feedback()
    return feedback[:2] if feedback else None


def turn_source_binding(turn):
    """绑定本轮消息和署名，不保存正文；候选定位符不能被另一个回合借用。"""
    value = [getattr(turn, "conversation_type", ""), getattr(turn, "conversation_id", ""),
             getattr(turn, "primary_message_id", ""), tuple(getattr(turn, "message_ids", ())),
             [(getattr(m, "message_id", ""), getattr(m, "sender_id", ""))
              for m in getattr(turn, "messages", ())]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def projection_source_group(view):
    """投影的所有有效来源均登记依赖，展示预算省略项也不能失去版本约束。"""
    if not isinstance(view, Mapping):
        return ContextSourceGroup("projection", read_path="projection_not_selected", state="not_applicable")
    diag = view.get("diagnostics", {})
    diag = diag if isinstance(diag, Mapping) else {}
    rows = view.get("records", ()) if view.get("rendered") else ()
    try:
        handles = (tuple(tuple(h) for h in view["source_dependencies"]) if view.get("rendered")
                   and "source_dependencies" in view else
                   tuple((r["record_id"], r["scope"], r["source_version"]) for r in rows))
        return ContextSourceGroup("projection", handles=handles,
            omitted_sources=max(0, len(handles) - sum(bool(r.get("text")) for r in rows)),
            read_path=str(diag.get("read_path") or "unified_projection"),
            state="selected" if handles else "disabled" if diag.get("state") == "disabled"
                  else "unavailable" if diag.get("state") in {"failed", "unavailable"}
                  else "empty" if view.get("rendered") else "not_applicable",
            reason_codes=tuple(diag.get("gap_reasons", ()))[:16])
    except (ValueError, TypeError, KeyError):
        return ContextSourceGroup("projection", state="invalid", reason_codes=("invalid_source_contract",))


def owner_source_group(view):
    """空的近期视图也保留签名，生成期间新到达但未展示的更正仍可使候选失效。"""
    if not isinstance(view, Mapping):
        return ContextSourceGroup("owner_continuity", read_path="recent_owner_sources", state="not_applicable")
    try:
        diag = view.get("diagnostics", {})
        handles = tuple(tuple(h) for h in view.get("handles", ()))
        selection_at = view.get("selection_at", 0)
        return ContextSourceGroup("owner_continuity", handles=handles, signature=view["signature"],
            # 新协议冻结本轮选材，不把五分钟窗口冒充原件TTL；旧适配器保持保守处理。
            expires_at=0 if selection_at else int(view.get("dependency_expires_at") or view.get("expires_at") or 0),
            selection_at=selection_at,
            omitted_sources=int(diag.get("omitted_sources", 0)), read_path="recent_owner_sources",
            state="selected" if handles else "empty")
    except (ValueError, TypeError, KeyError, AttributeError):
        return ContextSourceGroup("owner_continuity", state="invalid", reason_codes=("invalid_source_contract",))


def context_source_dependencies(context):
    """旧字典协议只在此兼容读取；新提示写入仅使用一个协议对象。"""
    stored = context.get(CONTEXT_SOURCE_KEY)
    if isinstance(stored, PromptSourceDependencies):
        return stored
    result = PromptSourceDependencies(turn_source_binding(context.get("qq_turn")))
    try:
        if stored is not None:
            raise ValueError("invalid_source_contract")
        handles = tuple(tuple(h) for h in context.get("_bot_prompt_source_handles", ()))
        if handles:
            result = result.with_group(ContextSourceGroup("legacy", handles=handles, read_path="legacy_adapter"))
        signature = context.get("_owner_continuity_signature")
        if signature is not None:
            result = result.with_group(ContextSourceGroup("owner_continuity", signature=signature,
                                                         read_path="recent_owner_sources"))
    except (ValueError, TypeError):
        result = result.with_group(ContextSourceGroup("legacy", state="invalid",
                                                     reason_codes=("invalid_source_contract",)))
    return result


def checked_prompt_snapshot(service, turn, handles):
    """只回查原件并返回本地原因；不发现候选、不做摘要、不调用语义模型。"""
    if not handles:
        return ContextSourceCheck(True)
    if service is None or turn is None or len(handles) > 64:
        return ContextSourceCheck(False, reason="source_budget" if len(handles) > 64 else "source_reader_unavailable")
    from shinku.memory.projection.contracts import Deadline, ProjectionRequest
    from shinku.memory.projection.sources import ProjectionSources
    try:
        primary = next((m for m in turn.messages if m.message_id == turn.primary_message_id), None)
        requester = str(getattr(primary, "sender_id", ""))
        if not requester:
            return ContextSourceCheck(False, reason="requester_unavailable")
        scope = service.scope_for(turn.conversation_id, turn.conversation_type)
        owner = service._is_master_request(conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type, message_meta={"sender_id": requester})
        allowed = service._history_scopes(scope, owner_request=owner, mode="retrieve")
        request = ProjectionRequest(scope, tuple(allowed), requester, "", "", turn.primary_message_id, int(service.clock()),
                                    participant_ids=service.policy.owner_participant_ids if owner else ())
        fresh = ProjectionSources(service.store).read_checked(tuple(handles), request, Deadline(300))
        counts = Counter(reason for reason in fresh.reasons if reason)
        if counts:
            return ContextSourceCheck(False, reason=next(iter(counts)),
                rejected=sum(counts.values()), reason_counts=tuple(sorted(counts.items())))
        return ContextSourceCheck(True, snapshot=tuple(
            (row.record_id, row.scope, row.metadata["source_version"]) for row in fresh.records))
    except Exception:
        return ContextSourceCheck(False, reason="source_read_failed")


def check_context_sources(service, context, *, owner_reader):
    """按本轮身份重新验证同一依赖集；签名只定位，不能继承旧授权。"""
    dependencies = context_source_dependencies(context)
    paths = tuple(dict.fromkeys(g.read_path for g in dependencies.groups if g.read_path))
    def invalid(reason):
        return ContextSourceCheck(False, reason=reason, read_paths=paths)
    if dependencies.binding != turn_source_binding(context.get("qq_turn")):
        return invalid("source_turn_changed")
    if any(g.state == "invalid" for g in dependencies.groups):
        return invalid("invalid_source_contract")
    if service is not None and any(g.expires_at for g in dependencies.groups):
        now = int(service.clock())
        if any(g.expires_at and now >= g.expires_at for g in dependencies.groups):
            return invalid("source_window_expired")
    checked = checked_prompt_snapshot(service, context.get("qq_turn"), dependencies.handles)
    if not checked.valid:
        return replace(checked, read_paths=paths)
    for group in dependencies.groups:
        if group.kind == "owner_continuity" and group.signature is not None:
            try:
                shared = owner_reader(context)
            except Exception:
                return invalid("owner_sources_unavailable")
            if shared is None:
                return invalid("owner_sources_unavailable")
            if shared.get("signature") != group.signature:
                return invalid("owner_dependency_changed")
    return replace(checked, read_paths=paths)
