"""Canonical results come from executor and verifier records, not model JSON."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from shinku.agent.task_state import redact
from .contracts import canonical_hash
from .information import bounded_projection
from shinku.agent.analysis_acceptance import advisory_ids
from shinku.agent.answer_protocol import current_answer_draft


_INTERNAL = {"reasoning", "reasoning_content", "analysis", "chain_of_thought", "transcript",
             "system_prompt", "file_text", "page_text", "stdout", "stderr", "raw_response"}
_RECEIPTS = {"verified_effects", "completion_targets", "artifact_receipts", "sha256", "bytes",
             "url", "http_status", "sources", "delivery_receipt", "result_valid", "result_facts", "self_state_receipt"}


def result_data(value, depth=0, path=()):
    return bounded_projection(value, excluded=_INTERNAL,
        protected_text_paths=("original_requests", "verified_answer", "execution_report", "answer_draft", "delivery_brief"))[0]


def seal_result(payload):
    projected, projection = bounded_projection(payload, excluded=_INTERNAL,
        # 答复正文沿用受保护文本的有界额度，避免导出将分析尾部变成 excerpt 对象。
        # 保留草稿不授予 verified_answer 或完成状态；核验事实仍由独立回执决定。
        protected_text_paths=("original_requests", "verified_answer", "execution_report", "answer_draft", "delivery_brief"))
    safe = redact(projected)
    if "information" in safe or projection["gap_count"]:
        information = safe.get("information") if isinstance(safe.get("information"), dict) else {}
        safe["information"] = {**information, "schema_version": 1, "projection": projection}
        if projection["gap_count"]:
            safe["information"]["status"] = "incomplete"
    safe["result_hash"] = canonical_hash(safe)
    return safe


def export_task_result(task, *, status=None, reason="", report="", available=True):
    completion = copy.deepcopy(task.get("completion", {}))
    state = status or str(task.get("status") or "unverified")
    if ("completion_attempt" in task
            and task.get("completion_attempt") != task.get("execution_attempt", 0)):
        completion = {"accepted": False, "reason": "completion_attempt_stale"}
    if state == "completed" and completion.get("accepted") is not True:
        state, reason = "unverified", "completion_receipt_missing"
    from .reply_delivery import is_reply_delivery
    reply_only = is_reply_delivery(completion)
    evidence, artifacts, facts = [], [], []
    selection_gaps = []
    calls = task.get("tool_calls", [])
    if len(calls) > 128:
        selection_gaps.append({"path": "/evidence", "kind": "list", "total": len(calls), "retained": 128,
            "omitted": len(calls) - 128, "reason": "latest_receipt_limit", "retrieval": "task_ledger_or_material_reference"})
    for call in calls[-128:]:
        receipt = (call.get("data") or {}).get("state_updates", {})
        if not isinstance(receipt, Mapping):
            receipt = {}
        fields = {key: copy.deepcopy(value) for key, value in receipt.items() if key in _RECEIPTS}
        evidence.append({"invocation_id": call.get("id", call.get("invocation_id", "")),
            "tool": call.get("name", ""), "execution_status": call.get("status", "unknown"),
            "source": "executor_receipt", "receipt": fields})
        if call.get("status") == "ok":
            for source_index, row in enumerate(receipt.get("result_facts", [])):
                if isinstance(row, Mapping):
                    facts.append({**{k: copy.deepcopy(row[k]) for k in (
                        "label", "value", "unit", "condition", "reference") if k in row},
                        "source": "executor_receipt", "invocation_id": call.get("id", ""),
                        **({"source_index": source_index} if task.get("material_package") else {})})
        for artifact in receipt.get("artifact_receipts", []):
            if isinstance(artifact, Mapping):
                artifacts.append({**dict(artifact), "source": "executor_receipt",
                    "result_version": task.get("revision", 0),
                    "verification": "task_passed" if not reply_only and state == "completed" and completion.get("accepted") is True else "unconfirmed"})
    required = [row["id"] for row in (task.get("contract") or {}).get("requirements", [])]
    if reply_only:
        required = []  # 历史模型条件仍保留原件，但不再伪装为本轮用户缺项。
    satisfied = completion.get("satisfied", [])
    missing = completion.get("missing", []) or [v for v in required if v not in satisfied and v not in advisory_ids(completion)]
    payload = {"schema_version": 1, "task_id": task.get("task_id", ""),
        "attempt": task.get("execution_attempt", 0), "version": task.get("revision", 0),
        "updated": task.get("updated", 0), "scope": {"requester": task["requester"], "session": task["session"]},
        "original_requests": task.get("requests", []), "constraints": task.get("confirmed_constraints", []),
        "contract": task.get("contract"), "execution": {"status": state, "reason": reason,
            "source": "agent_executor", "result_available": available},
        "verification": {**completion, "source": "completion_verifier"}, "evidence": evidence, "facts": facts,
        "artifacts": artifacts, "remaining": missing,
        "uncertainties": task.get("uncertain_attempts", []), "budget": task.get("budget", {}),
        "execution_report": {"text": report, "source": "execution_model", "is_fact_evidence": False},
        "delivery": {"status": "not_recorded", "sent": False, "source": "delivery_not_attached"}}
    # 原稿属于模型资料，不能变成 verified_answer；展示须由当前宿主失败路径
    # 重验来源后显式允许。取消、撤权和新轮次不从旧 progress 自动恢复正文。
    progress = task.get("progress", {})
    progress = progress if isinstance(progress, Mapping) else {}
    presentation = progress.get("answer_draft_presentation", {})
    presentation = presentation if isinstance(presentation, Mapping) else {}
    draft = current_answer_draft(progress, task)
    if (state in {"partial", "failed"} and presentation.get("allowed") is True and draft
            and draft["sha256"] == presentation.get("sha256")
            and reason == presentation.get("reason")):
        payload["answer_draft"] = {**draft, "retained_full": True, "source": "unverified_model_draft"}
        payload["execution_report"]["source"] = "host_failure_with_model_draft"
    material_information = task.get("material_information")
    payload["information"] = {"schema_version": 1, "status": "unknown", "required_status": "unknown",
        "reason": "legacy_source_completeness_unknown", "upstream_completeness": "unknown"}
    if isinstance(material_information, Mapping):
        payload["information"].update(copy.deepcopy(material_information))
        payload["information"].pop("read_usage", None)
        payload["information"].pop("read_receipts", None)
    payload["information"]["export_gaps"] = selection_gaps
    if task.get("snapshot_information"):
        payload["information"]["snapshot"] = copy.deepcopy(task["snapshot_information"])
    if selection_gaps or task.get("snapshot_information", {}).get("gap_count"):
        payload["information"]["status"] = "incomplete"
    payload["presentation"] = {"schema_version": 1, "optional_success_receipts": bool(facts or artifacts),
        "policy_source": "host_export", "detail_policy_version": 2,
        "material_recovery": task.get("material_package", {}).get("schema_version") == 1,
        "required": "goals_constraints_facts_conditions_artifacts_limits"}
    answer = task.get("result_answer_evidence")
    layers = completion.get("layers", {})
    # 同一任务的旧答案缓存不能越过当前普通分析的质量结论；交付成功不授予事实标记。
    quality_allows_verified = not (layers.get("policy") in {"ordinary_readonly_analysis", "ordinary_analysis"}
        and layers.get("answer_evidence") != "verified")
    if isinstance(answer, Mapping) and quality_allows_verified and not reply_only:
        from shinku.conversation.models import AnswerEvidence
        try:
            bound = AnswerEvidence(**dict(answer)).bound_to(report)
            if bound.kind in {"public", "history", "local", "state"} and bound.resolved:
                payload["verified_answer"] = {"text": report, "validated": True,
                    "answer_sha256": bound.answer_sha256, "source": "answer_verifier"}
                # The verified answer carries the useful result and sources;
                # retain successful tool receipts in the existing audit tier.
                if state == "completed" and completion.get("accepted") is True:
                    payload["presentation"]["optional_success_receipts"] = True
        except (TypeError, ValueError):
            pass
    from .delivery_brief import export_brief
    brief = export_brief(task, report)
    if brief is not None:
        payload['delivery_brief'] = brief
    return seal_result(payload)


def export_query_result(*, scope, request, route, status, completion, calls=(),
                        delivery_status="not_sent", sent=False, reason="", operation_id="", verified_answer=None):
    """Adapt an already-executed fast query without creating/running a task."""
    task = {"task_id": "", "requester": scope.requester, "session": scope.session,
        "requests": [{"message_id": "", "text": request}], "status": status,
        "completion": completion, "tool_calls": list(calls), "revision": 0}
    payload = export_task_result(task, status=status, reason=reason)
    payload.pop("result_hash", None)
    payload.update({"route": route, "task_identity_missing": True, "operation_id": operation_id})
    if verified_answer is not None:
        # Callers pass a text/evidence tuple; a model label is never enough.
        if isinstance(verified_answer, tuple) and len(verified_answer) == 2:
            text, answer = verified_answer
            bound = answer.bound_to(text)
            if bound.kind in {"public", "history", "local", "state"} and bound.resolved:
                payload["verified_answer"] = {"text": text, "validated": True,
                    "answer_sha256": bound.answer_sha256, "source": "answer_verifier"}
    payload["delivery"] = {"status": delivery_status, "sent": sent, "source": "host_delivery_receipt"}
    if route in {"public_query", "perception"}:
        # This describes the exported host receipts and current answer gate,
        # never completeness of all upstream pages or a model's call count.
        answer_ready = bool(payload["verification"].get("accepted") is True and payload.get("verified_answer"))
        gaps = payload["information"].get("export_gaps", [])
        payload["information"].update(query_scope="current_operation_export",
            status="incomplete" if gaps else "complete" if payload["evidence"] else "unknown",
            required_status="complete" if answer_ready else "incomplete",
            reason="query_receipt_and_answer_scope", upstream_completeness="unknown")
        payload["presentation"].update(query_receipt_scope="current_operation_export",
            optional_success_receipts=True)
        if not answer_ready and not payload["remaining"]:
            payload["remaining"] = ["原问题的答案尚未通过证据核验"]
    return seal_result(payload)


def capture_query_result(context, turn, reply, envelopes, *, route, answer_evidence=None):
    """Export existing read-only receipts after actual host delivery."""
    context["agent_query_result"] = build_query_result(context, turn, envelopes, route=route,
        status=reply.agent_status, reason=reply.reason, text=reply.final_text,
        answer_evidence=answer_evidence, delivery_status=reply.delivery_status, sent=reply.sent)


def build_query_result(context, turn, envelopes, *, route, status, reason="", text="",
                       answer_evidence=None, delivery_status="not_sent", sent=False):
    """Capture before expression/delivery without adding any query or task."""
    from .contracts import AgentScope
    from shinku.agent.request_scope import RequestScope
    scope = context.get("agent_request_scope")
    if not isinstance(scope, RequestScope):
        primary = next((v for v in turn.messages if str(getattr(v, "message_id", "")) == turn.primary_message_id), None)
        requester = str(getattr(primary, "sender_id", "") or "")
        if not requester:
            return None
        actor = AgentScope(requester, f"{turn.conversation_type.casefold()}:{turn.conversation_id}")
        request = str(getattr(primary, "text", "") or "")
    else:
        actor, request = AgentScope(scope.requester, scope.session), "\n".join(scope.user_messages())
    calls = [{"id": envelope.invocation_id, "name": (envelope.data or {}).get("tool_type", ""),
              "status": envelope.status, "data": envelope.data or {}} for envelope in envelopes]
    bound = answer_evidence.bound_to(text) if answer_evidence is not None else None
    evidence = bound.as_dict() if bound is not None else {}
    completion = {"accepted": bool(bound and bound.kind in {"public", "history", "local", "state"} and bound.resolved),
                  "status": status, "answer_evidence": evidence}
    result = export_query_result(scope=actor, request=request, route=route,
        status=status, completion=completion, calls=calls, reason=reason, operation_id=turn.primary_message_id,
        delivery_status=delivery_status, sent=sent,
        verified_answer=(text, answer_evidence) if answer_evidence is not None else None)
    # 历史原话是“读到了什么”，模型答复是未认证草稿。二者都须跨过表达边界，
    # 但不能因为读取成功就成为 result_facts、verified_answer 或完成认证。
    if (route == "perception" and bound is not None and bound.kind == "history"
            and not context.get("bot_history_sources_invalid")
            and not context.get("qq_dialogue_sources_changed")):
        observations = []
        for call in calls:
            if call["name"] != "memory_search" or call["status"] != "ok":
                continue
            updates = call["data"].get("state_updates", {})
            if not isinstance(updates, Mapping):
                continue
            inspection = updates.get("memory_inspection", {})
            lookup = updates.get("memory_lookup", {})
            if not isinstance(inspection, Mapping) or not isinstance(lookup, Mapping):
                continue
            rows = inspection.get("records", [])
            if not isinstance(rows, (list, tuple)) or not rows:
                continue
            records = []
            for row in rows[:12]:
                if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
                    continue
                item = {k: copy.deepcopy(row[k]) for k in (
                    "record_id", "scope", "source_id", "text", "created_at", "updated_at") if k in row}
                meta = row.get("metadata", {})
                item["metadata"] = {k: copy.deepcopy(meta[k]) for k in (
                    "role", "sender_id", "requester_id", "self_id", "source_version",
                    "owner_source_scope", "owner_participant_id", "reading_representation",
                    "delivery_status", "delivery_content_complete", "source_time")
                    if isinstance(meta, Mapping) and k in meta}
                records.append(item)
            if records:
                observations.append({"invocation_id": call["id"], "records": records,
                    "lookup": {k: copy.deepcopy(lookup[k]) for k in (
                        "status", "query_completed", "query", "mode", "read_at", "coverage") if k in lookup}})
        if observations:
            result.pop("result_hash", None)
            result["read_observations"] = observations
            result["presentation"]["history_read_material"] = 1
            if isinstance(text, str) and text.strip() and not result.get("verified_answer"):
                result["execution_report"] = {"text": text, "source": "unverified_query_draft",
                                              "is_fact_evidence": False}
            result = seal_result(result)
    return result
