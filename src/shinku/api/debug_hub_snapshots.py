"""Agent, trace, attention, memory and model-service snapshots."""
from __future__ import annotations
import os
from typing import Any
from ..qq.commands import AGENT_PERMISSION_LABELS
from .debug_hub_logs import _safe_call

def _agent_snapshot(assembly: Any) -> dict[str, Any]:
    if assembly is None:
        return {"enabled": False, "status": "not_attached"}

    diagnostics = _safe_call(getattr(assembly, "diagnostics", None), {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    bridge = getattr(assembly, "bridge", None)
    runtime = getattr(bridge, "runtime", None)
    metrics = _safe_call(getattr(runtime, "snapshot_metrics", None), {})
    last_error = _safe_call(getattr(runtime, "snapshot_last_error", None), {})
    memory_service = getattr(bridge, "memory_service", None)
    memory_store = getattr(memory_service, "store", None)
    memory_stats = _safe_call(getattr(memory_store, "stats", None), {})
    memory_path = str(getattr(memory_store, "path", "") or "")
    tool_host = getattr(bridge, "tool_host", None)
    registry = getattr(tool_host, "registry", None)
    names_getter = getattr(registry, "names", None)
    tool_names = _safe_call(names_getter, ())
    if not isinstance(tool_names, (list, tuple, set)):
        tool_names = ()
    tool_names = tuple(str(item) for item in tool_names if str(item).strip())
    runtime_health = {}
    health_getter = getattr(tool_host, "runtime_health", None)
    if callable(health_getter):
        health_value = _safe_call(lambda: health_getter(runtime), None)
        as_dict = getattr(health_value, "as_dict", None)
        if callable(as_dict):
            runtime_health = _safe_call(as_dict, {})
        elif isinstance(health_value, dict):
            runtime_health = health_value
    chat_bundle = getattr(runtime, "chat", None)
    aux_bundle = getattr(runtime, "aux", None)
    command_state = getattr(bridge, "command_state", None)
    permission_mode = str(
        _safe_call(getattr(command_state, 'permission_mode', None), 'traditional') or 'traditional',
    )
    task_coordinator = getattr(assembly, "task_coordinator", None)
    task_snapshot = _safe_call(getattr(task_coordinator, "list", None), []) if task_coordinator is not None else []
    if not isinstance(task_snapshot, list):
        task_snapshot = []
    return {
        "enabled": True,
        "status": "attached",
        "process": {
            "pid": os.getpid(),
            "mode": "embedded",
            "service": "backend",
            "note": "Agent 与 Shinku backend 共用一个进程，不是单独的 agent.exe。",
        },
        "controls": {
            "agent_enabled": bool(_safe_call(getattr(command_state, "is_agent_enabled", None), True)),
            "permission_mode": permission_mode,
            "permission_mode_label": AGENT_PERMISSION_LABELS.get(permission_mode, AGENT_PERMISSION_LABELS["traditional"]),
            "stopped": bool(_safe_call(getattr(command_state, "is_stopped", None), False)),
            "responses_enabled": bool(_safe_call(getattr(command_state, "responses_enabled", None), True)),
        },
        "live_models": {
            "chat": str(getattr(chat_bundle, "model", "") or ""),
            "aux": str(getattr(aux_bundle, "model", "") or ""),
        },
        "diagnostics": diagnostics,
        "runtime": {
            "metrics": metrics if isinstance(metrics, dict) else {},
            "last_error": last_error if isinstance(last_error, dict) else {},
        },
        "memory": {
            "path": memory_path,
            "stats": memory_stats if isinstance(memory_stats, dict) else {},
            "lifecycle": _safe_call(getattr(getattr(memory_store, "lifecycle", None), "status", None), {}),
        },
        "bridge": {
            "sender_configured": getattr(bridge, "sender", None) is not None,
            "allowed_tool_count": len(tool_names),
            "tool_names": list(tool_names),
            "tool_allowlist": sorted(getattr(bridge, "allowed_tool_names", set()) or set()),
            "runtime_health": runtime_health,
            "max_rounds": getattr(bridge, "max_rounds", None),
            "adaptive_budget_enabled": bool(getattr(bridge, "adaptive_budget_enabled", False)),
            "history_turns": getattr(bridge, "history_turns", None),
            "recent_context_messages": getattr(bridge, "recent_context_messages", None),
        },
        "tasks": {
            "count": len(task_snapshot),
            "items": task_snapshot[:50],
        },
    }


def _trace_snapshot(assembly: Any, *, limit: int = 30) -> list[dict[str, Any]]:
    """读取 QQ Agent 的有限结构化回合追踪，不触碰聊天原文。"""

    bridge = getattr(assembly, "bridge", None)
    store = getattr(bridge, "trace_store", None)
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        rows = snapshot(limit=max(1, min(100, int(limit or 30))))
    except Exception:
        return []
    return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _attention_snapshot(napcat_host: Any, assembly: Any, *, limit: int = 80) -> dict[str, Any]:
    """Join content-free ingress decisions with any resulting QQ turn trace."""

    ingress = getattr(napcat_host, "ingress", None)
    store = getattr(ingress, "attention_trace_store", None)
    snapshot = getattr(store, "snapshot", None)
    if not callable(snapshot):
        return {"available": False, "summary": {}, "items": []}
    try:
        payload = snapshot(limit=max(1, min(300, int(limit or 80))))
    except Exception:
        return {"available": False, "summary": {}, "items": []}
    if not isinstance(payload, dict):
        return {"available": False, "summary": {}, "items": []}

    traces = _trace_snapshot(assembly, limit=100)
    by_message_id: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in traces:
        message_ids = trace.get("message_ids")
        if not isinstance(message_ids, (list, tuple)):
            message_ids = [trace.get("message_id", "")]
        for message_id in message_ids:
            identifier = str(message_id or "").strip()
            if identifier:
                key = (str(trace.get("conversation_id") or "").strip(), identifier)
                by_message_id[key] = trace

    items: list[dict[str, Any]] = []
    for raw_item in payload.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        trace_key = (
            str(item.get("conversation_id") or "").strip(),
            str(item.get("message_id") or "").strip(),
        )
        trace = by_message_id.get(trace_key)
        if trace is not None:
            stages = trace.get("stages") if isinstance(trace.get("stages"), list) else []
            usage_stages = [
                stage for stage in stages
                if isinstance(stage, dict) and stage.get("name") == "llm_usage"
            ]
            usage: dict[str, Any] = {"calls": 0}
            for stage in usage_stages:
                usage["calls"] += max(0, int(stage.get("calls") or 0))
                for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens"):
                    if key in stage:
                        usage[key] = int(usage.get(key, 0)) + max(0, int(stage.get(key) or 0))
                channel = str(stage.get("channel") or "").strip()
                if channel:
                    channels = usage.setdefault("channels", {})
                    channels[channel] = int(channels.get(channel, 0)) + max(0, int(stage.get("calls") or 0))
            item["turn"] = {
                "trace_id": trace.get("trace_id", ""),
                "status": trace.get("status", ""),
                "agent_status": trace.get("agent_status", ""),
                "sent": bool(trace.get("sent")),
                "delivery_status": trace.get("delivery_status", ""),
                "reason": trace.get("reason", ""),
                "elapsed_ms": trace.get("elapsed_ms"),
                "llm_usage": usage if usage_stages else {"available": False},
            }
        elif not bool((item.get("attention") or {}).get("should_schedule")):
            item["turn"] = {"status": "not_scheduled", "llm_usage": {"calls": 0}}
        else:
            item["turn"] = {"status": "pending_or_trace_expired", "llm_usage": {"available": False}}
        items.append(item)
    return {
        "available": True,
        "summary": payload.get("summary", {}),
        "items": items,
    }


def _memory_routing_snapshot(assembly: Any, *, limit: int = 60) -> dict[str, Any]:
    """Expose bounded memory-routing metadata without query or message text."""

    bridge = getattr(assembly, "bridge", None)
    memory_service = getattr(bridge, "memory_service", None)
    store = getattr(memory_service, "store", None)
    reader = getattr(store, "recent_routing_audits", None)
    if not callable(reader):
        return {"available": False, "summary": {}, "items": []}
    try:
        rows = reader(limit=max(1, min(100, int(limit or 60))))
    except Exception:
        return {"available": False, "summary": {}, "items": []}
    if not isinstance(rows, list):
        return {"available": False, "summary": {}, "items": []}

    safe_reasons = {
        "empty_message",
        "forget_command",
        "explicit_summary_request",
        "explicit_recall_request",
        "obvious_short_chat",
        "current_task_request",
        "local_recent_topic",
        "local_summary_intent",
        "local_no_history_signal",
        "no_memory_judge_configured",
        "memory_judge_error",
        "memory_router_fallback",
        "model_memory_judgement",
    }
    items: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    model_judged = 0
    retrieved = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        mode = str(row.get("history_mode") or "none").strip().lower()
        if mode not in {"none", "recent", "retrieve", "summary"}:
            mode = "none"
        used_model = bool(row.get("used_model"))
        raw_reason = str(row.get("reason") or "")
        reason = raw_reason if raw_reason in safe_reasons else ("model_judgement" if used_model else "local_decision")
        signals = decision.get("local_signals")
        if not isinstance(signals, (list, tuple)):
            signals = ()
        safe_signals = [
            str(value)[:48]
            for value in signals[:8]
            if str(value).replace("_", "").isalnum()
        ]
        item = {
            "audit_id": str(row.get("audit_id") or "")[:80],
            "conversation_id": str(row.get("conversation_id") or "")[:160],
            "scope": str(row.get("scope") or "")[:160],
            "created_at": max(0, int(row.get("created_at") or 0)),
            "history_mode": mode,
            "need_retrieval": bool(row.get("need_retrieval")),
            "used_model": used_model,
            "confidence": max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
            "reason": reason,
            "local_signals": safe_signals,
            "retrieved_count": max(0, int(row.get("retrieved_count") or 0)),
            "injected_chars": max(0, int(row.get("injected_chars") or 0)),
        }
        items.append(item)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        model_judged += int(used_model)
        retrieved += int(item["retrieved_count"] > 0)
    return {
        "available": True,
        "summary": {
            "window_items": len(items),
            "mode_counts": mode_counts,
            "model_judged": model_judged,
            "with_results": retrieved,
        },
        "items": items,
    }


def _model_services_snapshot(controller: Any) -> dict[str, Any]:
    if controller is None:
        return {"available": False, "status": "not_attached", "providers": []}
    try:
        from ..providers.catalog import public_model_services_snapshot

        snapshot = public_model_services_snapshot(
            controller.store,
            controller.config_module,
            source="local_control_center",
        )
        snapshot["available"] = True
        return snapshot
    except Exception as exc:
        return {
            "available": True,
            "status": "error",
            "reason": str(exc)[:240],
            "providers": [],
        }
