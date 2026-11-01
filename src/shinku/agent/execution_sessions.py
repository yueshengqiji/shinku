"""Executor-authenticated lifecycle receipts and bounded terminal cleanup.

The optional hooks belong to registered executors. Model-provided session IDs
and status strings alone cannot enable automatic polling or exempt a stall.
"""
from __future__ import annotations

from collections.abc import Mapping

EVENT_KINDS = frozenset({"waiting", "output", "command_completed", "artifact_changed",
    "execution_error", "deadline_exceeded", "cancelled", "interrupted"})


def execution_event(handler, receipt, context):
    hook = getattr(handler, "agent_execution_event", None)
    if not callable(hook) or not isinstance(receipt, Mapping):
        return None
    session = receipt.get("execution_session")
    if not isinstance(session, Mapping):
        return None
    try:
        event = hook(receipt=receipt, context=context)
    except Exception:
        return None
    if not isinstance(event, Mapping) or event.get("kind") not in EVENT_KINDS:
        return None
    if (not session.get("session_id") or not session.get("command_sha256")
            or event.get("session_id") != session.get("session_id")
            or event.get("command_sha256") != session.get("command_sha256")
            or type(event.get("cursor")) is not int or event["cursor"] < 0
            or event["cursor"] != session.get("cursor")):
        return None
    allowed = {"running": {"waiting", "output"}, "completed": {"command_completed", "artifact_changed"},
        "failed": {"execution_error"}, "timed_out": {"deadline_exceeded"}, "cancelled": {"cancelled"},
        "interrupted": {"interrupted"}}
    if event["kind"] not in allowed.get(session.get("status"), set()):
        return None
    return {key: event[key] for key in ("kind", "session_id", "command_sha256", "cursor")}


def stop_execution_sessions(state, handlers, reason):
    """Stop scoped live executors, recording uncertainty without new execution."""
    latest = {}
    for call in state.tool_calls:
        data = call.get("data", {})
        receipt = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
        if not isinstance(receipt, Mapping):
            continue
        session = receipt.get("execution_session", {})
        if isinstance(session, Mapping) and session.get("session_id"):
            latest[(call.get("name"), session["session_id"])] = receipt
    reports = []
    for (name, session_id), receipt in latest.items():
        if receipt["execution_session"].get("status") != "running":
            continue
        report = {"tool": name, "session_id": session_id, "status": "unknown"}
        hook = getattr(handlers.get(name), "agent_stop_session", None)
        if callable(hook):
            try:
                stopped = hook(receipt=receipt, context=state.context, reason=reason)
                if (isinstance(stopped, Mapping) and stopped.get("session_id") == session_id
                        and stopped.get("status") in {"stopped", "already_stopped", "unknown", "failed"}):
                    report["status"] = stopped["status"]
            except Exception:
                pass
        reports.append(report)
    if reports:
        state.progress["execution_cleanup"] = reports
    # 浏览器上下文等资源不伪装成正在执行的 Shell 会话。只调用实际注册工具的
    # 终态释放钩子，按共享资源所有者去重；不在分片 yield/等待用户时关闭页面。
    released, resources = set(), []
    for name in dict.fromkeys(call.get('name') for call in state.tool_calls):
        handler = handlers.get(name)
        hook = getattr(handler, 'agent_release_task', None)
        owner = getattr(handler, 'agent_resource_owner', handler)
        if not callable(hook) or id(owner) in released:
            continue
        released.add(id(owner))
        report = {'tool': name, 'status': 'unknown'}
        try:
            result = hook(context=state.context, reason=reason)
            if isinstance(result, Mapping) and result.get('status') == 'released':
                report['status'] = 'released'
        except Exception:
            pass
        resources.append(report)
    if resources:
        state.progress['resource_cleanup'] = resources
