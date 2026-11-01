"""Host read-only task inquiries and fresh operational prompt projections."""
from __future__ import annotations

import re

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.task_view import TaskViewReader


_QUERY = re.compile(
    r"(?:任务)?(?:现在)?(?:进度(?:怎么样|如何)?|进展(?:怎么样|如何)?|状态(?:怎么样|如何)?|"
    r"做到(?:哪里|哪)(?:了)?|做(?:完|好)了(?:吗|没)|完成了(?:吗|没)|"
    r"还在(?:执行|运行|做)(?:吗)?|执行到哪(?:里)?(?:了)?|结果(?:怎么样|如何|出来了(?:吗|没)))",
    re.IGNORECASE,
)
_ID = re.compile(r"(?:任务号|任务编号|task_id)\s*[:：]?\s*([A-Za-z0-9_.:-]{1,160})", re.IGNORECASE)


def parse_status_query(text):
    text = str(text).strip()
    target = _ID.search(text)
    task_id = target.group(1) if target else ""
    if target:
        text = text[:target.start()] + text[target.end():]
    text = re.sub(r"[\s？?！!。，,]+", "", text).removeprefix("的")
    return {"task_id": task_id} if _QUERY.fullmatch(text) else None


def reader_for_bridge(bridge):
    def receipts(scope):
        from shinku.conversation import ConversationScope
        kind, channel = scope.session.split(":", 1)
        return bridge.conversation_runtime.task_receipts(ConversationScope(kind, channel, scope.requester))
    return TaskViewReader(bridge.task_store, bridge.task_coordinator, delivery_reader=receipts)


def task_operations(runtime, scope):
    values = runtime.task_receipts(scope)
    if not values.get("available"):
        return {"available": False, "records": [], "reason": "conversation_state_unavailable"}
    rows = [r for r in values["records"] if not r.get("task_id") and r.get("route") == "agent"
            and r.get("status") not in {"chat", "completed", "continued", "ignored", "silent"}]
    return {"available": True, "records": [{key: row[key] for key in (
        "turn_id", "status", "reason", "phase", "delivery_status", "sent", "started_at", "updated_at", "lease_until")}
        | {"task_identity_missing": True, "source": "conversation_runtime"} for row in rows[:4]]}


def operational_context(bridge, turn, exclude_turn):
    scope = bridge._conversation_scope(turn)
    shared = bridge.conversation_runtime.prompt_context(scope, exclude_turn=exclude_turn)
    requester, session = bridge._task_scope(turn)
    tasks = reader_for_bridge(bridge).candidates(AgentScope(requester, session))
    operations = task_operations(bridge.conversation_runtime, scope)
    return {**shared, "task_views": tasks["tasks"], "task_views_available": tasks["available"],
        "task_views_reason": tasks.get("reason", ""), "task_operations": [
            r for r in operations["records"] if r["turn_id"] != exclude_turn],
        "task_operations_available": operations["available"]}


def status_text(view):
    state = view["status"]
    label = {"completed": "整项任务验收通过", "unverified": "执行记录尚未通过当前任务的完整验收",
        "running": "正在执行", "queued": "正在排队", "waiting_user": "等待你的确认",
        "waiting_background": "等待后台执行", "partial": "只完成了部分要求",
        "failed": "执行失败", "blocked": "执行受阻", "canceled": "执行已取消",
        "conflict": "任务来源记录存在冲突，当前状态不能确认", "unknown": "当前状态无法确认",
        "interrupted": "执行已中断"}.get(state, "当前状态无法确认")
    pieces = [f"任务 {view['task_id']}：{label}。"]
    missing = view["verification"]["missing"]
    if missing:
        pieces.append("未完成或未核验：" + "；".join(missing[:4]) + "。")
    if view.get("clarification", {}).get("required"):
        pieces.append("必要资料尚未完整交接，需要补读或补充原文；查询状态没有启动补读或继续执行。")
    delivery = view["delivery"]["status"]
    pieces.append({"sent": "已有实际送达回执。", "failed": "结果送达失败，执行验收结论仍单独保留。",
        "dry_run": "使用测试发送端，未实际送达。", "uncertain": "送达结果尚未确认。",
        "partial": "只确认部分送达。", "suppressed": "结果发送已暂停。",
        "queued": "结果等待发送。"}.get(delivery, "尚无确认送达的回执。"))
    event = view.get("result_event") or {}
    if event.get("status") in {"pending", "processing"}:
        pieces.append("执行结果已保存，后续表达或送达仍待确认。")
    elif event.get("status") in {"failed", "expired"} and delivery in {"not_sent", "not_recorded"}:
        pieces.append("结果后续处理未成功，保留的记录需要核实。")
    if view["execution"]["cancel_requested"] and state != "canceled":
        pieces.append("已登记取消请求，实际停止仍待确认；取消不会自动回滚产物。")
    return "\n".join(pieces)


def route_task_status(bridge, turn, trace_id):
    if bridge._is_poke_turn(turn) or bridge._can_silence_autonomous_turn(turn):
        return None
    parsed = parse_status_query(bridge._primary_task_request(turn))
    if parsed is None:
        return None
    requester, session = bridge._task_scope(turn)
    scope = AgentScope(requester, session)
    primary = next((m for m in turn.messages if str(m.message_id) == turn.primary_message_id), None)
    references = getattr(primary, "reply_message_ids", ()) if primary else ()
    selection = reader_for_bridge(bridge).select(scope, task_id=parsed["task_id"], message_ids=references)
    # A newer request can fail before it has a task ID. It must not silently
    # inherit the status of an older completed task in the same conversation.
    if not parsed["task_id"] and not references:
        operations = task_operations(bridge.conversation_runtime, bridge._conversation_scope(turn))
        if not operations["available"]:
            selection = {"available": False, "reason": "conversation_state_unavailable"}
        elif operations["records"]:
            latest = max(r["started_at"] for r in operations["records"])
            if selection["available"]:
                bound_start = selection["task"]["delivery"].get("started_at", selection["task"]["updated"])
                if latest >= bound_start:
                    selection = {"available": False, "reason": "task_admission_unresolved"}
    if selection["available"]:
        text, reason = status_text(selection["task"]), "task_status_query"
    else:
        reason = selection["reason"]
        if reason == "task_admission_unresolved":
            text = "最近的一项请求尚未创建业务任务，准入或准备阶段没有完成；没有重新执行。若要查询旧任务，请指明任务号。"
        elif reason == "task_binding_not_found" and not parsed["task_id"] and not references:
            operations = task_operations(bridge.conversation_runtime, bridge._conversation_scope(turn))
            if not operations["available"]:
                text, reason = "当前无法读取任务的操作记录，暂时不能确认进展。", "conversation_state_unavailable"
            elif operations["records"]:
                text, reason = "上一项请求尚未创建业务任务，准入或准备阶段没有完成；没有重新执行。", "task_admission_unresolved"
            else:
                return None
        elif reason == "task_binding_ambiguous":
            choices = "\n".join(f"{r['task_id']}：{r['goal'][:100]}" for r in selection["candidates"])
            prefix = "当前还有尚未创建业务任务的请求。" if not parsed["task_id"] and not references and operations["records"] else ""
            text = prefix + "当前有多项任务，需要指出任务号或引用原始请求：\n" + choices
        elif reason in {"task_not_found", "task_binding_not_found"}:
            text = "当前请求者和会话范围内未找到对应任务，无法确认它的状态。"
        else:
            text = "当前无法读取可靠的任务状态，暂时不能确认进展。"
    # Its own chat operation must not overwrite the original task receipt.
    bridge._conversation_step(trace_id, route="chat", phase="query")
    bridge._trace(trace_id, "task_status_query", status=reason)
    return bridge._deliver_text(turn, text, agent_status="chat", reason=reason,
        trace_id=trace_id, record_memory=False, track_chat_flow=False)
