"""Evidence, discovery and duplicate-call progress for one run/checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .loop_models import AgentLoopDependencies, AgentLoopRun
from .execution_sessions import execution_event


def note_loop_execution(
    run: AgentLoopRun,
    invocation: ToolInvocation,
    envelope: ToolResultEnvelope,
    deps: AgentLoopDependencies,
) -> None:
    handler = deps.handlers().get(invocation.name)
    from shinku.agent_runtime.reply_delivery import delivery_turn
    direct = delivery_turn(run.context) is not None
    prior_candidates = set(run.state.research.candidates)
    if not direct:
        # ResearchLedger.observe 会自行开启研究模式；直接交付不能被一次搜索
        # 悄悄带回候选/答案复核。原工具原文和回执仍由原执行记录完整保留。
        run.state.research.observe(invocation, envelope, handler)
        run.state.local_answer.observe(invocation, envelope, handler, run.state.context)
        run.state.history_answer.observe(envelope, handler)
    _note_source_discovery(run, invocation, envelope, deps)
    receipt = (envelope.data or {}).get("state_updates", {})
    event = run.progress_ledger.observe(invocation, envelope, progress_class=deps.progress_class(invocation),
        candidate_only=not direct and bool(getattr(handler, "agent_research_candidate_field", "")),
        execution_event=execution_event(handler, receipt, run.context))
    if event.get("stage") == "candidate_discovery":
        event["new_candidates"] = bool(set(run.state.research.candidates) - prior_candidates)
    run.state.progress = {**run.state.progress, "stalled": run.progress_ledger.stalled, "last_event": event,
                      "fingerprints": list(run.progress_ledger.seen)[-128:],
                      "source_discovery_calls": run.state.source_discovery_calls,
                      "source_discovery_limit": deps.max_source_discovery_calls()}
    run.state.consecutive_observation_calls = run.progress_ledger.stalled
    if event.get("progress"):
        run.rejected_finals = 0
        if run.saved:
            run.saved.rejected_finals = 0
    if event["obstacle"]:
        run.state.transcript.append({"role": "system", "content":
            f"宿主诊断分类：{event['obstacle']}。依据实际错误选择替代方案；分类不是授权，修复环境不等于目标站可用。"})
    _note_signature_result(run, invocation, envelope, event, deps)
    from .investigation import note_investigation
    note_investigation(run.state, invocation, event)
    deps.observe_state(run.state, "running")


def _note_source_discovery(
    run: AgentLoopRun,
    invocation: ToolInvocation,
    envelope: ToolResultEnvelope,
    deps: AgentLoopDependencies,
) -> None:
    evidence_field = getattr(deps.handlers().get(invocation.name), "agent_evidence_collection_field", "")
    if evidence_field and deps.progress_class(invocation) == "observe":
        run.state.source_discovery_calls += 1
        if run.state.source_discovery_calls >= deps.max_source_discovery_calls():
            run.state.suppressed_tool_names.update(name for name in deps.handlers() if deps.is_source_discovery(name))
            run.state.transcript.append({"role": "system", "content":
                "本轮资料发现额度已用完，但正文读取和其他核验工具仍可用。请评估已有候选来源；有相关来源就读取必要正文，没有相关来源则明确说明缺口，不再扩搜。"})
        rows = (envelope.data or {}).get("state_updates", {}).get(evidence_field, [])
        urls = {str(row.get("url")) for row in rows if isinstance(row, Mapping) and row.get("url")}
        run.search_stalled = 0 if urls - run.search_urls else run.search_stalled + 1
        run.search_urls.update(urls)
        from shinku.agent_runtime.reply_delivery import delivery_turn
        if run.search_stalled >= 2 and run.search_urls and delivery_turn(run.context) is None:
            if run.state.research.enabled:
                run.state.suppressed_tool_names.update(name for name in deps.handlers() if deps.is_source_discovery(name))
                run.state.transcript.append({"role": "system", "content": "连续扩搜没有新增候选，暂停发现工具；已有关联候选可继续读正文，不能把发现停滞当成正文阅读也已结束。"})
            else:
                run.state.finalization_only = True


def _note_signature_result(
    run: AgentLoopRun,
    invocation: ToolInvocation,
    envelope: ToolResultEnvelope,
    event: dict[str, Any],
    deps: AgentLoopDependencies,
) -> None:
    signature = deps.call_signature(invocation)
    failure_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    failures = run.state.progress.setdefault("failure_attempts", {})
    run.duplicate_rejections.pop(signature, None)
    if envelope.status == "ok":
        failures.pop(failure_key, None)
        # The three-attempt cap is for failures, not successful polling
        # with fresh results. Duplicate/no-progress guards still apply.
        run.signature_attempts[signature] = 0
        progress_class = deps.progress_class(invocation)
        if event["verified_change"]:
            run.mutation_epoch += 1
        run.successful_signatures[signature] = (progress_class, run.mutation_epoch)
    else:
        failures[failure_key] = min(3, failures.get(failure_key, 0) + 1)
        run.successful_signatures.pop(signature, None)
    if run.saved:
        run.saved.mutation_epoch, run.saved.search_stalled = run.mutation_epoch, run.search_stalled
