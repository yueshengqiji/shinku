from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.budget_policy import PROFILES, choose_profile, configure_task_budget, grant_budget
from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry
from shinku.hosts.tool_host import ToolHost
from test_agent_async_processor import turn, wait_for


def paused_task(store=None):
    task = new_task(task_id="task:paused", requester="owner", session="private:owner",
                    request="Write the file, then verify it", message_id="old", contract=None)
    task["status"] = "blocked"
    task["budget"] = TaskBudget(max_requests=120, max_tokens=600000, max_tools=128,
                                  max_active_seconds=1800, requests=12, tokens=580000, tools=1,
                                  reason="task_token_budget_exhausted").snapshot()
    task["notes"]["blocker"] = "task_token_budget_exhausted"
    task["notes"]["remaining"] = ["Verify the written file"]
    task["tool_calls"] = [{"name": "write", "arguments": {"path": "fixture.txt"}, "status": "ok"}]
    return store.save(task) if store else task


def test_allocations_classify_work_and_obey_configured_ceiling():
    assert choose_profile({"requirements": [{"effect": "read", "targets": ["a"]}]}) == "standard"
    assert choose_profile({"requirements": [{"effect": "write"}]}) == "standard"
    assert choose_profile({"requirements": [{"effect": "write"}] * 4}) == "heavy"
    task = paused_task()
    task.pop("budget")
    task["contract"] = {"requirements": [{"effect": "write"}] * 4}
    budget = TaskBudget(requests=1, tokens=200)
    configure_task_budget(budget, task, {"max_tokens": 750000})
    assert (budget.profile, budget.max_tokens, budget.requests, budget.tokens) == ("heavy", 750000, 1, 200)


def test_old_stop_cannot_be_cleared_by_natural_continuation():
    task = paused_task()
    budget = TaskBudget(tokens=100)
    configure_task_budget(budget, task, PROFILES["heavy"])
    assert budget.reason == "task_token_budget_exhausted"
    assert (budget.tokens, budget.max_tokens) == (580100, 600000)


def test_explicit_grant_retains_counters_contract_and_receipts_after_restart(tmp_path):
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    original = paused_task(store)
    task = copy.deepcopy(original)
    grant_budget(task, profile="standard", ceilings=PROFILES["heavy"], request_id="grant1", requester="owner")
    store.save(task)
    store.close()
    restored = AgentTaskStore(tmp_path / "tasks.sqlite3").get(task["task_id"], requester="owner", session="private:owner")
    budget = TaskBudget(requests=1, tokens=123)
    configure_task_budget(budget, restored, PROFILES["heavy"])
    assert (budget.requests, budget.tokens, budget.tools) == (13, 580123, 1)
    assert budget.max_tokens == 4600000 and not budget.reason
    assert restored["tool_calls"] == original["tool_calls"]
    assert restored["requests"] == original["requests"]
    restored["budget"]["reason"] = "task_budget_low"
    with pytest.raises(ValueError, match="already_applied"):
        grant_budget(restored, profile="heavy", ceilings=PROFILES["heavy"], request_id="grant1", requester="owner")


@pytest.mark.parametrize("change,expected", [
    ({"status": "completed"}, "not_budget_paused"),
    ({"inflight": [{"name": "write"}]}, "uncertain_execution"),
    ({"uncertain_attempts": [{"name": "write"}]}, "uncertain_execution"),
])
def test_grant_rejects_completed_or_uncertain_task(change, expected):
    task = paused_task(); task.update(change)
    with pytest.raises(ValueError, match=expected):
        grant_budget(task, profile="standard", ceilings=PROFILES["heavy"], request_id="g", requester="owner")


def test_model_notes_cannot_grant_and_invalid_allocations_fail():
    from shinku.agent.task_state import update_notes
    task = paused_task()
    update_notes(task, {"max_tokens": 999999999, "budget_allocation": {"limits": {}}})
    assert task["budget"]["max_tokens"] == 600000
    task["budget_allocation"] = {"profile": "heavy", "limits": {**PROFILES["heavy"], "max_tokens": float("nan")}}
    with pytest.raises(ValueError):
        configure_task_budget(TaskBudget(), task, PROFILES["heavy"])


def test_low_budget_remains_advisory_and_allows_completion():
    budget = TaskBudget(max_requests=10, requests=9, managed=True)
    def planner(*, state):
        assert not state.finalization_only
        assert any("warning" in row["content"] for row in state.transcript)
        return AgentDecision.final("All steps verified.")
    with bind_task_budget(budget):
        result = AgentLoop(planner=planner, handlers={}, sliced=True).run(context={})
    assert result.status == "completed"
    assert budget.reason == ""


def test_large_previous_request_does_not_force_closeout():
    budget = TaskBudget(max_tokens=600000, tokens=430000, requests=15, last_reservation=100000, managed=True)
    assert budget.stage() == "warning"
    assert not budget.reason


class Writer:
    tool_type = "write"
    agent_permission_class = "write"
    agent_progress_class = "change"
    description = "fixture writer"
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    def __init__(self): self.calls = []
    def normalize_call(self, call): return call
    def execute(self, *, call, context):
        self.calls.append(call)
        return {"followup_context": "written"}


def test_restored_successful_mutation_cannot_be_replayed():
    writer = Writer()
    prior = [{"name": "write", "arguments": {"path": "fixture.txt"}, "status": "ok"}]
    with bind_task_budget(TaskBudget(managed=True)):
        result = AgentLoop(planner=lambda **_: AgentDecision.tool(ToolInvocation("write", {"path": "fixture.txt"})),
            handlers={"write": writer}, sliced=True).run(
                context={"agent_budget_resume": True, "agent_restored_tool_calls": prior})
    assert result.reason == "repeated_tool_call" and not writer.calls


def bridge_fixture(tmp_path):
    coordinator = AgentTaskCoordinator(max_workers=1)
    class Runtime:
        def __init__(self): self.calls = 0
        def chat_supports_native_tools(self): return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            return {"kind": "final", "text": "The saved evidence was reviewed.", "task_outcome": "partial"}
    runtime = Runtime()
    state = QQCommandState(owner_ids=("owner",), permission_mode="full")
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", command_state=state,
        master_user_ids=("owner",), task_coordinator=coordinator, async_agent_enabled=True,
        adaptive_budget_enabled=True, task_budget_limits=PROFILES["heavy"],
        task_store=AgentTaskStore(tmp_path / "tasks.sqlite3"), tool_host=ToolHost(ToolRegistry({})),
        completion_gate=lambda state, text: True)
    paused_task(bridge.task_store)
    return bridge, coordinator, runtime


def test_owner_control_adds_budget_and_resumes_same_task_without_routing_guess(tmp_path):
    bridge, coordinator, runtime = bridge_fixture(tmp_path)
    try:
        result = bridge.handle_turn(turn("/agent resume task:paused standard"))
        assert result.reason == "agent_task_queued"
        record = wait_for(coordinator, result.background_task_id, "blocked")
        saved = bridge.task_store.get("task:paused", requester="owner", session="private:owner")
        assert record["metadata"]["agent_task_id"] == "task:paused"
        assert saved["budget"]["max_tokens"] == 4600000
        assert saved["budget"]["tokens"] >= 580000
        assert len(saved["budget_allocation"]["grants"]) == 1
        assert len(saved["tool_calls"]) == 1 and runtime.calls == 1
    finally:
        coordinator.shutdown(wait=True)


def test_other_user_cannot_top_up_or_read_task(tmp_path):
    bridge, coordinator, runtime = bridge_fixture(tmp_path)
    try:
        request = turn("/agent resume task:paused heavy", sender="other")
        result = bridge.handle_turn(request)
        assert result.reason == "slash_command_forbidden" and runtime.calls == 0
        assert "budget_allocation" not in bridge.task_store.get("task:paused", requester="owner", session="private:owner")
    finally:
        coordinator.shutdown(wait=True)


def test_queue_rejection_does_not_commit_credit_or_clear_stop(tmp_path, monkeypatch):
    bridge, coordinator, runtime = bridge_fixture(tmp_path)
    before = bridge.task_store.get("task:paused", requester="owner", session="private:owner")
    monkeypatch.setattr(coordinator, "submit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("queue_full")))
    try:
        result = bridge.handle_turn(turn("/agent resume task:paused heavy"))
        assert result.reason == "background_task_admission_failed"
        after = bridge.task_store.get("task:paused", requester="owner", session="private:owner")
        assert after == before and runtime.calls == 0
    finally:
        coordinator.shutdown(wait=True)


def test_stale_revision_does_not_overwrite_new_checkpoint(tmp_path, monkeypatch):
    bridge, coordinator, runtime = bridge_fixture(tmp_path)
    original_prepare = bridge._prepare_agent_task
    def race(*args, **kwargs):
        task = bridge.task_store.get("task:paused", requester="owner", session="private:owner")
        task["notes"]["remaining"] = ["Newer checkpoint"]
        bridge.task_store.save(task)
        return original_prepare(*args, **kwargs)
    monkeypatch.setattr(bridge, "_prepare_agent_task", race)
    try:
        result = bridge.handle_turn(turn("/agent resume task:paused heavy"))
        assert result.reason == "task_state_conflict"
        after = bridge.task_store.get("task:paused", requester="owner", session="private:owner")
        assert after["notes"]["remaining"] == ["Newer checkpoint"]
        assert "budget_allocation" not in after and runtime.calls == 0
    finally:
        coordinator.shutdown(wait=True)


def test_lifetime_ceiling_does_not_reset_usage():
    task = paused_task()
    task["budget_allocation"] = {"profile": "heavy", "limits": {
        "max_requests": 1000, "max_tokens": 9000000, "max_tools": 1024, "max_active_seconds": 86400}, "grants": []}
    before = copy.deepcopy(task)
    with pytest.raises(ValueError, match="lifetime_limit"):
        grant_budget(task, profile="light", ceilings=PROFILES["heavy"], request_id="g", requester="owner")
    assert task == before


def test_budget_report_preserves_remaining_conditions_without_success_claim():
    from shinku.qq.budget_report import budget_stop_report
    task = paused_task()
    report = budget_stop_report(task, "task_budget_low", "work:1")
    assert "Verify the written file" in report
    assert "不代表所有要求已完成" in report and "/agent resume work:1 standard" in report


def test_fingerprint_blocks_replay_when_argument_body_was_redacted():
    writer = Writer()
    arguments = {"path": "fixture.txt", "content": "body" * 200}
    signature = 'write:' + json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prior = [{"name": "write", "arguments": {"path": "fixture.txt", "content": "[omitted]"}, "status": "ok",
              "call_fingerprint": hashlib.sha256(signature.encode()).hexdigest()}]
    with bind_task_budget(TaskBudget(managed=True)):
        result = AgentLoop(planner=lambda **_: AgentDecision.tool(ToolInvocation("write", arguments)),
            handlers={"write": writer}, sliced=True).run(context={"agent_budget_resume": True, "agent_restored_tool_calls": prior})
    assert result.reason == "repeated_tool_call" and not writer.calls


def test_light_cap_and_soft_pause_survive_fresh_budget_merge():
    task = paused_task();task.pop("budget")
    task["contract"] = {"requirements": [{"effect": "read", "targets": ["fixture.txt"]}]}
    task["budget_allocation"] = {"profile": "light", "limits": dict(PROFILES["light"]), "grants": []}
    budget = TaskBudget()
    configure_task_budget(budget, task, PROFILES["heavy"])
    assert budget.max_tokens == 200000
    budget.tokens = 175000;budget.reason = "task_budget_low"
    task["budget"] = budget.snapshot()
    resumed = TaskBudget()
    configure_task_budget(resumed, task, PROFILES["heavy"])
    assert resumed.reason == "task_budget_low" and resumed.max_tokens == 200000


def test_heavy_execution_can_cross_old_128_tool_limit():
    from test_agent_slices import Handler
    handler = Handler()
    def planner(*, state):
        return (AgentDecision.tool(ToolInvocation("read", {"n": state.rounds}))
                if state.rounds <= 140 else AgentDecision.final("verified"))
    budget = TaskBudget(**PROFILES["heavy"], managed=True, profile="heavy")
    loop = AgentLoop(planner=planner, handlers={"read": handler}, max_rounds=6, sliced=True)
    with bind_task_budget(budget):
        result = loop.run(context={})
        while result.status == "yielded": result = loop.run(checkpoint=result.checkpoint)
    assert result.status == "completed" and budget.tools == 140
    assert handler.calls == list(range(1, 141))
