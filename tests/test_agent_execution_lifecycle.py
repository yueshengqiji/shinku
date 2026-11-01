"""Independent lifecycle rules; no QQ, private extensions or live model calls."""
from copy import deepcopy

import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.completion import CompletionVerdict
from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation


class SessionHandler:
    agent_progress_class = "change"

    def __init__(self, quiet_polls=8, output=False):
        self.quiet_polls = quiet_polls
        self.output = output
        self.starts = self.polls = 0
        self.running = False
        self.stops = []
        self.receipt = None

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        if call.get("action") == "poll":
            self.polls += 1
        else:
            self.starts += 1
            self.running = True
        complete = self.polls > self.quiet_polls
        self.running = not complete
        cursor = self.polls if self.output else 0
        self.receipt = {"execution_session": {"session_id": "fixture-session", "command_sha256": "fixture-command",
            "status": "completed" if complete else "running", "cursor": cursor, "returncode": 0 if complete else None},
            "verified_effects": ["command_run"] if complete else [], "stdout": f"row {cursor}" if self.output else ""}
        return {"tool_type": "session", "followup_context": "executor receipt", "state_updates": deepcopy(self.receipt)}

    def agent_execution_event(self, *, receipt, context):
        if receipt != self.receipt:
            return None
        session = receipt["execution_session"]
        return {"kind": "command_completed" if session["status"] == "completed" else "output" if self.output else "waiting",
            "session_id": session["session_id"], "command_sha256": session["command_sha256"], "cursor": session["cursor"]}

    def agent_resume_session(self, *, receipt, context):
        return ToolInvocation("session", {"action": "poll", "session_id": "fixture-session", "cursor": receipt["execution_session"]["cursor"]})

    def agent_stop_session(self, *, receipt, context, reason):
        self.running = False
        self.stops.append(reason)
        return {"status": "stopped", "session_id": "fixture-session"}


def make_loop(handler, *, rounds=3, sliced=True, planner=None, **extra):
    return AgentLoop(planner=planner or (lambda **_: AgentDecision.final("command ended", outcome="completed")),
        handlers={"session": handler}, policy=ExecutionPolicy(risk_by_tool={"session": "low"}),
        max_rounds=rounds, sliced=sliced, **extra)


def start():
    return AgentDecision.tool(ToolInvocation("session", {"action": "start", "command": "fixture"}))


def test_r01_r03_eight_quiet_polls_cross_slices_without_replanning_or_stall():
    handler = SessionHandler()
    plans = []
    loop = make_loop(handler, planner=lambda **_: plans.append(True) or AgentDecision.final("command ended"))
    budget = TaskBudget(max_tools=20)
    checkpoint = None
    statuses = []
    with bind_task_budget(budget):
        for slice_index in range(5):
            result = loop.run(task_id="fixture-task", context={"original_goal": "unchanged"}, checkpoint=checkpoint,
                first_decision=start() if slice_index == 0 else None)
            statuses.append(result.status)
            assert result.progress["stalled"] == 0
            assert len(result.progress["fingerprints"]) <= 2
            if result.status != "yielded":
                break
            checkpoint = result.checkpoint
            assert checkpoint.state.context["original_goal"] == "unchanged"
            assert handler.running and not handler.stops
    assert statuses.count("yielded") >= 2
    assert result.status == "completed"
    assert handler.starts == 1 and handler.polls == 9 and len(plans) == 1
    assert budget.tools == 10 and budget.active_seconds > 0
    assert result.progress["last_event"]["execution_event"] == "command_completed"


def test_r02_output_advances_evidence_without_completing_task():
    handler = SessionHandler(output=True)
    result = make_loop(handler).run(first_decision=start())
    assert result.status == "yielded" and not result.completion
    assert result.progress["last_event"]["execution_event"] == "output"
    assert result.progress["last_event"]["new_evidence"]
    assert handler.starts == 1 and handler.polls == 2


@pytest.mark.parametrize("quiet_polls", [1, 4, 7, 10])
def test_completed_executor_on_last_slice_round_waits_for_task_acceptance(quiet_polls):
    handler = SessionHandler(quiet_polls=quiet_polls)
    plans, checks = [], []

    def verify(state, text):
        checks.append(state.rounds)
        assert handler.receipt["execution_session"]["status"] == "completed"
        assert handler.receipt["execution_session"]["returncode"] == 0
        assert handler.receipt["verified_effects"] == ["command_run"]
        return CompletionVerdict(True, satisfied=("fixture-command",),
            evidence=({"kind": "executor_receipt", "status": "completed"},))

    loop = make_loop(handler, completion_gate=verify,
        planner=lambda **_: plans.append(True) or AgentDecision.final("command ended", outcome="completed"))
    budget = TaskBudget(max_tools=20)
    with bind_task_budget(budget):
        result = loop.run(task_id="fixture-task", context={"original_goal": "unchanged"}, first_decision=start())
        for _ in range(8):
            if not handler.running:
                break
            assert result.status == "yielded"
            event = result.progress["last_event"]
            assert event["execution_event"] == "waiting" and event["waiting"]
            assert not event["new_evidence"] and not event["progress"]
            assert result.progress["stalled"] == 0
            result = loop.run(checkpoint=result.checkpoint)

        assert not handler.running
        assert result.status == "yielded" and result.reason == "slice_complete"
        assert result.rounds == quiet_polls + 2 and result.rounds % 3 == 0
        event = result.progress["last_event"]
        assert event["execution_event"] == "command_completed" and not event["waiting"]
        assert event["new_evidence"] and event["progress"] and not event["verified_change"]
        assert result.progress["stalled"] == 0 and not result.completion
        assert not plans and not checks
        assert result.checkpoint.state.context["original_goal"] == "unchanged"
        assert handler.starts == 1 and handler.polls == quiet_polls + 1
        assert budget.tools == quiet_polls + 2
        result = loop.run(checkpoint=result.checkpoint)

    assert result.status == "completed" and result.completion["accepted"]
    assert result.rounds == quiet_polls + 3 and len(plans) == len(checks) == 1
    assert handler.starts == 1 and handler.polls == quiet_polls + 1 and not handler.stops
    assert budget.tools == quiet_polls + 2


@pytest.mark.parametrize("reason", ["task_canceled", "agent_permission_changed", "task_time_budget_exhausted"])
def test_r05_terminal_interrupt_stops_session_without_resetting_budget(reason):
    handler = SessionHandler()
    loop = make_loop(handler, interruption_check=lambda _: reason if handler.polls >= 2 else "")
    result = loop.run(first_decision=start())
    assert result.reason == reason
    assert not handler.running and handler.stops == [reason]
    assert result.progress["execution_cleanup"][0]["status"] == "stopped"


def test_r05_round_exhaustion_is_terminal_and_stops_remaining_process():
    handler = SessionHandler()
    result = make_loop(handler, sliced=False).run(first_decision=start())
    assert result.reason == "max_rounds" and not handler.running
    assert handler.stops == ["max_rounds"]


def test_r07_forged_receipt_does_not_authorize_automatic_poll():
    handler = SessionHandler()
    handler.agent_execution_event = lambda **_: None
    plans = []
    result = make_loop(handler, planner=lambda **_: plans.append(True) or AgentDecision.final("not verified"),
        completion_gate=lambda **_: False).run(first_decision=start())
    assert handler.polls == 0 and plans
    assert result.status != "completed"


def test_r11_poll_budget_exhaustion_cleans_up_and_preserves_counters():
    handler = SessionHandler()
    budget = TaskBudget(max_tools=2)
    with bind_task_budget(budget):
        result = make_loop(handler, rounds=6).run(first_decision=start())
    assert result.reason == "task_tool_budget_exhausted"
    assert budget.tools == 2 and handler.starts == 1 and handler.polls == 1
    assert not handler.running


def test_failed_cleanup_remains_unknown_without_relabeling_execution():
    handler = SessionHandler()
    def fail(**kwargs):
        raise RuntimeError("private diagnostic must not escape")
    handler.agent_stop_session = fail
    result = make_loop(handler, sliced=False).run(first_decision=start())
    assert result.reason == "max_rounds"
    assert result.progress["execution_cleanup"][0]["status"] == "unknown"
    assert "private diagnostic" not in str(result)


def test_wait_preserves_prior_failures_instead_of_resetting_stall_counter():
    handler = SessionHandler()
    context = {"agent_task_state": {"progress": {"stalled": 3}}}
    result = make_loop(handler).run(context=context, first_decision=start())
    assert result.status == "yielded"
    assert result.progress["stalled"] == 3
    assert not result.progress["last_event"]["progress"]
    assert not result.progress["last_event"]["new_evidence"]


def test_repeating_output_cursor_does_not_create_new_evidence():
    from shinku.agent.progress import ProgressLedger
    from shinku.tools.invocation import ToolResultEnvelope
    ledger = ProgressLedger(stalled=2)
    event = {"kind": "output", "session_id": "same", "command_sha256": "same", "cursor": 1}
    receipt = ToolResultEnvelope("first", "ok", "line")
    call = ToolInvocation("session", {"action": "poll"})
    assert ledger.observe(call, receipt, progress_class="change", execution_event=event)["progress"]
    repeated = ledger.observe(call, receipt, progress_class="change", execution_event=event)
    assert not repeated["progress"] and not repeated["new_evidence"] and repeated["stalled"] == 0


def test_autopoll_still_checks_execution_policy_before_handler():
    handler = SessionHandler()
    class RevokingPolicy(ExecutionPolicy):
        def accepts(self, name):
            return handler.starts == 0
    policy = RevokingPolicy(risk_by_tool={"session": "low"})
    loop = make_loop(handler, completion_gate=lambda **_: False)
    loop.policy = policy
    result = loop.run(first_decision=start())
    assert handler.starts == 1 and handler.polls == 0
    assert result.status != "completed"


def test_waiting_time_consumes_hard_monotonic_budget(monkeypatch):
    import shinku.agent.budget as budget_module
    clock = [100.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: clock[0])
    handler = SessionHandler()
    original = handler.execute
    def advance(**kwargs):
        clock[0] += 1
        return original(**kwargs)
    handler.execute = advance
    budget = TaskBudget(max_active_seconds=4)
    with bind_task_budget(budget):
        result = make_loop(handler, rounds=6).run(first_decision=start())
    assert result.reason == "task_time_budget_exhausted"
    assert budget.active_seconds == 4 and budget.tools == 4
    assert handler.starts == 1 and not handler.running


def test_same_failed_call_budget_survives_semantic_resume():
    handler = SessionHandler()
    def failing(**kwargs):
        return {"status": "error", "message": "fixture failure"}
    handler.execute = failing
    invocation = ToolInvocation("session", {"action": "start", "command": "fixture"})
    loop = make_loop(handler, rounds=2, planner=lambda **_: AgentDecision.tool(invocation))
    first = loop.run(first_decision=AgentDecision.tool(invocation))
    assert first.status == "yielded"
    second = make_loop(handler, rounds=4, planner=lambda **_: AgentDecision.tool(invocation)).run(
        context={"agent_task_state": {"progress": first.progress}})
    assert second.reason == "repeated_tool_call"
    assert len(second.tool_envelopes) == 1


def test_queued_slice_cancellation_records_cleanup_before_persisting():
    import threading
    from types import SimpleNamespace
    from shinku.agent.execution_sessions import stop_execution_sessions
    from shinku.qq.background_tasks import _cancel_queued_work
    handler = SessionHandler()
    ctx = {"agent_task_state": {"notes": {}}}
    result = make_loop(handler, rounds=1).run(context=ctx, first_decision=start())
    stored = []
    work = SimpleNamespace(session={"checkpoint": result.checkpoint}, work_id="fixture-work", trace_id="fixture-trace", context=ctx)
    deps = SimpleNamespace(slice_lock=threading.RLock(), slice_sessions={"fixture-work": work.session},
        cleanup_checkpoint=lambda state, reason: stop_execution_sessions(state, {"session": handler}, reason),
        task_store=lambda: SimpleNamespace(save=lambda task: stored.append(deepcopy(task))),
        trace_store=lambda: SimpleNamespace(finish=lambda *a, **kw: None), conversation_finish=lambda *a: None)
    _cancel_queued_work(work, deps)
    assert not handler.running and handler.stops == ["task_canceled"]
    assert stored[0]["status"] == "canceled" and stored[0]["progress"]["execution_cleanup"][0]["status"] == "stopped"


def test_only_authenticated_command_attempt_is_excluded_from_durable_job_gate():
    from shinku.agent.loop import AgentState
    from shinku.agent.verification import SmartCompletionVerifier
    receipt = {"execution_session": {"session_id": "command", "command_sha256": "digest", "cursor": 0, "status": "failed"},
        "background_task": {"task_id": "command", "status": "failed", "completed": False}}
    class Executor:
        def agent_execution_event(self, **kwargs):
            return {"kind": "execution_error", "session_id": "command", "command_sha256": "digest", "cursor": 0}
    state = AgentState(transcript=[{"role": "tool", "name": "executor", "data": {"state_updates": receipt}}])
    assert SmartCompletionVerifier(result_handlers={"executor": Executor()})._background_tasks(state) == {}
    assert SmartCompletionVerifier()._background_tasks(state)["command"]["status"] == "failed"
    receipt["background_task"]["task_id"] = "durable-business-job"
    assert "durable-business-job" in SmartCompletionVerifier(result_handlers={"executor": Executor()})._background_tasks(state)
