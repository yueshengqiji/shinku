import pytest

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.progress import ProgressLedger, obstacle
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


class Handler:
    agent_progress_class = "observe"

    def __init__(self, new=True):
        self.calls = 0
        self.new = new

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        self.calls += 1
        return {"followup_context": f"evidence {self.calls if self.new else 0}",
                "state_updates": {"timestamp": self.calls}}


def test_r5_six_distinct_observations_are_allowed():
    handler = Handler()
    def planner(*, state):
        return AgentDecision.final("研究已整理。") if state.rounds == 7 else AgentDecision.tool(ToolInvocation("read", {"n": state.rounds}))
    result = AgentLoop(planner=planner, handlers={"read": handler}, max_rounds=8).run()
    assert result.status == "completed" and handler.calls == 6
    assert result.progress["stalled"] == 0


def test_timestamp_only_change_does_not_count_as_new_evidence():
    handler = Handler(new=False)
    result = AgentLoop(planner=lambda *, state: AgentDecision.tool(ToolInvocation("read", {"n": state.rounds})),
                       handlers={"read": handler}, max_rounds=12).run()
    assert result.reason == "no_progress" and handler.calls == 7


def test_failed_or_noop_mutation_never_resets_stall_budget():
    ledger = ProgressLedger()
    call = ToolInvocation("write", {})
    for n in range(6):
        receipt = ToolResultEnvelope(str(n), "error" if n % 2 else "ok", f"attempt {n}",
                                    {"state_updates": {"changed": False, "verified_effects": ["write"]}})
        assert not ledger.observe(call, receipt, progress_class="change")["progress"]
    assert ledger.stalled == 6


def test_verified_change_resets_stall_and_plain_success_does_not():
    ledger = ProgressLedger()
    call = ToolInvocation("write", {})
    ledger.observe(call, ToolResultEnvelope("1", "ok", "done"), progress_class="change")
    assert ledger.stalled == 1
    event = ledger.observe(call, ToolResultEnvelope("2", "ok", "done", {"state_updates": {"changed": True}}), progress_class="change")
    assert event["verified_change"] and ledger.stalled == 0


def test_evidence_fingerprints_restore_without_model_notes_becoming_proof():
    receipt = ToolResultEnvelope("1", "ok", "fact")
    first = ProgressLedger()
    first.observe(ToolInvocation("read", {}), receipt, progress_class="observe")
    restored = ProgressLedger(first.seen)
    assert not restored.observe(ToolInvocation("read", {}), receipt, progress_class="observe")["progress"]


@pytest.mark.parametrize("text,expected", [
    ("TimeoutError", "timeout"), ("ModuleNotFoundError", "dependency_missing"),
    ("PermissionError", "permission"), ("ConnectionError", "network"),
    ("proxy connection failed", "proxy"), ("HTTP 403", "target_denied"),
    ("bad_args", "invalid_arguments"), ("opaque failure", "unknown"),
])
def test_obstacle_categories(text, expected):
    assert obstacle(ToolResultEnvelope("1", "error", text)) == expected


def test_large_batch_is_rejected_before_any_execution():
    handler = Handler()
    decision = AgentDecision.tools([ToolInvocation("read", {"n": n}) for n in range(129)])
    result = AgentLoop(planner=lambda **_: decision, handlers={"read": handler}).run()
    assert result.reason == "tool_budget_exhausted" and handler.calls == 0


def test_navigation_transport_ok_does_not_hide_target_denial():
    ledger = ProgressLedger()
    receipt = ToolResultEnvelope("1", "ok", "navigation returned", {"state_updates": {
        "http_status": 403, "verified_effects": [], "site_access_ok": False}})
    event = ledger.observe(ToolInvocation("browser", {}), receipt, progress_class="change")
    assert event["obstacle"] == "target_denied" and not event["progress"]


def test_repeatable_observation_with_new_results_is_not_capped_at_three_successes():
    handler = Handler()
    handler.agent_repeatable_observation = True
    def planner(*, state):
        return AgentDecision.final("取得新结果。") if state.rounds == 7 else AgentDecision.tool(ToolInvocation("poll", {"task": "same"}))
    result = AgentLoop(planner=planner, handlers={"poll": handler}, max_rounds=8).run()
    assert result.status == "completed" and handler.calls == 6
