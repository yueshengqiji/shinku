"""Long tasks cross advisory thresholds and slices without manual top-ups."""
import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget, metered_completion
from shinku.agent.budget_policy import PROFILES, configure_task_budget
from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.qq.assembly_config import load_qq_agent_config
from shinku.qq.budget_report import budget_stop_report
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.invocation import ToolInvocation
from test_agent_async_processor import turn, wait_for
import test_agent_slices as slices
from test_agent_slices import Handler


def test_recorded_469787_pause_continues_through_95_percent_and_finishes():
    budget = TaskBudget(max_requests=120, max_tokens=600000, max_tools=128,
                        max_active_seconds=1800, requests=13, tokens=469787,
                        tools=5, managed=True, last_reservation=114103)
    handler = Handler()
    context = {}
    def planner(*, state):
        assert not state.finalization_only
        # Simulate usage-bearing provider requests, without real network calls.
        metered_completion(lambda **_: {"usage": {"total_tokens": 5000}},
                           {"messages": [], "max_tokens": 10000})
        return (AgentDecision.tool(ToolInvocation("read", {"n": state.rounds}))
                if state.rounds <= 23 else AgentDecision.final("All 23 observations verified."))
    loop = AgentLoop(planner=planner, handlers={"read":handler}, sliced=True, max_rounds=3,
                     completion_gate=lambda state,text: len(state.tool_calls)==23,
                     interruption_check=lambda state: budget.stop_reason())
    with bind_task_budget(budget):
        result = loop.run(task_id="same-long-task", context=context)
        slices = 0
        while result.status == "yielded":
            slices += 1
            assert not budget.reason
            assert not context.get("agent_budget_closeout")
            result = loop.run(checkpoint=result.checkpoint)
    assert result.status == "completed" and slices >= 7
    assert handler.calls == list(range(1,24))
    assert (budget.requests,budget.tokens,budget.tools)==(37,589787,28)
    assert not budget.reason


def test_actual_next_request_must_fit_and_does_not_reset_counters():
    budget = TaskBudget(max_tokens=600000,tokens=589787,requests=37,managed=True,
                        last_reservation=10004)
    reached = []
    with bind_task_budget(budget):
        with pytest.raises(TaskBudgetExceeded,match="task_token_budget_exhausted"):
            metered_completion(lambda **_: reached.append(True),{"messages":[],"max_tokens":20000})
        report = budget_stop_report({"budget":budget.snapshot()},budget.reason,"work:fixture")
    assert reached == [] and (budget.tokens,budget.requests)==(589787,37)
    assert budget.rejected_reservation == 20004
    assert "20004" in report and "10213" in report
    assert "不足以容纳下一次模型请求" in report
    assert "已达到上限" not in report


def test_managed_bridge_180_rounds_one_attempt_one_delivery():
    coordinator=AgentTaskCoordinator(max_workers=1)
    try:
        bridge,runtime,handler,sent=slices.BridgeSliceTests().make_bridge(coordinator,max_requests=180,stop=180)
        bridge.adaptive_budget_enabled=True
        created=bridge.handle_turn(turn())
        task=wait_for(coordinator,created.background_task_id,"completed",timeout=30)
        assert task["attempt"] == 1 and task["metadata"]["slices"] == 59
        assert runtime.calls == 180 and handler.calls == list(range(1,180))
        assert len(sent) == 1 and task["metadata"]["budget"]["managed"]
        assert task["metadata"]["budget"]["requests"] == 180
        assert not task["metadata"]["budget"]["reason"]
    finally:
        coordinator.shutdown(wait=True)


def test_production_config_preserves_long_profiles_and_explicit_lower_cap():
    cfg=load_qq_agent_config({})
    ceilings=dict(max_requests=cfg.task_max_model_requests,max_tokens=cfg.task_max_tokens,
                  max_tools=cfg.task_max_tools,max_active_seconds=cfg.task_max_active_seconds)
    for rows,profile in [([{"effect":"write"}],"standard"),([{"effect":"write"}]*4,"heavy")]:
        task={"contract":{"requirements":rows}}
        budget=TaskBudget()
        configure_task_budget(budget,task,ceilings)
        assert {key:getattr(budget,key) for key in ceilings} == PROFILES[profile]
    capped=load_qq_agent_config({"SHINKU_QQ_AGENT_TASK_MAX_TOKENS":"900000"})
    budget=TaskBudget()
    configure_task_budget(budget,{"contract":None},{"max_tokens":capped.task_max_tokens})
    assert budget.max_tokens == 900000
