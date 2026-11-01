"""防循环收尾与人工交接回归；合成工具不访问用户服务或账号。"""
from copy import deepcopy

import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.loop import AgentDecision, AgentLoop, AgentState
from shinku.agent.planner import LLMPlanner
from shinku.tools.invocation import ToolInvocation


class ServiceProbe:
    agent_progress_class = "change"

    def __init__(self):
        self.calls = 0

    def normalize_call(self, call):
        return call

    def execute(self, **kwargs):
        self.calls += 1
        return {"followup_context": "service running; manual login required",
                "state_updates": {"verified_effects": ["command_run"]}}


def invocation():
    return AgentDecision.tool(ToolInvocation("probe", {"action": "status"}))


@pytest.mark.parametrize("sliced", [False, True])
def test_repeated_checks_deliver_one_report_without_replaying_tool(sliced):
    probe, closeouts, gates = ServiceProbe(), [], []

    def planner(*, state):
        if state.tool_closeout_reason:
            closeouts.append(deepcopy(state.tool_calls))
            assert state.finalization_only
            return AgentDecision.final("服务已运行，账号需要用户扫码。", outcome="blocked")
        return invocation()

    loop = AgentLoop(planner=planner, handlers={"probe": probe}, sliced=sliced,
                     max_rounds=1 if sliced else 6,
                     completion_gate=lambda *a: gates.append(a) or True)
    result = loop.run()
    while result.status == "yielded":
        result = loop.run(checkpoint=result.checkpoint)
    assert (result.status, result.reason) == ("blocked", "repeated_tool_call")
    assert "需要用户扫码" in result.final_text
    assert probe.calls == len(closeouts) == 1 and not gates
    assert result.completion["accepted"] is False
    assert len(result.tool_envelopes) == 3
    assert [t["data"]["code"] for t in closeouts[0][1:]] == ["duplicate_tool_call"] * 2


@pytest.mark.parametrize("bad", [None, "throw", "tool", "empty", "completed", "metadata"])
def test_closeout_failure_or_false_success_has_nonempty_honest_fallback(bad):
    probe, attempts = ServiceProbe(), []

    def planner(*, state):
        if not state.tool_closeout_reason:
            return invocation()
        attempts.append(True)
        if bad == "throw":
            raise TimeoutError("private upstream details")
        if bad == "tool":
            return invocation()
        if bad == "empty":
            return AgentDecision.final("", outcome="blocked")
        if bad == "completed":
            return AgentDecision.final("全部完成，账号在线。", outcome="completed")
        if bad == "metadata":
            return {"task_notes": {"blocker": ""}, "task_outcome": "completed"}
        return None

    result = AgentLoop(planner=planner, handlers={"probe": probe}).run()
    assert probe.calls == len(attempts) == 1
    assert result.status == "blocked" and not result.completion["accepted"]
    assert "任务尚未完成" in result.final_text and "工具回执" in result.final_text
    assert "账号在线" not in result.final_text and "private upstream" not in result.final_text


@pytest.mark.parametrize("reason", ["task_canceled", "agent_permission_changed", "task_time_budget_exhausted"])
def test_interrupt_during_closeout_keeps_authority_and_no_extra_tools(reason):
    probe, stopped = ServiceProbe(), []

    def planner(*, state):
        if state.tool_closeout_reason:
            stopped.append(True)
            return AgentDecision.final("服务已运行", outcome="blocked")
        return invocation()

    result = AgentLoop(planner=planner, handlers={"probe": probe},
        interruption_check=lambda _: reason if stopped else "").run()
    assert result.reason == reason and probe.calls == 1
    assert result.status != "completed" and not result.final_text


def test_human_login_wait_delivers_actual_state_without_acceptance_or_more_checks():
    probe, checks = ServiceProbe(), []
    def planner(*, state):
        if not state.tool_calls:
            return invocation()
        return AgentDecision.wait("服务已运行，需要你扫码登录。", reason="human_action_required")
    result = AgentLoop(planner=planner, handlers={"probe": probe},
        completion_gate=lambda *args: checks.append(args) or False).run()
    assert result.status == "waiting_user" and result.reason == "human_action_required"
    assert probe.calls == 1 and not checks
    assert "扫码" in result.final_text and not result.completion.get("accepted")


def test_planner_closeout_disables_tools_reviews_and_reply_repair(monkeypatch):
    requests = []
    class Runtime:
        def call_chat_json(self, **kwargs):
            requests.append(kwargs)
            return {"task_outcome": "blocked"}  # 无正文不能触发第二次格式修复。
    planner = LLMPlanner(runtime=Runtime(), system_prompt="fixture",
        tool_specs=[{"type": "function", "function": {"name": "probe"}}])
    for name in ("_review_research_candidates", "_review_research_answer", "_review_local_answer", "_repair_final_reply"):
        monkeypatch.setattr(planner, name, lambda *a, **kw: pytest.fail("closeout must not review or repair"))
    state = AgentState(finalization_only=True, tool_closeout_reason="repeated_tool_call")
    planner(state=state)
    assert len(requests) == 1
    assert requests[0]["native_tools"] is None and not requests[0]["native_tool_choice"]


def test_closeout_preserves_managed_budget_counters():
    budget = TaskBudget(max_requests=10, max_tools=10, requests=2, tokens=30, managed=True)
    probe = ServiceProbe()
    def planner(*, state):
        return (AgentDecision.final("仍需人工登录", outcome="blocked")
                if state.tool_closeout_reason else invocation())
    with bind_task_budget(budget):
        result = AgentLoop(planner=planner, handlers={"probe": probe}).run()
    assert result.status == "blocked" and budget.requests == 2 and budget.tokens == 30
    assert budget.tools == 1 and probe.calls == 1


@pytest.mark.parametrize("human_wait", [False, True])
def test_real_service_archives_closeout_with_original_ledger_and_budget(tmp_path, monkeypatch, human_wait):
    from shinku.agent_runtime.demo import demo_request, demo_service
    from shinku.agent_runtime import AgentScope
    from test_agent_runtime_boundary import settled

    service, tool, runtime = demo_service(tmp_path)
    original = demo_request(tool)
    def respond(**kwargs):
        runtime.calls.append(kwargs)
        closing = any("宿主已停止重复调用" in s for s in kwargs.get("system_extra_blocks") or [])
        if closing:
            return {"kind": "final", "text": "文件已写入，后续步骤尚未完成。TAIL_REPORT", "task_outcome": "blocked"}
        if human_wait and tool.calls:
            return {"kind": "wait", "text": "文件已写入，后续需要用户确认。TAIL_REPORT", "reason": "human_action_required"}
        return {"tool_call": {"type": "demo_file", "path": str(tool.path)}}
    monkeypatch.setattr(runtime, 'call_chat_json', respond)
    port = service.bind(AgentScope('fixture', 'local:closeout'))
    try:
        h = port.submit(original)
        view = settled(port, h.task_id)
        result = port.result(h.task_id)
        assert view['status'] == ('waiting_user' if human_wait else 'blocked')
        assert result['execution_report']['text'].endswith('TAIL_REPORT')
        assert result['execution_report']['is_fact_evidence'] is False
        assert result['verification'].get('accepted') is not True and 'verified_answer' not in result
        assert tool.calls == 1 and tool.path.read_text() == 'standalone agent verified'
        assert result['original_requests'][0]['text'] == original.original_request
        assert result['contract'] == original.contract
        assert result['budget']['requests'] == len(runtime.calls) == (2 if human_wait else 4)
        assert len(result['evidence']) == (1 if human_wait else 3)
        assert result['artifacts'][0]['verification'] == 'unconfirmed'
        before = len(runtime.calls), result['result_hash']
        assert port.result(h.task_id)['result_hash'] == before[1]
        assert len(runtime.calls) == before[0]
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
