from shinku.agent.loop import AgentLoop, AgentDecision, AgentState
from shinku.agent.verification import SmartCompletionVerifier
from shinku.tools.invocation import ToolInvocation


class Search:
    agent_observation_required_fields = ("results",)
    agent_observation_receipt_fields = ("results",)
    agent_evidence_collection_field = "results"
    agent_progress_class = "observe"
    agent_permission_class = "read"

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        return {"tool_type": "private_web_search", "followup_context": "Search results",
                "state_updates": {"results": [{"title": "Announcement", "url": "https://example.com/announcement"}], "search_status": "ok"}}


def verifier():
    class Clock:
        agent_observation_required_fields = ("iso",)
    return SmartCompletionVerifier(result_handlers={"private_web_search": Search(), "private_current_time": Clock()}, tool_capabilities={
        "private_web_search": {"effects": ("tool:private_web_search",), "target_arguments": ()},
        "private_current_time": {"effects": ("tool:private_current_time",), "target_arguments": ()}})


def test_observation_labels_do_not_require_resource_target_receipts():
    state = AgentState(context={"agent_completion_contract": {"requirements": [
        {"id": "r1", "effect": "tool:private_current_time", "targets": ["系统时间"]},
        {"id": "r2", "effect": "tool:private_web_search", "targets": ["AI模型发布资讯"]}]}},
        tool_calls=[{"name": "private_current_time", "status": "ok", "data": {"state_updates": {"iso": "2026-10-01T11:34:21+08:00"}}},
                    {"name": "private_web_search", "status": "ok", "data": {"state_updates": {"results": [{"url": "https://example.com/a"}]}}}])
    assert verifier().evaluate(state, "整理结果及来源如下。").accepted
    state.tool_calls[-1]["data"]["state_updates"]["results"] = []
    assert not verifier().evaluate(state, "整理结果。").accepted


def test_budget_closeout_cannot_execute_more_tools():
    calls = []
    def planner(*, state):
        calls.append(state.finalization_only)
        return AgentDecision.tool(ToolInvocation("private_web_search", {"query": str(len(calls))}))
    result = AgentLoop(planner=planner, handlers={"private_web_search": Search()}, max_rounds=1).run()
    assert calls == [False, True]
    assert len(result.tool_envelopes) == 1
    assert result.status == "partial" and "https://example.com/announcement" in result.final_text


def test_duplicate_sources_force_closeout_even_with_different_queries():
    seen = []
    def planner(*, state):
        seen.append(state.finalization_only)
        if state.finalization_only:
            return AgentDecision.final("仅有部分待核实线索，日期尚未确认。")
        return AgentDecision.tool(ToolInvocation("private_web_search", {"query": str(len(seen))}))
    result = AgentLoop(planner=planner, handlers={"private_web_search": Search()}, max_rounds=6,
                       completion_gate=verifier()).run()
    assert seen == [False, False, False, True]
    assert result.status == "partial" and len(result.tool_envelopes) == 3


def test_observation_alias_cannot_satisfy_file_write():
    state = AgentState(context={"agent_completion_contract": {"requirements": [{"id": "write", "effect": "write", "targets": ["a.txt"]}]}},
        tool_calls=[{"name": "private_web_search", "status": "ok", "data": {"state_updates": {"results": [{"url": "https://example.com/a.txt"}]}}}])
    assert not verifier().evaluate(state, "已保存 a.txt。").accepted


def test_another_source_tool_uses_same_closeout_protocol():
    def planner(*, state):
        return AgentDecision.tool(ToolInvocation("catalog_lookup", {"query": str(state.rounds)}))
    result = AgentLoop(planner=planner, handlers={"catalog_lookup": Search()}, max_rounds=1).run()
    assert result.status == "partial" and len(result.tool_envelopes) == 1
    assert "https://example.com/announcement" in result.final_text
    assert "原问题" in result.final_text
    assert "新模型" not in result.final_text and "发布日期" not in result.final_text


def test_latest_observation_failure_cannot_reuse_old_success():
    state = AgentState(context={"agent_completion_contract": {"requirements": [
        {"id": "r1", "effect": "tool:private_current_time", "targets": ["系统时间"]}]}},
        tool_calls=[{"name": "private_current_time", "status": "ok", "data": {"state_updates": {"iso": "2026-10-01T11:00:00+08:00"}}},
                    {"name": "private_current_time", "status": "error"}])
    assert not verifier().evaluate(state, "当前时间如下。").accepted


def test_tool_collection_does_not_implicitly_complete_business_answer():
    state = AgentState(context={"agent_completion_contract": {"requirements": [
        {"id": "lookup", "effect": "tool:private_web_search", "targets": []}]}},
        tool_calls=[{"name": "private_web_search", "status": "ok", "data": {"state_updates": {"results": [{"url": "https://example.com"}]}}}])
    verdict = verifier().evaluate(state, "只查到泛泛线索，请缩小范围。")
    assert verdict.accepted and verdict.status == "partial"
    assert verifier().evaluate(state, "答案与来源如下。", outcome="completed").status == "completed"


def test_discovery_budget_caps_a_batch_without_disabling_source_reading():
    class NovelSearch(Search):
        def execute(self, *, call, context):
            result = super().execute(call=call, context=context)
            result["state_updates"]["results"][0]["url"] += str(call["query"])
            return result
    class Reader:
        agent_progress_class = "observe"
        def normalize_call(self, call):
            return call
        def execute(self, *, call, context):
            return {"followup_context": "Source text ready"}
    turns = []
    def planner(*, state):
        turns.append(state)
        if len(turns) == 1:
            return AgentDecision.tools([ToolInvocation("catalog_lookup", {"query": str(i)}) for i in range(6)])
        if len(turns) == 2:
            assert state.source_discovery_calls == 4
            assert state.suppressed_tool_names == {"catalog_lookup"}
            return AgentDecision.tool(ToolInvocation("read_source", {}))
        return AgentDecision.final("部分结果及缺口。", outcome="partial")
    result = AgentLoop(planner=planner, handlers={"catalog_lookup": NovelSearch(), "read_source": Reader()}).run()
    assert result.status == "partial"
    assert sum(e.status == "ok" for e in result.tool_envelopes) == 5
    assert sum((e.data or {}).get("code") == "source_discovery_budget_exhausted" for e in result.tool_envelopes) == 2
