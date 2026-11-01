"""URL targets and completion failures share preflight, execution and reporting."""

import pytest
from test_agent_async_processor import turn, wait_for
from test_agent_completion_batch_a import ReceiptHandler
from test_agent_contract_repair import Runtime
from test_research_evidence_ledger import read_ledger

from shinku.agent.budget import TaskBudgetExceeded
from shinku.agent.completion import CompletionVerdict
from shinku.agent.local_answer import LocalAnswerLedger
from shinku.agent.loop import AgentDecision, AgentLoop, AgentRunResult, AgentState
from shinku.agent.planner import LLMPlanner
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent.verification import SmartCompletionVerifier
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.registry import ToolRegistry

URL = "https://github.com/composio-community/awesome-codex-skills"
REQUEST = f"老婆帮我看一下{URL}项目是做什么的"


def contract(*urls):
    return {"goal": "读取项目页面并解释用途", "scope": ["指定网页，只读"],
            "deliverables": ["项目功能说明"], "requirements": [
                {"id": f"r{i}", "description": "读取页面", "effect": "browser_access",
                 "targets": [url], "checks": {}} for i, url in enumerate(urls, 1)]}


@pytest.mark.parametrize("request_text,expected", [
    (REQUEST, [URL]),
    (f"老婆帮我看一下 {URL} 项目是做什么的", [URL]),
    (f"解释[{URL}]({URL})。", [URL]),
    (f"查看（{URL}）。", [URL]),
    ("看看https://example.test/a.pdf?lang=zh#intro讲什么", ["https://example.test/a.pdf?lang=zh#intro"]),
    ('读取“https://example.test/中文.pdf”', ["https://example.test/中文.pdf"]),
    ("查看 https://example.test/a_(b).pdf。", ["https://example.test/a_(b).pdf"]),
    ("查看https://example.test/a_(b)然后解释", ["https://example.test/a_(b)"]),
    ("查看https://example.test/a%20b?x=1&y=2#top。", ["https://example.test/a%20b?x=1&y=2#top"]),
    ("查看http://[::1]:9998/page。", ["http://[::1]:9998/page"]),
    (f'读取 {URL} 并保存文件 "C:/out/report.pdf"', [URL, "c:\\out\\report.pdf"]),
    (f"查看{URL}和https://example.test/b", [URL, "https://example.test/b"]),
])
def test_url_targets_never_become_prose_or_partial_file_paths(request_text, expected):
    assert SmartCompletionVerifier._extract_targets(request_text) == expected


def test_url_preflight_and_final_gate_use_the_same_complete_target():
    verifier = SmartCompletionVerifier()
    value = contract(URL)
    assert verifier.target_coverage_gaps(value, [REQUEST]) == []
    receipt = {"name": "private_browser_open", "arguments": {"url": URL}, "status": "ok",
               "data": {"state_updates": {"verified_effects": ["browser_access"],
                                          "completion_targets": [URL]}}}
    state = AgentState(context={"agent_completion_contract": value}, tool_calls=[receipt],
                       transcript=[{"role": "user", "content": REQUEST}])
    assert verifier.evaluate(state, "这是一个技能集合。", outcome="completed").accepted
    missing = "https://example.test/second"
    assert verifier.target_coverage_gaps(value, [REQUEST + "，还要看" + missing]) == [
        {"target": missing, "effects": []}]
    receipt["data"]["state_updates"]["completion_targets"] = ["https://github.com"]
    assert not verifier.evaluate(state, "完成。", outcome="completed").accepted


def test_url_read_prose_cannot_acquire_local_file_roles():
    assert SmartCompletionVerifier._explicit_target_roles([f"读取 {URL}"]) == {}
    value = contract(URL)
    assert SmartCompletionVerifier().target_coverage_gaps(value, [f"读取 {URL}"]) == []
    request_text = f'读取 {URL}，输出文件 "C:/out/report.pdf"'
    assert SmartCompletionVerifier._explicit_target_roles([request_text]) == {"c:\\out\\report.pdf": {"write"}}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native_metadata_missing", [False, True])
def test_original_request_completes_without_spurious_contract_repair(tmp_path, asynchronous, native_metadata_missing):
    def prepare(kwargs):
        import json
        payload = json.loads(kwargs["user_prompt"])
        assert payload["task_link_options"] == [{"mode": "new"}]
        assert payload["explicit_targets"] == [URL]
        assert payload["selected_tools"] == ["private_browser_open"]
        assert "不能返回 mode=none" in "\n".join(kwargs["system_extra_blocks"])
        assert not kwargs.get("native_tools")
        return {"task_link": {"mode": "new"}, "task_contract": contract(URL)}

    responses = [prepare] if native_metadata_missing else []
    responses.append({"kind": "final", "text": "这是 Codex 技能集合，用来扩展任务工作流程。",
                      "task_outcome": "completed"})
    runtime = Runtime(*responses)
    handler = ReceiptHandler()
    sent = []
    coordinator = AgentTaskCoordinator(max_workers=1)
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    bridge = QQAgentBridge(runtime=runtime, system_prompt="隔离回归测试", memory_service=None,
        sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("fixture",)),
        tool_host=ToolHost(ToolRegistry({handler.tool_type: handler})),
        command_state=QQCommandState(permission_mode="full"), master_user_ids=("owner",),
        task_store=store, task_contract_enabled=True, task_coordinator=coordinator,
        async_agent_enabled=asynchronous)
    decision = AgentDecision.from_value({"kind": "tool", "tool_calls": [
        {"name": handler.tool_type, "arguments": {"url": URL}}],
        **({} if native_metadata_missing else {"task_link": {"mode": "new"}, "task_contract": contract(URL)})})
    bridge._run_chat_decision = lambda **kwargs: (decision, {handler.tool_type})
    try:
        reply = bridge.handle_turn(turn(REQUEST))
        if asynchronous:
            result = wait_for(coordinator, reply.background_task_id, "completed")
            assert "技能集合" in result["result_summary"]
        else:
            assert reply.agent_status == "completed", reply.as_dict()
        saved = store.candidates(requester="owner", session="private:owner")[0]
        assert saved["status"] == "completed"
        assert saved["contract"] == contract(URL)
        assert len(saved["tool_calls"]) == 1
        assert len(runtime.calls) == (2 if native_metadata_missing else 1)
        assert saved["last_report"] and sent
    finally:
        coordinator.shutdown(wait=True)
        store.close()


@pytest.mark.parametrize("stage", ["candidate", "local", "research"])
def test_review_budget_stop_is_not_misreported_as_invalid_evidence(stage):
    def exhausted(kwargs):
        raise TaskBudgetExceeded("task_model_budget_exhausted")
    runtime = Runtime(exhausted)
    planner = LLMPlanner(runtime=runtime, system_prompt="隔离测试", tool_specs=[])
    state = AgentState(context={})
    state.research = read_ledger()
    state.local_answer = LocalAnswerLedger(enabled=True)
    state.research._candidate({"url": "https://example.test/next", "title": "next source"})
    decision = AgentDecision.final("带来源的候选答复。")
    with pytest.raises(TaskBudgetExceeded, match="task_model_budget_exhausted"):
        if stage == "candidate":
            planner._review_research_candidates(state)
        elif stage == "local":
            planner._review_local_answer(state, decision)
        else:
            planner._review_research_answer(state, decision)
    assert len(runtime.calls) == 1




@pytest.mark.parametrize("sliced", [False, True])
def test_repeated_rejected_finals_preserve_exact_verdict_and_stop(sliced):
    verdict = CompletionVerdict(False, reason="完成验收未通过：缺少条件 r2 的实际结果证据",
                                satisfied=("r1",), missing=("r2",))
    loop = AgentLoop(planner=lambda **kwargs: AgentDecision.final("全部完成了。"), handlers={},
                     completion_gate=lambda *args: verdict, max_rounds=1 if sliced else 20,
                     sliced=sliced)
    result = loop.run()
    while result.status == "yielded":
        result = loop.run(checkpoint=result.checkpoint)
    assert (result.status, result.reason, result.rounds) == ("failed", "completion_no_progress", 3)
    assert result.final_text == ""  # Never deliver the rejected success claim.
    assert result.completion == verdict.as_dict()


def test_failure_reply_explains_missing_evidence_and_is_saved(tmp_path):
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    task = new_task(task_id="fixture", requester="owner", session="private:owner",
                    request=REQUEST, message_id="fixture-message",
                    contract=contract(URL, "https://example.test/second"))
    bridge = QQAgentBridge(runtime=Runtime(), system_prompt="隔离回归测试", task_store=store)
    verdict = CompletionVerdict(False, reason="PRIVATE_RAW_ERROR_SECRET", satisfied=("r1",), missing=("r2",))
    try:
        reply = bridge._finish_result(turn(REQUEST), AgentRunResult(status="failed", rounds=3,
            reason="completion_no_progress", completion=verdict.as_dict()),
            context={"agent_task_state": task}, initial_transcript=[], task_id="fixture")
        assert "页面访问与读取" in reply.final_text
        assert "1 项" in reply.final_text and "缺少实际结果证据" in reply.final_text
        assert "PRIVATE_RAW_ERROR_SECRET" not in reply.final_text
        assert store.get("fixture", requester="owner", session="private:owner")["last_report"] == reply.final_text
    finally:
        store.close()


@pytest.mark.parametrize("repair_valid", [False, True])
def test_initial_malformed_native_call_has_one_schema_retry_and_never_executes_fake_arguments(tmp_path, repair_valid):
    malformed = {"_native_tool_call": {"_malformed": True,
                 "protocol_error": "native_tool_arguments_malformed", "tool_calls": []}, "tool_call": None}
    selected = {"kind": "tool", "tool_calls": [{"name": "private_browser_open", "arguments": {"url": URL}}],
                "task_link": {"mode": "new"}, "task_contract": contract(URL)}
    responses = [malformed, selected if repair_valid else malformed]
    if repair_valid:
        responses.append({"kind": "final", "text": "这是技能集合。", "task_outcome": "completed"})
    runtime = Runtime(*responses)
    runtime.chat_supports_native_tools = lambda: True
    handler = ReceiptHandler()
    store = AgentTaskStore(tmp_path / "tasks.sqlite3")
    coordinator = AgentTaskCoordinator(max_workers=1)
    bridge = QQAgentBridge(runtime=runtime, system_prompt="隔离回归测试", task_store=store,
        task_contract_enabled=True, command_state=QQCommandState(permission_mode="full"),
        master_user_ids=("owner",), task_coordinator=coordinator, async_agent_enabled=True,
        tool_host=ToolHost(ToolRegistry({handler.tool_type: handler})))
    try:
        reply = bridge.handle_turn(turn(REQUEST))
        assert runtime.calls[0]["native_tools"][0]["function"]["name"] == "__shinku_tool_dispatch__"
        assert runtime.calls[1]["native_tools"][0]["function"]["name"] == "private_browser_open"
        assert "_shinku_request_kind" in runtime.calls[1]["native_tools"][0]["function"]["parameters"]["required"]
        if repair_valid:
            assert reply.agent_status == "waiting_background", reply.as_dict()
            wait_for(coordinator, reply.background_task_id, "completed")
            saved = store.candidates(requester="owner", session="private:owner")[0]
            assert len(saved["tool_calls"]) == 1
            assert len(runtime.calls) == 3
        else:
            assert reply.agent_status == "failed"
            assert "工具参数格式错误" in reply.final_text
            assert "尚未执行任何工具" in reply.final_text
            assert store.candidates(requester="owner", session="private:owner") == []
            assert len(runtime.calls) == 2
    finally:
        coordinator.shutdown(wait=True)
        store.close()
