"""说明性请求的交付规则；使用隔离文件、真实宿主核验与确定性模型响应。"""
from types import SimpleNamespace
from copy import deepcopy
import hashlib

import pytest

from shinku.agent.analysis_acceptance import ordinary_analysis
from shinku.agent.loop import AgentDecision
from shinku.agent.planner import LLMPlanner
from shinku.agent_runtime.context import execution_context
from test_analysis_acceptance_layers import case, REPORT


def bind(state, task, text):
    task["requests"][0]["text"] = text
    state.context = execution_context(task, SimpleNamespace(
        tool_context={}, permission_mode="read", authorization_ref="fixture-grant"), attempt=1)


@pytest.mark.parametrize("text", [
    "你看一下你本地记忆是什么架构", "帮我看看这个项目", "这个模块怎么组织的",
    "本地索引的工作方式是什么？", "What does this module do?",
    "部署流程是什么？", "解释统计模块的结构", "怎么运行测试？",
    "读取文档，说明里面的统计逻辑", "Explain how to calculate a sum",
])
def test_informational_variants_deliver_without_keyword_whitelist(case, text):
    state, gate, _, _, task = case
    bind(state, task, text)
    assert ordinary_analysis(state)
    result = gate.evaluate(state, REPORT, outcome="completed")
    assert result.accepted and result.layers["answer_evidence"] == "unverified"


def test_ordinary_answer_uses_no_additional_quality_model_request(case):
    state, _, *_ = case
    class Runtime:
        calls = 0
        def call_chat_json(self, **kwargs):
            self.calls += 1
            return {}
    runtime = Runtime()
    planner = LLMPlanner(runtime=runtime, system_prompt="fixture")
    planner._review_local_answer(state, AgentDecision.final(REPORT))
    assert runtime.calls == 0
    assert state.local_answer.answer_review_calls == 0


@pytest.mark.parametrize("text", [
    "精确统计各项金额", "读取所有文件，每个结论逐条提供证据",
    "解释之后修改配置", "Analyze this repo and run tests",
])
def test_strict_goals_are_not_relaxed(case, text):
    state, gate, _, _, task = case
    bind(state, task, text)
    assert not ordinary_analysis(state)
    assert not gate.evaluate(state, REPORT, outcome="completed").accepted


def model_plan(state, task):
    """模拟宿主接纳新会话契约，原请求和源码读取均来自隔离夹具。"""
    from shinku.agent.analysis_contract import origin_record
    task["input_mode"] = "conversation"
    task["analysis_plan_origin"] = origin_record({**task, "contract": None}, task["contract"])
    state.context["agent_completion_contract"] = task["contract"]


def test_model_optional_source_can_be_skipped_but_not_claimed_verified(case):
    state, gate, _, path, task = case
    task["contract"]["requirements"].append({
        "id": "optional", "description": "额外调查", "effect": "read", "targets": [str(path.parent/"other.txt")]})
    model_plan(state, task)
    verdict = gate.evaluate(state, REPORT, outcome="completed")
    assert verdict.accepted and verdict.status == "partial"
    assert verdict.layers["plan_gaps"] == ["optional"]
    assert verdict.layers["execution_results"] == "unverified"
    assert "optional" not in verdict.satisfied


def test_soft_plan_allows_further_reading_but_still_has_exploration_bound(case):
    from shinku.agent.local_analysis import prepare_local_analysis
    state, gate, reader, path, task = case
    task["contract"]["requirements"].append({
        "id": "optional", "description": "辅助阅读", "effect": "read", "targets": [str(path.parent/"other.txt")]})
    model_plan(state, task)
    deps = SimpleNamespace(completion_gate=lambda: gate, handlers=lambda: {reader.tool_type: reader},
                           observe_state=lambda *args: None)
    prepare_local_analysis(SimpleNamespace(state=state), deps)
    assert not state.finalization_only and state.progress["local_analysis"]["phase"] == "gathering"
    # 宿主探索限额只决定停止继续取证，不创建来源或完成证明。
    state.tool_calls = [deepcopy(state.tool_calls[0]) for _ in range(12)]
    prepare_local_analysis(SimpleNamespace(state=state), deps)
    assert state.finalization_only and state.progress["local_analysis"]["reason"] == "exploration_limit"
    assert len(state.local_answer.sources) == 1


@pytest.mark.parametrize("defect", ["explicit_target", "new_request", "changed_row", "legacy", "structured"])
def test_plan_provenance_cannot_relax_host_or_user_conditions(case, defect):
    from shinku.agent.analysis_contract import plan_ids
    state, gate, _, path, task = case
    if defect == "explicit_target":
        bind(state, task, "读取 " + str(path) + "，告诉我结构。")
    model_plan(state, task)
    if defect == "new_request": task["requests"].append({"message_id": "extra", "text": "精确审计"})
    if defect == "changed_row": task["contract"]["requirements"][0]["targets"] = [str(path.parent/"different")]
    if defect == "legacy": task.pop("analysis_plan_origin")
    if defect == "structured": task["input_mode"] = "task"
    assert "read" not in plan_ids(task, task["contract"])
    path.unlink()
    assert not gate.evaluate(state, REPORT, outcome="completed").accepted


def test_model_can_revise_auxiliary_plan_but_not_user_target(case):
    from shinku.agent.analysis_contract import merge_plan
    from shinku.agent.task_state import TaskContractError
    state, _, _, path, task = case
    bind(state, task, "读取 " + str(path) + "，看看怎么组织的。")
    task["contract"]["requirements"].append({
        "id": "optional", "description": "辅助读取", "effect": "read", "targets": [str(path.parent/"optional")]})
    model_plan(state, task)
    revised = deepcopy(task["contract"])
    revised["requirements"] = revised["requirements"][:2]
    revised["requirements"][1]["checks"]["min_sources"] = 1
    assert merge_plan(task, revised) == revised
    revised["requirements"] = revised["requirements"][1:]
    with pytest.raises(TaskContractError):
        merge_plan(task, revised)


@pytest.mark.parametrize("defect", ["ok", "failed", "wrong_command", "extra_write", "no_origin"])
def test_investigation_command_does_not_turn_prose_into_audit(case, defect):
    state, gate, _, _, task = case
    task["contract"]["requirements"].append({"id": "command", "description": "辅助查询", "effect": "command_run",
        "targets": [], "checks": {"command": "fixture-query", "exit_code": 0}})
    model_plan(state, task)
    handler = SimpleNamespace(agent_permission_class="execute")
    gate.result_handlers["query"] = handler
    gate.tool_capabilities["query"] = {"effects": ["command_run"]}
    receipt = {"verified_effects": ["command_run"], "execution_session": {
        "command_sha256": hashlib.sha256(b"fixture-query").hexdigest(), "status": "completed", "returncode": 0}}
    if defect == "failed": receipt["execution_session"]["returncode"] = 1
    if defect == "wrong_command": receipt["execution_session"]["command_sha256"] = "other"
    if defect == "extra_write": receipt["verified_effects"].append("write")
    if defect == "no_origin": task.pop("analysis_plan_origin")
    state.tool_calls.append({"id": "query-one", "name": "query", "status": "error" if defect == "failed" else "ok",
                             "executed": True, "data": {"state_updates": receipt}})
    verdict = gate.evaluate(state, REPORT, outcome="completed")
    if defect in {"extra_write", "no_origin"}:
        assert not verdict.accepted
    else:
        assert verdict.accepted
        assert verdict.status == ("completed" if defect == "ok" else "partial")
        assert verdict.layers["answer_evidence"] == "unverified"
        assert verdict.layers["execution_results"] == ("passed" if defect == "ok" else "unverified")
        from shinku.agent_runtime.results import export_task_result
        from shinku.conversation.models import AnswerEvidence, text_digest
        task.update(completion=verdict.as_dict(), status=verdict.status, completion_attempt=1)
        # 已有语义绑定也不能覆盖新策略下尚未满足的完整证据要求。
        task["result_answer_evidence"] = AnswerEvidence("local", "verified", True, 1, 1, text_digest(REPORT)).as_dict()
        assert "verified_answer" not in export_task_result(task, report=REPORT)


def test_review_exhaustion_stops_before_unreviewable_third_draft(case):
    from shinku.agent.loop import AgentLoop
    from shinku.agent.answer_protocol import retain_answer_draft
    from shinku.tools.execution import ExecutionPolicy
    state, gate, reader, _, task = case
    bind(state, task, "请精确审计，逐条核验")
    state.context["agent_restored_tool_calls"] = state.tool_calls
    generated = []
    def planner(*, state):
        answer = REPORT + "\n本次候选：" + str(len(generated))
        generated.append(answer)
        retain_answer_draft(state, {"kind": "final", "text": answer})
        state.local_answer.answer_review_calls += 1
        state.local_answer.apply_answer_review(answer, {}, state.context)
        return AgentDecision.final(answer)
    result = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, completion_gate=gate,
                       policy=ExecutionPolicy(default_risk="low"), sliced=True).run(context=state.context)
    assert len(generated) == 2
    assert result.reason == "local_answer_review_exhausted" and result.status == "failed"
    assert not result.completion["accepted"]
    assert generated[-1] in result.final_text and "未核实的模型原稿" in result.final_text
    assert "invalid_review_schema" in result.final_text
    assert len(result.progress["answer_drafts"]) == 2


@pytest.mark.parametrize("defect", ["none", "revoked", "changed", "new_request"])
def test_completion_rejection_preserves_only_current_authorized_draft(case, defect):
    from shinku.agent.loop import AgentLoop
    from shinku.agent.answer_protocol import retain_answer_draft
    from shinku.tools.execution import ExecutionPolicy
    state, gate, reader, path, task = case
    bind(state, task, "精确审计")
    state.context["agent_restored_tool_calls"] = state.tool_calls
    seen = []
    def planner(*, state):
        seen.append(1)
        if len(seen) == 1:
            retain_answer_draft(state, {"kind": "final", "text": REPORT})
        if len(seen) == 3:
            if defect == "revoked": reader.allowed = False
            if defect == "changed": path.write_text("内容变化")
            if defect == "new_request": task["requests"].append({"message_id": "new", "text": "换个目标"})
        return AgentDecision.final(REPORT)
    result = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, completion_gate=gate,
                       policy=ExecutionPolicy(default_risk="low"), sliced=True).run(context=state.context)
    assert result.reason == "completion_no_progress" and not result.completion["accepted"]
    assert (REPORT in result.final_text) is (defect == "none")


@pytest.mark.parametrize("attack", ["forged_import", "forged_update", "new_strict_request", "input_mode"])
def test_database_does_not_trust_proposed_plan_origin(case, tmp_path, attack):
    from shinku.agent.task_state import AgentTaskStore, TaskContractError
    from shinku.agent.analysis_contract import origin_record
    _, _, _, _, original = case
    task = deepcopy(original)
    task["input_mode"] = "conversation" if attack != "input_mode" else "task"
    task["analysis_plan_origin"] = origin_record({**task, "contract": None}, task["contract"])
    store = AgentTaskStore(tmp_path/"origin.sqlite")
    try:
        saved = store.save(task)
        assert not saved.get("analysis_plan_origin")
        proposal = deepcopy(saved)
        proposal["contract"]["requirements"] = proposal["contract"]["requirements"][1:]
        if attack in {"forged_import", "forged_update"}:
            proposal["analysis_plan_origin"] = task["analysis_plan_origin"]
        if attack == "new_strict_request":
            proposal["requests"].append({"message_id": "strict", "text": "请精确核验全部来源"})
        if attack == "input_mode":
            proposal["input_mode"] = "conversation"
        with pytest.raises(TaskContractError):
            store.save(proposal)
        persisted = store.get(saved["task_id"], requester=saved["requester"], session=saved["session"])
        assert persisted["contract"] == saved["contract"]
    finally:
        store.close()
