"""用户自述的既有操作不构成本轮指令；合成输入覆盖完成核验与 QQ 交付。"""
import json

import pytest

from shinku.agent.request_intent import execution_request_text
from shinku.agent.verification import SmartCompletionVerifier as Verifier
from shinku.tasks.coordinator import AgentTaskCoordinator
from test_agent_intent_boundaries import state_for
from test_contract_c2_5 import FakeResponse, _make_runtime
from test_qq_at_failure_regression import bridge, turn


SELF_REPORTS = [
    "这些资料我可是保存的很好的，不会弄丢。",
    "资料我保存得非常仔细。",
    "我已经保存了备份。",
    "我把 note.txt 保存了。",
    "我之前修改过 draft.txt。",
    "我一直保存着这些资料。",
]
ANSWER = "有备份就安心些。"


@pytest.mark.parametrize("text", SELF_REPORTS)
def test_user_self_report_does_not_require_assistant_execution(text):
    state = state_for(text)
    assert not Verifier.request_requires_action(text)
    verdict = Verifier().evaluate(state, ANSWER, outcome="completed")
    assert verdict.accepted and verdict.status == "completed", verdict
    # 投影不能修改宿主持有的原始请求或自行补出工具执行证据。
    assert state.context["agent_request_scope"].user_messages() == [text]
    assert state.tool_calls == []


@pytest.mark.parametrize("text", [
    "我想保存 note.txt。",
    "我要保存 note.txt。",
    "我需要你保存 note.txt。",
    "我保存的文件修改一下。",
    "我已经保存了 note.txt，你修改一下。",
    "我把 a.txt 保存了，然后修改 b.txt。",
    "我把 a.txt 保存了并发送 b.txt。",
    "我之前修改过 draft.txt 现在请保存 result.txt。",
    "这些资料我可是保存的很好的，帮我保存另一份。",
])
def test_new_operations_after_self_report_still_require_receipts(text):
    assert Verifier.request_requires_action(text)
    assert not Verifier().evaluate(state_for(text), ANSWER, outcome="completed").accepted


def test_projection_preserves_later_target_offsets():
    text = "我把 a.txt 保存了，然后修改 b.txt。"
    projected = execution_request_text(text)
    assert len(projected) == len(text)
    assert "a.txt" not in projected
    assert projected.index("b.txt") == text.index("b.txt")


@pytest.mark.parametrize("answer", ["已保存 result.txt。", "已删除 temp.txt。", "已发送 report.txt。"])
def test_assistant_success_claim_still_needs_execution(answer):
    assert not Verifier().evaluate(state_for(SELF_REPORTS[0]), answer, outcome="completed").accepted


def test_self_report_cannot_remove_host_completion_contract():
    state = state_for(SELF_REPORTS[0])
    state.context["agent_completion_contract"] = {"requirements": [
        {"id": "saved", "effect": "write", "targets": ["result.txt"], "checks": {}}]}
    verdict = Verifier().evaluate(state, ANSWER, outcome="completed")
    assert not verdict.accepted and "saved" in verdict.missing


@pytest.mark.parametrize("kind", ["private", "group"])
@pytest.mark.parametrize("text", SELF_REPORTS)
def test_self_report_reaches_real_qq_pipeline_without_tools_or_tasks(kind, text):
    runtime = _make_runtime(scripts=[("response", FakeResponse(json.dumps({
        "kind": "final", "text": ANSWER, "task_outcome": "completed"}, ensure_ascii=False)))])
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(runtime, coordinator, sent).handle_turn(turn(kind, text=text))
        assert (result.agent_status, result.reason, result.delivery_status) == (
            "completed", "direct_chat_reply", "sent")
        assert len(sent) == 1 and result.final_text == ANSWER
        assert not coordinator.list()
        assert runtime.snapshot_metrics()["chat_json_calls"] == 1
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()
