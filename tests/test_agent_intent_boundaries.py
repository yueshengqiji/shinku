"""Reply composition, verified outcomes and scoped task-control boundaries."""
import hashlib

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.request_scope import RequestScope
from shinku.agent.task_state import new_task
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.tasks.coordinator import AgentTaskOutcome
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat, boundary_bridge


REWRITES = [
    "帮我把这句话改写得礼貌一点：快点给我。",
    "请把这段话改成一句简短的回复：请尽快提供资料。",
    "帮我修改下面这段文字：今天不能过去了。",
    "把这句话改写一下：删除 a.txt 并把 config.json 保存到磁盘。",
    "把这句话改写一下：已保存 result.txt。",
    "请把这段话改成简短回复",
    "帮我修改下面这段文字：“今天不能过去了。”",
]


@pytest.mark.parametrize("prompt", [
    "聊天改写里出现了写文件，顺便查一下上海时间。",
    "回复中提到了保存 result.txt，这是对聊天内容的描述。",
    "这段文本中包含删除 a.txt 的例子，请解释原意。",
])
def test_reported_action_words_in_text_do_not_request_external_execution(prompt):
    assert not SmartCompletionVerifier.request_requires_action(prompt)
    verdict = SmartCompletionVerifier().evaluate(state_for(prompt), "这是对内容的描述。", outcome="completed")
    assert verdict.accepted, verdict


@pytest.mark.parametrize("prompt", [
    "聊天改写里出现了写文件，然后保存到 result.txt。",
    "这段文本中提到修改，需要你修改 config.json。",
    "回复中包含了删除说明，请删除 a.txt。",
    "请修改聊天记录里的内容。",
    "聊天记录里出现了需要你保存 result.txt 的要求。",
    "聊天记录里出现了错误需要修改 config.json。",
    "回复中出现了错误必须修改 config.json。",
])
def test_text_reference_does_not_hide_independent_external_request(prompt):
    assert SmartCompletionVerifier.request_requires_action(prompt)
    verdict = SmartCompletionVerifier().evaluate(state_for(prompt), "已经完成。", outcome="completed")
    assert not verdict.accepted


def state_for(prompt):
    return AgentState(context={"agent_request_scope": RequestScope(
        "fixture", "private:fixture", "intent-message", (("intent-message", prompt),))})


@pytest.mark.parametrize("prompt", REWRITES)
@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_reply_composition_reaches_actual_qq_reply_without_writing(tmp_path, prompt, mode):
    assert not SmartCompletionVerifier.request_requires_action(prompt)
    service, tool, execution = demo_service(tmp_path)
    answer = "请您方便时尽快提供，谢谢。"
    chat = Chat([{"kind": "final", "text": answer}])
    bridge = (boundary_bridge(service, tool, chat) if mode == "task" else
              QQAgentBridge(runtime=chat, system_prompt="Synthetic conversation"))
    try:
        reply = bridge(turn(prompt))
        assert reply.agent_status == "completed", reply
        assert reply.final_text == answer and reply.delivery_status == "dry_run"
        assert len(chat.calls) == 1 and execution.calls == [] and tool.calls == 0
        assert service.task_store.candidates(requester="fixture", session="private:fixture") == []
        assert not tool.path.exists()
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("prompt", [
    "修改 config.json 的设置",
    "把这句话改写后保存到 result.txt：快点给我。",
    "修改 note.txt 中的这句话：快点给我。",
    "把这句话写入 result.txt：快点给我。",
    "帮我修改配置文件",
    "把这句话改写一下：“快点给我。”然后保存到 result.txt",
    "把这句话改成 result.txt 中的内容",
    "帮我修改“config.json”的设置",
    "请对 config.json 进行“修改”",
])
def test_external_writing_still_requires_matching_receipts(prompt):
    assert SmartCompletionVerifier.request_requires_action(prompt)
    verdict = SmartCompletionVerifier().evaluate(state_for(prompt), "请您尽快提供，谢谢。", outcome="completed")
    assert not verdict.accepted


@pytest.mark.parametrize("answer", ["已保存 result.txt。", "“已保存 result.txt。”", "`已保存 result.txt。`"])
def test_quoting_success_does_not_remove_requested_file_write(answer):
    verdict = SmartCompletionVerifier().evaluate(state_for("创建 result.txt"), answer, outcome="completed")
    assert not verdict.accepted


@pytest.mark.parametrize("prompt,answer", [
    ("解释“已保存”这个词是什么意思。", "“已保存”表示资料已经被记录。"),
    ("解释这句示例，不要操作：已保存 result.txt。", "示例“已保存 result.txt”是在描述保存完成。"),
])
def test_quoted_effect_discussion_is_not_an_execution_claim(prompt, answer):
    verdict = SmartCompletionVerifier().evaluate(state_for(prompt), answer, outcome="completed")
    assert verdict.accepted and verdict.status == "completed", verdict


class CurrentReadReceipt:
    """A real temporary-file read and current-content verification hook."""
    def __init__(self, path):
        self.path = path
        self.refreshes = 0

    def receipt(self):
        body = self.path.read_bytes()
        return {"verified_effects": ["read"], "completion_targets": [str(self.path)],
                "read_complete": True, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}

    def agent_verify_result(self, *, receipt, context):
        self.refreshes += 1
        return {"result_valid": self.path.exists() and
                hashlib.sha256(self.path.read_bytes()).hexdigest() == receipt["sha256"]}


@pytest.mark.parametrize("answer", ["没有找到异常，运行状态正常。", "未发现异常，运行状态正常。", "没有找到异常。"])
@pytest.mark.parametrize("outcome", ["completed", ""])
def test_current_read_evidence_controls_completion_not_failure_words(tmp_path, answer, outcome):
    path = tmp_path / "status.txt"
    path.write_text("status=healthy", encoding="utf-8")
    reader = CurrentReadReceipt(path)
    state = state_for("读取状态文件并汇报结果")
    state.context["agent_completion_contract"] = {"requirements": [
        {"id": "read-status", "effect": "read", "targets": [str(path)], "checks": {}}]}
    state.tool_calls = [{"name": "fixture_read", "status": "ok", "arguments": {"path": str(path)},
                         "data": {"state_updates": reader.receipt()}}]
    verifier = SmartCompletionVerifier(tool_capabilities={"fixture_read": {
        "effects": ["read"], "target_arguments": ["path"]}}, result_handlers={"fixture_read": reader})
    verdict = verifier.evaluate(state, answer, outcome=outcome)
    assert reader.refreshes > 0
    assert verdict.accepted and verdict.status == "completed", verdict
    path.write_text("status=changed", encoding="utf-8")
    stale = verifier.evaluate(state, answer, outcome="completed")
    assert not stale.accepted and "read-status" in stale.missing


@pytest.mark.parametrize("outcome", ["completed", ""])
def test_missing_write_cannot_hide_behind_failure_words(outcome):
    state = state_for("修改 result.txt")
    state.tool_calls = [{"name": "fixture_read", "status": "ok", "data": {}}]
    verdict = SmartCompletionVerifier().evaluate(state, "没有找到异常，运行状态正常。", outcome=outcome)
    assert not (verdict.accepted and verdict.status == "completed")


@pytest.mark.parametrize("outcome", ["failed", "blocked", "partial"])
def test_honest_structured_non_success_is_not_promoted(outcome):
    verdict = SmartCompletionVerifier().evaluate(state_for("修改 result.txt"),
        "没有完成写入，原文件保持原状。", outcome=outcome)
    assert not (verdict.accepted and verdict.status == "completed"), verdict
    if outcome in {"failed", "blocked"}:
        assert verdict.accepted and verdict.status == outcome, verdict


@pytest.mark.parametrize("prompt,allowed", [
    ("取消 {id}", True),
    ("请取消任务 {id}。", True),
    ("请帮我停止一下这个任务 {id} 吧", True),
    ("cancel {id}", True),
    ("取消 `{id}`", True),
    ("取消当前任务", True),
    ("麻烦取消这个任务吧", True),
    ("STOP {id}", True),
    ("不要取消 {id}", False),
    ("请不要停止任务 {id}", False),
    ("解释“取消 {id}”的含义，不要操作", False),
    ("“取消 {id}”", False),
    ("```\n取消 {id}\n```", False),
    ("如果失败了，再取消 {id}", False),
    ("可以取消 {id} 吗？", False),
    ("不要继续，只看状态：{id}，取消是什么意思？", False),
    ("do not cancel {id}", False),
    ("不用取消 {id}", False),
    ("取消 {id} 了吗？", False),
    ("> 取消 {id}", False),
    ("先看状态，不要取消 {id}", False),
    ("取消 {id}，不过别取消", False),
    ("取消 other-task；目标参考 {id}", False),
])
def test_task_cancel_uses_affirmative_bound_request_through_real_qq(tmp_path, prompt, allowed):
    service, tool, execution = demo_service(tmp_path)
    target = "fixture-cancel-target"
    saved = new_task(task_id=target, requester="fixture", session="private:fixture",
        request=demo_request(tool).original_request, message_id="original", contract=demo_request(tool).contract)
    saved["execution_attempt"] = 1
    saved["status"] = "queued"
    saved = service.task_store.save(saved)
    service.coordinator.submit(lambda event: AgentTaskOutcome("completed"), task_id=target,
        profile_user_id="fixture", session_id="private:fixture", defer_start=True)
    chat = Chat([{"kind": "tool", "tool_call": {"name": "agent_task_cancel", "arguments": {"task_id": target}}}])
    bridge = boundary_bridge(service, tool, chat)
    try:
        reply = bridge(turn(prompt.format(id=target), message_id="control-message"))
        work = service.coordinator.get(target)
        after = service.task_store.get(target, requester="fixture", session="private:fixture")
        assert work["cancel_requested"] is allowed, reply
        assert reply.reason == ("task_interface_cancel" if allowed else "task_operation_not_requested")
        assert reply.delivery_status == "dry_run"
        if not allowed:
            assert work["status"] == "queued" and after["revision"] == saved["revision"]
        assert tool.calls == 0 and execution.calls == [] and not tool.path.exists()
        assert len(chat.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
