"""P3: real QQ routing reads facts without executing a task or picking one."""
from dataclasses import replace
import json

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.task_state import new_task
from shinku.conversation import ConversationScope
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage, MessageSegment
from test_qq_independent_agent import turn


class NoCalls:
    def call_chat_json(self, **kwargs):
        raise AssertionError("status inquiry must not call a model")
    def chat_supports_native_tools(self):
        return True


def seed(bridge, key="first", *, accepted=True):
    task = new_task(task_id=key, requester="fixture", session="private:fixture",
        request="生成报告" if key == "first" else "核对第二项内容", message_id=key + "-request", contract=None)
    task.update(status="completed", execution_attempt=1, completion_attempt=1,
        completion={"accepted": accepted, "status": "completed" if accepted else "partial",
                    "satisfied": ["报告"], "missing": [] if accepted else ["内容核验"]})
    task = bridge.task_store.save(task)
    scope = ConversationScope("private", "fixture", "fixture")
    bridge.conversation_runtime.begin(scope, key + "-turn")
    bridge.conversation_runtime.step(key + "-turn", task_id=key, task_attempt=1, task_version=task["revision"])
    bridge.conversation_runtime.finish(key + "-turn", status="completed", delivery_status="failed")
    return task


def test_state_question_reads_execution_verification_and_failed_delivery_without_model():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    task = seed(bridge)
    before = task["revision"]
    reply = bridge.handle_turn(turn("任务进度怎么样？"))
    assert reply.reason == "task_status_query" and reply.agent_status == "chat"
    assert "验收通过" in reply.final_text and "送达失败" in reply.final_text
    assert bridge.task_coordinator is None or not bridge.task_coordinator.list()
    assert bridge.task_store.get("first", requester="fixture", session="private:fixture")["revision"] == before


def test_multiple_tasks_ask_for_binding_then_quoted_original_selects_right_task():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    seed(bridge)
    seed(bridge, "second", accepted=False)
    reply = bridge.handle_turn(turn("做到哪里了？", message_id="inquiry"))
    assert reply.reason == "task_binding_ambiguous" and "first" in reply.final_text and "second" in reply.final_text
    question = turn("完成了吗？", message_id="quoted")
    original = question.messages[0]
    question = replace(question, messages=(replace(original, segments=(
        MessageSegment.from_value({"type": "reply", "data": {"id": "first-request"}}), *original.segments)),))
    reply = bridge.handle_turn(question)
    assert reply.reason == "task_status_query" and "验收通过" in reply.final_text
    assert "内容核验" not in reply.final_text


def test_explicit_id_cannot_read_other_requester_or_other_channel():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    seed(bridge)
    for message in [turn("任务号 first 的进度", user="other"), turn("任务号 first 的进度", channel="other")]:
        reply = bridge.handle_turn(message)
        assert reply.reason == "task_not_found" and "生成报告" not in reply.final_text


def test_admission_failure_without_task_survives_ordinary_chat():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    scope = ConversationScope("private", "fixture", "fixture")
    runtime = bridge.conversation_runtime
    runtime.begin(scope, "rejected")
    runtime.step("rejected", route="agent", phase="preparation")
    runtime.finish("rejected", status="blocked", reason="tool_unavailable_or_unauthorized")
    for index in range(6):
        runtime.begin(scope, f"ordinary-{index}")
        runtime.finish(f"ordinary-{index}", status="chat")
    reply = bridge.handle_turn(turn("任务进展如何？"))
    assert reply.reason == "task_admission_unresolved" and "尚未创建" in reply.final_text
    assert not bridge.task_store.candidates(requester="fixture", session="private:fixture")
    context = bridge._prompt_operational_context(turn("现在呢？"), "")
    assert any(row["reason"] == "tool_unavailable_or_unauthorized" for row in context["task_operations"])


def test_prompt_has_host_view_and_does_not_promote_previous_persona_report():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    task = seed(bridge, accepted=False)
    task["last_report"] = "已经全部部署成功。"
    task = bridge.task_store.save(task)
    state = AgentState(task_id="chat", context={"qq_turn": turn("聊一下刚才的任务"), "agent_task_candidates": [task]})
    prompt = bridge._build_prompt(state)
    assert "task_views" in prompt.user_prompt and "内容核验" in prompt.user_prompt
    assert "已经全部部署成功" not in prompt.user_prompt
    assert "状态查询不授权" in "\n".join(prompt.system_extra_blocks)


def test_status_phrase_plus_mutation_is_not_read_only_routed():
    from shinku.qq.task_status import parse_status_query
    for text in ["任务进度怎么样？然后重新执行", "完成了吗？把报告发给别人", "开始任务", "取消任务", "增加任务预算"]:
        assert parse_status_query(text) is None


def test_missing_task_store_stays_unknown_in_chat_status():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    def broken(**kwargs):
        raise OSError("private data")
    bridge.task_store.candidates = broken
    reply = bridge.handle_turn(turn("任务进度怎么样？"))
    assert reply.reason == "task_store_unavailable" and "无法读取" in reply.final_text
    assert "private data" not in reply.final_text


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("delivery", ["dry_run", "sent", "failed", "uncertain"])
def test_full_admission_execution_verification_delivery_and_state_query(tmp_path, asynchronous, independent, delivery):
    import time
    from shinku.agent_runtime.demo import demo_service, demo_request
    from shinku.qq.commands import QQCommandState
    from shinku.qq.delivery import DeliveryResult
    from shinku.qq.task_status import reader_for_bridge
    from shinku.agent_runtime import AgentScope
    service, tool, execution = demo_service(tmp_path)
    request = demo_request(tool)
    class Chat:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return {"kind": "final", "text": "The isolated file was created and verified.", "task_outcome": "completed"}
            return {"kind": "tool", "tool_call": {"type": tool.tool_type, "path": str(tool.path)},
                    "task_link": {"mode": "new"}, "task_contract": request.contract}
    chat = Chat()
    sent = []
    def sender(message):
        sent.append(message)
        if delivery == "uncertain":
            raise TimeoutError("simulated unknown send")
        return DeliveryResult(delivery == "sent", delivery, ("outbound-report",) if delivery == "sent" else (),
                              definitive_not_sent=delivery == "failed")
    bridge = QQAgentBridge(runtime=chat, system_prompt="fixture", tool_host=service.executor.tool_host,
        command_state=QQCommandState(agent_enabled=True, permission_mode="ultra"),
        task_store=service.task_store, task_coordinator=service.coordinator, task_contract_enabled=True,
        async_agent_enabled=asynchronous, execution_profile=service.executor.profile if independent else None,
        sender=None if delivery == "dry_run" else sender,
        task_budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30})
    try:
        reply = bridge.handle_turn(turn(request.original_request, message_id="original-request"))
        if asynchronous:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                record = service.coordinator.get(reply.background_task_id)
                if not record["worker_active"] and record["status"] not in {"queued", "running"}:
                    break
                time.sleep(.005)
            assert record["status"] == "completed", record
        else:
            assert reply.agent_status == "completed", reply
        task = bridge.task_store.candidates(requester="fixture", session="private:fixture")[0]
        view = reader_for_bridge(bridge).get(AgentScope("fixture", "private:fixture"), task["task_id"])
        assert view["status"] == "completed" and view["verification"]["verified"], view
        assert view["delivery"]["status"] == delivery, view
        assert view["delivery"]["sent"] is (delivery == "sent")
        assert view["attempt"] == 1 and view["delivery"]["version"] == task["revision"]
        if independent:
            assert bridge.agent_calls.status(turn("status"), task["task_id"])["delivery"] == view["delivery"]
        before = (tool.calls, chat.calls, len(execution.calls), len(sent), task["revision"])
        artifact_before = tool.path.read_bytes()
        result = bridge.handle_turn(turn("任务进度怎么样？", message_id="query"))
        assert result.agent_status == "chat" and "验收通过" in result.final_text
        if delivery in {"dry_run", "sent"}:
            assert result.reason == "task_status_query"
        else:
            assert "segment_" in result.reason
        after = bridge.task_store.get(task["task_id"], requester="fixture", session="private:fixture")
        assert before[:3] == (tool.calls, chat.calls, len(execution.calls)) and before[4] == after["revision"]
        assert tool.path.read_bytes() == artifact_before
        after_view = reader_for_bridge(bridge).get(AgentScope("fixture", "private:fixture"), task["task_id"])
        assert after_view["delivery"] == view["delivery"] and after_view["verification"] == view["verification"]
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_real_admission_refusal_survives_next_inquiry_without_execution():
    import test_agent_async_processor as helpers
    from threading import Event
    from shinku.tasks.coordinator import AgentTaskCoordinator
    coordinator = AgentTaskCoordinator()
    tool = helpers.BlockingInspector(Event(), Event())
    bridge = helpers.QQAsyncProcessorTests().make_bridge(coordinator, tool)
    try:
        first = bridge.handle_turn(helpers.turn("inspect", sender="other"))
        assert first.agent_status == "blocked" and tool.calls == 0
        second = bridge.handle_turn(helpers.turn("任务进度怎么样？", sender="other"))
        assert second.reason == "task_admission_unresolved" and "尚未创建" in second.final_text
        assert not coordinator.list() and tool.calls == 0
    finally:
        coordinator.shutdown(wait=True)


def test_new_unadmitted_request_cannot_inherit_older_completed_task_status():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    seed(bridge)
    scope = ConversationScope("private", "fixture", "fixture")
    runtime = bridge.conversation_runtime
    runtime.begin(scope, "new-admission")
    runtime.step("new-admission", route="agent", phase="preparation")
    runtime.finish("new-admission", status="blocked", reason="tool_unavailable_or_unauthorized")
    reply = bridge.handle_turn(turn("完成了吗？"))
    assert reply.reason == "task_admission_unresolved" and "验收通过" not in reply.final_text
    old = bridge.handle_turn(turn("任务号 first 的进度", message_id="old-query"))
    assert old.reason == "task_status_query" and "验收通过" in old.final_text


def test_unavailable_operations_cannot_bind_implicit_query_to_old_task():
    bridge = QQAgentBridge(runtime=NoCalls(), system_prompt="fixture")
    seed(bridge)
    bridge.conversation_runtime.task_receipts = lambda scope: {"available": False, "records": []}
    reply = bridge.handle_turn(turn("完成了吗？"))
    assert reply.reason == "conversation_state_unavailable" and "验收通过" not in reply.final_text
    explicit = bridge.handle_turn(turn("任务号 first 的进度", message_id="explicit"))
    assert explicit.reason == "task_status_query" and "验收通过" in explicit.final_text
