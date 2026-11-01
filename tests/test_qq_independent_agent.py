"""A02/A04: real QQ host adapters, injected offline execution runtime."""
from reply_review_fixture import dialogue_review_result
from reply_review_fixture import dialogue_review_result

from dataclasses import replace
import json
import threading
import time

import pytest

from shinku.agent.loop import AgentDecision
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime import AgentBoundaryError, AgentScope, ExecutionProfile, TaskHandle
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.agent_call import QQAgentCaller
from shinku.qq.commands import QQCommandState
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


def turn(text, *, user="fixture", channel="fixture", message_id="fixture-message"):
    message = IncomingMessage.from_event({"post_type": "message", "message_type": "private",
        "user_id": user, "self_id": "fixture-bot", "message_id": message_id,
        "message": [{"type": "text", "data": {"text": text}}]})
    return QQTurn(conversation_id=channel, conversation_type="private", primary_message_id=message_id,
                  message_ids=(message_id,), combined_text=text, messages=(message,))


def wait(caller, message, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        view = caller.status(message, task_id)
        if view["status"] not in {"running", "queued"} and not view["worker_active"]:
            return view
        time.sleep(.005)
    raise AssertionError("host task did not settle")


def test_a02_qq_caller_uses_port_for_submit_status_result_and_resume(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    caller = QQAgentCaller(service.bind)
    request = demo_request(tool)
    message = turn(request.original_request)
    try:
        service.executor.profile = ExecutionProfile(None)
        handle = caller.submit(message, request)
        assert caller.submit(message, request).task_id == handle.task_id
        blocked = wait(caller, message, handle.task_id)
        assert caller.result(message, handle.task_id)["execution"]["status"] == "blocked"
        service.executor.profile = ExecutionProfile(runtime)
        caller.resume(message, TaskHandle(handle.task_id, blocked["attempt"], blocked["version"]), request_key="resume")
        assert wait(caller, message, handle.task_id)["status"] == "completed"
        before = tool.calls, len(runtime.calls)
        assert caller.result(message, handle.task_id)["verification"]["accepted"] is True
        caller.status(message, handle.task_id)
        assert before == (tool.calls, len(runtime.calls)) == (1, 2)
        with pytest.raises(AgentBoundaryError, match="task_not_found"):
            caller.result(turn("read result", user="other"), handle.task_id)
        with pytest.raises(AgentBoundaryError, match="original_request_mismatch"):
            caller.submit(message, replace(request, original_request="A summary replacing the goal"))
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def make_bridge(service, tool, *, asynchronous=False):
    commands = QQCommandState(agent_enabled=True, permission_mode="ultra")
    class ChatRuntime:
        def __init__(self):
            self.calls = []
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            review = dialogue_review_result(kwargs)
            if review is not None:
                return review
            raise AssertionError("execution must not use persona runtime")
    chat = ChatRuntime()
    bridge = QQAgentBridge(runtime=chat, system_prompt="PERSONA_PRIVATE_SENTINEL", command_state=commands,
        tool_host=service.executor.tool_host, task_store=service.task_store, task_coordinator=service.coordinator,
        task_contract_enabled=True, async_agent_enabled=asynchronous,
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type: "low"}),
        execution_profile=service.executor.profile,
        task_budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30})
    return bridge, chat


def prepare_context(service, tool, *, task_id="admitted"):
    request = demo_request(tool)
    task = service.task_store.save(new_task(task_id=task_id, requester="fixture", session="private:fixture",
        request=request.original_request, message_id="fixture-message", contract=request.contract))
    from shinku.agent.request_scope import RequestScope
    context = {"agent_task_state": task, "agent_completion_contract": task["contract"],
        "agent_request_scope": RequestScope("fixture", "private:fixture", "fixture-message",
                                            (("fixture-message", request.original_request),)),
        "agent_permission_mode": "ultra", "agent_authority_revision": 0,
        "qq_turn": turn(request.original_request), "memory_context": "IRRELEVANT_MEMORY_SENTINEL"}
    return request, context


def test_a04_bridge_executes_admitted_task_without_persona_or_replacing_ledger(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    bridge, chat = make_bridge(service, tool)
    request, context = prepare_context(service, tool)
    try:
        result = bridge._run_agent(task_id="admitted", context=context,
            initial_transcript=[{"role": "assistant", "content": "PRIVATE_CHAT_SENTINEL"}])
        assert result.status == "completed" and result.completion["accepted"]
        assert len(runtime.calls) == 2 and chat.calls == [] and tool.calls == 1
        all_calls = json.dumps(runtime.calls)
        assert request.original_request in all_calls
        assert all(v not in all_calls for v in ("PERSONA_PRIVATE_SENTINEL", "PRIVATE_CHAT_SENTINEL", "IRRELEVANT_MEMORY_SENTINEL"))
        assert bridge.execution_adapter.task_store is bridge.task_store is service.task_store
        assert bridge.execution_adapter.coordinator is service.coordinator
        assert service.coordinator.list() == []
        payload = bridge.agent_calls.result(context["qq_turn"], "admitted")
        assert payload["execution"]["status"] == "completed" and payload["task_id"] == "admitted"
        before = tool.calls, len(runtime.calls)
        bridge.agent_calls.status(context["qq_turn"], "admitted")
        assert before == (tool.calls, len(runtime.calls))
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_a04_existing_single_worker_executes_slices_without_queuing_second_job(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    service.executor.profile = replace(service.executor.profile, max_rounds=1)
    bridge, _ = make_bridge(service, tool, asynchronous=True)
    _, context = prepare_context(service, tool)
    context.update(agent_work_id="existing-work", agent_work_attempt=1, agent_sliced_execution=True)
    from shinku.tasks.coordinator import AgentTaskContinuation, AgentTaskOutcome
    result_box = []
    checkpoint = None
    def handler(event):
        nonlocal checkpoint
        context["agent_task_interrupt"] = lambda: "task_canceled" if event.is_set() else ""
        result = bridge._run_agent(task_id="admitted", context=context, initial_transcript=[], checkpoint=checkpoint)
        checkpoint = result.checkpoint
        result_box.append(result)
        if result.status == "yielded":
            return AgentTaskContinuation(handler)
        return AgentTaskOutcome(result.status)
    try:
        service.coordinator.submit(handler, task_id="existing-work", profile_user_id="fixture", session_id="private:fixture",
                                   metadata={"agent_task_id": "admitted"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = service.coordinator.get("existing-work")
            if not record["worker_active"] and record["status"] == "completed":
                break
            if record["status"] == "failed":
                raise AssertionError(record["error"])
            time.sleep(.01)
        else:
            raise AssertionError("single worker deadlocked")
        assert len(service.coordinator.list()) == 1 and tool.calls == 1
        assert [v.status for v in result_box] == ["yielded", "completed"]
        assert len(runtime.calls) == 2
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_a03_stale_host_revision_or_checkpoint_does_not_execute(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    bridge, _ = make_bridge(service, tool)
    _, context = prepare_context(service, tool)
    changed = service.task_store.get("admitted", requester="fixture", session="private:fixture")
    service.task_store.save(changed)
    try:
        with pytest.raises(AgentBoundaryError, match="task_version_conflict"):
            bridge._run_agent(task_id="admitted", context=context, initial_transcript=[])
        assert tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_default_bridge_remains_on_existing_compatibility_path():
    class Runtime:
        def call_chat_json(self, **kwargs):
            self.dialogue_calls = getattr(self, "dialogue_calls", [])
            self.dialogue_calls.append(kwargs)
            review = dialogue_review_result(kwargs)
            if review is not None:
                return review
            return {"kind": "final", "text": "普通聊天"}
    bridge = QQAgentBridge(runtime=Runtime(), system_prompt="Existing persona")
    result = bridge(turn("你好"))
    assert result.final_text == "普通聊天"
    assert bridge.execution_adapter is None and bridge.agent_calls is None
    assert bridge.task_store.candidates(requester="fixture", session="private:fixture") == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a02_whole_qq_admission_execution_and_delivery_path(tmp_path, asynchronous):
    service, tool, runtime = demo_service(tmp_path)
    request = demo_request(tool)
    class Chat:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            return {"kind": "tool", "tool_call": {"type": tool.tool_type, "path": str(tool.path)},
                    "task_link": {"mode": "new"}, "task_contract": request.contract}
    chat = Chat()
    commands = QQCommandState(agent_enabled=True, permission_mode="ultra")
    message = turn(request.original_request)
    bridge = QQAgentBridge(runtime=chat, system_prompt="PERSONA_CHAT_SENTINEL", command_state=commands,
        tool_host=service.executor.tool_host, task_store=service.task_store, task_coordinator=service.coordinator,
        task_contract_enabled=True, async_agent_enabled=asynchronous, execution_profile=service.executor.profile,
        task_budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30})
    try:
        reply = bridge(message)
        if asynchronous:
            assert reply.agent_status == "waiting_background" and reply.delivery_status == "queued", reply
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                work = service.coordinator.get(reply.background_task_id)
                if not work["worker_active"] and work["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
            assert work["status"] == "completed", work
        else:
            assert reply.agent_status == "completed", reply
        tasks = service.task_store.candidates(requester="fixture", session="private:fixture")
        assert len(tasks) == 1 and tool.calls == 1 and chat.calls == 1 and len(runtime.calls) == 1
        result = bridge.agent_calls.result(message, tasks[0]["task_id"])
        assert result["execution"]["status"] == "completed" and result["verification"]["accepted"]
        assert "PERSONA_CHAT_SENTINEL" not in json.dumps(runtime.calls)
        assert len(service.coordinator.list()) == (1 if asynchronous else 0)
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_fast_query_result_contract_has_no_task_or_implicit_execution():
    from shinku.agent_runtime.results import capture_query_result
    from shinku.conversation import AnswerEvidence
    from shinku.qq.reply_models import QQAgentReply
    from shinku.tools.invocation import ToolResultEnvelope
    from shinku.agent.request_scope import RequestScope
    message = turn("Read the known source")
    context = {"agent_request_scope": RequestScope("fixture", "private:fixture", "fixture-message",
                                                 (("fixture-message", message.combined_text),))}
    receipt = ToolResultEnvelope("fixture-call", "ok", "ignored raw source log", {
        "tool_type": "public_fixture", "state_updates": {"verified_effects": ["read"],
            "completion_targets": ["https://example.invalid/source"], "http_status": 200}})
    # A successful read without answer verification remains unconfirmed.
    capture_query_result(context, message, QQAgentReply("completed", delivery_status="failed"),
                         [receipt], route="public_query")
    result = context["agent_query_result"]
    assert result["execution"]["status"] == "unverified" and result["task_identity_missing"]
    assert result["delivery"]["status"] == "failed" and result["task_id"] == ""
    assert result["evidence"][0]["receipt"]["http_status"] == 200
    assert "ignored raw source log" not in json.dumps(result)
