"""P3b: actual QQ admission, Agent, verifier, expression and test transport."""
import copy
from dataclasses import replace
import json
import time

import pytest

from shinku.agent.loop import AgentRunResult
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.conversation import AnswerEvidence
from shinku.conversation.models import text_digest
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.result_expression import ExpressionProfile
from shinku.result_expression.facts import identity
from shinku.tools.invocation import ToolResultEnvelope
from test_qq_independent_agent import turn
from test_result_expression import Model


def wait_work(service, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        work = service.coordinator.get(task_id)
        if not work["worker_active"] and work["status"] not in {"queued", "running"}:
            return work
        time.sleep(.01)
    raise AssertionError("fixture worker did not finish")


def make_host(tmp_path, *, asynchronous=False, independent=True, delivery="sent", model=None, task_store=None,
              coordinator=None):
    service, tool, execution = demo_service(tmp_path)
    if task_store is not None:
        from shinku.agent_runtime.task_view import TaskViewReader
        service.task_store.close()
        service.task_store = task_store
        service.task_views = TaskViewReader(task_store, coordinator or service.coordinator)
    if coordinator is not None:
        service.coordinator.shutdown(wait=True)
        service.coordinator = coordinator
    request = demo_request(tool)
    class Chat:
        def __init__(self):
            self.calls = []
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **options):
            self.calls.append(options)
            if len(self.calls) == 1:
                return {"kind": "tool", "tool_call": {"type": tool.tool_type, "path": str(tool.path)},
                    "task_link": {"mode": "new"}, "task_contract": request.contract}
            return {"kind": "final", "text": "Execution model report, created and verified.", "task_outcome": "completed"}
    chat, sent = Chat(), []
    def sender(message):
        sent.append(message)
        if delivery == "uncertain":
            raise TimeoutError("fixture unknown send")
        return DeliveryResult(delivery == "sent", delivery,
            (f"fixture-out-{len(sent)}",) if delivery == "sent" else (), definitive_not_sent=delivery == "failed")
    bridge = QQAgentBridge(runtime=chat, system_prompt="PERSONA_CHAT_PRIVATE_SENTINEL",
        tool_host=service.executor.tool_host, task_store=service.task_store, task_coordinator=service.coordinator,
        task_contract_enabled=True, async_agent_enabled=asynchronous,
        command_state=QQCommandState(agent_enabled=True, permission_mode="ultra"),
        execution_profile=service.executor.profile if independent else None,
        expression_profile=ExpressionProfile(model or Model(), configuration_id="offline-fixture",
            style_prompt="真红口吻；EXPRESSION_PERSONA_PRIVATE_SENTINEL"),
        sender=sender if delivery != "dry_run" else None,
        task_budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30})
    return service, tool, execution, request, bridge, chat, sent


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("delivery", ["sent", "failed", "uncertain", "dry_run"])
def test_full_pipeline_execution_expression_delivery_and_read_only_query(tmp_path, asynchronous, independent, delivery):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path,
        asynchronous=asynchronous, independent=independent, delivery=delivery)
    memories, dialogue = [], []
    bridge._record_agent_task_memory = lambda *args, **kw: memories.append(kw)
    bridge._record_assistant_reply = lambda *args, **kw: dialogue.append((args, kw))
    try:
        reply = bridge(turn(request.original_request))
        if asynchronous:
            assert reply.agent_status == "waiting_background" and reply.delivery_status == "queued"
            work = wait_work(service, reply.background_task_id)
            assert work["status"] == "completed", work
        else:
            assert reply.agent_status == "completed", reply
        task = service.task_store.candidates(requester="fixture", session="private:fixture")[0]
        result = task["execution_result"]
        key = canonical_hash(identity(result))
        record = bridge.result_expression.archive.get(key, scope=AgentScope("fixture", "private:fixture"))
        assert record["status"] == "ready", record
        assert record["delivery"]["status"] == delivery
        assert record["canonical_result"]["verification"]["accepted"] is True
        assert len(bridge.result_expression.profile.runtime.calls) == 1 and tool.calls == 1
        assert "PERSONA_CHAT_PRIVATE_SENTINEL" not in json.dumps(execution.calls)
        assert len(memories) == 1 and "原始目标" in memories[0]["outcome"]
        assert "……结果整理" not in memories[0]["outcome"] and "Execution model report" not in memories[0]["outcome"]
        if delivery == "sent":
            actual = "\n".join(part.text for msg in sent for part in msg.segments if part.kind == "text")
            assert actual == record["text"]
            assert record["delivered_content"]["content_complete"]
            assert dialogue and dialogue[0][1]["delivery_metadata"]["execution_result_hash"] == result["result_hash"]
            assert dialogue[0][1]["delivery_metadata"]["expression_is_execution_evidence"] is False
        else:
            assert dialogue == []
        before = (tool.calls, len(execution.calls), len(chat.calls), len(bridge.result_expression.profile.runtime.calls),
                  task["revision"], tool.path.read_bytes())
        bridge.sender = None  # Observe the read-only answer without another transport failure.
        state = bridge(turn("任务进度怎么样？", message_id="fixture-status"))
        assert state.reason == "task_status_query" and "验收通过" in state.final_text
        after_task = service.task_store.get(task["task_id"], requester="fixture", session="private:fixture")
        assert before == (tool.calls, len(execution.calls), len(chat.calls), len(bridge.result_expression.profile.runtime.calls),
                          after_task["revision"], tool.path.read_bytes())
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_repeated_terminal_callback_does_not_create_version_call_or_delivery(tmp_path):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path)
    try:
        reply = bridge(turn(request.original_request))
        assert reply.sent
        task = service.task_store.candidates(requester="fixture", session="private:fixture")[0]
        before = (task["revision"], len(sent), tool.calls, len(execution.calls))
        model = bridge.result_expression.profile.runtime
        repeated = bridge._finish_result(turn(request.original_request), AgentRunResult(status="completed", rounds=1,
            final_text=task["last_report"], completion=copy.deepcopy(task["completion"]), reason=task["stop_reason"]),
            context={"agent_task_state": task}, initial_transcript=[], task_id=task["task_id"])
        current = service.task_store.get(task["task_id"], requester="fixture", session="private:fixture")
        assert repeated.delivery_status == "suppressed" and len(model.calls) == 1
        assert before == (current["revision"], len(sent), tool.calls, len(execution.calls))
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_bad_polish_falls_back_through_same_sender_without_execution_retry(tmp_path):
    model = Model(lambda c, p: {**c, "text": "已上线并且发送成功"})
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path, model=model)
    try:
        reply = bridge(turn(request.original_request))
        assert reply.sent and "验收通过" in reply.final_text and "已上线并且" not in reply.final_text
        assert tool.calls == 1 and len(execution.calls) == 1 and len(model.calls) == 1
        task = service.task_store.candidates(requester="fixture", session="private:fixture")[0]
        record = bridge.result_expression.archive.get(canonical_hash(identity(task["execution_result"])),
            scope=AgentScope("fixture", "private:fixture"))
        assert record["status"] == "fallback" and record["delivery"]["status"] == "sent"
        assert task["completion"]["accepted"] is True
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("stop", ["cancel", "permission", "version"])
def test_change_during_polish_discards_and_does_not_send_or_overwrite_task(tmp_path, stop):
    holder = {}
    def invalidate(candidate, payload):
        bridge = holder["bridge"]
        task = bridge.task_store.candidates(requester="fixture", session="private:fixture")[0]
        if stop == "permission":
            bridge.command_state.set_permission_mode("readonly")
        else:
            if stop == "cancel":
                task["cancel_requested"] = True
            else:
                task["execution_attempt"] += 1
                task["status"] = "queued"
            holder["changed"] = bridge.task_store.save(task)
        return candidate
    model = Model(invalidate)
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path, model=model)
    holder["bridge"] = bridge
    try:
        reply = bridge(turn(request.original_request))
        assert reply.delivery_status == "suppressed" and sent == []
        assert len(model.calls) == 1 and tool.calls == 1 and len(execution.calls) == 1
        if "changed" in holder:
            now = bridge.task_store.get(holder["changed"]["task_id"], requester="fixture", session="private:fixture")
            assert now == holder["changed"]
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_permission_change_after_first_segment_stops_remaining_segments(tmp_path):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path)
    original = bridge.sender
    def sender(message):
        receipt = original(message)
        bridge.command_state.set_permission_mode("readonly")
        return receipt
    bridge.sender = sender
    try:
        reply = bridge(turn(request.original_request))
        assert not reply.sent and reply.delivery_status == "partial" and len(sent) == 1
        assert tool.calls == 1 and len(bridge.result_expression.profile.runtime.calls) == 1
        task = service.task_store.candidates(requester="fixture", session="private:fixture")[0]
        record = bridge.result_expression.archive.get(canonical_hash(identity(task["execution_result"])),
            scope=AgentScope("fixture", "private:fixture"))
        assert record["delivery"]["status"] == "partial"
        assert record["delivered_content"]["text"] != record["text"]
        assert record["canonical_result"]["verification"]["accepted"] is True
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("validated", [False, True])
def test_fast_query_adapts_existing_receipts_without_creating_or_running_task(tmp_path, validated):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path)
    text = "实际记录为 12.5 ms，条件为离线样本，来源 https://example.invalid/source。"
    evidence = AnswerEvidence("public", "verified" if validated else "unverified", validated, 1, 1,
                              text_digest(text) if validated else "")
    receipt = ToolResultEnvelope("fixture-read", "ok", "RAW_PAGE_PRIVATE_SENTINEL", {
        "tool_type": "fixture_read", "state_updates": {"http_status": 200, "url": "https://example.invalid/source",
            "result_facts": [{"label": "响应", "value": 12.5, "unit": "ms", "condition": "离线样本"}]}})
    context = {}
    try:
        reply = bridge._express_query_result(turn("查询离线样本"), text, context=context, status="completed",
            reason="fixture_query", route="public_query", envelopes=[receipt], answer_evidence=evidence)
        assert reply.sent and tool.calls == 0 and execution.calls == [] and chat.calls == []
        assert service.task_store.candidates(requester="fixture", session="private:fixture") == []
        result = context["agent_query_result"]
        assert result["task_identity_missing"] and result["task_id"] == ""
        assert "12.5 ms" in reply.final_text and "RAW_PAGE_PRIVATE_SENTINEL" not in json.dumps(context)
        if validated:
            assert text in reply.final_text and result["verified_answer"]["text"] == text
        else:
            assert "尚未确认" in reply.final_text and "verified_answer" not in result and reply.agent_status == "partial"
        duplicate = bridge._express_query_result(turn("查询离线样本"), text, context=context, status="completed",
            reason="fixture_query", route="public_query", envelopes=[receipt], answer_evidence=evidence)
        assert duplicate.delivery_status == "suppressed" and len(bridge.result_expression.profile.runtime.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_expression_does_not_run_after_ordinary_chat_or_each_slice(tmp_path):
    model = Model()
    bridge = QQAgentBridge(runtime=type("Chat", (), {"call_chat_json": lambda *a, **kw: {"kind": "final", "text": "你好"}})(),
        system_prompt="fixture", task_contract_enabled=True, expression_profile=ExpressionProfile(model))
    reply = bridge(turn("你好"))
    assert reply.final_text == "你好" and model.calls == []
    bridge.task_store.close()
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path, asynchronous=True, model=model)
    bridge.execution_adapter.executor.profile = replace(bridge.execution_adapter.executor.profile, max_rounds=1)
    try:
        reply = bridge(turn(request.original_request))
        assert reply.delivery_status == "queued"
        assert wait_work(service, reply.background_task_id)["status"] == "completed"
        assert tool.calls == 1 and len(model.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_full_public_fast_track_uses_its_verified_answer_and_no_heavy_agent():
    from test_qq_public_query_fast_track import Runtime, SourceTool, PersistentReader, call, final, turn as query_turn
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    from shinku.tasks.coordinator import AgentTaskCoordinator
    search, reader = SourceTool("private_web_search"), SourceTool("private_browser_open")
    chat = Runtime(call(), call("private_browser_open", {"url": "https://weather.example/source"}), final())
    model = Model()
    coordinator = AgentTaskCoordinator(max_workers=1)
    host = QQAgentBridge(runtime=chat, system_prompt="fixture persona", task_contract_enabled=True,
        tool_host=ToolHost(ToolRegistry({search.tool_type: search, reader.tool_type: PersistentReader(reader)})),
        allowed_tool_names={search.tool_type, reader.tool_type},
        policy=ExecutionPolicy(risk_by_tool={search.tool_type: "low", reader.tool_type: "low"}),
        master_user_ids=("master",), owner_gate_required=True,
        command_state=QQCommandState(agent_enabled=True, permission_mode="full"),
        async_agent_enabled=True, task_coordinator=coordinator,
        expression_profile=ExpressionProfile(model))
    def heavy(**options):
        raise AssertionError("fast query must not enter Agent or task preparation")
    host._run_agent = host._prepare_agent_task = heavy
    try:
        reply = host(query_turn())
        assert reply.agent_status == "completed" and reply.delivery_status == "dry_run", reply
        assert "18°C" in reply.final_text and "https://weather.example/source" in reply.final_text
        assert len(chat.calls) == 4 and len(model.calls) == len(search.calls) == len(reader.calls) == 1
        assert host.task_store.candidates(requester="master", session="private:master") == []
        prompt = model.calls[0]["user_prompt"]
        assert "page_text" not in prompt and "read_sources" not in prompt
    finally:
        coordinator.shutdown(wait=True)
        host.task_store.close()


def test_full_perception_fast_track_renders_executor_data_without_trusting_model_claim():
    from types import SimpleNamespace
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    class Sensor:
        tool_type = "fixture_sensor"
        agent_permission_class = "read"
        agent_progress_class = "observe"
        agent_completion_effects = ()
        calls = 0
        def tool_metadata(self):
            return SimpleNamespace(description="Read fixture sample", input_schema={"type": "object", "properties": {}})
        def normalize_call(self, value):
            return value if isinstance(value, dict) else None
        def execute(self, **options):
            self.calls += 1
            return {"tool_type": self.tool_type, "followup_context": "sample is 2 s",
                "state_updates": {"result_facts": [{"label": "校准值", "value": 2, "unit": "s", "condition": "离线"}]}}
    sensor = Sensor()
    class Chat:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **options):
            self.calls += 1
            if self.calls == 1:
                return {"kind": "tool", "tool_call": {"type": sensor.tool_type}, "task_link": {"mode": "none"}}
            return {"kind": "final", "text": "校准成功，已全部部署上线。"}
    chat, model = Chat(), Model()
    host = QQAgentBridge(runtime=chat, system_prompt="fixture", task_contract_enabled=True,
        tool_host=ToolHost(ToolRegistry({sensor.tool_type: sensor})),
        policy=ExecutionPolicy(risk_by_tool={sensor.tool_type: "low"}),
        command_state=QQCommandState(agent_enabled=True, permission_mode="full"),
        expression_profile=ExpressionProfile(model))
    try:
        reply = host(turn("当前校准值是多少？"))
        assert reply.delivery_status == "dry_run" and reply.agent_status == "partial", reply
        assert "2 s" in reply.final_text and "尚未确认" in reply.final_text
        assert "已全部部署上线" not in reply.final_text
        assert sensor.calls == 1 and chat.calls == 2 and len(model.calls) == 1
        assert host.task_store.candidates(requester="fixture", session="private:fixture") == []
    finally:
        host.task_store.close()


def test_background_expression_time_does_not_consume_execution_active_time(tmp_path):
    holder = {}
    def observe(candidate, payload):
        bridge = holder["bridge"]
        session = next(iter(bridge._slice_sessions.values()))
        budget = session["budget"]
        before = budget.snapshot()
        time.sleep(.15)
        after = budget.snapshot()
        assert before == after
        assert budget._depth == 0
        holder["snapshot"] = after
        return candidate
    model = Model(observe)
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path, asynchronous=True, model=model)
    holder["bridge"] = bridge
    try:
        reply = bridge(turn(request.original_request))
        assert wait_work(service, reply.background_task_id)["status"] == "completed"
        assert "snapshot" in holder and len(model.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
