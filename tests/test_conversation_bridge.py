from reply_review_fixture import dialogue_review_result
import json
from threading import Event

import pytest
import test_agent_async_processor as async_helpers
from test_qq_chat_agent_split import _turn

from shinku.agent.loop import AgentDecision
from shinku.conversation import AnswerEvidence, ConversationRuntime, ConversationScope
from shinku.hosts.tool_host import ToolHost
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.tools import MemorySearchTool
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry


@pytest.mark.parametrize("lookup_status,expected", [("candidates", "completed"),
    ("not_found", "partial"), ("scope_limited", "blocked"), ("read_failed", "failed")])
@pytest.mark.parametrize("routing", ["native_missing_metadata", "compact_mislabeled_public_query"])
def test_native_history_call_without_task_metadata_uses_real_query_not_contract_repair(tmp_path, lookup_status, expected, routing):
    memory = MemoryService(store=MemoryStore(tmp_path / "memory.db"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("u-19",)), clock=lambda: 1000)
    memory.record_turn(conversation_id="u-19", conversation_type="private", message_id="seed", text="夜色落在窗前。",
        metadata={"sender_id": "u-19"})
    tool = MemorySearchTool(memory)
    actual_execute = tool.execute
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        if lookup_status == "read_failed":
            return {"status": "error", "ok": False, "followup_context": "读取失败", "state_updates": {
                "memory_lookup": {"status": "read_failed", "query_completed": False}, "memory_inspection": {"records": []}}}
        return actual_execute(**kwargs)

    tool.execute = execute
    if lookup_status == "not_found":
        memory.inspect = lambda **kw: {"scope": "private:u-19", "retrieval_scopes": ["private:u-19"],
            "records": [], "followup_context": "未找到"}

    class Runtime:
        def __init__(self):
            self.calls = []

        def chat_supports_native_tools(self):
            return True

        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            review = dialogue_review_result(kwargs)
            if review is not None:
                return review
            if "历史答案复核" in "\n".join(kwargs.get("system_extra_blocks", [])):
                source = next(row for row in json.loads(kwargs["user_prompt"])["actual_records"] if "夜色落在窗前。" in row["text"])
                return {"disposition": "found", "claims": [{"source_id": source["id"],
                    "quote": "夜色落在窗前。", "answer_fragment": "夜色落在窗前。"}],
                    "coverage": [{"field": "原问题", "supported": True, "claim_indices": [0]}],
                    "answer_parts": [{"text": "那句是", "kind": "non_factual"},
                                     {"text": "夜色落在窗前。", "kind": "supported", "claim_indices": [0]}]}
            if kwargs.get("native_tools"):
                if routing == "compact_mislabeled_public_query":
                    return {"_native_tool_call": {"tool_calls": [{"name": "__shinku_tool_dispatch__", "arguments": {
                        "tool_name": "memory_search", "arguments": json.dumps({"query": "诗句", "mode": "recent"}),
                        "task_link": {"mode": "none", "track": "public_query"}}}]}}
                return {"_native_tool_call": {"tool_calls": [{"name": "memory_search", "arguments": {"query": "诗句", "mode": "recent"}}]}}
            return {"kind": "final", "text": "那句是夜色落在窗前。"}

    runtime = Runtime()
    store = ConversationRuntime()
    coordinator = AgentTaskCoordinator()
    try:
        bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", sender=None, memory_service=memory,
            tool_host=ToolHost(ToolRegistry({"memory_search": tool})), allowed_tool_names={"memory_search"},
            command_state=QQCommandState(permission_mode="full"), master_user_ids=("u-19",), owner_gate_required=True,
            task_contract_enabled=True, async_agent_enabled=True, task_coordinator=coordinator, conversation_runtime=store)
        request = "还记得前几天100001群里有人写的诗吗" if lookup_status == "scope_limited" else "核对以前那句诗的原话"
        result = bridge.handle_turn(_turn(request))
        assert result.agent_status == expected and result.reason == "lightweight_perception_fast_track"
        assert len(calls) == 1 and not coordinator.list()
        assert not bridge.task_store.candidates(requester="u-19", session="private:u-19")
        row = store.snapshot(ConversationScope("private", "u-19", "u-19"))["recent_verifications"][0]
        assert row["route"] == "history" and row["query_count"] == 1
        assert row["validated"] is (lookup_status != "candidates") and row["verified"] is False
        assert row["disposition"] == ("unverified" if lookup_status == "candidates" else lookup_status)
        assert row["delivery_status"] == "dry_run" and not row["sent"]
        assert all("元数据准备" not in "\n".join(call.get("system_extra_blocks", [])) for call in runtime.calls)
    finally:
        coordinator.shutdown(wait=True)


def test_ingress_invalidates_proactive_before_batched_handler_starts():
    from test_qq_chat_agent_split import _Runtime
    store = ConversationRuntime(quiet_seconds=0)
    bridge = QQAgentBridge(runtime=_Runtime({"speech": "我在。"}), system_prompt="fixture", conversation_runtime=store)
    turn = _turn()
    scope = bridge._conversation_scope(turn)
    permit = store.proactive_permit(scope.channel)
    assert bridge.observe_message(turn.messages[0]) is False  # No memory attached.
    assert not store.validate_proactive(permit)
    assert store.snapshot(scope)["recent_turns"] == []


def test_async_worker_completion_does_not_clear_more_recent_running_turn():
    entered, release = Event(), Event()
    coordinator = AgentTaskCoordinator(max_workers=1)
    helper = async_helpers.QQAsyncProcessorTests()
    bridge = helper.make_bridge(coordinator, async_helpers.BlockingInspector(entered, release))
    scope = ConversationScope("private", "owner", "owner")
    try:
        queued = bridge.handle_turn(async_helpers.turn("inspect"))
        assert queued.reason == "agent_task_queued" and entered.wait(2)
        row = bridge.conversation_runtime.snapshot(scope)["recent_turns"][0]
        assert row["phase"] != "finished"
        bridge.conversation_runtime.begin(scope, "new-running")
        release.set()
        async_helpers.wait_for(coordinator, queued.background_task_id, "completed")
        assert bridge.conversation_runtime.proactive_permit(scope.channel).reason == "conversation_active"
        rows = bridge.conversation_runtime.snapshot(scope)["recent_turns"]
        assert next(row for row in rows if row["turn_id"] != "new-running")["phase"] == "finished"
        assert next(row for row in rows if row["turn_id"] == "new-running")["phase"] != "finished"
    finally:
        release.set()
        coordinator.shutdown(wait=True)


def test_canceled_queued_worker_finishes_its_operational_record():
    entered, release = Event(), Event()
    coordinator = AgentTaskCoordinator(max_workers=1)
    helper = async_helpers.QQAsyncProcessorTests()
    bridge = helper.make_bridge(coordinator, async_helpers.BlockingInspector(entered, release))
    try:
        first = bridge.handle_turn(async_helpers.turn("inspect", message_id="first"))
        assert entered.wait(2)
        second = bridge.handle_turn(async_helpers.turn("inspect", message_id="second"))
        canceled = coordinator.cancel(second.background_task_id, profile_user_id="owner", session_id="private:owner")
        assert canceled["ok"]
        rows = bridge.conversation_runtime.snapshot(ConversationScope("private", "owner", "owner"))["recent_turns"]
        assert any(row["status"] == "canceled" and row["phase"] == "finished" for row in rows)
        assert bridge.conversation_runtime.proactive_permit(ConversationScope("private", "owner", "owner").channel).reason == "conversation_active"
        release.set()
        async_helpers.wait_for(coordinator, first.background_task_id, "completed")
    finally:
        release.set()
        coordinator.shutdown(wait=True)


def test_read_first_mutation_goal_cannot_enter_no_contract_history_route(tmp_path):
    memory = MemoryService(store=MemoryStore(tmp_path / "memory.db"), router=MemoryRouter())
    bridge = QQAgentBridge(runtime=async_helpers.Runtime(), system_prompt="fixture", tool_host=ToolHost(ToolRegistry({"memory_search": MemorySearchTool(memory)})),
        command_state=QQCommandState(permission_mode="full"), task_contract_enabled=True, async_agent_enabled=False)
    decision = AgentDecision.tool(ToolInvocation("memory_search", {"query": "地址"}))
    assert not bridge._is_lightweight_perception(decision, context={"qq_turn": _turn("查出以前的地址，然后把消息发送给他")})


@pytest.mark.parametrize("route", ["history", "unrelated", "invalid", "unknown_tool", "exception"])
def test_direct_reply_rechecks_unresolved_history_without_task_or_permission_expansion(tmp_path, route):
    memory = MemoryService(store=MemoryStore(tmp_path / "memory.db"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("u-19",)))
    tool = MemorySearchTool(memory)
    actual_execute, calls = tool.execute, []

    def execute(**kwargs):
        calls.append(kwargs)
        return actual_execute(**kwargs)

    tool.execute = execute

    class Runtime:
        def __init__(self):
            self.calls = []

        def chat_supports_native_tools(self):
            return True

        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            review = dialogue_review_result(kwargs)
            if review is not None:
                return review
            if "未完成核对的回复出口审查" in "\n".join(kwargs.get("system_extra_blocks", [])):
                if route == "exception":
                    raise RuntimeError("fixture_review_unavailable")
                if route == "unrelated":
                    return {"route": "chat", "relation": "unrelated"}
                if route == "invalid":
                    return {"route": "chat", "relation": "historical_fact"}
                return {"route": "history", "tool_name": "unknown_write" if route == "unknown_tool" else "memory_search",
                    "arguments": {"query": "原因", "mode": "retrieve"}}
            return {"kind": "final", "text": "还想陪你待一会儿。"}

    runtime = Runtime()
    store = ConversationRuntime()
    scope = ConversationScope("private", "u-19", "u-19")
    store.begin(scope, "old-unconfirmed")
    store.step("old-unconfirmed", route="history", evidence=AnswerEvidence("history", "unverified"))
    store.finish("old-unconfirmed", status="failed", reason="fixture_read_failed")
    coordinator = AgentTaskCoordinator()
    try:
        bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", sender=None, memory_service=memory,
            tool_host=ToolHost(ToolRegistry({"memory_search": tool})), allowed_tool_names={"memory_search"},
            command_state=QQCommandState(permission_mode="full"), master_user_ids=("u-19",), owner_gate_required=True,
            task_contract_enabled=True, async_agent_enabled=True, task_coordinator=coordinator, conversation_runtime=store)
        result = bridge.handle_turn(_turn("换个话题，聊聊今天的天气" if route == "unrelated" else "再核对一下那件事的原因"))
        assert not coordinator.list()
        assert calls == [] and len(runtime.calls) == 1
        assert result.final_text == "还想陪你待一会儿。" and result.agent_status == "completed"
        assert not hasattr(bridge, "_review_direct_reply_route")
        latest = store.snapshot(scope)["recent_turns"][0]
        assert not latest["verified"]
        assert len(store.snapshot(scope)["recent_verifications"]) >= 1
    finally:
        coordinator.shutdown(wait=True)
