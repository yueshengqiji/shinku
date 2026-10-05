"""Private/group parity, real runtime parsing, and fail-closed task budgets.

All senders and SDK responses are isolated fixtures; no production QQ sends.
"""
import copy
import json

import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent.task_state import AgentTaskStore, new_task, redact
from shinku.hosts.tool_host import ToolHost
from shinku.llm.runtime_core import ProviderToolProfile
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.reply_host import ReplyHost
from shinku.qq.turns import QQTurn
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.registry import ToolRegistry
from test_contract_c2_5 import _make_runtime, FakeResponse
from test_agent_async_processor import wait_for


class Inspector:
    tool_type = "inspect_fixture"
    agent_permission_class = "read"
    input_schema = {"type": "object", "properties": {}}
    description = "A fixture, never called for ordinary chat."

    def normalize_call(self, call):
        return dict(call)

    def execute(self, **kwargs):
        raise AssertionError("ordinary chat must not execute tools")


def turn(kind="group", text="今天怎么样", sender="owner"):
    parts = [{"type": "text", "data": {"text": text}}]
    if kind == "group":
        parts.insert(0, {"type": "at", "data": {"qq": "10001"}})
    message = IncomingMessage.from_event({"post_type": "message", "message_type": kind,
        "message_id": "fixture", "user_id": sender, "group_id": "42", "self_id": "10001",
        "message": parts})
    return QQTurn(conversation_id="42" if kind == "group" else sender, conversation_type=kind,
        primary_message_id=message.message_id, message_ids=(message.message_id,),
        combined_text=message.text, messages=(message,))


def bridge(runtime, coordinator, sent):
    return QQAgentBridge(runtime=runtime, system_prompt="fixture", bot_ids=("10001",),
        master_user_ids=("owner",), owner_gate_required=True,
        tool_host=ToolHost(ToolRegistry({"inspect_fixture": Inspector()})),
        command_state=QQCommandState(permission_mode="ultra"), task_contract_enabled=True,
        async_agent_enabled=True, task_coordinator=coordinator,
        reply_host=ReplyHost(strip_final_periods=True),
        sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("receipt",)))


@pytest.mark.parametrize("kind,sender", [("private", "owner"), ("group", "owner"), ("group", "guest")])
@pytest.mark.parametrize("schema_retry", [False, True])
def test_plain_model_reply_is_not_a_synthetic_failure(kind, sender, schema_retry):
    scripts = [("response", FakeResponse("今天还不错。你呢？"))]
    # Non-owner has no tools, hence no native-schema retry.
    if schema_retry and sender == "owner":
        scripts.insert(0, ("response", FakeResponse("")))
    runtime = _make_runtime(scripts=scripts)
    runtime._tool_profile = lambda bundle: ProviderToolProfile(supports_native_tools=True)
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(runtime, coordinator, sent).handle_turn(turn(kind, sender=sender))
        assert result.agent_status == "completed", result.reason
        assert [m.segments[0].text for m in sent] == ["今天还不错", "你呢？"]
        assert not coordinator.list()
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_explicit_failure_status_remains_honest_but_text_is_still_split(status):
    runtime = _make_runtime(scripts=[("response", FakeResponse(json.dumps({
        "kind": "final", "text": "这次没有完成。可以换个方式。", "task_outcome": status}, ensure_ascii=False)))])
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(runtime, coordinator, sent).handle_turn(turn())
        assert result.agent_status == status
        assert [m.segments[0].text for m in sent] == ["这次没有完成", "可以换个方式"]
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()


@pytest.mark.parametrize("status,count", [("partial", 2), ("waiting_user", 1), ("command", 1)])
def test_report_shaping_does_not_split_interactive_controls(status, count):
    runtime = _make_runtime()
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(runtime, coordinator, sent)._deliver_text(turn(), "第一句。第二句。",
            agent_status=status, record_memory=False, track_chat_flow=False)
        assert result.agent_status == status
        assert len(sent) == count
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()


def test_recovered_text_still_cannot_claim_unexecuted_action():
    runtime = _make_runtime(scripts=[("response", FakeResponse("已删除 C:/fixture/target.zip。"))])
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(runtime, coordinator, sent).handle_turn(turn(text="删除 C:/fixture/target.zip"))
        assert result.agent_status == "failed"
        assert result.reason == "direct_reply_unverified"
        assert "已删除" not in result.final_text
        assert not coordinator.list()
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()


def test_transport_fallback_is_not_promoted_to_success():
    class FailedRuntime:
        def chat_supports_native_tools(self):
            return True

        def call_chat_json(self, **kwargs):
            return copy.deepcopy(kwargs["fallback"])
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        result = bridge(FailedRuntime(), coordinator, sent).handle_turn(turn())
        assert result.agent_status == "failed"
        assert result.reason == "chat_generation_failed"
        assert not coordinator.list()
        assert "_agent_plain_text_verify" not in str(sent)
    finally:
        coordinator.shutdown(wait=True)


def test_saved_budget_metrics_remain_numeric_and_continue_without_reset():
    store = AgentTaskStore()
    try:
        task = new_task(task_id="old", requester="owner", session="private:owner",
            request="检查", message_id="old", contract=None)
        task["budget"] = TaskBudget(requests=4, tokens=100000, reported_tokens=90000).snapshot()
        saved = store.save(task)
        budget = TaskBudget(requests=1, tokens=5000)
        budget.absorb_prior(saved["budget"])
        assert (budget.requests, budget.tokens, budget.reported_tokens) == (5, 105000, 90000)
        assert not budget.stop_reason()
        assert redact({"tokens": "private-secret-value"})["tokens"] == "[redacted]"
    finally:
        store.close()


@pytest.mark.parametrize("key,value", [("tokens", "[redacted]"), ("max_tokens", "[redacted]"),
    ("requests", float("nan")), ("active_seconds", float("inf")), ("tools", -1), ("requests", True)])
def test_corrupted_legacy_budget_stops_without_typeerror_or_fresh_budget(key, value):
    budget = TaskBudget(requests=1, tokens=100)
    snapshot = TaskBudget(requests=4, tokens=90000).snapshot()
    snapshot[key] = value
    budget.absorb_prior(snapshot)
    assert budget.stop_reason() == "task_budget_snapshot_invalid"
    with bind_task_budget(budget), pytest.raises(TaskBudgetExceeded, match="task_budget_snapshot_invalid"):
        budget.use_tool()
    assert budget.tools == 0


@pytest.mark.parametrize("kind", ["private", "group"])
@pytest.mark.parametrize("corrupted", [False, True])
def test_async_browser_continuation_preserves_budget_and_blocks_invalid_records(kind, corrupted):
    class Browser:
        tool_type = "private_browser_open"
        agent_permission_class = "read"
        agent_progress_class = "observe"
        agent_completion_effects = ("browser_access",)
        agent_completion_target_arguments = ("url",)
        description = "Isolated browser receipt, no network or actual browser."
        input_schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        calls = 0

        def normalize_call(self, call):
            return dict(call)

        def execute(self, **kwargs):
            self.calls += 1
            return {"followup_context": "A fixture page opened.", "state_updates": {
                "verified_effects": ["browser_access"], "completion_targets": ["https://www.bilibili.com"],
                "site_access_ok": True, "http_status": 200}}

    old_contract = {"goal": "查看页面", "deliverables": ["页面访问结果"], "scope": ["隔离浏览回执"],
        "requirements": [{"id": "page", "description": "确认页面可访问", "effect": "browser_access",
                          "targets": ["https://www.bilibili.com"], "checks": {}}]}
    class Runtime:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"kind": "tool", "tool_call": {"name": "__shinku_tool_dispatch__",
                    "arguments": {"tool_name": "private_browser_open",
                        "arguments": json.dumps({"url": "https://www.bilibili.com"}),
                        "task_link": {"mode": "continue", "task_id": "old"}}}}
            return {"kind": "final", "text": "页面已打开。", "task_outcome": "completed"}
    coordinator, runtime, sent = AgentTaskCoordinator(), Runtime(), []
    bot = bridge(runtime, coordinator, sent)
    handler = Browser()
    bot.tool_host = ToolHost(ToolRegistry({handler.tool_type: handler}))
    item = turn(kind, text="继续确认这个页面")
    requester, session = bot._task_scope(item)
    old = new_task(task_id="old", requester=requester, session=session, request="查看页面",
        message_id="old", contract=old_contract)
    old["budget"] = TaskBudget(requests=4, tokens=100000).snapshot()
    if corrupted:
        old["budget"]["tokens"] = "[redacted]"
    bot.task_store.save(old)
    try:
        queued = bot.handle_turn(item)
        assert queued.agent_status == "waiting_background", queued.reason
        record = wait_for(coordinator, queued.background_task_id, "blocked" if corrupted else "completed")
        assert record["metadata"]["budget"]["requests"] == 4
        assert handler.calls == (0 if corrupted else 1)
        assert record["error"] == ("task_budget_snapshot_invalid" if corrupted else "")
        assert "后台任务发生内部错误" not in record["result_summary"]
    finally:
        coordinator.shutdown(wait=True)
        bot.task_store.close()
