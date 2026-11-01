import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_history_answer import receipt
from test_qq_chat_agent_split import _Runtime, _turn

from shinku.agent.history_answer import HistoryAnswerLedger
from shinku.agent.loop import AgentDecision
from shinku.conversation import (
    AnswerEvidence,
    ConversationRuntime,
    ConversationScope,
)
from shinku.conversation.models import text_digest
from shinku.conversation.routing import ObservationRoutePolicy
from shinku.conversation.runtime import ConversationStoreError
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.assembly import load_qq_agent_config
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


SCOPE = ConversationScope("private", "u-19", "u-19")


def test_separate_process_instances_share_state_and_exact_scope(tmp_path):
    clock = Clock()
    path = tmp_path / "state.db"
    foreground = ConversationRuntime(path, clock=clock)
    other = ConversationRuntime(path, clock=clock)
    foreground.begin(SCOPE, "read")
    foreground.step("read", route="history", evidence=AnswerEvidence("history", "unverified"), candidate_count=6)
    foreground.finish("read", status="failed", reason="task_contract_error:contract_repair_failed")
    value = other.snapshot(SCOPE)["recent_verifications"][0]
    assert value["query_count"] == 0 and value["candidate_count"] == 6
    assert not value["verified"] and value["status"] == "failed"
    assert not other.snapshot(ConversationScope("private", "u-19", "other"))["recent_turns"]
    assert not other.snapshot(ConversationScope("group", "u-19", "u-19"))["recent_turns"]
    with sqlite3.connect(path) as conn:
        stored = conn.execute("SELECT payload FROM conversation_turns").fetchone()[0]
    assert set(json.loads(stored)) == {
        "turn_id", "scope", "started_at", "updated_at", "lease_until", "status", "phase", "route",
        "origin", "evidence", "candidate_count", "reason", "delivery_status", "sent", "task_id",
        "task_attempt", "task_version", "message_ids",
    }


def test_completed_chat_does_not_erase_failed_query_or_old_worker_clear_new_turn():
    clock = Clock()
    runtime = ConversationRuntime(clock=clock)
    runtime.begin(SCOPE, "old")
    runtime.step("old", evidence=AnswerEvidence("history", "unverified"))
    runtime.begin(SCOPE, "new")
    runtime.finish("old", status="failed")
    assert runtime.proactive_permit(SCOPE.channel).reason == "conversation_active"
    runtime.finish("new", status="chat", delivery_status="dry_run")
    assert runtime.snapshot(SCOPE)["recent_verifications"][0]["status"] == "failed"
    clock.now += 100
    assert runtime.proactive_permit(SCOPE.channel).reason == "conversation_verification_unresolved"
    clock.now += 600
    assert runtime.proactive_permit(SCOPE.channel).allowed


def test_delivery_digest_and_terminal_ownership_are_independent_from_transport():
    runtime = ConversationRuntime()
    answer = "真实原话。"
    evidence = AnswerEvidence("history", "found", True, 1, 1, text_digest(answer))
    runtime.begin(SCOPE, "turn")
    runtime.step("turn", evidence=evidence)
    runtime.finish("turn", status="completed", delivery_status="partial", sent=True, answer="被改写的回答")
    assert not runtime.finish("turn", status="queued")
    row = runtime.snapshot(SCOPE)["recent_verifications"][0]
    assert row["status"] == "completed" and row["delivery_status"] == "partial"
    assert not row["sent"] and not row["verified"]


@pytest.mark.parametrize("status", ["not_found", "scope_limited", "read_failed"])
def test_actual_negative_receipts_remain_distinct(status):
    ledger = HistoryAnswerLedger(enabled=True)
    ledger.observe(receipt(status, records=[]), SimpleNamespace(agent_history_answer_source=True))
    ledger.apply_review("诚实说明限制", {"disposition": status, "claims": [], "limitation_fragment": "诚实说明限制"}, {})
    evidence = AnswerEvidence.from_history(ledger)
    assert evidence.validated and evidence.disposition == status and not evidence.verified
    assert evidence.resolved is (status != "read_failed")
    assert not evidence.bound_to("另一个答案").validated


def test_lease_expiry_is_interruption_and_retention_is_bounded():
    clock = Clock()
    runtime = ConversationRuntime(clock=clock, lease_seconds=30, retention_seconds=600, max_turns=16)
    runtime.begin(SCOPE, "abandoned")
    clock.now += 31
    assert runtime.snapshot(SCOPE)["recent_turns"][0]["status"] == "interrupted"
    assert not runtime.proactive_permit(SCOPE.channel).allowed
    for index in range(24):
        clock.now += 1
        runtime.begin(SCOPE, f"chat-{index}")
        runtime.finish(f"chat-{index}", status="chat")
    with runtime._database() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] <= 16
    clock.now += 601
    assert not runtime.snapshot(SCOPE)["recent_turns"]


def test_active_older_turn_is_not_hidden_by_row_limit():
    clock = Clock()
    runtime = ConversationRuntime(clock=clock, max_turns=16, quiet_seconds=0)
    runtime.begin(SCOPE, "active")
    for index in range(20):
        clock.now += 1
        runtime.begin(SCOPE, f"terminal-{index}")
        runtime.finish(f"terminal-{index}", status="chat")
    assert runtime.proactive_permit(SCOPE.channel).reason == "conversation_active"


def test_revision_invalidates_generation_even_when_new_turn_already_completed():
    runtime = ConversationRuntime(quiet_seconds=0)
    permit = runtime.proactive_permit(SCOPE.channel)
    runtime.begin(SCOPE, "new")
    runtime.finish("new", status="chat")
    assert not runtime.validate_proactive(permit)
    fresh = runtime.proactive_permit(SCOPE.channel)
    reserved = runtime.reserve_proactive(fresh, SCOPE, "scheduled")
    assert reserved and runtime.validate_proactive(reserved, exclude_turn="scheduled")
    assert not runtime.proactive_permit(SCOPE.channel).allowed
    runtime.notice_inbound(SCOPE.channel)
    assert not runtime.validate_proactive(reserved, exclude_turn="scheduled")


def test_unavailable_store_fails_closed_without_false_verification():
    runtime = ConversationRuntime()
    runtime.close()
    assert not runtime.prompt_context(SCOPE)["available"]
    permit = runtime.proactive_permit(SCOPE.channel)
    assert not permit.allowed and permit.reason == "conversation_state_unavailable"
    with pytest.raises(ConversationStoreError):
        runtime.begin(SCOPE, "closed")


def test_only_same_host_task_continuation_resolves_waiting_user():
    runtime = ConversationRuntime(quiet_seconds=0)
    runtime.begin(SCOPE, "pending")
    runtime.step("pending", task_id="task-original")
    runtime.finish("pending", status="waiting_user")
    runtime.begin(SCOPE, "chat")
    runtime.finish("chat", status="chat")
    assert not runtime.proactive_permit(SCOPE.channel).allowed
    runtime.begin(SCOPE, "continue")
    runtime.step("continue", task_id="task-original")
    runtime.finish("continue", status="completed")
    assert runtime.proactive_permit(SCOPE.channel).allowed
    assert next(row for row in runtime.snapshot(SCOPE)["recent_turns"] if row["turn_id"] == "pending")["status"] == "continued"


def test_bridge_failure_without_memory_persists_and_next_prompt_receives_it():
    clock = Clock()
    store = ConversationRuntime(clock=clock)
    runtime = _Runtime({"speech": "还在。"})
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", conversation_runtime=store)
    store.begin(SCOPE, "failed-history")
    store.step("failed-history", evidence=AnswerEvidence("history", "unverified"), candidate_count=4)
    reply = bridge._deliver_text(_turn(), "还没核对成功。", agent_status="failed", record_memory=False,
        track_chat_flow=False, reason="task_contract_error:contract_repair_failed", trace_id="failed-history")
    assert reply.delivery_status == "dry_run"
    clock.now += 1
    bridge.handle_turn(_turn("得确定一下，不然你记忆可能紊乱"))
    shared = "\n".join(call["user_prompt"] for call in runtime.calls)
    assert '"status": "failed"' in shared and '"query_count": 0' in shared
    assert "普通聊天回复完成不会消除旧的查询失败" in "\n".join(runtime.calls[-1]["system_extra_blocks"])
    verifications = store.snapshot(SCOPE)["recent_verifications"]
    assert any(row["turn_id"] == "failed-history" and row["status"] == "failed" for row in verifications)
    assert verifications[0]["turn_id"] == "failed-history" and not verifications[0]["verified"]


def test_config_same_path_for_backend_and_scheduled_host(tmp_path):
    config = load_qq_agent_config({"SHINKU_DATA_ROOT": str(tmp_path)})
    assert config.conversation_runtime_path == tmp_path / "runtime" / "conversation_state.sqlite3"
    config = load_qq_agent_config({"SHINKU_QQ_AGENT_STATE_DB_PATH": str(tmp_path / "tasks.db")})
    assert config.conversation_runtime_path == tmp_path / "conversation_state.sqlite3"
    chosen = tmp_path / "chosen.db"
    assert replace(config, conversation_state_store_path=str(chosen)).conversation_runtime_path == chosen


@pytest.mark.parametrize("action,link,history,expected", [
    (False, None, True, True), (False, None, False, False),
    (True, None, True, False), (False, {"mode": "continue"}, True, False),
    (False, {"mode": "new"}, True, False), (False, {"mode": "none", "track": "agent"}, True, True),
    (False, {"mode": "none", "track": "public_query"}, True, True),
    (False, {"mode": "none", "track": "public_query"}, False, False),
    (False, {"mode": "none"}, True, True),
])
def test_native_missing_metadata_admission_is_registered_bounded_history_only(action, link, history, expected):
    handler = SimpleNamespace(agent_history_answer_source=history, agent_fast_perception=True,
        agent_permission_class="read", agent_progress_class="observe", agent_completion_effects=())
    decision = AgentDecision.tool(ToolInvocation("history", {}))
    decision = replace(decision, task_link=link)
    assert ObservationRoutePolicy.admits(decision, handlers={"history": handler},
        policy=ExecutionPolicy(allowed=frozenset({"history"}), default_risk="low"),
        strict=True, asynchronous=True, action_goal=action) is expected


@pytest.mark.parametrize("limits", [{"lease_seconds": float("nan")}, {"max_turns": float("inf")}])
def test_nonfinite_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        ConversationRuntime(**limits)


def test_history_sources_preserve_role_and_scheduled_provenance():
    ledger = HistoryAnswerLedger(enabled=True)
    envelope = receipt(records=[{"source_id": "scheduled", "scope": "group:100001", "text": "主动问候",
        "created_at": 100, "metadata": {"role": "assistant", "origin": "scheduled_greeting", "greeting_routine": "recap"}}])
    ledger.observe(envelope, SimpleNamespace(agent_history_answer_source=True))
    source = next(iter(ledger.sources.values()))
    assert source["role"] == "assistant" and source["origin"] == "scheduled_greeting"


def test_history_found_requires_original_question_coverage_not_merely_one_bound_quote():
    value = HistoryAnswerLedger(enabled=True)
    value.observe(receipt(), SimpleNamespace(agent_history_answer_source=True))
    source = next(iter(value.sources))
    context = {"agent_completion_contract": {"requirements": [{"effect": "history_answer",
        "checks": {"required_fields": ["原问题"]}}]}}
    answer = "夜色落在窗前。"
    review = {"disposition": "found", "claims": [{"source_id": source, "quote": answer, "answer_fragment": answer}]}
    value.apply_review(answer, review, context)
    assert not value.answer_check["valid"] and not AnswerEvidence.from_history(value).verified
    value.apply_review(answer, {**review, "coverage": [{"field": "原问题", "supported": True, "claim_indices": [0]}]}, context)
    assert AnswerEvidence.from_history(value).verified


def test_queried_but_unconfirmed_is_not_found_or_absence():
    value = HistoryAnswerLedger(enabled=True)
    value.observe(receipt(), SimpleNamespace(agent_history_answer_source=True))
    answer = "我还没有确认这次的原因，不能只凭之前说过的话判断。"
    value.apply_review(answer, {"disposition": "unconfirmed", "claims": [], "limitation_fragment": answer}, {})
    evidence = AnswerEvidence.from_history(value)
    assert evidence.validated and evidence.disposition == "unconfirmed" and not evidence.resolved
    assert value.evaluate_answer(answer, "partial").accepted and not value.evaluate_answer(answer, "completed").accepted
