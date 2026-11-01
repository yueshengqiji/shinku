from types import SimpleNamespace
from dataclasses import replace
import pytest

from shinku.agent.history_answer import HISTORY_VERIFIER_SYSTEM, HistoryAnswerLedger, review_history_answer
from shinku.memory.tools import MemorySearchTool
from shinku.memory import MemoryService, MemoryStore, MemoryRouter, MemoryPolicy
from shinku.tools.invocation import ToolResultEnvelope
from test_qq_outgoing_mentions import _private_turn
import json
from shinku.agent.loop import AgentLoop, AgentDecision
from shinku.agent.planner import LLMPlanner
from shinku.agent.verification import SmartCompletionVerifier
from shinku.tools.invocation import ToolInvocation


def receipt(status="candidates", *, records=None, requested="group:100001"):
    records = records if records is not None else [{"source_id": "turn:100001:original-poem", "scope": "group:100001",
        "text": "夜色落在窗前。", "created_at": 100}]
    return ToolResultEnvelope("read-1", "error" if status == "read_failed" else "ok", "fixture", data={"state_updates": {
        "memory_lookup": {"status": status, "query_completed": status != "read_failed", "requested_scope_supported": status != "scope_limited",
                          "retrieval_scopes": ["group:100001"], "requested_scopes": [requested] if requested else []},
        "memory_inspection": {"records": records}}})


def ledger(envelope=None):
    # Ledger-level fixtures model a turn that *is* a verification request, so
    # both flags are on.  test_runtime_history_requirement.py covers the cases
    # where the receipt gate applies without one.
    value = HistoryAnswerLedger(enabled=True, verification_requested=True)
    value.observe(envelope or receipt(), SimpleNamespace(agent_history_answer_source=True))
    return value


def test_actual_original_message_binding_can_complete_but_candidates_alone_cannot():
    value = ledger()
    answer = "那句是夜色落在窗前。"
    assert not value.evaluate_answer(answer, "completed").accepted
    source = next(iter(value.sources))
    value.apply_review(answer, {"disposition": "found", "claims": [{"source_id": source,
        "answer_fragment": answer, "quote": "夜色落在窗前。"}],
        "coverage": [{"field": "诗句", "supported": True, "claim_indices": [0]}]}, {})
    assert value.evaluate_answer(answer, "completed").accepted
    assert value.satisfies({"required_fields": ["诗句"], "min_sources": 1})
    assert not value.satisfies({"required_fields": ["作者"]})


@pytest.mark.parametrize("review_disposition", ["unconfirmed", "unverified"])
def test_unconfirmed_reply_renders_only_receipt_limitation_not_extra_invented_facts(review_disposition):
    value = ledger(receipt(requested=""))
    invented = "我在另一个群听见你说了什么，所以把你拉回来；具体原因记不清。"
    value.apply_review(invented, {"disposition": review_disposition, "claims": [],
        "limitation_fragment": "具体原因记不清。"}, {})
    rendered = value.unconfirmed_feedback()
    assert rendered is not None
    answer, outcome, disposition = rendered
    assert "另一个群" not in answer and "所以把你" not in answer
    assert outcome == "partial" and disposition == "unconfirmed"
    value.apply_review(answer, {"disposition": disposition, "claims": [], "limitation_fragment": answer}, {})
    assert value.evaluate_answer(answer, outcome).accepted
    assert not value.evaluate_answer(invented, outcome).accepted
    assert not value.evaluate_answer(answer, "completed").accepted


@pytest.mark.parametrize("origin", ["", "scheduled_greeting"])
def test_assistant_self_statement_cannot_verify_event_cause_even_if_reviewer_says_found(origin):
    records = [{"source_id": "assistant-motive", "scope": "group:100001",
        "text": "只是想让你陪我待在这边。", "created_at": 100,
        "metadata": {"role": "assistant", "origin": origin}}]
    value = ledger(receipt(records=records))
    source = next(iter(value.sources))
    answer = "因为只是想让你陪我待在这边。"
    value.apply_review(answer, {"disposition": "found", "claims": [{"source_id": source,
        "quote": "只是想让你陪我待在这边。", "answer_fragment": "只是想让你陪我待在这边。"}],
        "coverage": [{"field": "原问题", "supported": True, "claim_indices": [0]}]}, {})
    assert not value.evaluate_answer(answer, "completed").accepted
    assert value.answer_check["reason"] == "utterance_only_not_event_proof"
    assert value.unconfirmed_feedback() is not None


def test_history_critic_is_independent_of_character_generation_persona():
    calls = []
    runtime = SimpleNamespace(call_chat_json=lambda **kwargs: calls.append(kwargs) or {})
    review_history_answer(ledger(), runtime=runtime, system_prompt="角色人设不要承认错误", answer="不能确认。", goal="原因", context={})
    assert calls[0]["system_prompt"] == HISTORY_VERIFIER_SYSTEM
    assert "角色人设不要承认错误" not in json.dumps(calls, ensure_ascii=False)


@pytest.mark.parametrize("status", ["not_found", "scope_limited", "read_failed"])
def test_query_absence_limit_and_failure_remain_distinct_not_success(status):
    value = ledger(receipt(status, records=[]))
    answer = "这次还没核对到目标。"
    value.apply_review(answer, {"disposition": status, "claims": [], "limitation_fragment": answer}, {})
    assert value.answer_check["disposition"] == status
    assert value.evaluate_answer(answer, "failed" if status == "read_failed" else "partial").accepted
    assert not value.evaluate_answer(answer, "completed").accepted
    assert not value.satisfies({})


def test_multiple_actual_candidates_require_ambiguity_report_not_unique_claim():
    records = [{"source_id": "one", "scope": "group:100001", "text": "第一首诗", "created_at": 100},
               {"source_id": "two", "scope": "group:100001", "text": "第二首诗", "created_at": 101}]
    value = ledger(receipt(records=records))
    answer = "有第一首诗和第二首诗两条，你指哪一条？"
    ids = list(value.sources)
    value.apply_review(answer, {"disposition": "ambiguous", "limitation_fragment": "你指哪一条？",
        "claims": [{"source_id": ids[0], "quote": "第一首诗", "answer_fragment": "第一首诗"},
                   {"source_id": ids[1], "quote": "第二首诗", "answer_fragment": "第二首诗"}],
        "answer_parts": [{"text":"有","kind":"non_factual"},
            {"text":"第一首诗","kind":"supported","claim_indices":[0]},
            {"text":"和","kind":"non_factual"},
            {"text":"第二首诗","kind":"supported","claim_indices":[1]},
            {"text":"两条，你指哪一条？","kind":"non_factual"}]}, {})
    assert value.evaluate_answer(answer, "partial").accepted
    assert not value.evaluate_answer(answer, "completed").accepted


def test_wrong_group_sources_and_fabricated_quote_cannot_satisfy_history():
    value = ledger(receipt(requested="group:elsewhere"))
    assert not value.sources
    value = ledger()
    source = next(iter(value.sources))
    value.apply_review("不存在的诗句", {"disposition": "found", "claims": [{"source_id": source,
        "answer_fragment": "不存在的诗句", "quote": "不存在的诗句"}]}, {})
    assert not value.evaluate_answer("不存在的诗句", "completed").accepted


def test_private_query_cannot_open_arbitrary_group_and_reports_scope_limit(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.db"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("10001",)), clock=lambda: 200)
    service.store.record_event(conversation_id="100001", scope="group:100001", source_id="group-poem",
        text="夜色落在窗前。", layer="episodic", created_at=100)
    tool = MemorySearchTool(service)
    assert tool.normalize_call({"query": "诗", "group_id": "100001"}) is None
    result = tool.execute(call={"query": "前几天100001群里有人写的诗", "mode": "retrieve"}, context={"qq_turn": _private_turn()})
    assert result["state_updates"]["memory_lookup"]["status"] == "scope_limited"
    assert "夜色落在窗前" not in result["followup_context"]


def test_requester_comes_from_primary_message_not_first_group_bystander(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.db"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("10001",)), clock=lambda: 200)
    tool = MemorySearchTool(service)
    seen = []
    service.inspect = lambda **kw: seen.append(kw) or {"records": [], "scope": "group:g", "followup_context": "empty"}
    turn = SimpleNamespace(conversation_id="g", conversation_type="group", primary_message_id="primary", combined_text="当前问题",
        messages=(SimpleNamespace(sender_id="10001", message_id="bystander", raw_event={}),
                  SimpleNamespace(sender_id="20002", message_id="primary", raw_event={})))
    tool.execute(call={"query": "诗"}, context={"qq_turn": turn})
    assert seen[0]["requester_id"] == "20002"


@pytest.mark.parametrize("value", [None, [], {"disposition": []},
    {"disposition": "found", "claims": [{"source_id": [], "quote": "诗", "answer_fragment": "诗"}]},
    {"disposition": "found", "claims": []}, {"disposition": "not_found", "limitation_fragment": []}])
def test_malformed_review_does_not_crash_or_manufacture_history(value):
    subject = ledger()
    subject.apply_review("诗", value, {})
    assert not subject.evaluate_answer("诗", "completed").accepted


@pytest.mark.parametrize("checks", [{"required_fields": [{}]}, {"required_fields": "诗"},
    {"min_sources": True}, {"min_sources": "1"}, {"min_sources": 0}, {"min_sources": []}])
def test_malformed_requirements_fail_closed(checks):
    assert not ledger().satisfies(checks)


@pytest.mark.parametrize("lookup_status,expected", [("scope_limited", "blocked"),
    ("not_found", "partial"), ("read_failed", "failed")])
def test_host_history_intent_cannot_be_bypassed_by_generic_observe_contract(lookup_status, expected):
    from shinku.agent.loop import AgentState
    state = AgentState(context={"agent_history_query_request": True,
        "agent_completion_contract": {"requirements": [{"id": "r1", "effect": "observe", "targets": []}]}})
    state.history_answer.configure(state.context)
    state.history_answer.observe(receipt(lookup_status, records=[]), SimpleNamespace(agent_history_answer_source=True))
    state.transcript = [{"role": "user", "content": "还记得前几天群里有人写的诗吗"}]
    class Runtime:
        def call_chat_json(self, **kwargs):
            if "历史答案复核" in "\n".join(kwargs.get("system_extra_blocks", [])):
                return {"disposition": "unverified"}
            return {"kind": "final", "text": "记不清了，提示一下？", "task_outcome": "failed"}
    decision = LLMPlanner(runtime=Runtime(), system_prompt="fixture")(state=state)
    assert decision.kind == "final" and decision.outcome == expected
    assert decision.text != "记不清了，提示一下？"
    assert state.history_answer.answer_check["disposition"] == lookup_status
    assert state.history_answer.evaluate_answer(decision.text, decision.outcome).accepted
    assert state.progress["history_host_feedback"] is True


def test_ordinary_memory_context_does_not_enable_history_goal_checks():
    value = HistoryAnswerLedger()
    value.configure({"agent_history_query_request": False, "memory_context": {"decision": {"history_mode": "summary"}}})
    assert not value.enabled and value.negative_feedback() is None


@pytest.mark.parametrize("case,expected", [("found", "completed"), ("ambiguous", "partial"), ("unconfirmed", "partial"),
    ("not_found", "partial"), ("scope_limited", "blocked"), ("read_failed", "failed")])
def test_full_agent_history_task_distinguishes_real_completion_from_query_result(tmp_path, case, expected):
    service = MemoryService(store=MemoryStore(tmp_path / "history.db"), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("10001",)), clock=lambda: 200)
    tool = MemorySearchTool(service)
    base = _private_turn()
    group_message = replace(base.messages[0], conversation_type="group", conversation_id="100001")
    group_turn = replace(base, conversation_type="group", conversation_id="100001", messages=(group_message,),
                         combined_text="查这条诗")
    sources = [{"source_id": "original-one", "scope": "group:100001", "text": "夜色落在窗前。", "created_at": 100}]
    if case == "ambiguous":
        sources.append({"source_id": "original-two", "scope": "group:100001", "text": "风声留在门外。", "created_at": 101})
    if case in {"not_found", "scope_limited", "read_failed"}:
        sources = []
    def inspect(**kwargs):
        if case == "read_failed":
            raise OSError("must not expose private exception text")
        return {"scope": "private:10001" if case == "scope_limited" else "group:100001",
                "records": sources, "followup_context": "synthetic history receipt"}
    service.inspect = inspect
    answers = {"unconfirmed": "夜色落在窗前。这次读到的资料不能确认是否还有后文。", "found": "那句是夜色落在窗前。", "ambiguous": "有夜色落在窗前。和风声留在门外。两条，你指哪条？",
               "not_found": "合法范围已查完，但没核对到目标诗。", "scope_limited": "不能在这里读取那个群其他成员的原话，还没核对到那首诗。",
               "read_failed": "历史读取失败，还没核对到目标。"}
    class Runtime:
        def __init__(self):
            self.calls = []
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            if "历史答案复核" not in "\n".join(kwargs.get("system_extra_blocks", [])):
                return {"kind": "final", "text": answers[case], "task_outcome": expected}
            payload = json.loads(kwargs["user_prompt"])
            rows = payload["actual_records"]
            parts = ([{"text":"那句是","kind":"non_factual"},
                      {"text":"夜色落在窗前。","kind":"supported","claim_indices":[0]}]
                     if case == "found" else
                     [{"text":"有","kind":"non_factual"},
                      {"text":"夜色落在窗前。","kind":"supported","claim_indices":[0]},
                      {"text":"和","kind":"non_factual"},
                      {"text":"风声留在门外。","kind":"supported","claim_indices":[1]},
                      {"text":"两条，你指哪条？","kind":"non_factual"}])
            if case == "unconfirmed":
                parts = [{"text":"夜色落在窗前。", "kind":"supported", "claim_indices":[0]},
                         {"text":"这次读到的资料不能确认是否还有后文。", "kind":"bounded_unknown",
                          "lookup_index":0, "scope_fragment":"这次读到的资料"}]
            return {"disposition": case, "limitation_fragment": answers[case], "answer_parts":parts,
                "claims": [{"source_id": row["id"], "quote": row["text"], "answer_fragment": row["text"]} for row in rows],
                "coverage": [{"field": "诗句", "supported": True, "claim_indices": [0]}] if rows else []}
    runtime = Runtime()
    context = {"qq_turn": group_turn, "agent_completion_contract": {"requirements": [
        {"id": "poem", "effect": "history_answer", "targets": [], "checks": {"required_fields": ["诗句"]}}]}}
    result = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt="fixture"), handlers={"memory_search": tool},
        completion_gate=SmartCompletionVerifier(result_handlers={"memory_search": tool}), max_rounds=4).run(
            context=context, initial_transcript=[{"role": "user", "content": "查这条诗"}],
            first_decision=AgentDecision.tool(ToolInvocation("memory_search", {"query": "100001群里的诗"})))
    assert result.status == expected
    negative = HistoryAnswerLedger(enabled=True, lookups=[{"status": case}]).negative_feedback()
    assert result.final_text == (negative[0] if negative else answers[case])
    assert len(result.tool_envelopes) == 1 and len(runtime.calls) == (1 if negative else 2)
    if not negative:
        assert runtime.calls[-1]["native_tools"] is None
    assert result.progress["history_disposition"] == case
