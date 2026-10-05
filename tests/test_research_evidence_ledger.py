"""Research receipts are bounded, current-run, and independent of tool names."""
import copy
import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentDecision, AgentLoop, AgentState
from shinku.agent.planner import LLMPlanner
from shinku.agent.progress import ProgressLedger
from shinku.agent.research import ResearchLedger, digest
from shinku.agent.task_state import TaskContractError, validate_contract
from shinku.agent.verification import SmartCompletionVerifier
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


URL = "https://source.example/release"
BODY = "Project Cedar was released on 2026-09-20. It supports offline use."
ANSWER = f"Project Cedar 于 2026-09-20 发布。来源：{URL}"


class Search:
    agent_research_candidate_field = "results"
    agent_evidence_collection_field = "results"
    agent_observation_receipt_fields = ("results",)
    agent_progress_class = "observe"
    agent_permission_class = "read"

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        return {"followup_context": "Candidate only", "state_updates": {"results": [
            {"url": URL, "title": "Cedar release", "snippet": BODY}]}}


class Reader:
    agent_research_content_field = "page_text"
    agent_research_url_argument = "url"
    agent_progress_class = "change"
    agent_completion_effects = ("browser_access",)
    agent_completion_target_arguments = ("url",)

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        return {"followup_context": BODY, "state_updates": {"url": URL, "page_text": BODY,
            "site_access_ok": True, "http_status": 200, "verified_effects": ["browser_access"],
            "completion_targets": [URL], "content_sha256": digest(BODY), "content_offset": 0}}


def receipt(updates, *, status="ok"):
    return ToolResultEnvelope("r", status, "data", {"state_updates": updates})


def read_ledger(body=BODY):
    ledger = ResearchLedger(enabled=True)
    ledger.observe(ToolInvocation("different_reader_name", {"url": URL}), receipt({
        "url": URL, "page_text": body, "http_status": 200, "site_access_ok": True,
        "content_sha256": digest(body), "content_offset": 0}), Reader())
    return ledger


def test_answer_review_is_body_only_without_title_or_candidate_summary_substitution():
    ledger = read_ledger()
    source = next(iter(ledger.sources.values()))
    source["title"] = "TITLE_ONLY_SENTINEL"
    ledger._candidate({"url": URL, "title": "CARD_TITLE_SENTINEL", "snippet": "CARD_SUMMARY_SENTINEL"})
    payload = ledger.review_payload("original", {}, answer=ANSWER)
    assert "title" not in payload["read_sources"][0] and payload["candidates"] == []
    assert payload["read_sources"][0]["text"] == BODY
    assert "SENTINEL" not in json.dumps(payload)
    discovery = ledger.review_payload("original", {})
    assert discovery["candidates"][0]["title"] == "CARD_TITLE_SENTINEL"
    assert source["title"] == "TITLE_ONLY_SENTINEL"


def review(ledger, *, fragment="Project Cedar 于 2026-09-20 发布。", quote="Project Cedar was released on 2026-09-20."):
    return {"verdict": "supported", "claims": [{"source_id": next(iter(ledger.sources)),
        "answer_fragment": fragment, "quote": quote, "date_text": "2026-09-20", "date_kind": "event"}],
        "coverage": [{"field": "发布日期", "supported": True, "claim_indices": [0]}], "missing_fields": []}


def test_new_candidates_are_not_source_evidence_or_goal_progress():
    ledger = ResearchLedger()
    call = ToolInvocation("renamed_catalog", {"query": "Cedar"})
    envelope = receipt({"results": [{"url": URL, "title": "Cedar", "snippet": BODY, "id": "forged"}]})
    ledger.observe(call, envelope, Search())
    assert ledger.enabled and not ledger.sources and next(iter(ledger.candidates)) != "forged"
    progress = ProgressLedger().observe(call, envelope, progress_class="observe", candidate_only=True)
    assert progress["new_candidates"] and not progress["new_evidence"] and not progress["progress"]
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_budget_closeout_with_only_candidates_is_failed_not_partial_answer():
    result = AgentLoop(planner=lambda **_: AgentDecision.tool(ToolInvocation("catalog", {})),
        handlers={"catalog": Search()}, max_rounds=1).run()
    assert result.status == "failed" and result.reason == "research_answer_unverified"
    assert "搜索候选不等于已核实事实" in result.final_text
    assert "恢复" not in result.final_text


def test_candidate_review_is_advice_and_only_valid_known_ids_are_accepted():
    ledger = ResearchLedger(reader_tools={"reader"})
    ledger.observe(ToolInvocation("catalog", {}), receipt({"results": [{"url": URL}]}), Search())
    identifier = next(iter(ledger.candidates))
    ledger.apply_candidate_review({"candidate_reviews": [
        {"candidate_id": identifier, "relevance": "relevant", "reason": "值得核验", "recommended_read": True},
        {"candidate_id": "fake", "relevance": "relevant", "reason": "fake", "recommended_read": True},
    ]}, [identifier])
    assert ledger.read_before_search() and not ledger.sources
    assert ledger.candidates[identifier]["review_origin"] == "model_recommendation_not_executor_proof"


def test_bound_current_source_quote_date_and_visible_url_can_support_answer():
    ledger = read_ledger()
    ledger.intent = {"answer_fields": ["发布日期"]}
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    assert ledger.evaluate_answer(ANSWER, "completed").accepted
    assert ledger.satisfies({"required_fields": ["发布日期"], "require_dates": True, "min_sources": 1})
    assert not ledger.satisfies({"min_sources": 2})
    assert not ledger.satisfies({"required_fields": ["价格"]})


@pytest.mark.parametrize("change", ["fake_source", "fake_quote", "fake_date", "missing_url", "lookalike_url", "fake_fragment", "oversized_quote", "malformed_id"])
def test_model_cannot_fabricate_bindings_even_if_it_says_supported(change):
    ledger = read_ledger()
    value, answer = review(ledger), ANSWER
    claim = value["claims"][0]
    if change == "fake_source":
        claim["source_id"] = "invented_source"
    elif change == "fake_quote":
        claim["quote"] = "Project Cedar was released in October."
    elif change == "fake_date":
        claim["date_text"] = "2026-10-01"
    elif change == "missing_url":
        answer = answer.replace(URL, "website")
    elif change == "lookalike_url":
        answer = answer.replace(URL, URL + "-fake")
    elif change == "fake_fragment":
        claim["answer_fragment"] = "It costs 100 dollars."
    elif change == "oversized_quote":
        claim["quote"] = BODY + "x" * 300
    else:
        claim["source_id"] = {"id": "wrong shape"}
    ledger.apply_answer_review(answer, value, None)
    assert not ledger.evaluate_answer(answer, "completed").accepted


def test_source_version_change_and_latest_denial_invalidate_old_quote():
    ledger = read_ledger()
    old = next(iter(ledger.sources))
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt({"url": URL, "page_text": "Updated page.",
        "http_status": 200, "site_access_ok": True, "content_sha256": digest("Updated page.")}), Reader())
    assert old not in ledger.sources
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt({"url": URL, "page_text": "Denied",
        "http_status": 403, "site_access_ok": False}), Reader())
    assert not ledger.sources


def test_snapshot_restores_candidates_but_not_stale_body_proof():
    ledger = read_ledger()
    ledger.observe(ToolInvocation("catalog", {}), receipt({"results": [{"url": URL}]}), Search())
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    snapshot = ledger.snapshot()
    assert "text" not in snapshot["source_receipts"][0]
    restored = ResearchLedger()
    restored.configure({"search": Search(), "reader": Reader()}, {"agent_task_state": {"research": snapshot}})
    assert restored.candidates and not restored.sources and not restored.answer_check
    assert not restored.evaluate_answer(ANSWER, "completed").accepted


def test_redirect_outside_requested_domain_is_not_answer_evidence():
    ledger = ResearchLedger()
    ledger.observe(ToolInvocation("catalog", {}), receipt({"results": [{"url": URL}], "source_scope": "source.example"}), Search())
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt({"url": "https://other.example/page", "page_text": BODY,
        "http_status": 200, "site_access_ok": True}), Reader())
    assert not ledger.sources and "source_redirect_outside_scope" in ledger.review_errors


def test_semantic_partial_is_not_upgraded_and_empty_partial_is_not_answer():
    ledger = read_ledger()
    ledger.intent = {"answer_fields": ["发布日期", "价格"]}
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted
    assert ledger.evaluate_answer(ANSWER, "partial").accepted
    failure = "本轮没有取得相关正文，尚未回答。"
    ledger.apply_answer_review(failure, {"verdict": "no_answer", "claims": [], "coverage": []}, None)
    assert ledger.evaluate_answer(failure, "failed").accepted
    assert not ledger.evaluate_answer(failure, "partial").accepted


def test_research_contract_uses_fields_not_descriptive_url_placeholders():
    contract = {"goal": "查证项目发布日期", "deliverables": ["带来源的日期"], "scope": [], "requirements": [{
        "id": "answer", "description": "实际正文支持的发布日期", "effect": "research_answer", "targets": [],
        "checks": {"required_fields": ["发布日期"], "min_sources": 1, "require_dates": True}}]}
    assert validate_contract(contract)
    wrong = copy.deepcopy(contract)
    wrong["requirements"][0]["targets"] = ["相关网页"]
    with pytest.raises(TaskContractError):
        validate_contract(wrong)
    state = AgentState(context={"agent_completion_contract": contract}, research=read_ledger())
    state.research.apply_answer_review(ANSWER, review(state.research), state.context)
    state.tool_calls = [{"name": "reader", "status": "ok", "data": {"state_updates": {"verified_effects": ["browser_access"], "completion_targets": [URL]}}}]
    assert SmartCompletionVerifier().evaluate(state, ANSWER, outcome="completed").accepted


class Runtime:
    def __init__(self):
        self.calls, self.steps = [], 0

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        system = "\n".join(kwargs.get("system_extra_blocks") or [])
        if "【候选相关性复核" in system:
            data = json.loads(kwargs["user_prompt"])
            return {"intent": {"topic": "Project Cedar", "answer_fields": ["发布日期"], "suggested_queries": []},
                "candidate_reviews": [{"candidate_id": row["id"], "relevance": "relevant", "reason": "实体及任务对应，需读正文", "recommended_read": True} for row in data["candidates"]]}
        if "【答案证据与覆盖复核" in system:
            data = json.loads(kwargs["user_prompt"])
            return {"verdict": "supported", "claims": [{"source_id": data["read_sources"][0]["id"],
                "answer_fragment": "Project Cedar 于 2026-09-20 发布。", "quote": "Project Cedar was released on 2026-09-20.", "date_text": "2026-09-20", "date_kind": "event"}],
                "coverage": [{"field": "发布日期", "supported": True, "claim_indices": [0]}], "missing_fields": []}
        self.steps += 1
        if self.steps == 1:
            return {"tool_call": {"name": "catalog", "arguments": {"query": "Cedar"}}}
        if self.steps == 2:
            return {"tool_call": {"name": "reader", "arguments": {"url": URL}}}
        return {"kind": "final", "text": ANSWER, "task_outcome": "completed"}


def test_planner_loop_moves_from_review_to_read_and_binds_final_answer():
    runtime = Runtime()
    specs = [{"function": {"name": "catalog"}}, {"function": {"name": "reader"}}]
    result = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt="p", tool_specs=specs),
        handlers={"catalog": Search(), "reader": Reader()}).run(initial_transcript=[{"role": "user", "content": "Project Cedar 何时发布？"}])
    assert result.status == "completed" and result.final_text == ANSWER and len(result.tool_envelopes) == 2
    assert result.progress["candidate_review_calls"] == 1 and result.progress["answer_review_calls"] == 1
    main_read = runtime.calls[2]
    assert [spec["function"]["name"] for spec in main_read["native_tools"]] == ["reader"]
    for call in runtime.calls:
        if "复核，不执行工具" in "\n".join(call.get("system_extra_blocks") or []):
            assert call["native_tools"] is None and call["native_tool_choice"] == ""


def test_read_before_search_is_enforced_even_if_planner_ignores_hidden_schema():
    search, reader = Search(), Reader()
    steps = 0

    def planner(*, state):
        nonlocal steps
        steps += 1
        if steps == 1:
            return AgentDecision.tool(ToolInvocation("catalog", {}))
        if steps == 2:
            identifier = next(iter(state.research.candidates))
            state.research.apply_candidate_review({"candidate_reviews": [{"candidate_id": identifier,
                "relevance": "relevant", "reason": "read", "recommended_read": True}]}, [identifier])
            return AgentDecision.tools([ToolInvocation("catalog", {"query": "again"}), ToolInvocation("reader", {"url": URL})])
        state.research.apply_answer_review(ANSWER, review(state.research), None)
        return AgentDecision.final(ANSWER, outcome="completed")

    result = AgentLoop(planner=planner, handlers={"catalog": search, "reader": reader}).run()
    assert result.status == "completed"
    rejected = [call for call in result.transcript if call.get("role") == "tool" and call.get("status") == "error"]
    assert rejected[0]["data"]["code"] == "source_read_required"
    assert result.progress["source_discovery_calls"] == 1


@pytest.mark.parametrize("value", [None, [], {"claims": {}}, {"verdict": [], "claims": []},
    {"verdict": "supported", "claims": [{"source_id": [], "quote": {}, "answer_fragment": []}]},
    {"verdict": "supported", "claims": [], "coverage": [{"supported": True, "claim_indices": [True]}]}])
def test_malformed_answer_review_never_crashes_or_proves_completion(value):
    ledger = read_ledger()
    ledger.apply_answer_review(ANSWER, value, None)
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_quote_outside_read_chunk_cannot_use_full_page_hash_as_proof():
    ledger = read_ledger("Only the introduction was read.")
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_answer_review_cache_invalidates_when_source_changes_and_budget_is_bounded():
    runtime = Runtime()
    planner = LLMPlanner(runtime=runtime, system_prompt="p")
    state = AgentState(research=read_ledger(), transcript=[{"role": "user", "content": "发布日期？"}])
    decision = AgentDecision.final(ANSWER, outcome="completed")
    planner._review_research_answer(state, decision)
    planner._review_research_answer(state, decision)
    assert state.research.answer_review_calls == 1
    state.research.observe(ToolInvocation("reader", {"url": URL}), receipt({"url": URL,
        "page_text": BODY + " Updated note.", "http_status": 200, "site_access_ok": True}), Reader())
    planner._review_research_answer(state, decision)
    assert state.research.answer_review_calls == 2
    planner._review_research_answer(state, AgentDecision.final(ANSWER + " More.", outcome="completed"))
    assert state.research.answer_review_calls == 2
    assert not state.research.evaluate_answer(ANSWER + " More.", "completed").accepted


def test_reader_only_contract_is_active_but_plain_chat_with_readers_is_not_research():
    ledger = ResearchLedger()
    contract = {"requirements": [{"effect": "research_answer", "targets": []}]}
    ledger.configure({"reader": Reader()}, {"agent_completion_contract": contract})
    assert ledger.enabled and not ledger.sources
    runtime = SimpleNamespace(call_chat_json=lambda **_: {"kind": "final", "text": "那就休息一会儿。"})
    result = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt="p"), handlers={"reader": Reader()}).run(
        initial_transcript=[{"role": "user", "content": "今天有点累"}])
    assert result.status == "completed" and not result.tool_envelopes


@pytest.mark.parametrize("text,expected", [("青杉编辑器 7.4 发布日期，功率 13.7 W", []),
    ('创建 "7.4"', ["7.4"]), (r"保存到 C:\out\7.4", [r"c:\out\7.4"]), ("保存 a.txt", ["a.txt"])])
def test_target_compatibility_distinguishes_versions_from_explicit_numeric_filenames(text, expected):
    assert SmartCompletionVerifier._extract_targets(text) == expected


def test_failed_search_does_not_emit_a_new_candidate_event():
    class FailedSearch(Search):
        def execute(self, *, call, context):
            return {"error": "search unavailable", "state_updates": {"search_status": "unavailable", "results": []}}
    result = AgentLoop(planner=lambda **_: AgentDecision.tool(ToolInvocation("search", {})),
        handlers={"search": FailedSearch()}, max_rounds=1).run()
    assert result.progress["last_event"]["new_candidates"] is False


@pytest.mark.parametrize("change", ["new_version", "denied", "invalid_final_url", "outside_scope"])
def test_host_gate_independently_rejects_already_bound_answer_after_invalidated_reread(change):
    ledger = read_ledger()
    ledger.observe(ToolInvocation("catalog", {}), receipt({"results": [{"url": URL}], "source_scope": "source.example"}), Search())
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    updates = {"url": URL, "page_text": BODY, "http_status": 200, "site_access_ok": True}
    if change == "new_version":
        updates["page_text"] = "New version no longer contains the old claim."
    elif change == "denied":
        updates.update(http_status=403, site_access_ok=False)
    elif change == "invalid_final_url":
        updates["url"] = "https://user:password@source.example/"
    else:
        updates["url"] = "https://other.example/"
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt(updates), Reader())
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted
    assert not ledger.satisfies({"min_sources": 1})


def test_at_source_cap_old_version_is_still_invalidated_before_capacity_check():
    ledger = read_ledger()
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    original = next(iter(ledger.sources.values()))
    for index in range(23):
        ledger.sources[f"other-{index}"] = {**original, "id": f"other-{index}", "url": f"https://other.example/{index}"}
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt({"url": URL, "page_text": "Changed page.",
        "http_status": 200, "site_access_ok": True}), Reader())
    assert len(ledger.sources) == 24 and not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_single_research_requirement_allows_bound_partial_without_weakening_complete_contract():
    contract = {"requirements": [{"id": "answer", "effect": "research_answer", "targets": [],
        "checks": {"required_fields": ["发布日期", "价格"], "min_sources": 2}}]}
    state = AgentState(context={"agent_completion_contract": contract}, research=read_ledger(),
        transcript=[{"role": "user", "content": "查发布日期和价格"}])
    state.research.apply_answer_review(ANSWER, review(state.research), state.context)
    state.tool_calls = [{"name": "reader", "status": "ok", "data": {"state_updates": {
        "verified_effects": ["browser_access"], "completion_targets": [URL]}}}]
    verifier = SmartCompletionVerifier()
    partial = verifier.evaluate(state, ANSWER, outcome="partial")
    assert partial.accepted and partial.status == "partial" and partial.missing == ("answer",)
    assert not verifier.evaluate(state, ANSWER, outcome="completed").accepted
    assert contract["requirements"][0]["checks"]["min_sources"] == 2
    state.research.sources.clear()
    assert not verifier.evaluate(state, ANSWER, outcome="partial").accepted


def test_overlapping_field_labels_can_share_a_valid_claim_without_removing_contract_fields():
    ledger = read_ledger()
    ledger.intent = {"answer_fields": ["发布日期"]}
    context = {"agent_completion_contract": {"requirements": [{"effect": "research_answer", "targets": [],
        "checks": {"required_fields": ["发布日"]}}]}}
    value = review(ledger)
    value["coverage"].append({"field": "发布日", "supported": True, "claim_indices": [0]})
    ledger.apply_answer_review(ANSWER, value, context)
    assert ledger.evaluate_answer(ANSWER, "completed").accepted
    assert ledger.required_fields(context) == ["发布日期", "发布日"]


def missing_date_answer():
    body = "Cedar warranty lasts 30 months. This page does not state a publication date."
    answer = f"保修期限为 30 个月。页面未标注公告发布日期，日期尚无法核实。来源：{URL}"
    contract = {"requirements": [{"id": "answer", "effect": "research_answer", "targets": [],
        "checks": {"required_fields": ["保修期限", "公告发布日期"], "require_dates": True, "min_sources": 1}}]}
    context = {"agent_completion_contract": contract}
    ledger = read_ledger(body)
    value = {"verdict": "partial", "claims": [
        {"source_id": next(iter(ledger.sources)), "answer_fragment": "保修期限为 30 个月。",
         "quote": "Cedar warranty lasts 30 months.", "date_text": "", "date_kind": "unknown"},
        {"source_id": next(iter(ledger.sources)), "answer_fragment": "页面未标注公告发布日期，日期尚无法核实。",
         "quote": "This page does not state a publication date.", "date_text": "", "date_kind": "unknown"}],
        "coverage": [{"field": "保修期限", "supported": True, "claim_indices": [0]}],
        "missing_fields": ["公告发布日期"], "limitations": [{"answer_fragment": "页面未标注公告发布日期，日期尚无法核实。",
            "fields": ["公告发布日期"], "condition_ids": ["answer"]}]}
    return ledger, context, answer, value


def test_review_receives_actual_contract_checks_and_cache_covers_changes():
    ledger, context, answer, value = missing_date_answer()
    payload = ledger.review_payload("保修期限和发布日期？", context, answer=answer)
    assert payload["research_requirements"][0]["checks"]["require_dates"] is True
    first = ledger.answer_review_key(answer, context)
    context["agent_completion_contract"]["requirements"][0]["checks"]["min_sources"] = 2
    second = ledger.answer_review_key(answer, context)
    assert second != first
    context["agent_completion_contract"]["requirements"][0]["checks"]["require_dates"] = False
    assert ledger.answer_review_key(answer, context) != second
    assert payload["research_requirements"][0]["checks"]["min_sources"] == 1


def test_overclaimed_completion_lowers_only_with_bound_fact_and_explicit_reviewed_gaps():
    ledger, context, answer, value = missing_date_answer()
    original = copy.deepcopy(context)
    ledger.apply_answer_review(answer, value, context)
    assert ledger.recommended_outcome(answer, "completed", context) == "partial"
    assert ledger.recommended_outcome(answer, "", context) == "partial"
    state = AgentState(context=context, research=ledger, tool_calls=[{"name": "reader", "status": "ok"}])
    verifier = SmartCompletionVerifier()
    assert verifier.evaluate(state, answer, outcome="partial").accepted
    assert not verifier.evaluate(state, answer, outcome="completed").accepted
    assert context == original


@pytest.mark.parametrize("change", ["missing_caveat", "forged_fragment", "forged_condition", "unacknowledged_field",
    "unsupported", "bad_quote", "changed_answer", "stale_source", "mixed_contract"])
def test_automatic_partial_does_not_relax_evidence_or_side_effects(change):
    ledger, context, answer, value = missing_date_answer()
    if change == "missing_caveat":
        value.pop("limitations")
    elif change == "forged_fragment":
        value["limitations"][0]["answer_fragment"] = "This caveat is not in the answer."
    elif change == "forged_condition":
        value["limitations"][0]["condition_ids"] = ["invented"]
    elif change == "unacknowledged_field":
        value["missing_fields"].append("未说明的价格")
    elif change == "unsupported":
        value["verdict"] = "unsupported"
    elif change == "bad_quote":
        value["claims"][0]["quote"] = "Forged warranty."
    elif change == "mixed_contract":
        context["agent_completion_contract"]["requirements"].append({"id": "send", "effect": "send", "targets": ["recipient"]})
    ledger.apply_answer_review(answer, value, context)
    if change == "changed_answer":
        answer += "Actually all requirements are complete."
    elif change == "stale_source":
        ledger.sources.clear()
    assert ledger.recommended_outcome(answer, "completed", context) == "completed"


@pytest.mark.parametrize("outcome", ["failed", "blocked", "waiting_background", "partial"])
def test_auto_partial_preserves_explicit_noncompletion_states(outcome):
    ledger, context, answer, value = missing_date_answer()
    ledger.apply_answer_review(answer, value, context)
    assert ledger.recommended_outcome(answer, outcome, context) == outcome


def test_live_planner_path_delivers_same_reviewed_answer_as_partial_without_retry_loop():
    ledger, context, answer, value = missing_date_answer()

    class MissingDateRuntime:
        calls = []

        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            if "【答案证据与覆盖复核" in "\n".join(kwargs.get("system_extra_blocks") or []):
                payload = json.loads(kwargs["user_prompt"])
                assert payload["research_requirements"][0]["checks"]["require_dates"] is True
                return value
            return {"kind": "final", "text": answer, "task_outcome": "completed"}

    runtime = MissingDateRuntime()
    state = AgentState(research=ledger, context=context)
    planner = LLMPlanner(runtime=runtime, system_prompt="p")
    decision = planner(state=state)
    assert decision.outcome == "partial" and decision.text == answer
    assert len(runtime.calls) == 2 and ledger.answer_review_calls == 1
    assert state.progress["research_outcome_downgrades"] == 1
    assert context["agent_completion_contract"]["requirements"][0]["checks"]["require_dates"] is True


def test_actual_page_links_are_candidates_not_body_evidence_and_preserve_scope():
    class LinkedReader(Reader):
        agent_research_link_field = "source_links"

    ledger = ResearchLedger(enabled=True)
    ledger.observe(ToolInvocation("catalog", {}), receipt({"results": [{"url": URL}], "source_scope": "source.example"}), Search())
    updates = {"url": URL, "page_text": "A documentation directory, not a release date.", "http_status": 200, "site_access_ok": True,
        "source_links": [{"url": "https://source.example/actual", "title": "Release notes"},
                         {"url": "https://other.example/actual", "title": "Outside scope"}]}
    ledger.observe(ToolInvocation("reader", {"url": URL}), receipt(updates), LinkedReader())
    assert len(ledger.sources) == 1
    linked = next(row for row in ledger.candidates.values() if row["url"].endswith("/actual"))
    assert linked["origin"] == "executor_page_link" and linked["source_scope"] == "source.example"
    assert linked["read_attempts"] == 0 and linked["relevance"] == "unreviewed"
    assert all(row["url"].startswith("https://source.example/") for row in ledger.candidates.values())
    ledger.apply_answer_review(ANSWER, review(ledger), None)
    assert not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_budget_finalization_keeps_honest_research_failure_instead_of_empty_partial():
    def planner(*, state):
        if not state.tool_calls:
            return AgentDecision.tool(ToolInvocation("catalog", {}))
        text = "本轮没有取得目标正文，尚未核实发布日期，任务未完成。"
        state.research.apply_answer_review(text, {"verdict": "no_answer", "claims": [], "coverage": []}, None)
        return AgentDecision.final(text, outcome="failed")

    result = AgentLoop(planner=planner, handlers={"catalog": Search()}, max_rounds=1).run()
    assert result.status == "failed" and "没有取得目标正文" in result.final_text
    assert result.reason != "research_answer_unverified"


def test_updated_date_requires_explicit_role_without_reinterpreting_older_contract():
    ledger = read_ledger()
    value = review(ledger)
    value["claims"][0]["date_kind"] = "updated"
    ledger.apply_answer_review(ANSWER, value, None)
    assert not ledger.satisfies({"require_dates": True})
    assert ledger.satisfies({"require_dates": True, "date_kinds": ["updated"]})
    assert not ledger.satisfies({"require_dates": True, "date_kinds": ["publication"]})
    assert not ledger.satisfies({"require_dates": True, "date_kinds": ["unknown"]})


@pytest.mark.parametrize("kinds", [[], ["unknown"], ["updated", "event", "publication", "updated"], "updated", [True]])
def test_malformed_date_roles_cannot_become_a_contract(kinds):
    contract = {"goal": "查询更新日期", "deliverables": ["日期及来源"], "scope": [], "requirements": [{
        "id": "date", "description": "核实更新日期", "effect": "research_answer", "targets": [],
        "checks": {"require_dates": True, "date_kinds": kinds}}]}
    with pytest.raises(TaskContractError):
        validate_contract(contract)


def test_research_fallback_when_sources_read_and_review_invalid():
    step = 0
    def planner(*, state):
        nonlocal step
        step += 1
        if step == 1:
            return AgentDecision.tool(ToolInvocation("reader", {"url": URL}))
        text = "论文主要提出了离线部署框架。"
        state.research.apply_answer_review(text, {"verdict": "invalid", "errors": ["invalid_semantic_review"]}, None)
        return AgentDecision.final(text)

    result = AgentLoop(
        planner=planner,
        handlers={"reader": Reader()},
        max_rounds=4,
    ).run(context={"agent_completion_contract": {"requirements": [{"effect": "research_answer", "targets": []}]}})
    assert result.status == "partial"
    assert result.reason == "research_evidence_partial_fallback"
    assert "离线部署框架" in result.final_text


def test_research_fallback_at_max_rounds_preserves_last_final_text():
    step = 0
    def planner(*, state):
        nonlocal step
        step += 1
        if step == 1:
            return AgentDecision.tool(ToolInvocation("reader", {"url": URL}))
        if step == 2:
            return AgentDecision.final("这是在第2轮总结的内容。")
        return AgentDecision.tool(ToolInvocation("reader", {"url": URL + f"?page={step}"}))

    class ReaderMulti(Reader):
        def execute(self, *, call, context):
            res = super().execute(call=call, context=context)
            res["state_updates"]["url"] = (call or {}).get("url", URL)
            return res

    result = AgentLoop(
        planner=planner,
        handlers={"reader": ReaderMulti()},
        max_rounds=3,
    ).run(context={"agent_completion_contract": {"requirements": [{"effect": "research_answer", "targets": []}]}})
    assert result.status == "partial"
    assert result.reason in {"research_evidence_partial_fallback", "max_rounds_partial"}
    assert "第2轮总结的内容" in result.final_text
