"""Unopened source state can inform coverage but cannot become body evidence."""
import copy
import json

from shinku.agent.loop import AgentDecision
from shinku.agent.planner import LLMPlanner
from test_research_answer_protocol import sdk_review, state_for_review
from test_research_evidence_ledger import ANSWER, BODY, URL, read_ledger, review


def test_answer_review_keeps_unread_ids_without_titles_summaries_or_claim_authority():
    ledger = read_ledger()
    ledger._candidate({"url": URL, "title": "READ_TITLE_SENTINEL", "snippet": "READ_CARD_SENTINEL"})
    identifier = ledger._candidate({"url": "https://source.example/other",
        "title": "UNREAD_TITLE_SENTINEL", "snippet": "UNREAD_CARD_SENTINEL"})
    before = copy.deepcopy(ledger.snapshot())
    payload = ledger.review_payload("核对两个来源的发布日期", {}, answer=ANSWER)
    assert payload["candidates"] == []
    assert payload["read_sources"][0]["text"] == BODY
    assert payload["unread_candidate_refs"] == [{"id": identifier, "url": "https://source.example/other",
        "relevance": "unreviewed", "body_evidence": False}]
    assert "SENTINEL" not in json.dumps(payload)
    assert ledger.snapshot() == before
    forged = review(ledger)
    forged["claims"][0]["source_id"] = identifier
    ledger.apply_answer_review(ANSWER, forged, {})
    assert ledger.answer_check["errors"] and not ledger.evaluate_answer(ANSWER, "completed").accepted


def test_available_unread_state_does_not_require_every_candidate_for_complete_answer():
    ledger = read_ledger()
    ledger._candidate({"url": "https://source.example/unrelated", "title": "Unused"})
    context = {"research_required_fields": ["发布日期"]}
    # Use the same established context contract as the existing evidence tests.
    ledger.intent["answer_fields"] = ["发布日期"]
    ledger.apply_answer_review(ANSWER, review(ledger), context)
    assert ledger.evaluate_answer(ANSWER, "completed").accepted
    assert ledger.review_payload("发布日期", context, answer=ANSWER)["unread_candidate_refs"]


def test_real_sdk_review_receives_composite_goal_coverage_and_read_state_rules():
    state = state_for_review()
    state.research._candidate({"url": "https://source.example/second", "title": "HIDDEN_TITLE_SENTINEL"})
    with sdk_review(json.dumps(review(state.research), ensure_ascii=False)) as (runtime, calls):
        LLMPlanner(runtime=runtime, system_prompt="neutral")._review_research_answer(state, AgentDecision.final(ANSWER))
        wire = json.dumps(calls[0], ensure_ascii=False)
        assert "unread_candidate_refs" in wire and "HIDDEN_TITLE_SENTINEL" not in wire
        assert "整个原始问题" in wire and "未读" in wire and "不能" in wire
        assert not calls[0].get("tools")
