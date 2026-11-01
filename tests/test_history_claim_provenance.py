"""旧话只证明说过：混合答复可转述，事件结论仍须独立来源。"""
import pytest

from test_history_answer import ledger, receipt


def mixed_ledger(origin=""):
    return ledger(receipt(records=[
        {"source_id": "user-correction", "scope": "group:100001", "text": "试件是紫色，不是绿色。",
         "created_at": 100, "metadata": {"role": "user"}},
        {"source_id": "assistant-reply", "scope": "group:100001", "text": "我会按紫色理解。",
         "created_at": 101, "metadata": {"role": "assistant", "origin": origin}},
    ]))


def mixed_review(value, *, utterance_kind="utterance", coverage_indices=None):
    user, assistant = value.sources
    return {"disposition": "found", "claims": [
        {"kind": "fact", "source_id": user, "quote": "试件是紫色，不是绿色。",
         "answer_fragment": "你说试件是紫色。"},
        {"kind": utterance_kind, "source_id": assistant, "quote": "我会按紫色理解。",
         "answer_fragment": "我当时说会按紫色理解。"},
    ], "coverage": [{"field": "原问题", "supported": True,
                     "claim_indices": [0] if coverage_indices is None else coverage_indices}],
        "answer_parts": [
            {"text": "你说试件是紫色。", "kind": "supported", "claim_indices": [0]},
            {"text": "我当时说会按紫色理解。", "kind": "supported", "claim_indices": [1]},
        ]}


@pytest.mark.parametrize("origin", ["", "scheduled_greeting"])
def test_attributed_utterance_does_not_block_independently_proven_answer(origin):
    value = mixed_ledger(origin)
    answer = "你说试件是紫色。我当时说会按紫色理解。"
    value.apply_review(answer, mixed_review(value), {})
    assert value.evaluate_answer(answer, "completed").accepted
    assert value.satisfies({"required_fields": ["原问题"], "min_sources": 1})


def test_utterance_cannot_cover_event_question_even_beside_user_evidence():
    value = mixed_ledger()
    answer = "你说试件是紫色。我当时说会按紫色理解。"
    value.apply_review(answer, mixed_review(value, coverage_indices=[1]), {})
    assert not value.evaluate_answer(answer, "completed").accepted
    assert value.answer_check["reason"] == "utterance_only_not_event_proof"


@pytest.mark.parametrize("kind", ["fact", "observation", "", None, [], {}])
def test_assistant_source_cannot_gain_fact_authority_by_malformed_kind(kind):
    value = mixed_ledger()
    answer = "你说试件是紫色。我当时说会按紫色理解。"
    value.apply_review(answer, mixed_review(value, utterance_kind=kind), {})
    assert not value.evaluate_answer(answer, "completed").accepted


def test_labeling_self_statement_utterance_does_not_prove_action_cause():
    value = ledger(receipt(records=[{
        "source_id": "assistant-motive", "scope": "group:100001", "text": "只是想陪你。",
        "created_at": 100, "metadata": {"role": "assistant"}}]))
    source = next(iter(value.sources))
    answer = "因为只是想陪你。"
    value.apply_review(answer, {"disposition": "found", "claims": [{"kind": "utterance",
        "source_id": source, "quote": "只是想陪你。", "answer_fragment": answer}],
        "coverage": [{"field": "原问题", "supported": True, "claim_indices": [0]}]}, {})
    assert not value.evaluate_answer(answer, "completed").accepted
    assert value.answer_check["reason"] == "utterance_only_not_event_proof"
