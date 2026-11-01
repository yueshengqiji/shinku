"""An unbound missing-object caveat cannot mint a completed original answer."""
import copy
import json

import pytest

from shinku.agent.research import digest
from test_public_query_partial_outcome import partial_evidence
from test_qq_query_continuation import PARTIAL
from test_research_answer_protocol import sdk_review


def overclaimed_review():
    value, _, review = partial_evidence()
    review["verdict"] = "supported"
    review["coverage"][0]["supported"] = True
    review["coverage"][0]["claim_indices"] = [0]
    review["missing_fields"] = []
    return value, review


@pytest.mark.parametrize("fault", ["orphan", "unknown_field", "unknown_condition", "unbound_text",
                                   "fields_type", "conditions_type", "row_type", "list_type", "too_many"])
def test_actual_sdk_invalid_limitation_does_not_get_silently_discarded(fault):
    value, review = overclaimed_review()
    row = review["limitations"][0]
    if fault == "orphan":
        row["fields"], row["condition_ids"] = [], []
    elif fault == "unknown_field":
        row["fields"] = ["not an original field"]
    elif fault == "unknown_condition":
        row["condition_ids"] = ["not an original condition"]
    elif fault == "unbound_text":
        row["answer_fragment"] = "This missing-object caveat is not in the answer."
    elif fault == "fields_type":
        row["fields"] = "原问题"
    elif fault == "conditions_type":
        row["condition_ids"] = {"id": "public_original"}
    elif fault == "row_type":
        review["limitations"] = [False]
    elif fault == "list_type":
        review["limitations"] = {"row": row}
    elif fault == "too_many":
        review["limitations"] = [copy.deepcopy(row) for _ in range(17)]
    sources = copy.deepcopy(value.ledger.sources)
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        assert value.review(PARTIAL, "completed") == (False, False)
        assert len(calls) == value.ledger.answer_review_calls == 1
        assert value.ledger.answer_check["errors"]
        assert value.ledger.sources == sources
        assert not calls[0].get("tools")


def test_supported_label_with_a_bound_unmet_field_still_cannot_be_completed():
    value, review = overclaimed_review()
    value.ledger.apply_answer_review(PARTIAL, review, value.state.context)
    assert not value.ledger.evaluate_answer(PARTIAL, "completed").accepted
    assert value.ledger.evaluate_answer(PARTIAL, "partial").accepted
    assert "原问题" in value.ledger.answer_check["missing_fields"]
    assert "原问题" not in value.ledger.answer_check["covered_fields"]
    assert not value.ledger.satisfies({"required_fields": ["原问题"], "min_sources": 1})
    assert value.ledger.recommended_outcome(PARTIAL, "completed", value.state.context) == "partial"


def test_valid_partial_keeps_sources_and_existing_status_only_downgrade():
    value, _, review = partial_evidence()
    value.ledger.apply_answer_review(PARTIAL, review, value.state.context)
    assert value.ledger.evaluate_answer(PARTIAL, "partial").accepted
    assert not value.ledger.evaluate_answer(PARTIAL, "completed").accepted
    assert value.ledger.recommended_outcome(PARTIAL, "completed", value.state.context) == "partial"


def test_old_review_cache_is_not_reused_or_given_a_new_budget():
    value, review = overclaimed_review()
    context = value.state.context
    old_key = digest(json.dumps({"answer": PARTIAL, "sources": sorted(value.ledger.sources),
        "fields": value.ledger.required_fields(context),
        "requirements": value.ledger.research_requirements(context)}, ensure_ascii=False, sort_keys=True))
    value.ledger.apply_answer_review(PARTIAL, review, context)
    value.ledger.answer_check.update(review_key=old_key, errors=[], limitations=[], missing_fields=[])
    value.ledger.answer_review_calls = 1
    review["limitations"][0].update(fields=[], condition_ids=[])
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        assert value.review(PARTIAL, "completed") == (False, False)
        assert len(calls) == 1 and value.ledger.answer_review_calls == 2
        value.review(PARTIAL + " changed", "completed")
        assert len(calls) == 1 and value.ledger.answer_review_calls == 2
        assert "answer_review_budget_or_length_exhausted" in value.ledger.review_errors
