"""Fast-query presentation keeps receipt scope and unsuccessful reads explicit."""
import copy
import json

import pytest

from shinku.agent_runtime.contracts import AgentScope
from shinku.agent_runtime.results import export_query_result
from shinku.result_expression.facts import fact_blocks
from shinku.result_expression.natural import check_paragraphs, paragraph_blocks


SCOPE = AgentScope("reader", "private:reader")
URL = "https://query.example/a"


def query_result(http_status=403, *, count=2):
    calls = [{"id": f"call-{i}", "name": "query_search" if i == 0 else "query_read",
              "status": "ok", "data": {"state_updates": {} if i == 0 else
                  {"url": URL, "http_status": http_status}}} for i in range(count)]
    return export_query_result(scope=SCOPE, request="核对杯子的价格并给来源", route="public_query",
        status="partial", completion={"accepted": False}, calls=calls,
        operation_id="query-1", reason="lightweight_public_query_fast_track")


def test_denied_body_is_a_required_scoped_observation_not_tool_success():
    value = query_result()
    before = copy.deepcopy(value)
    blocks = {b["id"]: b for b in paragraph_blocks(value, detail_policy_version=2)}
    observation = blocks["query_source_1"]
    assert observation["required"] and "403" in observation["text"] and URL in observation["text"]
    assert observation["semantic_review_required"]
    assert blocks["evidence_0"]["required"] is False
    assert blocks["evidence_1"]["required"] is False
    assert blocks["reason"]["required"] is False
    assert "legacy_source_completeness_unknown" not in json.dumps(value)
    assert not value["verification"]["accepted"] and "verified_answer" not in value
    assert value == before and value["evidence"][1]["execution_status"] == "ok"
    assert value["remaining"] and blocks["remaining"]["required"]


@pytest.mark.parametrize("count", [0, 2, 150])
def test_observed_count_only_describes_retained_receipts_and_preserves_gaps(count):
    value = query_result(count=count)
    blocks = {b["id"]: b for b in fact_blocks(value)}
    row = blocks["query_receipts"]
    assert row["required"] is False
    assert row["receipt_observation"]["retained_count"] == min(count, 128)
    assert row["receipt_observation"]["only_proves_retained_receipts"] is True
    assert "模型" in row["text"] and "总次数" in row["text"]
    if count > 128:
        assert value["information"]["export_gaps"]
        assert value["information"]["status"] == "incomplete"
        assert row["receipt_observation"]["has_export_gaps"]


def test_denied_http_value_cannot_be_changed_while_preserving_fact_ids():
    value = query_result()
    blocks = paragraph_blocks(value, detail_policy_version=2)
    candidate = {"schema_version": 2, "binding_hash": "bound", "paragraphs": [
        {"text": row["text"].replace("403", "200") if row["id"] == "query_source_1" else row["text"],
         "fact_ids": [row["id"]]} for row in blocks if row.get("required", True)]}
    error, _, _ = check_paragraphs(candidate, blocks, "bound")
    assert error


def test_model_marked_or_unbound_policy_does_not_hide_original_receipts():
    value = query_result()
    value["presentation"]["policy_source"] = "model"
    value["presentation"]["optional_success_receipts"] = False
    blocks = {b["id"]: b for b in paragraph_blocks(value, detail_policy_version=2)}
    assert "query_receipts" not in blocks
    assert blocks["evidence_1"].get("required", True)


@pytest.mark.parametrize("http_status", [True, "403", 200])
def test_non_error_or_untyped_http_value_does_not_invent_access_denial(http_status):
    blocks = {b["id"]: b for b in fact_blocks(query_result(http_status))}
    assert "query_source_1" not in blocks
