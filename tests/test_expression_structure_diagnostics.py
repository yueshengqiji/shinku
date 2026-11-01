"""Structure diagnostics never retain unknown IDs, text, or binding contents."""
import json
import pytest

from test_expression_semantic_boundary import sdk_pair
from test_result_expression import host, service, prepare


@pytest.mark.parametrize("change,missing,duplicates,unknown,binding", [
    (lambda c: {**c, "binding_hash": "private-secret-binding"}, [], [], 0, False),
    (lambda c: {**c, "layout": c["layout"][1:]}, ["status"], [], 0, True),
    (lambda c: {**c, "layout": [*c["layout"], c["layout"][0]]}, [], ["status"], 0, True),
    (lambda c: {**c, "layout": [*c["layout"], {"fact": "private-secret-id"}]}, [], [], 1, True),
])
def test_rejected_candidate_has_safe_structural_details(host, change, missing, duplicates, unknown, binding):
    with sdk_pair("普通衔接。", mutate=change) as (expression, checker, calls):
            expr = service(host, expression, checker=checker)
            record = prepare(expr)
            assert record["status"] == "fallback" and record["candidate"] is None
            checks = record["check"]["structure_checks"]
            assert checks["missing_fact_ids"] == missing
            assert checks["duplicate_fact_ids"] == duplicates
            assert checks["unknown_fact_count"] == unknown
            assert checks["binding_matches"] is binding
            assert "private-secret" not in json.dumps(checks)
            assert len(calls) == 1
            payload = calls[0][2]
            assert payload["layout_example"]["binding_hash"] == payload["binding_hash"]
            assert [row["fact"] for row in payload["layout_example"]["layout"] if "fact" in row] == payload["required_fact_ids"]
            assert prepare(expr)["expression_id"] == record["expression_id"]
            assert len(calls) == 1
