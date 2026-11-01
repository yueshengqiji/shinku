"""I03/I04/I08/I09: projection integrity and mandatory presentation rules."""
import copy
import json

from shinku.agent_runtime.information import information_for_result
from shinku.agent_runtime.results import seal_result
from shinku.result_expression.facts import check_candidate, fact_blocks, template
from test_result_expression import SCOPE, Model, canonical, prepare, service


def test_i04_every_projection_loss_has_counts_and_legacy_is_unknown():
    value = canonical()
    value.pop("result_hash")
    value["facts"] = [{"label": str(i), "value": i, "source": "executor_receipt"} for i in range(180)]
    value["facts"][0]["condition"] = "x" * 3000
    value["facts"][0]["nested"] = {str(i): i for i in range(90)}
    value["facts"][0]["deep"] = {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": 1}}}}}}}}
    sealed = seal_result(value)
    losses = sealed["information"]["projection"]["gaps"]
    assert {row["reason"] for row in losses} >= {"list_limit", "text_limit", "mapping_limit", "depth_limit"}
    assert next(row for row in losses if row["path"] == "/facts")["omitted"] == 52
    assert next(row for row in losses if row["path"] == "/facts/0/condition")["omitted"] == 600
    assert sealed["information"]["status"] == "incomplete"
    assert information_for_result({"schema_version": 1})["status"] == "unknown"


def test_i08_optional_audit_can_be_omitted_but_conditions_and_limits_cannot():
    value = canonical()
    value.pop("result_hash")
    value["presentation"] = {"schema_version": 1, "optional_success_receipts": True, "policy_source": "host_export"}
    value["information"] = {"schema_version": 1, "status": "incomplete", "required_status": "incomplete",
        "gaps": [{"source_id": "missing-source", "required": True, "reason": "source_unavailable"}]}
    blocks = fact_blocks(seal_result(value))
    required = [row for row in blocks if row.get("required", True)]
    assert "evidence_0" not in [row["id"] for row in required]
    candidate = {"binding_hash": "binding", "layout": [{"fact": row["id"]} for row in required]}
    reason, text, prose = check_candidate(candidate, blocks, "binding")
    assert not reason and not prose and "12.5 ms" in text and "离线隔离环境" in text and "source_unavailable" in text
    for missing in ("fact_0", "remaining", "information", "delivery", "deployment"):
        changed = copy.deepcopy(candidate)
        changed["layout"] = [row for row in changed["layout"] if row["fact"] != missing]
        assert check_candidate(changed, blocks, "binding")[0] == "expression_required_facts_missing_or_duplicated"
    duplicate = copy.deepcopy(candidate)
    duplicate["layout"].extend([{"fact": "evidence_0"}, {"fact": "evidence_0"}])
    assert check_candidate(duplicate, blocks, "binding")[0] == "expression_required_facts_missing_or_duplicated"


def test_i08_large_fallback_does_not_cut_mandatory_tail():
    blocks = [{"id": f"fact_{i}", "text": "x" * 2200 + f"；条件{i}：不能上线", "required": True} for i in range(16)]
    rendered = template(blocks)
    assert len(rendered) > 24000 and "条件15：不能上线" in rendered


def test_i09_old_cache_stays_bound_and_new_information_cannot_buy_another_expression(host):
    model = Model()
    svc = service(host, model)
    old = canonical()
    first = prepare(svc, old)
    assert prepare(svc, old) == first
    changed = copy.deepcopy(old)
    changed.pop("result_hash")
    changed["information"] = {"schema_version": 1, "status": "complete"}
    changed = seal_result(changed)
    new = prepare(svc, changed)
    assert new["status"] == "discarded" and new["fallback_reason"] == "expression_result_version_conflict"
    assert len(model.calls) == 1
    assert svc.archive.get(first["expression_id"], scope=SCOPE)["canonical_result"] == old


# Public fixture from the pre-existing expression suite; no private runtime.
from test_result_expression import host  # noqa: E402,F401
