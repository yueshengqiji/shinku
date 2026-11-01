"""Extractive annotations stay revocable interpretations, never facts."""
from dataclasses import replace
import pytest
from shinku.memory.projection.semantics import validate
from test_context_projection import engine, put, req


def output(rid, quote):
    return {"uncertain":True,"excerpts":[{"record_id":rid,"quote":quote}],
        "observations":[{"record_id":rid,"quote":quote,"subject":"Mira","attribute":"travel",
            "value":"not approved","condition":"if rain stops","time":"Friday",
            "author":"forged","state":"confirmed"}]}


@pytest.mark.parametrize("field,value", [("quote","Mira travel approved"), ("subject","Other"),
    ("condition","unconditional"), ("time","tomorrow"), ("value","complete")])
def test_observations_cannot_invent_or_rewrite_spans(field,value):
    quote = "Mira travel not approved; Friday only if rain stops."
    payload = {"records":[{"record_id":"a","text":quote}],"entities":[]}
    model = output("a",quote)
    model["observations"][0][field] = value
    with pytest.raises(ValueError):
        validate("distill",model,payload)


@pytest.mark.parametrize("mutation", ["none", "edit", "forget"])
def test_annotations_bound_to_author_version_and_revoked_after_model_work(tmp_path,mutation):
    store,projector = engine(tmp_path,semantic_enabled=True,max_chars=6000,record_chars=2400,
                             retrieval_strategy="model")
    quote = "Mira travel not approved; Friday only if rain stops."
    rows = [put(store,str(i),quote+" Some unrelated detail."*80,age=40*86400+i) for i in range(4)]
    calls = []
    def distill(payload):
        calls.append(payload)
        row = payload["records"][0]
        result = output(row["record_id"],quote)
        if mutation == "forget":
            store.forget(scope="private:owner",record_id=row["record_id"],query="",now=req().now)
        elif mutation == "edit":
            with store._connect() as conn:
                conn.execute("UPDATE memory_events SET text='Updated source' WHERE event_id=?",
                             (row["record_id"].removeprefix("event:"),))
        return result
    projector.semantic.model = distill
    try:
        view = projector.project(req("Mira travel",history_mode="retrieve"))
        assert len(calls) == 1
        annotations = [a for r in view.records for a in r["attribution"].get("source_observations",[])]
        if mutation != "none":
            assert not annotations
            assert "source_changed_during_projection" in view.diagnostics["gap_reasons"]
        else:
            assert annotations
            annotation = annotations[0]
            assert annotation["binding"]["author"] == "owner"
            assert annotation["binding"]["scope"] == "private:owner"
            assert annotation["binding"]["source_version"]
            assert annotation["state"] == "unconfirmed_interpretation"
            assert "author" not in annotation
            assert not annotation["execution_authority"]
            assert "if rain stops" in view.rendered
        with store._connect() as conn:
            assert conn.execute("SELECT count(*) FROM memory_facts").fetchone()[0] == 0
    finally:
        projector.stop()
