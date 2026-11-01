"""用真实 SQLite/服务入口验证诊断不改变检索、权限、渲染或强化。"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy
from shinku.memory.models import MemoryRecord
from shinku.memory.projection.contracts import ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.retrieval_trace import capture_retrieval, enabled, observe
from shinku.memory.source_reader import MemorySourceReader

NOW = 1_800_000_000
SCOPE = "private:owner"


class Provider:
    """可控向量只验证候选路径，不充当 BGE 或自然语义质量证明。"""
    dimension = 2

    def collection_key(self):
        return "retrieval_trace_fixture_v1"

    def embed_text(self, text):
        return [1.0, 0.0] if "宁静" in text or "安静" in text else [0.0, 1.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def store_at(tmp_path, *, semantic=True):
    return MemoryStore(tmp_path / "memory.sqlite3", embedding_provider=Provider() if semantic else None,
                       semantic_enabled=semantic, lifecycle_policy=LifecyclePolicy(mode="off"))


def put(store, text="我工作时喜欢安静的环境。", *, source="original", scope=SCOPE, at=NOW - 100):
    store.record_event(conversation_id=scope.split(":")[-1], scope=scope, source_id=source,
                       text=text, created_at=at, metadata={"role": "user", "sender_id": "owner"})
    return store.event_by_source_id(scope=scope, source_id=source)


def stages(trace, name):
    return [event for event in trace.snapshot()["events"] if event["stage"] == name]


def test_store_trace_preserves_results_database_and_excludes_text(tmp_path):
    store = store_at(tmp_path)
    target = put(store)
    store.backfill_embeddings(limit=20)
    expected = store.search(query="宁静的工作环境", scope=SCOPE, now=NOW)
    with store._connect() as conn:
        before = tuple(conn.iterdump())
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        actual = store.search(query="宁静的工作环境", scope=SCOPE, now=NOW)
    with store._connect() as conn:
        after = tuple(conn.iterdump())
    assert actual == expected
    assert after == before
    assert stages(trace, "store.semantic")[0]["watch"][0]["present"]
    assert stages(trace, "store.selected")[0]["watch"][0]["present"]
    report = json.dumps(trace.snapshot(), ensure_ascii=False)
    assert target.text not in report and "宁静的工作环境" not in report
    assert not enabled()


def test_trace_samples_do_not_mislabel_watched_candidate_as_absent():
    rows = [MemoryRecord(f"event:{i}", "episodic", SCOPE, f"private text {i}", NOW) for i in range(20)]
    with capture_retrieval(event_limit=2, record_limit=2, watch=(("event:19", SCOPE),)) as trace:
        observe("candidates", rows, query="must not appear", text="must not appear")
        observe("second")
        observe("third")
    snap = trace.snapshot()
    assert snap["dropped_events"] == 1 and not snap["complete"]
    event = snap["events"][0]
    assert event["count"] == 20 and len(event["sample"]) == 2 and event["sample_truncated"]
    assert event["watch"][0]["present"]
    assert "must not appear" not in json.dumps(snap)
    snap["events"].clear()
    assert len(trace.snapshot()["events"]) == 2


def test_nested_capture_restores_parent_even_on_failure():
    with capture_retrieval() as parent:
        observe("parent.before")
        with pytest.raises(RuntimeError, match="original failure"):
            with capture_retrieval() as child:
                observe("child")
                raise RuntimeError("original failure")
        observe("parent.after")
    assert [row["stage"] for row in parent.events] == ["parent.before", "parent.after"]
    assert [row["stage"] for row in child.events] == ["child"]
    observe("outside")
    assert not enabled() and len(parent.events) == 2


def test_concurrent_threads_own_their_trace():
    barrier = threading.Barrier(2)
    def work(label):
        with capture_retrieval() as trace:
            barrier.wait(timeout=5)
            observe(label)
        return trace.snapshot()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, label) for label in ("first", "second")]
        reports = [future.result(timeout=10) for future in futures]
    assert [[event["stage"] for event in report["events"]] for report in reports] == [["first"], ["second"]]


def test_child_async_task_cannot_write_into_inherited_parent_capture():
    async def run():
        async def child():
            observe("inherited.should.be.ignored")
            with capture_retrieval() as own:
                observe("child.own")
            return own
        with capture_retrieval() as parent:
            own = await asyncio.create_task(child())
            observe("parent")
        return parent, own
    parent, child = asyncio.run(run())
    assert [event["stage"] for event in parent.events] == ["parent"]
    assert [event["stage"] for event in child.events] == ["child.own"]


@pytest.mark.parametrize("options", [{"event_limit": 0}, {"record_limit": 1000},
                                      {"event_limit": True}, {"watch": (("a",),)}])
def test_invalid_trace_budget_is_rejected_before_business_call(options):
    with pytest.raises(ValueError):
        with capture_retrieval(**options):
            pytest.fail("invalid capture admitted")


def make_service(store, **overrides):
    policy = replace(MemoryPolicy(), semantic_enabled=True, evidence_package_enabled=True,
                     lifecycle=LifecyclePolicy(mode="off"), **overrides)
    return MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW, policy=policy)


def test_service_reports_no_query_instead_of_empty_search(tmp_path):
    service = make_service(store_at(tmp_path))
    with capture_retrieval() as trace:
        result = service.prepare_turn(conversation_id="owner", conversation_type="private", user_message="晚安")
    assert result.decision.history_mode == "none"
    assert stages(trace, "service.route")[0]["mode"] == "none"
    assert not stages(trace, "store.request")


@pytest.mark.parametrize("oversized", [False, True])
def test_service_distinguishes_selected_from_fully_rendered_source(tmp_path, oversized):
    store = store_at(tmp_path)
    body = "我工作时喜欢安静的环境。" * (12 if oversized else 1)
    target = put(store, body)
    service = make_service(store, render_record_chars=80)
    args = dict(conversation_id="owner", conversation_type="private", query="安静", mode="retrieve",
                requester_id="owner")
    expected = service.inspect(**args)
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        actual = service.inspect(**args)
    assert actual == expected
    assert stages(trace, "inspection.selected")[0]["watch"][0]["present"]
    assert stages(trace, "render.delivered")[-1]["watch"][0]["present"] is (not oversized)
    assert "retrieval-trace" not in actual["rendered"]


def test_legacy_partial_render_is_not_reported_as_complete_source(tmp_path):
    store = store_at(tmp_path)
    target = put(store, "我工作时喜欢安静的环境。" * 12)
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW,
                           policy=MemoryPolicy(render_record_chars=80, lifecycle=LifecyclePolicy(mode="off")))
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        text = service._render((target,))
    assert target.text not in text
    report = stages(trace, "render.delivered")[-1]
    assert report["partial_count"] == 1 and not report["watch"][0]["present"]


def test_revocation_is_visible_as_validation_rejection(tmp_path):
    store = store_at(tmp_path)
    target = put(store)
    store.forget(query="安静", scope=SCOPE, now=NOW)
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        result = MemorySourceReader(store).read_sources(requests=((target.record_id, SCOPE, ""),), now=NOW)
    assert result == (None,)
    event = stages(trace, "source.validated")[-1]
    assert event["requested"] == event["rejected"] == 1 and not event["watch"][0]["present"]


def test_projection_tracks_its_own_fusion_and_visible_sources(tmp_path):
    store = store_at(tmp_path)
    target = put(store, at=NOW - 40 * 86400)
    put(store, source="foreign", scope="private:someone_else")
    store.backfill_embeddings(limit=20)
    engine = ProjectionEngine(store, ProjectionPolicy(enabled=True))
    request = ProjectionRequest(scope=SCOPE, allowed_scopes=(SCOPE,), requester="owner", self_id="bot",
                                query="安静的环境", now=NOW, message_id="current", history_mode="retrieve")
    try:
        with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
            view = engine.project(request)
        assert stages(trace, "archive.vector")[0]["watch"][0]["present"]
        assert stages(trace, "archive.fused")[0]["watch"][0]["present"]
        assert stages(trace, "projection.visible")[0]["watch"][0]["present"]
        assert target.text in view.rendered
        assert "private:someone_else" not in json.dumps(trace.snapshot())
    finally:
        engine.stop()
