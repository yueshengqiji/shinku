"""旧来源缺少可用向量时仍能按正文定位，不改变层级或扩大授权范围。"""
from __future__ import annotations

import pytest

from shinku.memory import MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.retrieval_trace import capture_retrieval

NOW = 1_800_000_000
SCOPE = "private:owner"
QUERY = "枫桥资料"
TARGET = "枫桥资料只保存在离线磁盘，联网安装尚未得到同意。"


class UnavailableProvider:
    dimension = 2

    def collection_key(self):
        return "unavailable_coverage_fixture"

    def embed_text(self, text):
        raise RuntimeError("controlled encoding unavailable")

    def embed_texts(self, texts):
        raise RuntimeError("controlled encoding unavailable")


def make_store(tmp_path, *, provider_failure=False, kind="event"):
    store = MemoryStore(tmp_path / "memory.sqlite3",
                        embedding_provider=UnavailableProvider() if provider_failure else None,
                        semantic_enabled=provider_failure, event_scan_limit=10, fact_scan_limit=10,
                        lifecycle_policy=LifecyclePolicy(mode="off"))
    for i in range(25):
        body = TARGET if i == 0 else f"今日安排散步与整理书桌第{i}项"
        if kind == "event":
            store.record_event(conversation_id="owner", scope=SCOPE, source_id=f"case-{i}",
                               text=body, created_at=NOW - 40 * 86400 + i,
                               metadata={"role": "user", "sender_id": "owner"})
        else:
            store.upsert_fact(conversation_id="owner", scope=SCOPE, source_id=f"case-{i}",
                              fact_key=f"case-{i}", text=body, now=NOW - 40 * 86400 + i)
    return store


@pytest.mark.parametrize("kind", ["event", "fact"])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_old_literal_source_can_be_found_when_vectors_unavailable(tmp_path, kind, provider_failure):
    store = make_store(tmp_path, kind=kind, provider_failure=provider_failure)
    with capture_retrieval() as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=3)
    assert rows and rows[0].source_id == "case-0"
    assert rows[0].text == TARGET
    assert "semantic_similarity" not in rows[0].metadata
    assert any(event["stage"] == "store.lexical_fallback" for event in trace.events)


def test_fallback_keeps_scope_date_layer_and_revocation_guards(tmp_path):
    store = make_store(tmp_path)
    for source, scope, at, layer in [
        ("foreign", "private:other", NOW - 40 * 86400, "episodic"),
        ("outside-day", SCOPE, NOW, "episodic"),
        ("summary", SCOPE, NOW - 40 * 86400, "summary"),
        ("revoked", SCOPE, NOW - 40 * 86400, "episodic"),
    ]:
        store.record_event(conversation_id="owner", scope=scope, source_id=source, text=TARGET,
                           created_at=at, layer=layer)
    revoked = store.event_by_source_id(scope=SCOPE, source_id="revoked")
    store.forget(query="", scope=SCOPE, record_id=revoked.record_id, now=NOW)
    rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=12,
                        start_at=NOW - 40 * 86400, end_at=NOW - 39 * 86400,
                        layers={"episodic"}, require_term_overlap=True)
    assert [row.source_id for row in rows] == ["case-0"]


def test_projection_already_has_cross_window_literal_coverage(tmp_path):
    """对照证明原投影具备该能力，不把修复 Store 宣传成两条路径都新增。"""
    store = make_store(tmp_path)
    engine = ProjectionEngine(store, ProjectionPolicy(enabled=True))
    request = ProjectionRequest(scope=SCOPE, allowed_scopes=(SCOPE,), requester="owner",
                                self_id="bot", query=QUERY, now=NOW, message_id="current", history_mode="retrieve")
    try:
        handles, coverage, gaps = engine.archive.discover(request, Deadline(1000), limit=8)
        target = store.event_by_source_id(scope=SCOPE, source_id="case-0")
        assert (target.record_id, SCOPE, "") in handles
        assert coverage["source_fallback_scan"]
    finally:
        engine.stop()


def test_fallback_budget_exhaustion_keeps_existing_candidates(tmp_path, monkeypatch):
    """补查超过预算时保留近期结果并明确缺口，不把查询失败伪装成完整无命中。"""
    from shinku.memory import store_lexical_fallback
    store = make_store(tmp_path)
    ticks = iter([0.0, 1.0])
    # 替换模块自己的时钟对象，避免影响其它模块、测试调度或真实数据库。
    from types import SimpleNamespace
    monkeypatch.setattr(store_lexical_fallback, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with capture_retrieval() as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=3)
    assert rows and all(row.source_id != "case-0" for row in rows)
    event = next(row for row in trace.events if row["stage"] == "store.lexical_fallback")
    assert event["reason"] == "sql_budget_exhausted" and event["lexical_truncated"]
    assert store.stats()["active_events"] == 25


def test_complete_recent_window_skips_supplementary_read(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=False,
                        lifecycle_policy=LifecyclePolicy(mode="off"))
    store.record_event(conversation_id="owner", scope=SCOPE, source_id="only-source",
                       text=TARGET, created_at=NOW - 40 * 86400)
    with capture_retrieval() as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW)
    assert rows[0].source_id == "only-source"
    assert not any(row["stage"] == "store.lexical_fallback" for row in trace.events)
    assert any(row["reason"] == "scoped_sources_fit_window" for row in trace.events
               if row["stage"] == "store.lexical_fallback_skipped")


def test_sqlite_interrupt_stops_only_supplementary_read(tmp_path, monkeypatch):
    """实际触发 SQLite 进度回调，验证中断分支和连接清理，而非只模拟提前退出。"""
    from types import SimpleNamespace
    from shinku.memory import store_lexical_fallback
    store = make_store(tmp_path)
    with store._connect() as conn:
        conn.executemany("INSERT INTO memory_events(event_id,conversation_id,scope,text,layer,created_at,source_id) "
                         "VALUES(?,'owner',?,'无关天气','episodic',?,?)",
                         [(f"noise-{i}", SCOPE, NOW - i, f"noise-{i}") for i in range(1000)])
    ticks = iter([0.0, 0.0])
    monkeypatch.setattr(store_lexical_fallback, "time",
                        SimpleNamespace(monotonic=lambda: next(ticks, 1.0)))
    with capture_retrieval() as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW)
    assert rows and all(row.source_id != "case-0" for row in rows)
    event = next(row for row in trace.events if row["stage"] == "store.lexical_fallback")
    assert event["reason"] == "sql_budget_exhausted" and event["lexical_truncated"]
    assert store.stats()["active_events"] == 1025


@pytest.mark.parametrize("mode,tier", [("off", 1), ("shadow", 1), ("active", 1),
                                     ("active", 2), ("active", 3)])
def test_layered_source_lookup_does_not_reinforce_or_reclassify(tmp_path, mode, tier):
    """层级是合成夹具的受控状态；读取不能为了命中而更改强化或确认程度。"""
    store = make_store(tmp_path)
    store.configure_lifecycle(LifecyclePolicy(mode=mode))
    target = store.event_by_source_id(scope=SCOPE, source_id="case-0")
    if mode != "off":
        store.lifecycle.enroll(record_id=target.record_id, now=NOW)
        with store._connect() as conn:
            conn.execute("UPDATE memory_lifecycle SET tier=? WHERE record_type='event' AND record_id=?",
                         (tier, target.record_id.removeprefix("event:")))
    with store._connect() as conn:
        before = [tuple(row) for row in conn.execute("SELECT * FROM memory_lifecycle")]
        evidence_before = conn.execute("SELECT COUNT(*) FROM memory_lifecycle_evidence").fetchone()[0]
    for _ in range(3):
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=3)
        assert rows[0].source_id == "case-0"
    with store._connect() as conn:
        assert [tuple(row) for row in conn.execute("SELECT * FROM memory_lifecycle")] == before
        assert conn.execute("SELECT COUNT(*) FROM memory_lifecycle_evidence").fetchone()[0] == evidence_before


def test_retained_summary_does_not_recover_removed_original(tmp_path):
    store = make_store(tmp_path)
    target = store.event_by_source_id(scope=SCOPE, source_id="case-0")
    summary = "枫桥资料只保存在离线磁盘。"
    # 模拟已经完成回收的来源，不运行真实数据迁移或放宽回收许可。
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text=?,metadata_json=? WHERE event_id=?",
                     (summary, '{"raw_detail_unavailable":true,"lifecycle_representation":"summary"}',
                      target.record_id.removeprefix("event:")))
    rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=3)
    assert rows[0].text == summary
    assert rows[0].metadata["raw_detail_unavailable"]
    assert "联网安装" not in rows[0].text
