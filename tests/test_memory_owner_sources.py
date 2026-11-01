"""记忆服务公开来源接口与展示预算的合成边界测试，不读取生产记忆。"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.memory.owner_continuity import (
    OwnerContinuityBudget, owner_continuity, render_owner_continuity, select_owner_sources,
)
from test_desktop_qq_continuity import NOW, record, service


READ = dict(conversation_id="desktop:reader", conversation_type="desktop",
            requester="owner", self_id="bot", message_id="new")


def payload(view):
    return json.loads(view["rendered"].split("\n")[-1])


def test_public_read_keeps_undisplayed_sources_and_does_not_write(tmp_path, monkeypatch):
    memory = service(tmp_path / "memory.db", [NOW])
    for index in range(10):
        record(memory, f"计划{index}：只有确认才执行，目前尚未确认。", mid=str(index))
    with memory.store._connect() as conn:
        before = tuple(conn.iterdump())
    monkeypatch.setattr(memory.store, "expire", lambda **kw: pytest.fail("只读来源接口不得做过期清理"))
    batch = memory.read_recent_owner_sources(**READ)
    view = owner_continuity(memory, **READ)
    assert len(batch.records) == len(batch.versions) == 10
    assert len(view["handles"]) == 6
    assert payload(view)["omitted_recent_sources"] == 4
    assert all("目前尚未确认" in item["text"] for item in payload(view)["records"])
    with memory.store._connect() as conn:
        assert tuple(conn.iterdump()) == before


@pytest.mark.parametrize("kind,requester,enabled", [
    ("group", "owner", True), ("desktop", "other", True),
    ("private", "other", True), ("desktop", "owner", False),
])
def test_public_read_rejects_scope_or_identity_before_storage(tmp_path, monkeypatch,
                                                          kind, requester, enabled):
    from shinku.memory import owner_sources
    memory = service(tmp_path / "memory.db", [NOW])
    if not enabled:
        memory.policy = replace(memory.policy, owner_continuity_seconds=0)
    monkeypatch.setattr(owner_sources, "connection", lambda *a, **kw: pytest.fail("拒绝后不得读库"))
    assert memory.read_recent_owner_sources(**{**READ, "conversation_type": kind,
                                               "requester": requester}) is None


def test_public_read_rechecks_origin_revision_expiry_and_revocation(tmp_path):
    clock = [NOW]
    memory = service(tmp_path / "memory.db", clock)
    record(memory, "原件条件尚未确认。")
    first = memory.read_recent_owner_sources(**READ)
    with memory.store._connect() as conn:
        row = conn.execute("SELECT event_id,metadata_json FROM memory_events WHERE scope='private:owner'").fetchone()
        meta = {**json.loads(row["metadata_json"]), "source_revision": "corrected"}
        conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                     (json.dumps(meta), row["event_id"]))
    changed = memory.read_recent_owner_sources(**READ)
    assert changed.records == first.records
    assert changed.versions != first.versions
    clock[0] += 300
    assert not memory.read_recent_owner_sources(**READ).records
    clock[0] = NOW
    memory.store.forget(query="原件条件", scope="private:master", now=NOW)
    assert not memory.read_recent_owner_sources(**READ).records


def test_presentation_only_needs_public_source_contract(tmp_path):
    memory = service(tmp_path / "memory.db", [NOW])
    record(memory, "仅供本轮承接，尚未执行。")
    batch = memory.read_recent_owner_sources(**READ)
    # 无 store、policy 或内部读取器的调用方仍可组装提示；来源核验已由公开入口负责。
    facade = SimpleNamespace(read_recent_owner_sources=lambda **kwargs: batch)
    assert owner_continuity(facade, **READ) == owner_continuity(memory, **READ)


def test_smaller_budget_keeps_whole_conditions_and_all_version_dependencies(tmp_path):
    memory = service(tmp_path / "memory.db", [NOW])
    record(memory, "短安排：只有批准才执行。", mid="short")
    record(memory, "长材料。" * 100 + "但是尚未批准，不能执行。", mid="long")
    batch = memory.read_recent_owner_sources(**READ)
    budget = OwnerContinuityBudget(max_records=1, record_chars=80, total_chars=80)
    selected = select_owner_sources(batch, budget=budget)
    view = render_owner_continuity(batch, selected)
    assert [r.text for r in selected] == ["短安排：只有批准才执行。"]
    assert payload(view)["omitted_recent_sources"] == 1
    assert "长材料" not in view["rendered"]
    assert owner_continuity(memory, **READ, budget=budget) == view
    with memory.store._connect() as conn:
        row = conn.execute("SELECT event_id,metadata_json FROM memory_events WHERE scope='private:owner' AND source_id='turn:owner:long'").fetchone()
        meta = {**json.loads(row["metadata_json"]), "source_revision": "new_condition"}
        conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                     (json.dumps(meta), row["event_id"]))
    after = owner_continuity(memory, **READ, budget=budget)
    assert after["handles"] == view["handles"]
    assert after["signature"] != view["signature"]


def test_exact_budget_boundaries_and_newest_first_selection(tmp_path):
    clock = [NOW]
    memory = service(tmp_path / "memory.db", clock)
    for index, text in enumerate(("旧条件尚未确认", "中间条件未批准", "最新条件未批准")):
        clock[0] += 1
        record(memory, text, mid=str(index))
    batch = memory.read_recent_owner_sources(**READ)
    selected = select_owner_sources(batch, budget=OwnerContinuityBudget(
        max_records=2, record_chars=7, total_chars=14))
    assert [r.text for r in selected] == ["中间条件未批准", "最新条件未批准"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_display_budget_is_rejected(value):
    with pytest.raises(ValueError, match="invalid_owner_continuity_budget"):
        OwnerContinuityBudget(total_chars=value)
