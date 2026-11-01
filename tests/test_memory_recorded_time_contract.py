"""Explicit saved-time bounds for model-selected memory reads."""
from datetime import datetime

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.query_time import normalize_recorded_time, recorded_time_window
from test_context_projection import NOW, put
from test_private_dialogue_continuity import turn


EMPTY = {"start": "", "end": ""}


@pytest.fixture
def memory(tmp_path):
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"),
        router=MemoryRouter(), clock=lambda: NOW,
        policy=MemoryPolicy(event_ttl_seconds=0,
            projection=ProjectionPolicy(enabled=True, retrieval_strategy="model")))
    yield service
    service.projection.stop()
    service.store.stop_embedding_worker()
    service.lifecycle_review.stop()


@pytest.mark.parametrize("value", [
    None, "", {"start": ""}, {"start": "", "end": "", "timezone": "UTC"},
    {"start": None, "end": ""}, {"start": 20261001, "end": ""},
    {"start": True, "end": ""}, {"start": "2026-02-30", "end": ""},
    {"start": "2026-1-01", "end": ""},
    {"start": "2026-10-02", "end": "2026-10-02"},
    {"start": "2026-10-03", "end": "2026-10-02"},
])
def test_invalid_saved_time_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_recorded_time(value)


@pytest.mark.parametrize("call", [
    {"query": "old"}, {"query": "old", "recorded_time": None},
    {"query": "old", "recorded_time": ""},
    {"query": "old", "recorded_time": {"start": "", "end": "", "scope": "private"}},
])
def test_model_tool_rejects_missing_or_malformed_time_field(memory, call):
    assert MemorySearchTool(memory).normalize_call(call) is None


def test_explicit_local_date_window_is_start_inclusive_end_exclusive(memory):
    tool = MemorySearchTool(memory)
    start = int(datetime(2026, 10, 2, 12).timestamp())
    end_day = int(datetime(2026, 10, 3, 12).timestamp())
    memory.clock = lambda: NOW
    before = put(memory.store, "before", "Jade venue note before.", age=NOW-start+86400)
    inside = put(memory.store, "inside", "Jade venue approval pending.", age=NOW-start)
    after = put(memory.store, "after", "Jade venue note after.", age=NOW-end_day)

    result = tool.execute(call={"query": "Jade venue", "mode": "retrieve", "limit": 8,
                                "recorded_time": {"start": "2026-10-02", "end": "2026-10-03"}},
        context={"qq_turn": turn("Jade venue", mid="current")})
    inspection = result["state_updates"]["memory_inspection"]
    ids = {row["record_id"] for row in inspection["records"]}
    assert ids == {inside.record_id}
    assert before.record_id not in ids and after.record_id not in ids
    coverage = inspection["coverage"]
    assert coverage["time_filter_source"] == "explicit_recorded_time"
    assert coverage["requested_time_window"] == [start - 12 * 3600, end_day - 12 * 3600]


def test_exact_start_is_included_and_exact_end_is_excluded(memory):
    tool = MemorySearchTool(memory)
    start_at, end_at = recorded_time_window({"start":"2026-10-02", "end":"2026-10-03"})
    assert start_at is not None and end_at is not None
    rows = {
        "at_start": put(memory.store, "at-start", "Jade exact boundary start.", age=NOW-start_at),
        "before_end": put(memory.store, "before-end", "Jade just before end.", age=NOW-(end_at-1)),
        "at_end": put(memory.store, "at-end", "Jade exact boundary end.", age=NOW-end_at),
    }

    result = tool.execute(call={"query":"Jade exact boundary", "mode":"retrieve", "limit":8,
        "recorded_time":{"start":"2026-10-02", "end":"2026-10-03"}},
        context={"qq_turn":turn("Jade exact boundary", mid="exact-boundary")})

    inspection = result["state_updates"]["memory_inspection"]
    found = {row["record_id"] for row in inspection["records"]}
    assert found == {rows["at_start"].record_id, rows["before_end"].record_id}
    assert rows["at_end"].record_id not in found
    assert inspection["coverage"]["requested_time_window"] == [start_at, end_at]


@pytest.mark.parametrize("recorded_time, expected", [
    ({"start": "2026-10-02", "end": ""}, {"inside", "after"}),
    ({"start": "", "end": "2026-10-03"}, {"before", "inside"}),
])
def test_open_saved_time_bounds_filter_only_the_declared_side(memory, recorded_time, expected):
    stamps = {
        "before": int(datetime(2026, 10, 1, 12).timestamp()),
        "inside": int(datetime(2026, 10, 2, 12).timestamp()),
        "after": int(datetime(2026, 10, 3, 12).timestamp()),
    }
    rows = {name: put(memory.store, name, f"Jade record {name}.", age=NOW-stamp)
            for name, stamp in stamps.items()}
    result = MemorySearchTool(memory).execute(call={"query": "Jade record", "mode": "retrieve", "limit": 8,
        "recorded_time": recorded_time}, context={"qq_turn": turn("Jade record", mid="open-bound")})
    found = {row["record_id"] for row in result["state_updates"]["memory_inspection"]["records"]}
    assert found == {rows[name].record_id for name in expected}


@pytest.mark.parametrize("mode", ["recent", "activity"])
def test_nonempty_saved_time_bounds_are_rejected_for_non_search_modes(memory, mode):
    call = {"query": "Jade", "mode": mode,
            "recorded_time": {"start": "2026-10-01", "end": ""}}
    assert MemorySearchTool(memory).normalize_call(call) is None


def test_nonempty_saved_time_bounds_are_rejected_for_exact_task_reads(memory):
    call = {"query": "Jade", "mode": "retrieve", "task_id": "task-1",
            "recorded_time": {"start": "2026-10-01", "end": ""}}
    assert MemorySearchTool(memory).normalize_call(call) is None


def test_open_bounds_and_legacy_internal_none_remain_distinct(memory):
    assert normalize_recorded_time(EMPTY) == EMPTY
    with pytest.raises(ValueError):
        recorded_time_window({"start": "", "end": "2026-10-01", "timezone": "UTC"})
    # The internal compatibility route is explicit in coverage; model tools
    # require the new field and cannot reach this None default.
    result = memory.inspect(conversation_id="owner", conversation_type="private",
        requester_id="owner", query="Jade", mode="retrieve", limit=4,
        message_id="current", user_text="Jade", self_id="bot")
    assert result["coverage"]["time_filter_source"] == "legacy_query"


def test_local_day_bounds_are_converted_independently_across_dst(monkeypatch):
    import shinku.memory.query_time as query_time
    from datetime import time as daytime
    from zoneinfo import ZoneInfo

    converted = []
    test_zone = ZoneInfo("America/New_York")
    class LocalMidnight:
        def __init__(self, day):
            self.day = day
        def astimezone(self):
            converted.append(self.day.isoformat())
            return self
        def timestamp(self):
            return datetime.combine(self.day, daytime.min, tzinfo=test_zone).timestamp()

    class LocalDateTime:
        @staticmethod
        def combine(day, _time):
            return LocalMidnight(day)

    monkeypatch.setattr(query_time, "datetime", LocalDateTime)
    start, end = query_time.recorded_time_window({"start": "2026-03-08", "end": "2026-03-09"})
    # The controlled zone's spring-forward transition makes this local day 23h.
    assert end - start == 23 * 3600
    assert converted == ["2026-03-08", "2026-03-09"]
