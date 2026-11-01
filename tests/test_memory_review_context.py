"""Actual memory readers must preserve bounded coverage and host time for review.

Scripted verdicts here check transport and enforcement, not semantic quality.
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from shinku.agent.history_answer import review_history_answer
from shinku.memory.tools import MemorySearchTool
from test_history_answer import ledger, receipt
from test_memory_read_acceptance import ANSWER, QUERY, SOURCE, bridge_factory
from test_context_projection import NOW
from test_private_dialogue_continuity import turn


@pytest.mark.parametrize("entry", ["legacy", "task"])
@pytest.mark.parametrize("strategy", ["legacy", "model"])
@pytest.mark.parametrize("mode", ["recent", "retrieve", "summary"])
def test_actual_reader_transfers_time_query_and_incomplete_coverage(bridge_factory, entry, strategy, mode):
    capabilities = None
    if entry == "task" and strategy == "legacy":
        # This host explicitly grants the read; legacy retrieval alone does not.
        from shinku.qq.interaction_tools import ChatReadCapability
        capabilities = {"memory_search": ChatReadCapability(
            "memory_search", "Read the current conversation history.",
            {"type": "object", "additionalProperties": False, "properties": {
                "query": {"type": "string", "maxLength": 240},
                "mode": {"type": "string", "enum": ["recent", "retrieve", "summary"], "maxLength": 8},
                "task_id": {"type": "string", "maxLength": 240},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12}}},
            max_result_bytes=64000)}
    bridge, chat, memory, _ = bridge_factory(entry, "accepted", strategy=strategy, mode=mode,
                                       read_capabilities=capabilities)
    reads=[];original=memory.inspect
    def inspect(**kwargs):
        result=original(**kwargs);reads.append(result);return result
    memory.inspect=inspect
    reply = bridge(turn(QUERY, mid="coverage"))
    assert reply.agent_status == "completed", reply.final_text
    lookup = MemorySearchTool._lookup_receipt(reads[0], query=QUERY)
    assert lookup["query"] == "写长篇 安静 背景音乐"
    assert lookup["mode"] == mode
    assert lookup["read_at"]["unix"] == NOW
    assert datetime.fromisoformat(lookup["read_at"]["local_iso"]).timestamp() == NOW
    assert lookup["coverage"]["complete"] is False
    assert lookup["coverage"]["absence_means"] == "unknown"
    if strategy == "model":
        assert lookup["coverage"]["requested_time_window"] == [None, None]
    source = next(row for row in reads[0]["records"] if SOURCE in row["text"])
    assert source["created_at"] < NOW
    if not (strategy == "legacy" and mode == "summary"):
        assert source["created_at"] == NOW - 40 * 86400
    assert len(chat.calls) == 2 and chat.payload is None


def test_date_filtered_read_retains_its_actual_window_and_clock(bridge_factory):
    _, _, memory, _ = bridge_factory("legacy", "accepted")
    reader = MemorySearchTool(memory)
    result = reader.execute(call={"query": "2027-01-15 写长篇", "mode": "retrieve", "limit": 4,
                                 "recorded_time": {"start": "2027-01-15", "end": "2027-01-16"}},
                            context={"qq_turn": turn("核对那次安排", mid="date-read")})
    inspection = result["state_updates"]["memory_inspection"]
    lookup = result["state_updates"]["memory_lookup"]
    assert lookup["coverage"]["requested_time_window"] == inspection["coverage"]["requested_time_window"]
    assert all(type(v) is int for v in lookup["coverage"]["requested_time_window"])
    assert lookup["read_at"] == inspection["read_at"]


def test_missing_coverage_is_unknown_even_when_query_completed():
    lookup = MemorySearchTool._lookup_receipt({"scope": "private:owner", "records": []}, query="旧事")
    assert lookup["query_completed"] is True
    assert lookup["coverage"] == {"complete": False, "absence_means": "unknown"}
    assert lookup["read_at"] is None


@pytest.mark.parametrize("stamp", [None, "unknown", True, float("nan"), float("inf")])
def test_bad_source_timestamp_is_unknown_not_current_time(stamp):
    value = ledger(receipt(records=[{"source_id": "old", "scope": "group:100001",
                                    "text": SOURCE, "created_at": stamp}]))
    assert next(iter(value.sources.values()))["created_at_iso"] is None


def test_model_cannot_supply_clock_or_coverage(bridge_factory):
    _, _, memory, _ = bridge_factory("legacy", "accepted")
    reader = MemorySearchTool(memory)
    for injected in ({"read_at": NOW}, {"coverage": {"complete": True}}):
        assert reader.normalize_call({"query": "旧事", **injected}) is None


def test_review_policy_reaches_existing_call_without_extra_round_or_expected_answer():
    calls = []
    value = ledger()
    review_history_answer(value, runtime=SimpleNamespace(call_chat_json=lambda **kw: calls.append(kw) or {}),
                          system_prompt="Synthetic persona", answer=ANSWER, goal=QUERY, context={})
    assert len(calls) == 1
    instruction = "".join(calls[0]["system_extra_blocks"])
    # These assert the reviewed contract is delivered; they cannot assert that
    # an actual model follows it. That acceptance gate remains a separate run.
    for field in ("read_at", "created_at", "coverage.complete", "适用范围", "相对日期", "否定"):
        assert field in instruction
    assert "写长篇" not in instruction and "苍鹭" not in instruction
    assert calls[0]["native_tools"] is None
    assert calls[0]["prompt_cache_key"].endswith("history-evidence:v7")
