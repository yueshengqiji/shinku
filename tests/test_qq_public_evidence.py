"""Source presence, answer acceptance, model label and send receipt differ."""
import json

import pytest

from test_qq_public_query_fast_track import Runtime, SourceTool, bridge, call, final, turn
from test_qq_public_query_fast_track import close_coordinators
from shinku.qq.delivery import DeliveryResult


def pipeline(*, outcome="completed", answer=None):
    runtime = Runtime(call("private_browser_open", {"url": "https://weather.example/source"}),
                      final(answer or "烟台多云，18°C。来源：https://weather.example/source", outcome))
    host, search, reader = bridge(runtime)
    return runtime, host, search, reader


def test_supported_failed_label_gets_one_status_only_correction_not_an_auto_upgrade():
    runtime, host, search, reader = pipeline(outcome="failed")
    answer = runtime.responses[-1]["text"]
    reply = host.handle_turn(turn())
    assert reply.agent_status == "completed" and reply.final_text == answer
    assert len(runtime.calls) == 4 and len(reader.calls) == 1 and not search.calls
    assert all(row["native_tools"] is None for row in runtime.calls[-2:])
    trace = next(r for r in host.trace_store.snapshot(limit=1)[0]["stages"] if r["name"] == "fast_public_query_finished")
    assert trace["model_outcome"] == "failed" and trace["goal_status"] == "verified" and trace["outcome"] == "completed"


@pytest.mark.parametrize("correction", [
    {"task_outcome": "failed"}, {"task_outcome": "completed", "text": "改成伪造的答案"},
    {"task_outcome": "completed", "tool_call": {"name": "private_web_search", "arguments": {}}},
    {}, {"task_outcome": ["completed"]},
])
def test_unresolved_conflict_or_action_in_correction_never_invents_success(correction):
    runtime, host, _, reader = pipeline(outcome="failed")
    runtime.status_correction = correction
    reply = host.handle_turn(turn())
    assert reply.agent_status == "failed" and "冲突" in reply.final_text
    assert len(runtime.calls) == 4 and len(reader.calls) == 1


@pytest.mark.parametrize("review", [
    {"verdict": "unsupported", "claims": []},
    {"verdict": "supported", "claims": [{"source_id": "invented", "quote": "烟台", "answer_fragment": "烟台"}]},
    {"verdict": "supported", "claims": []}, {},
    {"verdict": ["supported"], "claims": {}},
])
def test_unbound_off_topic_or_failed_semantic_review_is_not_delivered_as_fact(review):
    runtime, host, _, _ = pipeline(answer="上海晴，30°C。来源：https://weather.example/source")
    # A protocol correction gets the same failing fixture, never the fake
    # runtime's default semantic approval after its response list is empty.
    invalid_protocol = not isinstance(review.get("verdict"), str)
    runtime.answer_reviews = [review, review] if invalid_protocol else [review]
    reply = host.handle_turn(turn())
    assert reply.agent_status == "partial" and "上海晴" not in reply.final_text
    assert "证据核对" in reply.final_text and len(runtime.calls) == (5 if invalid_protocol else 4)
    if invalid_protocol:
        first, second = runtime.calls[2:4]
        assert first["user_prompt"] == second["user_prompt"]
        assert all(row["native_tools"] is None for row in (first, second))
        assert "仅纠正复核协议" in "\n".join(second["system_extra_blocks"])


def test_protocol_correction_then_semantic_rejection_keeps_off_topic_answer_unverified():
    runtime, host, _, reader = pipeline(answer="上海晴，30°C。来源：https://weather.example/source")
    runtime.answer_reviews = [{}, {"verdict": "unsupported", "claims": []}]
    reply = host.handle_turn(turn())
    assert reply.agent_status == "partial" and "上海晴" not in reply.final_text
    assert "证据核对" in reply.final_text and len(runtime.calls) == 5 and len(reader.calls) == 1
    assert all(row["native_tools"] is None for row in runtime.calls[2:])
    assert runtime.calls[2]["user_prompt"] == runtime.calls[3]["user_prompt"]


def test_missing_field_cannot_count_as_complete_even_with_a_successful_body():
    runtime, host, _, _ = pipeline()
    def missing(payload):
        source = payload["read_sources"][0]
        return {"verdict": "partial", "claims": [{"source_id": source["id"],
            "quote": "多云18°C", "answer_fragment": "18°C"}], "missing_fields": ["降水"],
            "coverage": [{"field": "原问题", "supported": False, "claim_indices": [0]}]}
    runtime.answer_reviews = [missing]
    reply = host.handle_turn(turn("烟台今天的温度和降水分别是多少？"))
    assert reply.agent_status == "partial" and "证据核对" in reply.final_text
    assert len(runtime.calls) == 4


def test_bounded_answer_correction_rechecks_body_not_just_failed_label():
    runtime, host, _, reader = pipeline(outcome="failed", answer="烟台多云，18°C。")
    runtime.answer_reviews = [{"verdict": "supported", "claims": []}]
    runtime.answer_correction = final()
    reply = host.handle_turn(turn())
    assert reply.agent_status == "completed" and "https://weather.example/source" in reply.final_text
    assert len(runtime.calls) == 5 and len(reader.calls) == 1
    assert all(row["native_tools"] is None for row in runtime.calls[-3:])


def test_second_evidence_review_failure_cannot_retry_or_upgrade():
    runtime, host, _, reader = pipeline(outcome="failed", answer="烟台多云，18°C。")
    runtime.answer_reviews = [{"verdict": "supported", "claims": []}, {}]
    runtime.answer_correction = final()
    reply = host.handle_turn(turn())
    assert reply.agent_status == "partial" and "证据核对" in reply.final_text
    assert len(runtime.calls) == 5 and len(reader.calls) == 1


def test_complete_goal_and_failed_delivery_are_independent():
    runtime, host, _, reader = pipeline()
    sent = []
    host.sender = lambda m: sent.append(m) or DeliveryResult(False, "rejected", error="fixture_rejected")
    reply = host.handle_turn(turn())
    assert reply.agent_status == "completed" and not reply.sent and reply.delivery_status == "rejected"
    assert len(reader.calls) == 1 and sent


def test_second_search_requires_offtopic_evidence_and_keeps_original_scope_in_prompt():
    runtime = Runtime(call(), call("private_web_search", {"query": "烟台 2026-10-04 天气", "provider": "duckduckgo"}))
    host, search, _ = bridge(runtime)
    reply = host.handle_turn(turn())
    assert len(search.calls) == 1 and reply.reason == "public_query_search_not_justified"
    assert len(runtime.calls) == 3 and runtime.calls[-1]["native_tools"] is None
    assert json.loads(runtime.calls[-1]["user_prompt"])["original_request"] == "今天烟台天气怎么样？"


def test_unavailable_search_allows_one_alternate_not_unlimited_queries():
    result = {"tool_type": "private_web_search", "followup_context": "blocked",
              "state_updates": {"results": [], "private_tool_error": True, "error_code": "search_provider_blocked"}}
    search = SourceTool("private_web_search", result)
    runtime = Runtime(call(), call("private_web_search", {"query": "烟台 天气", "provider": "duckduckgo"}),
                      call("private_web_search", {"query": "烟台 当前天气"}))
    host, _, _ = bridge(runtime, search=search)
    reply = host.handle_turn(turn())
    assert len(search.calls) == 2 and reply.agent_status == "failed"
    assert reply.reason == "public_query_search_not_justified"
