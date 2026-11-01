"""合成异常响应经过真实 SDK、QQ 校验和持久追踪；不连接供应商或发送 QQ。"""
from contextlib import contextmanager
from dataclasses import replace
import json

import httpx
from openai import OpenAI
import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.llm.policy import RuntimePolicy
from shinku.qq.delivery import DeliveryResult
from shinku.qq.trace import TurnTraceStore
from test_qq_independent_agent import turn
from test_qq_at_failure_regression import turn as contextual_turn
from test_qq_tool_decision_boundary import boundary_bridge, contract_runtime
from test_task_reply_protocol import ProtocolRuntime


PROMPT = "你刚才的解释让我安心了一些。"
ANSWER = "可以慢慢说，我在听。"
MALFORMED = '```json\n{"kind":"final"'


@contextmanager
def sdk_responses(answers):
    """同时支持真实 SSE、非流式响应及原生工具；请求列表是行为证据。"""
    answers, calls = list(answers), []

    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert answers, "unexpected third provider call"
        value = answers.pop(0)
        if isinstance(value, Exception):
            raise value
        body, finish = value if isinstance(value, tuple) else (value, "stop")
        if isinstance(body, dict):
            payload = {"tool_calls": [{"index": 0, "id": "fixture-call", "type": "function",
                "function": {"name": body["name"], "arguments": json.dumps(body["arguments"])}}]}
            finish = "tool_calls"
        else:
            payload = {"content": body}
        if wire.get("stream"):
            chunks = [{"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                "model": "fixture", "choices": [{"index": 0, "delta": payload, "finish_reason": None}]},
                {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                "model": "fixture", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "fixture", "choices": [{"index": 0, "message": {"role": "assistant", **payload},
                "finish_reason": finish}], "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                    "total_tokens": 15}})

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client, \
            OpenAI(api_key="offline-fixture", base_url="https://example.invalid/v1", max_retries=0,
                   http_client=client) as sdk:
        runtime = ProtocolRuntime(sdk.chat.completions.create)
        runtime.runtime_policy = RuntimePolicy(http_attempts=1, stream_attempts=1)
        yield runtime, calls


@contextmanager
def host(tmp_path, responses, *, streaming=True):
    service, tool, execution = demo_service(tmp_path)
    sent = []
    try:
        with sdk_responses(responses) as (runtime, calls):
            bridge = boundary_bridge(service, tool, runtime,
                sender=lambda msg: sent.append(msg) or DeliveryResult(True, "sent", ("fixture-receipt",)))
            bridge.command_state.set_stream_enabled(streaming)
            trace_path = tmp_path / "trace.json"
            bridge.trace_store = TurnTraceStore(path=trace_path)
            yield bridge, service, tool, execution, calls, sent, trace_path
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def no_execution(service, tool, execution):
    assert not tool.calls and not tool.path.exists() and not execution.calls
    assert not service.task_store.candidates(requester="fixture", session="private:fixture")


@pytest.mark.parametrize("bad,shape", [("", "empty"), (MALFORMED, "fenced")])
def test_stream_failure_changes_transport_once_and_keeps_request_and_permissions(tmp_path, bad, shape):
    with host(tmp_path, [bad, ANSWER]) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "completed" and reply.final_text == ANSWER
        assert len(calls) == 2 and calls[0].get("stream") and not calls[1].get("stream")
        assert calls[0]["tools"] == calls[1]["tools"]
        assert calls[0]["messages"][-1] == calls[1]["messages"][-1]
        assert len(sent) == 1
        no_execution(service, tool, execution)
        stages = TurnTraceStore(path=path).snapshot()[0]["stages"]
        diagnostics = [s for s in stages if s["name"] == "chat_response_protocol"]
        assert len(diagnostics) == 2
        assert diagnostics[0]["response_shape"] == {"transport": "stream", "body_shape": shape,
            "body_chars": len(bad), "native_call": "none"}
        assert diagnostics[0]["finish_reason"] == "stop"
        assert diagnostics[1]["response_shape"]["transport"] == "json"
        assert diagnostics[0]["stages"] == (["empty_response", "parse_invalid"] if not bad else ["parse_invalid"])
        assert PROMPT not in path.read_text() and ANSWER not in path.read_text()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("bad,reason", [("", "chat_model_output_empty"), (MALFORMED, "chat_model_output_invalid")])
def test_double_failure_has_specific_cause_and_never_calls_a_third_time(tmp_path, streaming, bad, reason):
    with host(tmp_path, [bad, bad], streaming=streaming) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "blocked" and reply.reason == reason
        assert "任务接口" not in reply.final_text and len(calls) == 2 and len(sent) == 1
        no_execution(service, tool, execution)


@pytest.mark.parametrize("streaming", [False, True])
def test_complete_final_text_newline_recovers_locally_without_another_call(tmp_path, streaming):
    answer = "先把问题讲清楚。\n再一起看下一步。"
    body = '{"kind":"final","text":"' + answer + '"}'
    with host(tmp_path, [body], streaming=streaming) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "completed" and reply.final_text == answer
        assert len(calls) == 1
        # 正文换行沿用现有分段策略：两段各送一次，不把分段误当成重复模型调用。
        assert len(sent) == 2
        assert "\n".join(segment.text for msg in sent for segment in msg.segments) == answer
        no_execution(service, tool, execution)


def test_normal_success_keeps_single_stream_call(tmp_path):
    with host(tmp_path, [ANSWER]) as (bridge, service, tool, execution, calls, sent, path):
        assert bridge(turn(PROMPT)).final_text == ANSWER
        assert len(calls) == len(sent) == 1 and calls[0]["stream"]
        no_execution(service, tool, execution)


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_retry_may_select_an_existing_authorized_task_once(tmp_path, streaming):
    service, tool, execution = demo_service(tmp_path)
    request = demo_request(tool)
    metadata = contract_runtime(execution, request.contract)
    try:
        with sdk_responses(["", {"name": "agent_task_submit", "arguments": {"goal": request.original_request}}]) as (runtime, calls):
            bridge = boundary_bridge(service, tool, runtime)
            bridge.command_state.set_stream_enabled(streaming)
            reply = bridge(turn(request.original_request))
            assert reply.agent_status == "completed", reply
            assert len(calls) == 2 and len(metadata) == 1 and tool.calls == 1
            assert len(service.task_store.candidates(requester="fixture", session="private:fixture")) == 1
            assert calls[0]["tools"] == calls[1]["tools"]
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("body", [
    '{"kind":"final","text":"未执行。","authorization":"approved"}',
    '{"kind":"final","text":"未执行。","task_outcome":"completed"}',
])
def test_recovered_complete_object_keeps_forbidden_fields_visible(tmp_path, body):
    body = body.replace("未执行。", "未执行。\n保持原样。")
    with host(tmp_path, [body, body]) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "blocked"
        no_execution(service, tool, execution)
        assert "未执行。\n保持原样。" != reply.final_text


@pytest.mark.parametrize("change", ["source", "stale", "budget"])
def test_host_change_or_budget_blocks_the_recovery_request(tmp_path, change):
    with host(tmp_path, [""]) as (bridge, service, tool, execution, calls, sent, path):
        state = {"changed": False}
        base = contextual_turn("group", text=PROMPT, sender="fixture") if change == "stale" else turn(PROMPT)
        item = replace(base, attention_mode="autonomous",
            autonomous_guard=lambda phase: "autonomous_turn_superseded" if state["changed"] else "")
        context = {"qq_turn": item, "audience_context": {"is_master": True, "role": "master"}}
        original = bridge._trace
        def trace(trace_id, name, **details):
            original(trace_id, name, **details)
            if name == "chat_interface_recovery" and details.get("status") == "started":
                if change == "source": context["qq_dialogue_sources_changed"] = True
                if change == "stale": state["changed"] = True
        bridge._trace = trace
        with bind_task_budget(TaskBudget(max_requests=1 if change == "budget" else 10)):
            decision, _ = bridge._run_chat_decision(task_id="fixture", context=context,
                initial_transcript=[{"role": "user", "content": PROMPT}], trace_id="")
        assert len(calls) == 1
        if change == "source": assert decision.outcome == "failed"
        if change == "stale": assert context["qq_autonomous_stale_reason"] == "autonomous_turn_superseded"
        if change == "budget": assert decision.reason == "chat_model_budget_exhausted"
        no_execution(service, tool, execution)


def test_response_diagnostic_persistence_uses_exact_event_and_field_allowlist(tmp_path):
    path = tmp_path / "trace.json"
    store = TurnTraceStore(path=path)
    trace = store.begin(turn(PROMPT))
    shape = {"transport": "stream", "body_shape": "empty", "body_chars": 0, "native_call": "none",
        "text": "SECRET_BODY", "private_field": "SECRET_VALUE"}
    store.stage(trace, "chat_response_protocol", stages=["empty_response", "parse_invalid", "SECRET_STAGE"],
        finish_reason="stop", response_shape=shape)
    store.stage(trace, "other_event", response_shape=shape, finish_reason="SECRET_REASON")
    first, second = TurnTraceStore(path=path).snapshot()[0]["stages"]
    assert first["response_shape"] == {k: v for k, v in shape.items() if k not in {"text", "private_field"}}
    assert first["stages"] == ["empty_response", "parse_invalid"]
    assert first["finish_reason"] == "stop"
    assert "response_shape" not in second and "finish_reason" not in second
    assert "SECRET" not in path.read_text()


@pytest.mark.parametrize("bad", [True, -1, 2**32, float("nan"), "SECRET_VALUE", {}, []])
def test_response_diagnostic_rejects_invalid_counts_and_labels(tmp_path, bad):
    path = tmp_path / "trace.json"
    store = TurnTraceStore(path=path)
    trace = store.begin(turn(PROMPT))
    store.stage(trace, "chat_response_protocol", stages=[bad], finish_reason=bad,
        response_shape={"transport": bad, "body_shape": bad, "native_call": bad, "body_chars": bad})
    event = TurnTraceStore(path=path).snapshot()[0]["stages"][0]
    assert event["response_shape"] == {} and event["stages"] == []
    assert "finish_reason" not in event


@pytest.mark.parametrize("streaming", [False, True])
def test_native_call_with_no_body_is_not_misclassified_as_empty(tmp_path, streaming):
    with host(tmp_path, [{"name": "fixture_forbidden", "arguments": {}}, ANSWER], streaming=streaming) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "completed" and len(calls) == 2
        stages = TurnTraceStore(path=path).snapshot()[0]["stages"]
        first = next(s for s in stages if s["name"] == "chat_response_protocol")
        assert first["response_shape"]["native_call"] == "present"
        assert "empty_response" not in first["stages"]
        assert not calls[1].get("tools")  # 被拒工具仍走原有纯文字恢复，不能执行。
        no_execution(service, tool, execution)


@pytest.mark.parametrize("fault,reason", [(('半截答复', 'length'), 'chat_model_output_length'),
    (('被过滤的正文', 'content_filter'), 'chat_model_output_filtered')])
def test_incomplete_output_never_recovers_or_publishes_partial_text(tmp_path, fault, reason):
    with host(tmp_path, [fault, fault]) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.reason == reason and reply.agent_status == "blocked" and len(calls) == 2
        assert fault[0] not in reply.final_text
        no_execution(service, tool, execution)


@pytest.mark.parametrize("second", [
    {"name": "fixture_forbidden", "arguments": {}},
    {"name": "agent_task_submit", "arguments": {"authorization": "approved"}},
    '{"kind":"final","text":"不应交付","authorization":"approved"}',
    '{"kind":"tool","tool_call":{"type":"demo_file","path":"ignored"}}',
])
def test_complete_response_recovery_cannot_bypass_the_existing_interface(tmp_path, second):
    with host(tmp_path, ["", second]) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn(PROMPT))
        assert reply.agent_status == "blocked" and len(calls) == 2
        assert reply.final_text != "不应交付" and len(sent) == 1
        no_execution(service, tool, execution)


@pytest.mark.parametrize("streaming", [False, True])
def test_complete_response_recovery_keeps_action_and_claim_checks(tmp_path, streaming):
    body = json.dumps({"kind": "final", "text": "我已经联网搜索并查证完成了。"}, ensure_ascii=False)
    with host(tmp_path, ["", body], streaming=streaming) as (bridge, service, tool, execution, calls, sent, path):
        reply = bridge(turn("请联网查证这句话并说明来源。"))
        assert reply.agent_status in {"blocked", "failed"} and len(calls) == 2
        assert reply.final_text != "我已经联网搜索并查证完成了。"
        no_execution(service, tool, execution)


def test_shape_metadata_remains_call_local_under_concurrent_sdk_responses():
    from concurrent.futures import ThreadPoolExecutor
    from shinku.llm.response_protocol import observe_response
    def call(body):
        with sdk_responses([body]) as (runtime, calls):
            with observe_response(allow_plain_reply=True, classify_invalid_output=True) as observed:
                runtime.call_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={})
            return observed.response_shape, observed.events
    with ThreadPoolExecutor(2) as pool:
        blank, full = list(pool.map(call, ["", ANSWER]))
    assert blank[0] == {"transport": "json", "body_shape": "empty", "body_chars": 0, "native_call": "none"}
    assert full[0] == {"transport": "json", "body_shape": "text", "body_chars": len(ANSWER), "native_call": "none"}
    assert blank[1] == [{"stage": "empty_response"}, {"stage": "parse_invalid"}]
    assert full[1] == [{"stage": "plain_reply_candidate"}]
