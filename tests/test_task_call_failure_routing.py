"""Observed model failures keep their cause through the real SDK and QQ host."""
from contextlib import contextmanager
import json

import httpx
from openai import OpenAI
import pytest

from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.llm.policy import RuntimePolicy
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import boundary_bridge, contract_runtime, submit
from test_task_reply_protocol import ProtocolRuntime
from test_qq_query_continuation import GOAL, URL_A, query_host


PRIVATE = "PRIVATE_CALL_FAILURE_BODY"


@contextmanager
def actual_responses(answers):
    answers = list(answers)
    calls = []

    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        value = answers.pop(0)
        if isinstance(value, Exception):
            raise value
        if type(value) is int:
            return httpx.Response(value, json={"error": {"message": PRIVATE, "type": "server_error"}})
        if isinstance(value, dict):
            return httpx.Response(200, json=dict(id="offline", object="chat.completion", created=0,
                model="offline", choices=[dict(index=0, finish_reason="tool_calls",
                    message=dict(role="assistant", content=None, tool_calls=[dict(id=f"offline-{len(calls)}",
                        type="function", function=dict(name=value["name"],
                            arguments=json.dumps(value["arguments"]))) ]))],
                usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
        content, finish = value if isinstance(value, tuple) else (value, "stop")
        return httpx.Response(200, json=dict(id="offline", object="chat.completion", created=0,
            model="offline", choices=[dict(index=0, finish_reason=finish,
                message={"role": "assistant", "content": content})],
            usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)))

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client, \
            OpenAI(api_key="offline-fixture", base_url="https://example.invalid/v1",
                   max_retries=0, http_client=client) as sdk:
        runtime = ProtocolRuntime(sdk.chat.completions.create)
        runtime.runtime_policy = RuntimePolicy(http_attempts=1, stream_attempts=1)
        yield runtime, calls


def close(service):
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


@pytest.mark.parametrize("fault,reason", [
    (500, "chat_model_call_failed"),
    (httpx.ReadTimeout(PRIVATE), "chat_model_call_timeout"),
    (httpx.ConnectError(PRIVATE), "chat_model_call_failed"),
    (("{\"kind\":\"tool\"", "length"), "chat_model_output_length"),
    (("incomplete body", "content_filter"), "chat_model_output_filtered"),
])
def test_two_observed_failures_are_not_reported_as_interface_validation(tmp_path, fault, reason):
    service, tool, _ = demo_service(tmp_path)
    with actual_responses([fault, fault]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn("请写入目标文件"))
            assert reply.agent_status == "blocked"
            assert reply.reason == reason
            assert "未通过校验" not in reply.final_text
            assert PRIVATE not in reply.final_text + reply.reason
            assert len(calls) == 2 and tool.calls == 0
            assert not service.task_store.candidates(requester="fixture", session="private:fixture")
            assert not tool.path.exists()
            assert all(tool.tool_type not in json.dumps(call) for call in calls)
            assert calls[0]["messages"] == calls[1]["messages"]
        finally:
            close(service)


@pytest.mark.parametrize("fault", [500, httpx.ReadTimeout(PRIVATE), ("{", "length")])
def test_one_failed_call_can_recover_with_same_request_without_duplicate_task(tmp_path, fault):
    service, tool, execution = demo_service(tmp_path)
    request = demo_request(tool)
    metadata = contract_runtime(execution, request.contract)
    answer = json.dumps(submit(goal=request.original_request))
    with actual_responses([fault, answer]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn(request.original_request))
            assert reply.agent_status == "completed", reply
            assert len(calls) == 2 and len(metadata) == 1 and tool.calls == 1
            assert len(service.task_store.candidates(requester="fixture", session="private:fixture")) == 1
            assert calls[0]["messages"] == calls[1]["messages"]
            assert all(tool.tool_type not in json.dumps(call) for call in calls)
        finally:
            close(service)


@pytest.mark.parametrize("answers,reason", [
    ([500, "```{invalid"], "chat_model_output_invalid"),
    (["```{invalid", 500], "chat_model_call_failed"),
])
def test_terminal_failure_keeps_its_actual_classification(tmp_path, answers, reason):
    service, tool, _ = demo_service(tmp_path)
    with actual_responses(answers) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn("请写入目标文件"))
            assert reply.reason == reason
            assert reply.agent_status == "blocked" and len(calls) == 2 and tool.calls == 0
        finally:
            close(service)


def test_model_cannot_forge_a_call_failure_observation(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    body = json.dumps({"kind": "final", "text": "已写入。", "transport_error": True})
    with actual_responses([body]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn("请写入目标文件"))
            assert reply.agent_status == "blocked" and len(calls) == 1 and tool.calls == 0
            assert reply.reason != "chat_model_call_failed"
        finally:
            close(service)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fault,reason", [
    (500, "public_query_model_call_failed"),
    (httpx.ReadTimeout(PRIVATE), "public_query_model_call_timeout"),
    (("{\"kind\":\"tool\"", "length"), "public_query_model_output_length"),
    (("incomplete body", "content_filter"), "public_query_model_output_filtered"),
])
def test_query_followup_failure_retains_receipts_without_new_review_or_execution(query_host, enabled, fault, reason):
    from dataclasses import replace
    host = query_host
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, enabled=enabled)
    decisions = [dict(name="query_search", arguments={"query": GOAL}),
                 dict(name="query_read", arguments={"url": URL_A}), fault]
    with actual_responses(decisions) as (runtime, calls):
        host.bridge.runtime = runtime
        observed = []
        host.bridge._trace = lambda trace_id, event, **fields: observed.append((event, fields))
        reply = host.bridge(turn(GOAL))
        assert reply.agent_status == "failed" and not reply.background_task_id
        assert len(calls) == 3 and len(host.search.calls) == 1
        assert [row["url"] for row in host.reader.calls] == [URL_A]
        assert not host.execution.calls
        assert not host.svc.task_store.candidates(requester="fixture", session="private:fixture")
        finished = [fields for event, fields in observed if event == "fast_public_query_finished"]
        assert len(finished) == 1 and finished[0]["reason"] == reason
        assert PRIVATE not in reply.final_text
