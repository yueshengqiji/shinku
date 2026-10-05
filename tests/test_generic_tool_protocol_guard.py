from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.planner import LLMPlanner
from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD, invocation_to_legacy_tool_call, legacy_tool_call_to_invocation
from shinku.tools.execution import execute_invocation
from test_contract_c2_5 import _make_runtime, FakeResponse, FakeStreamChunk


class ZeroArgumentTool:
    def __init__(self):
        self.normalized, self.executed = 0, 0
    def normalize_call(self, call):
        self.normalized += 1
        return {}
    def execute(self, **kwargs):
        self.executed += 1
        return {"followup_context": "actual result"}


@pytest.mark.parametrize("arguments", ["{not-json", "[]", "null", "", None, [], True])
def test_shared_planner_and_executor_do_not_execute_malformed_json_as_zero_args(arguments):
    tool = ZeroArgumentTool()
    decision = LLMPlanner._parse({"tool_calls": [{"function": {"name": "zero", "arguments": arguments}}]})
    assert decision and decision.invocation.protocol_error
    _, envelope = execute_invocation(decision.invocation, handlers={"zero": tool})
    assert envelope.status != "ok" and envelope.data["code"] == "tool_arguments_malformed"
    assert tool.normalized == tool.executed == 0
    assert legacy_tool_call_to_invocation(invocation_to_legacy_tool_call(decision.invocation)) is None


@pytest.mark.parametrize("arguments", ["{not-json", "[]", "null", "", None])
def test_sdk_nonstream_malformed_arguments_are_not_converted_to_an_executable_call(arguments):
    response = FakeResponse("")
    response.choices[0].message.tool_calls = [SimpleNamespace(id="call", function=SimpleNamespace(name="zero", arguments=arguments))]
    runtime = _make_runtime(scripts=[("response", response)])
    try:
        value = runtime.call_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={})
        marker = value[NATIVE_TOOL_CALL_FIELD]
        assert marker["_malformed"] is True and not marker["tool_calls"]
        assert "type" not in marker and "arguments" not in marker and LLMPlanner._parse(value) is None
    finally:
        runtime.close()


def exhaust(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def test_sdk_stream_truncated_argument_object_is_rejected_without_fake_zero_args():
    chunk = FakeStreamChunk("")
    chunk.choices[0].delta.tool_calls = [SimpleNamespace(index=0, id="call", function=SimpleNamespace(name="zero", arguments="{bad"))]
    runtime = _make_runtime(scripts=[("stream", [chunk])])
    try:
        result = exhaust(runtime.stream_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={}))
        assert result.parsed[NATIVE_TOOL_CALL_FIELD]["_malformed"] is True
        assert not result.rescue_attempted and LLMPlanner._parse(result.parsed) is None
    finally:
        runtime.close()


def test_valid_empty_object_remains_valid_for_real_zero_argument_tools():
    tool = ZeroArgumentTool()
    decision = LLMPlanner._parse({"tool_calls": [{"function": {"name": "zero", "arguments": "{}"}}]})
    _, envelope = execute_invocation(decision.invocation, handlers={"zero": tool})
    assert envelope.status == "ok" and tool.executed == 1


def test_stream_http_reconnect_and_rescue_share_flat_attempt_limit_and_request_meter():
    runtime = _make_runtime(scripts=[("raise", TimeoutError("timeout"))] * 10)
    runtime.runtime_policy = replace(runtime.runtime_policy, http_attempts=3, stream_attempts=3,
        retry_backoff_seconds=0, stream_retry_delay_seconds=0)
    budget = TaskBudget(max_requests=20)
    try:
        with bind_task_budget(budget):
            result = exhaust(runtime.stream_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={"kind": "final", "text": "failed"}))
        assert len(runtime.chat.client.payloads) == 4 and budget.requests == 4
        assert result.error and not result.rescue_attempted
    finally:
        runtime.close()


def test_cache_fallback_consumes_same_nonstream_transport_attempt_limit():
    runtime = _make_runtime(scripts=[("raise", TypeError("unexpected prompt_cache_key")),
        ("raise", TimeoutError("timeout")), ("response", FakeResponse('{"speech":"should not run"}'))])
    budget = TaskBudget(max_requests=20)
    try:
        with bind_task_budget(budget), pytest.raises(TimeoutError):
            runtime._dispatch_completion(bundle=runtime.chat, payload={"messages": [], "prompt_cache_key": "fixture"})
        assert len(runtime.chat.client.payloads) == 2 and budget.requests == 2
    finally:
        runtime.close()
