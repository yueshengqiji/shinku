"""Actual SDK/parser/QQ reply gates over isolated HTTP and task storage."""
import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI
import pytest

from shinku.agent_runtime.demo import demo_service
from shinku.llm.runtime import ProviderToolProfile
from shinku.llm.response_protocol import observe_response
from test_expression_runtime import FixtureRuntime
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import boundary_bridge
from reply_review_fixture import wire_dialogue_review_result


PLAIN = "嗯，先歇一会儿吧。今天最累的是哪件事？"


class ProtocolRuntime(FixtureRuntime):
    def _tool_profile(self, bundle):
        return ProviderToolProfile(supports_native_tools=True, force_response_json=False)

    def _note_parse_failure(self, *args, **kwargs):
        pass

    def _note_truncated(self, *args, **kwargs):
        pass


@contextmanager
def actual_sdk(answers, *, streaming=False):
    calls = []
    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        review = wire_dialogue_review_result(wire)
        value = json.dumps(review) if review is not None else answers.pop(0)
        if isinstance(value, Exception):
            raise value
        content, finish = value if isinstance(value, tuple) else (value, "stop")
        if wire.get("stream"):
            chunks = [dict(id="offline", object="chat.completion.chunk", created=0, model="offline",
                choices=[dict(index=0, delta={"content": content}, finish_reason=None)]),
                dict(id="offline", object="chat.completion.chunk", created=0, model="offline",
                choices=[dict(index=0, delta={}, finish_reason=finish)])]
            data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=dict(id="offline", object="chat.completion", created=0,
            model="offline", choices=[dict(index=0, finish_reason=finish,
                message={"role": "assistant", "content": content})],
            usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
    with httpx.Client(transport=httpx.MockTransport(handle)) as client, \
            OpenAI(api_key="offline-fixture", base_url="https://example.invalid/v1",
                   max_retries=0, http_client=client) as sdk:
        runtime = ProtocolRuntime(sdk.chat.completions.create)
        from shinku.llm.policy import RuntimePolicy
        runtime.runtime_policy = RuntimePolicy(http_attempts=1, stream_attempts=1)
        yield runtime, calls


@pytest.mark.parametrize("route,completed", [("chat", True), ("defer", False), ("unverified", False)])
@pytest.mark.parametrize("streaming", [False, True])
def test_plain_candidate_requires_original_request_review(tmp_path, route, completed, streaming):
    service, tool, _ = demo_service(tmp_path)
    with actual_sdk([PLAIN, json.dumps({"route": route, "relation": "unrelated"})]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        bridge._stream_enabled = lambda: streaming
        try:
            reply = bridge(turn("我今天有点累"))
            assert reply.agent_status == "completed", reply
            assert len(calls) == 1 and tool.calls == 0
            assert service.task_store.candidates(requester="fixture", session="private:fixture") == []
            assert not any("独立的对话依据审查者" in str(c["messages"]) for c in calls)
            assert "我今天有点累" in calls[0]["messages"][-1]["content"]
            if completed:
                assert reply.final_text == PLAIN
        finally:
            service.coordinator.shutdown(wait=True)
            service.task_store.close()


@pytest.mark.parametrize("body", ["agent_task_submit(goal='write a file')", "已经把文件写好了。"])
def test_tool_or_success_prose_never_creates_execution(tmp_path, body):
    service, tool, _ = demo_service(tmp_path)
    with actual_sdk([body, json.dumps({"route": "defer"})]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn("请写入目标文件"))
            assert reply.agent_status != "completed"
            assert len(calls) == 1 and tool.calls == 0
            assert not service.task_store.candidates(requester="fixture", session="private:fixture")
        finally:
            service.coordinator.shutdown(wait=True)
            service.task_store.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("body,finish", [('{"kind":"final","text":"安全正文","task_outcome":"completed"}', "stop"),
    ('{"kind":"final","text":"漏掉尾部"', "length"), ('截断的自然正文', "length"),
    ('屏蔽的正文', "content_filter"), ('```json\n{"speech":"半截"', "stop")])
def test_forbidden_fields_partial_and_incomplete_output_are_not_prose(body, finish, streaming):
    with actual_sdk([(body, finish)]) as (runtime, calls):
        with observe_response(allow_plain_reply=True) as observation:
            options = dict(system_prompt="JSON rules", user_prompt="request", fallback={})
            if streaming:
                generator = runtime.stream_chat_json(**options)
                while True:
                    try: next(generator)
                    except StopIteration as done:
                        result = done.value.parsed
                        break
            else:
                result = runtime.call_chat_json(**options)
        assert len(calls) == 1  # No stream rescue request.
        if "task_outcome" in body:
            assert result["task_outcome"] == "completed"  # The host must see and reject it.
        else:
            assert result == {}, result
        assert not any(row["stage"] == "plain_reply_candidate" for row in observation.events)


def test_transport_error_and_plain_reply_diagnostics_are_call_local():
    def call(plain):
        answer = PLAIN if plain else httpx.ReadTimeout("private payload must not enter diagnostics")
        with actual_sdk([answer]) as (runtime, calls):
            with observe_response(allow_plain_reply=plain) as observation:
                result = runtime.call_chat_json(system_prompt="rules", user_prompt="request", fallback={})
            return result, observation.events, len(calls)
    with ThreadPoolExecutor(2) as pool:
        good, bad = list(pool.map(call, [True, False]))
    assert good == ({"kind": "final", "text": PLAIN}, [{"stage": "plain_reply_candidate"}], 1)
    assert bad == ({}, [{"stage": "transport_error"}], 1)
    assert "private" not in json.dumps(bad[1])
    with actual_sdk([PLAIN]) as (runtime, calls):
        assert runtime.call_chat_json(system_prompt="review", user_prompt="request", fallback={}) == {}


def test_task_prompt_has_one_format_contract(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    with actual_sdk([json.dumps({"kind": "final", "text": PLAIN})]) as (runtime, calls):
        bridge = boundary_bridge(service, tool, runtime)
        try:
            reply = bridge(turn("我今天有点累"))
            assert reply.agent_status == "completed"
            from shinku.qq.task_interaction import TASK_CHAT_GUIDANCE, TASK_RESPONSE_FORMAT
            system = "\n".join(row["content"] for row in calls[0]["messages"] if row["role"] == "system")
            assert system.count(TASK_CHAT_GUIDANCE) == system.count(TASK_RESPONSE_FORMAT) == 1
        finally:
            service.coordinator.shutdown(wait=True)
            service.task_store.close()
