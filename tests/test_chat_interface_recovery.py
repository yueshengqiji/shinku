"""错误工具调用后的纯文字补答；真实 SDK 解析与 QQ 链路使用隔离回执。"""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.llm.runtime_core import ProviderToolProfile
from shinku.qq.delivery import DeliveryResult
from test_contract_c2_5 import FakeResponse, _make_runtime
from test_qq_at_failure_regression import turn
from test_qq_tool_decision_boundary import boundary_bridge
from shinku.agent_runtime.demo import demo_service
from shinku.agent.budget import TaskBudget, bind_task_budget


PROMPT = "archive envoy 这个虚构职位怎样译得自然？"
ANSWER = "可以暂译为档案使者，具体还要看这个职位的职责。"


def tool_response(name, *, native=True):
    if not native:
        return FakeResponse(json.dumps({"kind": "tool", "tool_call": {
            "name": name, "arguments": {}}}))
    response = FakeResponse(None, finish_reason="tool_calls")
    response.choices[0].message.tool_calls = [SimpleNamespace(id="isolated-call",
        function=SimpleNamespace(name=name, arguments="{}"))]
    return response


def runtime_for(*responses):
    runtime = _make_runtime(scripts=[("response", response) for response in responses], dialogue_reviews=True)
    runtime._tool_profile = lambda bundle: ProviderToolProfile(supports_native_tools=True)
    return runtime


@pytest.fixture
def host(tmp_path):
    service, tool, execution = demo_service(tmp_path)
    runtimes, sent = [], []
    def create(responses, **options):
        runtime = runtime_for(*responses)
        runtimes.append(runtime)
        bridge = boundary_bridge(service, tool, runtime,
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("fixture-receipt",)),
            **options)
        return bridge, runtime
    yield create, service, tool, execution, sent
    service.coordinator.shutdown(wait=True)
    service.task_store.close()
    for runtime in runtimes:
        runtime.close()


def assert_no_execution(service, tool, execution):
    assert tool.calls == 0 and not tool.path.exists() and execution.calls == []
    assert service.task_store.candidates(requester="fixture", session="private:fixture") == []
    assert service.task_store.candidates(requester="fixture", session="group:42") == []
    assert not service.coordinator.list()


@pytest.mark.parametrize("kind", ["private", "group"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_rejected_call_can_only_recover_a_text_answer(host, kind, native, registered):
    create, service, tool, execution, sent = host
    name = tool.tool_type if registered else "fixture_unknown_translate"
    bridge, runtime = create([tool_response(name, native=native),
        FakeResponse(json.dumps({"kind": "final", "text": ANSWER}, ensure_ascii=False))])
    result = bridge(turn(kind, text=PROMPT, sender="fixture"))
    assert (result.agent_status, result.reason, result.delivery_status) == (
        "completed", "direct_chat_reply", "sent")
    assert result.final_text == ANSWER and len(sent) == 1
    payloads = runtime.chat.client.payloads
    assert len(payloads) == 2 and payloads[0].get("tools")
    assert not payloads[1].get("tools") and payloads[1].get("tool_choice") in {None, "none"}
    assert_no_execution(service, tool, execution)
    stages = bridge.trace_store.snapshot()[0]["stages"]
    assert any(s["name"] == "chat_interface_validation" and s["reason"] == "chat_execution_call_rejected"
               for s in stages)
    assert any(s["name"] == "chat_interface_recovery" and s["kind"] == "reply_only" for s in stages)
    # 被拒响应只作本地诊断，不能把任意名称/参数回填到纠正提示词中。
    assert name not in json.dumps(payloads[1], ensure_ascii=False)


@pytest.mark.parametrize("second", ["unknown", "registered", "submit", "malformed", "authority"])
def test_second_response_cannot_call_tools_or_start_a_third_request(host, second):
    create, service, tool, execution, sent = host
    response = (tool_response("fixture_again") if second == "unknown" else
                tool_response(tool.tool_type) if second == "registered" else
                tool_response("agent_task_submit") if second == "submit" else
                FakeResponse("{}") if second == "malformed" else
                FakeResponse('{"kind":"final","text":"done","success":true}'))
    bridge, runtime = create([tool_response("fixture_unknown_translate"), response])
    result = bridge(turn("group", text=PROMPT, sender="fixture"))
    assert result.agent_status == "blocked" and result.reason != "direct_chat_reply"
    assert len(runtime.chat.client.payloads) == 2 and len(sent) == 1
    assert "任务接口" not in result.final_text
    assert_no_execution(service, tool, execution)


@pytest.mark.parametrize("text", ["已保存 result.txt。", "我已经联网搜索并查证完成了。", "我马上替你修改文件。"])
def test_text_recovery_keeps_claim_and_action_promise_gates(host, text):
    create, service, tool, execution, sent = host
    bridge, runtime = create([tool_response("fixture_unknown_translate"),
                             FakeResponse(json.dumps({"kind": "final", "text": text}, ensure_ascii=False))])
    result = bridge(turn("private", text="请查证这个词的用法并给出来源。", sender="fixture"))
    assert result.agent_status in {"blocked", "failed"}
    assert result.final_text != text and len(runtime.chat.client.payloads) == 2
    assert_no_execution(service, tool, execution)


def test_repaired_wait_can_clarify_without_admitting_the_rejected_action(host):
    create, service, tool, execution, sent = host
    bridge, runtime = create([tool_response(tool.tool_type),
        FakeResponse('{"kind":"wait","text":"还没有执行操作，请补充具体目标。"}')])
    result = bridge(turn("private", text="帮我保存 result.txt。", sender="fixture"))
    assert result.agent_status == "waiting_user" and len(runtime.chat.client.payloads) == 2
    assert_no_execution(service, tool, execution)


def test_format_correction_and_reply_recovery_share_the_same_two_call_limit(host):
    create, service, tool, execution, sent = host
    bridge, runtime = create([FakeResponse("{}"), tool_response("fixture_unknown_translate")])
    result = bridge(turn("private", text=PROMPT, sender="fixture"))
    assert result.agent_status == "blocked" and len(runtime.chat.client.payloads) == 2
    assert_no_execution(service, tool, execution)


@pytest.mark.parametrize("response", [
    '{"kind":"final","text":"done","success":true}',
    '{"kind":"tool","tool_call":{"name":"agent_task_submit","arguments":{"authorization":"approved"}}}',
    '{"tool_call":{"type":"demo_file","path":"ignored"}}',
])
def test_authority_and_argument_rejections_do_not_enter_text_recovery(host, response):
    create, service, tool, execution, sent = host
    bridge, runtime = create([FakeResponse(response)])
    result = bridge(turn("private", text=PROMPT, sender="fixture"))
    assert result.agent_status == "blocked" and len(runtime.chat.client.payloads) == 1
    assert_no_execution(service, tool, execution)


@pytest.mark.parametrize("change", ["source", "stale"])
def test_invalidated_context_stops_before_the_correction_request(host, change):
    create, service, tool, execution, sent = host
    bridge, runtime = create([tool_response("fixture_unknown_translate")])
    changed = [False]
    item = replace(turn("group", text=PROMPT, sender="fixture"), attention_mode="autonomous",
        autonomous_guard=lambda phase: "autonomous_turn_superseded" if changed[0] else "")
    context = {"qq_turn": item, "audience_context": {"is_master": True, "role": "master"}}
    original = bridge._trace
    def trace(trace_id, name, **details):
        original(trace_id, name, **details)
        if name == "chat_interface_recovery" and details.get("status") == "started":
            if change == "source":
                context["qq_dialogue_sources_changed"] = True
            else:
                changed[0] = True
    bridge._trace = trace
    decision, _ = bridge._run_chat_decision(task_id="isolated", context=context,
        initial_transcript=[{"role": "user", "content": PROMPT}], trace_id="")
    assert len(runtime.chat.client.payloads) == 1
    if change == "source":
        assert decision.outcome == "failed"
    else:
        assert context.get("qq_autonomous_stale_reason") == "autonomous_turn_superseded"
    assert_no_execution(service, tool, execution)


def test_correction_cannot_reset_or_exceed_the_host_request_budget(host):
    create, service, tool, execution, sent = host
    bridge, runtime = create([tool_response("fixture_unknown_translate")])
    budget = TaskBudget(max_requests=1)
    item = turn("private", text=PROMPT, sender="fixture")
    context = {"qq_turn": item, "audience_context": {"is_master": True, "role": "master"}}
    with bind_task_budget(budget):
        decision, _ = bridge._run_chat_decision(task_id="isolated", context=context,
            initial_transcript=[{"role": "user", "content": PROMPT}], trace_id="")
    assert len(runtime.chat.client.payloads) == budget.requests == 1
    assert decision.reason == "chat_model_budget_exhausted"
    assert_no_execution(service, tool, execution)


def test_failed_ambient_recovery_does_not_publish_a_local_error_template(host):
    from test_autonomous_reply_failure import turn as ambient_turn
    create, service, tool, execution, sent = host
    bridge, runtime = create([tool_response("fixture_unknown_translate"), tool_response("agent_task_submit")])
    item = ambient_turn()
    item = replace(item, messages=(replace(item.messages[0], sender_id="fixture"),))
    result = bridge(item)
    # 自主出口沿用既有失败终态；内部 blocked 草稿不能冒充有效模型发言。
    assert (result.agent_status, result.reason, result.delivery_status) == (
        "failed", "chat_generation_failed", "suppressed")
    assert not result.sent and not result.final_text and not sent
    assert len(runtime.chat.client.payloads) == 2
    assert_no_execution(service, tool, execution)
