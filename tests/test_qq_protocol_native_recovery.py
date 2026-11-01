"""Native protocol failures switch to real schemas, preserving intent and authority."""
import copy
from types import SimpleNamespace
import json
import pytest
from test_contract_c2_5b import _runtime
from test_qq_public_query_fast_track import Runtime, bridge, malformed_native, turn
from shinku.qq.tool_selection import native_repair_tool_specs, resolve_native_repair_intent
from shinku.agent.loop import AgentDecision
from shinku.tools.invocation import ToolInvocation

def test_provider_rejection_and_local_decode_failure_are_distinguishable_without_raw_data():
    runtime = _runtime()
    provider = SimpleNamespace(choices=[SimpleNamespace(finish_reason="MALFORMED_FUNCTION_CALL")])
    result = runtime._read_native_tool_call(provider)
    assert result["malformed_reason"] == "provider_malformed_function_call"
    for name, arguments, expected in [
        ("fixture", "PRIVATE_SECRET_INVALID_JSON", "invalid_argument_json"),
        ("PRIVATE SECRET INVALID NAME", "{}", "invalid_tool_name"),
    ]:
        parts = {0: {"name": name, "arguments_parts": [arguments]}}
        result = runtime._assemble_stream_tool_call(parts)
        assert result["malformed_reason"] == expected
        assert result["tool_calls"] == []
        assert "PRIVATE_SECRET" not in json.dumps(result)
        assert "PRIVATE SECRET" not in json.dumps(result)
    parts = {}
    runtime._collect_stream_tool_parts(provider, parts)
    assert runtime._assemble_stream_tool_call(parts)["malformed_reason"] == "provider_malformed_function_call"



@pytest.mark.parametrize("intent,mode", [("task", "new"), ("public_query", "none")])
def test_native_recovery_preserves_intent_and_strips_host_metadata(intent, mode):
    repaired={"kind":"tool","tool_call":{"name":"private_web_search","arguments":{
        "query":"拉格朗日点", "_shinku_request_kind":intent}}}
    runtime=Runtime(malformed_native(),repaired)
    host,search,reader=bridge(runtime)
    context={"qq_turn":turn("深入研究并写四份报告。" if mode=="new" else "查证拉格朗日点。"),
             "audience_context":{"is_master":True,"role":"master"}}
    try:
        decision,_=host._run_chat_decision(task_id="probe",context=context,initial_transcript=[])
        assert decision.kind=="tool" and decision.invocation.arguments=={"query":"拉格朗日点"}
        assert decision.task_link["mode"]==mode
        repair=runtime.calls[1]
        assert len(runtime.calls)==2 and repair["native_tool_choice"]=="auto"
        assert all(s["function"]["name"]!="__shinku_tool_dispatch__" for s in repair["native_tools"])
        assert repair["system_prompt"].startswith("你是任务工具决定恢复器")
        assert repair["history_turns"]==[] and repair["user_prompt"]==runtime.calls[0]["user_prompt"]
        assert not search.calls and not reader.calls and not host.task_coordinator.list()
    finally:host.task_coordinator.shutdown(wait=True)


def test_schema_copy_keeps_original_required_parameters_and_does_not_mutate_catalog():
    original=[{"type":"function","function":{"name":"fixture","parameters":{"type":"object",
        "properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False}}}]
    prior=copy.deepcopy(original);result=native_repair_tool_specs(original)
    assert original==prior
    params=result[0]["function"]["parameters"]
    assert params["required"]==["path","_shinku_request_kind"] and not params["additionalProperties"]


@pytest.mark.parametrize("name,args",[("unknown",{"_shinku_request_kind":"task"}),
    ("fixture",{}),("fixture",{"_shinku_request_kind":"invalid"}),
    ("fixture",{"_shinku_request_kind":"continue","_shinku_task_id":{}})])
def test_invalid_repair_is_failed_without_a_tool_decision(name,args):
    result=resolve_native_repair_intent(AgentDecision.tool(ToolInvocation(name,args)),{"fixture"})
    assert result.kind=="final" and result.outcome=="failed"


def test_continuation_retains_scoped_target_and_native_call_identity():
    decision=AgentDecision.tool(ToolInvocation("fixture",{"path":"C:/fixture","_shinku_request_kind":"continue",
        "_shinku_task_id":"task-one"},source="native_openai",id="call-one"))
    result=resolve_native_repair_intent(decision,{"fixture"})
    assert result.task_link=={"mode":"continue","task_id":"task-one"}
    assert result.invocation.arguments=={"path":"C:/fixture"} and result.invocation.id=="call-one"
    assert result.invocation.source=="native_openai"


def test_unknown_tool_cannot_execute_after_native_repair():
    runtime=Runtime(malformed_native(),{"kind":"tool","tool_call":{"name":"unknown","arguments":{
        "_shinku_request_kind":"task"}}})
    host,search,reader=bridge(runtime)
    try:
        reply=host.handle_turn(turn("执行多步骤任务"))
        assert reply.agent_status=="failed" and len(runtime.calls)==2
        assert not search.calls and not reader.calls and not host.task_coordinator.list()
    finally:host.task_coordinator.shutdown(wait=True)
