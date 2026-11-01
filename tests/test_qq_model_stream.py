from types import SimpleNamespace
import pytest
from shinku.qq.model_stream import call_reply_model
from shinku.qq.commands import QQCommandState
from test_qq_public_query_fast_track import Runtime, bridge, final, turn, close_coordinators, malformed_native

class StreamRuntime(Runtime):
    def stream_chat_json(self, **kwargs):
        kwargs.pop("early_tool_call_validator")
        self.stream_calls = getattr(self, "stream_calls", 0) + 1
        yield {"type": "speech_chunk", "text": "UNVERIFIED"}
        result = self.call_chat_json(**kwargs)
        return SimpleNamespace(parsed=result, error="", rescue_succeeded=False, first_chunk_ms=3, transport_chunks=2)

def test_stream_toggle_is_live_and_preserves_full_options():
    state = QQCommandState(stream_enabled=True)
    runtime = StreamRuntime(final("你好", "completed"), final("关了", "completed"))
    host, _, _ = bridge(runtime)
    host.command_state = state
    model = host._model_call_dependencies()
    args = dict(system_prompt="sys", user_prompt="user", fallback={"kind":"final", "text":"failed"}, native_tools=[{"function":{"name":"read"}}])
    assert model.call_json(**args)["text"] == "你好"
    state.set_stream_enabled(False)
    assert model.call_json(**args)["text"] == "关了"
    assert runtime.calls == [args, args]

def test_host_stream_never_publishes_draft_and_sends_final_once():
    runtime = StreamRuntime(final("你好。", "completed"))
    host, _, _ = bridge(runtime)
    host.command_state.set_stream_enabled(True)
    sent = []
    from shinku.qq.delivery import DeliveryResult
    host.sender = lambda msg: (sent.append(msg), DeliveryResult(True, "sent", ("receipt",)))[1]
    reply = host.handle_turn(turn("你好"))
    assert reply.sent and len(sent) == 1
    assert "UNVERIFIED" not in str(sent)
    assert len(runtime.calls) == 1

@pytest.mark.parametrize("broken", [True, False])
def test_stream_error_does_not_publish_partial_and_generator_closes(broken):
    closed=[]
    class Fake:
        def stream_chat_json(self, **options):
            try:
                yield {"text":"draft"}
                return SimpleNamespace(parsed={"kind":"final","text":"draft"}, error="broken" if broken else "", rescue_succeeded=False)
            finally:closed.append(True)
        def call_chat_json(self, **options):raise AssertionError("unbounded fallback")
    model=SimpleNamespace(runtime=Fake(),stream_enabled=lambda:True,effects=SimpleNamespace(trace=lambda *a,**k:None))
    fallback={"kind":"final","text":"failed","task_outcome":"failed"}
    result=call_reply_model(model,fallback=fallback)
    assert result == (fallback if broken else {"kind":"final","text":"draft"})
    assert closed == [True]

def test_streamed_public_query_preserves_tool_route_and_evidence_gate():
    selected={"kind":"tool","tool_call":{"name":"private_web_search","arguments":{"query":"拉格朗日点"}},"task_link":{"mode":"none","track":"public_query"}}
    runtime=StreamRuntime(malformed_native(),selected,final("未取得正文，无法完成查证。","failed"))
    host, search, _=bridge(runtime)
    host.command_state.set_stream_enabled(True)
    result=host.handle_turn(turn("什么是拉格朗日点？"))
    assert len(search.calls)==1 and not result.background_task_id
    assert result.reason=="lightweight_public_query_fast_track"
    assert len(runtime.calls)==3 and runtime.stream_calls==3



def test_deleted_route_review_is_not_reachable_from_bridge():
    runtime=StreamRuntime(final("你好。","completed"));host,_,_=bridge(runtime)
    assert not hasattr(host,"_review_direct_reply_route")
    reply=host.handle_turn(turn("你好"))
    assert reply.agent_status=="completed" and len(runtime.calls)==1
