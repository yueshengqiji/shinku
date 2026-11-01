"""A tool follow-up must retain authorship evidence despite new group traffic."""
import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.self_state import SelfStateTool
from shinku.hosts.tool_host import ToolHost
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from test_context_projection import NOW, engine, put, req


POEM = "Once upon a time there was a capital pig. The piglets overtook the streets."


def turn(mid="current", sender="owner", group="g"):
    message = IncomingMessage.from_event({"post_type":"message", "message_type":"group",
        "group_id":group, "user_id":sender, "self_id":"bot", "message_id":mid, "time":NOW,
        "sender":{"nickname":"Owner" if sender == "owner" else "Friend"},
        "message":[{"type":"at","data":{"qq":"bot"}},
                   {"type":"text","data":{"text":"资本主义是猪"}}]})
    return QQTurn(conversation_id=group, conversation_type="group", primary_message_id=mid,
                  message_ids=(mid,), combined_text="资本主义是猪", messages=(message,))


def fixture(tmp_path):
    store, projector = engine(tmp_path, max_chars=5000)
    clock = [NOW]
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda:clock[0],
        policy=MemoryPolicy(master_user_ids=("owner",), projection=projector.policy))
    row = put(store, "poem", POEM, scope="group:g", sender="owner", sender_name="Owner", age=47)
    put(store, "poem:assistant", "那些小猪是你在比喻什么？", scope="group:g", sender="bot", role="assistant",
        age=30, requester_id="owner", delivery_status="sent", delivery_message_ids=["sent-reply"],
        delivery_content_complete=True)
    return store, service, clock, row


def flood(store, clock):
    for i in range(24):
        put(store, f"new-{i}", f"群友的插话{i}：这段童谣里的小猪真有意思。", scope="group:g",
            sender="friend", sender_name="Friend", age=-i-1)
    clock[0] = NOW + 40


def test_real_perception_route_retains_original_author_after_message_flood(tmp_path):
    store, service, clock, row = fixture(tmp_path)
    calls = []
    def model(**kwargs):
        calls.append(kwargs)
        return ({"kind":"tool", "tool_call":{"name":"read_self_state", "arguments":{"scope":"all"}}}
                if len(calls) == 1 else {"kind":"final", "text":"这是你刚才那段里的比喻。"})
    def observe():
        flood(store, clock)
        return {"model":"fixture", "persona":"真红"}
    runtime = SimpleNamespace(call_chat_json=model, chat_supports_native_tools=lambda:True)
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", memory_service=service,
        tool_host=ToolHost(ToolRegistry({"read_self_state":SelfStateTool(fresh_provider=observe)})),
        allowed_tool_names={"read_self_state"}, master_user_ids=("owner",), owner_gate_required=True,
        policy=ExecutionPolicy(risk_by_tool={"read_self_state":"low"}))
    result = bridge.handle_turn(turn())
    assert result.reason == "lightweight_perception_fast_track" and len(calls) == 2
    assert clock[0] == NOW + 40
    for call in calls:
        prompt = call["user_prompt"]
        assert POEM in prompt and '"sender_id":"owner"' in prompt
        assert row.record_id in prompt and "主人（Owner）" in prompt
        source = next(json.loads(line) for line in prompt.splitlines()
                      if line.startswith('{"record_id":') and row.record_id in line)
        assert source["text"] == POEM and source["attribution"]["sender_id"] == "owner"
    assert "群友的插话" not in calls[-1]["user_prompt"]
    assert "当前实例的只读状态如下" in calls[-1]["user_prompt"]
    # A fresh turn still selects recent traffic; this is not global pinning.
    fresh = service.project_context(conversation_id="g", conversation_type="group",
        user_message="资本主义是猪", message_meta={"sender_id":"owner", "self_id":"bot", "message_id":"next"})
    assert POEM not in fresh["rendered"]


@pytest.mark.parametrize("change", ["forget", "edit", "expire"])
def test_retained_locators_revalidate_content_and_revocation(tmp_path, change):
    store, service, clock, row = fixture(tmp_path)
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}), system_prompt="fixture",
                           memory_service=service, master_user_ids=("owner",))
    state = AgentState(context={"qq_turn":turn()}, transcript=[])
    assert POEM in bridge._build_prompt(state).user_prompt
    assert POEM not in repr(state.context["_projection_continuity"])
    flood(store, clock)
    if change == "forget":
        store.forget(scope=row.scope, record_id=row.record_id, query="", now=clock[0])
    else:
        with store._connect() as conn:
            if change == "edit":
                conn.execute("UPDATE memory_events SET text=? WHERE event_id=?",
                             ("更正后的原文",row.record_id.split(":",1)[1]))
            else:
                conn.execute("UPDATE memory_events SET expires_at=? WHERE event_id=?",
                             (NOW+1,row.record_id.split(":",1)[1]))
    followup = bridge._build_prompt(AgentState(context=state.context, transcript=[])).user_prompt
    assert POEM not in followup
    if change == "edit":
        assert "更正后的原文" in followup


@pytest.mark.parametrize("next_turn", [turn(mid="next"), turn(sender="friend"), turn(group="other")])
def test_selection_does_not_leak_to_another_turn_requester_or_group(tmp_path, next_turn):
    store, service, clock, row = fixture(tmp_path)
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}), system_prompt="fixture",
                           memory_service=service, master_user_ids=("owner",))
    state = AgentState(context={"qq_turn":turn()}, transcript=[])
    assert POEM in bridge._build_prompt(state).user_prompt
    flood(store, clock)
    state.context["qq_turn"] = next_turn
    assert POEM not in bridge._build_prompt(state).user_prompt


def test_retained_handles_cannot_widen_group_access(tmp_path):
    store, service, clock, row = fixture(tmp_path)
    secret = put(store, "private", "私人秘密不得带入群聊", scope="private:master")
    view = service.project_context(conversation_id="g", conversation_type="group", user_message="资本主义是猪",
        message_meta={"sender_id":"owner", "self_id":"bot", "message_id":"current"},
        continuity_handles=((secret.record_id, secret.scope, ""),))
    assert "私人秘密" not in view["rendered"]
    assert view["diagnostics"]["continuity_requested"] == 0


def test_empty_selection_stays_empty_until_next_turn(tmp_path):
    store, projector = engine(tmp_path)
    first = projector.project(req(scope="group:g", allowed_scopes=("group:g",)))
    assert not first.records
    put(store, "late", "小明说这是另一个话题。", scope="group:g", sender="friend")
    followup = projector.project(req(scope="group:g", allowed_scopes=("group:g",), continuity_handles=()))
    assert not followup.records and "另一个话题" not in followup.rendered
    assert projector.project(req(scope="group:g", allowed_scopes=("group:g",))).records
