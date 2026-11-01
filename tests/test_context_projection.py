from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.projection.jobs import SingleFlight
from shinku.memory.projection.semantics import validate
from shinku.agent.loop import AgentState
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn

NOW=1_800_000_000


def req(query="继续聊", **kwargs):
    values=dict(scope="private:owner", allowed_scopes=("private:owner",), requester="owner",
                self_id="bot", query=query, now=NOW, message_id="current")
    values.update(kwargs)
    return ProjectionRequest(**values)


def put(store, mid, text, *, scope="private:owner", sender="owner", age=100, role="user", **metadata):
    source=f"turn:{scope.split(':',1)[-1]}:{mid}"
    store.record_event(conversation_id=scope.split(':',1)[-1], scope=scope, text=text,
        source_id=source, created_at=NOW-age,
        metadata={"role":role, "sender_id":sender, "message_id":mid.removesuffix(":assistant"),
                  "self_id":"bot", **metadata})
    return store.event_by_source_id(source_id=source,scope=scope)


def engine(tmp_path, **kwargs):
    store=MemoryStore(tmp_path/"m.db")
    return store, ProjectionEngine(store, ProjectionPolicy(enabled=True, **kwargs))


def test_multiple_actual_pairs_conditions_and_authority(tmp_path):
    store, projector=engine(tmp_path)
    for i in range(3):
        put(store,str(i),f"条件{i}：如果下雨就不去，我还没答应。",age=60-i*10)
        put(store,str(i)+":assistant",f"我提议第{i}种办法，还没有执行。",sender="bot", role="assistant",
            age=59-i*10, requester_id="owner", delivery_status="sent", delivery_message_ids=[f"r{i}"],
            delivery_content_complete=True)
    view=projector.project(req())
    assert len(view.records)==6
    assert all("还没" in r["text"] or "没有执行" in r["text"] for r in view.records)
    assert view.execution_authority is False
    assert view.diagnostics["semantic"]["calls"]==0
    assert len(view.rendered)<=projector.policy.max_chars
    assert "条件" not in json.dumps(view.diagnostics,ensure_ascii=False)


@pytest.mark.parametrize("metadata", [{}, {"delivery_status":"failed"},
    {"delivery_status":"sent", "delivery_message_ids":["r"], "requester_id":"intruder"}])
def test_unsent_or_wrong_recipient_assistant_excluded(tmp_path, metadata):
    store,projector=engine(tmp_path)
    put(store,"old:assistant","从未送达的计划",sender="bot",role="assistant",**metadata)
    assert not projector.project(req()).records


def test_partial_delivery_is_explicit(tmp_path):
    store,projector=engine(tmp_path)
    put(store,"old:assistant","实际发送片段",sender="bot",role="assistant",requester_id="owner",
        delivery_status="partial",delivery_message_ids=["r"],delivery_content_complete=False)
    row=projector.project(req()).records[0]
    assert row["attribution"]["delivery_status"]=="partial"
    assert row["attribution"]["delivery_content_complete"] is False


def test_identical_words_independent_group_authors_are_not_merged(tmp_path):
    store,projector=engine(tmp_path)
    put(store,"a","我想去",scope="group:g",sender="A")
    put(store,"b","我想去",scope="group:g",sender="B")
    view=projector.project(req(scope="group:g",allowed_scopes=("group:g",),requester="A"))
    assert {r["attribution"]["sender_id"] for r in view.records}=={"A","B"}


def test_single_referent_and_ambiguous_two_names(tmp_path):
    store,projector=engine(tmp_path)
    put(store,"a","小红说她打算去看展。")
    one=projector.project(req("她后来去了吗"))
    assert one.topic["last_subject"] and not one.topic["unresolved"]
    put(store,"b","小雪说她也在考虑。",age=90)
    two=projector.project(req("她后来去了吗"))
    assert not two.topic["last_subject"] and two.topic["unresolved"]
    assert "ambiguous_referent" in two.diagnostics["gap_reasons"]


def test_same_name_two_sources_keeps_independent_candidates(tmp_path):
    store,projector=engine(tmp_path)
    put(store,"a","小明说他在上海。")
    put(store,"b","小明说他在北京。",age=90)
    view=projector.project(req("那个小明后来呢"))
    assert len(view.entities)==2
    assert len({e["candidate_id"] for e in view.entities})==2
    assert view.topic["unresolved"]


def test_forget_restart_and_stale_seed_do_not_restore_sources(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,"a","小红说她不喝咖啡。")
    first=projector.project(req())
    version=first.records[0]["source_version"]
    store.forget(scope=row.scope,record_id=row.record_id,query="",now=NOW)
    restarted=ProjectionEngine(store,projector.policy)
    view=restarted.project(req(seed_handles=((row.record_id,row.scope,version),)))
    assert not view.records and "咖啡" not in view.rendered


def test_source_change_during_model_wait_rejects_derived_selection(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True)
    first=put(store,"a","小红说她不去。")
    put(store,"b","小雪说她也不去。")
    def model(payload):
        with store._connect() as conn:
            conn.execute("UPDATE memory_events SET text='来源已更正' WHERE event_id=?",(first.record_id.split(':')[1],))
        return {"uncertain":False,"missing":[],"selected_entity":payload["entities"][0]["candidate_id"],
                "evidence":[{"record_id":payload["records"][0]["record_id"],"quote":payload["records"][0]["text"]}],
                "relevant_ids":[],"excluded_ids":[],"read_ids":[]}
    projector.semantic.model=model
    view=projector.project(req("做建筑设计的她后来呢"))
    assert "judge" not in view.topic
    assert "source_changed_during_projection" in view.diagnostics["gap_reasons"]
    assert first.record_id not in [r["record_id"] for r in view.records]


def test_judge_cache_question_key_and_model_failure_fallback(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True)
    put(store,"a","小红说她不去。")
    put(store,"b","小雪说她也不去。")
    calls=[]
    def model(payload):
        calls.append(payload)
        return {"uncertain":True,"missing":["无法区分"],"selected_entity":"",
                "evidence":[{"record_id":payload["records"][0]["record_id"],"quote":payload["records"][0]["text"]}]}
    projector.semantic.model=model
    first=projector.project(req("做建筑设计的她后来呢"))
    second=projector.project(req("做建筑设计的她后来呢"))
    projector.project(req("做建筑设计的她后来去哪里了"))
    assert len(calls)==2
    assert first.diagnostics["semantic"]["calls"]==1 and second.diagnostics["semantic"]["status"]=="cache_hit"
    assert second.topic["unresolved"]
    projector.semantic.model=lambda p: {"uncertain":False,"evidence":[{"record_id":"foreign","quote":"虚构"}]}
    failed=projector.project(req("做建筑设计的她在哪里"))
    assert failed.records and "judge" not in failed.topic
    assert failed.diagnostics["semantic"]["status"]=="failed"


def test_slow_path_daily_combined_cap(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True,semantic_daily_calls=1)
    put(store,"a","小红说她不去。")
    put(store,"b","小雪说她也不去。")
    projector.semantic.model=lambda p: {}
    projector.project(req("做建筑设计的她后来呢"))
    second=projector.project(req("做建筑设计的她去哪了"))
    assert second.diagnostics["semantic"]["status"]=="call_budget"


def test_long_source_does_not_silently_drop_trailing_negation(tmp_path):
    store,projector=engine(tmp_path,record_chars=80)
    put(store,"a","我给你介绍这段计划。"*20+"但是最终我没有同意，不要执行。")
    view=projector.project(req())
    assert view.records[0]["representation"]=="omitted_original"
    assert not view.records[0]["text"]
    assert "record_budget" in view.diagnostics["gap_reasons"]


def test_singleflight_coalesces_and_never_queues_after_timeout():
    flight=SingleFlight()
    release=threading.Event()
    started=threading.Event()
    count=[]
    def work():
        count.append(1); started.set(); release.wait(2); return {"ok":True}
    answers=[]
    t=threading.Thread(target=lambda:answers.append(flight.run("a",work,1)))
    t.start(); assert started.wait(1)
    value,diag=flight.run("b",work,.01)
    assert value is None and diag["status"]=="busy"
    value,diag=flight.run("a",work,.01)
    assert diag["status"]=="timeout"
    release.set();t.join(2)
    assert len(count)==1 and answers[0][0] is None
    assert not flight.cache


def test_distillation_rejects_summary_as_input_and_unbound_quotes():
    payload={"records":[{"record_id":"event:a","text":"我没有同意。"}],"entities":[]}
    with pytest.raises(ValueError):
        validate("distill",{"uncertain":False,"excerpts":[{"record_id":"event:a","quote":"我同意。"}]},payload)


def test_service_scope_recomputed_even_with_injected_foreign_handles(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    hidden=put(store,"secret","私人秘密",scope="private:master")
    service=MemoryService(store=store,router=MemoryRouter(),clock=lambda:NOW,
        policy=MemoryPolicy(master_user_ids=("owner",),projection=ProjectionPolicy(enabled=True)))
    view=service.project_context(conversation_id="g",conversation_type="group",user_message="之前的秘密",
        message_meta={"sender_id":"intruder","self_id":"bot","message_id":"current"},
        memory_context={"decision":{"history_mode":"retrieve"},"records":[hidden.as_dict()]})
    assert view and "私人秘密" not in view["rendered"]


def test_prompt_integration_render_once_and_revalidate_each_build(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    row=put(store,"old","只在周末喝热茶，工作日不喝。")
    service=MemoryService(store=store,router=MemoryRouter(),clock=lambda:NOW,
        policy=MemoryPolicy(projection=ProjectionPolicy(enabled=True)))
    message=IncomingMessage.from_event({"post_type":"message","message_type":"private","user_id":"owner",
        "self_id":"bot","message_id":"current","message":[{"type":"text","data":{"text":"然后呢"}}]})
    turn=QQTurn(conversation_id="owner",conversation_type="private",primary_message_id="current",
        message_ids=("current",),combined_text="然后呢",messages=(message,))
    bridge=QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}),system_prompt="test persona",memory_service=service,sender=None)
    state=AgentState(transcript=[],context={"qq_turn":turn,"memory_context":{"rendered":row.text,
        "recent_dialogue":row.text},"chat_flow":{"previous_exchange":row.text}})
    first=bridge._build_prompt(state)
    assert first.user_prompt.count(row.text)==1
    assert row.text not in "\n".join(first.system_extra_blocks)
    store.forget(scope=row.scope,record_id=row.record_id,query="",now=NOW)
    assert row.text not in bridge._build_prompt(state).user_prompt


def test_default_disabled_creates_no_projection_tables(tmp_path):
    service=MemoryService(store=MemoryStore(tmp_path/"m.db"),router=MemoryRouter())
    assert service.projection is None
    with service.store._connect() as conn:
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'memory_projection_%'").fetchall()


def test_environment_budget_validation():
    assert MemoryPolicy.from_environment({"SHINKU_CONTEXT_PROJECTION_ENABLED":"true"}).projection.enabled
    with pytest.raises(ValueError):
        ProjectionPolicy.from_environment({"SHINKU_CONTEXT_PROJECTION_CANDIDATE_LIMIT":"9999"})


def test_group_projection_preserves_quotes_targets_and_current_request(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    quoted='小明说“随你便”是在开玩笑，不是同意执行。'
    put(store,"quoted",quoted,scope="group:g",sender="A",at_targets=["B"],reply_to_message_id="prior")
    put(store,"secret","不应进入群聊的私聊内容",scope="private:owner")
    service=MemoryService(store=store,router=MemoryRouter(),clock=lambda:NOW,
        policy=MemoryPolicy(projection=ProjectionPolicy(enabled=True)))
    message=IncomingMessage.from_event({"post_type":"message","message_type":"group","group_id":"g",
        "user_id":"B","self_id":"bot","message_id":"current","message":[
            {"type":"at","data":{"qq":"A"}},{"type":"reply","data":{"id":"quoted"}},
            {"type":"text","data":{"text":"你刚才那句是什么意思？"}}]})
    turn=QQTurn(conversation_id="g",conversation_type="group",primary_message_id="current",
        message_ids=("current",),combined_text="你刚才那句是什么意思？",messages=(message,))
    bridge=QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}),system_prompt="test persona",memory_service=service,sender=None)
    current={"role":"user","content":"B → @A：你刚才那句是什么意思？"}
    state=AgentState(transcript=[current],context={"qq_turn":turn,"group_context":"旧的重复群聊快照",
        "chat_flow":{"reason":"different_visual_source","previous_exchange":"过时的图片猜测"}})
    prompt=bridge._build_prompt(state)
    assert prompt.user_prompt.count(quoted)==1
    assert '"at_targets":["B"]' in prompt.user_prompt
    assert '"reply_to_message_id":"prior"' in prompt.user_prompt
    assert state.transcript==[current]  # Current dialogue travels separately to the runtime.
    assert "不应进入群聊" not in prompt.user_prompt
    assert "过时的图片猜测" not in prompt.user_prompt
    assert "旧的重复群聊快照" not in prompt.user_prompt
    assert "图片归属" in "\n".join(prompt.system_extra_blocks)


def test_volatile_handoff_retains_existing_prompt_boundary(tmp_path):
    service=MemoryService(store=MemoryStore(tmp_path/"m.db"),router=MemoryRouter(),clock=lambda:NOW,
        policy=MemoryPolicy(projection=ProjectionPolicy(enabled=True)))
    message=IncomingMessage.from_event({"post_type":"message","message_type":"private","user_id":"owner",
        "self_id":"bot","message_id":"current","message":[{"type":"text","data":{"text":"真红"}}]})
    turn=QQTurn(conversation_id="owner",conversation_type="private",primary_message_id="current",
        message_ids=("current",),combined_text="真红",messages=(message,))
    bridge=QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}),system_prompt="test persona",memory_service=service,sender=None)
    context={"qq_turn":turn,"owner_handoff_context":"未落库的短时交接线索，仅用于承接"}
    assert bridge._project_prompt_context(context) is None
    prompt=bridge._build_prompt(AgentState(transcript=[],context=context))
    assert context["owner_handoff_context"] in prompt.user_prompt
