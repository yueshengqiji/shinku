"""Actual scoped tool/reader regressions; no semantic model quality claims."""
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.qq.assembly_tools import memory_read_capabilities
from shinku.qq.interaction_tools import audited_handlers
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry
from test_context_projection import NOW, put
from test_private_dialogue_continuity import turn


@pytest.fixture
def memory(tmp_path):
    def judge(**kwargs):
        pytest.fail("The main-model strategy must not add a memory classification call")
    service = MemoryService(store=MemoryStore(tmp_path/"memory.sqlite3"),
        router=MemoryRouter(judge, deterministic_guards=False), clock=lambda:NOW,
        policy=MemoryPolicy(event_ttl_seconds=0, event_scan_limit=10, max_results=12,
                            projection=ProjectionPolicy(enabled=True,retrieval_strategy="model")))
    yield service
    service.projection.stop()


def invoke(service, query, *, current="它是什么规格？", kind="private", sender="owner", mode="retrieve"):
    return MemorySearchTool(service).execute(call={"query":query,"mode":mode,"limit":8,
        "recorded_time":{"start":"","end":""}},
        context={"qq_turn":turn(current,conversation="g" if kind=="group" else sender,kind=kind,sender=sender)})


@pytest.mark.parametrize("query", ["苍鹭 K24", "Oriole 77", "2027年1月1日的方案", "你好呀", "她的接口呢？"])
def test_normal_prompt_does_not_classify_or_force_archive_candidates(memory, query):
    put(memory.store,"old", "Oriole 77 uses cyan ink.",age=40*86400)
    memory.projection.archive.discover = lambda *a, **kw: pytest.fail("Automatic projection is not an archive request")
    context = memory.prepare_turn(conversation_id="owner",conversation_type="private",user_message=query,
                                 recent_context="刚才的双方对话",message_meta={"sender_id":"owner","self_id":"bot"})
    assert context.decision.reason == "main_model_retrieval"
    assert not context.decision.used_model and not context.records
    view = memory.project_context(conversation_id="owner",conversation_type="private",user_message=query,
                                  message_meta={"sender_id":"owner","self_id":"bot"},memory_context=context.as_dict())
    assert not view["records"] and view["diagnostics"]["semantic"]["calls"] == 0


def test_model_query_reads_cold_source_outside_legacy_window_without_maintenance(memory, monkeypatch):
    target = put(memory.store,"old", "Oriole 77 uses cyan ink; laboratory prototypes only.",age=40*86400)
    for i in range(14):
        put(memory.store,f"noise{i}",f"Unrelated journal {i}",age=100+i)
    # 直接验证旧来源在近期窗口外，保留后续独立发现、无维护和完整正文的断言。
    with memory.store._connect() as conn:
        recent = conn.execute("SELECT event_id FROM memory_events WHERE scope=? ORDER BY created_at DESC LIMIT ?",
                              (target.scope, memory.store.event_scan_limit)).fetchall()
    assert target.record_id.removeprefix("event:") not in {row[0] for row in recent}
    assert target.record_id in {r.record_id for r in memory.store.search(query="Oriole 77",scope=target.scope,now=NOW)}
    monkeypatch.setattr(memory.store,"backfill_embeddings",lambda **kw:pytest.fail("No request-time backfill"))
    monkeypatch.setattr(memory.projection.archive,"maintain",lambda:pytest.fail("No request-time maintenance"))
    monkeypatch.setattr(memory.store,"search",lambda **kw:pytest.fail("The explicit query must use shared discovery"))
    result = invoke(memory,"Oriole 77")
    inspection = result["state_updates"]["memory_inspection"]
    assert target.record_id in {r["record_id"] for r in inspection["records"]}
    assert "laboratory prototypes only" in result["followup_context"]
    assert inspection["coverage"]["maintenance_in_request"] == 0
    assert inspection["coverage"]["absence_means"] == "unknown"
    assert inspection["model_calls"] == 0 and not inspection["execution_authority"]


def test_explicit_reply_context_is_read_in_one_bounded_batch(memory):
    user = put(memory.store,"old", "蓝玻璃材料只能用于样品，尚未批准用于食品容器。",age=40*86400)
    reply = put(memory.store,"old:assistant", "我只是解释选项，还没有开始采购。",age=40*86400-1,
        role="assistant",sender="bot",requester_id="owner",delivery_status="sent",delivery_message_ids=["receipt"])
    result = invoke(memory,"蓝玻璃")
    inspection = result["state_updates"]["memory_inspection"]
    assert {user.record_id,reply.record_id} <= {r["record_id"] for r in inspection["records"]}
    assert "尚未批准" in result["followup_context"] and "还没有开始" in result["followup_context"]
    assert inspection["coverage"]["candidate_reads"] <= memory.policy.projection.candidate_limit


def test_scope_and_delivery_still_bound_model_reads(memory):
    put(memory.store,"secret","K24 secret private note.",scope="private:other",sender="other",age=40*86400)
    put(memory.store,"draft:assistant","K24 has already shipped.",scope="group:g",role="assistant",sender="bot",
        requester_id="owner",delivery_status="failed",delivery_message_ids=["fake"],age=40*86400)
    result = invoke(memory,"K24 private:other",kind="group")
    inspection = result["state_updates"]["memory_inspection"]
    assert not inspection["records"]
    assert inspection["retrieval_scopes"] == ["group:g"]
    assert "secret private note" not in result["followup_context"] and "already shipped" not in result["followup_context"]
    assert MemorySearchTool(memory).normalize_call({"query":"K24","scope":"private:other"}) is None


@pytest.mark.parametrize("mutation", ["edit", "forget"])
def test_late_source_changes_are_not_published(memory, mutation):
    row = put(memory.store,"old","Ginkgo specification: provisional.",age=40*86400)
    read = memory.projection.sources.read
    done = []
    def changed(handles, request, deadline):
        values = read(handles, request, deadline)
        if any(values) and not done:
            done.append(True)
            if mutation == "forget":
                memory.store.forget(scope=row.scope,record_id=row.record_id,query="",now=NOW)
            else:
                with memory.store._connect() as conn:
                    conn.execute("UPDATE memory_events SET text='Corrected source.' WHERE event_id=?",
                                 (row.record_id.removeprefix("event:"),))
        return values
    memory.projection.sources.read = changed
    result = invoke(memory,"Ginkgo specification")
    inspection = result["state_updates"]["memory_inspection"]
    assert done and not inspection["records"]
    assert "provisional" not in result["followup_context"]
    assert any(g["reason"] == "source_changed_during_projection" for g in inspection["gaps"])
    assert result["state_updates"]["memory_lookup"]["status"] == "read_failed"


def test_real_vector_table_and_lexical_failure_keep_source_fallback(memory):
    put(memory.store,"old","Magnolia dimensions are tentative.",age=40*86400)
    with memory.store._connect() as conn:
        conn.execute("DROP TABLE memory_projection_fts")
    result = invoke(memory,"Magnolia")
    assert "dimensions are tentative" in result["followup_context"]
    assert result["state_updates"]["memory_inspection"]["coverage"]["state"] == "bounded_source_fallback"


def test_timeout_is_a_failed_read_not_proof_of_absence(memory):
    def timeout(request, deadline, **kwargs):
        deadline.until = 0
        raise TimeoutError("synthetic deadline")
    memory.projection.archive.search = timeout
    result = invoke(memory,"Old specification")
    assert result["state_updates"]["memory_lookup"]["status"] == "read_failed"
    assert not result["state_updates"]["memory_lookup"]["query_completed"]
    assert result["state_updates"]["memory_inspection"]["coverage"]["absence_means"] == "unknown"


def test_host_only_grants_builtin_opted_in_reader_and_preserves_explicit_capability(memory):
    handler = MemorySearchTool(memory)
    host = ToolHost(ToolRegistry({"memory_search":handler}))
    caps = memory_read_capabilities(None,memory,host)
    assert set(caps) == {"memory_search"}
    # 主模型读取最多三次，补查仍使用同一轮的当前授权与预算。
    assert caps["memory_search"].max_calls == 3
    assert set(audited_handlers(host.registry.view(),caps)) == {"memory_search"}
    narrower = replace(caps["memory_search"],max_calls=1,max_result_bytes=1000)
    assert memory_read_capabilities({"memory_search":narrower},memory,host)["memory_search"] is narrower
    class Impostor(MemorySearchTool):
        pass
    other = ToolHost(ToolRegistry({"memory_search":Impostor(memory)}))
    assert memory_read_capabilities(None,memory,other) == {}
    memory.policy = replace(memory.policy,projection=replace(memory.policy.projection,retrieval_strategy="legacy"))
    assert memory_read_capabilities(None,memory,host) == {}


def test_tool_result_stays_within_audited_byte_budget(memory):
    for i in range(12):
        put(memory.store,f"long-{i}","Longcase "+"这是带有限定范围的原始资料。"*140,age=40*86400+i)
    result = invoke(memory,"Longcase")
    cap = MemorySearchTool(memory).chat_read_capability()
    assert len(json.dumps(result,ensure_ascii=False).encode()) <= cap.max_result_bytes
    assert not result["state_updates"]["memory_inspection"]["coverage"]["complete"]


def test_large_or_malformed_model_query_is_rejected_whole(memory):
    tool = MemorySearchTool(memory)
    assert tool.normalize_call({"query":"x"*2001, "recorded_time":{"start":"", "end":""}}) is None
    assert tool.normalize_call({"query":["x"], "recorded_time":{"start":"", "end":""}}) is None
    query = "适用范围不变。"*100
    assert tool.normalize_call({"query":query, "recorded_time":{"start":"", "end":""}})["query"] == query


def test_task_entry_can_request_cold_memory_without_starting_execution(tmp_path, memory):
    from shinku.agent_runtime.demo import demo_service
    from test_qq_tool_decision_boundary import Chat, boundary_bridge
    svc, execution_tool, executor = demo_service(tmp_path/"runtime")
    text = "Oriole 77 uses cyan ink; laboratory prototypes only."
    put(memory.store,"old",text,age=40*86400)
    handler = MemorySearchTool(memory)
    svc.executor.tool_host.registry.register(handler)
    caps = memory_read_capabilities(None,memory,svc.executor.tool_host)
    class ReadingChat(Chat):
        def call_chat_json(self, **options):
            if "历史答案复核" in "\n".join(options.get("system_extra_blocks", [])):
                self.calls.append(options)
                row = next(r for r in json.loads(options["user_prompt"])["actual_records"] if "Oriole" in r["text"])
                return {"disposition":"found", "claims":[{"source_id":row["id"],"quote":text,"answer_fragment":text}],
                        "coverage":[{"field":"原问题","supported":True,"claim_indices":[0]}]}
            return super().call_chat_json(**options)
    chat = ReadingChat([{"kind":"tool","tool_call":{"name":"memory_search","arguments":{"query":"Oriole 77","mode":"retrieve","limit":4,
        "recorded_time":{"start":"","end":""}}}},
                        {"kind":"final","text":text}])
    bridge = boundary_bridge(svc,execution_tool,chat,chat_read_capabilities=caps,memory_service=memory,
                             master_user_ids=("owner",))
    try:
        result = bridge(turn("Oriole 77",mid="current"))
        assert result.reason == "lightweight_perception_fast_track", result
        assert result.agent_status == "completed", result
        assert not result.background_task_id and not svc.coordinator.list()
        assert not executor.calls and not execution_tool.calls
        assert not svc.task_store.candidates(requester="owner",session="private:owner")
        assert any(spec["function"]["name"] == "memory_search" for spec in chat.calls[0]["native_tools"])
        assert len(chat.calls) == 2  # existing main tool turn, answer, source-use review
        assert "laboratory prototypes only" in chat.calls[1]["user_prompt"]
    finally:
        svc.coordinator.shutdown(wait=True)
        svc.task_store.close()
        bridge.conversation_runtime.close()


@pytest.mark.parametrize("mode", ["retrieve", "summary"])
def test_calendar_window_applies_to_requested_and_linked_sources(memory, mode):
    from datetime import datetime
    stamp = int(datetime(2026, 8, 5, 12).timestamp())
    memory.clock = lambda: stamp
    def row(mid, text, date):
        return put(memory.store,mid,text,age=NOW-int(datetime.fromisoformat(date).timestamp()))
    old = row("old", "Jade meeting before window.", "2026-08-03T15:00:00")
    wanted = row("wanted", "Jade meeting not approved.", "2026-08-04T15:00:00")
    future = row("future", "Jade meeting after window.", "2026-08-05T10:00:00")
    result = MemorySearchTool(memory).execute(call={"query":"2026年8月4日 Jade", "mode":mode, "limit":8,
        "recorded_time":{"start":"2026-08-04", "end":"2026-08-05"}},
        context={"qq_turn":turn("2026年8月4日的记录",mid="current")})
    rows = result["state_updates"]["memory_inspection"]["records"]
    assert {r["record_id"] for r in rows} == {wanted.record_id}
    assert old.record_id not in {r["record_id"] for r in rows}
    assert future.record_id not in {r["record_id"] for r in rows}


def test_partial_receipt_and_activity_scope_remain_explicit(memory):
    put(memory.store,"old:assistant","K24 only this fragment was delivered.",age=40*86400,
        role="assistant",sender="bot",requester_id="owner",delivery_status="partial",
        delivery_message_ids=["receipt"],delivery_content_complete=False)
    row = invoke(memory,"K24")["state_updates"]["memory_inspection"]["records"][0]
    assert row["metadata"]["delivery_status"] == "partial"
    assert row["metadata"]["delivery_content_complete"] is False
    result = invoke(memory,"今天在群里",mode="activity",sender="visitor")
    assert not result["state_updates"]["memory_inspection"]["records"]


def test_assembly_installs_model_memory_read_capability(tmp_path, monkeypatch):
    import shinku.qq.assembly as assembly
    monkeypatch.setenv("SHINKU_CONTEXT_PROJECTION_ENABLED","true")
    monkeypatch.setenv("SHINKU_CONTEXT_PROJECTION_RETRIEVAL_STRATEGY","model")
    monkeypatch.setattr(assembly,"build_embedding_provider",lambda *a,**kw:
                        SimpleNamespace(provider=None,status="offline-fixture",error=""))
    persona = tmp_path/"persona.md"
    persona.write_text("Public synthetic persona.",encoding="utf-8")
    config = assembly.QQAgentConfig(enabled=True,send_enabled=False,memory_enabled=True,
        memory_db_path=str(tmp_path/"memory.db"),memory_judge_enabled=False,
        persona_file=str(persona),chat_api_key="fixture",chat_base_url="https://example.invalid/v1",
        chat_model_name="fixture",agent_state_store_path=str(tmp_path/"tasks.db"),
        trace_store_path=str(tmp_path/"trace.json"),conversation_state_store_path=str(tmp_path/"conversation.db"))
    result = assembly.assemble_qq_agent(config,runtime=SimpleNamespace(call_chat_json=lambda **kw: {}))
    try:
        bridge = result.bridge
        assert bridge.sender is None
        assert bridge.memory_service.policy.projection.retrieval_strategy == "model"
        assert "memory_search" in bridge.chat_read_capabilities
        assert bridge.chat_read_capabilities["memory_search"].max_calls == 3
    finally:
        result.task_coordinator.shutdown(wait=True)
        result.bridge.memory_service.projection.stop()
        result.bridge.memory_service.store.stop_embedding_worker()
        result.bridge.task_store.close()
        result.bridge.conversation_runtime.close()


def test_detailed_coverage_footer_cannot_discard_all_retrieved_evidence(memory):
    target = put(memory.store,"cold","Spruce reader requires offline driver; installation not approved.",age=40*86400)
    discover = memory.projection.archive.discover
    def diagnostics(*args,**kwargs):
        handles,coverage,gaps = discover(*args,**kwargs)
        coverage.update({"diagnostic_"+str(i):"bounded source coverage" for i in range(30)})
        return handles,coverage,gaps
    memory.projection.archive.discover = diagnostics
    for i in range(10):
        put(memory.store,str(i),"Spruce noise "+str(i),age=40*86400+10+i)
    result = invoke(memory,"Spruce reader")
    inspection = result["state_updates"]["memory_inspection"]
    assert target.record_id in {r["record_id"] for r in inspection["records"]}
    assert "not approved" in inspection["rendered"]
    assert len(inspection["rendered"]) <= 4500


def test_generated_intent_is_not_overridden_by_incidental_input_calendar(memory):
    target = put(memory.store,"old","Kestrel Q4 has a sealed battery; replacement not approved.",age=40*86400)
    result = invoke(memory,"Kestrel Q4 sealed battery",current="今天开会。顺便问下 Kestrel Q4 电池如何更换？")
    inspection = result["state_updates"]["memory_inspection"]
    assert target.record_id in {r["record_id"] for r in inspection["records"]}
    assert inspection["coverage"]["requested_time_window"] == [None,None]


def test_discovery_failure_with_no_sources_is_not_reported_as_successful_absence(memory):
    def broken(*args,**kwargs):
        raise sqlite3.OperationalError("isolated source index failure")
    memory.projection.archive.search = broken
    result = invoke(memory,"Kestrel prior conditions")
    assert result["state_updates"]["memory_lookup"]["status"] == "read_failed"
    assert not result["state_updates"]["memory_lookup"]["query_completed"]
