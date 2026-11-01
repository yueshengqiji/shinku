from dataclasses import replace
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from shinku.memory import MemoryStore
from shinku.memory.projection.archive import ArchiveLocator
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.projection.retry import compare_main_retry
from shinku.qq.turn_preparation import _prepare_memory
from test_context_projection import NOW, req, put, engine


@pytest.mark.parametrize("days",[1,7,15,30])
def test_cross_day_restart_cold_lookup_and_explicit_link(tmp_path,days):
    store,projector=engine(tmp_path)
    original=put(store,"old","蓝晶项目仅限离线演练；我没有同意上线。",age=days*86400)
    projector.stop()
    restarted=ProjectionEngine(store,ProjectionPolicy(enabled=True))
    for request in (req("之前的蓝晶项目限制是什么",history_mode="retrieve"),
                    req("那件事的条件呢",reply_message_ids=("old",))):
        view=restarted.project(request)
        assert original.record_id in [r["record_id"] for r in view.records]
        assert "没有同意上线" in view.rendered
        assert view.diagnostics["semantic"]["calls"]==0
    restarted.stop()


def test_exact_date_does_not_import_today_or_wrong_day(tmp_path):
    store,projector=engine(tmp_path)
    right=put(store,"yesterday","蓝晶项目昨天未通过。",age=86410)
    put(store,"today","蓝晶项目今天通过。",age=100)
    view=projector.project(req("昨天的蓝晶项目",history_mode="retrieve",start_at=NOW-2*86400,end_at=NOW-86400))
    assert {r["record_id"] for r in view.records}=={right.record_id}


def test_missing_predecessor_is_not_complete_history(tmp_path):
    _,projector=engine(tmp_path)
    view=projector.project(req("那个人呢",reply_message_ids=("missing",)))
    assert not view.records
    assert "missing_predecessor" in view.diagnostics["gap_reasons"]
    assert view.coverage["absence_means"]=="unknown"


def test_topic_handles_persist_but_are_revalidated_after_restart(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,"old","小红说她没有接受邀请。",age=7*86400)
    first=projector.project(req("之前小红提过什么",history_mode="retrieve"))
    assert first.records
    projector.stop()
    restarted=ProjectionEngine(store,ProjectionPolicy(enabled=True))
    second=restarted.project(req("她接受了吗",now=NOW+30))
    assert second.topic["parent_topic"]==first.topic["topic_id"]
    assert row.record_id in second.topic["source_ids"]
    with store._connect() as conn:
        data=conn.execute("SELECT handles_json FROM memory_projection_topics").fetchone()[0]
        assert "小红" not in data and "接受" not in data


def test_complete_index_skips_full_fallback_but_source_update_invalidates(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,"old","蓝晶项目值0.42")
    for _ in range(8):
        if projector.archive.maintain()["state"]=="ready": break
    request=req("蓝晶项目",history_mode="retrieve")
    _,coverage=projector.archive.search(request,Deadline(500),limit=24)
    assert coverage["index_ready"] and not coverage["source_fallback_scan"]
    store.search(query="蓝晶项目",scope=row.scope,now=NOW)
    _,coverage=projector.archive.search(request,Deadline(500),limit=24)
    assert coverage["index_ready"]  # usage stamps are not source changes
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='琥珀项目值0.52' WHERE event_id=?",(row.record_id.split(':')[1],))
    found,coverage=projector.archive.search(req("琥珀项目"),Deadline(500),limit=24)
    assert row.record_id in [h[0] for h in found] and coverage["source_fallback_scan"]


def test_external_store_update_forces_index_fallback(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,"old","蓝晶项目值0.42")
    for _ in range(8):
        if projector.archive.maintain()["state"]=="ready":break
    other=MemoryStore(store.path)
    with other._connect() as conn:
        conn.execute("UPDATE memory_events SET text='琥珀项目值0.52' WHERE event_id=?",(row.record_id.split(':')[1],))
    found,coverage=projector.archive.search(req("琥珀项目"),Deadline(500),limit=24)
    assert coverage["source_fallback_scan"] and row.record_id in [r[0] for r in found]


def test_database_busy_stays_bounded_and_does_not_claim_absence(tmp_path):
    store,projector=engine(tmp_path)
    put(store,"old","只在周末备份。")
    lock=sqlite3.connect(store.path);lock.execute("BEGIN IMMEDIATE")
    try:
        stamp=time.monotonic();view=projector.project(req())
        assert time.monotonic()-stamp<1
        assert view.records and "topic_state_write_failed" in view.diagnostics["gap_reasons"]
    finally:
        lock.rollback();lock.close()


def test_model_timeout_late_result_never_cached(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True,semantic_timeout_ms=20)
    put(store,"a","小红说她不去。")
    put(store,"b","小雪说她不去。")
    release=threading.Event()
    def slow(payload):
        release.wait(1)
        return {"uncertain":True,"evidence":[{"record_id":payload["records"][0]["record_id"],
                                              "quote":payload["records"][0]["text"]}]}
    projector.semantic.model=slow
    view=projector.project(req("做建筑设计的她后来呢"))
    assert view.records and view.diagnostics["semantic"]["status"]=="timeout"
    release.set()
    deadline=time.monotonic()+1
    while projector.semantic.jobs.pending and time.monotonic()<deadline:
        time.sleep(.005)
    assert not projector.semantic.jobs.cache


def test_distillation_extracts_sources_only_and_can_be_disabled(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True,max_chars=3500,record_chars=1000)
    for i in range(4):put(store,str(i),f"项目{i}，"+"讨论细节。"*110+"但是未批准上线。")
    calls=[]
    def distill(payload):
        calls.append(payload["kind"])
        return {"uncertain":True,"missing":["节选不完整"],"excerpts":[
            {"record_id":row["record_id"],"quote":"但是未批准上线。"} for row in payload["records"]]}
    projector.semantic.model=distill
    view=projector.project(req("之前的项目约束",history_mode="retrieve"))
    assert calls==["distill"]
    assert "未批准上线" in view.rendered and len(view.rendered)<=3500
    projector.policy=replace(projector.policy,enabled=False)
    assert projector.project(req()) is None


def test_ordinary_greeting_never_distills_unrelated_large_recent_slice(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True,max_chars=3500,record_chars=1000)
    for i in range(4):put(store,str(i),"很长的旧话题。"*100)
    def forbidden(payload):raise AssertionError("ordinary greeting must not add a model call")
    projector.semantic.model=forbidden
    view=projector.project(req("你好呀"))
    assert view.diagnostics["semantic"]=={"status":"not_needed","calls":0}


def test_main_retry_typed_control_once_and_foreign_handle_rejected(tmp_path):
    store,projector=engine(tmp_path,ambiguity_strategy="main_retry")
    put(store,"a","小红说她不去。")
    view=projector.project(req("做建筑设计的她去哪了"));rid=view.records[0]["record_id"]
    calls=[]
    def model(payload):
        calls.append(payload)
        return {"control":{"type":"need_context","record_ids":[rid]}} if len(calls)==1 else {"draft":"尚未确认。"}
    result=compare_main_retry(call=model,request=req(),view=view,sources=projector.sources)
    assert len(calls)==2 and result["retries"]==1 and not result["publish"]
    normal=compare_main_retry(call=lambda p:{"draft":"NEED_CONTEXT 是一个词。"},request=req(),view=view,sources=projector.sources)
    assert normal["retries"]==0 and normal["status"]=="answer"
    bad=compare_main_retry(call=lambda p:{"control":{"type":"need_context","record_ids":["event:foreign"]}},
        request=req(),view=view,sources=projector.sources)
    assert bad["status"]=="unauthorized_handle"


def test_prepare_failure_preserves_user_write_and_safe_diagnostics():
    writes=[];traces=[]
    def fail(**kw):raise RuntimeError("secret body must not be logged")
    service=SimpleNamespace(prepare_turn=fail,record_turn=lambda **kw:writes.append(kw))
    deps=SimpleNamespace(memory_service=service,memory_message_meta=lambda turn:{"sender_id":"owner"},
        primary_task_request=lambda turn:"之前说过什么",recent_context=lambda turn:"",
        effects=SimpleNamespace(trace=lambda *a,**kw:traces.append(kw)),trace_memory=lambda *a:None)
    turn=SimpleNamespace(conversation_id="owner",conversation_type="private",primary_message_id="current",
                         combined_text="之前说过什么",visual_inputs=())
    result=_prepare_memory(turn,is_poke=False,trace_id="fixture",deps=deps)
    assert len(writes)==1 and result.audit["error_type"]=="RuntimeError"
    assert result.decision.history_mode=="retrieve"
    assert "secret body" not in json.dumps(traces)


class Embeddings:
    dimension=2
    def collection_key(self):return "fixture-synonyms-v1"
    def embed_text(self,text):return [1.,0.] if "安静" in text or "宁静" in text else [0.,1.]
    def embed_texts(self,texts):return [self.embed_text(t) for t in texts]


def test_semantic_paraphrase_beyond_actual_default_window_and_stale_vector(tmp_path):
    store=MemoryStore(tmp_path/"m.db",embedding_provider=Embeddings(),semantic_enabled=True)
    row=put(store,"old","工作时需要安静",age=30*86400)
    for i in range(801):put(store,str(i),"车辆与噪声",age=100+i)
    while store.backfill_embeddings(limit=256)>0:pass
    projector=ProjectionEngine(store,ProjectionPolicy(enabled=True))
    request=req("之前说喜欢宁静的条件",history_mode="retrieve")
    view=projector.project(request)
    assert row.record_id in [r["record_id"] for r in view.records]
    assert row.record_id in view.diagnostics["injected_source_ids"]
    assert "工作时需要安静" in view.rendered
    assert view.coverage["vector_rows"]==802
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='不再表达原偏好' WHERE event_id=?",(row.record_id.split(':')[1],))
    handles,_=projector.archive.semantic(request,Deadline(1000),limit=24)
    assert row.record_id not in [h[0] for h in handles]


def test_model_budget_survives_service_restart(tmp_path):
    store,projector=engine(tmp_path,semantic_enabled=True,semantic_daily_calls=1)
    put(store,"a","小红说她不去。")
    put(store,"b","小雪说她也不去。")
    projector.semantic.model=lambda p:{}
    projector.project(req("做建筑设计的她后来呢"))
    projector.stop()
    restarted=ProjectionEngine(store,projector.policy,lambda p:{})
    assert restarted.project(req("做建筑设计的她在哪里" )).diagnostics["semantic"]["status"]=="call_budget"


def test_owner_origin_correction_invalidates_copy(tmp_path):
    store,projector=engine(tmp_path)
    original=put(store,"old","旧条件",scope="group:g")
    store.record_event(conversation_id="master",scope="private:master",text=original.text,
        source_id="owner:"+original.source_id,created_at=original.created_at,
        metadata={**original.metadata,"owner_memory":True,"owner_source_scope":"group:g"})
    copy=store.event_by_source_id(source_id="owner:"+original.source_id,scope="private:master")
    request=req(allowed_scopes=("private:owner","private:master"),seed_handles=((copy.record_id,copy.scope,""),))
    assert copy.record_id in projector.project(request).diagnostics["injected_source_ids"]
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='新条件' WHERE event_id=?",(original.record_id.split(':')[1],))
    assert copy.record_id not in projector.project(request).diagnostics["injected_source_ids"]


def test_current_understanding_condition_and_confirmation_are_preserved(tmp_path):
    store,projector=engine(tmp_path)
    projector.understanding_enabled=True
    store.upsert_fact(conversation_id="owner",scope="private:owner",fact_key="coffee",text="我工作时不喝咖啡。",now=NOW-100,
        metadata={"role":"user","sender_id":"owner","confirmation_kind":"explicit_memory_request"})
    view=projector.project(req("之前咖啡的偏好",history_mode="retrieve"))
    assert view.records
    entry=view.records[0]["attribution"]["current_understanding"][0]
    assert entry["condition"]=="工作时" and entry["state"]=="confirmed"


def test_chinese_two_character_term_uses_complete_index(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,"coffee","我工作时不喝咖啡。",age=15*86400)
    for _ in range(8):
        if projector.archive.maintain()["state"]=="ready":break
    handles,coverage=projector.archive.search(req("之前咖啡的偏好"),Deadline(500),limit=24)
    assert row.record_id in [h[0] for h in handles]
    assert not coverage["source_fallback_scan"]


def test_fact_outside_actual_default_120_window_is_located(tmp_path):
    store,projector=engine(tmp_path)
    store.upsert_fact(conversation_id="owner",scope="private:owner",fact_key="old",text="蓝晶项目限定0.42",now=NOW-86400)
    for i in range(121):
        store.upsert_fact(conversation_id="owner",scope="private:owner",fact_key=str(i),text=f"天气与食谱{i}",now=NOW-100+i)
    view=projector.project(req("蓝晶项目限定",history_mode="retrieve"))
    assert "0.42" in view.rendered
    assert view.diagnostics["candidate_reads"]<=projector.policy.candidate_limit


def test_derived_index_observer_failure_cannot_break_canonical_write(tmp_path):
    store,projector=engine(tmp_path)
    projector.archive.after_commit=lambda *a: (_ for _ in ()).throw(RuntimeError("derived offline"))
    row=put(store,"new","原始消息必须能写入。")
    assert row.text=="原始消息必须能写入。"
