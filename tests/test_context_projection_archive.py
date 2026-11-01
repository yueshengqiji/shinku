from dataclasses import replace
import sqlite3

import pytest

from shinku.memory import MemoryStore
from shinku.memory.projection.archive import ArchiveLocator
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.sources import ProjectionSources

NOW=1_800_000_000


def request(query="琥珀项目参数", **values):
    return ProjectionRequest(scope="private:owner",allowed_scopes=("private:owner",),requester="owner",
        self_id="bot",query=query,message_id="current",now=NOW,history_mode="retrieve",**values)


def event(store,source,text,*,created=NOW-100,scope="private:owner"):
    store.record_event(conversation_id="owner",scope=scope,text=text,source_id=source,created_at=created,
        metadata={"role":"user","sender_id":"owner"})
    return store.event_by_source_id(source_id=source,scope=scope)


def test_old_original_outside_recent_window_found_without_index_or_date(tmp_path):
    store=MemoryStore(tmp_path/"m.db",event_scan_limit=10)
    old=event(store,"old","琥珀项目参数设定为0.72",created=NOW-30*86400)
    for i in range(11):
        event(store,f"noise-{i}",f"无关天气记录编号{i}",created=NOW-20+i)
    # 验证目标位于原始近期窗口外；Store 的跨窗口补查不再要求复现旧缺陷。
    with store._connect() as conn:
        recent = conn.execute("SELECT event_id FROM memory_events WHERE scope=? "
                              "ORDER BY created_at DESC LIMIT ?", (old.scope, store.event_scan_limit)).fetchall()
    assert old.record_id.removeprefix("event:") not in [row[0] for row in recent]
    assert old.record_id in [r.record_id for r in store.search(query="琥珀项目参数",scope=old.scope,now=NOW)]
    archive=ArchiveLocator(store,ProjectionPolicy())
    handles,coverage=archive.search(request(),Deadline(3000),limit=20)
    rows=ProjectionSources(store).read(handles,request(),Deadline(3000))
    assert old.record_id in [r.record_id for r in rows if r]
    assert coverage["maintenance_in_request"]==0
    assert coverage["state"]=="source_checked"


def test_old_fact_outside_window_and_date_filter(tmp_path):
    store=MemoryStore(tmp_path/"m.db",fact_scan_limit=10)
    store.upsert_fact(conversation_id="owner",scope="private:owner",fact_key="old",text="琥珀项目参数0.72",now=NOW-90000)
    for i in range(11):
        store.upsert_fact(conversation_id="owner",scope="private:owner",fact_key=str(i),text=f"无关事实{i}",now=NOW-100+i)
    archive=ArchiveLocator(store,ProjectionPolicy())
    handles,_=archive.search(request(),Deadline(3000),limit=20)
    assert any(r and "0.72" in r.text for r in ProjectionSources(store).read(handles,request(),Deadline(3000)))
    handles,_=archive.search(request(start_at=NOW-1000,end_at=NOW),Deadline(3000),limit=20)
    assert not handles


def test_index_is_only_locator_and_correction_forget_scope_are_rechecked(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    old=event(store,"old","琥珀项目参数0.72")
    foreign=event(store,"foreign","琥珀项目参数0.99",scope="private:foreign")
    archive=ArchiveLocator(store,ProjectionPolicy())
    assert archive.maintain()["written"]==2
    handles,_=archive.search(request(),Deadline(3000),limit=20)
    assert foreign.record_id not in [x[0] for x in handles]
    reader=ProjectionSources(store)
    before=reader.read(handles,request(),Deadline(3000))[0]
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='蓝晶项目参数0.61' WHERE event_id=?",(old.record_id.split(':')[1],))
    assert reader.read(((old.record_id,old.scope,before.metadata["source_version"]),),request(),Deadline(3000))==(None,)
    newreq=request("蓝晶项目参数")
    changed,_=archive.search(newreq,Deadline(3000),limit=20)
    assert old.record_id in [x[0] for x in changed]
    store.forget(scope=old.scope,record_id=old.record_id,query="",now=NOW)
    assert reader.read(changed,newreq,Deadline(3000))==(None,)


def test_missing_derived_index_does_not_break_source_writes(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    archive=ArchiveLocator(store,ProjectionPolicy())
    with store._connect() as conn:
        conn.execute("DROP TABLE memory_projection_fts")
    old=event(store,"old","琥珀项目参数0.72")
    handles,coverage=archive.search(request(),Deadline(3000),limit=20)
    assert old.record_id in [x[0] for x in handles]
    assert coverage["state"]=="bounded_source_fallback"
    assert archive.maintain()["state"]=="fallback"


def test_bounded_reader_checks_expiry_without_mutating_source(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    old=event(store,"old","琥珀项目参数0.72")
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET expires_at=? WHERE event_id=?",(NOW-1,old.record_id.split(':')[1]))
    assert ProjectionSources(store).read(((old.record_id,old.scope,""),),request(),Deadline(3000))==(None,)
    with store._connect() as conn:
        assert conn.execute("SELECT status FROM memory_events").fetchone()[0]=="active"


def test_unknown_scope_and_exhausted_budget_fail_closed(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    old=event(store,"old","琥珀项目参数0.72",scope="private:foreign")
    reader=ProjectionSources(store)
    assert reader.read(((old.record_id,old.scope,""),),request(),Deadline(100))==(None,)
    deadline=Deadline(10); deadline.until=0
    with pytest.raises(TimeoutError):
        reader.read(((old.record_id,old.scope,""),),request(),deadline)


def test_locator_has_restartable_bounded_backfill(tmp_path):
    store=MemoryStore(tmp_path/"m.db")
    for i in range(3):
        event(store,str(i),f"琥珀项目参数{i}")
    policy=replace(ProjectionPolicy(),maintenance_batch=1)
    first=ArchiveLocator(store,policy)
    assert first.maintain()["written"]==1
    second=ArchiveLocator(store,policy)
    assert second.maintain()["written"]==1
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_projection_documents").fetchone()[0]==2
