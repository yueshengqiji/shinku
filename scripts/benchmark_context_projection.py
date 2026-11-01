"""Fictional-only locator scaling; no production DB, configuration or network."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from shinku.memory import MemoryStore
from shinku.memory.projection.archive import ArchiveLocator
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.sources import ProjectionSources

NOW=1_800_000_000
SCOPE="private:fictional-owner"
POLICY=ProjectionPolicy(enabled=True)


def seed(store, count):
    targets={"far":0,"middle":count//2,"near":count-1}
    wanted={f"event:synthetic-{index}" for index in targets.values()}
    with store._connect() as conn:
        conn.executemany("""INSERT INTO memory_events
            (event_id,conversation_id,scope,text,layer,created_at,source_id,metadata_json)
            VALUES(?,?,?,?,?,?,?,?)""", [
            (f"synthetic-{i}","fictional-owner",SCOPE,
             f"琥珀项目：阶段{i}，参数0.72；只在离线模式下使用，不要直接发布。" if f"event:synthetic-{i}" in wanted
             else f"虚构无关日志条目{i}，今日天气与读书进度。",
             "episodic",NOW-40*86400+i,f"fictional:{i}",json.dumps({"role":"user","sender_id":"fictional-owner"}))
            for i in range(count)])
    return wanted


def run_scale(root,count):
    store=MemoryStore(root/f"scale-{count}.sqlite3")
    wanted=seed(store,count)
    locator=ArchiveLocator(store,POLICY)
    reader=ProjectionSources(store)
    request=ProjectionRequest(scope=SCOPE,allowed_scopes=(SCOPE,),requester="fictional-owner",
        self_id="bot",query="琥珀项目",message_id="current",now=NOW,history_mode="retrieve")
    output=[]
    for state in ("partial_index", "complete_index"):
        if state=="complete_index":
            started=time.monotonic(); steps=0
            while steps < (count//POLICY.maintenance_batch+3)*3:
                steps+=1
                result=locator.maintain()
                if result["state"]=="ready":
                    break
                if result["state"]=="fallback":
                    raise RuntimeError("fixture_index_build_failed")
            indexing={"steps":steps,"seconds":round(time.monotonic()-started,3),**result}
        else:
            locator.maintain()
            indexing={"steps":1,"state":"partial"}
        for sample in range(5):
            stamp=time.monotonic()
            baseline=store.search(query=request.query,scope=SCOPE,now=NOW,limit=POLICY.source_limit)
            baseline_ms=(time.monotonic()-stamp)*1000
            stamp=time.monotonic()
            handles,coverage=locator.search(request,Deadline(POLICY.retrieval_ms),limit=POLICY.candidate_limit)
            rows=reader.read(handles[:POLICY.source_limit],request,Deadline(300))
            actual={r.record_id for r in rows if r}
            output.append({"scale":count,"index_state":state,"sample":sample,
                "cache":"cold" if sample==0 else "warm", "baseline_ms":round(baseline_ms,3),
                "projection_ms":round((time.monotonic()-stamp)*1000,3),
                "expected_ids":sorted(wanted),"actual_ids":sorted(actual),
                "baseline_recall_at_24":len(wanted & {r.record_id for r in baseline})/len(wanted),
                "projection_recall_at_24":len(wanted & actual)/len(wanted),
                "irrelevant_fraction":len(actual-wanted)/max(1,len(actual)),"coverage":coverage,
                "indexing":indexing})
    locator.stop()
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for count in (1000,10000,50000):
        rows.extend(run_scale(root,count))
        (root/"scale-results.json").write_text(json.dumps({"fictional_only":True,
            "policy":asdict(POLICY),"rows":rows},ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"scale":count,"recall":min(r["projection_recall_at_24"] for r in rows if r["scale"]==count),
            "median_ms":statistics.median(r["projection_ms"] for r in rows if r["scale"]==count)},ensure_ascii=False),flush=True)
    if not all(r["projection_recall_at_24"]==1 for r in rows):
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
