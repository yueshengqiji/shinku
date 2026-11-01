"""Paired saved-model/persona replay. Fictional SQLite only; sender=None."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import gray_memory_stage_a as a
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.projection.semantics import ProjectionModel
from shinku.agent.loop import AgentState

FIXTURE=a.ROOT/"tests/fixtures/context_projection_v1.json"


class Observed(a.ObservedRuntime):
    def call_aux_json(self, **kwargs):
        before=self.runtime.snapshot_metrics();started=time.monotonic()
        try:
            return self.runtime.call_aux_json(**kwargs)
        finally:
            after=self.runtime.snapshot_metrics()
            self.calls.append({"kind":"aux", "seconds":round(time.monotonic()-started,3),
                "input_chars":len(kwargs.get("user_prompt","")),
                "metrics_delta":{k:v-before.get(k,0) for k,v in after.items() if v!=before.get(k,0)}})


def seed(store, case, now):
    aliases={}
    for i,row in enumerate(case["history"]):
        role=row.get("role","user")
        source="turn:fictional-owner:"+row["id"]+(":assistant" if role=="assistant" else "")
        meta={"source":"qq_turn","role":role,"sender_id":"bot" if role=="assistant" else a.OWNER,
              "self_id":"bot","message_id":row["id"]}
        if role=="assistant":
            meta.update(requester_id=a.OWNER,delivery_status="sent",delivery_message_ids=["fixture-receipt"],
                        delivery_content_complete=True,fixture_only=True)
        store.record_event(conversation_id=a.OWNER,scope="private:"+a.OWNER,text=row["text"],source_id=source,
            created_at=now-row["age_days"]*86400-120+i,metadata=meta)
        aliases[store.event_by_source_id(source_id=source,scope="private:"+a.OWNER).record_id]=row["id"]
    count=case.get("noise",0)
    with store._connect() as conn:
        conn.executemany("""INSERT INTO memory_events
            (event_id,conversation_id,scope,text,layer,created_at,source_id,metadata_json) VALUES(?,?,?,?,?,?,?,?)""",
            [(f"noise-{i}",a.OWNER,"private:"+a.OWNER,f"虚构天气日志和阅读进度{i}","episodic",now-110+i%90,
              f"fixture:noise:{i}",json.dumps({"role":"user","sender_id":a.OWNER})) for i in range(count)])
    return aliases


def run_case(root, case, enabled, profile, now, evidence_char_budget=None):
    runtime,config,base_policy,build,persona,frozen=profile
    observed=Observed(runtime)
    policy=replace(base_policy, master_user_ids=(a.OWNER,),
                   projection=ProjectionPolicy(enabled=enabled,semantic_enabled=enabled),event_ttl_seconds=0)
    if evidence_char_budget is not None:
        policy=replace(policy,render_total_chars=evidence_char_budget,render_record_chars=2400,
                       retrieval_limit=24,max_results=24,
                       projection=replace(policy.projection,max_chars=evidence_char_budget))
    store=a.MemoryStore(root/"memory.sqlite3",embedding_provider=build.provider,semantic_enabled=policy.semantic_enabled)
    # No canonical profile or production data is used as seed material.
    aliases=seed(store,case,now)
    # Warm the retained vector index OUTSIDE measured requests, identically in both arms.
    if build.provider is not None:
        while store.backfill_embeddings(limit=256)>0:
            pass
    service=a.MemoryService(store=store,router=a.MemoryRouter(),clock=lambda:now,policy=policy,
                            projection_model=ProjectionModel(observed) if enabled else None)
    for rid, alias in tuple(aliases.items()):
        with store._connect() as conn:
            original=conn.execute("SELECT source_id FROM memory_events WHERE event_id=?",(rid.split(':',1)[1],)).fetchone()
        copy=store.event_by_source_id(source_id="owner:"+original[0],scope=service.owner_scope)
        if copy:
            aliases[copy.record_id]=alias
    if service.projection:
        while service.projection.archive.maintain()["state"]!="ready":
            pass
    message=a.IncomingMessage.from_event({"post_type":"message","message_type":"private",
        "user_id":a.OWNER,"self_id":"bot","message_id":"current","time":now,
        "message":[{"type":"text","data":{"text":case["query"]}}]})
    turn=a.QQTurn(conversation_id=a.OWNER,conversation_type="private",primary_message_id="current",
        message_ids=("current",),combined_text=case["query"],messages=(message,))
    bridge=a.QQAgentBridge(runtime=observed,system_prompt=persona,memory_service=service,sender=None,
        master_user_ids=(a.OWNER,),temperature=config.temperature,history_turns=config.history_turns)
    started=time.monotonic()
    memory=service.prepare_turn(conversation_id=a.OWNER,conversation_type="private",user_message=case["query"],
        message_meta={"sender_id":a.OWNER,"self_id":"bot","message_id":"current"})
    state=AgentState(transcript=[{"role":"user","content":case["query"]}], context={
        "qq_turn":turn,"memory_context":memory.as_dict(), "audience_context":{"role":"master","conversation_type":"private"},
        "chat_flow":{},"conversation_turn_id":"isolated-fixture"})
    prompt=bridge._build_prompt(state,include_agent_control=False)
    if evidence_char_budget is not None and len(prompt.user_prompt)>evidence_char_budget:
        raise ValueError("paired_evidence_budget_exceeded")
    prompt_seconds=time.monotonic()-started
    before=runtime.snapshot_metrics();call_started=time.monotonic()
    result=observed.call_chat_json(system_prompt=persona, user_prompt=prompt.user_prompt,
        system_extra_blocks=list(prompt.system_extra_blocks)+["请直接回应当前用户；返回 JSON {\"reply\":\"完整自然回复\"}，不要调用工具。"],
        history_turns=state.transcript, fallback={}, temperature=config.temperature,
        prompt_cache_key="shinku:projection-isolated-reply:v1")
    elapsed=time.monotonic()-call_started
    after=runtime.snapshot_metrics()
    # Complete reply is saved for evidence-based review, not scored as correct by keywords.
    text=str(result.get("reply") or "") if isinstance(result,dict) else ""
    injected_ids = state.context.get("context_projection_diagnostics",{}).get("injected_source_ids",[])
    found=sorted({alias for rid,alias in aliases.items() if rid in injected_ids})
    if not enabled:
        found=[aliases[r.record_id] for r in memory.records if r.record_id in aliases]
    row={"case":case["id"],"split":case["split"],"projection":enabled,
         "reply":text,"raw_result":result,"expected_sources":case["required"],"actual_sources":found,
         "source_recall":len(set(found)&set(case["required"]))/len(case["required"]),
         "constraints":case["constraints"],"forbidden":case["forbidden"],
         "prompt_seconds":round(prompt_seconds,3),"reply_seconds":round(elapsed,3),
         "total_seconds":round(time.monotonic()-started,3),"first_displayable_seconds":round(prompt_seconds+elapsed,3),
         "display_timing_note":"non-streaming JSON diagnostic; no production TTFT inference",
         "prompt_chars":len(prompt.user_prompt),"projection_diagnostics":state.context.get("context_projection_diagnostics",{}),
         "calls":observed.calls,"reply_metrics":{k:v-before.get(k,0) for k,v in after.items() if v!=before.get(k,0)},
         "error_type":runtime.snapshot_last_error().get("type","") if not text else "",
         "review_status":"pending_evidence_review", "sender":None,
         "paired_evidence_char_budget":evidence_char_budget,
         "protocol_note":"actual QQ prompt builder and saved persona; fixed non-streaming reply JSON diagnostic, no tools or delivery"}
    if service.projection:
        service.projection.stop()
    store.stop_embedding_worker()
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--evidence-char-budget",type=int,choices=[10000],default=None)
    parser.add_argument("--http-attempts",type=int,choices=[1,2],default=2)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    fixture=json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_kind"]=="fictional_only"
    profile=a.saved_profile(root)
    profile[0].runtime_policy=replace(profile[0].runtime_policy,http_attempts=args.http_attempts)
    profile[-1]['diagnostic_differences']=[v for v in profile[-1]['diagnostic_differences'] if 'one HTTP attempt' not in v]
    profile[-1]['diagnostic_differences'].append(f'45s chat/30s AUX SDK limits; {args.http_attempts} HTTP attempts; projection AUX additionally has a shared deadline')
    rows=[]
    source_paths=list((a.ROOT/"src/shinku/memory/projection").glob("*.py")) + [a.ROOT/"src/shinku/qq/prompting.py",
        a.ROOT/"src/shinku/memory/service.py",a.ROOT/"src/shinku/qq/agent_bridge.py",Path(__file__)]
    report={"fictional_only":True,"fixture_sha256":a.sha(FIXTURE),"profile":profile[-1],"rows":rows,
            "source_hashes":{str(p.relative_to(a.ROOT)):a.sha(p) for p in source_paths}}
    report["profile"]["diagnostic_differences"].append("sender=None; actual prompt-builder with fixed reply JSON; existing route judge disabled equally")
    for repeat in range(args.repeats):
        for case in fixture["cases"]:
            for enabled in (False,True) if repeat%2==0 else (True,False):
                target=root/f"r{repeat}-{case['id']}-{'projection' if enabled else 'baseline'}";target.mkdir()
                row=run_case(target,case,enabled,profile,fixture["now"],args.evidence_char_budget);row["repeat"]=repeat;rows.append(row)
                (root/"replay.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(json.dumps({k:row[k] for k in ("case","projection","source_recall","total_seconds","error_type")},ensure_ascii=False),flush=True)
    report["protected_configs_unchanged"]={path:a.sha(Path(path))==sha for path,sha in profile[-1]["protected_config_hashes"].items()}
    (root/"replay.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
