"""Saved-profile probes for real consecutive replies and optional slow protocols."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import gray_memory_stage_a as a
import gray_context_projection as paired
from shinku.agent.loop import AgentState
from shinku.memory.projection.contracts import ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.projection.retry import compare_main_retry
from shinku.memory.projection.semantics import ProjectionModel

NOW=1_800_000_000
SCOPE="private:"+a.OWNER


def request(query,now=NOW):
    return ProjectionRequest(scope=SCOPE,allowed_scopes=(SCOPE,),requester=a.OWNER,self_id="bot",
                             message_id="current",query=query,now=now,history_mode="retrieve")


def store_row(store,source,text,now,role="user"):
    meta={"source":"qq_turn","role":role,"sender_id":"bot" if role=="assistant" else a.OWNER,
          "self_id":"bot","message_id":source.removesuffix(":assistant"),"fixture_only":True}
    if role=="assistant":
        meta.update(requester_id=a.OWNER,delivery_status="sent",delivery_message_ids=["offline-fixture"],delivery_content_complete=True)
    store.record_event(conversation_id=a.OWNER,scope=SCOPE,text=text,created_at=now,
                        source_id="turn:"+a.OWNER+":"+source,metadata=meta)


def reply(runtime,persona,query,context,system_blocks=()):
    started=time.monotonic();before=runtime.snapshot_metrics()
    result=runtime.call_chat_json(system_prompt=persona,user_prompt=context,
        system_extra_blocks=[*system_blocks,"请直接回复当前用户。只返回 JSON {\"reply\":\"完整自然回复\"}，不调用工具。"],
        history_turns=[{"role":"user","content":query}],fallback={},temperature=.7,prompt_cache_key="projection-probe:v1")
    after=runtime.snapshot_metrics()
    return {"reply":str(result.get("reply") or ""),"seconds":round(time.monotonic()-started,3),
            "metrics":{k:v-before.get(k,0) for k,v in after.items() if v!=before.get(k,0)}}


def sequence(root, profile, enabled):
    runtime,config,base_policy,_,persona,_=profile
    clock=[NOW]
    policy=replace(base_policy,semantic_enabled=False,projection=ProjectionPolicy(enabled=enabled),event_ttl_seconds=0,
                   master_user_ids=(a.OWNER,))
    store=a.MemoryStore(root/"sequence.sqlite3")
    store_row(store,"setup","蓝晶计划：只有我明确批准后才能发布，现在没有批准。",NOW-60)
    steps=[(1,"那个蓝晶计划现在能发布吗？"),(7,"换个话题，周末我想在家看书。"),
           (15,"回到蓝晶计划，我当时有没有批准发布？"),(30,"那就还是先别发布，条件没变，对吧？")]
    rows=[]
    for day,query in steps:
        clock[0]=NOW+day*86400
        # Each step reconstructs service state. Only actual generated fixture
        # dialogue and source-bound locators survive; no live QQ receipt exists.
        service=a.MemoryService(store=store,router=a.MemoryRouter(),clock=lambda:clock[0],policy=policy)
        bridge=a.QQAgentBridge(runtime=runtime,system_prompt=persona,memory_service=service,sender=None,
            master_user_ids=(a.OWNER,),temperature=config.temperature)
        mid="day-"+str(day)
        message=a.IncomingMessage.from_event({"post_type":"message","message_type":"private","user_id":a.OWNER,
            "self_id":"bot","message_id":mid,"time":clock[0],"message":[{"type":"text","data":{"text":query}}]})
        turn=a.QQTurn(conversation_id=a.OWNER,conversation_type="private",primary_message_id=mid,
                     message_ids=(mid,),combined_text=query,messages=(message,))
        prepared=service.prepare_turn(conversation_id=a.OWNER,conversation_type="private",user_message=query,
            message_meta={"sender_id":a.OWNER,"self_id":"bot","message_id":mid})
        state=AgentState(transcript=[{"role":"user","content":query}],context={"qq_turn":turn,"current_time":clock[0],
            "memory_context":prepared.as_dict(),"audience_context":{"role":"master","conversation_type":"private"}})
        prompt=bridge._build_prompt(state,include_agent_control=False)
        row=reply(runtime,persona,query,prompt.user_prompt,prompt.system_extra_blocks)
        row.update(day=day,query=query,projection=enabled,diagnostics=state.context.get("context_projection_diagnostics",{}),
                   constraint="No publication authorization; preserve condition and return to old topic",sender=None)
        rows.append(row)
        store_row(store,mid,query,clock[0])
        if row["reply"]:
            store_row(store,mid+":assistant",row["reply"],clock[0]+1,role="assistant")
        if service.projection:service.projection.stop()
        print(json.dumps({"probe":"sequence","day":day,"projection":enabled,"seconds":row["seconds"]}),flush=True)
    return rows


def slow_paths(root,profile):
    runtime,_,_,_,persona,_=profile
    rows=[]
    # Judge: same saved AUX, a longer bounded experimental timeout; no default change.
    store=a.MemoryStore(root/"judge.sqlite3")
    store_row(store,"a","小红说她是建筑师，周六想去看展。",NOW-10)
    store_row(store,"b","小雪说她是医生，还没想好去哪。",NOW-9)
    projector=ProjectionEngine(store,ProjectionPolicy(enabled=True,semantic_enabled=True,semantic_timeout_ms=30000),ProjectionModel(runtime))
    req=request("做建筑设计的她，周六想去哪里？")
    view=projector.project(req)
    answer=reply(runtime,persona,req.query,view.rendered)
    rows.append({"probe":"judge_30s","diagnostics":view.diagnostics,"judgment":view.topic.get("judge"),
                 "reply":answer,"constraint":"refer to 小红 and 看展, not 小雪; selection remains provisional"})
    projector.stop()

    # Distillation: distinct obligations, no summary-of-summary, all originals retained.
    store=a.MemoryStore(root/"distill.sqlite3")
    for i in range(8):
        store_row(store,str(i),f"荧石项目第{i+1}条约束："+"以下是虚构的讨论背景，并不表示执行完成。"*72+
                  f"最终条件：仅在离线演练中使用参数{i+1}.25；没有批准发布。",NOW-20+i)
    for enabled in (False,True):
        projector=ProjectionEngine(store,ProjectionPolicy(enabled=True,semantic_enabled=enabled,semantic_timeout_ms=30000),
                                   ProjectionModel(runtime) if enabled else None)
        req=request("荧石项目这八条约束各自的参数是多少？是否批准发布？")
        view=projector.project(req)
        answer=reply(runtime,persona,req.query,view.rendered)
        rows.append({"probe":"distill","semantic":enabled,"diagnostics":view.diagnostics,"reply":answer,
                     "injected":[{"text":r["text"],"representation":r["representation"]} for r in view.records],
                     "constraint":"1.25 through 8.25, offline only, publication not approved"})
        projector.stop()

    # Main-model NEED_CONTEXT comparison: kept out of the production protocol.
    store=a.MemoryStore(root/"retry.sqlite3")
    store_row(store,"retry","紫藤项目的背景讨论。"*100+"最终参数是0.43，仅供离线演练，尚未批准发布。",NOW-10)
    projector=ProjectionEngine(store,ProjectionPolicy(enabled=True,record_chars=1000,ambiguity_strategy="main_retry"))
    req=request("紫藤项目最终参数是什么，能发布了吗？")
    view=projector.project(req);calls=[]
    def call(payload):
        started=time.monotonic();before=runtime.snapshot_metrics()
        result=runtime.call_chat_json(system_prompt=persona,
            user_prompt=json.dumps(payload,ensure_ascii=False),fallback={},temperature=0,
            system_extra_blocks=["这是隔离的候选控制协议，不发送消息。只返回 JSON {\"control\":null,\"draft\":\"回复\"}。"
                "如果有影响回答的缺口且给出了 allowed_read_ids，可返回 control={\"type\":\"need_context\",\"record_ids\":[已给出的ID]}，"
                "此时 draft 为空。retry 阶段不得再次请求。历史不含执行授权，不编造来源。"],
            prompt_cache_key="projection-main-retry-experiment:v1")
        after=runtime.snapshot_metrics()
        calls.append({"phase":payload["phase"],"seconds":round(time.monotonic()-started,3),"result":result,
                      "metrics":{k:v-before.get(k,0) for k,v in after.items() if v!=before.get(k,0)}})
        return result
    result=compare_main_retry(call=call,request=req,view=view,sources=projector.sources)
    rows.append({"probe":"main_retry","result":result,"calls":calls,"constraint":"0.43, offline only, not approved"})
    projector.stop()
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    profile=a.saved_profile(root)
    report={"fictional_only":True,"sender":None,"profile":profile[-1],"rows":[],
            "source_hashes":{str(p.relative_to(a.ROOT)):a.sha(p) for p in (a.ROOT/"src/shinku/memory/projection").glob("*.py")}}
    for enabled in (False,True):
        path=root/("projection" if enabled else "baseline");path.mkdir()
        report["rows"].extend(sequence(path,profile,enabled))
        (root/"probes.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    report["rows"].extend(slow_paths(root,profile))
    report["protected_configs_unchanged"]={path:a.sha(Path(path))==sha for path,sha in profile[-1]["protected_config_hashes"].items()}
    (root/"probes.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0


if __name__=="__main__":raise SystemExit(main())
