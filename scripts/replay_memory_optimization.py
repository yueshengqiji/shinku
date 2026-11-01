"""Offline synthetic replay through real memory and QQ prompt adapters.

No response model is called. Expected answers are only used after projection.
Local embedding weights may be loaded explicitly, with socket access blocked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(store, case, now, owner, scope):
    aliases = {}
    previous_user = ""
    for row in case["history"]:
        role = row.get("role", "user")
        row_scope = row.get("scope", scope)
        mid = row.get("reply_to", previous_user) if role == "assistant" else row["id"]
        mid = mid or row["id"]
        meta = {"role":role, "sender_id":row.get("sender", "bot" if role == "assistant" else owner),
                "self_id":"bot", "message_id":mid}
        if role == "assistant":
            meta.update(requester_id=owner, delivery_status=row.get("delivery_status", "sent"),
                        delivery_message_ids=["synthetic-receipt-"+row["id"]],
                        delivery_content_complete=row.get("delivery_content_complete", True))
        else:
            previous_user = row["id"]
        source = "turn:"+row_scope.split(":",1)[1]+":"+mid+(":assistant" if role=="assistant" else "")
        store.record_event(conversation_id=row_scope.split(":",1)[1], scope=row_scope, text=row["text"],
                           source_id=source, created_at=now-row.get("age_seconds", 40*86400), metadata=meta)
        record = store.event_by_source_id(source_id=source, scope=row_scope)
        aliases[record.record_id] = row["id"]
    with store._connect() as conn:
        conn.executemany("""INSERT INTO memory_events
            (event_id,conversation_id,scope,text,layer,created_at,source_id,metadata_json)
            VALUES(?,?,?,?,?,?,?,?)""", [
            (f"noise-{i}",scope.split(":",1)[1],scope,f"虚构天气与读书日志条目{i}","episodic",now-4000+i,
             f"synthetic-noise:{i}",json.dumps({"role":"user","sender_id":owner}))
            for i in range(case.get("noise",0))])
    return aliases


def run_case(case, root, now, provider, strategy, explicit_probe=False, recorded_time=None):
    from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
    from shinku.memory.projection.contracts import ProjectionPolicy
    from shinku.agent.loop import AgentState
    from shinku.agent_runtime.execution import ExecutionProfile
    from shinku.qq.agent_bridge import QQAgentBridge
    from shinku.qq.message import IncomingMessage
    from shinku.qq.turns import QQTurn
    from shinku.tasks.coordinator import AgentTaskCoordinator

    owner = "fictional-owner"
    kind = case.get("conversation_type", "group" if case.get("fault")=="tool_message_flood" else "private")
    conversation = "fixture-group" if kind == "group" else owner
    scope = kind+":"+conversation
    options = dict(enabled=True, semantic_enabled=False)
    if strategy != "current":
        options["retrieval_strategy"] = strategy
    policy = MemoryPolicy(master_user_ids=(), event_ttl_seconds=0, projection=ProjectionPolicy(**options))
    store = MemoryStore(root/"memory.sqlite3", embedding_provider=provider, semantic_enabled=provider is not None)
    aliases = seed(store, case, now, owner, scope)
    if provider:
        for _ in range(100):
            if not store.backfill_embeddings(limit=256):
                break
        else:
            raise RuntimeError("fixture_backfill_exceeded")
        if case.get("vector_state") in {"partial","other_model"}:
            with store._connect() as conn:
                if case["vector_state"] == "partial":
                    conn.execute("DELETE FROM memory_embeddings WHERE record_id=?",
                                 (next(iter(aliases)).removeprefix("event:"),))
                else:
                    conn.execute("UPDATE memory_embeddings SET model_key='fixture_other_model'")
    ticks = [now]
    service = MemoryService(store=store, router=MemoryRouter(), policy=policy, clock=lambda:ticks[0])
    projector = service.projection
    for _ in range(100):
        if projector.archive.maintain()["state"] == "ready":
            break
    else:
        raise RuntimeError("fixture_maintenance_exceeded")
    if case.get("index_state") == "missing":
        with store._connect() as conn:
            conn.execute("DROP TABLE memory_projection_fts")
    if case.get("mutation") == "forget":
        for rid in aliases:
            store.forget(scope=scope, record_id=rid, query="", now=now)
    if case.get("restart"):
        projector.stop()
        store.stop_embedding_worker()
        store = MemoryStore(root/"memory.sqlite3", embedding_provider=provider, semantic_enabled=provider is not None)
        service = MemoryService(store=store, router=MemoryRouter(), policy=policy, clock=lambda:ticks[0])
        projector = service.projection

    calls = []
    def no_model(**kwargs):
        calls.append("unexpected_response_model")
        raise AssertionError("offline_replay_must_not_call_response_models")
    runtime = SimpleNamespace(call_chat_json=no_model)
    coordinator = AgentTaskCoordinator(max_workers=1) if case.get("entry")=="task" else None
    bridge = QQAgentBridge(runtime=runtime, system_prompt="Synthetic offline prompt adapter.", memory_service=service,
        sender=None, tool_decision_mode=case.get("entry", "legacy"),
        task_coordinator=coordinator, task_contract_enabled=coordinator is not None,
        execution_profile=ExecutionProfile(SimpleNamespace(call_chat_json=no_model)) if case.get("entry")=="task" else None)
    meta = {"sender_id":owner,"self_id":"bot","message_id":"current"}
    message = IncomingMessage.from_event({"post_type":"message","message_type":kind,"user_id":owner,
        "group_id":conversation if kind=="group" else None,"self_id":"bot","message_id":"current","time":now,
        "message":[{"type":"text","data":{"text":case["query"]}}]})
    turn = QQTurn(conversation_id=conversation,conversation_type=kind,primary_message_id="current",
                  message_ids=("current",),combined_text=case["query"],messages=(message,))
    recent = "\n".join(row["text"] for row in case["history"] if row.get("age_seconds",40*86400)<1800
                       and row.get("scope",scope)==scope and row.get("delivery_status","sent") in {"sent","partial"})
    evidence = {}
    try:
        context = service.prepare_turn(conversation_id=conversation, conversation_type=kind,
            user_message=case["query"], recent_context=recent, message_meta=meta)
        fault = case.get("fault")
        if fault == "encoder_unavailable":
            # Inject only the encoder failure; discovery and source reads are real.
            def fail_vector(*args, **kwargs):
                return (), {"semantic":"encoder_failed"}
            projector.archive.semantic = fail_vector
        elif fault == "deadline":
            original_search = projector.archive.search
            def deadline_search(request, deadline, **kwargs):
                deadline.until = 0
                return original_search(request, deadline, **kwargs)
            projector.archive.search = deadline_search
        elif fault == "database_busy":
            def busy(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked (injected fixture failure)")
            projector.sources.read = busy
        elif fault == "fabricated_quote":
            from shinku.memory.projection.semantics import validate
            try:
                validate("distill",{"uncertain":False,"excerpts":[{"record_id":"invented","quote":"不存在"}]},
                         {"records":[{"record_id":"s0","text":case["history"][0]["text"]}]})
            except ValueError as exc:
                evidence["fabricated_quote_rejected"] = str(exc)
        mutation = case.get("mutation")
        if mutation in {"edit_after_read","forget_after_read"}:
            original_read = projector.sources.read
            changed = []
            def mutate_read(handles, request, deadline):
                rows = original_read(handles, request, deadline)
                row = next((r for r in rows if r),None)
                if row and not changed:
                    changed.append(row.record_id)
                    if mutation == "forget_after_read":
                        store.forget(scope=row.scope,record_id=row.record_id,query="",now=now)
                    else:
                        with store._connect() as conn:
                            conn.execute("UPDATE memory_events SET text=? WHERE event_id=?",
                                         ("隔离案例的新版本，旧句已失效。",row.record_id.removeprefix("event:")))
                return rows
            projector.sources.read = mutate_read
        state = AgentState(transcript=[{"role":"user","content":case["query"]}],context={
            "qq_turn":turn,"memory_context":context.as_dict(),"chat_flow":{},
            "conversation_turn_id":"synthetic-"+case["id"]})
        started = time.monotonic()
        prompt = bridge._build_prompt(state, include_agent_control=False)
        if fault == "tool_message_flood":
            for i in range(24):
                store.record_event(conversation_id=conversation,scope=scope,text=f"后来另一位群友的新话题{i}",
                    source_id=f"turn:{conversation}:late-{i}",created_at=now+i+1,
                    metadata={"role":"user","sender_id":"other","message_id":f"late-{i}"})
            ticks[0] = now+40
            prompt = bridge._build_prompt(state, include_agent_control=False)
            evidence["late_messages_injected"] = "后来另一位群友的新话题" in prompt.user_prompt
        elapsed = (time.monotonic()-started)*1000
        # The final prompt is the actual material seen by the reply model.
        records = []
        for line in prompt.user_prompt.splitlines():
            if line.startswith('{"record_id":'):
                records.append(json.loads(line))
        if explicit_probe:
            from shinku.memory import MemorySearchTool
            probe_started = time.monotonic()
            inspection = MemorySearchTool(service).execute(
                call={"query":case["query"],"mode":"retrieve","limit":12,
                      **({"recorded_time":recorded_time} if recorded_time is not None else {})},
                context={"qq_turn":turn})
            data = inspection["state_updates"]["memory_inspection"]
            records = [{**r,"attribution":r.get("metadata",{})} for r in data["records"]]
            evidence["explicit_probe"] = {"query_source":"unchanged_user_input", "model_selected_query":False,
                "recorded_time":recorded_time,
                "recorded_time_source":"fixture_control_not_model" if recorded_time is not None else "legacy_query",
                "coverage":data.get("coverage",{}), "error":data.get("error",""),
                "elapsed_ms":round((time.monotonic()-probe_started)*1000,3),
                "rendered_chars":len(data.get("rendered","")),
                "result_bytes":len(json.dumps(inspection,ensure_ascii=False).encode("utf-8"))}
        actual = [aliases.get(r["record_id"],r["record_id"]) for r in records if r["text"]]
        required = set(case["required_sources"])
        eligible = bool(required) and not case.get("fault") and not case.get("mutation")
        return {"id":case["id"],"category":case["category"],"split":case["split"],"entry":case.get("entry","legacy"),
            "state":"executed","eligible_recall":eligible,"required_sources":sorted(required),"actual_sources":actual,
            "recall_at_12":len(required&set(actual[:12]))/len(required) if eligible else None,
            "history_mode":context.decision.history_mode,"projection":service.projection_status,
            "prompt_ms":round(elapsed,3),"prompt_chars":len(prompt.user_prompt),"model_calls":len(calls),
            "records":records,"prompt":prompt.user_prompt,"fault_evidence":evidence,
            "answer_quality":"not_run", "measurement":"forced_explicit_read_capability" if explicit_probe else "automatic_projection","delivery_count":0,"policy":asdict(policy.projection)}
    finally:
        projector.stop()
        store.stop_embedding_worker()
        bridge.task_store.close()
        bridge.conversation_runtime.close()
        if coordinator:
            coordinator.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--source-root",default="src")
    parser.add_argument("--fixture",default="tests/fixtures/memory_optimization_v1.json")
    parser.add_argument("--split",choices=("dev","holdout","all"),default="dev")
    parser.add_argument("--strategy",default="current")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--explicit-probe",action="store_true")
    parser.add_argument("--recorded-start", default=None,
                        help="Inclusive record date YYYY-MM-DD, or empty for no lower bound; diagnostic control only")
    parser.add_argument("--recorded-end", default=None,
                        help="Exclusive record date YYYY-MM-DD, or empty for no upper bound; diagnostic control only")
    args = parser.parse_args()
    recorded_time = None
    if args.recorded_start is not None or args.recorded_end is not None:
        if args.recorded_start is None or args.recorded_end is None or not args.explicit_probe or args.strategy != "model":
            parser.error("both recorded date bounds require --explicit-probe --strategy model")
        recorded_time = {"start":args.recorded_start,"end":args.recorded_end}
    elif args.explicit_probe and args.strategy == "model":
        parser.error("model tool requires --recorded-start and --recorded-end; use empty bounds for an unbounded diagnostic")
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(args.source_root).resolve()))
    fixture = Path(args.fixture).resolve()
    data = json.loads(fixture.read_text(encoding="utf-8"))
    assert data["source_kind"] == "fictional_only"
    blocked = []
    def deny(*args, **kwargs):
        blocked.append("socket_connection_denied")
        raise PermissionError("offline_replay_no_network")
    socket.socket.connect = socket.socket.connect_ex = deny
    for key in list(os.environ):
        if key.startswith("SHINKU_"):
            del os.environ[key]
    for key in ("SHINKU_DATA_ROOT","SHINKU_PRIVATE_DATA_ROOT","SHINKU_LOG_ROOT","SHINKU_LOG_DIR","SHINKU_CONFIG_ROOT"):
        os.environ[key] = str(root)
    os.environ["SHINKU_ENV_FILE"] = str(root/"missing.env")
    os.environ["SHINKU_MODEL_SERVICES_PATH"] = str(root/"missing-model-services.json")
    os.environ["SHINKU_MEMORY_DB_PATH"] = str(root/"unused.sqlite3")
    provider = None
    if args.embedding_cache:
        from shinku.providers.huggingface import HuggingFaceEmbeddingProvider
        provider = HuggingFaceEmbeddingProvider(model_name="BAAI/bge-small-zh-v1.5",device="cpu",
            local_files_only=True,cache_folder=str(Path(args.embedding_cache).resolve()))
        provider.embed_text("离线查询预热")
    rows = []
    try:
        for case in data["cases"]:
            if args.split != "all" and args.split != case["split"]:
                continue
            case_root = root/case["id"]
            case_root.mkdir()
            try:
                row = run_case(case,case_root,data["now"],provider,args.strategy,args.explicit_probe,recorded_time)
            except Exception as exc:
                row = {"id":case["id"],"category":case["category"],"state":"failed",
                       "error_type":type(exc).__name__,"error":str(exc)[:500]}
            rows.append(row)
            (case_root/"result.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps({k:row[k] for k in ("id","state","recall_at_12","error_type") if k in row}),flush=True)
    finally:
        eligible = [r for r in rows if r.get("eligible_recall")]
        summary = {"fixture_sha256":digest(fixture),"source_root":str(Path(args.source_root).resolve()),
            "measurement":"forced_explicit_read_capability" if args.explicit_probe else "automatic_projection",
            "split":args.split,"strategy":args.strategy,"embedding_model":provider.collection_key() if provider else None,
            "recorded_time_control":recorded_time,"model_selected_time_filter":False,
            "network_connections_blocked":len(blocked),"response_model_calls":0,"production_writes":0,"qq_sends":0,
            "executed":sum(r["state"]=="executed" for r in rows),"failed":sum(r["state"]=="failed" for r in rows),
            "eligible_recall_cases":len(eligible),"mean_recall_at_12":sum(r["recall_at_12"] for r in eligible)/len(eligible) if eligible else None,
            "semantic_and_human_acceptance":"pending","rows":[{k:v for k,v in r.items() if k not in {"prompt","records"}} for r in rows]}
        (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return int(any(r["state"]=="failed" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
