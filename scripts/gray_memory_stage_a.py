"""Frozen fictional baseline through the existing Shinku memory/QQ reply path.

No QQ connection, production memory reads/writes, configuration saves, downloads,
or new memory features. Only memory_search is registered; delivery is a local sink.
--live uses the saved chat model. Oracle runs are labelled reading diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "private/src")]

from shinku.memory import MemoryContext, MemoryDecision, MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.embedding_factory import build_embedding_provider
from shinku.memory.router import LLMMemoryJudge
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

FIXTURE = ROOT / "tests/fixtures/memory_stage_a_v1.json"
OWNER = "fictional-owner"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(text: str) -> int:
    value = datetime.fromisoformat(text)
    if value.utcoffset() is None:
        raise ValueError("fixture_time_requires_timezone")
    return int(value.timestamp())


def load_cases(path: Path = FIXTURE) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source_kind") != "fictional_only":
        raise ValueError("only_fictional_fixtures_allowed")
    timestamp(data["question_date"])
    seen: set[str] = set()
    categories: dict[str, set[str]] = {}
    for case in data["cases"]:
        if case["id"] in seen or case["split"] not in {"dev", "holdout"}:
            raise ValueError("fixture_id_or_split_invalid")
        seen.add(case["id"])
        categories.setdefault(case["category"], set()).add(case["split"])
        sources = {item["id"] for item in case["history"]}
        if len(sources) != len(case["history"]) or not set(case["required_sources"]) <= sources:
            raise ValueError("fixture_evidence_invalid")
        for item in case["history"]:
            timestamp(item["at"])
        for rule in case["answer_rule"].get("all", []) + case["answer_rule"].get("forbidden", []):
            re.compile(rule)
    if any(splits != {"dev", "holdout"} for splits in categories.values()):
        raise ValueError("each_category_requires_both_splits")
    return data


def score_answer(text: str, rule: dict[str, Any]) -> dict[str, Any]:
    missing = [pattern for pattern in rule.get("all", []) if not re.search(pattern, text, re.I)]
    forbidden = [pattern for pattern in rule.get("forbidden", []) if re.search(pattern, text, re.I)]
    return {"rule_pass": bool(text.strip()) and not missing and not forbidden,
            "missing_patterns": missing, "forbidden_patterns": forbidden,
            "semantic_review_required": True}


def failure_layer(reply: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    """Do not mistake transport/host failures for wrong memory reasoning."""
    failed_calls = [call for call in calls if call.get("metrics_delta", {}).get("errors", 0) > 0]
    if failed_calls and (str(reply.get("reason") or "").startswith("chat_") or
                         "刚才那句我没接好" in str(reply.get("final_text") or "")):
        return "model_transport_or_runtime_failure"
    if str(reply.get("reason") or "").startswith("task_contract_error:"):
        return "host_task_contract_blocked"
    if reply.get("reason") == "reported_failure":
        return "model_reported_failure_requires_semantic_review"
    if reply.get("agent_status") == "failed":
        return "host_execution_failure_requires_review"
    return ""


class ObservedRuntime:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def call_chat_json(self, **kwargs: Any) -> Any:
        before = self.runtime.snapshot_metrics()
        started = time.monotonic()
        try:
            result = self.runtime.call_chat_json(**kwargs)
            return result
        finally:
            after = self.runtime.snapshot_metrics()
            error = self.runtime.snapshot_last_error()
            self.calls.append({"seconds": round(time.monotonic() - started, 3),
                "user_prompt_chars": len(str(kwargs.get("user_prompt") or "")),
                "evidence_source_ids": sorted(set(re.findall(r"source=([^）\s]+)",
                    str(kwargs.get("user_prompt") or "") + "\n" + "\n".join(
                        str(item.get("content") or "") for item in kwargs.get("history_turns") or []
                        if isinstance(item, dict))))),
                "metrics_delta": {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)},
                "runtime_error": {k: error[k] for k in ("type", "phase") if k in error}})


class ObservedMemory(MemoryService):
    def __init__(self, *, oracle_ids: set[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.contexts: list[dict[str, Any]] = []
        self.tool_reads: list[dict[str, Any]] = []
        self.source_aliases: dict[str, str] = {}
        self.record_aliases: dict[str, str] = {}
        self.oracle_ids = oracle_ids

    def canonical(self, record: dict[str, Any]) -> str:
        source = str(record.get("source_id") or "")
        return self.source_aliases.get(source, self.record_aliases.get(str(record.get("record_id") or ""), source))

    def prepare_turn(self, **kwargs: Any) -> MemoryContext:
        if self.oracle_ids is None:
            result = super().prepare_turn(**kwargs)
        else:
            scope = self.scope_for(kwargs["conversation_id"], kwargs["conversation_type"])
            owner_request = self._is_master_request(conversation_id=kwargs["conversation_id"],
                conversation_type=kwargs["conversation_type"], message_meta=kwargs.get("message_meta"))
            scopes = self._history_scopes(scope, owner_request=owner_request, mode="retrieve")
            records = [record for target_scope in scopes for record in self.store.timeline(
                scope=target_scope, now=int(self.clock()), limit=200, include_facts=True)]
            unique = {}
            for record in records:
                alias = self.canonical(record.as_dict())
                if alias in self.oracle_ids:
                    unique.setdefault(alias, record)
            selected = tuple(unique.values())
            selected = tuple(sorted(selected, key=lambda r: (r.created_at, r.source_id)))
            rendered = self._render(selected)
            result = MemoryContext(MemoryDecision(True, query=kwargs["user_message"],
                reason="fixture_oracle_reading_diagnostic", history_mode="retrieve"),
                selected, rendered, len(selected), len(rendered))
        self.contexts.append(result.as_dict())
        return result

    def inspect(self, **kwargs: Any) -> dict[str, Any]:
        result = super().inspect(**kwargs)
        self.tool_reads.append(result)
        return result


def make_service(path: Path, policy: MemoryPolicy, provider: Any, status: str, runtime: Any,
                 judge_enabled: bool, clock: list[int], oracle_ids: set[str] | None = None) -> ObservedMemory:
    store = MemoryStore(path, embedding_provider=provider, semantic_enabled=policy.semantic_enabled,
        semantic_min_score=policy.semantic_min_score, semantic_backfill_batch_size=policy.semantic_backfill_batch_size,
        embedding_status=status)
    judge = LLMMemoryJudge(runtime, recent_context_chars=policy.recent_context_chars) if runtime and judge_enabled else None
    return ObservedMemory(store=store, router=MemoryRouter(judge=judge,
        deterministic_guards=policy.deterministic_guards, max_query_chars=policy.max_query_chars),
        policy=policy, clock=lambda: clock[0], oracle_ids=oracle_ids)


def seed(service: ObservedMemory, case: dict[str, Any], clock: list[int], question_time: int) -> list[dict[str, Any]]:
    group = case.get("conversation_type", "private") == "group"
    conversation = "fictional-group" if group else OWNER
    kind = "group" if group else "private"
    history = list(case["history"])
    for i in range(6):
        history.append({"id": f"noise-{i}", "text": f"虚构无关记录{i}：今天整理文件夹和草稿，没有新的决定。",
                        "at": "2026-09-30T15:00:00+08:00"})
    for item in sorted(history, key=lambda row: timestamp(row["at"])):
        clock[0] = timestamp(item["at"])
        source = f"fixture:{case['id']}:{item['id']}"
        service.source_aliases[source] = item["id"]
        service.source_aliases[f"owner:{source}"] = item["id"]
        target_kind = item.get("conversation_type", kind)
        target_conversation = item.get("conversation_id", conversation)
        actor = item.get("actor", OWNER)
        metadata = {"message_id": item["id"], "sender_id": actor,
            "sender_name": "测试主人" if actor == OWNER else "测试群友", "requester_id": OWNER,
            "shinku_involved": actor == OWNER, "created_at": clock[0]}
        if "layer" in item:
            service.store.record_event(conversation_id=target_conversation,
                scope=service.scope_for(target_conversation, target_kind), text=item["text"], source_id=source,
                layer=item["layer"], created_at=clock[0], metadata=metadata)
        else:
            service.record_turn(conversation_id=target_conversation, conversation_type=target_kind,
                message_id=item["id"], source_id=source, text=item["text"],
                role=item.get("role", "user"), metadata=metadata)
    clock[0] = question_time
    service.store.backfill_embeddings(limit=128)
    scopes = {service.scope_for(conversation, kind), service.owner_scope, f"private:{OWNER}"}
    for scope in scopes:
        for record in service.store.timeline(scope=scope, now=question_time, limit=200, include_facts=True):
            alias = service.canonical(record.as_dict())
            service.record_aliases[record.record_id] = alias
    return apply_actions(service, case, clock, conversation, kind)


def apply_actions(service: ObservedMemory, case: dict[str, Any], clock: list[int], conversation: str, kind: str) -> list[dict[str, Any]]:
    results = []
    scope = service.scope_for(conversation, kind)
    for action in case.get("actions", []):
        operation = action["type"]
        if operation == "correct_fact":
            records = service.store.timeline(scope=scope, now=clock[0], limit=200, include_facts=True)
            facts = [r for r in records if r.layer == "fact" and service.canonical(r.as_dict()) == action["target"]]
            if not facts:
                raise ValueError("fixture_fact_not_found")
            changed = []
            for target_scope in (scope, service.owner_scope):
                for record in service.store.timeline(scope=target_scope, now=clock[0], limit=200, include_facts=True):
                    if record.layer == "fact" and service.canonical(record.as_dict()) == action["target"]:
                        changed.append(service.store.lifecycle.correct_fact(record_id=record.record_id,
                            scope=target_scope, expected_hash=hashlib.sha256(record.text.encode()).hexdigest(),
                            replacement=action["replacement"], now=clock[0], confirmed=True))
            if not changed or not all(changed):
                raise ValueError("fixture_correction_failed")
            results.append({"type": operation, "confirmed": True, "changed_count": len(changed)})
        elif operation == "forget":
            result = service.record_turn(conversation_id=conversation, conversation_type=kind,
                message_id="fixture-forget", text=f"忘记{action['query']}", metadata={"sender_id": OWNER})
            results.append({"type": operation, "result": result.as_dict() if result else {}})
        elif operation == "backfill":
            results.append({"type": operation, "written": service.store.backfill_embeddings(limit=128)})
        elif operation == "drop_vectors":
            with service.store._connect() as conn:
                conn.execute("DELETE FROM memory_embeddings")
            provider = service.store.embedding_provider
            if provider:
                provider.clear_cache()
            results.append({"type": operation, "isolated_only": True})
        elif operation == "restore_after_forget":
            backup = service.store.path.parent / "isolated-backup.sqlite3"
            with closing(sqlite3.connect(service.store.path)) as old, closing(sqlite3.connect(backup)) as destination:
                old.backup(destination)
            service.record_turn(conversation_id=conversation, conversation_type=kind,
                message_id="fixture-forget", text=f"忘记{action['query']}", metadata={"sender_id": OWNER})
            revocations = [service.store.lifecycle.export_revocations(scope=s) for s in (scope, service.owner_scope)]
            with closing(sqlite3.connect(backup)) as old, closing(sqlite3.connect(service.store.path)) as destination:
                old.backup(destination)
            outcomes = [service.store.lifecycle.import_revocations(value, approved_scope=value["scope"], confirmed=True)
                        for value in revocations]
            results.append({"type": operation, "isolated_only": True, "revocation_imports": outcomes})
        else:
            raise ValueError("fixture_action_not_allowed")
    return results


def run_case(case: dict[str, Any], *, run_root: Path, policy: MemoryPolicy, build: Any,
             runtime: Any, config: Any, persona: str, question_time: int,
             permission_mode: str = "traditional", oracle: bool = False) -> dict[str, Any]:
    case_root = run_root / ("oracle-" + case["id"] if oracle else case["id"])
    case_root.mkdir(exist_ok=False)
    setup_started = time.monotonic()
    clock = [question_time]
    observed = ObservedRuntime(runtime) if runtime else None
    memory = make_service(case_root / "memory.sqlite3", policy, build.provider, build.status,
        observed, bool(config and config.memory_judge_enabled), clock,
        set(case["required_sources"]) if oracle else None)
    actions = seed(memory, case, clock, question_time)
    kind = case.get("conversation_type", "private")
    conversation = "fictional-group" if kind == "group" else OWNER
    requester = case.get("requester", OWNER)
    event = {"post_type": "message", "message_type": kind, "user_id": requester,
        "group_id": "fictional-group", "message_id": f"query-{case['id']}", "time": question_time,
        "sender": {"nickname": "测试主人" if requester == OWNER else "测试群友"},
        "message": [{"type": "text", "data": {"text": case["question"]}}]}
    message = IncomingMessage.from_event(event)
    turn = QQTurn(conversation_id=conversation, conversation_type=kind,
        primary_message_id=message.message_id, message_ids=(message.message_id,),
        combined_text=case["question"], messages=(message,))
    available = set()
    for scope in {memory.scope_for(conversation, kind), memory.owner_scope}:
        for record in memory.store.timeline(scope=scope, now=question_time, limit=200, include_facts=True):
            available.add(memory.canonical(record.as_dict()))
    setup_seconds = round(time.monotonic() - setup_started, 3)
    started = time.monotonic()
    if observed:
        registry = ToolRegistry({"memory_search": MemorySearchTool(memory)})
        bridge = QQAgentBridge(runtime=observed, system_prompt=persona, memory_service=memory,
            tool_host=ToolHost(registry), allowed_tool_names={"memory_search"},
            policy=ExecutionPolicy(risk_by_tool={"memory_search": "low"}),
            master_user_ids=(OWNER,), owner_gate_required=True,
            command_state=QQCommandState(permission_mode=permission_mode, owner_ids=(OWNER,)),
            temperature=config.temperature, max_rounds=config.max_rounds,
            history_turns=config.history_turns, recent_context_messages=config.recent_context_messages,
            prompt_cache_key="shinku:memory-stage-a:v1", task_contract_enabled=True,
            sender=lambda _: DeliveryResult(True, "isolated-local-sink"))
        reply = bridge.handle_turn(turn).as_dict()
    else:
        context = memory.prepare_turn(conversation_id=conversation, conversation_type=kind,
            user_message=case["question"], recent_context="", message_meta={"sender_id": requester})
        reply = {"agent_status": "not_run", "final_text": "", "reason": "offline_retrieval_only"}
    elapsed = round(time.monotonic() - started, 3)
    evidence_records = [record for context in memory.contexts + memory.tool_reads for record in context.get("records", [])]
    retrieved = sorted({memory.canonical(record) for record in evidence_records})
    rendered = "\n".join(str(context.get("rendered") or "") for context in memory.contexts + memory.tool_reads)
    delivered_ids = set(source for call in observed.calls for source in call["evidence_source_ids"]) if observed else set(
        re.findall(r"source=([^）\s]+)", rendered))
    visible_sources = sorted({memory.canonical(record) for record in evidence_records if record.get("source_id") in delivered_ids})
    required = set(case["required_sources"])
    all_required = bool(required) and required <= set(retrieved)
    leaks = [text for text in case.get("forbidden_evidence", []) if text in rendered]
    score = score_answer(str(reply["final_text"]), case["answer_rule"]) if runtime else None
    record = {"id": case["id"], "split": case["split"], "category": case["category"], "oracle": oracle,
        "question": case["question"], "rubric": case["rubric"], "history": case["history"],
        "actions": actions, "required_sources": sorted(required), "available_sources": sorted(available),
        "retrieved_sources": retrieved, "all_required_sources_retrieved": all_required if required else None,
        "visible_sources": visible_sources, "all_required_sources_visible": required <= set(visible_sources) if required else None,
        "evidence_leaks": leaks, "memory_contexts": memory.contexts, "tool_reads": memory.tool_reads,
        "reply": reply, "answer_check": score, "seconds": elapsed,
        "model_calls": observed.calls if observed else [], "injected_chars": len(rendered)}
    if not runtime:
        attribution = "retrieval_only_qa_not_run"
    elif leaks:
        attribution = "access_or_revocation_leak"
    elif not score["rule_pass"]:
        if required and not required <= available:
            attribution = "source_unavailable"
        elif required and not all_required:
            mode = memory.contexts[0]["decision"]["history_mode"] if memory.contexts else "unknown"
            attribution = "route_skipped" if mode == "none" else "retrieval_incomplete"
        else:
            attribution = "reading_or_abstention_requires_review"
    elif required and not all_required:
        attribution = "answer_rule_pass_without_complete_memory_evidence"
    else:
        attribution = "rule_pass_requires_semantic_review"
    record["attribution"] = attribution
    record["execution_failure_layer"] = failure_layer(reply, record["model_calls"])
    if record["execution_failure_layer"]:
        record["attribution"] = record["execution_failure_layer"]
    record["setup_and_index_seconds"] = setup_seconds
    memory.lifecycle_review.stop()
    memory.store.stop_embedding_worker()
    memory.store.lifecycle.stop_worker()
    return record


def saved_profile(run_root: Path) -> tuple[Any, Any, MemoryPolicy, Any, str, dict[str, Any]]:
    from shinku.api.model_runtime import RuntimeModelServiceConfig, RuntimeModelServiceController
    from shinku.llm.runtime import LLMRuntime
    from shinku.providers.catalog import ModelServiceConfigStore
    from shinku.qq.assembly import load_qq_agent_config
    from shinku_private.extension import augment_system_prompt
    spec = importlib.util.spec_from_file_location("memory_stage_a_launch", ROOT / "private/launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    env = dict(os.environ)
    launch._load_private_env(env)
    os.environ.update(env)
    catalog_path = Path(env.get("SHINKU_MODEL_SERVICES_PATH") or ROOT / "private/data/model_services.json")
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    protected = [ROOT / "private/config/.env.private", catalog_path]
    config = load_qq_agent_config()
    # The actual CLI supplies this canonical path when the QQ config omits it.
    # Mirror that read-only derivation rather than mistaking an empty path for
    # the user's live permission mode (the constructor default is traditional).
    state_path = Path(config.command_state_path) if config.command_state_path else Path(
        env.get("SHINKU_DATA_ROOT") or ROOT / "private/data") / "qq/command_state.json"
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    protected.append(state_path)
    hashes = {str(p): sha(p) for p in protected if p.is_file()}
    state = QQCommandState.from_environment(state_path=state_path)
    permission = state.permission_mode()
    RuntimeModelServiceController(store=ModelServiceConfigStore(catalog_path, environ=os.environ),
        config_module=RuntimeModelServiceConfig.from_agent_config(config)).load_saved()
    policy = MemoryPolicy.from_environment()
    cache = Path(policy.embedding_cache_dir or ROOT / "private/data/memory/models/bge-small-zh-v1.5")
    if not cache.is_absolute():
        cache = ROOT / cache
    policy = replace(policy, master_user_ids=(OWNER,), embedding_device="cpu",
        embedding_local_files_only=True, embedding_cache_dir=str(cache))
    for key in ("SHINKU_DATA_ROOT", "SHINKU_PRIVATE_DATA_ROOT", "SHINKU_LOG_ROOT", "SHINKU_LOG_DIR"):
        os.environ[key] = str(run_root / "isolated-runtime")
    os.environ["SHINKU_MEMORY_MASTER_USER_IDS"] = OWNER
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SHINKU_LLM_HTTP_ATTEMPTS"] = "1"
    # Diagnostic transport bounds only; saved model/profile/config remain untouched.
    os.environ["SHINKU_LLM_CHAT_TIMEOUT_SECONDS"] = "120"
    os.environ["SHINKU_LLM_AUX_TIMEOUT_SECONDS"] = "30"
    runtime = LLMRuntime(metrics_path=run_root / "metrics.sqlite3")
    # Do not leave the private system-persona plaintext in report artifacts.
    with tempfile.TemporaryDirectory(prefix="shinku-stage-a-persona-") as persona_scratch:
        persona_path = launch._compose_persona({**env, "SHINKU_PRIVATE_DATA_ROOT": persona_scratch})
        persona = augment_system_prompt(prompt=persona_path.read_text(encoding="utf-8"), config=config)
    build = build_embedding_provider(policy, default_cache_dir=cache)
    frozen = {"chat_model": runtime.chat.model, "aux_model": runtime.aux.model,
        "temperature": config.temperature, "max_rounds": config.max_rounds,
        "permission_mode": permission, "memory_judge_enabled": config.memory_judge_enabled,
        "command_state_path_source": "explicit_config" if config.command_state_path else "canonical_cli_default",
        "history_turns": config.history_turns, "recent_context_messages": config.recent_context_messages,
        "memory_policy": {**asdict(policy), "master_user_ids": [OWNER]},
        "embedding_status": build.status, "embedding_error_type": build.error.split(":", 1)[0][:80] if build.error else "",
        "persona_sha256": hashlib.sha256(persona.encode()).hexdigest(), "persona_chars": len(persona),
        "protected_config_hashes": hashes,
        "diagnostic_differences": ["fictional owner/scopes", "CPU embeddings; cached model only",
            "synchronous index warmup; no background workers", "only memory_search tool",
            "local delivery sink; no QQ transport", "120s chat/30s AUX timeouts; one HTTP attempt",
            "isolated logs/metrics/persona composition", "fixed fictional question clock"],
        "service_state_kind": "saved_profile_not_a_live_backend_snapshot"}
    return runtime, config, policy, build, persona, frozen


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("dev", "holdout", "all"):
        selected = [r for r in rows if split == "all" or r["split"] == split]
        positives = [r for r in selected if r["required_sources"]]
        checked = [r for r in selected if r["answer_check"] is not None]
        result[split] = {"cases": len(selected), "required_evidence_cases": len(positives),
            "complete_evidence_cases": sum(r["all_required_sources_retrieved"] for r in positives),
            "visible_complete_evidence_cases": sum(r["all_required_sources_visible"] for r in positives),
            "qa_rule_pass": sum(r["answer_check"]["rule_pass"] for r in checked), "qa_checked": len(checked),
            "evidence_leak_cases": sum(bool(r["evidence_leaks"]) for r in selected),
            "model_calls": sum(len(r["model_calls"]) for r in selected),
            "median_seconds": round(statistics.median([r["seconds"] for r in selected]), 3) if selected else 0,
            "attributions": dict(Counter(r["attribution"] for r in selected))}
    return result


def render_report(report: dict[str, Any]) -> str:
    lines = ["# 阶段 A：真红记忆现状基线", "", f"运行 ID：{report['run_id']}", "",
        "仅虚构语料、隔离数据库与本地投递。保存配置对应的真实模型回答；不是线上 QQ 效果证明。",
        "关键词规则分数只是初筛，不能等同语义正确率；逐案复核另行记录。", "",
        "## 固定条件", "", f"- 语料版本：{report['fixture_version']}；SHA-256：`{report['fixture_sha256']}`。",
        f"- 模式：{'真实模型回复' if report['live'] else '仅离线检索，没有模型回答'}。",
        f"- 配置未改写：{report.get('protected_configs_unchanged', '待运行结束检查')}。", ""]
    if report.get("profile"):
        p = report["profile"]
        lines += [f"- Chat：`{p['chat_model']}`；温度：{p['temperature']}；权限档位：`{p['permission_mode']}`。",
            f"- 本地 embedding：`{p['memory_policy']['embedding_model']}`，状态 `{p['embedding_status']}`。",
            "- 与生产的隔离差异：" + "；".join(p["diagnostic_differences"]) + "。", ""]
    lines += ["## 分组结果", "", "| 分组 | 题数 | 证据取齐／需要证据 | 回答规则通过／已测 | 证据泄漏案例 | 模型调用 | 中位秒数 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for split, row in summarize(report["cases"]).items():
        lines.append(f"| {split} | {row['cases']} | {row['complete_evidence_cases']}/{row['required_evidence_cases']} | {row['qa_rule_pass']}/{row['qa_checked']} | {row['evidence_leak_cases']} | {row['model_calls']} | {row['median_seconds']} |")
    lines += ["", "## 实际问答与来源", ""]
    for row in report["cases"]:
        lines += [f"### {row['id']} · {row['category']} · {row['split']}", "",
            f"问题：{row['question']}", "", "实际回答：", ""]
        lines += ["> " + line for line in (row["reply"]["final_text"] or "【未运行模型或没有最终回答】").splitlines()]
        lines += ["", f"核对标准：{row['rubric']}", "",
            f"所需来源：{', '.join(row['required_sources']) or '无答案证据／控制题'}；实际命中：{', '.join(row['retrieved_sources']) or '无'}；模型输入可见来源：{', '.join(row['visible_sources']) or '无'}。",
            f"初步归因：`{row['attribution']}`；耗时 {row['seconds']} 秒。", "", "演示历史：", ""]
        for item in row["history"]:
            lines.append(f"- `{item['id']}` · {item['at']} · {item.get('role', 'user')} / {item.get('actor', OWNER)}：{item['text']}")
        if row["actions"]:
            lines += ["", "隔离操作：`" + ", ".join(a["type"] for a in row["actions"]) + "`。"]
        lines += [""]
    if report.get("oracle_cases"):
        lines += ["## 正确来源直接提供的阅读诊断", "", "这不是改进后成绩，也不合并进基线。仅对选定问题直接提供固定正确来源。", ""]
        for row in report["oracle_cases"]:
            lines += [f"### {row['id']}", "", "> " + row["reply"]["final_text"].replace("\n", "\n> "), "",
                f"规则通过：{row['answer_check']['rule_pass']}；所需来源取齐：{row['all_required_sources_retrieved']}。", ""]
    if report.get("lifecycle_replay"):
        replay = report["lifecycle_replay"]
        lines += ["## 既有生命周期隔离回放", "", f"通过 {replay['passed']}/{replay['total']}；独立工程检查，不当作模型问答成绩。", ""]
        for row in replay["cases"]:
            lines.append(f"- `{row.get('case', row.get('name', 'unknown'))}`：{row['passed']}。")
    lines += ["", "## 限制", "", "- 小型合成语料，不代表真实群聊、长期质量或零风险。",
        "- 已索引的暖状态为主；模型有随机性，同输入可复跑但不保证逐字一致。",
        "- 留出集未用于调整生产规则；当前运行是初始基线，后续阶段需固定同一配置对照。",
        "- 无独立人类盲评；需要将规则初筛与本助手的逐案复核分开报告。",
        "- 结构化 JSON 包含检索记录、渲染证据、调用计数与复现元数据，不包含密钥或隐藏思维。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--oracle-limit", type=int, default=6)
    parser.add_argument("--oracle-only", action="store_true",
                        help="Only direct-evidence reading diagnostics; never count as normal baseline")
    args = parser.parse_args()
    if not 0 <= args.oracle_limit <= 8:
        parser.error("oracle limit must be 0..8")
    fixture_path = args.fixture.resolve(strict=True)
    if fixture_path.parent != (ROOT / "tests/fixtures").resolve():
        parser.error("fixture must be in this repository's tests/fixtures")
    data = load_cases(fixture_path)
    cases = [r for r in data["cases"] if not args.case_id or r["id"] in args.case_id]
    if not cases or (args.case_id and set(args.case_id) - {r["id"] for r in cases}):
        parser.error("unknown or empty case selection")
    if args.oracle_only and (not args.live or not args.case_id or any(not r["required_sources"] for r in cases)):
        parser.error("oracle-only requires live, explicit case ids and positive evidence")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "docs" / f"MEMORY_STAGE_A_{run_id}"
    run_root.mkdir(exist_ok=False)
    output = run_root / "baseline.json"
    report = {"run_id": run_id, "live": args.live, "fixture_version": data["version"],
        "selected_case_ids": [case["id"] for case in cases], "full_fixture_selected": len(cases) == len(data["cases"]),
        "run_kind": "oracle_reading_only" if args.oracle_only else "normal_baseline",
        "fixture_path": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": sha(fixture_path), "question_date": data["question_date"],
        "source_snapshot": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), fixture_path,
            ROOT / "src/shinku/memory/router.py", ROOT / "src/shinku/memory/store.py",
            ROOT / "src/shinku/memory/service.py", ROOT / "src/shinku/memory/lifecycle.py",
            ROOT / "src/shinku/qq/agent_bridge.py"]},
        "isolation": {"fictional_only": True, "production_memory_read": False, "production_memory_write": False,
            "qq_transport": False, "config_save": False, "model_download": False, "tools": ["memory_search"]},
        "cases": [], "oracle_cases": [], "status": "running"}
    runtime = config = None
    profile = {}
    try:
        if args.live:
            runtime, config, policy, build, persona, profile = saved_profile(run_root)
            report["profile"] = profile
            if policy.semantic_enabled and build.status != "huggingface":
                raise RuntimeError("saved_semantic_model_unavailable_no_silent_substitution")
        else:
            from types import SimpleNamespace
            policy = MemoryPolicy(master_user_ids=(OWNER,))
            build = SimpleNamespace(provider=None, status="disabled")
            persona = ""
        def save() -> None:
            report["summary"] = summarize(report["cases"])
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_root / "BASELINE.md").write_text(render_report(report), encoding="utf-8")
        save()
        print("REPORT " + str(run_root), flush=True)
        print("PROFILE " + json.dumps({k: profile.get(k) for k in ("chat_model", "permission_mode", "embedding_status")}), flush=True)
        for case in ([] if args.oracle_only else cases):
            print("START " + case["id"], flush=True)
            row = run_case(case, run_root=run_root, policy=policy, build=build, runtime=runtime,
                config=config, persona=persona, question_time=timestamp(data["question_date"]),
                permission_mode=profile.get("permission_mode", "traditional"))
            report["cases"].append(row)
            save()
            print("CASE " + json.dumps({"id": row["id"], "sources_complete": row["all_required_sources_retrieved"],
                "rule_pass": row["answer_check"]["rule_pass"] if row["answer_check"] else None,
                "seconds": row["seconds"], "attribution": row["attribution"]}, ensure_ascii=False), flush=True)
        if args.live:
            # Fixed bounded diagnostic selection, not tuning or replacement of failed baseline answers.
            selected = cases if args.oracle_only else [r for r in report["cases"] if r["required_sources"] and
                (not r["all_required_sources_retrieved"] or not r["answer_check"]["rule_pass"])][:args.oracle_limit]
            by_id = {case["id"]: case for case in cases}
            for original in selected:
                row = run_case(by_id[original["id"]], run_root=run_root, policy=policy, build=build,
                    runtime=runtime, config=config, persona=persona, question_time=timestamp(data["question_date"]),
                    permission_mode=profile["permission_mode"], oracle=True)
                report["oracle_cases"].append(row)
                save()
                print("ORACLE " + json.dumps({"id": row["id"], "rule_pass": row["answer_check"]["rule_pass"]}), flush=True)
        from gray_memory_lifecycle import replay
        report["lifecycle_replay"] = replay()
        report["protected_configs_unchanged"] = all(Path(p).is_file() and sha(Path(p)) == value
            for p, value in profile.get("protected_config_hashes", {}).items())
        report["status"] = "completed"  # completion of measurement, not correctness of every case
        if runtime:
            report["metrics"] = runtime.snapshot_metrics()
        save()
        print("SUMMARY " + json.dumps(report["summary"], ensure_ascii=False), flush=True)
        return 0 if report["protected_configs_unchanged"] else 2
    except Exception as exc:
        report["status"] = "harness_error"
        report["error_type"] = type(exc).__name__
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "BASELINE.md").write_text(render_report(report), encoding="utf-8")
        print("HARNESS_ERROR " + type(exc).__name__, flush=True)
        raise
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
