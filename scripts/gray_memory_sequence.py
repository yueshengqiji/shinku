"""Isolated E consecutive turns through the real reply chain, never QQ transport.

Recent ordinary conversation and newly injected memory are measured separately.
The same bridge/service survive the sequence; confirmed corrections occur between
turns using the existing maintenance engine. No alternate planner or ideal reply.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import re
from types import SimpleNamespace
import time
import hashlib

import gray_memory_stage_a as a
import gray_memory_upgrade as upgrade

ROOT = a.ROOT
FIXTURE = ROOT / "tests/fixtures/memory_stage_e_sequence_v1.json"


class SequenceRuntime(a.ObservedRuntime):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.forbidden = []

    def call_chat_json(self, **kwargs):
        prompt = str(kwargs.get("user_prompt") or "")
        ordinary = "\n".join(str(row.get("content") or "") for row in kwargs.get("history_turns") or []
                             if isinstance(row, dict))
        before = len(self.calls)
        try:
            return super().call_chat_json(**kwargs)
        finally:
            if len(self.calls) > before:
                self.calls[-1].update({"current_prompt_forbidden_hits": [word for word in self.forbidden if word in prompt],
                    "recent_dialogue_forbidden_hits": [word for word in self.forbidden if word in ordinary],
                    "recent_dialogue_chars": len(ordinary)})


class SequenceBridge(a.QQAgentBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handoffs = []

    def _owner_handoff_context(self, *args, **kwargs):
        text = super()._owner_handoff_context(*args, **kwargs)
        self.handoffs.append(text)
        return text


def load_fixture():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if data.get("source_kind") != "fictional_only":
        raise ValueError("only_fictional_sequences_allowed")
    a.timestamp(data["question_date"])
    splits = set()
    ids = set()
    for case in data["cases"]:
        if case["id"] in ids or case["split"] not in {"dev", "holdout"}:
            raise ValueError("sequence_identity_invalid")
        ids.add(case["id"])
        splits.add(case["split"])
        history_ids = {row["id"] for row in case["history"]}
        for row in case["history"]:
            a.timestamp(row["at"])
        for step in case["steps"]:
            if not set(step["required_sources"]) <= history_ids:
                raise ValueError("sequence_required_source_unknown")
            for pattern in step["answer_rule"].get("all", []) + step["answer_rule"].get("forbidden", []):
                re.compile(pattern)
    if splits != {"dev", "holdout"}:
        raise ValueError("sequence_requires_dev_and_holdout")
    return data


def snapshot():
    paths = [Path(__file__), ROOT / "src/shinku/memory/workbench.py",
             ROOT / "src/shinku/qq/owner_handoff.py", ROOT / "src/shinku/qq/chat_flow.py"]
    return {**upgrade.snapshot(), **{str(path.relative_to(ROOT)): a.sha(path) for path in paths}}


def apply_sequence_actions(memory, step, clock):
    """Correct known fixture fact handles, not a recency-capped timeline page.

    The normal timeline remains bounded. Chat turns may push an old saved fact
    off its first page; that does not revoke a previously known source handle.
    All handles are reauthorized before the already-confirmed local action.
    """
    from shinku.memory.source_reader import MemorySourceReader
    results = []
    scope = memory.scope_for(a.OWNER, "private")
    for action in step.get("actions", []):
        if action["type"] != "correct_fact":
            results.extend(a.apply_actions(memory, {"actions": [action]}, clock, a.OWNER, "private"))
            continue
        ids = [rid for rid, alias in memory.record_aliases.items()
               if alias == action["target"] and rid.startswith("fact:")]
        rows = MemorySourceReader(memory.store).read_sources(requests=tuple(
            (rid, allowed_scope, "") for allowed_scope in (scope, memory.owner_scope) for rid in ids), now=clock[0])
        facts = [row for row in rows if row is not None and row.layer == "fact"
                 and row.metadata.get("role") == "user" and row.metadata.get("sender_id") == a.OWNER
                 and row.metadata.get("confirmation_kind") == "explicit_memory_request"]
        if not any(row.scope == scope for row in facts):
            raise ValueError("fixture_fact_not_found")
        changed = [memory.store.lifecycle.correct_fact(record_id=row.record_id, scope=row.scope,
            expected_hash=hashlib.sha256(row.text.encode()).hexdigest(), replacement=action["replacement"],
            now=clock[0], confirmed=True) for row in facts]
        if not changed or not all(changed):
            raise ValueError("fixture_correction_failed")
        results.append({"type": "correct_fact", "confirmed": True, "changed_count": len(changed),
                        "selection": "known_fixture_fact_handles_revalidated"})
    return results


def run_sequence(case, *, root, policy, build, runtime, config, persona, question_time, permission_mode,
                 progress_callback=None):
    clock = [question_time]
    observed = SequenceRuntime(runtime) if runtime else None
    memory = a.make_service(root / "memory.sqlite3", policy, build.provider, build.status, observed,
        bool(config and config.memory_judge_enabled), clock)
    a.seed(memory, {**case, "required_sources": []}, clock, question_time)
    bridge = None
    if observed:
        registry = a.ToolRegistry({"memory_search": a.MemorySearchTool(memory)})
        bridge = SequenceBridge(runtime=observed, system_prompt=persona, memory_service=memory,
            tool_host=a.ToolHost(registry), allowed_tool_names={"memory_search"},
            policy=a.ExecutionPolicy(risk_by_tool={"memory_search": "low"}), master_user_ids=(a.OWNER,),
            owner_gate_required=True, command_state=a.QQCommandState(permission_mode=permission_mode, owner_ids=(a.OWNER,)),
            temperature=config.temperature, max_rounds=config.max_rounds, history_turns=config.history_turns,
            recent_context_messages=config.recent_context_messages, prompt_cache_key="shinku:memory-stage-a:v1",
            task_contract_enabled=True, sender=lambda _: a.DeliveryResult(True, "isolated-local-sink"))
    steps = []
    try:
        for index, step in enumerate(case["steps"]):
            clock[0] = question_time + index * 30
            actions = apply_sequence_actions(memory, step, clock)
            before_contexts, before_tools = len(memory.contexts), len(memory.tool_reads)
            before_calls = len(observed.calls) if observed else 0
            before_handoffs = len(bridge.handoffs) if bridge else 0
            if observed:
                observed.forbidden = step.get("forbidden_evidence", [])
            started = time.monotonic()
            if bridge:
                message = a.IncomingMessage.from_event({"post_type": "message", "message_type": "private",
                    "user_id": a.OWNER, "message_id": "sequence-" + case["id"] + "-" + step["id"], "time": clock[0],
                    "sender": {"nickname": "测试主人"}, "message": [{"type": "text", "data": {"text": step["question"]}}]})
                turn = a.QQTurn(conversation_id=a.OWNER, conversation_type="private", primary_message_id=message.message_id,
                    message_ids=(message.message_id,), combined_text=step["question"], messages=(message,))
                reply = bridge.handle_turn(turn).as_dict()
            else:
                memory.prepare_turn(conversation_id=a.OWNER, conversation_type="private", user_message=step["question"],
                    message_meta={"sender_id": a.OWNER, "message_id": "sequence-" + step["id"]})
                reply = {"agent_status": "not_run", "final_text": "", "reason": "offline_retrieval_only"}
            contexts, tools = memory.contexts[before_contexts:], memory.tool_reads[before_tools:]
            calls = observed.calls[before_calls:] if observed else []
            handoffs = bridge.handoffs[before_handoffs:] if bridge else []
            evidence = [row for context in contexts + tools for row in context.get("records", [])]
            rendered = "\n".join(str(context.get("rendered") or "") for context in contexts + tools)
            visible_ids = set(source for call in calls for source in call["evidence_source_ids"]) if observed else set(
                re.findall(r"source=([^）\s]+)", rendered))
            visible = sorted({memory.canonical(row) for row in evidence if row["source_id"] in visible_ids})
            required = set(step["required_sources"])
            score = a.score_answer(reply["final_text"], step["answer_rule"]) if observed else None
            steps.append({**step, "actions_result": actions, "reply": reply,
                "memory_contexts": contexts, "tool_reads": tools, "model_calls": calls,
                "owner_handoff_chars": sum(map(len, handoffs)), "injected_chars": len(rendered),
                "visible_sources": visible, "required_sources_visible": required <= set(visible) if required else None,
                "evidence_leaks": [word for word in step.get("forbidden_evidence", []) if word in rendered],
                "answer_check": score, "execution_failure_layer": a.failure_layer(reply, calls),
                "seconds": round(time.monotonic() - started, 3)})
            if progress_callback:
                progress_callback({"id": case["id"], "split": case["split"], "category": case["category"],
                                   "history": case["history"], "steps": steps})
        return {"id": case["id"], "split": case["split"], "category": case["category"],
                "history": case["history"], "steps": steps}
    finally:
        memory.lifecycle_review.stop()
        memory.store.stop_embedding_worker()
        memory.store.lifecycle.stop_worker()


def render(report):
    lines = ["# E 连续话题实测", "", "状态：" + report["status"], "",
        "同一桥接器和隔离记忆库连续运行；raw=B+C+D，enhanced仅增开E。不是五个独立初始化的查询。",
        "保存配置、人设和工具权限不变；只允许 memory_search，本地投递不发送 QQ。",
        "新注入记忆、主人问候线索和普通近期对话分别计数；普通对话上下文不等于来源证据。",
        "规则是初筛，实际最终回复和首次失败均保留。", ""]
    runs = list(report["runs"])
    if report.get("status") == "harness_error" and report.get("in_progress_sequence"):
        runs.append(report["in_progress_sequence"])
    for run in runs:
        sequence = run["sequence"]
        lines += [f"## {run['repeat']} / {run['branch']} / {sequence['id']}", "", "虚构历史："]
        for row in sequence["history"]:
            lines.append(f"- {row['id']} / {row['at']}：{row['text']}")
        for step in sequence["steps"]:
            lines += ["", "### " + step["id"], "", "问题：" + step["question"], "",
                "实际回复：" + (step["reply"]["final_text"] or "（离线未调用／无有效回答）"), "",
                f"模型可见来源：{step['visible_sources']}；本轮记忆{step['injected_chars']}字；问候线索{step['owner_handoff_chars']}字；失败={step['execution_failure_layer']}。"]
            for context in step["memory_contexts"]:
                lines += ["", "实际注入：", "", "```text", context["rendered"] or "（无）", "```"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3 or args.retrieval_only and not args.live:
        parser.error("repeats=1..3; retrieval-only requires --live")
    fixture = load_fixture()
    output = ROOT / "docs" / ("MEMORY_STAGE_E_SEQUENCE_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    report = {"status": "running", "live": args.live and not args.retrieval_only,
        "repeats": args.repeats, "fixture_hash": a.sha(FIXTURE), "source_snapshot_before": snapshot(),
        "runs": [], "isolation": {"fictional_only": True, "qq_transport": False, "production_memory_access": False}}
    runtime = None
    profile = {}
    def save():
        (output / "sequence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "SEQUENCE.md").write_text(render(report), encoding="utf-8")
    try:
        if args.live:
            runtime, config, policy, build, persona, profile = a.saved_profile(output)
            if policy.semantic_enabled and build.status != "huggingface":
                raise RuntimeError("saved_embedding_unavailable_no_substitution")
            report["profile"] = profile
        else:
            policy = a.MemoryPolicy(master_user_ids=(a.OWNER,))
            build, config, persona = SimpleNamespace(provider=None, status="disabled"), None, ""
        save()
        print("REPORT " + str(output), flush=True)
        for repeat in range(1, args.repeats + 1):
            for case in fixture["cases"]:
                for branch in (("raw", "enhanced") if repeat % 2 else ("enhanced", "raw")):
                    root = output / f"{repeat}-{branch}-{case['id']}"
                    root.mkdir()
                    print(f"START {repeat} {branch} {case['id']}", flush=True)
                    changed = replace(policy, search_keys_enabled=True, current_understanding_enabled=True,
                        evidence_package_enabled=True, history_needs_enabled=branch == "enhanced")
                    def progress(value):
                        report["in_progress_sequence"] = {"repeat": repeat, "branch": branch, "sequence": value}
                        save()
                    sequence = run_sequence(case, root=root, policy=changed, build=build,
                        runtime=runtime if not args.retrieval_only else None, config=config, persona=persona,
                        question_time=a.timestamp(fixture["question_date"]), permission_mode=profile.get("permission_mode", "traditional"),
                        progress_callback=progress)
                    report["runs"].append({"repeat": repeat, "branch": branch, "sequence": sequence})
                    report.pop("in_progress_sequence", None)
                    save()
                    print("CASE " + json.dumps({"repeat": repeat, "branch": branch, "id": case["id"],
                        "complete": [row["required_sources_visible"] for row in sequence["steps"]],
                        "failures": [row["execution_failure_layer"] for row in sequence["steps"]]}, ensure_ascii=False), flush=True)
        report["status"] = "completed"
        return 0
    except Exception as exc:
        report["status"], report["error_type"] = "harness_error", type(exc).__name__
        raise
    finally:
        report["source_snapshot_after"] = snapshot()
        report["source_unchanged_during_run"] = report["source_snapshot_before"] == report["source_snapshot_after"]
        report["protected_configs_unchanged"] = all(Path(path).is_file() and a.sha(Path(path)) == value
            for path, value in profile.get("protected_config_hashes", {}).items())
        if runtime:
            report["metrics"] = runtime.snapshot_metrics()
            runtime.close()
        save()
        print("STATUS " + json.dumps({key: report[key] for key in
            ("status", "source_unchanged_during_run", "protected_configs_unchanged")}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
