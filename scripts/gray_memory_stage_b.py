"""Paired Stage B measurement using Stage A's unchanged fictional reply harness.

Raw-source retrieval (including the existing BGE) versus local hints+source reading.
Three paired repetitions are the default. Failures remain in the report. No QQ,
production memory, downloads, provider switches or production configuration saves.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

import gray_memory_stage_a as a

ROOT = a.ROOT


def snapshot() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "scripts/gray_memory_stage_a.py",
             ROOT / "src/shinku/memory/policy.py", ROOT / "src/shinku/memory/search_index.py",
             ROOT / "src/shinku/memory/store.py", ROOT / "src/shinku/memory/service.py",
             ROOT / "src/shinku/memory/router.py", ROOT / "src/shinku/memory/lifecycle.py",
             ROOT / "src/shinku/qq/agent_bridge.py"]
    return {str(path.relative_to(ROOT)): a.sha(path) for path in paths}


def render(report: dict[str, Any]) -> str:
    lines = ["# 阶段 B 成对测量：搜索线索与原文", "",
             f"状态：{report['status']}；真实模型：{report['live']}；重复次数：{report['repeats']}。", "",
             "只用虚构语料与隔离数据库，不是线上 QQ 证明。实验类型：" + report["experiment"] + "。",
             "existing_hybrid_paired 保留已有 BGE；lexical_fallback_fault 是双方均关闭向量的故障对照；offline_lexical_only 不调用模型且只测本地词项。",
             "existing_hybrid_retrieval_only 使用保存配置与已有 BGE，仅验证检索，不调用模型或给回答评分。",
             "关键词规则只是初筛，不是语义正确率；补充集与冻结 A 集分别计分。", "",
             "## 汇总", "", "| 集合／次数／分支 | 取齐来源 | 输入可见取齐 | 规则初筛 | 调用数 | 中位秒 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, repetitions in report.get("summary", {}).items():
        for rep, branches in repetitions.items():
            for branch, stats in branches.items():
                s = stats["all"]
                lines.append(f"| {name}/{rep}/{branch} | {s['complete_evidence_cases']}/{s['required_evidence_cases']} | "
                             f"{s['visible_complete_evidence_cases']}/{s['required_evidence_cases']} | "
                             f"{s['qa_rule_pass']}/{s['qa_checked']} | {s['model_calls']} | {s['median_seconds']} |")
    lines += ["", "## 实际问答与来源（失败也保留）", ""]
    for run in report["runs"]:
        row = run["case"]
        lines += [f"### {run['fixture']} / {run['repeat']} / {run['branch']} / {row['id']}", "",
                  "问题：" + row["question"], "", "历史："]
        for h in row["history"]:
            lines.append(f"- {h['id']} / {h.get('role', 'user')} / {h['at']}：{h['text']}")
        lines += ["", "真实回复：", "", row["reply"]["final_text"] or "（本轮没有有效模型回复／离线未调用）", "",
                  f"来源：{row['retrieved_sources']}；模型可见：{row['visible_sources']}；归因：{row['attribution']}。", ""]
        for context in row["memory_contexts"]:
            for record in context["records"]:
                meta = record["metadata"]
                lines.append(f"- 原文来源 `{record['source_id']}` / role={meta.get('role', meta.get('evidence_kind', '未记录'))} / "
                             f"线索={meta.get('search_hints', [])} / 版本={meta.get('source_version', '旧路径未登记')}：{record['text']}")
        lines.append("")
    return "\n".join(lines)


def summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run in runs:
        result.setdefault(run["fixture"], {}).setdefault(str(run["repeat"]), {}).setdefault(run["branch"], [])
        result[run["fixture"]][str(run["repeat"])][run["branch"]].append(run["case"])
    return {name: {rep: {branch: a.summarize(rows) for branch, rows in branches.items()}
                   for rep, branches in reps.items()} for name, reps in result.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="With --live, load the saved BGE/profile but make no model calls")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--fixture", choices=("main", "expansion", "scope", "all"), default="all")
    parser.add_argument("--lexical-fallback", action="store_true",
                        help="Separate fault experiment: disable embeddings for BOTH branches, never main baseline")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    if args.retrieval_only and (not args.live or args.lexical_fallback):
        parser.error("retrieval-only requires --live and the unchanged existing BGE baseline")
    if args.lexical_fallback and args.fixture != "expansion":
        parser.error("lexical fallback is a separately labelled expansion experiment only")
    fixtures = {"main": a.FIXTURE, "expansion": ROOT / "tests/fixtures/memory_stage_b_v1.json",
                "scope": ROOT / "tests/fixtures/memory_stage_a_scope_v1.json"}
    selected = fixtures if args.fixture == "all" else {args.fixture: fixtures[args.fixture]}
    data = {name: a.load_cases(path) for name, path in selected.items()}
    all_ids = {case["id"] for fixture in data.values() for case in fixture["cases"]}
    if set(args.case_id) - all_ids:
        parser.error("unknown case id")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "docs" / ("MEMORY_STAGE_B_" + run_id)
    run_root.mkdir(exist_ok=False)
    report: dict[str, Any] = {"run_id": run_id, "live": bool(args.live and not args.retrieval_only), "repeats": args.repeats,
        "saved_profile_loaded": args.live,
        "experiment": "lexical_fallback_fault" if args.lexical_fallback else "existing_hybrid_retrieval_only" if args.retrieval_only else "existing_hybrid_paired" if args.live else "offline_lexical_only",
        "fixture_hashes": {name: a.sha(path) for name, path in selected.items()},
        "source_snapshot_before": snapshot(), "runs": [], "status": "running",
        "isolation": {"fictional_only": True, "production_memory_read": False, "production_memory_write": False,
                      "qq_transport": False, "model_download": False, "extra_extraction_calls": 0}}
    runtime = config = None
    profile: dict[str, Any] = {}
    def save():
        report["summary"] = summaries(report["runs"])
        (run_root / "paired.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "PAIRED.md").write_text(render(report), encoding="utf-8")
    try:
        if args.live:
            runtime, config, policy, build, persona, profile = a.saved_profile(run_root)
            if policy.semantic_enabled and build.status != "huggingface":
                raise RuntimeError("saved_semantic_model_unavailable_no_silent_substitution")
            report["profile"] = profile
        else:
            policy = a.MemoryPolicy(master_user_ids=(a.OWNER,))
            build = SimpleNamespace(provider=None, status="disabled")
            persona = ""
        if args.lexical_fallback:
            policy = replace(policy, semantic_enabled=False)
            build = SimpleNamespace(provider=None, status="fault_disabled")
        save()
        print("REPORT " + str(run_root), flush=True)
        print("PROFILE " + json.dumps({k: profile.get(k) for k in ("chat_model", "permission_mode", "embedding_status")}), flush=True)
        for rep in range(1, args.repeats + 1):
            for name, fixture in data.items():
                for case in fixture["cases"]:
                    if args.case_id and case["id"] not in args.case_id:
                        continue
                    # Alternate order to reduce fixed provider/time/cache-order effects.
                    branches = ("raw", "enhanced") if rep % 2 else ("enhanced", "raw")
                    for branch in branches:
                        print(f"START {name} {rep} {branch} {case['id']}", flush=True)
                        case_root = run_root / f"{name}-{rep}-{branch}"
                        case_root.mkdir(exist_ok=True)
                        row = a.run_case(case, run_root=case_root, policy=replace(policy, search_keys_enabled=branch == "enhanced"),
                            build=build, runtime=None if args.retrieval_only else runtime, config=config, persona=persona,
                            question_time=a.timestamp(fixture["question_date"]),
                            permission_mode=profile.get("permission_mode", "traditional"))
                        report["runs"].append({"fixture": name, "repeat": rep, "branch": branch, "case": row})
                        save()
                        print("CASE " + json.dumps({"fixture": name, "repeat": rep, "branch": branch,
                            "id": case["id"], "sources_complete": row["all_required_sources_retrieved"],
                            "visible_complete": row["all_required_sources_visible"],
                            "rule_pass": row["answer_check"]["rule_pass"] if row["answer_check"] else None,
                            "seconds": row["seconds"], "failure": row["execution_failure_layer"]}, ensure_ascii=False), flush=True)
        report["status"] = "completed"
        return 0
    except Exception as exc:
        report["status"] = "harness_error"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["protected_configs_unchanged"] = all(Path(path).is_file() and a.sha(Path(path)) == value
            for path, value in profile.get("protected_config_hashes", {}).items())
        report["source_snapshot_after"] = snapshot()
        report["source_unchanged_during_run"] = report["source_snapshot_after"] == report["source_snapshot_before"]
        if runtime:
            report["metrics"] = runtime.snapshot_metrics()
            runtime.close()
        save()
        print("STATUS " + json.dumps({key: report[key] for key in ("status", "protected_configs_unchanged", "source_unchanged_during_run")}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
