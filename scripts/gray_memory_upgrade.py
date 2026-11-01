"""Paired C-F offline upgrades on the unchanged fictional Stage A reply harness.

Each pair toggles just the requested stage above its previous-stage baseline.
Saved models/persona are read-only. No production memory or QQ transport; failures
are retained. Local-only mode creates no model replies and never scores answers.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import gray_memory_stage_a as a
import gray_memory_stage_b as b

ROOT = a.ROOT


def snapshot():
    paths = [Path(__file__), ROOT / "src/shinku/memory/current_understanding.py",
             ROOT / "src/shinku/memory/source_reader.py", ROOT / "src/shinku/api/memory_admin.py",
             ROOT / "src/shinku/memory/evidence.py", ROOT / "src/shinku/memory/workbench.py"]
    return {**b.snapshot(), **{str(path.relative_to(ROOT)): a.sha(path) for path in paths}}


def render(report):
    stage = report["stage"]
    lines = [f"# 阶段 {stage} 成对测量", "", f"状态：{report['status']}；实际模型回复：{report['live']}；三轮配置：{report['repeats']}。", "",
             "只用虚构语料、隔离数据库、保存的模型配置和本地投递，不是线上证明。",
             "raw 是前阶段基线，enhanced 只增开本阶段；两边都保留已有 BGE 和 B 的来源回读。",
             "没有远程提炼调用；首次失败保留，规则初筛不等于语义正确率。", "",
             "## 汇总", "", "| 集合／次数／分支 | 取齐来源 | 模型可见取齐 | 规则初筛 | 调用数 | 中位秒 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for fixture, repetitions in report["summary"].items():
        for repeat, branches in repetitions.items():
            for branch, stats in branches.items():
                row = stats["all"]
                lines.append(f"| {fixture}/{repeat}/{branch} | {row['complete_evidence_cases']}/{row['required_evidence_cases']} | "
                    f"{row['visible_complete_evidence_cases']}/{row['required_evidence_cases']} | {row['qa_rule_pass']}/{row['qa_checked']} | "
                    f"{row['model_calls']} | {row['median_seconds']} |")
    original = b.render(report)
    lines += ["", original[original.index("## 实际问答与来源"):]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("C", "D", "E"), required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixture", choices=("stage", "regression", "all"), default="stage")
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3 or args.retrieval_only and not args.live:
        parser.error("repeats=1..3 and retrieval-only requires saved-profile --live")
    paths = {"stage": ROOT / f"tests/fixtures/memory_stage_{args.stage.lower()}_v1.json"}
    if args.fixture in {"regression", "all"}:
        paths.update({"main": a.FIXTURE, "scope": ROOT / "tests/fixtures/memory_stage_a_scope_v1.json"})
    if args.fixture == "regression":
        paths.pop("stage")
    fixtures = {name: a.load_cases(path) for name, path in paths.items()}
    all_ids = {case["id"] for fixture in fixtures.values() for case in fixture["cases"]}
    if set(args.case_id) - all_ids:
        parser.error("unknown case id")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "docs" / f"MEMORY_STAGE_{args.stage}_{run_id}"
    output.mkdir(exist_ok=False)
    report = {"stage": args.stage, "run_id": run_id, "repeats": args.repeats,
              "live": bool(args.live and not args.retrieval_only), "saved_profile_loaded": args.live,
              "experiment": "stage_upgrade_model_paired" if args.live and not args.retrieval_only else "stage_upgrade_retrieval_only",
              "baseline_features": ["source_reading", "local_search_keys"] + (["current_understanding"] if args.stage in {"D", "E"} else []) + (["evidence_package"] if args.stage == "E" else []),
              "new_feature": {"C": "current_understanding", "D": "evidence_package", "E": "history_needs_workbench"}[args.stage],
              "fixture_hashes": {name: a.sha(path) for name, path in paths.items()},
              "source_snapshot_before": snapshot(), "runs": [], "status": "running",
              "isolation": {"fictional_only": True, "production_memory_read": False, "production_memory_write": False,
                            "qq_transport": False, "model_download": False, "extra_extraction_calls": 0}}
    runtime = config = None
    profile = {}
    def save():
        report["summary"] = b.summaries(report["runs"])
        (output / "paired.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "PAIRED.md").write_text(render(report), encoding="utf-8")
    try:
        if args.live:
            runtime, config, policy, build, persona, profile = a.saved_profile(output)
            if policy.semantic_enabled and build.status != "huggingface":
                raise RuntimeError("saved_embedding_unavailable_no_substitution")
            report["profile"] = profile
        else:
            policy = a.MemoryPolicy(master_user_ids=(a.OWNER,))
            build = SimpleNamespace(provider=None, status="disabled")
            persona = ""
        save()
        print("REPORT " + str(output), flush=True)
        for repeat in range(1, args.repeats + 1):
            for fixture_name, fixture in fixtures.items():
                for case in fixture["cases"]:
                    if args.case_id and case["id"] not in args.case_id:
                        continue
                    for branch in (("raw", "enhanced") if repeat % 2 else ("enhanced", "raw")):
                        print(f"START {fixture_name} {repeat} {branch} {case['id']}", flush=True)
                        root = output / f"{fixture_name}-{repeat}-{branch}"
                        root.mkdir(exist_ok=True)
                        changed = replace(policy, search_keys_enabled=True,
                            current_understanding_enabled=args.stage in {"D", "E"} or branch == "enhanced",
                            evidence_package_enabled=args.stage == "E" or args.stage == "D" and branch == "enhanced",
                            history_needs_enabled=args.stage == "E" and branch == "enhanced")
                        row = a.run_case(case, run_root=root, policy=changed, build=build,
                            runtime=None if args.retrieval_only else runtime, config=config, persona=persona,
                            question_time=a.timestamp(fixture["question_date"]), permission_mode=profile.get("permission_mode", "traditional"))
                        report["runs"].append({"fixture": fixture_name, "repeat": repeat, "branch": branch, "case": row})
                        save()
                        print("CASE " + json.dumps({"fixture": fixture_name, "repeat": repeat, "branch": branch,
                            "id": case["id"], "source_complete": row["all_required_sources_retrieved"],
                            "visible_complete": row["all_required_sources_visible"], "seconds": row["seconds"],
                            "failure": row["execution_failure_layer"]}, ensure_ascii=False), flush=True)
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
        print("STATUS " + json.dumps({key: report[key] for key in
            ("status", "protected_configs_unchanged", "source_unchanged_during_run")}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
