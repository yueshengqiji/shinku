"""Local metrics and first-context version comparisons; never a semantic judge.

Reads explicit measurement JSON files only. Writes a new report directory and
keeps old failures. Does not load production memory, saved models or providers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_context(case):
    rows = case.get("memory_contexts", [])
    return str(rows[0].get("rendered", "")) if rows else ""


def normalize_correction_source(text):
    # Only randomly allocated correction source IDs are normalized. Do not
    # normalize dates, content, roles, author identity, conditions or numbers.
    return re.sub(r"(?<=source=)(?:owner:)?correction:[0-9a-f]{32}", "correction:<generated-id>", text)


def compare(before, after):
    earlier, later = (json.loads(path.read_text(encoding="utf-8")) for path in (before, after))
    index = {(r["fixture"], r["repeat"], r["branch"], r["case"]["id"]): r["case"] for r in later["runs"]}
    rows = []
    for run in earlier["runs"]:
        key = (run["fixture"], run["repeat"], run["branch"], run["case"]["id"])
        current = index.get(key)
        old_text = first_context(run["case"])
        new_text = first_context(current) if current is not None else ""
        rows.append({"fixture": key[0], "repeat": key[1], "branch": key[2], "id": key[3],
                     "matched": current is not None, "exact": current is not None and old_text == new_text,
                     "correction_id_normalized_equal": current is not None and
                        normalize_correction_source(old_text) == normalize_correction_source(new_text),
                     "before_hash": digest(old_text), "after_hash": digest(new_text)})
    return {"before": str(before.relative_to(ROOT)), "after": str(after.relative_to(ROOT)),
            "kind": "first_rendered_context_only_not_reply_equivalence",
            "rows": rows, "exact": sum(r["exact"] for r in rows),
            "normalized": sum(r["correction_id_normalized_equal"] for r in rows), "total": len(rows)}


def metrics(report, path):
    groups = {}
    for run in report["runs"]:
        groups.setdefault((run["fixture"], run["branch"], run["repeat"]), []).append(run["case"])
        groups.setdefault((run["fixture"], run["branch"], "pooled"), []).append(run["case"])
    rows = []
    for (fixture, branch, repeat), cases in groups.items():
        calls = [call for case in cases for call in case["model_calls"]]
        times = sorted(case["seconds"] for case in cases)
        rows.append({"fixture": fixture, "branch": branch, "repeat": repeat, "executions": len(cases),
            "required": sum(case["all_required_sources_visible"] is not None for case in cases),
            "visible_complete": sum(case["all_required_sources_visible"] is True for case in cases),
            "execution_failures": dict(Counter(case["execution_failure_layer"] for case in cases if case["execution_failure_layer"])),
            "evidence_leak_cases": sum(bool(case["evidence_leaks"]) for case in cases),
            "model_calls": len(calls), "reported_tokens_chat": sum(call["metrics_delta"].get("total_tokens_chat", 0) for call in calls),
            "median_seconds": median(times), "p90_seconds_nearest_rank": times[max(0, (9 * len(times) + 9) // 10 - 1)],
            "median_injected_chars": median(case["injected_chars"] for case in cases)})
    return {"path": str(path.relative_to(ROOT)), "status": report["status"], "live": report["live"],
            "source_stable": report["source_unchanged_during_run"], "configs_stable": report["protected_configs_unchanged"],
            "fixture_hashes": report["fixture_hashes"], "metrics": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--compare", nargs=2, action="append", default=[])
    args = parser.parse_args()
    if not args.report and not args.compare:
        parser.error("explicit reports required")
    result = {"kind": "local_metrics_not_semantic_judgment", "reports": [], "comparisons": []}
    for name in args.report:
        path = (ROOT / name).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["status"] != "completed":
            raise ValueError("unfinished_report_not_final_evidence")
        result["reports"].append(metrics(data, path))
    for earlier, later in args.compare:
        result["comparisons"].append(compare((ROOT / earlier).resolve(), (ROOT / later).resolve()))
    output = ROOT / "docs" / ("MEMORY_UPGRADE_RESULTS_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 离线测量汇总", "", "不做语义评分；各版本、首次失败与源报告分别保留。", ""]
    for report in result["reports"]:
        lines += ["## " + report["path"], "", f"实际回复={report['live']}；源码稳定={report['source_stable']}；配置稳定={report['configs_stable']}。", ""]
        for row in report["metrics"]:
            lines.append(f"- {row['fixture']}/{row['branch']}/{row['repeat']}：来源{row['visible_complete']}/{row['required']}，"
                         f"调用{row['model_calls']}，上报token{row['reported_tokens_chat']}，"
                         f"中位{row['median_seconds']:.3f}s，P90{row['p90_seconds_nearest_rank']:.3f}s，"
                         f"执行失败{row['execution_failures']}，禁止片段{row['evidence_leak_cases']}。")
    for comparison in result["comparisons"]:
        lines += ["", "## 初始来源渲染版本比较", "", f"{comparison['before']} → {comparison['after']}："
                  f"原样相同{comparison['exact']}/{comparison['total']}，仅规范化随机更正source ID后{comparison['normalized']}/{comparison['total']}。",
                  "不是完整提示、普通近期对话、后续工具读取或模型回答的等价证明；不同源码版本不能混算。"]
    (output / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT " + str(output), flush=True)
    for row in result["comparisons"]:
        print(json.dumps({k: row[k] for k in ("before", "after", "exact", "normalized", "total")}), flush=True)


if __name__ == "__main__":
    main()
