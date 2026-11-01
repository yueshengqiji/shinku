"""Prepare a fresh, bounded acceptance batch; execution is always explicit.

Reuse the existing isolated QQ bridge driver. Never overwrite/resume a stopped
batch or silently adapt a recorded model tool decision to a newer protocol.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp/memory_optimization_mr_20261014/increment09/remaining/manifest.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for folder in ("src", "private/src", "scripts", "tests", "private/tests")
            for p in sorted((ROOT / folder).rglob("*.py"))}


def source_path(value, previous_root=None):
    """Resolve repository inputs, relocating historical roots only explicitly."""
    value = str(value)
    windows = PureWindowsPath(value)
    foreign_windows = bool(windows.drive) and os.name != "nt"
    if not foreign_windows:
        path = (ROOT / value).resolve()
        try:
            path.relative_to(ROOT)
            return path
        except ValueError:
            pass
    if not previous_root:
        raise ValueError("source_path_outside_repository_supply_source_root")
    path_type = PureWindowsPath if PureWindowsPath(previous_root).is_absolute() else PurePosixPath
    old_root, old_path = path_type(previous_root), path_type(value)
    if not old_root.is_absolute() or not old_path.is_absolute():
        raise ValueError("source_relocation_requires_absolute_paths")
    try:
        relative = old_path.relative_to(old_root)
    except ValueError as exc:
        raise ValueError("source_path_outside_declared_source_root") from exc
    if ".." in relative.parts:
        raise ValueError("source_relocation_parent_traversal")
    path = ROOT.joinpath(*relative.parts).resolve()
    path.relative_to(ROOT)
    return path


def key(row):
    return row["id"] + "/" + row["arm"]


def prepare(args, output):
    source = Path(args.source_manifest).resolve()
    source.relative_to(ROOT)
    old = read(source)
    previous_root = getattr(args, "source_root", None)
    fixture = source_path(old["fixture"], previous_root) if "fixture" in old else source.parent / "fixtures.json"
    driver_key = "imported_driver" if "imported_driver" in old else "driver"
    driver = source_path(old[driver_key], previous_root)
    for name, path in (("fixture", fixture), (driver_key, driver)):
        if sha(path) != old[name + "_sha256"]:
            raise ValueError("source_manifest_hash_mismatch:" + name)
    previous = read(source.parent / "results.json") if (source.parent / "results.json").exists() else []
    attempted = {key(row) for row in previous}
    retries = set(args.retry)
    if not retries <= attempted:
        raise ValueError("retry_must_name_a_previously_attempted_case_arm")
    schedule = [row for row in old["schedule"] if key(row) not in attempted or key(row) in retries]
    if not schedule or len({key(row) for row in schedule}) != len(schedule):
        raise ValueError("empty_or_duplicate_schedule")
    if any(not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", row["id"])
           or row["arm"] not in {"legacy", "model"} for row in schedule):
        raise ValueError("invalid_case_identifier_or_strategy")
    corpus = read(fixture)
    kind = corpus.get("source_kind")
    if kind not in (None, "fictional_only") or kind is None and not args.synthetic_fixtures:
        raise ValueError("synthetic_fixtures_required")
    wanted = {row["id"] for row in schedule}
    cases = [copy.deepcopy(row) for row in corpus["cases"] if row["id"] in wanted]
    if len(cases) != len(wanted):
        raise ValueError("missing_or_duplicate_fixture")
    converted = []
    for row in cases:
        if row.pop("replayed_candidate", None) is not None or "replayed_response" in row:
            converted.append(row["id"])
        row.pop("replayed_response", None)
    # Generate a fresh main decision. This tests current protocol generation,
    # not whether the original recorded tool choice can be repaired verbatim.
    corpus["cases"] = cases
    corpus["source_kind"] = "fictional_only"
    prior = read(source.parent / "summary.json") if (source.parent / "summary.json").exists() else {}
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "fixtures.json", corpus)
    dump(output / "source-freeze.json", {"files": source_files()})
    manifest = {
        "schema_version": 1, "source_manifest": str(source), "source_manifest_sha256": sha(source),
        "source_repository_root": previous_root,
        "resolved_source_paths": {"fixture": str(fixture), "driver": str(driver)},
        "driver": str(driver), "driver_sha256": sha(driver), "runner_sha256": sha(__file__),
        "fixture_sha256": sha(output / "fixtures.json"),
        "source_freeze_sha256": sha(output / "source-freeze.json"), "schedule": schedule,
        "explicit_retries": sorted(retries), "fresh_main_instead_of_recorded_response": converted,
        "stimulus": "fresh_main_generation", "authorization": old.get("authorization", ""),
        "synthetic_fixture_asserted": args.synthetic_fixtures,
        "max_requests": args.max_requests, "max_tokens": args.max_tokens,
        "max_seconds": args.max_seconds, "per_case": args.per_case,
        "prior_ledger": str(source.parent / "summary.json"), "prior_budget": prior.get("budget", {}),
        "prior_usage_history": old.get("prior_usage_history", old.get("prior_unknown_usage_preserved", {})),
        "prior_consecutive_technical_failures": 1 if prior.get("budget", {}).get("estimated_requests", 0) else 0,
        "semantic_acceptance": "pending_independent_review", "production_operations": False,
    }
    dump(output / "manifest.json", manifest)
    print(json.dumps({"prepared": str(output), "planned": len(schedule), "model_calls": 0}, ensure_ascii=False))


def frozen(output):
    manifest = read(output / "manifest.json")
    pairs = [(Path(manifest["driver"]), manifest["driver_sha256"]),
             (Path(__file__), manifest["runner_sha256"]),
             (output / "fixtures.json", manifest["fixture_sha256"]),
             (output / "source-freeze.json", manifest["source_freeze_sha256"])]
    if any(sha(path) != digest for path, digest in pairs):
        raise ValueError("batch_artifact_changed_prepare_new_batch")
    files = read(output / "source-freeze.json")["files"]
    if source_files() != files:
        raise ValueError("working_copy_changed_prepare_new_batch")
    return manifest, files


def load_driver(output, manifest):
    for name in ("SHINKU_DATA_ROOT", "SHINKU_PRIVATE_DATA_ROOT", "SHINKU_LOG_ROOT", "SHINKU_LOG_DIR"):
        os.environ[name] = str(output / "host-runtime")
    spec = importlib.util.spec_from_file_location("memory_acceptance_bridge", manifest["driver"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != ROOT:
        raise ValueError("driver_repository_mismatch")
    module.HERE = output

    class Observed(module.Observed):
        def invoke(self, method, kw):
            start_index = len(self.guard.calls)
            count_before = len(self.calls)
            started = time.monotonic()
            try:
                return super().invoke(method, kw)
            finally:
                if len(self.calls) > count_before:
                    row = self.calls[-1]
                    dispatches = self.guard.calls[start_index:]
                    row["elapsed_seconds"] = time.monotonic() - started
                    row["provider_dispatches"] = copy.deepcopy(dispatches)
                    row["dispatch_failures"] = [d["error_type"] for d in dispatches if d.get("error_type")]
                    row["budget_failure"] = row.get("error_type") == "TaskBudgetExceeded" or "TaskBudgetExceeded" in row["dispatch_failures"]
                    row["usage_missing"] = any(d.get("budget_after", {}).get("estimated_requests", 0) >
                                               d.get("budget_before", {}).get("estimated_requests", 0)
                                               for d in dispatches)
                    result = row.get("result")
                    row["technical_failure"] = bool(
                        row.get("error_type") not in (None, "TaskBudgetExceeded")
                        or any(kind != "TaskBudgetExceeded" for kind in row["dispatch_failures"])
                        or row["usage_missing"] or isinstance(result, dict) and result.get("_agent_transport_fallback"))
                    row["invalid_protocol_result"] = not isinstance(result, dict) or not result
                    row["request_usage"] = {name: sum(max(0, d.get("budget_after", {}).get(name, 0)
                                                         - d.get("budget_before", {}).get(name, 0))
                                                     for d in dispatches)
                                            for name in ("requests", "reported_tokens", "tokens", "estimated_requests")}
                    row["request_usage"]["unreconciled_reservation"] = max(0,
                        row["request_usage"]["tokens"] - row["request_usage"]["reported_tokens"])
                    row["system_extra_chars"] = sum(len(str(block)) for block in kw.get("system_extra_blocks") or ())
                    # JSON calls expose completed text only. Do not fabricate
                    # streaming latency from total generation time.
                    row["first_token_seconds"] = None
                    row["first_visible_text_seconds"] = None
                    dump(self.root / "calls.json", self.calls)

    module.Observed = Observed
    return module


def preflight(output, manifest, driver):
    with (output / "preflight-started.json").open("x", encoding="utf-8") as file:
        json.dump({"started": time.time()}, file)
    original = driver.Scripted

    class CurrentScripted(original):
        def call_chat_json(self, **kw):
            value = copy.deepcopy(super().call_chat_json(**kw))
            call = value.get("tool_call")
            if isinstance(call, dict) and call.get("name") == "memory_search":
                # This is a scripted preflight fixture, never a real model
                # decision. Keep the required protocol field explicit.
                call["arguments"]["recorded_time"] = {"start": "", "end": ""}
            if value.get("kind") == "final":
                value["text"] = "苍鹭 K24 是桌面试验泵，密封圈采用丁腈橡胶，尚未做户外试验。"
            return value

    driver.Scripted = CurrentScripted
    driver.preflight()
    frozen(output)
    dump(output / "ready.json", {"passed": True, "network_requests": 0,
                                "manifest_sha256": sha(output / "manifest.json")})


def live(output, manifest, driver):
    ready = read(output / "ready.json")
    if not ready.get("passed") or ready["manifest_sha256"] != sha(output / "manifest.json"):
        raise ValueError("fresh_preflight_required")
    if not manifest.get("authorization"):
        raise ValueError("missing_recorded_model_test_authorization")
    with (output / "started.json").open("x", encoding="utf-8") as file:
        json.dump({"started": time.time(), "manifest_sha256": sha(output / "manifest.json")}, file)
    guard = driver.Guard(output, **{name: manifest[name] for name in
                                   ("max_requests", "max_tokens", "max_seconds", "per_case")})
    rows, error, profile_data = [], "", {}
    consecutive = manifest["prior_consecutive_technical_failures"]
    try:
        import gray_memory_stage_a as profile
        runtime, config, policy, build, persona, profile_data = profile.saved_profile(output)
        if build.provider is None or getattr(build, "status", "") != "huggingface":
            raise RuntimeError("saved_semantic_model_unavailable_no_silent_substitution")
        build.provider.embed_text("隔离预热")
        profile_data["diagnostic_differences"] = [
            "sender=None; no QQ delivery; isolated SQLite; no task execution",
            "fresh main generation; current recorded_time protocol; no old response injected"]
        dump(output / "profile.json", profile_data)
        corpus = read(output / "fixtures.json")
        cases = {row["id"]: row for row in corpus["cases"]}
        with guard.install(runtime):
            for item in manifest["schedule"]:
                guard.check()
                frozen(output)
                case_root = output / (item["id"] + "--" + item["arm"])
                for name in ("SHINKU_DATA_ROOT", "SHINKU_PRIVATE_DATA_ROOT", "SHINKU_LOG_ROOT", "SHINKU_LOG_DIR"):
                    os.environ[name] = str(case_root / "host-runtime")
                row = driver.run_case(cases[item["id"]], item["arm"], case_root, corpus["now"], build.provider,
                    policy, runtime, config, persona, profile_data["permission_mode"], guard)
                row["stimulus"] = "fresh_main_generation"
                row["history_query_exercised"] = bool(row["tool_reads"])
                row["technical_failure_calls"] = sum(bool(c.get("technical_failure")) for c in row["model_calls"])
                row["protocol_failure_calls"] = sum(bool(c.get("invalid_protocol_result")) for c in row["model_calls"])
                row["unfinished_model_calls"] = sum("finished" not in c for c in row["model_calls"])
                failed = bool(row["technical_failure_calls"] or row["protocol_failure_calls"])
                rows.append(row)
                dump(case_root / "result.json", row)
                dump(output / "results.json", rows)
                consecutive = consecutive + 1 if failed else 0
                print(json.dumps({"case": key(item), "host_status": row["reply"].get("agent_status"),
                    "technical_failures": row["technical_failure_calls"], "stop": guard.stop}, ensure_ascii=False), flush=True)
                if row["unfinished_model_calls"]:
                    guard.deny("pilot_model_call_not_finished")
                guard.check()
                if consecutive >= 2:
                    guard.deny("two_consecutive_technical_failures")
    except BaseException as exc:
        error = type(exc).__name__
        dump(output / "failure.json", {"error_type": error, "stop": guard.stop, "case": guard.case})
        raise
    finally:
        changed = [name for name, digest in read(output / "source-freeze.json")["files"].items()
                   if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
        protected = [name for name, digest in profile_data.get("protected_config_hashes", {}).items()
                     if not Path(name).is_file() or sha(name) != digest]
        dump(output / "summary.json", {"planned": len(manifest["schedule"]), "returned": len(rows),
            "error": error, "stop": guard.stop, "budget": guard.ledger.snapshot(),
            "prior_ledger": manifest["prior_ledger"], "prior_budget": manifest["prior_budget"],
            "prior_usage_history": manifest["prior_usage_history"],
            "source_changed": changed, "protected_changed": protected,
            "qq_sends": sum(r.get("qq_sends", 0) for r in rows),
            "execution_calls": sum(r.get("execution_calls", 0) for r in rows),
            "semantic_acceptance": "pending_independent_review",
            "rows": [{"id": r["id"], "arm": r["arm"], "status": r["reply"].get("agent_status"),
                      "technical_failures": r["technical_failure_calls"]} for r in rows]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "live"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-root", help="Explicit former repository root for relocated historical inputs; hashes must still match")
    parser.add_argument("--retry", action="append", default=[], metavar="CASE/ARM")
    parser.add_argument("--synthetic-fixtures", action="store_true",
                        help="Assert the explicitly selected legacy fixture without source_kind is fictional")
    parser.add_argument("--max-requests", type=int, default=66)
    parser.add_argument("--max-tokens", type=int, default=1400000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--per-case", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.relative_to(ROOT / "tmp")
    os.chdir(ROOT)
    if any(getattr(args, key) <= 0 for key in ("max_requests", "max_tokens", "max_seconds", "per_case")):
        parser.error("budgets must be positive")
    if args.mode == "prepare":
        prepare(args, output)
        return
    manifest, _ = frozen(output)
    # Exclusive logs and start markers prevent accidentally replaying a batch.
    with (output / (args.mode + "-stdout.log")).open("x", encoding="utf-8") as stdout, \
         (output / (args.mode + "-stderr.log")).open("x", encoding="utf-8") as stderr, \
         redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            driver = load_driver(output, manifest)
            (preflight if args.mode == "preflight" else live)(output, manifest, driver)
        except BaseException:
            traceback.print_exc(file=stderr)
            raise


if __name__ == "__main__":
    main()
