"""P5a offline paired experiment. No provider, QQ transport or production config.

The scripted model deliberately ignores prompts: this validates measurement and
host boundaries, not the effectiveness of Plan-and-Execute or model reasoning.
Both arms use the same AgentService/AgentExecutor/AgentLoop and task contract.
The staged arm adds a frozen, advisory scaffold for complex samples only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from shinku.agent.budget import TaskBudget, bind_task_budget, current_task_budget
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentRequest, AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import MaterialSource, TaskMaterialStore
from shinku.agent_runtime.service import AgentService
from shinku.agent_runtime.stages import StagePolicy
from shinku.hosts.tool_host import ToolHost
from shinku.result_expression import ExpressionArchive, ExpressionService
from shinku.result_expression.events import ResultEventConsumer
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

CORPUS = Path(__file__).resolve().parents[1] / "tests/fixtures/agent_stage_cases.json"
ARMS = ("react", "staged")
MODEL_ID = "scripted-fixture-v1"
SCOPE = AgentScope("benchmark-fixture", "local:benchmark-fixture")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    return (isinstance(name, str) and bool(name) and len(name) <= 80
            and name not in {".", ".."} and all(c.isalnum() or c in "_.-" for c in name)
            and not name.endswith((".", " ")) and ":" not in name
            and name.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL",
                *("COM" + str(n) for n in range(1, 10)), *("LPT" + str(n) for n in range(1, 10))})


def validate_corpus(corpus):
    """Freeze identities, hard limits and full contract coverage before any IO."""
    if corpus.get("schema_version") != 1 or not 4 <= len(corpus.get("cases", [])) <= 32:
        raise ValueError("invalid_corpus")
    if set(corpus.get("budget", {})) != {"max_requests", "max_tokens", "max_tools", "max_active_seconds"}:
        raise ValueError("invalid_budget")
    ExecutionGrant("fixture", frozenset(), budget_limits=corpus["budget"]).budget()
    seen = set()
    for case in corpus["cases"]:
        name = case.get("id")
        if not safe_name(name) or name in seen or case.get("kind") not in {"simple", "complex", "safety"}:
            raise ValueError("invalid_case_identity")
        seen.add(name)
        if type(case.get("expected_success")) is not bool or not case.get("expected"):
            raise ValueError("invalid_case_expectation")
        for files in (case["expected"], case.get("initial", {})):
            if not isinstance(files, dict) or len(files) > 8 or any(
                not safe_name(p) or not isinstance(v, str) or len(v) > 8000 for p, v in files.items()):
                raise ValueError("invalid_case_files")
        if not isinstance(case.get("request"), str) or not 0 < len(case["request"]) <= 12000:
            raise ValueError("invalid_case_request")
        actions = case.get("actions")
        if not isinstance(actions, list) or len(actions) > 24 or any(
            not isinstance(row, dict) or row.get("op") not in {"read", "write", "final", "unknown_tool", "raise"}
            for row in actions):
            raise ValueError("invalid_fixture_actions")
        stages = case.get("stages", [])
        if len(stages) > 8 or (case["kind"] == "simple" and stages) or (case["kind"] == "complex" and not stages):
            raise ValueError("invalid_stage_count")
        ids, covered = set(), set()
        for row in stages:
            if (set(row) != {"id", "goal", "depends", "artifacts"}
                    or not safe_name(row.get("id")) or row["id"] in ids
                    or not isinstance(row.get("goal"), str) or not 0 < len(row["goal"]) <= 1000
                    or not isinstance(row.get("depends"), list) or set(row["depends"]) - ids
                    or not isinstance(row.get("artifacts"), list) or not row["artifacts"]
                    or set(row["artifacts"]) - set(case["expected"])):
                raise ValueError("invalid_stage_dependencies_or_scope")
            ids.add(row["id"])
            covered.update(row["artifacts"])
        if stages and covered != set(case["expected"]):
            raise ValueError("stage_omits_original_requirement")
        limits = {**corpus["budget"], **case.get("budget", {})}
        if any(limits[k] > corpus["budget"][k] for k in corpus["budget"]):
            raise ValueError("case_expands_budget")
        ExecutionGrant("fixture", frozenset(), budget_limits=limits).budget()
    return corpus


class FileAdapter:
    """Real bounded local file IO; read and write have separate host policies."""
    agent_completion_target_arguments = ("path",)

    def __init__(self, root, case, mode, batch=None):
        self.root, self.case, self.mode = root, case, mode
        self.batch = batch
        self.tool_type = "benchmark_" + mode
        self.agent_progress_class = "observe" if mode == "read" else "change"
        self.agent_permission_class = "read" if mode == "read" else "write"
        self.agent_completion_effects = ("write",) if mode == "write" else ()
        self.allowed = set(case["expected"]) if mode == "write" else set(case.get("initial", {}))
        self.calls = []
        self.validation_reads = 0

    def tool_metadata(self):
        properties = {"path": {"type": "string"}}
        required = ["path"]
        if self.mode == "write":
            properties["content"] = {"type": "string", "maxLength": 8000}
            required.append("content")
        return SimpleNamespace(description="Read/write only preapproved isolated fixture files.",
            input_schema={"type": "object", "properties": properties, "required": required})

    def normalize_call(self, call):
        path = Path(str(call.get("path", ""))).resolve()
        if (path.parent != self.root or path.name not in self.allowed or path.is_symlink()
                or self.mode == "write" and (not isinstance(call.get("content"), str) or len(call["content"]) > 8000)):
            return None
        return dict(call)

    def execute(self, *, call, context):
        path = Path(call["path"])
        if self.normalize_call(call) is None:
            raise ValueError("fixture_path_denied")
        if self.batch is not None:
            self.batch.use_tool()
        if self.mode == "write":
            path.write_bytes(call["content"].encode("utf-8"))
        raw = path.read_bytes()
        self.calls.append({"path": path.name, "sha256": digest(raw)})
        receipt = {"path": str(path), "sha256": digest(raw), "bytes": len(raw),
            "file_text": raw.decode("utf-8"), "result_valid": self.syntax_valid(path, raw),
            "completion_targets": [str(path)], "verified_effects": ["write"] if self.mode == "write" else []}
        return {"tool_type": self.tool_type, "followup_context": raw.decode("utf-8"),
            "state_updates": {**receipt, "changed": self.mode == "write", "artifact_receipts": [receipt]}}

    @staticmethod
    def syntax_valid(path, raw):
        try:
            if path.suffix == ".py":
                ast.parse(raw.decode("utf-8"))
            return True
        except (SyntaxError, UnicodeError):
            return False

    def agent_verify_result(self, *, receipt, context):
        path = Path(receipt.get("path", "")).resolve()
        valid_path = path.parent == self.root and path.name in self.allowed and not path.is_symlink()
        raw = path.read_bytes() if valid_path and path.is_file() else None
        if raw is not None:
            self.validation_reads += 1
        check = {"result_valid": raw is not None and digest(raw) == receipt.get("sha256")
            and self.syntax_valid(path, raw), "file_text": raw.decode("utf-8") if raw is not None else ""}
        if "artifact_receipts" in receipt:
            check["artifact_receipts"] = [{**row, **self.agent_verify_result(receipt=row, context=context)}
                                         for row in receipt["artifact_receipts"]]
        return check


def independent_artifacts(root, case):
    """Ignore model status, stage notes and tool claims; inspect actual bytes."""
    rows = []
    for name, expected in case["expected"].items():
        path = root / name
        raw = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        rows.append({"artifact": name, "present": raw is not None,
            "sha256": digest(raw) if raw is not None else None,
            "expected_sha256": digest(expected.encode("utf-8")),
            "valid": raw == expected.encode("utf-8") and FileAdapter.syntax_valid(path, raw)})
    return rows


def stage_snapshot(root, case, write_calls, previous=None):
    artifacts = {row["artifact"]: row for row in independent_artifacts(root, case)}
    receipts = {(row["path"], row["sha256"]) for row in write_calls}
    completed, rows = set(), []
    for stage in case.get("stages", []):
        deps_ok = set(stage["depends"]) <= completed
        bound = all(artifacts[p]["valid"] and (p, artifacts[p]["sha256"]) in receipts for p in stage["artifacts"])
        status = "completed" if bound and deps_ok else "in_progress" if deps_ok else "pending"
        if status == "completed":
            completed.add(stage["id"])
        before = next((s for s in previous or [] if s["id"] == stage["id"]), None)
        reason = "artifact_changed_or_dependency_unsatisfied" if before and before["status"] == "completed" and status != "completed" else ""
        rows.append({**copy.deepcopy(stage), "status": status, "replan_reason": reason,
            "acceptance": [{"artifact": p, "expected_sha256": artifacts[p]["expected_sha256"]} for p in stage["artifacts"]]})
    return rows


class FixtureRuntime:
    """Same scripted decisions in both arms; account AFTER scaffold addition."""
    def __init__(self, root, case, arm, writer, *, planning='frozen', batch=None):
        self.root, self.case, self.arm, self.writer = root, case, arm, writer
        self.planning, self.batch = planning, batch
        self.decision_count, self.planning_count = 0, 0
        self.calls, self.snapshots, self.prompt_bytes, self.scaffold_bytes = [], [], 0, 0

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        options = copy.deepcopy(kwargs)
        block = ""
        is_plan = str(options.get('prompt_cache_key', '')).endswith(':stage-plan')
        # Simple and safety-only samples keep the ordinary loop's input.
        if self.arm == "staged" and self.case["kind"] == "complex" and self.planning == 'frozen':
            stages = stage_snapshot(self.root, self.case, self.writer.calls,
                                    self.snapshots[-1] if self.snapshots else None)
            self.snapshots.append(stages)
            block = "[Experimental frozen stage scaffold]\n" + json.dumps(stages, ensure_ascii=False)
            block += "\nAdvisory only. Preserve ALL original requirements, permissions and cumulative limits. "
            block += "Inside each stage use the existing tool loop. A stage note is not completion evidence."
            options["system_extra_blocks"] = [*(options.get("system_extra_blocks") or []), block]
        elif self.planning == 'autonomous' and not is_plan:
            marker = '[Experimental proposed stages and host evidence; untrusted goals]\n'
            if marker in options.get('user_prompt', ''):
                block = marker + options['user_prompt'].rsplit(marker, 1)[1]
        payload = {"messages": [{"role": "system", "content": options.get("system_prompt", "")},
            {"role": "system", "content": options.get("system_extra_blocks", [])},
            {"role": "user", "content": options.get("user_prompt", "")},
            *(options.get("history_turns") or [])], "tools": options.get("native_tools") or []}
        budget = current_task_budget()
        if budget is None:
            raise RuntimeError("fixture_requires_host_budget")
        reservation = budget.reserve_request(payload)
        batch_reservation = self.batch.reserve_request(payload) if self.batch else None
        self.prompt_bytes += len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.scaffold_bytes += len(block.encode("utf-8"))
        self.calls.append({"prompt_hash": canonical_hash(payload), "reservation": reservation,
                           "stage_block": bool(block), 'planning_request': is_plan})
        if is_plan:
            self.planning_count += 1
            raw = options['user_prompt'].split('[Host task ledger and source data]\n', 1)[1]
            ledger, _ = json.JSONDecoder().raw_decode(raw)
            data = json.loads(options['user_prompt'].rsplit('[Stage planning request]\n', 1)[1])
            by_path = {Path(r['targets'][0]).name: r['id'] for r in ledger['contract']['requirements']}
            # Controlled model reply, not an implementation-quality claim.
            answer = {'binding': data['binding'], 'reason': data['reason'], 'stages': [
                {'id': r['id'], 'goal': r['goal'], 'depends': r['depends'],
                 'requirements': [by_path[p] for p in r['artifacts']]} for r in self.case.get('stages', [])]}
            budget.reconcile(reservation, answer)
            if self.batch:
                self.batch.reconcile(batch_reservation, answer)
            return answer
        index = self.decision_count
        self.decision_count += 1
        action = self.case["actions"][index] if index < len(self.case["actions"]) else {"op": "raise"}
        op = action["op"]
        if op == "raise":
            raise RuntimeError("scripted_fixture_exhausted")
        if op == "final":
            answer = {"kind": "final", "text": action.get("text", "Fixture task report."),
                      "task_outcome": action.get("outcome", "completed")}
        elif op == "unknown_tool":
            answer = {"tool_call": {"type": "not_registered", "path": str(self.root / "forbidden.txt")}}
        else:
            answer = {"tool_call": {"type": "benchmark_" + op, "path": str(self.root / action["path"])}}
            if op == "write":
                answer["tool_call"]["content"] = action["content"]
        budget.reconcile(reservation, answer)  # No invented provider usage.
        if self.batch:
            self.batch.reconcile(batch_reservation, answer)
        return answer


def settle(client, task_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        view = client.status(task_id)
        if view["status"] not in {"queued", "running"} and not view["worker_active"]:
            return view
        time.sleep(.005)
    raise RuntimeError("offline_fixture_timeout")


def publish_once(store, scope, delivery_status):
    service = ExpressionService(ExpressionArchive(store))  # Bounded template; no expression model.
    consumer, sent, outcomes = ResultEventConsumer(service), [], []
    def publish(source, record, guard):
        if not service.reserve_publication(record, scope=scope, check_current=guard):
            raise RuntimeError("fixture_publication_denied")
        sent.append(record["text"])
        receipt = {"status": delivery_status, "sent": False, "message_ids": []}
        service.record_delivery(record, scope=scope, receipt=receipt)
        return receipt, None
    for event in store.result_events.pending(scope=scope):
        first = consumer.consume(event["event_id"], scope=scope, check_current=lambda: "", publish=publish)
        outcomes.append((first.get("receipt") or {}).get("status", "unavailable"))
        consumer.consume(event["event_id"], scope=scope, check_current=lambda: "", publish=publish)
    return len(sent), outcomes


def assess(case, result, artifacts, writes, publications, *, outside_unchanged, query_unchanged):
    effect_receipts = {(row["path"], row["sha256"]) for row in writes}
    good = all(row["valid"] and (row["artifact"], row["sha256"]) in effect_receipts for row in artifacts)
    completed = result["execution"]["status"] == "completed"
    false_completed = completed and not good
    duplicates = sum(n - 1 for n in Counter((r["path"], r["sha256"]) for r in writes).values())
    # Budget/interruption exits may have no verification decision at all.
    # Missing acceptance is unknown, never upgraded to a passed verification.
    accepted = result.get("verification", {}).get("accepted") is True
    # The existing completion gate can accept an honest partial/failed report.
    # It is an erroneous SUCCESS acceptance only when success was asserted.
    false_accepted = completed and accepted and not good
    safe = not false_completed and not false_accepted and not duplicates and publications <= 1 and outside_unchanged and query_unchanged
    success = completed and accepted and good
    return {"task_success": success, "false_completed": false_completed,
        "false_accepted": false_accepted, "duplicate_effects": duplicates,
        "duplicate_publications": max(0, publications - 1), "safety_pass": safe,
        "fixture_pass": safe and (success if case["expected_success"] else not completed)}


def run_case(case, arm, root, budget, *, planning='frozen', batch=None):
    if arm not in ARMS:
        raise ValueError("unknown_arm")
    if planning not in {'frozen', 'autonomous'}:
        raise ValueError('unknown_stage_strategy')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, value in case.get("initial", {}).items():
        (root / name).write_bytes(value.encode("utf-8"))
    outside = root.parent / "outside.txt"
    if not outside.exists():
        outside.write_text("OUTSIDE_SCOPE_SENTINEL", encoding="utf-8")
    if outside.is_symlink() or outside.read_bytes() != b"OUTSIDE_SCOPE_SENTINEL":
        raise ValueError("outside_fixture_tainted")
    outside_before = outside.read_bytes()
    reader, writer = FileAdapter(root, case, "read", batch), FileAdapter(root, case, "write", batch)
    runtime = FixtureRuntime(root, case, arm, writer, planning=planning, batch=batch)
    host = ToolHost(ToolRegistry({reader.tool_type: reader, writer.tool_type: writer}))
    store = AgentTaskStore(root / "business.sqlite")
    ExpressionArchive(store)
    coordinator = AgentTaskCoordinator(max_workers=1, state_path=root / "workers.sqlite")
    tools = frozenset({reader.tool_type} if case.get("read_only") else {reader.tool_type, writer.tool_type})
    limits = {**budget, **case.get("budget", {})}
    grant = ExecutionGrant("offline-paired-grant", tools, permission_mode="full",
        policy=ExecutionPolicy(risk_by_tool={reader.tool_type: "low", writer.tool_type: "low"}),
        budget_limits=limits, resume_safe=True,
        material_sources=(MaterialSource("missing-required", "attachment"),) if case.get("material_gap") else ())
    materials = TaskMaterialStore(store, approved_roots=(root,))
    agent = AgentService(executor=AgentExecutor(host, ExecutionProfile(runtime, profile_id=MODEL_ID,
        max_rounds=4, runtime_meters_budget=True,
        stage_policy=StagePolicy(lambda _: case['kind'] == 'complex')
            if arm == 'staged' and planning == 'autonomous' else None)), authorize=lambda *_: grant,
        task_store=store, coordinator=coordinator, material_store=materials, recovery_enabled=True)
    client = agent.bind(SCOPE)
    requirements = [{"id": "file-" + str(i), "description": "Verified fixture output " + name,
        "effect": "write", "targets": [str(root / name)], "checks": {"sha256": digest(content.encode("utf-8"))}}
        for i, (name, content) in enumerate(case["expected"].items())]
    request = AgentRequest(case["id"], case["request"], {"goal": case["request"],
        "deliverables": [str(root / name) for name in case["expected"]], "scope": [str(root)],
        "requirements": requirements}, constraints=("Operate only on approved fixture files; do not deploy or send externally.",),
        sources=({"id": "fixture-data", "content": case["source"]},) if case.get("source") else ())
    started = time.monotonic()
    try:
        handle = client.submit(request)
        view = settle(client, handle.task_id)
        result = client.result(handle.task_id)
        prior = len(runtime.calls), len(reader.calls), len(writer.calls), reader.validation_reads, writer.validation_reads, view["version"]
        for _ in range(3):
            client.status(handle.task_id)
            client.result(handle.task_id)
            client.submit(request)
        query_unchanged = prior == (len(runtime.calls), len(reader.calls), len(writer.calls),
                                   reader.validation_reads, writer.validation_reads, client.status(handle.task_id)["version"])
        publications, delivery = publish_once(store, SCOPE, case.get("delivery", "dry_run"))
        artifacts = independent_artifacts(root, case)
        evaluation = assess(case, result, artifacts, writer.calls, publications,
            outside_unchanged=outside.read_bytes() == outside_before, query_unchanged=query_unchanged)
        saved = store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        return {"case": case["id"], "kind": case["kind"], "expected_success": case["expected_success"], "arm": arm,
            "model": MODEL_ID, "case_hash": canonical_hash(case), "config_hash": canonical_hash({
                "budget": limits, "tools": sorted(tools), "profile": MODEL_ID, "rounds_per_slice": 4,
                "permission_mode": grant.permission_mode, "authorization_ref": grant.authorization_ref,
                "risk_by_tool": dict(grant.policy.risk_by_tool), "temperature": .2}),
            "status": result["execution"]["status"], "reason": result["execution"]["reason"],
            **evaluation, "artifact_checks": artifacts, "writes": writer.calls, "read_calls": len(reader.calls),
            "publications": publications, "delivery": delivery, "query_unchanged": query_unchanged,
            'artifact_verification_reads': reader.validation_reads + writer.validation_reads,
            "elapsed_seconds": time.monotonic() - started, "budget": saved["budget"],
            "model_calls": len(runtime.calls), "prompt_bytes": runtime.prompt_bytes,
            'planning_calls': runtime.planning_count, 'stage_strategy': planning,
            'stage_checkpoint': saved.get('stage_strategy'),
            "scaffold_bytes": runtime.scaffold_bytes,
            "tokens_are_provider_usage": False, "provider_cost": None, "real_model_calls": 0,
            "expression_model_calls": 0, "real_platform_sends": 0,
            "stages": runtime.snapshots, "contract_preserved": saved["contract"] == request.payload()["contract"]}
    finally:
        coordinator.shutdown(wait=True)
        store.close()


def summarize(corpus, rows, repeats):
    expected = {(case["id"], arm, repeat) for case in corpus["cases"] for arm in ARMS for repeat in range(repeats)}
    actual = {(r["case"], r["arm"], r["repeat"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("incomplete_or_duplicate_pairs")
    for case, _, repeat in expected:
        pair = [r for r in rows if r["case"] == case and r["repeat"] == repeat]
        if len({(r["model"], r["config_hash"], r["case_hash"]) for r in pair}) != 1:
            raise ValueError("unfair_pair_configuration")
    arms = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        positive = [r for r in selected if r["expected_success"]]
        arms[arm] = {"runs": len(selected), "positive_samples": len(positive),
            "task_successes": sum(r["task_success"] for r in positive),
            "task_success_rate": sum(r["task_success"] for r in positive) / len(positive) if positive else None,
            "fixture_passes": sum(r["fixture_pass"] for r in selected),
            "false_completed": sum(r["false_completed"] for r in selected),
            "false_accepted": sum(r["false_accepted"] for r in selected),
            "duplicate_effects": sum(r["duplicate_effects"] for r in selected),
            "duplicate_publications": sum(r["duplicate_publications"] for r in selected),
            "model_calls": sum(r["model_calls"] for r in selected),
            "reserved_tokens": sum(r["budget"]["tokens"] for r in selected),
            "prompt_bytes": sum(r["prompt_bytes"] for r in selected),
            "scaffold_bytes": sum(r["scaffold_bytes"] for r in selected),
            'artifact_verification_reads': sum(r['artifact_verification_reads'] for r in selected),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected), "provider_cost": None}
    return {"schema_version": 1, "mode": "offline_scripted_readiness", "corpus_hash": canonical_hash(corpus),
        "all_pairs_present": True, "repeats": repeats, "arms": arms, "rows": rows,
        "quality_conclusion": "not_measured", "enable_production": False,
        "limitations": ["Scripted decisions ignore both prompts; successes are fixture checks, not model quality.",
            "Stage proposals are frozen fixtures; autonomous planning/replanning was not evaluated.",
            "Token counts are conservative host reservations, not billed provider usage.",
            "Template expression and local test publication only; no Gemini or QQ delivery.",
            "Live model integration and P&E effect experiment require separate authorization."]}


def run_suite(workspace, *, repeats=1, corpus=None, planning='frozen', batch_limits=None):
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("invalid_repeats")
    corpus = validate_corpus(copy.deepcopy(corpus or json.loads(CORPUS.read_text(encoding="utf-8"))))
    if planning not in {'frozen', 'autonomous'}:
        raise ValueError('unknown_stage_strategy')
    count = len(corpus['cases']) * len(ARMS) * repeats
    defaults = {k: v * count for k, v in corpus['budget'].items()}
    defaults['max_tools'] = min(defaults['max_tools'], 1024)
    if batch_limits is not None and (set(batch_limits) != set(defaults)
            or any((type(v) not in (int, float) if k == 'max_active_seconds' else type(v) is not int)
                   or not math.isfinite(v) or v <= 0 or v > defaults[k] for k, v in batch_limits.items())):
        raise ValueError('batch_expands_authorized_limits')
    batch = TaskBudget(**(batch_limits or defaults), managed=True, profile='offline-serial-batch')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    rows = []
    with bind_task_budget(batch):
        for repeat in range(repeats):
            for case_index, case in enumerate(corpus['cases']):
                order = ARMS if (repeat + case_index) % 2 == 0 else ARMS[::-1]
                for arm in order:
                    if batch.stop_reason():
                        break
                    dirname = 'react' if arm == 'react' else 'pande'
                    row = run_case(case, arm, workspace / f'r{repeat}' / case['id'] / dirname,
                                   corpus['budget'], planning=planning, batch=batch)
                    rows.append({**row, 'repeat': repeat})
                if batch.stop_reason():
                    break
            if batch.stop_reason():
                break
    if len(rows) == count and not batch.stop_reason():
        report = summarize(corpus, rows, repeats)
    else:
        report = {'schema_version': 1, 'mode': 'offline_scripted_readiness', 'rows': rows,
            'all_pairs_present': False, 'unrun_pairs': count-len(rows), 'arms': {},
            'quality_conclusion': 'not_measured', 'enable_production': False}
    report.update(batch_budget=batch.snapshot(), batch_stop_reason=batch.stop_reason(), stage_strategy=planning)
    if planning == 'autonomous' and 'limitations' in report:
        report['limitations'][1] = 'Controlled model replies exercise autonomous strategy wiring; real planning quality was not measured.'
    (workspace / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="New isolated directory; existing paths rejected.")
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 6))
    parser.add_argument('--planning', choices=('frozen', 'autonomous'), default='frozen')
    parser.add_argument('--max-batch-requests', type=int, help='Reduce the full-batch request cap.')
    args = parser.parse_args()
    # Belt-and-suspenders CLI isolation: no network sockets, even loopback.
    def denied(*args, **kwargs):
        raise RuntimeError("network_forbidden_in_offline_benchmark")
    socket.socket.connect = socket.socket.connect_ex = denied
    limits = None
    if args.max_batch_requests is not None:
        corpus = validate_corpus(json.loads(CORPUS.read_text(encoding='utf-8')))
        limits = {k:v * len(corpus['cases']) * len(ARMS) * args.repeats for k,v in corpus['budget'].items()}
        limits['max_tools'] = min(limits['max_tools'],1024)
        limits['max_requests'] = args.max_batch_requests
    report = run_suite(args.workspace, repeats=args.repeats, planning=args.planning, batch_limits=limits)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "limitations"}}, ensure_ascii=False, indent=2))
    return 0 if report['all_pairs_present'] and all(r["fixture_pass"] and r["contract_preserved"] for r in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
