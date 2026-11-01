"""Explicit recovery with real SQLite, file effects and independent verification."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from shinku.agent.budget import TaskBudget
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime import AgentBoundaryError, AgentScope, TaskHandle
from shinku.agent_runtime.demo import DemoFileTool, demo_request
from shinku.agent_runtime.execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialStore
from shinku.agent_runtime.recovery import TaskRecoveryGate, record_recovery
from shinku.agent_runtime.result_events import ResultEventStore
from shinku.agent_runtime.service import AgentService
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

SCOPE = AgentScope("fixture", "local:fixture")


class Runtime:
    def __init__(self, decisions=()):
        self.decisions, self.calls = list(decisions), []
    def chat_supports_native_tools(self):
        return True
    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.decisions:
            raise RuntimeError("isolated provider unavailable")
        result = self.decisions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def write(tool):
    return {"tool_call": {"type": tool.tool_type, "path": str(tool.path)}}


FINAL = {"kind": "final", "text": "The isolated file was created and verified.", "task_outcome": "completed"}


def service_at(root, runtime=None, *, tool=None, materials=False):
    root.mkdir(exist_ok=True)
    tool = tool or DemoFileTool(root)
    runtime = runtime or Runtime()
    store = AgentTaskStore(root / "business.sqlite")
    ResultEventStore(store)
    material_store = TaskMaterialStore(store, approved_roots=(root,)) if materials else None
    grant = ExecutionGrant("fixture-grant", frozenset({tool.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type: "low"}),
        budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30},
        resume_safe=True)
    service = AgentService(executor=AgentExecutor(ToolHost(ToolRegistry({tool.tool_type: tool})), ExecutionProfile(runtime)),
        authorize=lambda *_: grant, task_store=store,
        coordinator=AgentTaskCoordinator(max_workers=1, state_path=root / "workers.sqlite"),
        material_store=material_store, recovery_enabled=True)
    return service, service.bind(SCOPE), tool, runtime


def close(service):
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


def settle(client, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        view = client.status(task_id)
        if view["status"] not in {"queued", "running"} and not view["worker_active"]:
            return view
        time.sleep(.005)
    raise AssertionError("isolated recovery worker did not settle")


def handle(view):
    return TaskHandle(view["task_id"], view["attempt"], view["version"])


def failed_write(root):
    tool = DemoFileTool(root)
    service, client, tool, runtime = service_at(root, Runtime([write(tool), RuntimeError("fixture failure")]), tool=tool)
    initial = client.submit(demo_request(tool))
    view = settle(client, initial.task_id)
    assert tool.calls == 1 and view["status"] == "failed"
    task = service.task_store.get(initial.task_id, requester=SCOPE.requester, session=SCOPE.session)
    return service, client, tool, runtime, task


@pytest.mark.parametrize("restart", [False, True])
def test_verified_file_is_preserved_and_repeat_is_blocked_across_explicit_resume(tmp_path, restart):
    first, client, old_tool, _, before = failed_write(tmp_path)
    if restart:
        close(first)
        tool = DemoFileTool(tmp_path)
        service, client, tool, runtime = service_at(tmp_path, Runtime([write(tool), FINAL]), tool=tool)
    else:
        service, tool = first, old_tool
        runtime = Runtime([write(tool), FINAL])
        service.executor.profile = ExecutionProfile(runtime)
    try:
        current = handle(client.status(before["task_id"]))
        proof = client.recovery(current)
        assert proof["status"] == "continuable" and proof["checks"][0]["status"] == "preserve_effect"
        assert service.task_store.get(before["task_id"], requester=SCOPE.requester, session=SCOPE.session) == before
        client.resume(current, request_key="resume-1")
        assert settle(client, current.task_id)["status"] == "completed"
        after = service.task_store.get(current.task_id, requester=SCOPE.requester, session=SCOPE.session)
        assert old_tool.calls == 1 and (tool.calls == 0 if restart else tool.calls == 1)
        assert before["contract"] == after["contract"] and before["requests"] == after["requests"]
        assert after["budget"]["requests"] == before["budget"]["requests"] + 2
        assert after["budget"]["tools"] == before["budget"]["tools"]
        assert any(row.get("executed") is False for row in after["tool_calls"])
        assert after["recovery_history"][0]["original_stop_reason"] == before["stop_reason"]
        assert after["recovery_history"][0]["recovery_reason"] == ("service_restart" if restart else "explicit_continuation")
        assert client.result(current.task_id)["verification"]["accepted"]
        assert client.resume(current, request_key="resume-1").attempt == 2
        assert len(runtime.calls) == 2
    finally:
        close(service)


@pytest.mark.parametrize("change", ["alter", "remove", "revoke"])
def test_missing_changed_or_denied_effect_never_starts_recovery(tmp_path, change):
    service, client, tool, runtime, before = failed_write(tmp_path)
    if change == "alter":
        tool.path.write_text("different result", encoding="utf-8")
    elif change == "remove":
        tool.path.unlink()
    else:
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), still_authorized=lambda: False)
    try:
        with pytest.raises(AgentBoundaryError, match="recovery_effect_unconfirmed|host_authorization_denied"):
            client.resume(handle(client.status(before["task_id"])), request_key="retry")
        assert service.task_store.get(before["task_id"], requester=SCOPE.requester, session=SCOPE.session) == before
        assert tool.calls == 1 and len(runtime.calls) == 2
    finally:
        close(service)


@pytest.mark.parametrize("bucket", ["inflight", "uncertain_attempts", "tool_calls"])
def test_possible_write_effect_is_review_required_even_if_file_exists(tmp_path, bucket):
    service, client, tool, runtime, task = failed_write(tmp_path)
    task[bucket] = [{"name": tool.tool_type, "id": "unknown-call", "arguments": {"path": str(tool.path)}, "status": "unknown"}]
    task = service.task_store.save(task)
    try:
        report = client.recovery(TaskHandle(task["task_id"], task["execution_attempt"], task["revision"]))
        assert report["status"] == "review_required" and "recovery_effect_unknown" in report["reasons"]
        with pytest.raises(AgentBoundaryError, match="recovery_effect_unknown"):
            client.resume(TaskHandle(task["task_id"], task["execution_attempt"], task["revision"]), request_key="retry")
        assert tool.calls == 1 and len(runtime.calls) == 2
        assert service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task
    finally:
        close(service)


@pytest.mark.parametrize("counter,limit", [("requests", "max_requests"), ("tokens", "max_tokens"),
                                          ("tools", "max_tools"), ("active_seconds", "max_active_seconds")])
def test_insufficient_cumulative_budget_does_not_start_or_reset(tmp_path, counter, limit):
    service, client, tool, _, task = failed_write(tmp_path)
    task["budget"][counter] = task["budget"][limit]
    task = service.task_store.save(task)
    try:
        with pytest.raises(AgentBoundaryError, match="recovery_budget_insufficient"):
            client.resume(TaskHandle(task["task_id"], task["execution_attempt"], task["revision"]), request_key="retry")
        assert service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)["budget"] == task["budget"]
        assert tool.calls == 1
    finally:
        close(service)


def test_concurrent_services_claim_one_recovery_and_stale_writer_is_rejected(tmp_path):
    first, _, tool, _, task = failed_write(tmp_path)
    close(first)
    a, ca, ta, ra = service_at(tmp_path, Runtime([FINAL]))
    b, cb, tb, rb = service_at(tmp_path, Runtime([FINAL]))
    current = TaskHandle(task["task_id"], 1, task["revision"])
    def attempt(client):
        try:
            return client.resume(current, request_key="same-request").attempt
        except (AgentBoundaryError, ValueError) as exc:
            return str(exc)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, [ca, cb]))
        assert 2 in outcomes
        view = settle(ca if a.coordinator.get(task["task_id"])["attempt"] == 2 else cb, task["task_id"])
        assert view["status"] == "completed"
        after = a.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        assert after["execution_attempt"] == 2 and len(after["recovery_history"]) == 1
        assert len(ra.calls) + len(rb.calls) == 1 and ta.calls + tb.calls == 0
        with pytest.raises(AgentBoundaryError, match="stale_execution_attempt"):
            a._write(SCOPE, task["task_id"], 1, lambda row: row.update(status="completed"))
    finally:
        close(a)
        close(b)


@pytest.mark.parametrize("field", ["host_grant", "budget"])
def test_legacy_missing_grant_or_budget_is_explicitly_rejected(tmp_path, field):
    service, client, _, _, task = failed_write(tmp_path)
    task.pop(field)
    task = service.task_store.save(task)
    try:
        with pytest.raises(AgentBoundaryError, match="resume_legacy_grant_unavailable|resume_budget_unavailable"):
            client.resume(TaskHandle(task["task_id"], 1, task["revision"]), request_key="retry")
    finally:
        close(service)


@pytest.mark.parametrize("other", [AgentScope("other", SCOPE.session), AgentScope(SCOPE.requester, "local:other")])
def test_cross_scope_cannot_review_or_resume(tmp_path, other):
    service, client, _, _, task = failed_write(tmp_path)
    foreign = service.bind(other)
    current = TaskHandle(task["task_id"], 1, task["revision"])
    try:
        for action in (lambda: foreign.recovery(current), lambda: foreign.resume(current, request_key="r")):
            with pytest.raises(AgentBoundaryError, match="task_not_found"):
                action()
    finally:
        close(service)


def test_admission_failure_keeps_stop_history_and_duplicate_cannot_queue_again(tmp_path, monkeypatch):
    service, client, tool, _, before = failed_write(tmp_path)
    monkeypatch.setattr(service.coordinator, "submit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("queue rejected")))
    current = TaskHandle(before["task_id"], 1, before["revision"])
    try:
        with pytest.raises(RuntimeError, match="queue rejected"):
            client.resume(current, request_key="r")
        after = service.task_store.get(current.task_id, requester=SCOPE.requester, session=SCOPE.session)
        assert after["status"] == "blocked" and after["stop_reason"] == "recovery_admission_failed"
        assert after["recovery_history"][0]["original_stop_reason"] == before["stop_reason"]
        assert after["budget"] == before["budget"] and tool.calls == 1
        assert client.resume(current, request_key="r").attempt == 2
    finally:
        close(service)


def test_file_change_while_queued_is_checked_before_any_planner_call(tmp_path, monkeypatch):
    service, client, tool, _, before = failed_write(tmp_path)
    runtime = Runtime([FINAL])
    service.executor.profile = ExecutionProfile(runtime)
    original = service.coordinator.dispatch
    def dispatch(*args, **kw):
        tool.path.write_text("tampered during admission", encoding="utf-8")
        return original(*args, **kw)
    monkeypatch.setattr(service.coordinator, "dispatch", dispatch)
    try:
        client.resume(TaskHandle(before["task_id"], 1, before["revision"]), request_key="r")
        assert settle(client, before["task_id"])["status"] == "blocked"
        assert client.result(before["task_id"])["execution"]["reason"] == "recovery_effect_unconfirmed"
        assert runtime.calls == [] and tool.calls == 1
    finally:
        close(service)


@pytest.fixture
def ledger(tmp_path):
    clock = [1000.0]
    store = AgentTaskStore(tmp_path / "task.sqlite", clock=lambda: clock[0])
    tool = DemoFileTool(tmp_path)
    task = new_task(task_id="fixture-task", requester=SCOPE.requester, session=SCOPE.session,
        request="Create an isolated file", message_id="fixture-message", contract=demo_request(tool).contract)
    task.update(execution_attempt=1, status="blocked", stop_reason="interrupted", budget=TaskBudget().snapshot())
    task = store.save(task)
    yield store, task, tool, clock
    store.close()


def review(ledger, **kwargs):
    store, task, tool, _ = ledger
    return TaskRecoveryGate(store, {tool.tool_type: tool}, **kwargs).review(task,
        context={}, still_authorized=lambda: True)


@pytest.mark.parametrize("effect", ["send", "delete", "process_stop", "reminder", "command_run"])
def test_external_or_unverifiable_effects_require_review(ledger, effect):
    store, task, tool, _ = ledger
    task["tool_calls"] = [{"name": tool.tool_type, "arguments": {}, "status": "ok",
        "data": {"state_updates": {"verified_effects": [effect], "delivery_receipt": "receipt-is-not-current-proof"}}}]
    ledger = (store, store.save(task), tool, ledger[3])
    assert review(ledger)["status"] == "review_required"


@pytest.mark.parametrize("change", ["expire", "version", "revoke", "origin", "permission", "missing_store", "incomplete"])
def test_material_binding_checked_without_spending_reads(ledger, tmp_path, change):
    store, task, tool, clock = ledger
    materials = TaskMaterialStore(store, policy=MaterialPolicy(ttl_seconds=10), approved_roots=(tmp_path,))
    path = tmp_path / "input.txt"
    path.write_text("original source", encoding="utf-8")
    task["material_package"] = materials.package(task, [MaterialSource("input", "attachment", local_path=path,
        complete=change != "incomplete")], authorization_ref="fixture")
    task = store.save(task)
    before = materials.limits(task)
    if change == "expire":
        clock[0] += 11
    elif change == "version":
        task["material_package"]["items"][0]["version"] = "new-version"
        task = store.save(task)
    elif change == "revoke":
        materials.revoke(task["material_package"]["items"][0]["reference"], task=task)
    elif change == "origin":
        path.write_text("replacement source", encoding="utf-8")
    authorized = lambda: change != "permission"
    gate = TaskRecoveryGate(store, {tool.tool_type: tool}, material_store=None if change == "missing_store" else materials)
    if change == "permission":
        with pytest.raises(AgentBoundaryError, match="host_authorization_denied"):
            gate.review(task, context={}, still_authorized=authorized)
    else:
        assert gate.review(task, context={}, still_authorized=authorized)["status"] == "review_required"
    if change != "expire":
        assert materials.limits(task)["reads"] == before["reads"]


def test_intact_retained_sources_are_checked_without_replacing_or_reading_pages(ledger):
    store, task, tool, _ = ledger
    materials = TaskMaterialStore(store, policy=MaterialPolicy(inline_chars=8))
    task["material_package"] = materials.package(task, [MaterialSource("long", "document", "original tail" * 40)], authorization_ref="fixture")
    task = store.save(task)
    before = materials.limits(task)
    proof = TaskRecoveryGate(store, {tool.tool_type: tool}, material_store=materials).review(task, context={}, still_authorized=lambda: True)
    assert proof["status"] == "continuable"
    assert materials.limits(task) == before
    assert store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task


@pytest.mark.parametrize("bound", ["attempts", "checks", "deadline"])
def test_recovery_is_bounded_and_preserves_source(ledger, bound):
    store, task, tool, _ = ledger
    options = {}
    if bound == "attempts":
        task["recovery_history"] = [{}] * 16
    else:
        task["tool_calls"] = [{"name": tool.tool_type, "arguments": {}, "status": "error"}] * 2
        if bound == "checks":
            options["max_checks"] = 1
        else:
            ticks = iter([0, 11, 12, 13])
            options["clock"] = lambda: next(ticks, 14)
    task = store.save(task)
    result = TaskRecoveryGate(store, {tool.tool_type: tool}, **options).review(task, context={}, still_authorized=lambda: True)
    expected = {"attempts": "recovery_attempt_limit", "checks": "recovery_check_limit", "deadline": "recovery_check_deadline"}[bound]
    assert expected in result["reasons"]
    assert result["status"] == "review_required"
    assert store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task


def test_effect_verifier_cannot_race_task_identity_or_leak_exception(ledger, monkeypatch):
    store, task, tool, _ = ledger
    result = tool.execute(call={"path": str(tool.path)}, context={})
    task["tool_calls"] = [{"name": tool.tool_type, "arguments": {}, "status": "ok", "data": result}]
    task = store.save(task)
    original = tool.agent_verify_result
    def raced(**kwargs):
        current = store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        current["notes"]["blocker"] = "newer revision"
        store.save(current)
        return original(**kwargs)
    monkeypatch.setattr(tool, "agent_verify_result", raced)
    with pytest.raises(AgentBoundaryError, match="recovery_binding_changed"):
        TaskRecoveryGate(store, {tool.tool_type: tool}).review(task, context={}, still_authorized=lambda: True)


def test_completed_result_routes_only_to_existing_delivery_and_never_reexecutes(tmp_path):
    tool = DemoFileTool(tmp_path)
    service, client, tool, runtime = service_at(tmp_path, Runtime([write(tool), FINAL]), tool=tool)
    try:
        initial = client.submit(demo_request(tool))
        view = settle(client, initial.task_id)
        before = tool.calls, len(runtime.calls), client.result(initial.task_id)
        report = client.recovery(handle(view))
        assert report["status"] == "delivery_only" and not report["automatic_send"]
        assert report["delivery"]["event_status"] == "pending"
        with pytest.raises(AgentBoundaryError, match="task_not_resumable"):
            client.resume(handle(view), request_key="r")
        assert before == (tool.calls, len(runtime.calls), client.result(initial.task_id))
    finally:
        close(service)


def test_expired_business_record_cannot_recover_from_historical_event(ledger):
    store, task, tool, clock = ledger
    clock[0] += store.ttl_seconds + 1
    with pytest.raises(AgentBoundaryError, match="recovery_task_expired_or_unavailable"):
        review(ledger)


def test_unknown_read_is_retained_as_unknown_audit_and_only_replanned(ledger):
    store, task, tool, _ = ledger
    class Reader:
        agent_permission_class = "read"
        agent_completion_effects = ("read",)
    task["inflight"] = [{"name": "read", "id": "unknown-read", "arguments": {"path": "fixture"}, "status": "unknown"}]
    task = store.save(task)
    proof = TaskRecoveryGate(store, {"read": Reader()}).review(task, context={}, still_authorized=lambda: True)
    assert proof["status"] == "continuable"
    record_recovery(task, proof, request_key="r", reason="restart")
    after = store.save(task)
    assert after["inflight"] == [] and after["uncertain_attempts"] == []
    assert after["recovery_uncertain_reads"][0]["status"] == "unknown"
    assert after["tool_calls"] == [] and after["completion"] == {}


@pytest.mark.parametrize("stage", ["during_write", "after_receipt", "after_verification", "after_commit"])
def test_actual_process_crash_keeps_receipts_and_recovery_never_replays_command(tmp_path, stage):
    import os
    child = tmp_path / "crash.py"
    child.write_text('''import os, sys, time
from pathlib import Path
from test_agent_recovery import service_at, Runtime, DemoFileTool, write, FINAL
from shinku.agent_runtime.demo import demo_request
root, stage = Path(sys.argv[1]), sys.argv[2]
tool = DemoFileTool(root)
service, client, tool, runtime = service_at(root, Runtime([write(tool), FINAL]), tool=tool)
if stage == "during_write":
    original = tool.execute
    def crash(**kw):
        original(**kw)
        os._exit(43)
    tool.execute = crash
elif stage in {"after_receipt", "after_verification"}:
    original = service._observe
    def crash(task_id, session, state, status):
        original(task_id, session, state, status)
        if ((stage == "after_receipt" and state.tool_calls and not state.inflight_tool_calls)
                or (stage == "after_verification" and status == "completed")):
            os._exit(43)
    service._observe = crash
else:
    original = service._write
    def crash(*args, **kwargs):
        saved = original(*args, **kwargs)
        if saved.get("execution_result", {}).get("verification", {}).get("accepted"):
            os._exit(43)
        return saved
    service._write = crash
client.submit(demo_request(tool))
time.sleep(5)
os._exit(99)
''', encoding="utf-8")
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "tests")])
    env["SHINKU_ENV_FILE"] = str(tmp_path / "missing.env")
    crashed = subprocess.run([sys.executable, "-X", "utf8", "-B", str(child), str(tmp_path), stage],
        env=env, cwd=tmp_path, capture_output=True, timeout=12)
    assert crashed.returncode == 43, crashed.stderr.decode("utf-8")
    tool = DemoFileTool(tmp_path)
    service, client, tool, runtime = service_at(tmp_path, Runtime([write(tool), FINAL]), tool=tool)
    try:
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        current = TaskHandle(task["task_id"], task["execution_attempt"], task["revision"])
        assert tool.path.read_text(encoding="utf-8") == "standalone agent verified"
        assert service.coordinator.get(task["task_id"])["metadata"]["recovered_after_restart"]
        if stage == "during_write":
            with pytest.raises(AgentBoundaryError, match="recovery_effect_unknown"):
                client.resume(current, request_key="r")
            assert task["inflight"] and not task["tool_calls"]
        elif stage == "after_commit":
            assert client.recovery(current)["status"] == "delivery_only"
            with pytest.raises(AgentBoundaryError, match="execution_already_committed"):
                client.resume(current, request_key="r")
        else:
            if stage == "after_verification":
                assert task["completion"]["accepted"] and task["result_commit_pending"] == "completed"
                assert not task.get("execution_result")
            client.resume(current, request_key="r")
            assert settle(client, task["task_id"])["status"] == "completed"
            assert client.result(task["task_id"])["verification"]["accepted"]
            fresh = service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
            assert fresh["recovery_history"][0]["prior_execution_status"] == task["status"]
        assert tool.calls == 0
        assert len(runtime.calls) == (2 if stage in {"after_receipt", "after_verification"} else 0)
    finally:
        close(service)


def test_approval_cannot_repeat_a_recorded_file_effect(tmp_path):
    from shinku.agent.loop import AgentLoop
    from shinku.agent.budget import bind_task_budget
    from shinku.tools.invocation import ToolInvocation
    tool = DemoFileTool(tmp_path)
    args = {"path": str(tool.path)}
    receipt = tool.execute(call=args, context={})
    old = {"name": tool.tool_type, "arguments": args, "status": "ok", "data": receipt}
    with bind_task_budget(TaskBudget(managed=True)):
        result = AgentLoop(planner=lambda **kw: (_ for _ in ()).throw(AssertionError("must not plan")),
            handlers={tool.tool_type: tool}, sliced=True).run(
                context={"agent_recovery_resume": True, "agent_restored_tool_calls": [old]},
                approved_invocations=[ToolInvocation(tool.tool_type, args)])
    assert result.reason == "approved_effect_already_recorded"
    assert tool.calls == 1 and tool.path.read_bytes() == b"standalone agent verified"


@pytest.mark.parametrize("classification,blocked", [(0, False), (1, True), (None, True)])
def test_recovery_distinguishes_optional_omission_from_missing_required(ledger, classification, blocked):
    """仅非必需记录省略不能阻塞恢复；旧未知分类和真实必需缺项仍保护。"""
    store, task, tool, _ = ledger
    materials = TaskMaterialStore(store)
    task["material_package"] = materials.package(task, [], authorization_ref="fixture")
    package = task["material_package"]
    package["omitted_gap_records"] = 3
    if classification is None:
        package.pop("omitted_required_gap_records")
    else:
        package["omitted_required_gap_records"] = classification
    task = store.save(task)
    proof = TaskRecoveryGate(store, {tool.tool_type: tool}, material_store=materials).review(
        task, context={}, still_authorized=lambda: True)
    assert ("recovery_required_material_missing" in proof.get("reasons", [])) is blocked
    if not blocked:
        assert proof["status"] == "continuable"
