"""P4b business/event/receipt facts -> one bounded historical memory card."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.materials import TaskMaterialStore, MaterialSource, MaterialPolicy
from shinku.agent_runtime.results import export_task_result
from shinku.agent_runtime.result_events import ResultEventStore
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.experience_facts import DeploymentEvidence
from shinku.memory.source_reader import MemorySourceReader
from shinku.memory.task_experience import TaskExperienceProjector
from shinku.result_expression import ExpressionArchive, ExpressionService, ExpressionProfile
from test_result_events import SCOPE, terminal
from test_result_expression import Model


def setup(tmp_path, *, material=False, max_versions=4096):
    clock = [2_000_000_000.0]
    tasks = AgentTaskStore(tmp_path / "tasks.db", clock=lambda: clock[0])
    events = ResultEventStore(tasks)
    archive = ExpressionArchive(tasks)
    store = MemoryStore(tmp_path / "memory.db")
    memory = MemoryService(store=store, router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=(SCOPE.requester,)), clock=lambda: clock[0])
    materials = TaskMaterialStore(tasks, policy=MaterialPolicy(ttl_seconds=20), approved_roots=(tmp_path,)) if material else None
    projector = TaskExperienceProjector(memory, tasks, expression_archive=archive,
        material_store=materials, max_versions=max_versions)
    return tasks, events, archive, store, memory, projector, materials, clock


def project(projector, task_id="fixture-task", **kw):
    return projector.project(task_id=task_id, scope=SCOPE, still_authorized=lambda: True, **kw)


def card(store, task_id="fixture-task"):
    return store.event_by_source_id(source_id="owner:agent-task:" + task_id, scope=MemoryService.OWNER_SCOPE)


def finish_delivery(tasks, archive, status, *, sent=None):
    task = tasks.get("fixture-task", requester=SCOPE.requester, session=SCOPE.session)
    result = task["execution_result"]
    service = ExpressionService(archive, ExpressionProfile(Model(), configuration_id="fixture"))
    record = service.prepare(result, scope=SCOPE, check_current=lambda: "")
    assert service.reserve_publication(record, scope=SCOPE, check_current=lambda: "")
    return service.record_delivery(record, scope=SCOPE, receipt={"status": status,
        "sent": status == "sent" if sent is None else sent, "message_ids": ["fixture-out"] if status in {"sent", "partial"} else []})


@pytest.mark.parametrize("status", ["completed", "failed", "blocked", "partial", "canceled", "unverified"])
def test_distinct_execution_states_never_use_model_report(tmp_path, status):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    task = terminal(tasks, status=status)
    task["last_report"] = "我成功完成全部并已上线，MODEL_REPORT_SECRET"
    task["execution_result"]["execution_report"]["text"] = "我成功完成全部并已上线，MODEL_REPORT_SECRET"
    task["execution_result"].pop("result_hash")
    from shinku.agent_runtime.results import seal_result
    task["execution_result"] = seal_result(task["execution_result"])
    tasks.save(task)
    result = project(projector, deployment=DeploymentEvidence("not_deployed", "fixture-review"))
    assert result["status"] == "stored", result
    record = card(store)
    snapshot = record.metadata["snapshot"]
    assert snapshot["execution_status"] == status
    assert snapshot["verified"] is (status == "completed")
    assert snapshot["deployment"]["status"] == "not_deployed"
    assert "MODEL_REPORT_SECRET" not in json.dumps(record.as_dict(), ensure_ascii=False)
    assert "已上线" not in record.text and "宿主确认未上线" in record.text
    assert "不证明当前项目状态" in record.text
    assert memory.inspect_agent_task(requester_id=SCOPE.requester, task_id=task["task_id"])["found"]
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM memory_facts").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory_lifecycle_evidence").fetchone()[0] == 0


@pytest.mark.parametrize("status", ["sent", "failed", "uncertain", "partial", "dry_run", "suppressed", "alien"])
def test_delivery_and_execution_are_independent_and_durable(tmp_path, status):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    assert project(projector)["status"] == "stored"
    first = card(store).metadata["projection_hash"]
    finish_delivery(tasks, archive, status)
    assert not projector.validate_record(card(store))
    assert project(projector)["status"] == "stored"
    snapshot = card(store).metadata["snapshot"]
    expected = "uncertain" if status == "alien" else status
    assert snapshot["verified"] and snapshot["delivery"]["status"] == expected
    assert snapshot["delivery"]["sent"] is (status == "sent")
    assert card(store).metadata["supersedes"] == first
    assert len(projector.versions.history(task_id="fixture-task", scope=memory.owner_scope)) == 2
    assert project(projector)["status"] == "duplicate"


def test_sent_label_without_confirmed_receipt_remains_unknown(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    finish_delivery(tasks, archive, "sent", sent=False)
    assert project(projector)["status"] == "stored"
    assert card(store).metadata["snapshot"]["delivery"]["status"] == "uncertain"
    assert "已确认送达" not in card(store).text


def test_reply_delivery_does_not_become_verified_goal_in_historical_memory(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    task = terminal(tasks)
    task['completion']['layers'] = {'policy': 'agent_reply_delivery', 'goal_completion': 'not_verified'}
    task['execution_result'] = export_task_result({**task, 'revision': 1, 'updated': tasks.clock()})
    tasks.save(task)
    result = project(projector)
    assert result['status'] == 'stored', result
    record = card(store)
    assert record.metadata['snapshot']['verified'] is False
    assert record.metadata['snapshot']['delivery_policy'] == 'agent_reply_delivery'
    assert '答复已交付；整项目标未独立验收' in record.text
    assert '目标验收通过' not in record.text


def test_parallel_consumers_commit_one_card_and_one_version(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    peers = [projector]
    for _ in range(7):
        ms = MemoryStore(store.path)
        service = MemoryService(store=ms, router=MemoryRouter(), policy=memory.policy, clock=memory.clock)
        peers.append(TaskExperienceProjector(service, tasks, expression_archive=archive))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(project, peers))
    assert sum(r["status"] == "stored" for r in results) == 1, results
    assert sum(r["status"] == "duplicate" for r in results) == 7
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM memory_task_experience_versions").fetchone()[0] == 1
    assert events.pending(scope=SCOPE)[0]["status"] == "pending"


@pytest.mark.parametrize("tamper", ["scope", "version", "hash", "status", "accepted", "event", "permission"])
def test_untrusted_or_stale_sources_cannot_write(tmp_path, tamper):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    if tamper == "permission":
        result = projector.project(task_id="fixture-task", scope=SCOPE, still_authorized=lambda: False)
    elif tamper == "scope":
        result = projector.project(task_id="fixture-task", scope=AgentScope("other", SCOPE.session), still_authorized=lambda: True)
    else:
        with tasks._transaction() as db:
            if tamper == "event":
                db.execute("DELETE FROM agent_result_events")
            else:
                raw = db.execute("SELECT payload FROM agent_conversation_tasks").fetchone()[0]
                task = json.loads(raw)
                if tamper == "version":
                    task["revision"] += 1
                elif tamper == "hash":
                    task["execution_result"]["result_hash"] = "0" * 64
                elif tamper == "status":
                    task["status"] = "failed"
                else:
                    task["completion"]["accepted"] = False
                db.execute("UPDATE agent_conversation_tasks SET payload=?", (json.dumps(task),))
        result = project(projector)
    assert result["status"] == "rejected", result
    assert card(store) is None


@pytest.mark.parametrize("purge", [False, True])
def test_forget_revokes_future_versions_and_clears_archived_bodies(tmp_path, purge):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    task = tasks.save(terminal(tasks))
    project(projector)
    old = card(store)
    store.forget(query="", record_id=old.record_id, scope=memory.owner_scope, purge=purge, now=int(memory.clock()))
    assert project(projector)["reason"] == "experience_revoked"
    task.update(status="failed", completion={"accepted": False})
    task["execution_result"] = export_task_result({**task, "revision": task["revision"] + 1, "updated": tasks.clock()})
    tasks.save(task)
    assert project(projector)["reason"] == "experience_revoked"
    assert projector.versions.history(task_id="fixture-task", scope=memory.owner_scope) == []
    with store._connect() as db:
        assert db.execute("SELECT payload FROM memory_task_experience_versions").fetchone()[0] == "{}"
    assert not projector.validate_record(old)


def test_correction_supersedes_card_without_rewriting_task_history_or_facts(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    task = tasks.save(terminal(tasks))
    source = copy.deepcopy(task["execution_result"])
    project(projector)
    old = card(store)
    assert store.record_event(conversation_id="master", scope=memory.owner_scope, text="旧任务的有来源摘要",
        source_id="fixture-derived", layer="summary", metadata={"source_record_ids": [old.record_id],
        "source_content_hashes": {old.record_id: __import__('hashlib').sha256(old.text.encode()).hexdigest()}})
    task.update(status="partial", completion={"accepted": False, "missing": ["fixture-gap"]})
    task["execution_result"] = export_task_result({**task, "revision": task["revision"] + 1, "updated": tasks.clock()})
    saved = tasks.save(task)
    assert project(projector)["status"] == "stored"
    assert "部分完成" in card(store).text and not card(store).metadata["snapshot"]["verified"]
    assert not projector.validate_record(old)
    assert store.event_by_source_id(source_id="fixture-derived", scope=memory.owner_scope) is None
    assert len(projector.versions.history(task_id="fixture-task", scope=memory.owner_scope)) == 2
    assert events.get(ResultEventStore.key(source), scope=SCOPE)["source_hash"] == source["result_hash"]
    assert saved["execution_attempt"] == source["attempt"]


@pytest.mark.parametrize("change", ["revoke", "expiry", "file", "origin", "resolver"])
def test_source_changes_hide_stale_card_then_retain_explicit_gap(tmp_path, change):
    tasks, events, archive, store, memory, projector, materials, clock = setup(tmp_path, material=True)
    valid = [True]
    attachment = tmp_path / "fixture.txt"
    attachment.write_text("SOURCE_ORIGINAL_NOT_COPIED", encoding="utf-8")
    task = terminal(tasks)
    task.pop("execution_result")
    task["status"] = "running"
    task = tasks.save(task)
    source = MaterialSource("attachment", "attachment", local_path=attachment,
                            revalidate=lambda: valid[0])
    task["material_package"] = materials.package(task, [source], authorization_ref="fixture-grant")
    task.update(status="completed", material_information={"status": "complete", "required_status": "complete"})
    task["execution_result"] = export_task_result({**task, "revision": task["revision"] + 1, "updated": tasks.clock()})
    task = tasks.save(task)
    assert project(projector)["status"] == "stored"
    record = card(store)
    reader = MemorySourceReader(store)
    assert reader.read_source(record_id=record.record_id, scope=memory.owner_scope, now=int(clock[0]))
    if change == "revoke":
        materials.revoke(task["material_package"]["items"][0]["reference"], task=task)
    elif change == "expiry":
        clock[0] += 21
    elif change == "file":
        attachment.write_text("CHANGED_BYTES", encoding="utf-8")
    elif change == "origin":
        valid[0] = False
    else:
        projector.materials = None
    assert reader.read_source(record_id=record.record_id, scope=memory.owner_scope, now=int(clock[0])) is None
    assert project(projector)["status"] == "stored"
    fresh = card(store)
    assert fresh.metadata["snapshot"]["verified"]  # A later invalidation doesn't rewrite past acceptance.
    assert fresh.metadata["snapshot"]["information"]["source_status"] == "unavailable_or_incomplete"
    assert reader.read_source(record_id=fresh.record_id, scope=memory.owner_scope, now=int(clock[0]))
    assert "SOURCE_ORIGINAL_NOT_COPIED" not in json.dumps(fresh.as_dict())
    with tasks._transaction() as db:
        assert db.execute("SELECT count(*) FROM agent_task_material_reads").fetchone()[0] == 0


def test_restart_requires_explicit_source_and_authority_rebind(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    project(projector)
    fresh = MemoryStore(store.path)
    row = card(fresh)
    reader = MemorySourceReader(fresh)
    assert reader.read_source(record_id=row.record_id, scope=memory.owner_scope, now=int(memory.clock())) is None
    service = MemoryService(store=fresh, router=MemoryRouter(), policy=memory.policy, clock=memory.clock)
    restarted = TaskExperienceProjector(service, tasks, expression_archive=archive)
    assert reader.read_source(record_id=row.record_id, scope=memory.owner_scope, now=int(memory.clock())) is None
    assert restarted.bind(task_id="fixture-task", scope=SCOPE, still_authorized=lambda: True)
    assert reader.read_source(record_id=row.record_id, scope=memory.owner_scope, now=int(memory.clock()))
    assert project(restarted)["status"] == "duplicate"


def test_capacity_is_auxiliary_and_does_not_evict_dedupe_or_change_task(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path, max_versions=1)
    tasks.save(terminal(tasks))
    project(projector)
    before = tasks.get("fixture-task", requester=SCOPE.requester, session=SCOPE.session)
    finish_delivery(tasks, archive, "sent")
    assert project(projector)["reason"] == "experience_capacity"
    assert tasks.get("fixture-task", requester=SCOPE.requester, session=SCOPE.session) == before


def test_privacy_no_raw_results_reasoning_credentials_or_full_requests(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    task = terminal(tasks)
    task["requests"][0]["text"] = "排查 API key=TOKEN_PRIVATE_SENTINEL https://user:pass@example.test/?token=SECRET " + "tail" * 500
    task["tool_calls"] = [{"id": "fixture-call", "name": "fixture-tool", "status": "ok",
        "data": {"stdout": "RAW_OUTPUT_SENTINEL", "reasoning": "INTERNAL_SENTINEL"}}]
    task["execution_result"] = export_task_result({**task, "revision": 1, "updated": tasks.clock()})
    tasks.save(task)
    assert project(projector)["status"] == "stored"
    raw = json.dumps(card(store).as_dict())
    for secret in ("TOKEN_PRIVATE_SENTINEL", "user:pass", "token=SECRET", "RAW_OUTPUT_SENTINEL", "INTERNAL_SENTINEL"):
        assert secret not in raw
    assert card(store).metadata["snapshot"]["goal_excerpt_omitted_chars"] > 0


def test_real_subprocess_exit_rolls_back_card_and_dedupe_together(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    snapshot, text = projector._load("fixture-task", SCOPE, lambda: True, None)
    payload = tmp_path / "snapshot.json"
    payload.write_text(json.dumps([snapshot, text]), encoding="utf-8")
    code = """
import json, os, sys
from shinku.memory import MemoryStore
from shinku.memory.experience_store import ExperienceStore
store = MemoryStore(sys.argv[1]); versions = ExperienceStore(store)
snapshot, text = json.load(open(sys.argv[2], encoding='utf-8'))
checks = [0]
def guard():
    checks[0] += 1
    if checks[0] == 2: os._exit(81)
    return True
versions.commit(snapshot, text, scope='private:master', now=2000000000, check_current=guard)
"""
    child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(store.path), str(payload)], timeout=15)
    assert child.returncode == 81
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory_task_experience_versions").fetchone()[0] == 0
    assert project(projector)["status"] == "stored"
    assert project(projector)["status"] == "duplicate"


def test_reset_erases_archived_card_bodies_and_retains_revocation(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    project(projector)
    store.reset_all()
    assert project(projector)["reason"] == "experience_revoked"
    with store._connect() as db:
        assert db.execute("SELECT payload FROM memory_task_experience_versions").fetchone()[0] == "{}"


def test_projection_is_not_enrolled_reinforced_or_promoted_to_current_fact(tmp_path):
    from shinku.memory.lifecycle_policy import LifecyclePolicy
    from shinku.memory.current_understanding import claims
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    store.lifecycle.policy = LifecyclePolicy(mode="active")
    memory.policy = replace(memory.policy, lifecycle=store.lifecycle.policy)
    tasks.save(terminal(tasks))
    project(projector)
    record = card(store)
    assert claims(record) == []
    assert not store.lifecycle.enroll(record_id=record.record_id, now=int(memory.clock()))
    assert not store.lifecycle.reinforce_used(record=record, quote=record.text, final_text=record.text,
        actor_id=SCOPE.requester, evidence_id="fixture-internal-search", now=int(memory.clock()))["accepted"]
    for _ in range(3):
        memory.inspect(conversation_id="fixture", conversation_type="private", requester_id="fixture",
            mode="retrieve", query="生成核验隔离报告", limit=5)
    store.lifecycle.scan(now=int(memory.clock()), scope=memory.owner_scope)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM memory_lifecycle WHERE record_id=?",
            (record.record_id.removeprefix("event:"),)).fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory_lifecycle_evidence").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory_facts").fetchone()[0] == 0


@pytest.mark.parametrize("involved", [False, True])
def test_owner_group_scope_requires_explicit_involvement(tmp_path, involved):
    from shinku.agent.task_state import new_task
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    scope = AgentScope("fixture", "group:fixture-group")
    task = new_task(task_id="fixture-group-task", requester=scope.requester, session=scope.session,
        request="群内任务", message_id="fixture-message", contract=None)
    task.update(status="failed", execution_attempt=1, completion_attempt=1, completion={"accepted": False})
    task["execution_result"] = export_task_result({**task, "revision": 1, "updated": tasks.clock()})
    tasks.save(task)
    result = projector.project(task_id=task["task_id"], scope=scope, still_authorized=lambda: True,
        metadata={"shinku_involved": involved})
    assert result["status"] == ("stored" if involved else "rejected")
    if involved:
        record = card(store, task["task_id"])
        assert record.metadata["owner_source_scope"] == scope.session
        assert not memory.inspect_agent_task(requester_id="spectator", task_id=task["task_id"])["found"]


def test_missing_required_source_and_summary_only_remain_explicit(tmp_path):
    tasks, events, archive, store, memory, projector, materials, *_ = setup(tmp_path, material=True)
    task = terminal(tasks, status="partial")
    task["material_information"] = {"status": "incomplete", "required_status": "missing",
        "gaps": [{"reason": "required_source_unavailable"}], "upstream_completeness": "unknown"}
    task["material_package"] = {"schema_version": 1, "items": [], "gaps": [{"reason": "source_summary_only"}]}
    task["execution_result"] = export_task_result({**task, "revision": 1, "updated": tasks.clock()})
    tasks.save(task)
    project(projector)
    snapshot = card(store).metadata["snapshot"]
    assert snapshot["information"]["required_status"] == "missing"
    assert snapshot["information"]["upstream"] == "unknown"
    assert snapshot["information"]["source_status"] == "unavailable_or_incomplete"
    assert snapshot["information"]["input_gaps"] == 1
    assert "incomplete" in card(store).text and "missing" in card(store).text


def test_card_tamper_and_permission_changes_are_rejected_by_exact_and_retrieved_reads(tmp_path):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    valid = [True]
    projector.project(task_id="fixture-task", scope=SCOPE, still_authorized=lambda: valid[0])
    row = card(store)
    valid[0] = False
    assert not memory.inspect_agent_task(requester_id="fixture", task_id="fixture-task")["found"]
    assert MemorySourceReader(store).read_source(record_id=row.record_id, scope=memory.owner_scope,
        now=int(memory.clock())) is None
    valid[0] = True
    with store._connect() as db:
        db.execute("UPDATE memory_events SET text=? WHERE source_id=?", ("伪造已上线", row.source_id))
    assert not memory.inspect_agent_task(requester_id="fixture", task_id="fixture-task")["found"]
    assert project(projector)["reason"] == "experience_card_integrity_failed"


def test_source_changed_during_memory_transaction_rolls_back(tmp_path, monkeypatch):
    tasks, events, archive, store, memory, projector, *_ = setup(tmp_path)
    tasks.save(terminal(tasks))
    snapshot, text = projector._load("fixture-task", SCOPE, lambda: True, None)
    checks = [0]
    def guard():
        checks[0] += 1
        return checks[0] == 1
    with pytest.raises(ValueError, match="experience_source_changed_during_commit"):
        projector.versions.commit(snapshot, text, scope=memory.owner_scope, now=int(memory.clock()), check_current=guard)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory_task_experience_versions").fetchone()[0] == 0
