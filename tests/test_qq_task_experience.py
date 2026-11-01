"""Actual QQ paths, file executor, verifier, archived send and memory consumer."""
import copy
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from shinku.agent_runtime import AgentScope
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.task_experience import TaskExperienceProjector
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.result_expression import replay_task_expression
from test_qq_independent_agent import turn
import test_qq_result_expression as fixture
from test_qq_result_events import Crash


SCOPE = AgentScope("fixture", "private:fixture")


def host(tmp_path, monkeypatch, **kw):
    store = MemoryStore(tmp_path / "memory.db")
    memory = MemoryService(store=store, router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("fixture",)))
    def build(**options):
        return QQAgentBridge(**options, memory_service=memory, experience_enabled=True)
    monkeypatch.setattr(fixture, "QQAgentBridge", build)
    result = fixture.make_host(tmp_path, **kw)
    return (*result, store, memory)


def card(store, task_id):
    return store.event_by_source_id(source_id="owner:agent-task:" + task_id, scope=MemoryService.OWNER_SCOPE)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("delivery", ["sent", "failed", "uncertain", "dry_run"])
def test_existing_qq_paths_project_execution_and_actual_delivery(tmp_path, monkeypatch, asynchronous, independent, delivery):
    service, tool, execution, request, bridge, chat, sent, store, memory = host(tmp_path, monkeypatch,
        asynchronous=asynchronous, independent=independent, delivery=delivery)
    try:
        reply = bridge(turn(request.original_request))
        if asynchronous:
            assert reply.agent_status == "waiting_background"
            work = fixture.wait_work(service, reply.background_task_id)
            assert work["status"] == "completed"
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        record = card(store, task["task_id"])
        assert record is not None
        snapshot = record.metadata["snapshot"]
        assert snapshot["verified"] and snapshot["execution_status"] == "completed"
        assert snapshot["delivery"]["status"] == delivery
        assert snapshot["delivery"]["sent"] is (delivery == "sent")
        assert snapshot["result_hash"] == task["execution_result"]["result_hash"]
        assert snapshot["deployment"]["status"] == "unknown"
        assert "Execution model report" not in record.text and "……结果整理" not in record.text
        assert tool.calls == 1 and len(bridge.result_expression.profile.runtime.calls) == 1
        assert bridge.task_experience.validate_record(record)
        assert len(bridge.task_experience.versions.history(task_id=task["task_id"], scope=memory.owner_scope)) == 2
        before = (tool.calls, tool.path.read_bytes(), len(execution.calls), len(chat.calls), len(sent), task["revision"])
        replay_task_expression(bridge, turn(request.original_request), task_id=task["task_id"], context={})
        after = service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        assert before == (tool.calls, tool.path.read_bytes(), len(execution.calls), len(chat.calls), len(sent), after["revision"])
        assert len(bridge.task_experience.versions.history(task_id=task["task_id"], scope=memory.owner_scope)) == 2
        with store._connect() as db:
            assert db.execute("SELECT count(*) FROM memory_events WHERE source_id LIKE 'owner:agent-task:%'").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM memory_events WHERE source_id LIKE 'owner:agent-task-detail:%'").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM memory_facts").fetchone()[0] == 0
    finally:
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize("crash", ["before_memory", "after_memory", "after_delivery", "during_send"])
def test_memory_crashes_replay_only_auxiliary_projection(tmp_path, monkeypatch, crash):
    service, tool, execution, request, bridge, chat, sent, store, memory = host(tmp_path, monkeypatch)
    original_commit = bridge.task_experience.versions.commit
    original_sender = bridge.sender
    original_project = bridge.task_experience.project
    if crash in {"before_memory", "after_memory"}:
        def fail(*args, **kwargs):
            if crash == "after_memory":
                original_commit(*args, **kwargs)
            raise Crash()
        monkeypatch.setattr(bridge.task_experience.versions, "commit", fail)
    elif crash == "after_delivery":
        calls = [0]
        def fail(**kw):
            calls[0] += 1
            if calls[0] == 2:
                raise Crash()
            return original_project(**kw)
        monkeypatch.setattr(bridge.task_experience, "project", fail)
    else:
        def fail(message):
            original_sender(message)
            raise Crash()
        bridge.sender = fail
    try:
        with pytest.raises(Crash):
            bridge(turn(request.original_request))
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["execution_result"]["verification"]["accepted"]
        before = (tool.calls, len(execution.calls), len(chat.calls), task["revision"], tool.path.read_bytes())
        monkeypatch.setattr(bridge.task_experience.versions, "commit", original_commit)
        monkeypatch.setattr(bridge.task_experience, "project", original_project)
        bridge.sender = original_sender
        # Expire a claimed expression lease as in P4a's crash recovery.
        clock = bridge.task_store.clock
        bridge.task_store.clock = lambda: clock() + 301
        sent_before = len(sent)
        replay_task_expression(bridge, turn(request.original_request), task_id=task["task_id"], context={})
        fresh = service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        assert before == (tool.calls, len(execution.calls), len(chat.calls), fresh["revision"], tool.path.read_bytes())
        snapshot = card(store, task["task_id"]).metadata["snapshot"]
        assert snapshot["verified"]
        assert snapshot["delivery"]["status"] == ("uncertain" if crash == "during_send" else "sent")
        if crash in {"during_send", "after_delivery"}:
            assert len(sent) == sent_before
        assert len(bridge.result_expression.profile.runtime.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)


def test_memory_operational_error_does_not_turn_verified_delivered_result_into_failure(tmp_path, monkeypatch):
    service, tool, execution, request, bridge, chat, sent, store, memory = host(tmp_path, monkeypatch)
    monkeypatch.setattr(bridge.task_experience.versions, "commit", lambda *a, **k: (_ for _ in ()).throw(OSError("sensitive fixture text")))
    try:
        reply = bridge(turn(request.original_request))
        assert reply.sent and reply.agent_status == "completed"
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["execution_result"]["verification"]["accepted"]
        assert card(store, task["task_id"]) is None
        event = bridge.task_store.result_events.get(bridge.task_store.result_events.key(task["execution_result"]), scope=SCOPE)
        assert event["status"] == "done" and event["delivery"]["sent"]
    finally:
        service.coordinator.shutdown(wait=True)


def test_no_migration_or_new_memory_calls_without_explicit_option(tmp_path):
    service, tool, execution, request, bridge, chat, sent = fixture.make_host(tmp_path)
    try:
        assert bridge.task_experience is None
        reply = bridge(turn(request.original_request))
        assert reply.sent
        assert not (tmp_path / "memory.db").exists()
    finally:
        service.coordinator.shutdown(wait=True)


def test_foreign_owner_does_not_enter_experiences(tmp_path, monkeypatch):
    service, tool, execution, request, bridge, chat, sent, store, memory = host(tmp_path, monkeypatch)
    memory.policy = replace(memory.policy, master_user_ids=("different-owner",))
    try:
        reply = bridge(turn(request.original_request))
        assert reply.sent
        with store._connect() as db:
            assert db.execute("SELECT count(*) FROM memory_task_experience_versions").fetchone()[0] == 0
        assert tool.calls == 1
    finally:
        service.coordinator.shutdown(wait=True)


def test_status_inquiry_does_not_reproject_run_or_reinforce_experience(tmp_path, monkeypatch):
    service, tool, execution, request, bridge, chat, sent, store, memory = host(tmp_path, monkeypatch)
    try:
        bridge(turn(request.original_request))
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        count = len(bridge.task_experience.versions.history(task_id=task["task_id"], scope=memory.owner_scope))
        before = (tool.calls, len(execution.calls), len(chat.calls), len(bridge.result_expression.profile.runtime.calls),
                  task["revision"], tool.path.read_bytes())
        bridge.sender = None
        reply = bridge(turn("任务进度怎么样？", message_id="fixture-status"))
        assert reply.reason == "task_status_query"
        fresh = service.task_store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        assert before == (tool.calls, len(execution.calls), len(chat.calls), len(bridge.result_expression.profile.runtime.calls),
                          fresh["revision"], tool.path.read_bytes())
        assert len(bridge.task_experience.versions.history(task_id=task["task_id"], scope=memory.owner_scope)) == count
        with store._connect() as db:
            assert db.execute("SELECT count(*) FROM memory_lifecycle_evidence").fetchone()[0] == 0
    finally:
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_assembly_experience_option_uses_existing_memory_and_task_stores(tmp_path, monkeypatch, enabled):
    from types import SimpleNamespace
    import shinku.qq.assembly as assembly_module
    from shinku.result_expression import ExpressionProfile
    from test_result_expression import Model
    for key, value in {"SHINKU_ENV_FILE": "missing.env", "SHINKU_DATA_ROOT": "data",
                       "SHINKU_CONFIG_ROOT": "config", "SHINKU_LOG_ROOT": "logs"}.items():
        monkeypatch.setenv(key, str(tmp_path / value))
    monkeypatch.setattr(assembly_module, "build_embedding_provider",
        lambda *a, **k: SimpleNamespace(provider=None, status="fixture-offline", error=""))
    persona = tmp_path / "persona.md"
    persona.write_text("Fixture public persona.", encoding="utf-8")
    config = assembly_module.QQAgentConfig(enabled=True, send_enabled=False, memory_enabled=True,
        memory_db_path=str(tmp_path / "memory.db"), memory_judge_enabled=False,
        persona_file=str(persona), chat_api_key="fixture-key", chat_base_url="https://example.invalid/v1",
        chat_model_name="fixture", agent_state_store_path=str(tmp_path / "tasks.db"),
        trace_store_path=str(tmp_path / "trace.json"), conversation_state_store_path=str(tmp_path / "conversation.db"))
    assembly = assembly_module.assemble_qq_agent(config,
        runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "fixture"}),
        expression_profile=ExpressionProfile(Model(), configuration_id="fixture"), experience_enabled=enabled)
    try:
        bridge = assembly.bridge
        assert (bridge.task_experience is not None) == enabled
        if enabled:
            assert bridge.task_experience.tasks is bridge.task_store
            assert bridge.task_experience.memory is bridge.memory_service
            assert bridge.task_experience.archive is bridge.result_expression.archive
        with bridge.memory_service.store._connect() as db:
            assert bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_task_experience_versions'").fetchone()) == enabled
        assert bridge.sender is None
    finally:
        assembly.task_coordinator.shutdown(wait=True)
        assembly.bridge.task_store.close()
        assembly.bridge.conversation_runtime.close()
