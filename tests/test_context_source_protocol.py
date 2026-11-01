"""来源协议、双时点重核与无正文诊断的离线合成验收。"""
from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService
from shinku.memory.context_sources import (
    CONTEXT_DIAGNOSTIC_KEY, CONTEXT_SOURCE_KEY, ContextSourceGroup, PromptSourceDependencies,
)
from shinku.memory.projection.contracts import Deadline
from shinku.memory.projection.sources import ProjectionSources
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.reply_sources import check_context_sources, turn_source_binding
from test_context_projection import NOW, engine, put, req
from test_desktop_qq_continuity import qq_turn, record, setup


def mode_setup(tmp_path, mode, request):
    """task 模式通过正式构造入口启用，保留独立执行模型边界。"""
    rig = setup(tmp_path)
    if mode == "task":
        original = rig.bridge
        coordinator = AgentTaskCoordinator(max_workers=1)
        request.addfinalizer(lambda: coordinator.shutdown(wait=True))
        rig.bridge = QQAgentBridge(runtime=rig.runtime, system_prompt="测试人格",
            sender=original.sender, bot_ids=("bot",), master_user_ids=("owner",),
            memory_service=rig.memory, conversation_runtime=original.conversation_runtime,
            tool_decision_mode=mode, execution_profile=ExecutionProfile(SimpleNamespace()),
            task_contract_enabled=True, task_coordinator=coordinator)
    return rig


def stages(rig):
    return [s for t in rig.bridge.trace_store.snapshot(limit=8) for s in t["stages"]
            if s["name"] == "context_sources"]


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_both_modes_store_one_bound_contract_and_recheck_twice(tmp_path, mode, request):
    rig = mode_setup(tmp_path, mode, request)
    record(rig.memory, "私密标记：只有批准才执行，尚未批准。")
    state = AgentState(context={"qq_turn": qq_turn("接着聊")}, transcript=[])
    parts = rig.bridge._build_prompt(state)
    contract = state.context[CONTEXT_SOURCE_KEY]
    assert isinstance(contract, PromptSourceDependencies)
    assert contract.handles and "私密标记" not in repr(contract)
    assert "私密标记" in parts.user_prompt
    assert "_bot_prompt_source_handles" not in state.context
    assert "_owner_continuity_signature" not in state.context
    deps = rig.bridge._model_call_dependencies()
    reply = deps.call_json(source_context=state.context, trace_id="synthetic",
        user_prompt=parts.user_prompt, fallback={"speech": "fallback"})
    assert reply["speech"] == rig.runtime.text and len(rig.runtime.calls) == 1
    diagnostic = state.context[CONTEXT_DIAGNOSTIC_KEY]
    assert set(diagnostic["checks"]) == {"before_generation", "after_generation"}
    assert all(c["state"] == "valid" for c in diagnostic["checks"].values())
    assert "私密标记" not in json.dumps(diagnostic, ensure_ascii=False)
    assert "event:" not in json.dumps(diagnostic)


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("change,reason", [
    ("expire", "source_expired"), ("forget", "source_revoked"),
    ("edit", "owner_origin_changed"), ("identity", "scope_not_allowed"),
    ("new_omitted", "owner_dependency_changed"),
])
def test_real_bridge_cancels_changed_basis_without_extra_model_calls(tmp_path, mode, change, reason, request):
    rig = mode_setup(tmp_path, mode, request)
    record(rig.memory, "合成约定：只有批准才执行，尚未批准。", conversation="desktop:first", kind="desktop")
    def request(**options):
        rig.runtime.calls.append(options)
        if change == "expire":
            # 真正的记忆TTL仍按实时钟失效；自然跨过选材窗口另有正向回归。
            with rig.memory.store._connect() as conn:
                conn.execute("UPDATE memory_events SET expires_at=?", (NOW + 1,))
            rig.clock[0] += 2
        elif change == "forget":
            rig.memory.store.forget(query="合成约定", scope="private:master", now=NOW)
        elif change == "identity":
            rig.memory.policy = replace(rig.memory.policy, master_user_ids=())
        elif change == "edit":
            with rig.memory.store._connect() as conn:
                conn.execute("UPDATE memory_events SET text=? WHERE scope='private:desktop:first'",
                             ("原件条件已经更正。",))
        else:
            record(rig.memory, "长材料。" * 500 + "但是旧约定取消了。", mid="long",
                   conversation="desktop:first", kind="desktop")
        return {"kind": "final", "text": "继续之前的合成约定。"}
    rig.runtime.call_chat_json = request
    result = rig.bridge.handle_turn(qq_turn("接着聊这件事"))
    assert result.agent_status == "failed" and result.final_text != "继续之前的合成约定。"
    assert len(rig.runtime.calls) == 1
    assert any(s.get("phase") == "after_generation" and s.get("reason") == reason for s in stages(rig))


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_real_ttl_expiry_after_prompt_before_generation_skips_model(tmp_path, mode, request):
    rig = mode_setup(tmp_path, mode, request)
    record(rig.memory, "只供短期承接的安排。")
    state = AgentState(context={"qq_turn": qq_turn("继续")}, transcript=[])
    parts = rig.bridge._build_prompt(state)
    with rig.memory.store._connect() as conn:
        conn.execute("UPDATE memory_events SET expires_at=?", (NOW + 1,))
    rig.clock[0] += 2
    response = rig.bridge._model_call_dependencies().call_json(source_context=state.context,
        user_prompt=parts.user_prompt, fallback={"speech": "fallback"})
    assert response == {"speech": "fallback"} and not rig.runtime.calls
    check = state.context[CONTEXT_DIAGNOSTIC_KEY]["checks"]["before_generation"]
    assert check["reason"] == "source_expired"


@pytest.mark.parametrize("turn", [qq_turn("继续", mid="other"), qq_turn("继续", sender="other"),
                                  qq_turn("继续", kind="group")])
def test_contract_cannot_move_to_another_turn_or_requester(tmp_path, turn, monkeypatch):
    rig = setup(tmp_path)
    record(rig.memory, "私人来源。")
    state = AgentState(context={"qq_turn": qq_turn("继续")}, transcript=[])
    rig.bridge._build_prompt(state)
    state.context["qq_turn"] = turn
    from shinku.qq import reply_sources
    monkeypatch.setattr(reply_sources, "checked_prompt_snapshot", lambda *a: pytest.fail("跨轮拒绝须先于读库"))
    assert rig.bridge._snapshot_prompt_sources(state.context).reason == "source_turn_changed"


def test_omitted_projection_source_is_still_versioned(tmp_path):
    store, projector = engine(tmp_path, max_chars=1200, record_chars=400)
    memory = MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW,
                           policy=MemoryPolicy(projection=projector.policy))
    for i in range(6):
        put(store, str(i), f"来源{i}。" + "讨论细节。" * 100 + "但是未批准执行。")
    runtime = SimpleNamespace(calls=[], chat_supports_native_tools=lambda: False)
    runtime.call_chat_json = lambda **kw: runtime.calls.append(kw) or {"speech": "fixture"}
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", memory_service=memory)
    state = AgentState(context={"qq_turn": qq_turn("继续")}, transcript=[])
    bridge._build_prompt(state)
    group = next(g for g in state.context[CONTEXT_SOURCE_KEY].groups if g.kind == "projection")
    assert group.omitted_sources > 0 and len(group.handles) == 6
    # 无需从提示反推被裁去的正文；协议保留所有已核验来源的固定版本。
    view = memory.project_context(conversation_id="owner", conversation_type="private", user_message="继续",
        message_meta={"sender_id": "owner", "self_id": "bot", "message_id": "q1"})
    visible = {r["record_id"] for r in view["records"] if r["text"]}
    omitted = next(handle for handle in view["source_dependencies"] if handle[0] not in visible)
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text=? WHERE event_id=?",
                     ("来源已有新更正。", omitted[0].removeprefix("event:")))
    result = bridge._model_call_dependencies().call_json(source_context=state.context, fallback={})
    assert result == {} and not runtime.calls
    assert state.context[CONTEXT_DIAGNOSTIC_KEY]["checks"]["before_generation"]["reason"] == "source_version_changed"


@pytest.mark.parametrize("change,reason", [
    ("expire", "source_expired"), ("forget", "source_revoked"),
    ("edit", "source_version_changed"), ("scope", "scope_not_allowed"),
    ("delete", "source_missing"), ("invalid", "invalid_source_handle"),
])
def test_checked_reader_reports_observed_reason_without_writes(tmp_path, change, reason):
    store, _ = engine(tmp_path)
    row = put(store, "old", "原文包含私密条件，尚未批准。")
    reader = ProjectionSources(store)
    handle = (row.record_id, row.scope, "")
    fresh = reader.read((handle,), req(), Deadline(300))[0]
    handle = (row.record_id, row.scope, fresh.metadata["source_version"])
    if change == "forget":
        store.forget(scope=row.scope, record_id=row.record_id, query="", now=NOW)
    elif change == "scope":
        handle = (row.record_id, "private:other", handle[2])
    elif change == "invalid":
        handle = ("bad:id", row.scope, handle[2])
    else:
        with store._connect() as conn:
            if change == "expire":conn.execute("UPDATE memory_events SET expires_at=?", (NOW-1,))
            elif change == "edit":conn.execute("UPDATE memory_events SET text='已经更正'")
            else:conn.execute("DELETE FROM memory_events")
    with store._connect() as conn:
        before = tuple(conn.iterdump())
    checked = reader.read_checked((handle,), req(), Deadline(300))
    assert checked.records == (None,) and checked.reasons == (reason,)
    with store._connect() as conn:
        assert tuple(conn.iterdump()) == before


def test_diagnostic_reader_does_not_add_queries(tmp_path, monkeypatch):
    from shinku.memory.projection import sources
    store, _ = engine(tmp_path)
    row = put(store, "old", "合成来源。")
    sql = []
    original = sources.connection
    @contextmanager
    def traced(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            conn.set_trace_callback(sql.append)
            yield conn
    monkeypatch.setattr(sources, "connection", traced)
    reader, handles = ProjectionSources(store), ((row.record_id, row.scope, ""),)
    old_result = reader.read(handles, req(), Deadline(300))
    old_sql = tuple(sql);sql.clear()
    checked = reader.read_checked(handles, req(), Deadline(300))
    assert checked.records == old_result and checked.reasons == ("",)
    assert tuple(sql) == old_sql


def test_combined_source_budget_is_rejected_without_silent_truncation():
    turn = qq_turn("继续")
    contract = PromptSourceDependencies(turn_source_binding(turn), (
        ContextSourceGroup("projection", tuple((f"event:{i}", "private:owner", "v") for i in range(64))),
        ContextSourceGroup("owner_continuity", (("event:extra", "private:master", "v"),)),
    ))
    assert len(contract.handles) == 65
    checked = check_context_sources(object(), {"qq_turn": turn, CONTEXT_SOURCE_KEY: contract},
                                    owner_reader=lambda context: pytest.fail("预算失败不得扩展读取"))
    assert not checked.valid and checked.reason == "source_budget"


def test_projection_disabled_and_legacy_reference_are_not_reported_as_verified(tmp_path):
    rig = setup(tmp_path)
    state = AgentState(context={"qq_turn": qq_turn("继续"), "memory_context": {
        "rendered": "旧式参考尚未核验。", "decision": {"history_mode": "recent"}}}, transcript=[])
    rig.bridge._build_prompt(state)
    groups = state.context[CONTEXT_DIAGNOSTIC_KEY]["groups"]
    assert any(g["read_path"] == "projection_disabled" and g["state"] == "disabled" for g in groups)
    assert any(g["read_path"] == "legacy_memory_reference" and g["state"] == "untracked" for g in groups)
    assert state.context[CONTEXT_DIAGNOSTIC_KEY]["absence_means"] == "unknown"


def test_diagnostic_protocol_rejects_body_or_path_payloads():
    with pytest.raises(ValueError, match="invalid_context_source_diagnostics"):
        ContextSourceGroup("projection", read_path="/private/path/secret")
    with pytest.raises(ValueError, match="invalid_context_source_diagnostics"):
        ContextSourceGroup("projection", reason_codes=("私密正文",))
