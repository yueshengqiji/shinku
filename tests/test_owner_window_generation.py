"""五分钟仅限制本轮选材；真实桥接和模拟时钟验证生成前后的来源边界。"""
from dataclasses import replace
import json

import pytest

from shinku.agent.loop import AgentState
from shinku.memory.context_sources import CONTEXT_SOURCE_KEY
from test_context_source_protocol import mode_setup
from test_desktop_qq_continuity import NOW, qq_turn, record


MARKER = "合成短期条件：只有确认才执行，目前未确认。"


def near_boundary(rig, *, omitted=False):
    rig.clock[0] = NOW - 299
    record(rig.memory, ("长材料。" * 500 if omitted else "") + MARKER, mid="near-window")
    rig.clock[0] = NOW
    if omitted:
        record(rig.memory, "本轮还未批准。", mid="visible")


def checks(rig):
    return [s for t in rig.bridge.trace_store.snapshot(limit=1) for s in t["stages"]
            if s["name"] == "context_sources"]


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_real_bridge_keeps_one_call_when_source_leaves_recent_window(tmp_path, mode, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig)
    rig.runtime.before_return = lambda: rig.clock.__setitem__(0, NOW + 16)
    result = rig.bridge.handle_turn(qq_turn("接着这个条件说。"))
    assert result.agent_status == "completed" and result.final_text == rig.runtime.text
    assert len(rig.runtime.calls) == len(rig.sends) == 1
    assert MARKER in rig.runtime.calls[0]["user_prompt"]
    assert [s.get("reason") for s in checks(rig)] == ["", ""]


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_omitted_old_source_natural_age_does_not_cancel_candidate(tmp_path, mode, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig, omitted=True)
    rig.runtime.before_return = lambda: rig.clock.__setitem__(0, NOW + 16)
    result = rig.bridge.handle_turn(qq_turn("接着聊。"))
    assert result.agent_status == "completed" and len(rig.runtime.calls) == 1
    assert MARKER not in rig.runtime.calls[0]["user_prompt"]
    assert "omitted_recent_sources" in rig.runtime.calls[0]["user_prompt"]


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_unpersisted_turn_uses_same_selection_time_at_both_checks(tmp_path, mode, phase, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig)
    # 直接构造提示也有明确的本轮选材时间，不依赖当前消息已经写入数据库。
    state = AgentState(context={"qq_turn": qq_turn("继续", mid="not-persisted")}, transcript=[])
    parts = rig.bridge._build_prompt(state)
    if phase == "before":
        rig.clock[0] = NOW + 16
    else:
        rig.runtime.before_return = lambda: rig.clock.__setitem__(0, NOW + 16)
    response = rig.bridge._model_call_dependencies().call_json(source_context=state.context,
        user_prompt=parts.user_prompt, fallback={"speech": "fallback"})
    assert response["speech"] == rig.runtime.text and len(rig.runtime.calls) == 1
    assert not state.context.get("qq_dialogue_sources_changed")


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_next_turn_selects_again_and_does_not_extend_automatic_window(tmp_path, mode, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig)
    rig.runtime.before_return = lambda: rig.clock.__setitem__(0, NOW + 16)
    first = rig.bridge.handle_turn(qq_turn("继续", mid="first"))
    second = rig.bridge.handle_turn(qq_turn("现在继续", mid="second"))
    assert first.agent_status == second.agent_status == "completed"
    assert len(rig.runtime.calls) == 2
    assert MARKER in rig.runtime.calls[0]["user_prompt"]
    assert MARKER not in rig.runtime.calls[1]["user_prompt"]


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("change,reason", [
    ("forget", "source_revoked"), ("edit", "owner_origin_changed"),
    ("ttl", "source_expired"), ("receipt", "owner_dependency_changed"),
    ("identity", "scope_not_allowed"), ("disabled", "owner_sources_unavailable"),
    ("new_omitted", "owner_dependency_changed"),
])
def test_real_invalidations_still_cancel_after_window_crossing(tmp_path, mode, change, reason, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig)
    def change_before_return():
        rig.clock[0] = NOW + 16
        if change == "forget":
            rig.memory.store.forget(query="合成短期条件", scope="private:master", now=rig.clock[0])
        elif change == "identity":
            rig.memory.policy = replace(rig.memory.policy, master_user_ids=())
        elif change == "disabled":
            rig.memory.policy = replace(rig.memory.policy, owner_continuity_seconds=0)
        elif change == "new_omitted":
            record(rig.memory, "新更正。" * 500 + "旧条件已撤销。", mid="new-correction")
        else:
            with rig.memory.store._connect() as conn:
                if change == "edit":
                    conn.execute("UPDATE memory_events SET text=? WHERE scope='private:owner' AND source_id=?",
                                 ("原件已经更正。", "turn:owner:near-window"))
                elif change == "ttl":
                    conn.execute("UPDATE memory_events SET expires_at=? WHERE text=?", (NOW + 10, MARKER))
                else:
                    row = conn.execute("SELECT event_id,metadata_json FROM memory_events WHERE scope='private:owner' AND source_id=?",
                                       ("turn:owner:near-window",)).fetchone()
                    meta = {**json.loads(row["metadata_json"]), "delivery_content_complete": False}
                    conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                                 (json.dumps(meta), row["event_id"]))
    rig.runtime.before_return = change_before_return
    result = rig.bridge.handle_turn(qq_turn("接着这个条件说。"))
    assert result.agent_status == "failed" and result.final_text != rig.runtime.text
    assert len(rig.runtime.calls) == 1
    assert checks(rig)[-1]["reason"] == reason


def test_selection_anchor_is_bound_to_turn_and_checks_remain_read_only(tmp_path, request):
    rig = mode_setup(tmp_path, "legacy", request)
    near_boundary(rig)
    state = AgentState(context={"qq_turn": qq_turn("继续", mid="first")}, transcript=[])
    rig.bridge._build_prompt(state)
    contract = state.context[CONTEXT_SOURCE_KEY]
    rig.clock[0] = NOW + 16
    with rig.memory.store._connect() as conn:
        before = tuple(conn.iterdump())
    assert rig.bridge._snapshot_prompt_sources(state.context).valid
    assert rig.bridge._snapshot_prompt_sources(state.context).valid
    with rig.memory.store._connect() as conn:
        assert tuple(conn.iterdump()) == before
    other = AgentState(context={**state.context, "qq_turn": qq_turn("新一轮", mid="other")}, transcript=[])
    prompt = rig.bridge._build_prompt(other)
    assert MARKER not in prompt.user_prompt
    assert other.context[CONTEXT_SOURCE_KEY].binding != contract.binding


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("change", ["edit", "ttl", "forget"])
def test_omitted_dependency_still_detects_real_invalidation(tmp_path, mode, change, request):
    rig = mode_setup(tmp_path, mode, request)
    near_boundary(rig, omitted=True)
    def invalidate():
        rig.clock[0] = NOW + 16
        if change == "forget":
            rig.memory.store.forget(query="合成短期条件", scope="private:master", now=rig.clock[0])
        else:
            with rig.memory.store._connect() as conn:
                if change == "edit":
                    conn.execute("UPDATE memory_events SET text=? WHERE source_id=? AND scope='private:owner'",
                                 ("省略原件已经更正。", "turn:owner:near-window"))
                else:
                    conn.execute("UPDATE memory_events SET expires_at=? WHERE source_id IN (?,?)",
                                 (NOW + 10, "turn:owner:near-window", "owner:turn:owner:near-window"))
    rig.runtime.before_return = invalidate
    result = rig.bridge.handle_turn(qq_turn("接着聊。"))
    assert result.agent_status == "failed" and len(rig.runtime.calls) == 1
    assert checks(rig)[-1]["reason"] == "owner_dependency_changed"


@pytest.mark.parametrize("selection_at", [-1, True, 1.5, NOW + 1])
def test_public_source_reader_rejects_invalid_or_future_selection_time(tmp_path, request, selection_at):
    rig = mode_setup(tmp_path, "legacy", request)
    with pytest.raises(ValueError, match="invalid_owner_selection_time"):
        rig.memory.read_recent_owner_sources(conversation_id="owner", conversation_type="private",
            requester="owner", self_id="bot", message_id="q1", selection_at=selection_at)
