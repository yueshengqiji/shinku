"""Pre-execution metadata recovery is not permission or executor evidence."""
import copy
import json
from types import SimpleNamespace

import pytest

from shinku.agent.task_state import AgentTaskStore, TaskContractError, new_task
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


class Runtime:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        assert self.responses, "unexpected extra model call"
        value = self.responses.pop(0)
        return value(kwargs) if callable(value) else copy.deepcopy(value)


class DeleteRecorder:
    tool_type = "delete_fixture"
    agent_permission_class = "change"
    agent_progress_class = "change"
    agent_completion_effects = ("delete",)
    agent_completion_target_arguments = ("path",)
    agent_auto_confirm_in_full_access = True

    def __init__(self):
        self.calls = []

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {
            "path": {"type": "string"}, "confirm": {"type": "boolean"}}, "required": ["path", "confirm"]})

    def build_prompt_instruction(self):
        return "隔离记录删除效果，不操作真实文件。"

    def normalize_call(self, call):
        return dict(call) if call.get("path") and call.get("confirm") is True else None

    def execute(self, *, call, context):
        self.calls.append(dict(call))
        return {"followup_context": "隔离删除回执", "state_updates": {
            "verified_effects": ["delete"], "completion_targets": [call["path"]], "path": call["path"]}}


def contract(effect="delete", path="C:/fixture/target.zip"):
    return {"goal": "查找并删除指定压缩包", "deliverables": ["指定压缩包已删除的核验结果"], "scope": ["只处理指定文件"],
            "requirements": [{"id": "r1", "description": "指定压缩包必须实际删除", "effect": effect,
                              "targets": [path], "checks": {}}]}


def tool_decision(*, bad=True, metadata=True, link=None):
    value = {"kind": "tool", "tool_calls": [{"id": "locked-call", "name": "delete_fixture",
              "arguments": {"path": "C:/fixture/target.zip"}}]}
    if metadata:
        value.update(task_link=link or {"mode": "new"},
                     task_contract=contract("tool:delete_fixture" if bad else "delete"))
    return value


def turn(text="找找指定压缩包然后帮我删掉", *, user="owner", message="m1"):
    item = IncomingMessage.from_event({"post_type": "message", "message_type": "private", "user_id": user,
        "message_id": message, "message": [{"type": "text", "data": {"text": text}}]})
    return QQTurn(conversation_id="s1", conversation_type="private", primary_message_id=message,
                  message_ids=(message,), combined_text=text, messages=(item,))


def bot(runtime, handler, *, mode="ultra", store=None):
    return QQAgentBridge(runtime=runtime, system_prompt="测试身份", tool_host=ToolHost(ToolRegistry({handler.tool_type: handler})),
        master_user_ids=("owner",), owner_gate_required=True, task_contract_enabled=True, task_store=store,
        command_state=QQCommandState(permission_mode=mode),
        policy=ExecutionPolicy(risk_by_tool={handler.tool_type: "high"}))


FINAL = {"kind": "final", "text": "指定压缩包已删除。", "task_outcome": "completed"}


@pytest.mark.parametrize("mode", ["full", "ultra"])
def test_bad_side_effect_condition_is_repaired_before_exact_selected_call(mode):
    handler = DeleteRecorder()

    def repair(kwargs):
        assert handler.calls == []
        assert not kwargs.get("native_tools") and not kwargs.get("history_turns")
        data = json.loads(kwargs["user_prompt"])
        assert data["locked_task_link"] == {"mode": "new"}
        assert data["effect_catalog"] == {"delete_fixture": ["delete"]}
        assert data["selected_tools"] == ["delete_fixture"]
        assert data["locked_repair_contract"] == contract()
        return {"task_contract": contract()}

    runtime = Runtime(tool_decision(), repair, FINAL)
    bridge = bot(runtime, handler, mode=mode)
    reply = bridge.handle_turn(turn())
    assert reply.agent_status == "completed", reply.reason
    assert handler.calls == [{"path": "C:/fixture/target.zip", "type": "delete_fixture", "confirm": True}]
    stored = bridge.task_store.candidates(requester="owner", session="private:s1")[0]
    assert stored["requests"][0]["text"] == turn().combined_text
    assert stored["contract"] == contract()
    assert stored["tool_calls"][0]["id"] == "locked-call"
    stages = bridge.trace_store.snapshot()[0]["stages"]
    assert [row["status"] for row in stages if row["name"] == "task_contract_repair"] == ["started", "accepted"]
    assert len(runtime.calls) == 3


def test_missing_native_metadata_preparation_then_repair_is_bounded():
    handler = DeleteRecorder()
    runtime = Runtime(tool_decision(metadata=False), {"task_link": {"mode": "new"},
        "task_contract": contract("tool:delete_fixture")}, {"task_contract": contract()}, FINAL)
    assert bot(runtime, handler).handle_turn(turn()).agent_status == "completed"
    assert len(runtime.calls) == 4 and len(handler.calls) == 1


def test_valid_contract_has_no_extra_repair_call():
    handler = DeleteRecorder()
    runtime = Runtime(tool_decision(bad=False), FINAL)
    assert bot(runtime, handler).handle_turn(turn()).agent_status == "completed"
    assert len(runtime.calls) == 2 and len(handler.calls) == 1


def test_invalid_tool_alias_with_unresolved_selector_retains_exact_predicate():
    handler = DeleteRecorder()
    initial = tool_decision()
    initial["task_contract"]["requirements"][0].update(targets=[],
        target_selector={"kind": "file_stem", "value": "target", "suffixes": [".zip"]})
    corrected = copy.deepcopy(initial["task_contract"])
    corrected["requirements"][0]["effect"] = "delete"
    runtime = Runtime(initial, {"task_contract": corrected}, FINAL)
    bridge = bot(runtime, handler)
    assert bridge.handle_turn(turn()).agent_status == "completed"
    stored = bridge.task_store.candidates(requester="owner", session="private:s1")[0]
    assert stored["contract"] == corrected and len(handler.calls) == 1


def test_misplaced_delivery_check_is_repaired_before_any_selected_execution():
    handler = DeleteRecorder()
    initial = tool_decision(bad=False)
    initial["task_contract"]["requirements"][0]["checks"] = {"receipt_required": True}
    runtime = Runtime(initial, {"task_contract": contract()}, FINAL)
    bridge = bot(runtime, handler)
    assert bridge.handle_turn(turn()).agent_status == "completed"
    assert len(handler.calls) == 1 and len(runtime.calls) == 3


@pytest.mark.parametrize("change", ["effect", "targets", "goal", "checks"])
def test_misplaced_receipt_repair_cannot_change_any_other_condition(change):
    handler = DeleteRecorder()
    initial = tool_decision(bad=False)
    initial["task_contract"]["requirements"][0]["checks"] = {"receipt_required": True}
    corrected = contract()
    if change == "goal":
        corrected["goal"] = "新目标"
    elif change == "effect":
        corrected["requirements"][0]["effect"] = "write"
    elif change == "targets":
        corrected["requirements"][0]["targets"] = []
    else:
        corrected["requirements"][0]["checks"] = {"min_bytes": 10}
    reply = bot(Runtime(initial, {"task_contract": corrected}), handler).handle_turn(turn())
    assert reply.agent_status == "failed" and handler.calls == []


@pytest.mark.parametrize("extra", [{"tool_calls": [{"name": "delete_fixture", "arguments": {"path": "other.zip"}}]},
    {"kind": "tool"}, {"speech": "已经删掉"}, {"approved": True}, {"task_outcome": "completed"}])
def test_repair_cannot_supply_actions_authorization_or_final(extra):
    handler = DeleteRecorder()
    runtime = Runtime(tool_decision(), {"task_contract": contract(), **extra})
    bridge = bot(runtime, handler)
    reply = bridge.handle_turn(turn())
    assert reply.reason == "task_contract_error:contract_repair_failed" and reply.agent_status == "failed"
    assert handler.calls == [] and len(runtime.calls) == 2
    assert bridge.task_store.candidates(requester="owner", session="private:s1") == []


def test_repair_does_not_repeat_when_corrected_contract_still_invalid():
    handler = DeleteRecorder()
    runtime = Runtime(tool_decision(), {"task_contract": contract("tool:delete_fixture")})
    reply = bot(runtime, handler).handle_turn(turn())
    assert reply.agent_status == "failed" and "权限不足" in reply.final_text
    assert handler.calls == [] and len(runtime.calls) == 2


def test_repair_exception_is_honest_failure_without_execution():
    handler = DeleteRecorder()

    def unavailable(_):
        raise RuntimeError("model down")

    runtime = Runtime(tool_decision(), unavailable)
    assert bot(runtime, handler).handle_turn(turn()).agent_status == "failed"
    assert handler.calls == [] and len(runtime.calls) == 2


def test_invalid_proposal_and_user_text_stay_in_data_not_system_instructions():
    marker = "UNTRUSTED_CONTRACT_OVERRIDE_X7"
    handler = DeleteRecorder()
    initial = tool_decision()
    initial["task_notes"] = {"hypotheses": [marker]}
    runtime = Runtime(initial, {"task_contract": contract()}, FINAL)
    assert bot(runtime, handler).handle_turn(turn(marker)).agent_status == "completed"
    repair = runtime.calls[1]
    assert marker in repair["user_prompt"]
    assert marker not in repair["system_prompt"] + "".join(repair["system_extra_blocks"])


@pytest.mark.parametrize("change", ["effect", "target", "checks", "goal", "scope", "requirements"])
def test_well_formed_wrong_alias_cannot_be_repaired_by_weakening_or_expanding(change):
    handler = DeleteRecorder()
    corrected = contract()
    if change == "effect":
        corrected["requirements"][0]["effect"] = "tool:some_read"
    elif change == "target":
        corrected["requirements"][0]["targets"] = []
    elif change == "checks":
        corrected["requirements"][0]["checks"] = {"receipt_required": True}
    elif change == "goal":
        corrected["goal"] = "只查找而不删除"
    elif change == "scope":
        corrected["scope"].append("其他文件也删除")
    else:
        corrected["requirements"].append({"id": "r2", "description": "扩大目标", "effect": "delete", "targets": ["other.zip"]})
    runtime = Runtime(tool_decision(), {"task_contract": corrected})
    reply = bot(runtime, handler).handle_turn(turn())
    assert reply.agent_status == "failed" and not handler.calls


@pytest.mark.parametrize("mode,user", [("readonly", "owner"), ("ultra", "other")])
def test_permission_and_owner_rejections_are_not_repaired(mode, user):
    handler = DeleteRecorder()
    runtime = Runtime(tool_decision())
    reply = bot(runtime, handler, mode=mode).handle_turn(turn(user=user))
    assert reply.agent_status in {"blocked", "rejected"}
    assert handler.calls == [] and len(runtime.calls) <= 1


def test_traditional_mode_still_asks_for_approval_after_repair():
    handler = DeleteRecorder()
    initial = tool_decision()
    initial["tool_calls"][0]["arguments"]["confirm"] = True
    runtime = Runtime(initial, {"task_contract": contract()})
    reply = bot(runtime, handler, mode="traditional").handle_turn(turn())
    assert reply.agent_status == "waiting_user" and reply.reason == "tool_approval_required"
    assert handler.calls == [] and len(runtime.calls) == 2


def seed(store):
    task = new_task(task_id="old", requester="owner", session="private:s1", request="删除指定压缩包",
                    message_id="old", contract=contract())
    task["status"] = "failed"
    return store.save(task)


def test_repair_cannot_relink_or_drop_old_conditions():
    for change in ("link", "target"):
        store = AgentTaskStore()
        old = seed(store)
        handler = DeleteRecorder()
        initial = tool_decision(link={"mode": "continue", "task_id": "old"})
        initial["task_contract"]["scope"] = "malformed"
        corrected = {"task_contract": contract()}
        if change == "link":
            corrected["task_link"] = {"mode": "new"}
        else:
            corrected["task_contract"]["requirements"][0]["targets"] = ["other.zip"]
        runtime = Runtime(initial, corrected)
        reply = bot(runtime, handler, store=store).handle_turn(turn())
        assert reply.agent_status == "failed" and not handler.calls
        assert store.get("old", requester="owner", session="private:s1") == old


def test_old_requirement_change_and_wrong_session_are_not_repairable():
    for change in ("requirement", "session"):
        store = AgentTaskStore()
        old = seed(store)
        handler = DeleteRecorder()
        initial = tool_decision(bad=False, link={"mode": "continue", "task_id": "old"})
        if change == "requirement":
            initial["task_contract"]["requirements"][0]["effect"] = "tool:delete_fixture"
        else:
            initial["task_link"]["task_id"] = "foreign-id"
        runtime = Runtime(initial)
        reply = bot(runtime, handler, store=store).handle_turn(turn())
        assert reply.agent_status == "blocked" and not handler.calls
        assert len(runtime.calls) == 1
        assert store.get("old", requester="owner", session="private:s1") == old


def test_oversized_repair_input_is_rejected_without_extra_model_call():
    handler = DeleteRecorder()
    initial = tool_decision()
    initial["task_contract"]["goal"] = "x" * 40000
    runtime = Runtime(initial)
    assert bot(runtime, handler).handle_turn(turn()).agent_status == "failed"
    assert not handler.calls and len(runtime.calls) == 1


def test_uncertain_state_commit_is_not_repaired_or_repeated():
    class ConflictingStore(AgentTaskStore):
        def save(self, state, **kwargs):
            raise TaskContractError("task_state_conflict")

    handler = DeleteRecorder()
    runtime = Runtime(tool_decision(bad=False))
    reply = bot(runtime, handler, store=ConflictingStore()).handle_turn(turn())
    assert reply.reason == "task_contract_error:task_state_conflict"
    assert not handler.calls and len(runtime.calls) == 1
