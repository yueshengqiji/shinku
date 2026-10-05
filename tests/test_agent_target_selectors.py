"""Exact unresolved file selectors retain goals; only executor receipts bind them."""
import copy

import pytest

from shinku.agent.loop import AgentLoop, AgentState
from shinku.agent.planner import LLMPlanner
from shinku.agent.task_state import TaskContractError, extend_contract, validate_contract
from shinku.agent.verification import SmartCompletionVerifier
from shinku.llm.runtime_core import RuntimeCore


def contract(**selector):
    return {"goal": "查找并删除名称指定的压缩包", "deliverables": ["删除核验结果"], "scope": ["只处理指定文件"],
        "requirements": [{"id": "r1", "description": "指定压缩包实际删除", "effect": "delete", "targets": [],
            "target_selector": {"kind": "file_stem", "value": "示例包", **selector}}]}


def receipt(path="C:/fixture/示例包.zip", **updates):
    return {"name": "private_file_delete", "status": "ok", "arguments": {"path": path},
        "data": {"state_updates": {"path": path, "verified_effects": ["delete"], "completion_targets": [path], **updates}}}


def state(*calls, value=None):
    return AgentState(context={"agent_completion_contract": value or contract()}, tool_calls=list(calls),
                      transcript=[{"role": "user", "content": "查找示例包的压缩包并帮我删掉"}])


def test_selector_is_kept_and_resolves_only_from_real_effect_target():
    value = validate_contract(contract(suffixes=[".zip", ".7z"]))
    original = copy.deepcopy(value)
    verdict = SmartCompletionVerifier().evaluate(state(receipt(), value=value), "已删掉指定压缩包，另一个文件没动。")
    assert verdict.accepted and verdict.status == "completed" and verdict.satisfied == ("r1",)
    assert value == original


@pytest.mark.parametrize("call", [None, receipt("C:/fixture/别的包.zip"), receipt("C:/fixture/示例包.zip.bak"),
    receipt("示例包.zip"), receipt(verified_effects=[]), receipt(completion_targets=[]),
    {"name": "private_file_delete", "status": "ok", "arguments": {"path": "C:/fixture/示例包.zip"}},
    {**receipt(), "status": "error"}, {**receipt(), "name": "unregistered_delete"},
    receipt("https://example.test/示例包.zip")])
def test_unresolved_or_wrong_or_unverified_target_cannot_complete(call):
    verdict = SmartCompletionVerifier().evaluate(state(*([call] if call else [])), "已经删掉了。")
    assert not verdict.accepted and "r1" in verdict.missing


def test_ambiguous_two_full_paths_are_not_one_resolved_target():
    verdict = SmartCompletionVerifier().evaluate(state(receipt(), receipt("D:/other/示例包.zip")), "已经删掉了。")
    assert not verdict.accepted


def test_exact_filename_and_suffix_constraints_are_not_fuzzy():
    verifier = SmartCompletionVerifier()
    assert verifier.evaluate(state(receipt(), value=contract(kind="file_name", value="示例包.zip")), "已删掉。").accepted
    verdict = verifier.evaluate(state(receipt(), value=contract(suffixes=[".7z"])), "已删掉。")
    assert not verdict.accepted


def test_later_failed_delete_or_recreation_supersedes_old_receipt():
    for later in [{**receipt(), "status": "error"}, {**receipt(), "name": "private_file_write",
                    "data": {"state_updates": {"verified_effects": ["write"], "completion_targets": ["C:/fixture/示例包.zip"]}}}]:
        assert not SmartCompletionVerifier().evaluate(state(receipt(), later), "已删掉。" ).accepted


@pytest.mark.parametrize("selector", [{"kind": "contains"}, {"value": "../示例包"}, {"value": "示例*"},
    {"value": "C:/x.zip"}, {"value": ".."}, {"suffixes": []}, {"suffixes": ["*.zip"]},
    {"extra": "model binding"}])
def test_invalid_selector_shape_is_rejected(selector):
    with pytest.raises(TaskContractError):
        validate_contract(contract(**selector))


def test_selector_cannot_replace_existing_target_or_change_on_followup():
    old = contract()
    changed = contract(value="更简单的目标")
    with pytest.raises(TaskContractError, match="removed_or_changed"):
        extend_contract(old, changed)
    added = copy.deepcopy(old)
    added["requirements"].append({"id": "r2", "description": "实际完整路径也必须删除", "effect": "delete",
                                  "targets": ["C:/fixture/示例包.zip"]})
    assert extend_contract(old, added)["requirements"][0] == old["requirements"][0]


def test_compound_suffix_is_valid_and_exact_stem_strips_only_declared_suffix():
    value = validate_contract(contract(suffixes=[".tar.gz"]))
    assert SmartCompletionVerifier().evaluate(state(receipt("C:/fixture/示例包.tar.gz"), value=value), "已删掉。").accepted
    value = contract(kind="file_name", value="示例包.tar.gz", suffixes=[".tar.gz"])
    assert SmartCompletionVerifier().evaluate(state(receipt("C:/fixture/示例包.tar.gz"), value=value), "已删掉。").accepted


def test_selector_cannot_cover_or_weaken_user_explicit_absolute_path():
    run = state(receipt("D:/other/示例包.zip"))
    run.transcript = [{"role": "user", "content": '请删除 "C:/required/示例包.zip"'}]
    verdict = SmartCompletionVerifier().evaluate(run, "已经删掉了。")
    assert not verdict.accepted and any("uncovered_target" in key for key in verdict.missing)
    run.transcript = [{"role": "user", "content": '请查找并删除 "示例包.zip"'}]
    assert SmartCompletionVerifier().evaluate(run, "已经删掉了。").accepted


def test_external_delivery_receipt_check_does_not_apply_to_delete():
    value = contract()
    value["requirements"][0]["checks"] = {"receipt_required": True}
    with pytest.raises(TaskContractError, match="receipt_check_requires_send_effect"):
        validate_contract(value)


class PlainTextRuntime:
    def __init__(self, text, *, explicit_failure=False):
        self.text, self.explicit_failure, self.calls = text, explicit_failure, []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.explicit_failure:
            return {"kind": "final", "text": self.text, "task_outcome": "failed"}
        return RuntimeCore._recover_agent_plain_text(self, self.text, fallback=kwargs["fallback"])


@pytest.mark.parametrize("actual", ["C:/fixture/示例包.zip", "C:/fixture/别的包.zip", None])
def test_action_plain_text_recovery_is_gated_not_auto_completed(actual):
    runtime = PlainTextRuntime("已经删掉指定压缩包，其他文件没有修改。")
    planner = LLMPlanner(runtime=runtime, system_prompt="isolated", tool_specs=[])
    context = {"agent_completion_contract": contract(), "agent_restored_tool_calls": [receipt(actual)] if actual else []}
    result = AgentLoop(planner=planner, handlers={}, max_rounds=1,
                       completion_gate=SmartCompletionVerifier()).run(context=context)
    assert runtime.calls[0]["fallback"]["_agent_plain_text_verify"] is True
    assert (result.status == "completed") == (actual == "C:/fixture/示例包.zip")


def test_plain_text_opt_in_needs_enabled_evidence_gate_and_active_contract():
    for gate, context in [(None, {"agent_completion_contract": contract()}),
                          (SmartCompletionVerifier(enabled=False), {"agent_completion_contract": contract()}),
                          (SmartCompletionVerifier(), {}), (SmartCompletionVerifier(), {"agent_completion_contract": {}})]:
        runtime = PlainTextRuntime("普通聊天")
        AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt="isolated", tool_specs=[]), handlers={},
                  max_rounds=1, completion_gate=gate).run(context=context)
        assert runtime.calls[0]["fallback"]["_agent_plain_text_verify"] is False


def test_explicit_model_failure_is_preserved_even_with_satisfied_action():
    runtime = PlainTextRuntime("删除已核验，但任务整体失败。", explicit_failure=True)
    result = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt="isolated", tool_specs=[]), handlers={},
        completion_gate=SmartCompletionVerifier()).run(context={"agent_completion_contract": contract(),
            "agent_restored_tool_calls": [receipt()]})
    assert result.status == "failed" and result.reason == "reported_failure"


def test_colloquial_delete_claim_with_failure_cannot_hide_missing_effect():
    verdict = SmartCompletionVerifier().evaluate(state(), "没找到其它文件，但目标已经删掉了。", outcome="failed")
    assert not verdict.accepted
