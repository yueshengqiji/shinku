"""执行上下文总体预算回归：提示投影不能损坏真实来源与读取账本。"""
import copy
import json
from types import SimpleNamespace

import pytest

from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime.context import EXECUTION_CONTEXT_MAX_BYTES, build_execution_prompt
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.information import source_text
from shinku.agent_runtime.materials import MaterialSource, TaskMaterialStore, TaskMaterialSession


def decode_payload(prompt):
    return json.JSONDecoder().raw_decode(prompt.split("\n", 1)[1])[0]


@pytest.fixture
def material_context(tmp_path):
    ledger = AgentTaskStore(tmp_path / "tasks.sqlite", clock=lambda: 1000.0)
    task = ledger.save(new_task(task_id="context-fixture", requester="fixture", session="local:context",
        request="读取项目并提出优化建议，保留原始文件。", message_id="original", contract=None))
    task["execution_attempt"] = 1
    task["status"] = "running"
    task = ledger.save(task)
    store = TaskMaterialStore(ledger)
    try:
        yield ledger, task, store
    finally:
        ledger.close()


def state_for(task, store, sources, *, transcript=()):
    task["material_package"] = store.package(task, sources, authorization_ref="fixture-authority")
    saved = store.store.save(task)
    task.clear()
    task.update(saved)
    session = TaskMaterialSession(store, task, still_authorized=lambda: True)
    context = {"agent_task_state": task, "agent_authority_revision": "unchanged-authority",
               "agent_permission_mode": "read", "agent_material_session": session}
    return SimpleNamespace(context=context, transcript=list(transcript)), session


@pytest.mark.parametrize("unit", ["长中文资料", "资料🙂🧭", "控制\n\t\"\\\x01"])
def test_many_sources_fit_total_utf8_budget_without_changing_sources_or_reads(material_context, unit):
    ledger, task, store = material_context
    originals = [{"feedback": unit * 5000 + "SOURCE_TAIL_" + str(index),
                  "state_updates": {"path": "/synthetic/project/part-" + str(index)}} for index in range(14)]
    required = MaterialSource("original-input", "caller_data", "原始约束：只读分析，不修改或部署。")
    sources = [required, *(MaterialSource("tool:fixture-" + str(i), "executor_result", text,
        required=False, provenance={"invocation_id": "fixture-" + str(i)}) for i, text in enumerate(originals))]
    transcript = [{"role": "tool", "name": "fixture_read", "content": "最新工具反馈" * 1400}]
    state, active = state_for(task, store, sources, transcript=transcript)
    before_package = copy.deepcopy(task["material_package"])
    last = before_package["items"][-1]
    # 先留下真实尾部读取回执，确认提示压缩不会重写或清零累计读取事实。
    text = source_text(originals[-1])
    receipt = active.read(last["reference"], offset=len(text) - 40, length=40)
    reads_before = store.read_receipts(task)
    limits_before = store.limits(task)
    information_before = active.information()
    assert len(json.dumps(before_package, ensure_ascii=False).encode()) > EXECUTION_CONTEXT_MAX_BYTES

    parts = build_execution_prompt(state)
    payload = decode_payload(parts.user_prompt)
    projected = payload["task_materials"]
    assert len(parts.user_prompt.encode("utf-8")) <= EXECUTION_CONTEXT_MAX_BYTES
    assert len(parts.user_prompt.encode("utf-8")) + sum(
        len(block.encode("utf-8")) for block in parts.system_extra_blocks) <= EXECUTION_CONTEXT_MAX_BYTES
    assert payload["requests"] == task["requests"]
    assert payload["contract"] == task["contract"]
    assert payload["authorization"] == {"reference": "unchanged-authority", "mode": "read"}
    assert projected["items"][0] == before_package["items"][0]
    assert projected["prompt_projection"]["read_records_unchanged"] is True
    assert len(projected["items"]) == len(before_package["items"])
    for original, view in zip(before_package["items"], projected["items"]):
        assert {k: v for k, v in view.items() if k not in {"content", "omitted_chars"}} == {
            k: v for k, v in original.items() if k not in {"content", "omitted_chars"}}
        assert original["content"].startswith(view["content"])
        assert view["omitted_chars"] == view["content_chars"] - len(view["content"])
    assert task["material_package"] == before_package
    assert store.read_receipts(task) == reads_before
    assert store.limits(task) == limits_before
    assert active.information() == information_before
    assert build_execution_prompt(state) == parts
    for item, original in zip(before_package["items"][1:], originals):
        assert store.resolve(item["reference"], task=task, still_authorized=lambda: True) == original
    assert active.read(last["reference"], offset=len(text) - 40, length=40) == receipt
    assert store.limits(task) == limits_before
    # 存储重开后，源哈希、正文和已有读取回执仍可核验。
    ledger.save(task)
    restored = TaskMaterialStore(ledger)
    assert restored.resolve(last["reference"], task=task, still_authorized=lambda: True) == originals[-1]
    assert restored.read_receipts(task) == reads_before


def test_small_context_does_not_rewrite_material_descriptors(material_context):
    _, task, store = material_context
    state, active = state_for(task, store, [MaterialSource("tool:small", "executor_result", "small", required=False)])
    payload = decode_payload(build_execution_prompt(state).user_prompt)
    assert payload["task_materials"] == active.prompt_package()
    assert "prompt_projection" not in payload["task_materials"]


def test_many_large_tool_feedbacks_share_budget_and_preserve_original_receipts(material_context):
    _, task, store = material_context
    source = "长工具原文和读取证据\n" * 5000
    state, active = state_for(task, store, [MaterialSource("tool:large", "executor_result", source, required=False)])
    reference = active.prompt_package()["items"][0]
    state.transcript = [{"role": "tool", "name": "fixture_read", "content": source,
                         "material_reference": {"reference": reference["reference"]},
                         "feedback_information": {"complete": True}} for _ in range(12)]
    before = copy.deepcopy(state.transcript)
    original_task = copy.deepcopy(task)
    parts = build_execution_prompt(state)
    assert len(parts.user_prompt.encode()) + sum(len(x.encode()) for x in parts.system_extra_blocks) <= EXECUTION_CONTEXT_MAX_BYTES
    assert state.transcript == before and task == original_task
    assert '"full_source_retained": true' in parts.user_prompt
    assert store.resolve(reference["reference"], task=task, still_authorized=lambda: True) == source


def test_feedback_without_a_retained_reference_is_not_silently_cut(material_context):
    _, task, store = material_context
    state, _ = state_for(task, store, [MaterialSource("tool:small", "executor_result", "small", required=False)])
    state.transcript = [{"role": "tool", "content": "无法重新取得的原始反馈" * 20000}]
    with pytest.raises(AgentBoundaryError, match="execution_context_oversized"):
        build_execution_prompt(state)


@pytest.mark.parametrize("oversized", ["request", "contract", "required_source"])
def test_core_or_required_input_is_never_silently_shortened(material_context, oversized):
    _, task, store = material_context
    sources = [MaterialSource("tool:large", "executor_result", "工具正文" * 6000, required=False)]
    if oversized == "required_source":
        # 多份必需资料全部保留；过大的核心输入仍明确拒绝。
        sources += [MaterialSource("input-" + str(i), "caller_data", "必需原文" * 3000) for i in range(10)]
    state, _ = state_for(task, store, sources)
    if oversized == "request":
        task["requests"][0]["text"] = "必须保留的用户原始要求" * 10000
    elif oversized == "contract":
        task["contract"] = {"goal": "不能裁剪的验收约定" * 10000}
    original = copy.deepcopy(task)
    with pytest.raises(AgentBoundaryError, match="execution_context_oversized"):
        build_execution_prompt(state)
    assert task == original


def test_large_core_without_material_session_still_refuses():
    task = {"requests": [{"text": "原始要求" * 30000}]}
    state = SimpleNamespace(context={"agent_task_state": task}, transcript=[])
    with pytest.raises(AgentBoundaryError, match="execution_context_oversized"):
        build_execution_prompt(state)
