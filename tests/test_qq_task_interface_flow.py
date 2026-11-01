"""P4d task interfaces through materials, approval, recovery and delivery."""

import copy
from dataclasses import replace
import hashlib
import json
import threading
import time

import pytest

from shinku.agent.approval import ConsentDecision
from shinku.agent.loop import AgentRunResult
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.result_expression import replay_task_expression
from shinku.result_expression import ExpressionProfile
from shinku.result_expression.facts import identity
from test_agent_material_integration import media_turn, PIXEL
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat, boundary_bridge, contract_runtime, submit
from test_result_expression import Model


SCOPE = AgentScope("fixture", "private:fixture")
FINAL = {"kind": "final", "text": "The isolated file was created and verified.", "task_outcome": "completed"}


def task(svc):
    rows = svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)
    assert len(rows) == 1
    return rows[0]


def drain(svc, reply):
    if not reply.background_task_id:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        work = svc.coordinator.get(reply.background_task_id)
        if not work["worker_active"] and work["status"] not in {"queued", "running"}:
            return work
        time.sleep(.005)
    raise AssertionError("isolated worker did not stop")


@pytest.fixture
def host(tmp_path):
    svc, tool, execution = demo_service(tmp_path)
    request = demo_request(tool)
    metadata = contract_runtime(execution, request.contract)
    chat, expression, sent = Chat([submit()]), Model(), []
    def sender(message):
        sent.append(message)
        return DeliveryResult(True, "sent", (f"test-send-{len(sent)}",))
    bridge = boundary_bridge(svc, tool, chat, recovery_enabled=True,
        material_store=TaskMaterialStore(svc.task_store, approved_roots=(tmp_path,)),
        expression_profile=ExpressionProfile(expression, configuration_id="task-boundary-fixture"), sender=sender)
    yield svc, tool, execution, request, bridge, chat, expression, sent, metadata
    svc.coordinator.shutdown(wait=True)
    svc.task_store.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("delivery", ["sent", "failed", "uncertain", "dry_run"])
def test_td10_actual_expression_event_transport_and_replay_keep_one_effect(host, asynchronous, delivery):
    svc, tool, execution, request, bridge, chat, model, sent, metadata = host
    bridge.async_agent_enabled = asynchronous
    def sender(message):
        sent.append(message)
        if delivery == "uncertain":
            raise TimeoutError("isolated transport status unknown")
        return DeliveryResult(delivery == "sent", delivery,
            (f"test-out-{len(sent)}",) if delivery == "sent" else (), definitive_not_sent=delivery == "failed")
    bridge.sender = sender if delivery != "dry_run" else None
    reply = bridge(turn(request.original_request))
    drain(svc, reply)
    saved = task(svc)
    result = saved["execution_result"]
    event = svc.task_store.result_events.get(svc.task_store.result_events.key(result), scope=SCOPE)
    record = bridge.result_expression.archive.get(canonical_hash(identity(result)), scope=SCOPE)
    assert saved["status"] == "completed" and result["verification"]["accepted"]
    assert event["status"] == {"failed": "failed", "uncertain": "uncertain"}.get(delivery, "done")
    assert record["delivery"]["status"] == delivery
    assert len(model.calls) == 1 and len(execution.calls) == 2 and len(metadata) == 1 and tool.calls == 1
    assert tool.tool_type not in json.dumps(chat.calls) and all(not c.get("native_tools") for c in model.calls)
    if delivery == "sent":
        assert record["text"] == "\n".join(part.text for msg in sent for part in msg.segments if part.kind == "text")
        assert event["delivery"]["sent"]
    before = saved["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls), len(sent)
    replay_task_expression(bridge, turn("读取旧事件", message_id="old-event"), task_id=saved["task_id"], context={})
    repeated = bridge._finish_result(turn(request.original_request), AgentRunResult(status="completed", rounds=1,
        final_text=saved["last_report"], completion=copy.deepcopy(saved["completion"]), reason=saved["stop_reason"]),
        context={"agent_task_state": saved}, initial_transcript=[], task_id=saved["task_id"])
    assert repeated.delivery_status == "suppressed"
    assert before == (task(svc)["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls), len(sent))


@pytest.mark.parametrize("operation", ["status", "result"])
def test_td05_native_task_read_does_not_plan_execute_or_repeat_polish(host, operation):
    svc, tool, execution, request, bridge, chat, model, sent, metadata = host
    bridge(turn(request.original_request))
    saved = task(svc)
    before = saved["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls)
    chat.replies.append({"kind": "tool", "tool_call": {"name": "agent_task_" + operation,
                       "arguments": {"task_id": saved["task_id"]}}})
    answer = bridge(turn("读取这个编号的规范信息", message_id="interface-read"))
    assert answer.reason == "task_interface_" + operation and "验收通过" in answer.final_text, answer
    assert before == (task(svc)["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls))
    assert all(tool.tool_type not in json.dumps(c.get("native_tools")) + json.dumps(c["system_extra_blocks"])
               for c in chat.calls)


def test_td08_duplicate_original_submission_does_not_repeat_contract_effect_or_polish(host):
    svc, tool, execution, request, bridge, chat, model, sent, metadata = host
    original = turn(request.original_request)
    bridge(original)
    before = task(svc)["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls), len(sent)
    chat.replies.append(submit())
    reply = bridge(original)
    assert reply.reason in {"task_request_already_admitted", "conversation_turn_already_handled"}, reply
    assert before[:5] == (task(svc)["revision"], tool.calls, len(execution.calls), len(metadata), len(model.calls))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_td06_original_command_quote_file_tail_and_image_reach_executor(host, tmp_path, asynchronous):
    svc, tool, execution, request, bridge, chat, model, _, metadata = host
    bridge.async_agent_enabled = asynchronous
    bridge.execution_adapter.executor.profile = replace(bridge.execution_adapter.executor.profile, supports_images=True)
    attachment = tmp_path / "source.txt"
    attachment.write_text("Source body. " * 600 + "TAIL_CONSTRAINT: offline only; retain this file.", encoding="utf-8")
    original = request.original_request + " 用户指定命令 echo RAW_COMMAND；按之前的方案，保留附件。"
    source = media_turn(original, quote=[{"type": "text", "data": {"text": "原方案及约束 QUOTED_CONSTRAINT"}}])
    file_segment = IncomingMessage.from_event({"user_id": "fixture", "message": [
        {"type": "file", "data": {"path": str(attachment)}}]}).segments[0]
    source = replace(source, messages=(replace(source.messages[0], segments=(*source.messages[0].segments, file_segment)),))
    actual = execution.call_chat_json
    def read_then_write(**options):
        if options.get("native_tools") and not tool.calls:
            data = json.loads(options["user_prompt"].split("\n", 1)[1].split("\n[Task materials", 1)[0])
            package = data["task_materials"]
            item = next(row for row in package["items"] if row["kind"] == "attachment")
            receipts = bridge.material_store.read_receipts({"task_id": package["task_id"],
                "requester": SCOPE.requester, "session": SCOPE.session})
            offset = max((row["offset"] + row["returned_chars"] for row in receipts
                          if row["reference"] == item["reference"]), default=0)
            if offset < len(attachment.read_text(encoding="utf-8")):
                execution.calls.append(options)
                return {"tool_call": {"type": "task_material_read", "reference": item["reference"],
                                      "offset": offset, "length": 4096}}
        return actual(**options)
    execution.call_chat_json = read_then_write
    reply = bridge(source)
    drain(svc, reply)
    saved = task(svc)
    assert saved["status"] == "completed" and tool.calls == 1, (reply.reason, saved["stop_reason"],
        saved["last_report"], len(execution.calls))
    prompts = json.dumps(execution.calls, ensure_ascii=False)
    assert "RAW_COMMAND" in prompts and "QUOTED_CONSTRAINT" in prompts and "TAIL_CONSTRAINT" in prompts
    images = next(c["user_images"] for c in execution.calls if c.get("user_images"))
    assert images[0]["data_url"] == PIXEL and images[0]["message_id"] == "fixture-message"
    assert saved["execution_result"]["information"]["required_status"] == "complete"
    assert "PERSONA_CHAT_SENTINEL" not in prompts and len(model.calls) == 1
    assert attachment.read_text(encoding="utf-8").endswith("retain this file.")
    assert "RAW_COMMAND" in json.dumps(metadata, ensure_ascii=False)


@pytest.mark.parametrize("gap", ["unsupported_image", "missing_context", "missing_image"])
def test_td06_missing_or_unsupported_material_is_truthful_and_never_falls_back(host, gap):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    if gap == "unsupported_image":
        bridge.execution_adapter.executor.profile = replace(bridge.execution_adapter.executor.profile, supports_images=False)
        source = media_turn(request.original_request)
    elif gap == "missing_image":
        source = media_turn(request.original_request, payload={"url": ""})
    else:
        source = turn(request.original_request + " 按之前那个方案。")
    reply = bridge(source)
    saved = task(svc)
    assert reply.agent_status == "waiting_user" and tool.calls == 0 and execution.calls == [], reply
    assert len(chat.calls) == 1 and len(metadata) == 1
    assert saved["execution_result"]["information"]["required_status"] == "incomplete"
    assert not saved["execution_result"]["verification"]["accepted"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_td08_executor_selected_approval_uses_original_store_and_revision(host, asynchronous, stale):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    bridge.async_agent_enabled = asynchronous
    bridge.command_state.set_permission_mode("confirm")
    bridge.allowed_tool_names = {tool.tool_type}
    class Consent:
        def resolve(self, **kwargs):
            return ConsentDecision("approve", 1.0, "explicit isolated approval")
    bridge.consent_resolver = Consent()
    first = bridge(turn(request.original_request))
    drain(svc, first)
    saved = task(svc)
    pending = bridge.approval_store.get("private:fixture")
    assert pending is not None and tool.calls == 0 and len(execution.calls) == 1
    assert pending.request.invocations[0].name == tool.tool_type
    if stale:
        saved["notes"]["blocker"] = "newer host record"
        svc.task_store.save(saved)
    reply = bridge(turn("同意执行", message_id="approval"))
    drain(svc, reply)
    if stale:
        assert reply.reason == "task_version_conflict" and tool.calls == 0, reply
        assert bridge.approval_store.get("private:fixture") is not None
    else:
        assert task(svc)["status"] == "completed" and tool.calls == 1, reply
        assert bridge.approval_store.get("private:fixture") is None
        assert task(svc)["execution_attempt"] == 2
    assert len(chat.calls) == 1 and len(metadata) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("uncertain", [False, True])
def test_td08_resume_preserves_real_effect_receipt_and_cumulative_budget(host, asynchronous, uncertain):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    bridge.async_agent_enabled = asynchronous
    base = execution.call_chat_json
    def outage(**options):
        if options.get("native_tools") and tool.calls:
            execution.calls.append(options)
            raise RuntimeError("isolated execution interrupted after write")
        return base(**options)
    execution.call_chat_json = outage
    drain(svc, bridge(turn(request.original_request)))
    before = task(svc)
    assert before["status"] == "failed" and tool.calls == 1
    digest = hashlib.sha256(tool.path.read_bytes()).hexdigest()
    if uncertain:
        before["inflight"] = [{"name": tool.tool_type, "arguments": {"path": str(tool.path)}, "status": "unknown"}]
        before = svc.task_store.save(before)
    execution.call_chat_json = lambda **options: (execution.calls.append(options) or FINAL)
    chat.replies.append({"kind": "tool", "tool_call": {"name": "agent_task_resume",
        "arguments": {"task_id": before["task_id"]}}})
    reply = bridge(turn("继续核验 " + before["task_id"], message_id="resume-original"))
    drain(svc, reply)
    after = task(svc)
    assert tool.calls == 1 and hashlib.sha256(tool.path.read_bytes()).hexdigest() == digest
    assert len(metadata) == 1
    assert all(tool.tool_type not in json.dumps(c.get("native_tools")) + json.dumps(c["system_extra_blocks"])
               for c in chat.calls)
    if uncertain:
        assert after["execution_attempt"] == before["execution_attempt"] and after["status"] != "completed"
    else:
        assert after["status"] == "completed" and after["execution_attempt"] == 2, reply
        assert after["contract"] == before["contract"] and len(after["recovery_history"]) == 1
        assert after["budget"]["requests"] >= before["budget"]["requests"] + 1
        assert after["budget"]["tools"] == before["budget"]["tools"]
        assert after["execution_result"]["verification"]["accepted"]


def test_td08_real_single_worker_cross_slice_keeps_one_task_and_first_planner(host):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    bridge.async_agent_enabled = True
    bridge.execution_adapter.executor.profile = replace(bridge.execution_adapter.executor.profile, max_rounds=1)
    results = []
    actual = bridge._run_agent
    def capture(**kwargs):
        result = actual(**kwargs)
        results.append(result.status)
        return result
    bridge._run_agent = capture
    reply = bridge(turn(request.original_request))
    drain(svc, reply)
    assert results == ["yielded", "completed"] and len(svc.coordinator.list()) == 1
    assert task(svc)["status"] == "completed" and tool.calls == 1 and len(execution.calls) == 2
    assert len(chat.calls) == 1 and len(metadata) == 1


@pytest.mark.parametrize("failure", ["wrong_tool", "bad_arguments", "adapter_failure"])
def test_td07_initial_executor_error_is_corrected_in_same_loop_and_budget(host, failure, tmp_path):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    actual = execution.call_chat_json
    first = [True]
    attempts = []
    if failure == "adapter_failure":
        real_tool = tool.execute
        def transient(**options):
            attempts.append(options)
            if len(attempts) == 1:
                return {"tool_type": tool.tool_type, "status": "error", "ok": False,
                        "followup_context": "FIXTURE_TRANSIENT_FAILURE; no file effect"}
            return real_tool(**options)
        tool.execute = transient
    def wrong_then_correct(**options):
        if options.get("native_tools") and first[0]:
            first[0] = False
            execution.calls.append(options)
            if failure == "wrong_tool":
                return {"tool_call": {"type": "unregistered", "path": str(tool.path)}}
            if failure == "bad_arguments":
                return {"tool_call": {"type": tool.tool_type, "path": str(tmp_path / "denied.txt")}}
            return {"tool_call": {"type": tool.tool_type, "path": str(tool.path)}}
        return actual(**options)
    execution.call_chat_json = wrong_then_correct
    reply = bridge(turn(request.original_request))
    saved = task(svc)
    assert reply.agent_status == "completed" and tool.calls == 1, reply
    assert len(execution.calls) == 3 and len(metadata) == 1 and len(chat.calls) == 1
    assert saved["contract"] == request.contract and saved["execution_attempt"] == 1
    assert saved["execution_result"]["verification"]["accepted"]
    assert any(row["status"] == "error" for row in saved["tool_calls"])
    assert saved["budget"]["requests"] == 4 and saved["budget"]["tools"] <= 2
    assert "tool_use_error" in json.dumps(execution.calls[1]) or "FIXTURE_TRANSIENT_FAILURE" in json.dumps(execution.calls[1])
    assert not (tmp_path / "denied.txt").exists()


@pytest.mark.parametrize("failure", ["budget", "revoked", "metadata_authority"])
def test_td09_budget_revocation_and_metadata_authority_stop_without_fallback(host, failure):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    if failure == "budget":
        bridge.task_budget_limits["max_requests"] = 1
    elif failure == "metadata_authority":
        execution.call_chat_json = lambda **kw: {"task_link": {"mode": "new"}, "task_contract": request.contract,
                                               "first_decision": {"tool_call": {"type": tool.tool_type}}}
    else:
        actual = execution.call_chat_json
        def revoke(**options):
            if options.get("native_tools"):
                bridge.command_state.set_agent_enabled(False)
            return actual(**options)
        execution.call_chat_json = revoke
    reply = bridge(turn(request.original_request))
    assert reply.agent_status not in {"completed", "chat"} and tool.calls == 0 and len(chat.calls) == 1, reply
    assert not tool.path.exists()
    for row in svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session):
        assert row["status"] != "completed" and row["completion"].get("accepted") is not True
    assert tool.tool_type not in json.dumps(chat.calls)


def test_td08_scoped_cancel_of_running_worker_prevents_late_planner_effect(host):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    bridge.async_agent_enabled = True
    entered, release = threading.Event(), threading.Event()
    actual = execution.call_chat_json
    def blocked_plan(**options):
        if options.get("native_tools"):
            entered.set()
            assert release.wait(3)
        return actual(**options)
    execution.call_chat_json = blocked_plan
    first = bridge(turn(request.original_request))
    try:
        assert entered.wait(2)
        saved = task(svc)
        chat.replies.append({"kind": "tool", "tool_call": {"name": "agent_task_cancel",
                             "arguments": {"task_id": saved["task_id"]}}})
        canceled = bridge(turn("取消 " + saved["task_id"], message_id="cancel-running"))
        assert canceled.reason == "task_interface_cancel", canceled
        release.set()
        work = drain(svc, first)
        assert not work["worker_active"] and work["status"] == "canceled"
        assert task(svc)["status"] == "canceled" and tool.calls == 0
        assert len(svc.coordinator.list()) == 1 and len(metadata) == 1
    finally:
        release.set()


@pytest.mark.parametrize("bad", ["wrong_scope", "unrequested", "completed", "mode_conflict"])
def test_td08_task_mutation_rejects_wrong_scope_unrequested_or_completed_resume(host, bad):
    svc, tool, execution, request, bridge, chat, model, _, metadata = host
    bridge(turn(request.original_request))
    saved = task(svc)
    if bad == "mode_conflict":
        saved["status"] = "failed"
        saved["tool_decision_owner"] = "legacy"
        saved = svc.task_store.save(saved)
    chat.replies.append({"kind": "tool", "tool_call": {"name": "agent_task_resume",
                         "arguments": {"task_id": saved["task_id"]}}})
    before = tool.calls, len(execution.calls), len(metadata), len(model.calls), saved["revision"]
    reply = bridge(turn("继续 " + saved["task_id"] if bad != "unrequested" else "看看这个编号的记录",
                        user="another" if bad == "wrong_scope" else "fixture", message_id="rejected-resume"))
    assert reply.agent_status == ("failed" if bad == "mode_conflict" else "blocked"), reply
    assert before == (tool.calls, len(execution.calls), len(metadata), len(model.calls), task(svc)["revision"])


def test_td06_changed_attachment_is_not_used_for_an_effect_or_completion(host, tmp_path):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    path = tmp_path / "input.txt"
    path.write_text("original file constraint", encoding="utf-8")
    source = media_turn(request.original_request, kind="file", payload={"path": str(path)})
    actual = execution.call_chat_json
    def conflict(**options):
        if options.get("native_tools"):
            path.write_text("another actor changed the source", encoding="utf-8")
        return actual(**options)
    execution.call_chat_json = conflict
    reply = bridge(source)
    assert reply.agent_status != "completed" and tool.calls == 0, reply
    assert task(svc)["execution_result"]["verification"].get("accepted") is not True and len(chat.calls) == 1
    assert len(metadata) == 1


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_td10_non_success_result_is_not_upgraded_by_expression(host, status):
    svc, tool, execution, request, bridge, chat, model, sent, metadata = host
    if status == "partial":
        request.contract["requirements"].append({"id": "other", "description": "A second verified file",
            "effect": "write", "targets": [str(tool.path.parent / "other.txt")]})
    actual = execution.call_chat_json
    def limited(**options):
        if options.get("native_tools") and (status == "failed" or tool.calls):
            execution.calls.append(options)
            return {"kind": "final", "text": "The other file is still missing.", "task_outcome": status}
        return actual(**options)
    execution.call_chat_json = limited
    reply = bridge(turn(request.original_request))
    saved = task(svc)
    assert reply.agent_status == status and saved["status"] == status, reply
    assert saved["execution_result"]["remaining"]
    assert saved["execution_result"]["verification"]["status"] == status
    assert not saved["execution_result"]["verification"]["evidence"][-1]["business_goal"]
    assert tool.calls == int(status == "partial") and len(model.calls) == 1 and len(metadata) == 1 and len(chat.calls) == 1
    event = svc.task_store.result_events.get(svc.task_store.result_events.key(saved["execution_result"]), scope=SCOPE)
    assert event["delivery"]["sent"] and event["status"] == "done"
    assert "验收通过" not in reply.final_text


def test_td08_supplement_is_scoped_and_does_not_auto_resume_or_add_budget(host):
    svc, tool, execution, request, bridge, chat, model, _, metadata = host
    bridge(turn(request.original_request + " 按之前那个方案。"))
    saved = task(svc)
    assert saved["status"] == "waiting_user" and tool.calls == 0
    before = tool.calls, len(execution.calls), len(metadata), len(model.calls), copy.deepcopy(saved["budget"])
    chat.replies.append({"kind": "tool", "tool_call": {"name": "agent_task_supplement",
                         "arguments": {"task_id": saved["task_id"]}}})
    reply = bridge(turn("补充 " + saved["task_id"] + " 原始约束仍为离线，保留已写文件。", message_id="supplement"))
    assert reply.reason == "task_interface_supplement", reply
    assert "尚未继续执行" in reply.final_text
    after = task(svc)
    assert before == (tool.calls, len(execution.calls), len(metadata), len(model.calls), after["budget"])
    assert after["execution_attempt"] == saved["execution_attempt"] and after["revision"] > saved["revision"]


def test_td05_direct_verified_image_chat_does_not_create_execution_task(host):
    svc, tool, execution, request, bridge, chat, model, _, metadata = host
    chat.replies[:] = [{"kind": "final", "text": "这张图是一个很小的点阵。"}]
    source = media_turn("看这张图片。")
    reply = bridge(source)
    assert reply.reason == "direct_chat_reply" and reply.agent_status == "completed", reply
    assert any(c.get("user_images") for c in chat.calls)
    assert not svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)
    assert tool.calls == 0 and execution.calls == [] and metadata == [] and model.calls == []


def test_td08_competing_resumes_cannot_claim_second_execution_attempt(host):
    svc, tool, execution, request, bridge, chat, _, _, metadata = host
    actual = execution.call_chat_json
    def outage(**options):
        if options.get("native_tools") and tool.calls:
            execution.calls.append(options)
            raise RuntimeError("isolated interruption after effect")
        return actual(**options)
    execution.call_chat_json = outage
    bridge(turn(request.original_request))
    saved = task(svc)
    bridge.async_agent_enabled = True
    entered, release = threading.Event(), threading.Event()
    def continue_plan(**options):
        execution.calls.append(options)
        entered.set()
        assert release.wait(3)
        return FINAL
    execution.call_chat_json = continue_plan
    resume = {"kind": "tool", "tool_call": {"name": "agent_task_resume", "arguments": {"task_id": saved["task_id"]}}}
    chat.replies.extend([resume, resume])
    first = bridge(turn("继续 " + saved["task_id"], message_id="first-resume"))
    try:
        assert entered.wait(2)
        count = len(execution.calls)
        second = bridge(turn("继续 " + saved["task_id"], message_id="second-resume"))
        release.set()
        drain(svc, first)
        drain(svc, second)
        after = task(svc)
        assert after["status"] == "completed" and after["execution_attempt"] == 2
        assert len(after["recovery_history"]) == 1 and len(execution.calls) == count and tool.calls == 1
        assert len(metadata) == 1
    finally:
        release.set()
