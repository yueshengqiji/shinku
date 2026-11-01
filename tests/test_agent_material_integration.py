"""I01-I10 exercised through actual admission, executor, media and persistence."""
from dataclasses import replace
import hashlib
import json
import time

import pytest
import copy

from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentSupplement, ExecutionGrant, ExecutionProfile, TaskHandle
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatVisualInputBridge
from shinku.qq.turns import QQTurn
from shinku.result_expression import ExpressionProfile
from shinku.tools.execution import ExecutionPolicy
from test_qq_independent_agent import turn
from test_result_expression import Model


PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a9sAAAAASUVORK5CYII="
SCOPE = AgentScope("fixture", "private:fixture")


def wait_client(service, task_id):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        view = service.bind(SCOPE).status(task_id)
        if not view["worker_active"] and view["status"] not in {"running", "queued"}:
            return view
        time.sleep(.005)
    raise AssertionError("offline material task did not settle")


def grant_for(tool, *, sources=(), gaps=(), authorized=lambda: True):
    return ExecutionGrant("fixture-material-grant", frozenset({tool.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type: "low"}), permission_mode="ultra",
        budget_limits={"max_requests": 16, "max_tokens": 600000, "max_tools": 12, "max_active_seconds": 30},
        still_authorized=authorized, resume_safe=True, material_sources=tuple(sources), material_gaps=tuple(gaps))


def close(service):
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


def test_i03_i07_actual_standalone_reads_required_pages_before_file_effect(tmp_path):
    svc, tool, _ = demo_service(tmp_path)
    text = "a" * 125 + "条件：仅离线，不部署。"
    materials = TaskMaterialStore(svc.task_store, policy=MaterialPolicy(inline_chars=32, page_chars=64))
    svc.materials = materials
    grant = grant_for(tool, sources=[MaterialSource("source", "document", text)])
    svc.authorize = lambda scope, task, operation: grant
    class Runtime:
        calls = []
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            if not tool.calls:
                package = json.loads(kwargs["user_prompt"].split("\n", 1)[1].split("\n[Task materials", 1)[0])["task_materials"]
                reference = package["items"][0]["reference"]
                receipts = materials.read_receipts({"task_id": package["task_id"], "requester": SCOPE.requester, "session": SCOPE.session})
                offset = max((r["offset"] + r["returned_chars"] for r in receipts), default=0)
                if offset < len(text):
                    return {"tool_call": {"type": "task_material_read", "reference": reference, "offset": offset, "length": 64}}
                assert "不部署" in json.dumps(receipts, ensure_ascii=False)
                return {"tool_call": {"type": tool.tool_type, "path": str(tool.path)}}
            return {"kind": "final", "text": "The isolated file was created and verified, offline only.", "task_outcome": "completed"}
    runtime = Runtime()
    svc.executor.profile = ExecutionProfile(runtime, max_rounds=8)
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        view = wait_client(svc, handle.task_id)
        payload = svc.bind(SCOPE).result(handle.task_id)
        assert view["status"] == "completed" and tool.calls == 1 and len(runtime.calls) == 5
        assert payload["information"]["required_status"] == "complete"
        assert payload["budget"]["tools"] == 4
        before = tool.calls, len(runtime.calls), tool.path.read_bytes()
        svc.bind(SCOPE).submit(demo_request(tool))
        svc.bind(SCOPE).result(handle.task_id)
        svc.bind(SCOPE).status(handle.task_id)
        assert before == (tool.calls, len(runtime.calls), tool.path.read_bytes())
        assert "task_material_read" not in svc.executor.tool_host.registry.names()
    finally:
        close(svc)


def test_i07_unread_required_source_blocks_admitted_write_and_false_completion(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    svc.materials = TaskMaterialStore(svc.task_store, policy=MaterialPolicy(inline_chars=8, page_chars=64))
    grant = grant_for(tool, sources=[MaterialSource("source", "document", "a" * 100)])
    svc.authorize = lambda scope, task, operation: grant
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        view = wait_client(svc, handle.task_id)
        assert view["status"] != "completed" and tool.calls == 0 and not tool.path.exists()
        payload = svc.bind(SCOPE).result(handle.task_id)
        assert payload["information"]["required_status"] == "incomplete"
    finally:
        close(svc)


def media_turn(request, *, kind="image", payload=None, quote=None):
    parts = [{"type": "text", "data": {"text": request}}, {"type": kind, "data": payload or {"url": PIXEL}}]
    if quote is not None:
        parts.append({"type": "reply", "data": {"id": "original-plan", "message": quote,
            "user_id": "fixture", "conversation_id": "fixture"}})
    message = IncomingMessage.from_event({"post_type": "message", "message_type": "private", "user_id": "fixture",
        "message_id": "fixture-message", "self_id": "fixture-bot", "message": parts})
    visual = NapCatVisualInputBridge().build(message)
    return QQTurn("fixture", "private", "fixture-message", ("fixture-message",), request, (message,),
                  visual_inputs=((message.message_id, visual.inputs),) if visual.inputs else ())


def bridge_for(svc, tool, *, executor=True, asynchronous=False, supports_images=True):
    request = demo_request(tool)
    class Chat:
        def __init__(self):
            self.calls = []
        def chat_supports_native_tools(self):
            return True
        def chat_supports_images(self):
            return supports_images
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            if not tool.calls:
                return {"kind": "tool", "tool_call": {"type": tool.tool_type, "path": str(tool.path)},
                    "task_link": {"mode": "new"}, "task_contract": request.contract}
            return {"kind": "final", "text": "The isolated file was created and verified.", "task_outcome": "completed"}
    chat, expression = Chat(), Model()
    profile = replace(svc.executor.profile, supports_images=supports_images)
    materials = TaskMaterialStore(svc.task_store, approved_roots=(tool.path.parent,))
    bridge = QQAgentBridge(runtime=chat, system_prompt="PRIVATE_PERSONA_SENTINEL", command_state=QQCommandState(
        agent_enabled=True, permission_mode="ultra"), tool_host=svc.executor.tool_host,
        task_store=svc.task_store, task_coordinator=svc.coordinator, task_contract_enabled=True,
        execution_profile=profile if executor else None, material_store=materials,
        expression_profile=ExpressionProfile(expression, configuration_id="fixture-materials"),
        async_agent_enabled=asynchronous,
        task_budget_limits={"max_requests": 16, "max_tokens": 600000, "max_tools": 12, "max_active_seconds": 30})
    return bridge, chat, expression, materials


@pytest.mark.parametrize("executor", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_i02_i10_real_qq_media_and_attachment_to_execution_and_expression(tmp_path, executor, asynchronous):
    svc, tool, runtime = demo_service(tmp_path)
    request = demo_request(tool)
    attachment = tmp_path / "input.txt"
    attachment.write_text("Fixture input. Condition: offline only, no deployment.", encoding="utf-8")
    base = media_turn(request.original_request)
    message = replace(base.messages[0], segments=(*base.messages[0].segments,
        IncomingMessage.from_event({"user_id": "fixture", "message": [{"type": "file", "data": {"path": str(attachment)}}]}).segments[0]))
    source = replace(base, messages=(message,))
    bridge, chat, expression, materials = bridge_for(svc, tool, executor=executor, asynchronous=asynchronous)
    try:
        reply = bridge(source)
        if asynchronous:
            assert reply.agent_status == "waiting_background"
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                rows = svc.coordinator.list()
                if rows and not rows[0]["worker_active"] and rows[0]["status"] != "queued":
                    break
                time.sleep(.005)
        task = svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        result = task["execution_result"]
        assert task["status"] == "completed" and tool.calls == 1
        assert result["information"]["required_status"] == "complete"
        calls = runtime.calls if executor else chat.calls[1:]
        assert calls and any(row.get("user_images") for row in calls)
        images = next(row["user_images"] for row in calls if row.get("user_images"))
        assert images[0]["data_url"] == PIXEL
        assert images[0]["message_id"] == "fixture-message"
        assert "Condition: offline only, no deployment." in json.dumps(calls)
        if executor:
            assert "PRIVATE_PERSONA_SENTINEL" not in json.dumps(calls)
        assert len(expression.calls) == 1
        assert "资料交接完整性" in expression.calls[0]["user_prompt"]
        assert "data:image" not in expression.calls[0]["user_prompt"]
        assert len(svc.coordinator.list()) == (1 if asynchronous else 0)
        image_source = next(row for row in task["material_package"]["items"] if row["kind"] == "image")
        assert image_source["media"]["sha256"] == hashlib.sha256(__import__("base64").b64decode(PIXEL.split(",")[1])).hexdigest()
        assert attachment.read_text(encoding="utf-8").startswith("Fixture input.")
    finally:
        close(svc)


@pytest.mark.parametrize("supports_images", [False, None])
def test_i02_unsupported_media_stops_before_first_effect_and_executor_model(tmp_path, supports_images):
    svc, tool, runtime = demo_service(tmp_path)
    bridge, _, _, _ = bridge_for(svc, tool, supports_images=supports_images)
    try:
        reply = bridge(media_turn(demo_request(tool).original_request))
        assert reply.agent_status == "waiting_user" and tool.calls == 0 and runtime.calls == []
        task = svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["execution_result"]["information"]["required_status"] == "incomplete"
        assert task["execution_result"]["verification"]["accepted"] is not True
    finally:
        close(svc)


def test_i01_missing_referenced_original_requests_clarification(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    bridge, _, _, _ = bridge_for(svc, tool)
    request = demo_request(tool).original_request + " 按之前那个方案。"
    try:
        reply = bridge(turn(request))
        assert reply.agent_status == "waiting_user" and tool.calls == 0 and runtime.calls == []
        task = svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["requests"][0]["text"] == request
        assert any(g["reason"] == "material_related_context_missing" for g in task["execution_result"]["information"]["gaps"])
    finally:
        close(svc)


def test_i01_quote_original_enters_independent_input_without_unrelated_private_history(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    bridge, _, _, _ = bridge_for(svc, tool)
    request = demo_request(tool).original_request + " 按之前那个方案，不要改那个文件。"
    source = media_turn(request, quote=[{"type": "text", "data": {"text": "原方案：输出报告；保持 input.txt 不变。"}}])
    try:
        reply = bridge(source)
        assert reply.agent_status == "completed" and tool.calls == 1
        prompts = json.dumps(runtime.calls, ensure_ascii=False)
        assert "原方案：输出报告；保持 input.txt 不变。" in prompts
        assert "original-plan" in prompts and "PRIVATE_PERSONA_SENTINEL" not in prompts
    finally:
        close(svc)


def test_i07_i09_supplement_keeps_task_budget_and_old_result_event_without_auto_resume(tmp_path):
    from shinku.result_expression import ExpressionArchive, ExpressionService
    from shinku.result_expression.events import ResultEventConsumer
    svc, tool, runtime = demo_service(tmp_path)
    materials = TaskMaterialStore(svc.task_store, policy=MaterialPolicy(inline_chars=32))
    svc.materials = materials
    grant = [grant_for(tool, sources=[MaterialSource("related_context", "related_message", "a" * 200)])]
    class PartialRuntime:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            return {"kind": "final", "text": "The necessary source has not been read yet.", "task_outcome": "partial"}
    partial_runtime = PartialRuntime()
    svc.executor.profile = ExecutionProfile(partial_runtime)
    svc.authorize = lambda scope, task, operation: grant[0]
    expression_model = Model()
    expression = ExpressionService(ExpressionArchive(svc.task_store), ExpressionProfile(expression_model))
    client = svc.bind(SCOPE)
    try:
        initial = client.submit(demo_request(tool))
        view = wait_client(svc, initial.task_id)
        old_result = client.result(initial.task_id)
        events = svc.task_store.result_events
        old_event = events.get(events.key(old_result), scope=SCOPE)
        before = tool.calls, len(runtime.calls), len(expression_model.calls)
        deadline = materials.limits({"task_id": initial.task_id, "requester": SCOPE.requester, "session": SCOPE.session})["deadline"]
        grant[0] = grant_for(tool, sources=[MaterialSource("related_context", "related_message",
            "先前方案：仅生成隔离报告，不部署。")])
        request = AgentSupplement("clarify-once", "确认原方案，仅生成报告，不部署。", ("不部署",))
        old_handle = TaskHandle(initial.task_id, view["attempt"], view["version"])
        supplemented = client.supplement(old_handle, request)
        assert client.supplement(old_handle, request) == supplemented
        assert before == (tool.calls, len(runtime.calls), len(expression_model.calls)) == (0, 0, 0)
        assert partial_runtime.calls == 1
        assert client.result(initial.task_id) == old_result
        current = svc.task_store.get(initial.task_id, requester=SCOPE.requester, session=SCOPE.session)
        assert len(current["requests"]) == 2 and current["requests"][0]["text"] == demo_request(tool).original_request
        assert current["confirmed_constraints"] == ["不部署"]
        assert materials.limits(current)["deadline"] == deadline
        assert events.get(events.key(old_result), scope=SCOPE)["event_id"] == old_event["event_id"]
        consumer = ResultEventConsumer(expression)
        old_replay = consumer.consume(old_event["event_id"], scope=SCOPE, check_current=lambda: "",
            publish=lambda *_: pytest.fail("supplement may not resend the old result"))
        assert old_replay["error"] == "stale_result_event" and expression_model.calls == []
        svc.executor.profile = ExecutionProfile(runtime)
        client.resume(supplemented, request_key="continue-approved")
        assert wait_client(svc, initial.task_id)["status"] == "completed"
        final = client.result(initial.task_id)
        assert tool.calls == 1 and len(runtime.calls) == 2
        assert final["budget"]["requests"] == 3 and final["budget"]["tools"] == 1
        assert len(svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)) == 1
        assert old_result["information"]["required_status"] == "incomplete"
        assert final["information"]["required_status"] == "complete"
    finally:
        close(svc)


def test_i06_i07_supplement_revalidates_host_permission_and_conflicting_retries(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    svc.materials = TaskMaterialStore(svc.task_store)
    enabled = [True]
    grant = grant_for(tool, gaps=[{"source_id": "related_context", "required": True, "reason": "source_missing"}],
                      authorized=lambda: enabled[0])
    svc.authorize = lambda *_: grant
    client = svc.bind(SCOPE)
    try:
        handle = client.submit(demo_request(tool))
        view = wait_client(svc, handle.task_id)
        pending = TaskHandle(handle.task_id, view["attempt"], view["version"])
        enabled[0] = False
        with pytest.raises(AgentBoundaryError, match="host_authorization_denied"):
            client.supplement(pending, AgentSupplement("supplement", "提供确认约束"))
        enabled[0] = True
        supplied = client.supplement(pending, AgentSupplement("supplement", "提供确认约束"))
        with pytest.raises(AgentBoundaryError, match="material_supplement_key_conflict"):
            client.supplement(pending, AgentSupplement("supplement", "另一项确认约束"))
        assert tool.calls == 0 and runtime.calls == []
        assert supplied.task_id == handle.task_id
    finally:
        close(svc)


def receipt_fixture(tmp_path, *, fact_count=1, policy=None):
    svc, tool, runtime = demo_service(tmp_path)
    original = tool.execute
    def execute(*, call, context):
        value = original(call=call, context=context)
        value["state_updates"]["result_facts"] = [{"label": "测量" + str(i), "value": 12.5 + i, "unit": "ms",
            "condition": ("a" * 3000 if i == 0 else "") + f"条件{i}：仅离线，不部署。",
            "reference": "https://example.invalid/fixture"} for i in range(fact_count)]
        value["followup_context"] = "result " + "b" * 3200
        return value
    tool.execute = execute
    materials = TaskMaterialStore(svc.task_store, policy=policy or MaterialPolicy(inline_chars=32))
    svc.materials = materials
    return svc, tool, runtime, materials


def test_delivery_does_not_reread_large_auxiliary_receipts_with_exhausted_quota(tmp_path):
    from shinku.agent_runtime.material_presentation import recover_presentation
    svc, tool, runtime = demo_service(tmp_path)
    materials = TaskMaterialStore(svc.task_store, policy=MaterialPolicy(inline_chars=32, max_read_chars=16))
    svc.materials = materials
    execute = tool.execute
    def large(**kwargs):
        result = execute(**kwargs)
        result['followup_context'] = 'PROCESS_DIAGNOSTIC_ONLY' * 1000
        return result
    tool.execute = large
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        wait_client(svc, handle.task_id)
        result = svc.bind(SCOPE).result(handle.task_id)
        task = svc.task_store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        before = copy.deepcopy(materials.limits(task))
        projection = recover_presentation(materials, result, scope=SCOPE, still_authorized=lambda: True, delivery=True)
        assert projection['status'] == 'complete' and not projection['gaps']
        assert not projection['sources'] and materials.limits(task) == before
        assert tool.calls == 1 and svc.bind(SCOPE).result(handle.task_id) == result
    finally:
        close(svc)


@pytest.mark.parametrize("fact_count", [1, 140])
@pytest.mark.parametrize('delivery', [False, True])
def test_i03_i04_i09_long_actual_receipt_restored_before_expression_and_cache_never_rereads(tmp_path, fact_count, delivery):
    from shinku.agent_runtime.material_presentation import recover_presentation
    from shinku.result_expression import ExpressionArchive, ExpressionService
    svc, tool, runtime, materials = receipt_fixture(tmp_path, fact_count=fact_count)
    model = Model()
    expression = ExpressionService(ExpressionArchive(svc.task_store), ExpressionProfile(model, max_input_chars=60000),
        material_resolver=lambda result, **kw: recover_presentation(materials, result, delivery=delivery, **kw))
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        assert wait_client(svc, handle.task_id)["status"] == "completed"
        result = svc.bind(SCOPE).result(handle.task_id)
        before = copy.deepcopy(result)
        assert len(result["facts"]) == min(fact_count, 128)
        assert result["facts"][0]["condition"]["omitted"] == "text_limit"
        record = expression.prepare(result, scope=SCOPE, check_current=lambda: "")
        assert record["status"] == "ready", record["fallback_reason"]
        assert "条件0：仅离线，不部署。" in record["text"] and f"条件{fact_count - 1}：仅离线，不部署。" in record["text"]
        assert len(record["presentation_projection"]["facts"]) == fact_count
        assert record["canonical_result"] == before and result == before
        usage = materials.limits({"task_id": handle.task_id, "requester": SCOPE.requester, "session": SCOPE.session})
        record2 = expression.prepare(result, scope=SCOPE, check_current=lambda: "")
        assert record2 == record and len(model.calls) == 1 and tool.calls == 1 and len(runtime.calls) == 2
        assert usage == materials.limits({"task_id": handle.task_id, "requester": SCOPE.requester, "session": SCOPE.session})
        assert model.calls[0]["native_tools"] is None
        assert all(x not in model.calls[0]["user_prompt"] for x in ["b" * 1200, "PERSONA_PROMPT_PRIVATE_SENTINEL"])
    finally:
        close(svc)


@pytest.mark.parametrize("action", ["revoke", "deadline", "quota", "unknown_upstream"])
@pytest.mark.parametrize('delivery', [False, True])
def test_i05_i07_failed_full_receipt_recovery_uses_one_honest_template_no_model_or_reexecution(tmp_path, action, delivery):
    from shinku.agent_runtime.material_presentation import recover_presentation
    from shinku.result_expression import ExpressionArchive, ExpressionService
    policy = MaterialPolicy(max_read_chars=16) if action == "quota" else MaterialPolicy()
    svc, tool, runtime, materials = receipt_fixture(tmp_path, policy=policy)
    if action == "unknown_upstream":
        original = tool.execute
        def execute(**kw):
            value = original(**kw)
            value["state_updates"]["raw_detail_unavailable"] = True
            return value
        tool.execute = execute
    model = Model()
    expression = ExpressionService(ExpressionArchive(svc.task_store), ExpressionProfile(model),
        material_resolver=lambda result, **kw: recover_presentation(materials, result, delivery=delivery, **kw))
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        wait_client(svc, handle.task_id)
        result = svc.bind(SCOPE).result(handle.task_id)
        task = svc.task_store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        if action == "revoke":
            materials.revoke(task["material_package"]["items"][0]["reference"], task=task)
        elif action == "deadline":
            deadline = materials.limits(task)["deadline"]
            svc.task_store.clock = lambda: deadline + 1
        record = expression.prepare(result, scope=SCOPE, check_current=lambda: "")
        assert record["status"] == "fallback" and record["fallback_reason"] == "expression_required_material_unavailable"
        assert "incomplete" in record["text"] and model.calls == []
        assert expression.prepare(result, scope=SCOPE, check_current=lambda: "") == record
        assert tool.calls == 1 and len(runtime.calls) == 2 and svc.bind(SCOPE).result(handle.task_id) == result
    finally:
        close(svc)


@pytest.mark.parametrize("executor", [False, True])
def test_i03_i10_real_qq_long_receipt_tail_reaches_delivery_candidate(tmp_path, executor):
    from shinku.agent_runtime.contracts import canonical_hash
    from shinku.result_expression.facts import identity
    svc, tool, runtime, _ = receipt_fixture(tmp_path, fact_count=140)
    bridge, chat, model, materials = bridge_for(svc, tool, executor=executor)
    try:
        reply = bridge(media_turn(demo_request(tool).original_request))
        assert reply.agent_status == "completed" and tool.calls == 1
        task = svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        source = task["execution_result"]
        record = bridge.result_expression.archive.get(canonical_hash(identity(source)), scope=SCOPE)
        assert record["status"] in {"ready", "fallback"}
        assert "条件0：仅离线，不部署。" in record["text"] and "条件139：仅离线，不部署。" in record["text"]
        assert record["canonical_result"] == source and len(record["presentation_projection"]["facts"]) == 140
        assert all(row["native_tools"] is None for row in model.calls)
        assert record["delivery"]["status"] == "dry_run"
        assert source["delivery"]["status"] == "not_recorded"
    finally:
        close(svc)


def test_i05_i09_cached_full_expression_ref_revoked_blocks_publication_without_another_read_or_model(tmp_path):
    from shinku.agent_runtime.material_presentation import recover_presentation, validate_presentation
    from shinku.result_expression import ExpressionArchive, ExpressionService
    svc, tool, runtime, materials = receipt_fixture(tmp_path)
    model = Model()
    expression = ExpressionService(ExpressionArchive(svc.task_store), ExpressionProfile(model),
        material_resolver=lambda result, **kw: recover_presentation(materials, result, **kw),
        material_validator=lambda result, projection, **kw: validate_presentation(materials, result, projection, **kw))
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        wait_client(svc, handle.task_id)
        source = svc.bind(SCOPE).result(handle.task_id)
        record = expression.prepare(source, scope=SCOPE, check_current=lambda: "")
        task = svc.task_store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        materials.revoke(task["material_package"]["items"][0]["reference"], task=task)
        assert expression.prepare(source, scope=SCOPE, check_current=lambda: "") == record
        assert expression.reserve_publication(record, scope=SCOPE, check_current=lambda: "") is False
        assert len(model.calls) == 1 and tool.calls == 1 and svc.bind(SCOPE).result(handle.task_id) == source
    finally:
        close(svc)


def test_i06_foreign_quoted_media_and_files_never_enter_selected_materials(tmp_path):
    from shinku.qq.task_materials import sources_for_turn
    source = media_turn("按引用图片生成报告。", quote=[{"type": "image", "data": {"url": PIXEL}},
        {"type": "file", "data": {"path": str(tmp_path / "foreign.txt")}}])
    message = source.messages[0]
    quoted = {**message.quoted_message, "conversation_id": "foreign", "conversation_type": "private"}
    message = replace(message, quoted_message=quoted)
    selected, gaps = sources_for_turn(replace(source, messages=(message,)),
        {"requester": SCOPE.requester, "session": SCOPE.session}, {})
    assert not any(row.provenance.get("message_id") == "original-plan" for row in selected)
    assert not any(row.kind == "attachment" for row in selected)
    assert any(row["reason"] == "material_quote_scope_conflict" for row in gaps)


def test_i02_missing_second_image_is_not_hidden_by_first_valid_image(tmp_path):
    from shinku.qq.task_materials import sources_for_turn
    request = "用这两张图片生成报告。"
    source = media_turn(request)
    message = replace(source.messages[0], segments=(*source.messages[0].segments,
        IncomingMessage.from_event({"user_id": "fixture", "message": [{"type": "image", "data": {"file": "unavailable"}}]}).segments[0]))
    selected, gaps = sources_for_turn(replace(source, messages=(message,)),
        {"requester": SCOPE.requester, "session": SCOPE.session}, {})
    assert sum(row.kind == "image" for row in selected) == 1
    assert any(row["reason"] == "material_image_unavailable" for row in gaps)


def test_i10_explicit_store_does_not_turn_ordinary_chat_into_a_task(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    bridge, chat, expression, _ = bridge_for(svc, tool)
    def chat_reply(**kw):
        chat.calls.append(kw)
        return {"kind": "final", "text": "你好。", "task_outcome": "completed"}
    chat.call_chat_json = chat_reply
    try:
        bridge(turn("你好。"))
        assert svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session) == []
        assert tool.calls == 0 and runtime.calls == [] and expression.calls == []
    finally:
        close(svc)


def test_i09_interrupted_bound_presentation_recovery_keeps_complete_tail_and_never_calls_model(tmp_path):
    from shinku.agent_runtime.contracts import canonical_hash
    from shinku.agent_runtime.material_presentation import recover_presentation
    from shinku.result_expression import ExpressionArchive, ExpressionService
    from shinku.result_expression.facts import identity
    svc, tool, runtime, materials = receipt_fixture(tmp_path)
    model = Model()
    archive = ExpressionArchive(svc.task_store)
    expression = ExpressionService(archive, ExpressionProfile(model),
        material_resolver=lambda result, **kw: recover_presentation(materials, result, **kw))
    try:
        handle = svc.bind(SCOPE).submit(demo_request(tool))
        wait_client(svc, handle.task_id)
        source = svc.bind(SCOPE).result(handle.task_id)
        key = canonical_hash(identity(source))
        projection = recover_presentation(materials, source, scope=SCOPE, still_authorized=lambda: True)
        binding = canonical_hash({**identity(source), "result_hash": source["result_hash"],
                                  "presentation_hash": canonical_hash(projection)})
        archive.claim(key, scope=SCOPE, result=source, config_hash=expression.profile.config_hash,
                      binding_hash=binding, processing_seconds=1)
        archive.bind_presentation(key, scope=SCOPE, projection=projection, binding_hash=binding)
        archive.update(key, scope=SCOPE, change=lambda r: r.update(processing_until=0))
        recovered = expression.prepare(source, scope=SCOPE, check_current=lambda: "", recover_interrupted=True)
        assert recovered["status"] == "fallback" and recovered["fallback_reason"] == "expression_interrupted"
        assert "条件0：仅离线，不部署。" in recovered["text"]
        assert expression.prepare(source, scope=SCOPE, check_current=lambda: "") == recovered
        assert model.calls == [] and tool.calls == 1 and len(runtime.calls) == 2
        # An archive projection cannot be replaced with different fact content
        # merely by recalculating the outer record digest.
        def corrupt(record):
            record["presentation_projection"]["facts"][0]["value"] = 13
        archive.update(key, scope=SCOPE, change=corrupt)
        assert expression.prepare(source, scope=SCOPE, check_current=lambda: "")["fallback_reason"] == "expression_archive_unavailable"
    finally:
        close(svc)


@pytest.mark.parametrize("enabled", [False, True])
def test_i10_real_assembly_materials_are_explicit_and_share_original_task_store(tmp_path, monkeypatch, enabled):
    from shinku.qq.assembly import assemble_qq_agent, QQAgentConfig
    monkeypatch.setenv("SHINKU_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("SHINKU_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SHINKU_CONFIG_ROOT", str(tmp_path / "config"))
    monkeypatch.setenv("SHINKU_LOG_ROOT", str(tmp_path / "logs"))
    persona = tmp_path / "persona.md"
    persona.write_text("Fixture public persona.", encoding="utf-8")
    config = QQAgentConfig(enabled=True, send_enabled=False, memory_enabled=False,
        persona_file=str(persona), chat_api_key="fixture-key", chat_base_url="https://example.invalid/v1",
        chat_model_name="fixture", agent_state_store_path=str(tmp_path / "tasks.sqlite"),
        trace_store_path=str(tmp_path / "trace.json"), conversation_state_store_path=str(tmp_path / "conversation.sqlite"))
    from types import SimpleNamespace
    assembly = assemble_qq_agent(config, runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "fixture"}), material_policy=MaterialPolicy() if enabled else None,
                                 material_roots=(tmp_path,))
    try:
        bridge = assembly.bridge
        assert (bridge.material_store is not None) == enabled
        if enabled:
            assert bridge.material_store.store is bridge.task_store
        assert "task_material_read" not in bridge.tool_host.registry.names()
        with bridge.task_store._transaction() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert ("agent_task_materials" in tables) == enabled
        assert bridge.sender is None
    finally:
        assembly.task_coordinator.shutdown(wait=True)
        assembly.bridge.task_store.close()
        assembly.bridge.conversation_runtime.close()


def test_i01_i04_oversized_original_is_rejected_before_admission_or_effect(tmp_path):
    svc, tool, runtime = demo_service(tmp_path)
    bridge, chat, expression, _ = bridge_for(svc, tool)
    request = demo_request(tool).original_request + "x" * 12500
    try:
        reply = bridge(turn(request))
        assert reply.agent_status == "failed" and tool.calls == 0 and runtime.calls == []
        assert svc.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session) == []
    finally:
        close(svc)
