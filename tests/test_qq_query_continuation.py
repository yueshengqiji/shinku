"""P4e real host/worker/receipt handoff with isolated source bytes and models."""
from reply_review_fixture import dialogue_review_result
import copy
from contextvars import copy_context
from dataclasses import replace
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.qq.delivery import DeliveryResult
from shinku.qq.interaction_tools import ChatReadCapability
from shinku.qq.query_continuation import ForegroundQueryPolicy
from shinku.qq.reply_host import ReplyHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from test_qq_tool_decision_boundary import boundary_bridge
from test_qq_independent_agent import turn
from shinku.agent_runtime.demo import demo_service


URL_A, URL_B = "https://query.example/a", "https://query.example/b"
GOAL = "比较甲店和乙店的价格，给出两个店的来源。"
BODY_A, BODY_B = "甲店价格10元。", "乙店价格20元。"
PARTIAL = f"{BODY_A}来源：{URL_A}；乙店价格尚未核实。"
ANSWER = f"{BODY_A}来源：{URL_A}；{BODY_B}来源：{URL_B}。"


def invocation(name, **arguments):
    return {"kind": "tool", "tool_call": {"name": name, "arguments": arguments}}


def schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


class SourceRead:
    agent_fast_public_query = True
    agent_permission_class = "read"
    agent_progress_class = "observe"
    agent_completion_effects = ()
    agent_repeatable_observation = True

    def __init__(self, name, paths):
        self.tool_type, self.paths, self.calls = name, paths, []
        self.hook = None
        self.failure = None
        if name == "query_search":
            self.agent_research_candidate_field = "results"
            self.parameters = schema({"query": {"type": "string", "minLength": 1, "maxLength": 200}}, ["query"])
        else:
            self.agent_research_content_field = "page_text"
            self.parameters = schema({"url": {"type": "string", "maxLength": 500},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000}}, ["url"])

    def normalize_call(self, value):
        return dict(value)

    def tool_metadata(self):
        return SimpleNamespace(description="Read the isolated original query's sources.", input_schema=self.parameters)

    def build_prompt_instruction(self):
        return "Read the isolated original query's sources."

    def agent_query_source_version(self, *, arguments, context):
        return hashlib.sha256(self.paths[arguments["url"]].read_bytes()).hexdigest()

    def execute(self, *, call, context):
        self.calls.append(copy.deepcopy(call))
        if self.hook:
            self.hook(call, context)
        if self.failure:
            return copy.deepcopy(self.failure)
        if self.tool_type == "query_search":
            return {"tool_type": self.tool_type, "followup_context": "两个尚未读取的来源。", "state_updates": {
                "results": [{"url": url, "title": body} for url, body in ((URL_A, BODY_A), (URL_B, BODY_B))]}}
        path, offset = self.paths[call["url"]], call.get("offset", 0)
        body = path.read_text(encoding="utf-8")
        return {"tool_type": self.tool_type, "followup_context": body[offset:], "state_updates": {
            "url": call["url"], "site_access_ok": True, "http_status": 200, "page_text": body[offset:],
            "content_offset": offset, "content_next_offset": None, "content_truncated": False,
            "content_sha256": hashlib.sha256(body.encode()).hexdigest()}}


def answer_review(payload, complete):
    claims = [{"source_id": source["id"], "answer_fragment": source["text"].split("。", 1)[0] + "。",
        "quote": source["text"].split("。", 1)[0] + "。",
        "date_kind": "unknown", "date_text": ""} for source in payload["read_sources"]]
    return {"verdict": "supported" if complete else "partial", "claims": claims,
        "coverage": [{"field": "原问题", "supported": complete, "claim_indices": list(range(len(claims)))}],
        "missing_fields": [] if complete else ["原问题"],
        "limitations": [] if complete else [{"answer_fragment": "乙店价格尚未核实", "fields": ["原问题"]}]}


class QueryChat:
    def __init__(self, replies=None):
        self.calls = []
        self.review_complete = False
        self.replies = list(replies or [invocation("query_search", query=GOAL),
            invocation("query_read", url=URL_A), {"kind": "final", "text": PARTIAL, "task_outcome": "partial"}])

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        blocks = str(kwargs.get("system_extra_blocks"))
        if "答案证据与覆盖复核" in blocks:
            return answer_review(json.loads(kwargs["user_prompt"]), self.review_complete)
        if "候选相关性复核" in blocks:
            payload = json.loads(kwargs["user_prompt"])
            return {"candidate_reviews": [{"candidate_id": row["id"], "relevance": "relevant",
                "reason": "原问题的实际店铺", "recommended_read": True} for row in payload["candidates"]]}
        if "仅纠正" in blocks:
            return {}
        if not self.replies:
            raise AssertionError("unexpected chat model request")
        value = self.replies.pop(0)
        return value(kwargs) if callable(value) else value


class QueryExecution:
    def __init__(self):
        self.calls = []
        self.plans = [invocation("query_read", url=URL_B), {"kind": "final", "text": ANSWER, "task_outcome": "completed"}]
        self.review_transform = None
        self.metadata_transform = None
        self.before_plan = None

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        blocks = str(kwargs.get("system_extra_blocks"))
        if "Foreground query checkpoint review" in blocks:
            payload = json.loads(kwargs["user_prompt"])
            source = payload["read_sources"][0]
            candidate = next(row for row in payload["candidates"] if row["url"] == URL_B)
            value = {"status": "continue", "progress": [{"receipt_id": source["id"], "quote": source["text"][:80],
                "reason": "甲店价格已经实际读到，乙店仍缺正文"}], "remaining": ["乙店价格及来源"], "next_ids": [candidate["id"]]}
            return self.review_transform(value, payload) if self.review_transform else value
        if "任务契约与语义承接" in blocks:
            value = {"task_link": {"mode": "new"}, "task_contract": {"goal": GOAL,
                "deliverables": ["两店价格与来源"], "scope": ["原公开来源，只读"], "requirements": [{
                    "id": "prices", "description": GOAL, "effect": "research_answer", "targets": [],
                    "checks": {"required_fields": ["原问题"], "min_sources": 2}}]}}
            return self.metadata_transform(value) if self.metadata_transform else value
        if "候选相关性复核" in blocks:
            payload = json.loads(kwargs["user_prompt"])
            return {"candidate_reviews": [{"candidate_id": row["id"], "relevance": "relevant",
                "reason": "原问题的实际店铺", "recommended_read": True} for row in payload["candidates"]]}
        if "答案证据与覆盖复核" in blocks:
            return answer_review(json.loads(kwargs["user_prompt"]), True)
        if self.before_plan:
            self.before_plan(kwargs)
        if not self.plans:
            raise AssertionError("unexpected execution request")
        return self.plans.pop(0)


@pytest.fixture
def query_host(tmp_path):
    svc, mutation, _ = demo_service(tmp_path)
    paths = {URL_A: tmp_path / "a.txt", URL_B: tmp_path / "b.txt"}
    paths[URL_A].write_text(BODY_A, encoding="utf-8")
    paths[URL_B].write_text(BODY_B, encoding="utf-8")
    search, reader = SourceRead("query_search", paths), SourceRead("query_read", paths)
    for handler in (search, reader):
        svc.executor.tool_host.registry.register(handler)
    def source_guard(arguments, context):
        return any(arguments["url"] == row.get("url") for receipt in context.get("_chat_read_receipts", [])
            for row in receipt["result"].get("state_updates", {}).get("results", []))
    caps = {handler.tool_type: ChatReadCapability(handler.tool_type, "Read the original public query.", handler.parameters,
        route="public_query", max_calls=3, argument_guard=source_guard if handler is reader else None)
        for handler in (search, reader)}
    chat, execution, sent = QueryChat(), QueryExecution(), []
    profile = replace(svc.executor.profile, runtime=execution, max_rounds=6)
    def sender(message):
        sent.append(message)
        return DeliveryResult(True, "sent", (f"query-fixture-{len(sent)}",))
    bridge = boundary_bridge(svc, mutation, chat, profile=profile,
        foreground_query_policy=ForegroundQueryPolicy(enabled=True, max_tools=2, runtime_meters_budget=False),
        material_store=TaskMaterialStore(svc.task_store, approved_roots=(tmp_path,)),
        chat_read_capabilities=caps, sender=sender, reply_host=ReplyHost(enabled=False))
    bridge.allowed_tool_names |= set(caps)
    yield SimpleNamespace(svc=svc, bridge=bridge, search=search, reader=reader, mutation=mutation,
        chat=chat, execution=execution, paths=paths, sent=sent)
    svc.coordinator.shutdown(wait=True)
    svc.task_store.close()


def drain(host, reply):
    until = time.monotonic() + 5
    while reply.background_task_id and time.monotonic() < until:
        record = host.svc.coordinator.get(reply.background_task_id)
        if not record["worker_active"] and record["status"] not in {"queued", "running"}:
            return record
        time.sleep(.005)
    raise AssertionError("isolated query worker did not settle")


def saved_task(host):
    rows = host.svc.task_store.candidates(requester="fixture", session="private:fixture")
    assert len(rows) == 1
    return rows[0]


def perception_read(host, *, max_calls=2, hook=None):
    from shinku.agent.self_state import SelfStateTool
    goal = "确认当前运行状态和 Agent 状态，两项都需要实际读取。"
    calls = []
    read = SelfStateTool(fresh_provider=lambda: {"runtime": {"ready": True}, "agent": {"enabled": True}})
    actual = read.execute
    def execute(*, call, context):
        calls.append(dict(call))
        result = actual(call=call, context=context)
        if hook:
            hook(context)
        return result
    read.execute = execute
    host.svc.executor.tool_host.registry.register(read)
    cap = ChatReadCapability(read.tool_type, "Read the two originally requested state sections.", schema({
        "scope": {"type": "string", "enum": ["runtime", "agent"], "maxLength": 16}}, ["scope"]),
        max_calls=max_calls)
    host.bridge.chat_read_capabilities = {cap.name: cap}
    host.bridge.runtime.capabilities = {cap.name: cap}
    host.bridge.allowed_tool_names.add(cap.name)
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, max_tools=1)
    host.chat.replies = [invocation(read.tool_type, scope="runtime"),
        {"kind": "final", "text": "运行已就绪；Agent 状态尚未读取。", "task_outcome": "partial"}]
    actual_model = host.execution.call_chat_json
    def model(**kwargs):
        blocks = str(kwargs.get("system_extra_blocks"))
        if "Foreground query checkpoint review" in blocks:
            host.execution.calls.append(kwargs)
            row = json.loads(kwargs["user_prompt"])["observations"][0]
            return {"status": "continue", "progress": [{"receipt_id": row["receipt_id"],
                "quote": '"ready": true', "reason": "原问题的运行状态已有实际回执"}],
                "remaining": ["Agent 状态"], "next_ids": ["capability:" + cap.name]}
        if "任务契约与语义承接" in blocks:
            host.execution.calls.append(kwargs)
            return {"task_link": {"mode": "new"}, "task_contract": {"goal": goal,
                "deliverables": ["运行和 Agent 状态"], "scope": ["原实例的只读状态"], "requirements": [{
                    "id": "state", "description": goal, "effect": "tool:" + cap.name,
                    "targets": [], "checks": {}}]}}
        return actual_model(**kwargs)
    host.execution.call_chat_json = model
    host.execution.plans = [invocation(read.tool_type, scope="agent"),
        {"kind": "final", "text": "运行就绪且 Agent 已启用，两项状态均已读取。", "task_outcome": "completed"}]
    return goal, calls


@pytest.mark.parametrize("max_calls", [1, 2])
def test_fh11_perception_followup_retains_original_read_quota_and_receipt(query_host, max_calls):
    host = query_host
    goal, calls = perception_read(host, max_calls=max_calls)
    reply = host.bridge(turn(goal))
    if max_calls == 1:
        assert not reply.background_task_id and not host.svc.coordinator.list()
        assert [row["scope"] for row in calls] == ["runtime"] and len(host.execution.calls) == 1
        return
    assert reply.reason == "agent_task_queued", (reply, host.execution.calls)
    record, task = drain(host, reply), saved_task(host)
    assert record["status"] == task["status"] == "completed", (record, task.get("completion"))
    assert [row["scope"] for row in calls] == ["runtime", "agent"]
    assert task["budget"]["tools"] == 2 and task["budget"]["requests"] == 6
    assert task["budget"]["requests"] == len(host.chat.calls) + len(host.execution.calls)
    assert goal in host.execution.calls[-2]["user_prompt"] and '"ready"' in host.execution.calls[-2]["user_prompt"]
    assert task["tool_calls"][0]["id"] == task["foreground_query"]["receipt_ids"][0]
    assert len(host.sent) == 2 and host.mutation.calls == 0


@pytest.mark.parametrize("stop", ["budget", "authority", "injection"])
def test_fh04_fh05_perception_stops_before_followup_or_new_model(query_host, stop):
    from shinku.agent.budget import current_task_budget
    host = query_host
    def hook(context):
        if stop == "budget":
            current_task_budget().use_tool()
            current_task_budget().use_tool()
        elif stop == "authority":
            host.bridge.command_state.set_permission_mode("readonly")
    goal, calls = perception_read(host, hook=hook)
    if stop == "budget":
        host.bridge.task_budget_limits["max_tools"] = 2
    if stop == "injection":
        host.chat.replies[-1]["budget"] = {"max_tools": 1000}
    reply = host.bridge(turn(goal))
    assert not reply.background_task_id and not host.svc.coordinator.list()
    assert [row["scope"] for row in calls] == ["runtime"] and not host.execution.calls and host.mutation.calls == 0
    assert len(host.chat.calls) == (2 if stop == "injection" else 1)


def test_fh01_query_finishes_only_missing_source_with_original_receipts_and_budget(query_host):
    host = query_host
    reply = host.bridge(turn(GOAL))
    assert reply.reason == "agent_task_queued", reply
    record = drain(host, reply)
    task = saved_task(host)
    assert record["status"] == task["status"] == "completed", (record, task["completion"])
    assert task["completion"]["accepted"] and task["execution_result"]["verification"]["accepted"]
    assert [row.get("url") for row in host.reader.calls] == [URL_A, URL_B]
    assert len(host.search.calls) == 1 and host.mutation.calls == 0
    assert [row["id"] for row in task["tool_calls"][:2]] == task["foreground_query"]["receipt_ids"]
    assert task["budget"]["tools"] == 3
    assert task["budget"]["requests"] == len(host.chat.calls) + len(host.execution.calls)
    assert task["budget"]["max_requests"] == 16 and task["budget"]["max_tools"] == 8
    assert len(host.sent) == 2 and record["metadata"]["foreground_delivery"]["sent"]
    plans = [row for row in host.execution.calls if row.get("native_tools")]
    assert plans and host.mutation.tool_type not in json.dumps(plans)
    assert "PERSONA_CHAT_SENTINEL" not in json.dumps(host.execution.calls)


def test_fh11_disabled_preserves_honest_original_partial_without_task(query_host):
    host = query_host
    host.bridge.foreground_query_policy = ForegroundQueryPolicy()
    reply = host.bridge(turn(GOAL))
    assert reply.agent_status == "partial" and not reply.background_task_id
    assert not host.svc.coordinator.list() and host.execution.calls == [] and host.mutation.calls == 0


@pytest.mark.parametrize("complete", [False, True])
def test_fh02_answer_within_threshold_never_creates_background_task(query_host, complete):
    host = query_host
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, max_tools=3)
    host.chat.review_complete = complete
    if complete:
        host.chat.replies[-1] = {"kind": "final", "text": f"{BODY_A}来源：{URL_A}", "task_outcome": "completed"}
    reply = host.bridge(turn("甲店价格是多少？" if complete else GOAL))
    assert reply.agent_status == ("completed" if complete else "partial"), reply
    assert not host.svc.coordinator.list() and not host.execution.calls


def test_fh02_search_cards_and_http_success_without_source_progress_never_queue(query_host):
    host = query_host
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, max_tools=1)
    host.chat.replies = [invocation("query_search", query=GOAL), {"kind": "final", "text": "只取得卡片，还不能比较。", "task_outcome": "partial"}]
    reply = host.bridge(turn(GOAL))
    assert reply.agent_status != "completed" and len(host.search.calls) == 1
    assert not host.reader.calls and not host.execution.calls and not host.svc.coordinator.list()


@pytest.mark.parametrize("code", ["source_http_timeout", "source_challenge_blocked", "source_http_transport_error",
    "search_provider_blocked", "source_access_unverified"])
def test_fh03_failed_or_blocked_source_is_not_a_background_retry(query_host, code):
    host = query_host
    host.reader.failure = {"tool_type": "query_read", "status": "error", "ok": False,
        "code": code, "followup_context": "原来源读取失败。", "state_updates": {"site_access_ok": False}}
    reply = host.bridge(turn(GOAL))
    assert reply.agent_status != "completed" and not reply.background_task_id
    assert not host.execution.calls and not host.svc.coordinator.list() and host.mutation.calls == 0


@pytest.mark.parametrize("change", ["stop", "permission", "cancel"])
def test_fh04_fh05_authority_change_or_cancel_at_return_does_not_handoff(query_host, change):
    host = query_host
    def revoke(call, context):
        if change == "permission":
            host.bridge.command_state.set_permission_mode("readonly")
        elif change == "stop":
            host.bridge.command_state.set_stopped(True)
        else:
            context["agent_task_interrupt"] = lambda: "task_canceled"
    host.reader.hook = revoke
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list() and not host.execution.calls


@pytest.mark.parametrize("key,value", [("max_requests", 4), ("max_tools", 2), ("max_tokens", 100)])
def test_fh05_total_limits_prevent_new_execution_without_reallocation(query_host, key, value):
    host = query_host
    host.bridge.task_budget_limits[key] = value
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list() and not host.execution.calls
    assert host.mutation.calls == 0


@pytest.mark.parametrize("status", ["stop", "waiting_user"])
def test_fh03_fh04_semantic_no_progress_or_missing_input_stays_foreground(query_host, status):
    host = query_host
    host.execution.review_transform = lambda value, _payload: {**value, "status": status}
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list()
    assert len(host.execution.calls) == 1 and not host.mutation.calls


@pytest.mark.parametrize("injection", ["quote", "source", "next", "grant", "calls", "budget", "success"])
def test_fh04_fh06_unbound_progress_or_host_field_injection_cannot_submit(query_host, injection):
    host = query_host
    def inject(value, payload):
        if injection == "quote":
            value["progress"][0]["quote"] = "正文不存在的任意断言"
        elif injection == "source":
            value["progress"][0]["receipt_id"] = "fake-source"
        elif injection == "next":
            value["next_ids"] = ["capability:" + host.mutation.tool_type]
        else:
            value[injection] = {"approved": True}
        return value
    host.execution.review_transform = inject
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list()
    assert len(host.execution.calls) == 1 and host.mutation.calls == 0


def test_fh04_prepared_contract_cannot_upgrade_query_into_write(query_host):
    host = query_host
    def mutate(value):
        value["task_contract"]["requirements"] = [{"id": "write", "effect": "write", "description": "错误新增写入",
            "targets": [str(host.mutation.path)], "checks": {"contains": "not authorized"}}]
        return value
    host.execution.metadata_transform = mutate
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list() and host.mutation.calls == 0


def test_fh07_full_queue_has_no_background_claim_or_admitted_task(query_host):
    host = query_host
    host.svc.coordinator.max_pending_tasks = 1
    host.svc.coordinator.submit(lambda cancel: None, task_id="occupied", defer_start=True)
    reply = host.bridge(turn(GOAL))
    assert reply.reason == "background_task_admission_failed" and not reply.background_task_id
    assert len(host.svc.coordinator.list()) == 1 and not host.svc.task_store.candidates(requester="fixture", session="private:fixture")
    assert host.mutation.calls == 0


def test_fh07_committed_queue_acknowledgement_loss_reconciles_same_job(query_host):
    host = query_host
    actual, submits = host.svc.coordinator.submit, []
    def ack_loss(*args, **kwargs):
        submits.append(kwargs)
        actual(*args, **kwargs)
        raise TimeoutError("isolated admission acknowledgement lost")
    host.svc.coordinator.submit = ack_loss
    reply = host.bridge(turn(GOAL))
    assert reply.reason == "agent_task_queued", reply
    record = drain(host, reply)
    assert record["status"] == "completed" and len(submits) == 1 and len(host.svc.coordinator.list()) == 1
    assert [row.get("url") for row in host.reader.calls] == [URL_A, URL_B]


def test_fh07_existing_operation_after_completion_does_not_repeat_reads_or_send(query_host):
    from shinku.qq.query_continuation import existing_query_operation
    host = query_host
    reply = host.bridge(turn(GOAL))
    drain(host, reply)
    before = len(host.reader.calls), len(host.search.calls), len(host.execution.calls), len(host.sent)
    repeated = existing_query_operation(host.bridge, turn(GOAL), {})
    assert repeated.reason == "query_operation_already_linked" and repeated.delivery_status == "suppressed"
    assert before == (len(host.reader.calls), len(host.search.calls), len(host.execution.calls), len(host.sent))


@pytest.mark.parametrize("changed", [False, True])
def test_fh08_prior_source_reread_requires_independent_version_change(query_host, changed):
    host = query_host
    host.execution.plans.insert(0, invocation("query_read", url=URL_A))
    if changed:
        def change_before_plan(kwargs):
            host.paths[URL_A].write_text("甲店价格11元。", encoding="utf-8")
            host.execution.before_plan = None
        host.execution.before_plan = change_before_plan
        host.execution.plans[-1]["text"] = ANSWER.replace("10元", "11元")
    reply = host.bridge(turn(GOAL))
    record = drain(host, reply)
    task = saved_task(host)
    assert record["status"] == "completed", (record, task["completion"])
    assert [row.get("url") for row in host.reader.calls] == ([URL_A, URL_A, URL_B] if changed else [URL_A, URL_B])
    if changed:
        assert task["progress"]["query_source_refreshes"][0]["reason"] == "adapter_observed_source_version_change"
    else:
        assert any(row.get("data", {}).get("code") == "duplicate_tool_call" for row in task["tool_calls"])


@pytest.mark.parametrize("exhausted", [False, True])
def test_fh01_fh05_clock_threshold_uses_safe_return_and_original_deadline(query_host, exhausted):
    host, clock = query_host, [0.0]
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy,
        max_tools=3, max_seconds=1, clock=lambda: clock[0])
    host.reader.hook = lambda call, context: clock.__setitem__(0, 31.0 if exhausted else 1.1)
    reply = host.bridge(turn(GOAL))
    if exhausted:
        assert not reply.background_task_id and not host.execution.calls and not host.svc.coordinator.list()
    else:
        assert reply.reason == "agent_task_queued", reply
        assert drain(host, reply)["status"] == "completed"
        assert len(host.search.calls) == 1 and len(host.reader.calls) == 2


def test_fh05_deadline_elapsed_while_queued_prevents_first_planning_or_read(query_host):
    host, clock = query_host, [0.0]
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, clock=lambda: clock[0])
    actual = host.svc.coordinator.dispatch
    def dispatch(task_id, **kwargs):
        clock[0] = 31.0
        return actual(task_id, **kwargs)
    host.svc.coordinator.dispatch = dispatch
    reply = host.bridge(turn(GOAL))
    record = drain(host, reply)
    assert record["status"] == "blocked" and record["error"] == "task_time_budget_exhausted", record
    assert len(host.execution.calls) == 2 and [row.get("url") for row in host.reader.calls] == [URL_A]
    assert host.mutation.calls == 0


@pytest.mark.parametrize("total_tokens", [800000, 1600000])
def test_fh06_full_receipt_tail_is_read_from_original_material_reference(query_host, total_tokens):
    host = query_host
    host.bridge.task_budget_limits["max_tokens"] = total_tokens
    host.paths[URL_A].write_text(BODY_A + " retained source body" * 360 + " TAIL_CONSTRAINT_6721", encoding="utf-8")
    actual = host.execution.call_chat_json
    def complete_materials(**kwargs):
        if kwargs.get("native_tools"):
            raw = kwargs["user_prompt"].split("\n", 1)[1].split("\n[Task materials", 1)[0]
            payload = json.loads(raw)
            item = next((row for row in payload["task_materials"]["items"]
                if row["source_id"].startswith("query-receipt:") and row.get("omitted_chars")), None)
            if item:
                task = saved_task(host)
                receipts = host.bridge.material_store.read_receipts(task)
                offset = max((row["offset"] + row["returned_chars"] for row in receipts
                    if row["reference"] == item["reference"]), default=0)
                if offset < item["content_chars"]:
                    host.execution.calls.append(kwargs)
                    return invocation("task_material_read", reference=item["reference"], offset=offset, length=4096)
        return actual(**kwargs)
    host.execution.call_chat_json = complete_materials
    reply = host.bridge(turn(GOAL))
    record = drain(host, reply)
    task = saved_task(host)
    if total_tokens == 800000:
        assert record["status"] == "blocked" and record["error"] == "task_token_budget_exhausted"
        assert task["budget"]["max_tokens"] == total_tokens and task.get("completion", {}).get("status") != "completed"
        return
    assert record["status"] == "completed", (record["error"], task.get("stop_reason"),
        [(row["name"], row["status"]) for row in task["tool_calls"]], task.get("budget"), task.get("completion"))
    assert "TAIL_CONSTRAINT_6721" in json.dumps(host.execution.calls, ensure_ascii=False)
    assert task["execution_result"]["information"]["required_status"] == "complete"
    assert any(row["name"] == "task_material_read" for row in task["tool_calls"])
    assert [row.get("url") for row in host.reader.calls] == [URL_A, URL_B]


def test_fh07_unknown_queue_confirmation_keeps_one_key_until_inspected(query_host):
    from shinku.qq.query_continuation import existing_query_operation
    host = query_host
    actual_submit, actual_get = host.svc.coordinator.submit, host.svc.coordinator.get
    lost = [False]
    def submit(*args, **kwargs):
        actual_submit(*args, **kwargs)
        lost[0] = True
        raise TimeoutError("isolated submit acknowledgement loss")
    def get(*args, **kwargs):
        if lost[0]:
            raise TimeoutError("isolated record read unavailable")
        return actual_get(*args, **kwargs)
    host.svc.coordinator.submit, host.svc.coordinator.get = submit, get
    reply = host.bridge(turn(GOAL))
    assert reply.reason == "query_handoff_confirmation_unknown" and reply.agent_status == "blocked", reply
    assert not any("后台继续" in part.text for msg in host.sent for part in msg.segments)
    host.svc.coordinator.get = actual_get
    before = len(host.reader.calls), len(host.search.calls), len(host.execution.calls)
    known = existing_query_operation(host.bridge, turn(GOAL), {})
    assert known.background_task_id == reply.background_task_id and known.reason == "query_operation_already_linked"
    assert before == (len(host.reader.calls), len(host.search.calls), len(host.execution.calls))
    assert len(host.svc.coordinator.list()) == 1
    host.svc.coordinator.dispatch(reply.background_task_id)
    record = drain(host, reply)
    assert record["status"] == "completed", record


def test_fh08_timed_out_inflight_read_and_its_late_result_never_handoff(query_host):
    host, release, finished = query_host, threading.Event(), threading.Event()
    old = host.bridge.chat_read_capabilities["query_read"]
    host.bridge.chat_read_capabilities["query_read"] = replace(old, max_seconds=.02)
    host.bridge.runtime.capabilities = host.bridge.chat_read_capabilities
    def delayed(call, context):
        release.wait(2)
        finished.set()
    host.reader.hook = delayed
    try:
        reply = host.bridge(turn(GOAL))
        assert not reply.background_task_id and not host.execution.calls and not host.svc.coordinator.list()
        release.set()
        assert finished.wait(1)
        assert not host.svc.coordinator.list() and host.mutation.calls == 0
    finally:
        release.set()


@pytest.mark.parametrize("status", ["sent", "failed", "uncertain", "dry_run"])
def test_fh09_stage_and_final_delivery_are_separate_without_repeat_on_old_operation(query_host, status):
    from shinku.qq.query_continuation import existing_query_operation
    from shinku.result_expression import ExpressionProfile, ExpressionService, ExpressionArchive
    from shinku.result_expression.facts import identity
    from shinku.agent_runtime.contracts import canonical_hash
    from test_result_expression import Model
    host, model = query_host, Model()
    host.bridge.result_expression = ExpressionService(ExpressionArchive(host.svc.task_store),
        ExpressionProfile(model, configuration_id="query-fixture"))
    def sender(message):
        host.sent.append(message)
        if status == "uncertain":
            raise TimeoutError("isolated delivery unknown")
        return DeliveryResult(status == "sent", status, ("query-one",) if status == "sent" else (),
            definitive_not_sent=status == "failed")
    host.bridge.sender = sender if status != "dry_run" else None
    reply = host.bridge(turn(GOAL))
    record = drain(host, reply)
    task = saved_task(host)
    assert record["status"] == task["status"] == "completed", (record, task["completion"])
    archive = host.bridge.result_expression.archive.get(canonical_hash(identity(task["execution_result"])),
        scope=SimpleNamespace(requester="fixture", session="private:fixture"))
    assert archive["delivery"]["status"] == status and len(model.calls) == 1
    assert record["metadata"]["foreground_delivery"]["status"] == ("transport_error" if status == "uncertain" else status)
    if status == "uncertain":
        assert not record["metadata"]["foreground_delivery"]["receipt"]["definitive_not_sent"]
    before = len(host.execution.calls), len(host.sent), len(model.calls)
    existing_query_operation(host.bridge, turn(GOAL), {})
    assert before == (len(host.execution.calls), len(host.sent), len(model.calls))
    from shinku.qq.result_expression import replay_task_expression
    from shinku.qq.task_status import reader_for_bridge
    from shinku.agent_runtime.contracts import AgentScope
    scope = AgentScope("fixture", "private:fixture")
    view = reader_for_bridge(host.bridge).get(scope, task["task_id"])
    event = host.bridge.result_expression.archive.events.get(view["result_event"]["event_id"], scope=scope)
    assert event["source_hash"] == task["execution_result"]["result_hash"]
    assert event["version"] == task["revision"]
    replay_task_expression(host.bridge, turn(GOAL), task_id=task["task_id"], context={})
    assert before == (len(host.execution.calls), len(host.sent), len(model.calls))


def test_fh07_persisted_operation_after_restart_never_starts_another_worker(query_host, tmp_path):
    host = query_host
    host.svc.coordinator.shutdown(wait=True)
    original = AgentTaskCoordinator(state_path=tmp_path / "workers.sqlite3", max_workers=1)
    host.svc.coordinator = host.bridge.task_coordinator = original
    host.bridge.execution_adapter.coordinator = original
    reply = host.bridge(turn(GOAL))
    record = drain(host, reply)
    assert record["status"] == "completed", record
    original.shutdown(wait=True)
    fresh = AgentTaskCoordinator(state_path=tmp_path / "workers.sqlite3", max_workers=1)
    host.svc.coordinator = host.bridge.task_coordinator = fresh
    before = len(host.search.calls), len(host.reader.calls), len(host.chat.calls), len(host.execution.calls), len(host.sent)
    repeated = host.bridge(turn(GOAL))
    assert repeated.reason == "query_operation_already_linked" and repeated.background_task_id == reply.background_task_id
    assert len(fresh.list()) == 1 and not fresh.get(reply.background_task_id)["worker_active"]
    assert before == (len(host.search.calls), len(host.reader.calls), len(host.chat.calls), len(host.execution.calls), len(host.sent))


def test_fh07_generic_executor_cannot_resume_query_from_json_without_live_capability(query_host):
    from shinku.agent_runtime.execution import ExecutionGrant
    host = query_host
    reply = host.bridge(turn(GOAL))
    drain(host, reply)
    task = saved_task(host)
    before = len(host.execution.calls), len(host.reader.calls), len(host.search.calls), host.mutation.calls
    executor = host.bridge.execution_adapter.executor
    grant = ExecutionGrant(str(host.bridge.command_state.authority_revision()),
        frozenset({"query_search", "query_read"}), permission_mode="ultra", budget_limits=host.bridge.task_budget_limits)
    result = executor.run(task_id=task["task_id"], context={"agent_task_state": task}, grant=grant)
    assert result.status == "failed" and result.reason == "query_live_checkpoint_unavailable", result
    assert before == (len(host.execution.calls), len(host.reader.calls), len(host.search.calls), host.mutation.calls)


@pytest.mark.parametrize("missing_answer", [False, True])
def test_fh04_browser_access_contract_cannot_replace_verified_research(query_host, missing_answer):
    host = query_host
    def contract(value):
        row = {"id": "browser", "description": "Access must remain in the original source scope.",
            "effect": "browser_access", "targets": [URL_A if missing_answer else "https://unrelated.example/page"], "checks": {}}
        value["task_contract"]["requirements"] = ([row] if missing_answer else value["task_contract"]["requirements"] + [row])
        return value
    host.execution.metadata_transform = contract
    reply = host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list() and not saved_tasks(host)
    assert len(host.reader.calls) == len(host.search.calls) == 1 and host.mutation.calls == 0


def saved_tasks(host):
    return host.svc.task_store.candidates(requester="fixture", session="private:fixture")


def test_fh01_fh06_exact_original_urls_and_validated_source_tail_continue_without_search(query_host):
    host = query_host
    goal = f"比较甲店 {URL_A} 和乙店 {URL_B} 的价格，给出两个店的来源。"
    tail = "原网页的后续正文，保留原始范围。"
    host.paths[URL_A].write_text(BODY_A + tail, encoding="utf-8")
    original_read = host.reader.execute
    def paginated(*, call, context):
        result = original_read(call=call, context=context)
        updates = result["state_updates"]
        if call["url"] == URL_A and not call.get("offset"):
            result["followup_context"] = updates["page_text"] = BODY_A
            updates.update(content_next_offset=len(BODY_A), content_truncated=True)
        updates.update(verified_effects=["tool:query_read"], completion_targets=[call["url"]], result_valid=True)
        return result
    host.reader.execute = paginated
    host.reader.agent_completion_target_arguments = ("url",)
    def guard(arguments, context):
        if arguments["url"] not in {URL_A, URL_B}:
            return False
        return not arguments.get("offset") or any(
            row["arguments"].get("url") == arguments["url"]
            and row["result"]["state_updates"].get("content_next_offset") == arguments["offset"]
            for row in context.get("_chat_read_receipts", []))
    cap = replace(host.bridge.chat_read_capabilities["query_read"], argument_guard=guard)
    host.bridge.chat_read_capabilities = {"query_read": cap}
    host.bridge.runtime.capabilities = {"query_read": cap}
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, max_tools=1)
    host.chat.replies = [invocation("query_read", url=URL_A),
        {"kind": "final", "text": PARTIAL, "task_outcome": "partial"}]
    actual = host.execution.call_chat_json
    def model(**kwargs):
        if "Foreground query checkpoint review" in str(kwargs.get("system_extra_blocks")):
            host.execution.calls.append(kwargs)
            source = json.loads(kwargs["user_prompt"])["read_sources"][0]
            assert source["truncated"] and source["next_offset"] == len(BODY_A)
            return {"status": "continue", "progress": [{"receipt_id": source["id"], "quote": BODY_A,
                "reason": "甲店价格实际已读，原正文尚有尾部"}], "remaining": ["原网页尾部与乙店价格"],
                "next_ids": [source["id"]]}
        if "答案证据与覆盖复核" in str(kwargs.get("system_extra_blocks")):
            host.execution.calls.append(kwargs)
            payload = json.loads(kwargs["user_prompt"])
            payload["read_sources"] = [row for row in payload["read_sources"] if row["text"] != tail]
            return answer_review(payload, True)
        return actual(**kwargs)
    host.execution.call_chat_json = model
    def contract(value):
        value["task_contract"]["goal"] = goal
        value["task_contract"]["requirements"].append({"id": "urls", "description": "核实原明确网址的只读回执",
            "effect": "tool:query_read", "targets": [URL_A, URL_B], "checks": {"http_status": 200}})
        return value
    host.execution.metadata_transform = contract
    host.execution.plans.insert(0, invocation("query_read", url=URL_A, offset=len(BODY_A)))
    reply = host.bridge(turn(goal))
    assert reply.reason == "agent_task_queued", reply
    record, task = drain(host, reply), saved_task(host)
    assert record["status"] == "completed", (record, task.get("completion"))
    assert [row.get("url") for row in host.reader.calls] == [URL_A, URL_A, URL_B]
    assert host.reader.calls[1]["offset"] == len(BODY_A) and not host.search.calls
    assert task["completion"]["accepted"] and tail in json.dumps(host.execution.calls, ensure_ascii=False)
    assert task["requests"][0]["text"] == goal and task["budget"]["tools"] == 3


@pytest.mark.parametrize("action", [False, True])
def test_fh10_ordinary_chat_and_unstarted_action_claim_have_no_automatic_query_task(query_host, action):
    host = query_host
    question = f"创建 {host.mutation.path}。" if action else "我回来了，陪我说句话。"
    text = f"已创建 {host.mutation.path}。" if action else "我在这里。"
    host.chat.replies = [{"kind": "final", "text": text}]
    reply = host.bridge(turn(question))
    assert not reply.background_task_id and not host.svc.coordinator.list() and not saved_tasks(host)
    assert not host.search.calls and not host.reader.calls and not host.execution.calls and host.mutation.calls == 0
    if action:
        assert reply.agent_status != "completed" and text != reply.final_text
    else:
        assert reply.agent_status == "completed" and reply.final_text == text and len(host.chat.calls) == 1


def test_fh07_simultaneous_threshold_checks_share_one_preparation_worker_and_notice(query_host, monkeypatch):
    from shinku.qq import query_continuation
    host, barrier, replies, failures = query_host, threading.Barrier(2), [], []
    actual = query_continuation.try_continue_query
    def parallel(bridge, run):
        def trigger():
            try:
                barrier.wait(timeout=2)
                replies.append(actual(bridge, run))
            except BaseException as exc:
                failures.append(exc)
        contexts = [copy_context(), copy_context()]
        workers = [threading.Thread(target=lambda context=context: context.run(trigger)) for context in contexts]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=3)
        assert not failures and not any(worker.is_alive() for worker in workers)
        assert len(replies) == 2 and all(reply is not None for reply in replies)
        return next(reply for reply in replies if reply.reason == "agent_task_queued")
    monkeypatch.setattr(query_continuation, "try_continue_query", parallel)
    reply = host.bridge(turn(GOAL))
    assert drain(host, reply)["status"] == "completed"
    assert len(host.svc.coordinator.list()) == 1 and len(saved_tasks(host)) == 1 and len(host.sent) == 2
    assert len(host.search.calls) == 1 and len(host.reader.calls) == 2
    assert sum("Foreground query checkpoint review" in str(row.get("system_extra_blocks"))
        for row in host.execution.calls) == 1
