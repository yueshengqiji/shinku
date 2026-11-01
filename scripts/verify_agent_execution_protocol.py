"""P2b preparation: actual SDK/runtime/Agent path over an offline HTTP transport.

No live option, credential/config lookup, QQ adapter or production state. The
fixture transport controls responses, not tool execution or completion checks.
Passing fixtures never establishes a provider's capabilities or model quality.
"""
from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import re
import socket
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentRequest, AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import MaterialSource, TaskMaterialStore
from shinku.agent_runtime.service import AgentService
from shinku.hosts.tool_host import ToolHost
from shinku.llm.policy import RuntimePolicy
from shinku.llm.runtime import LLMRuntime, ModelBundle, ProviderToolProfile
from shinku.result_expression import ExpressionArchive
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

# Reuse the reviewed real file adapter and independent artifact/receipt assessor.
_spec = importlib.util.spec_from_file_location(
    "_agent_protocol_benchmark", Path(__file__).with_name("benchmark_agent_stages.py"))
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

MODEL = "offline-sdk-protocol-fixture"
BASE_URL = "https://execution-fixture.invalid/v1"
SCOPE = AgentScope("protocol-fixture", "local:protocol-fixture")
REASONING = "INTERNAL_PROTOCOL_REASONING_SENTINEL"
TAIL = "ATTACHMENT_TAIL_CONDITION_SENTINEL"
IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1pQAAAAASUVORK5CYII=")
DATA_URL = "data:image/png;base64," + base64.b64encode(IMAGE).decode()
TASK_LIMITS = dict(max_requests=12, max_tokens=600000, max_tools=8, max_active_seconds=30)
BATCH_LIMITS = dict(max_requests=64, max_tokens=4000000, max_tools=48, max_active_seconds=300)


def case(name, actions, *, success=True, **extra):
    return dict(id=name, expected_success=success, expected={"output.txt": "verified fixture\n"},
                actions=actions, **extra)


WRITE = {"op": "write", "path": "output.txt", "content": "verified fixture\n"}
FINAL = {"op": "final"}
CASES = (
    case("native_two_tools", [{"op": "read", "path": "spec.txt"}, WRITE, FINAL],
         initial={"spec.txt": "CONTEXT_INPUT_SENTINEL: write and verify output."}),
    case("failed_tool_corrected", [{"op": "read", "path": "missing.txt"},
         {"op": "read", "path": "spec.txt"}, WRITE, FINAL], initial={"spec.txt": "correct source"}),
    case("cross_slice", [{"op": "read", "path": "spec.txt"}, WRITE, FINAL],
         initial={"spec.txt": "CROSS_SLICE_RECEIPT_SENTINEL"}, max_rounds=1),
    case("malformed_tool", [{"op": "malformed"}], success=False),
    case("timeout", [{"op": "timeout"}], success=False, max_requests=2),
    case("retry_then_success", [{"op": "retryable_error"}, WRITE, FINAL]),
    case("false_final", [FINAL], success=False, max_requests=2),
    case("unknown_tool", [{"op": "unknown"}, {"op": "final", "outcome": "failed"}], success=False),
    case("denied_path", [{**WRITE, "path": "../outside.txt"},
         {"op": "final", "outcome": "failed"}], success=False),
    case("long_attachment_tail", [{"op": "material_tail"}, WRITE, FINAL], attachment="f" * 10000 + TAIL),
    case("image_supported", [WRITE, FINAL], image=True, supports_images=True),
    case("image_unsupported", [WRITE, FINAL], image=True, supports_images=False, success=False),
)


def validate_candidate(value):
    """Review only. A valid draft is neither permission nor a live invocation."""
    fields = {"model", "base_url", "mode", "limits", "samples"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("candidate_fields_invalid_no_credentials_allowed")
    if not isinstance(value["model"], str) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", value["model"]):
        raise ValueError("candidate_model_invalid")
    url = value["base_url"]
    if not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url):
        raise ValueError("candidate_endpoint_invalid")
    parts = urlsplit(url)
    try:
        port = parts.port
        loopback = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        loopback = False
        # Invalid port must not be treated like a non-IP hostname.
        try:
            port = parts.port
        except ValueError:
            raise ValueError("candidate_endpoint_invalid") from None
    if (not parts.hostname or parts.username is not None or parts.password is not None or parts.query
            or parts.fragment or (parts.scheme != "https" and not (parts.scheme == "http" and loopback))):
        raise ValueError("candidate_endpoint_invalid")
    if value["mode"] not in {"thinking", "nonthinking"}:
        raise ValueError("candidate_mode_invalid")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != set(BATCH_LIMITS):
        raise ValueError("candidate_limits_invalid")
    for name, ceiling in BATCH_LIMITS.items():
        n = limits[name]
        valid_type = type(n) in (int, float) if name == "max_active_seconds" else type(n) is int
        if not valid_type or not math.isfinite(n) or not 0 < n <= ceiling:
            raise ValueError("candidate_limits_invalid")
    samples = value["samples"]
    if (not isinstance(samples, list) or not samples or any(type(n) is not str for n in samples)
            or len(set(samples)) != len(samples) or set(samples) - {c["id"] for c in CASES}):
        raise ValueError("candidate_samples_invalid")
    return {**copy.deepcopy(value), "draft_only": True, "live_authorized": False,
            "real_model_calls": 0, "provider_cost": None}


def message_text(payload):
    for message in reversed(payload["messages"]):
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else "\n".join(
                item.get("text", "") for item in content if item.get("type") == "text")
    return ""


class SDKFixture:
    def __init__(self, root, sample):
        self.root, self.sample = root, sample
        self.payloads = []
        self.responses = []
        self.reported_usage = 0

    def __call__(self, request):
        if request.url.host != "execution-fixture.invalid" or request.url.path != "/v1/chat/completions":
            raise AssertionError("unapproved_fixture_http_target")
        if request.headers.get("authorization") != "Bearer offline-fixture-key":
            raise AssertionError("production_credential_reached_fixture")
        payload = json.loads(request.content)
        if payload.get("model") != MODEL:
            raise AssertionError("unapproved_fixture_model")
        self.payloads.append(payload)
        index = len(self.payloads) - 1
        action = self.sample["actions"][min(index, len(self.sample["actions"]) - 1)]
        op = action["op"]
        if op == "timeout":
            raise httpx.ReadTimeout("fixture model timeout", request=request)
        if op == "retryable_error":
            return httpx.Response(429, json={"error": {"message": "fixture rate limit", "type": "rate_limit"}})
        message = {"role": "assistant", "content": None, "reasoning_content": REASONING}
        if op in {"read", "write", "unknown", "malformed", "material_tail"}:
            name = "benchmark_" + op
            arguments = {k: v for k, v in action.items() if k != "op"}
            if "path" in arguments:
                arguments["path"] = str(self.root / arguments["path"])
            if op == "unknown":
                name, arguments = "unregistered_fixture_tool", {}
            if op == "malformed":
                name, arguments = "benchmark_write", "{"
            if op == "material_tail":
                text = message_text(payload)
                ledger, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
                item = next(i for i in ledger["task_materials"]["items"] if i["source_id"] == "fixture-attachment")
                start = item["content_chars"] - item["omitted_chars"]
                name, arguments = "task_material_read", {"reference": item["reference"],
                    "offset": start, "length": min(12000, item["omitted_chars"])}
            message["tool_calls"] = [{"id": "call_fixture_" + str(index), "type": "function", "function": {
                "name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}]
        else:
            message["content"] = json.dumps({"kind": "final", "text": "Fixture result; no external delivery.",
                                             "task_outcome": action.get("outcome", "completed")})
        usage = {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
        self.responses.append(message)
        self.reported_usage += 100
        if payload.get("stream"):
            # Real SDK SSE parsing, including an independent reasoning delta.
            chunks = [({"reasoning_content": REASONING}, None, None),
                      ({"content": message["content"][:20]}, None, None),
                      ({"content": message["content"][20:]}, "stop", None), ({}, None, usage)]
            content = "".join("data: " + json.dumps({"id": "fixture-stream", "object": "chat.completion.chunk",
                "created": 1, "model": MODEL, "choices": [] if u else [{"index": 0, "delta": delta,
                "finish_reason": finish}], **({"usage": u} if u else {})}) + "\n\n" for delta, finish, u in chunks)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=(content + "data: [DONE]\n\n").encode())
        return httpx.Response(200, json={"id": "fixture-completion", "object": "chat.completion", "created": 1,
            "model": MODEL, "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}], "usage": usage})


class ProbeRuntime(LLMRuntime):
    """Only configuration/audit isolation and fixture capability are replaced.

    Payload, SDK serialization, parsing, retries and task metering stay real.
    This runtime cannot create an actual network transport or reload live config.
    """
    def __init__(self, fixture, batch):
        self.fixture = fixture
        self.sdk = OpenAI(api_key="offline-fixture-key", base_url=BASE_URL, max_retries=0,
            timeout=2, http_client=httpx.Client(transport=httpx.MockTransport(fixture), trust_env=False))
        self.sdk.protocol = "openai"
        create = self.sdk.chat.completions.create

        def bounded_create(**payload):
            reservation = batch.reserve_request(payload)
            response = create(**payload)
            # Streaming usage reconciles at stream end in the probe below.
            if not payload.get("stream"):
                batch.reconcile(reservation, response)
            return response
        self.sdk.chat.completions.create = bounded_create
        super().__init__(runtime_policy=RuntimePolicy(http_attempts=2, retry_backoff_seconds=0,
            stream_attempts=1, chat_timeout_seconds=2, circuit_failure_threshold=100000))

    def _env(self, name, default=""):
        return default

    def _make_chat_bundle(self):
        return ModelBundle(self.sdk, MODEL)

    _make_aux_bundle = _make_chat_bundle

    def _tool_profile(self, bundle):
        return ProviderToolProfile(supports_native_tools=True, verified=False,
                                   notes="Synthetic SDK transport fixture only; provider support unknown.")

    def _build_cache_kwargs(self, **kwargs):
        return {}

    def _record_context_budget(self, record):
        return {}  # No repository/production audit writes; all wire observations are local.

    def close(self):
        self.sdk.close()


class FailureReadAdapter(benchmark.FileAdapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.allowed.add("missing.txt")
        self.attempts = []

    def execute(self, *, call, context):
        self.attempts.append(Path(call["path"]).name)
        return super().execute(call=call, context=context)  # Actual FileNotFoundError, then real read.


class ProbeMaterialStore(TaskMaterialStore):
    def __init__(self, store, root, batch):
        super().__init__(store, approved_roots=(root,))
        self.batch, self.read_dispatches = batch, 0

    def read(self, *args, **kwargs):
        self.batch.use_tool()
        self.read_dispatches += 1
        return super().read(*args, **kwargs)


class ObservedExecutor(AgentExecutor):
    def __init__(self, *args):
        super().__init__(*args)
        self.slice_outcomes = []

    def run(self, **kwargs):
        result = super().run(**kwargs)
        self.slice_outcomes.append(result.status)
        return result


def run_case(sample, root, batch, *, runtime_factory=ProbeRuntime, native_tool_history=False):
    root.mkdir(exist_ok=False)
    for name, text in sample.get("initial", {}).items():
        (root / name).write_text(text, encoding="utf-8")
    outside = root.parent / "outside.txt"
    before = outside.read_bytes()
    reader, writer = FailureReadAdapter(root, sample, "read", batch), benchmark.FileAdapter(root, sample, "write", batch)
    fixture = SDKFixture(root, sample)
    runtime = runtime_factory(fixture, batch)
    host = ToolHost(ToolRegistry({reader.tool_type: reader, writer.tool_type: writer}))
    store = AgentTaskStore(root / "business.sqlite")
    ExpressionArchive(store)
    coordinator = AgentTaskCoordinator(max_workers=1, state_path=root / "workers.sqlite")
    materials = []
    if sample.get("attachment"):
        materials.append(MaterialSource("fixture-attachment", "attachment", sample["attachment"]))
    if sample.get("image"):
        materials.append(MaterialSource("fixture-image", "image", {"data_url": DATA_URL},
            provenance={"sender_id": "original-fixture-author", "relation": "related_message"}))
    limits = {**TASK_LIMITS, **({"max_requests": sample["max_requests"]} if "max_requests" in sample else {})}
    grant = ExecutionGrant("offline-sdk-grant", frozenset({reader.tool_type, writer.tool_type}), permission_mode="full",
        policy=ExecutionPolicy(risk_by_tool={reader.tool_type: "low", writer.tool_type: "low"}),
        budget_limits=limits, resume_safe=True, material_sources=tuple(materials))
    executor = ObservedExecutor(host, ExecutionProfile(runtime, profile_id=MODEL,
        max_rounds=sample.get("max_rounds", 4), runtime_meters_budget=True,
        supports_images=sample.get("supports_images"), native_tool_history=native_tool_history))
    material_store = ProbeMaterialStore(store, root, batch)
    service = AgentService(executor=executor, authorize=lambda *_: grant,
        task_store=store, coordinator=coordinator, material_store=material_store)
    client = service.bind(SCOPE)
    path = root / "output.txt"
    request = AgentRequest(sample["id"], "Use the supplied fixture context, write output, and verify the original conditions.",
        {"goal": "Write and verify the approved fixture output", "deliverables": [str(path)], "scope": [str(root)],
         "requirements": [{"id": "output", "description": "original approved output", "effect": "write",
             "targets": [str(path)], "checks": {"sha256": benchmark.digest(b"verified fixture\n")}}]},
        constraints=("CONFIRMED_CONSTRAINT_SENTINEL: no external delivery or deployment.",),
        sources=({"id": "prior-context", "content": "RELATED_CONTEXT_SENTINEL: approved local task."},))
    try:
        handle = client.submit(request)
        view = benchmark.settle(client, handle.task_id)
        result = client.result(handle.task_id)
        prior = (len(fixture.payloads), len(reader.calls), len(writer.calls), view["version"])
        for _ in range(3):
            client.status(handle.task_id)
            client.result(handle.task_id)
            client.submit(request)
        quiet_queries = prior == (len(fixture.payloads), len(reader.calls), len(writer.calls),
                                  client.status(handle.task_id)["version"])
        publications, _ = benchmark.publish_once(store, SCOPE, "dry_run")
        checks = benchmark.independent_artifacts(root, sample)
        evaluation = benchmark.assess(sample, result, checks, writer.calls, publications,
            outside_unchanged=outside.read_bytes() == before, query_unchanged=quiet_queries)
        task = store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        wire = json.dumps(fixture.payloads, ensure_ascii=False)
        images = [item["image_url"]["url"] for p in fixture.payloads for m in p["messages"]
            if isinstance(m.get("content"), list) for item in m["content"] if item.get("type") == "image_url"]
        return {"case": sample["id"], **evaluation, "status": result["execution"]["status"],
            "reason": result["execution"]["reason"], "sdk_requests": len(fixture.payloads),
            "slices": len(executor.slice_outcomes), "slice_outcomes": executor.slice_outcomes,
            "budget": task["budget"], "reported_fixture_usage": fixture.reported_usage,
            "tool_read_attempts": reader.attempts, "writes": writer.calls, "artifact_checks": checks,
            "material_read_dispatches": material_store.read_dispatches,
            "publications": publications, "query_unchanged": quiet_queries,
            "original_contract_preserved": task["contract"] == request.payload()["contract"],
            "prior_context_on_wire": "RELATED_CONTEXT_SENTINEL" in wire,
            "confirmed_constraint_on_wire": "CONFIRMED_CONSTRAINT_SENTINEL" in wire,
            "read_result_on_later_wire": any("CROSS_SLICE_RECEIPT_SENTINEL" in json.dumps(p["messages"], ensure_ascii=False)
                for p in fixture.payloads[1:]) if sample["id"] == "cross_slice" else None,
            "attachment_tail_on_wire": TAIL in wire if sample.get("attachment") else None,
            "image_bytes_on_wire": bool(images) and all(i == DATA_URL for i in images),
            "image_source_on_wire": "original-fixture-author" in wire if sample.get("image") else None,
            "reasoning_excluded_from_result": REASONING not in json.dumps(result),
            "native_history_on_wire": any(m.get("role") == "tool" or m.get("tool_calls")
                or m.get("reasoning_content") for p in fixture.payloads for m in p["messages"])}
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()
        store.close()


def history_probe(root, batch):
    fixture = SDKFixture(root, case("history-fields", [FINAL]))
    runtime = ProbeRuntime(fixture, batch)
    history = [{"role": "assistant", "content": "", "reasoning_content": REASONING,
        "tool_calls": [{"id": "original-call", "type": "function", "function": {
            "name": "benchmark_read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "original-call", "content": "ORIGINAL_TOOL_RECEIPT_SENTINEL"}]
    task_budget = TaskBudget(**TASK_LIMITS)
    try:
        with bind_task_budget(task_budget):
            response = runtime.call_chat_json(system_prompt="fixture execution", user_prompt="next fixture step",
                history_turns=history, fallback={"kind": "final", "text": "fixture failure"},
                native_tools=[{"type": "function", "function": {"name": "benchmark_read",
                    "parameters": {"type": "object", "properties": {}}}}])
        messages = fixture.payloads[0]["messages"]
        return {"sdk_requests": len(fixture.payloads), "budget": task_budget.snapshot(),
            "assistant_native_history_preserved": history[0] in messages,
            "tool_role_and_id_preserved": history[1] in messages,
            "reasoning_history_preserved": any(m.get("reasoning_content") == REASONING for m in messages),
            "provider_message_retained": any(k in response for k in ("_native_response", "_provider_message")),
            "reasoning_excluded_from_public_response": REASONING not in json.dumps(response),
            "provider_capability": "not_measured", "protocol_conclusion": "observe_fields_independently"}
    finally:
        runtime.close()


def stream_probe(root, batch):
    fixture = SDKFixture(root, case("sse", [FINAL]))
    runtime = ProbeRuntime(fixture, batch)
    task_budget = TaskBudget(**TASK_LIMITS)
    try:
        with bind_task_budget(task_budget):
            generator = runtime.stream_chat_json(system_prompt="fixture execution", user_prompt="fixture final",
                fallback={"kind": "final", "text": "fixture failure"})
            events = []
            while True:
                try:
                    events.append(next(generator))
                except StopIteration as ended:
                    result = ended.value
                    break
        # TaskBudget reconciles streamed usage through the existing stream proxy.
        batch.reconcile(batch.last_reservation, {"usage": {"total_tokens": fixture.reported_usage}})
        return {"sdk_requests": len(fixture.payloads), "sent_stream": fixture.payloads[0].get("stream") is True,
            "final_json_parsed": result.parsed.get("kind") == "final", "usage": task_budget.tokens,
            "reasoning_excluded": REASONING not in json.dumps(events) and REASONING not in result.raw_text,
            "native_thinking_stream_continuity": "not_measured"}
    finally:
        runtime.close()


def run_suite(workspace, *, candidate=None):
    reviewed = validate_candidate(candidate) if candidate is not None else None
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "outside.txt").write_text("OUTSIDE_SCOPE_SENTINEL", encoding="utf-8")
    batch = TaskBudget(**BATCH_LIMITS, managed=True, profile="offline-sdk-protocol-batch")
    with bind_task_budget(batch):
        rows = []
        for sample in CASES:
            row = run_case(copy.deepcopy(sample), root / sample["id"], batch)
            rows.append(row)
            (root / sample["id"] / "observations.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        history = history_probe(root, batch)
        stream = stream_probe(root, batch)
    blockers = [name for name in ("assistant_native_history_preserved", "tool_role_and_id_preserved",
                                  "reasoning_history_preserved", "provider_message_retained") if not history[name]]
    report = {"schema_version": 1, "mode": "offline-sdk-fixtures", "corpus_hash": canonical_hash(CASES),
        "rows": rows, "history_fields": history, "sse": stream, "batch": batch.snapshot(),
        "batch_tools_accounting": "actual_fixture_file_and_material_dispatches",
        "fixture_gates_pass": all(row["fixture_pass"] for row in rows) and stream["final_json_parsed"],
        "native_thinking_continuation_blockers": blockers, "candidate_review": reviewed,
        "provider_acceptance": "not_run", "model_quality": "not_measured", "provider_cost": None,
        "real_model_calls": 0, "real_platform_sends": 0, "enable_production": False,
        "ready_for_thinking_tool_continuation": False,
        "next_required_inputs": ["model_and_endpoint", "execution_mode", "approved_samples_and_request_or_cost_limit"]}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="Optional credential-free review draft; never invokes a provider.")
    args = parser.parse_args()
    candidate = None
    if args.candidate is not None:
        if args.candidate.stat().st_size > 16384:
            parser.error("candidate_draft_oversized")
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    # The CLI runs in its own process; every socket connection is forbidden.
    def no_network(*args, **kwargs):
        raise AssertionError("offline_probe_network_forbidden")
    socket.socket.connect = socket.socket.connect_ex = no_network
    report = run_suite(args.workspace, candidate=candidate)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))
    return 0 if report["fixture_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
