"""Host-audited interaction capabilities, distinct from execution tools.

The host supplies this allowlist. Handler read/observe markers alone cannot
add a capability, its arguments, or its source scope to a chat request.
"""

from __future__ import annotations

import copy
from contextvars import copy_context
from dataclasses import dataclass
import json
import math
import re
import threading
import time
from types import SimpleNamespace
from types import MappingProxyType
from typing import Callable, Mapping

from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.tools.invocation import ToolInvocation


TASK_TOOL_NAMES = frozenset("agent_task_" + value for value in
    ("submit", "status", "result", "cancel", "resume", "supplement"))


def _object(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


def _workflow_catalog(values):
    """目录只来自可信宿主；复制并校验有界名称/说明，不包含执行权限或参数。"""
    if (not isinstance(values, Mapping) or len(values) > 8
            or any(not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name)
                   or not isinstance(description, str) or not 0 < len(description) <= 600
                   for name, description in values.items())):
        raise ValueError('invalid_host_task_workflows')
    return dict(sorted(values.items()))


def task_tool_specs(*, delegated=False, workflows=None):
    """保留原任务接口；可选工作流仅选择宿主入口，原消息与授权仍由宿主绑定。"""
    workflows = _workflow_catalog({} if workflows is None else workflows)
    if workflows and not delegated:
        raise ValueError('host_task_workflow_requires_delegation')
    task_id = {"type": "string", "minLength": 1, "maxLength": 160}
    schemas = {
        "submit": _object({"goal": {"type": "string", "minLength": 1, "maxLength": 2000}}),
        **{name: _object({"task_id": task_id}, ("task_id",)) for name in
           ("status", "result", "cancel", "resume", "supplement")},
    }
    descriptions = {
        "submit": "Submit the complete original user request and host-selected materials for execution effects or sustained multi-stage work. "
                  "For a bounded public fact lookup or a small comparison, start with the offered host-audited read capabilities. "
                  "Searching, reading a few sources, and citing them does not by itself require a formal task. "
                  "The host manages read limits and eligible background continuation; a denied source is a limitation, not authorization to escalate. "
                  "An optional goal is caller data; it does not replace the original request or authorize an operation.",
        "status": "Read scoped task status without planning or executing.",
        "result": "Read a scoped, version-checked task result without executing or redelivering it.",
        "cancel": "Request cancellation of an explicitly selected task; stopping is receipt-based, without rollback.",
        "resume": "Explicitly continue a scoped task using its existing receipts and cumulative limits.",
        "supplement": "Attach the user's original clarification to a scoped task; does not resume execution.",
    }
    if delegated:
        schemas['supplement']['properties']['resume'] = {'type': 'boolean'}
        descriptions['supplement'] = ("Attach the user's complete original clarification or choice to the existing task. "
            "Set resume=true when the user wants work to continue after applying it; default false only saves. "
            "Reuse the task's goal, receipts and cumulative budget. If the old worker is still stopping, "
            "the host reports pending rather than executing concurrently.")
        schemas['result']['properties']['view'] = {'type': 'string', 'enum': ['summary', 'full']}
        descriptions['result'] += ' Default summary preserves the deliverable; select view=full for original report and execution receipts.'
        descriptions['submit'] = ('Delegate the complete user goal to the connected independent Angelos. '
            'Choose whether this capability is useful using current context and actual capabilities; '
            'there is no task-category, keyword, source-count or step-count threshold. '
            'The host supplies original requests and materials; goal is advisory data, not authority.')
    if workflows:
        schemas['submit']['properties']['workflow'] = {
            'type': 'string', 'enum': list(workflows),
            'description': '可选宿主工作流：' + '；'.join(name + '：' + text for name, text in workflows.items())}
        descriptions['submit'] += (' A workflow selects a host-audited task ingress. Omit it for ordinary delegation. '
            'It grants no tools, targets, credentials, publishing permission or budget; the host validates the original request.')
    return [{"type": "function", "function": {"name": "agent_task_" + name,
        "description": descriptions[name], "parameters": schema}} for name, schema in schemas.items()]


def validate_arguments(value, schema):
    """A deliberately bounded schema subset, rejecting unknown fields/types."""
    if not isinstance(schema, Mapping):
        return False
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return (isinstance(value, Mapping) and not set(value) - set(props)
            and set(schema.get("required", ())) <= set(value)
            and all(validate_arguments(item, props[name]) for name, item in value.items()))
    if kind == "array":
        return (isinstance(value, list) and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 16)
            and all(validate_arguments(item, schema.get("items", {})) for item in value))
    if kind == "string":
        valid = isinstance(value, str) and schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 2000)
    elif kind in {"integer", "number"}:
        valid = (isinstance(value, int if kind == "integer" else (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and schema.get("minimum", -1e9) <= value <= schema.get("maximum", 1e9))
    elif kind == "boolean":
        valid = isinstance(value, bool)
    else:
        return False
    return valid and ("enum" not in schema or value in schema["enum"])


def _check_schema(schema, depth=0):
    if not isinstance(schema, dict) or depth > 4 or set(schema) - {
        "type", "description", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
        "minLength", "maxLength", "minimum", "maximum", "enum"}:
        raise ValueError("unsupported_chat_read_schema")
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        if (schema.get("additionalProperties") is not False or not isinstance(props, dict) or len(props) > 16
                or not set(schema.get("required", ())) <= set(props)):
            raise ValueError("unbounded_chat_read_schema")
        for name, value in props.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
                raise ValueError("invalid_chat_read_argument")
            # No generic command/path dispatcher can be registered as a read.
            if name.casefold() in {"command", "script", "code", "path", "cwd", "tool", "tools", "arguments"} and not value.get("enum"):
                raise ValueError("unbounded_chat_read_argument")
            _check_schema(value, depth + 1)
    elif kind == "array":
        if not 0 <= schema.get("minItems", 0) <= schema.get("maxItems", -1) <= 16:
            raise ValueError("unbounded_chat_read_schema")
        _check_schema(schema.get("items"), depth + 1)
    elif kind == "string":
        if not 0 <= schema.get("minLength", 0) <= schema.get("maxLength", -1) <= 4000:
            raise ValueError("unbounded_chat_read_schema")
    elif kind in {"integer", "number"}:
        if not -1e6 <= schema.get("minimum", -math.inf) <= schema.get("maximum", math.inf) <= 1e6:
            raise ValueError("unbounded_chat_read_schema")
    elif kind != "boolean":
        raise ValueError("unsupported_chat_read_schema")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not 1 <= len(schema["enum"]) <= 32):
        raise ValueError("invalid_chat_read_enum")


@dataclass(frozen=True)
class ChatReadCapability:
    name: str
    description: str
    parameters: dict
    route: str = "perception"
    max_calls: int = 1
    max_result_bytes: int = 24000
    max_seconds: float = 10.0
    # Source-specific checks (e.g. a public URL must be original input or a
    # receipt-backed candidate), never a model-provided authority callback.
    argument_guard: Callable[[dict, Mapping], bool] | None = None

    def __post_init__(self):
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.name) or self.name in TASK_TOOL_NAMES
                or not self.description or len(self.description) > 900 or self.route not in {"perception", "public_query"}
                or not 1 <= self.max_calls <= 3 or not 256 <= self.max_result_bytes <= 64000
                or not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 30
                or (self.argument_guard is not None and not callable(self.argument_guard))):
            raise ValueError("invalid_chat_read_capability")
        _check_schema(self.parameters)
        if self.parameters.get("type") != "object":
            raise ValueError("chat_read_object_required")
        if any("url" in name.casefold() for name in self.parameters.get("properties", {})) and self.argument_guard is None:
            raise ValueError("chat_read_source_guard_required")
        # Take an owned snapshot so later caller changes cannot expand schemas.
        object.__setattr__(self, "parameters", copy.deepcopy(self.parameters))
        object.__setattr__(self, "description", " ".join(self.description.split()))

    def spec(self):
        purpose = ("Host-audited capability for a bounded public fact lookup or a small comparison. "
                   "Use these reads before submitting sustained work; source scope and any background continuation are host-managed. "
                   if self.route == "public_query" else
                   "Host-audited scoped read without starting an execution task. ")
        bounds = (f" Per-turn call limit: {self.max_calls}; per-call result limit: {self.max_result_bytes} bytes; "
                  f"per-call time limit: {self.max_seconds:g} seconds. These limits do not grant new source access.")
        return {"type": "function", "function": {"name": self.name, "description": purpose + self.description + bounds,
                "parameters": copy.deepcopy(self.parameters)}}

    def admits(self, arguments, context):
        return (validate_arguments(arguments, self.parameters)
            and (self.argument_guard is None or self.argument_guard(dict(arguments), context)))


class AuditedReadHandler:
    """Apply size, time, count and argument limits at the real adapter boundary."""
    agent_audited_query_read = True
    def __init__(self, handler, capability):
        self.handler, self.capability = handler, capability
        self.tool_type = capability.name

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def tool_metadata(self):
        return SimpleNamespace(description=self.capability.spec()["function"]["description"],
            input_schema=copy.deepcopy(self.capability.parameters))

    def build_prompt_instruction(self):
        return self.capability.spec()["function"]["description"]

    def normalize_call(self, call):
        args = {key: value for key, value in call.items() if key != "type" and not key.startswith("_tool_")}
        if not validate_arguments(args, self.capability.parameters):
            return None
        result = self.handler.normalize_call(call)
        if not isinstance(result, Mapping):
            return None
        # A normalizer cannot insert an unchecked executable argument.
        normalized = {key: value for key, value in result.items() if key != "type" and not key.startswith("_tool_")}
        return dict(result) if validate_arguments(normalized, self.capability.parameters) else None

    def execute(self, *, call, context):
        try:
            return self._execute_read(call=call, context=context)
        except AgentBoundaryError as exc:
            return {"tool_type": self.tool_type, "ok": False, "status": "error", "error_code": exc.code,
                    "followup_context": "Read capability stopped: " + exc.code,
                    "state_updates": {"result_valid": False}}

    def _execute_read(self, *, call, context):
        from shinku.agent.request_scope import RequestScope
        scope = context.get("agent_request_scope") if isinstance(context, dict) else None
        if not isinstance(scope, RequestScope):
            raise AgentBoundaryError("chat_read_scope_required")
        args = {key: value for key, value in call.items() if key != "type" and not key.startswith("_tool_")}
        if not self.capability.admits(args, context):
            raise AgentBoundaryError("chat_read_arguments_rejected")
        state = context.setdefault("_chat_read_limits", {"started": time.monotonic(), "counts": {}, "lock": threading.Lock()})
        with state["lock"]:
            count = state["counts"].get(self.tool_type, 0)
            if count >= self.capability.max_calls:
                raise AgentBoundaryError("chat_read_call_limit")
            state["counts"][self.tool_type] = count + 1
        finished, box = threading.Event(), []
        def read():
            try:
                box.append(self.handler.execute(call=call, context=context))
            except Exception as exc:
                box.append(exc)
            finally:
                finished.set()
        # These audited adapters have no effects. A timed-out read may settle
        # later, but its late return never enters evidence or starts another call.
        copied = copy_context()
        thread = threading.Thread(target=lambda: copied.run(read), name="chat-audited-read", daemon=True)
        thread.start()
        if not finished.wait(self.capability.max_seconds):
            raise AgentBoundaryError("chat_read_timeout")
        if isinstance(box[0], Exception):
            raise box[0]
        result = box[0]
        if len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")) > self.capability.max_result_bytes:
            raise AgentBoundaryError("chat_read_result_oversized")
        from shinku.tools.execution import result_to_envelope
        if result_to_envelope(invocation=ToolInvocation(self.tool_type, args), result=result).status == "ok":
            context.setdefault("_chat_read_receipts", []).append({"tool": self.tool_type,
                "arguments": copy.deepcopy(args), "result": copy.deepcopy(result), "source": "executed_read_adapter"})
        return result


def audited_handlers(handlers, capabilities, *, route=None):
    result = {}
    for name, capability in capabilities.items():
        if route is not None and capability.route != route:
            continue
        handler = handlers.get(name)
        if handler is None:
            continue
        if capability.route == "public_query":
            factory = getattr(handler, "public_query_variant", None)
            if callable(factory):
                handler = factory()
            if getattr(handler, "agent_fast_public_query", None) is not True:
                continue
        effects = set(getattr(handler, "agent_completion_effects", ()) or ())
        if (getattr(handler, "tool_type", "") != name or getattr(handler, "agent_permission_class", "") != "read"
                or getattr(handler, "agent_progress_class", "") != "observe"
                or effects & {"write", "delete", "send", "process_stop", "reminder", "browser_access", "command_run"}):
            continue
        result[name] = AuditedReadHandler(handler, capability)
    return result


class InteractionRuntime:
    """Guard every chat request, including provider retry/repair callers."""
    def __init__(self, runtime, registry, capabilities, *, delegated=False):
        self.runtime, self.registry, self.capabilities = runtime, registry, capabilities
        self.delegated = bool(delegated)
        self.task_workflows = MappingProxyType({})

    def configure_task_workflows(self, values):
        """装配阶段绑定一次目录；模型响应和上下文不能修改真实SDK准入上界。"""
        catalogue = _workflow_catalog(values)
        if not self.delegated or self.task_workflows:
            raise ValueError('host_task_workflow_binding_conflict')
        self.task_workflows = MappingProxyType(catalogue)

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def _guard(self, kwargs):
        safe = {row["function"]["name"]: row for row in task_tool_specs(delegated=self.delegated)}
        extended = {row['function']['name']: row for row in task_tool_specs(
            delegated=self.delegated, workflows=self.task_workflows)}
        safe.update({name: value.spec() for name, value in self.capabilities.items()})
        for spec in kwargs.get("native_tools") or ():
            name = spec.get("function", {}).get("name") if isinstance(spec, Mapping) else ""
            # 无工作流可用的会话仍用原schema；扩展schema必须逐字等于装配时的目录。
            if name not in safe or spec != safe[name] and spec != extended.get(name):
                raise AgentBoundaryError("chat_execution_tools_rejected")
        system = str(kwargs.get("system_prompt") or "") + json.dumps(kwargs.get("system_extra_blocks") or [], ensure_ascii=False)
        formal = set(self.registry.names()) - set(self.capabilities)
        if any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", system) for name in formal):
            raise AgentBoundaryError("chat_execution_catalog_rejected")

    def call_chat_json(self, **kwargs):
        self._guard(kwargs)
        return self.runtime.call_chat_json(**kwargs)

    def stream_chat_json(self, **kwargs):
        self._guard(kwargs)
        stream = getattr(self.runtime, "stream_chat_json", None)
        if not callable(stream):
            # A non-stream runtime retains the same guarded call route.
            def fallback():
                if False:
                    yield None
                return SimpleNamespace(parsed=self.call_chat_json(**{
                    key: value for key, value in kwargs.items() if key != "early_tool_call_validator"}), error="")
            return fallback()
        return stream(**kwargs)


def validate_interaction_response(response, capabilities, context):
    """Validate before the permissive, legacy protocol parser drops fields."""
    from shinku.agent.planner import LLMPlanner
    from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD
    if not isinstance(response, Mapping):
        raise AgentBoundaryError("chat_interface_format_invalid")
    allowed = {"kind", "text", "speech", "reply", "speech_segments", "status", "reason", "reply_routing",
               "tool_call", "tool_calls", "invocation", NATIVE_TOOL_CALL_FIELD}
    if response.get("kind") == "final":
        allowed.add("dialogue_basis")
    if set(response) - allowed:
        raise AgentBoundaryError("chat_interface_fields_rejected")
    _validate_call_shapes(response)
    decision = LLMPlanner._parse(response)
    if decision is None:
        raise AgentBoundaryError("chat_interface_format_invalid")
    if decision.kind in {"final", "wait"}:
        if any(response.get(key) for key in ("tool_call", "tool_calls", "invocation", NATIVE_TOOL_CALL_FIELD)):
            raise AgentBoundaryError("chat_interface_mixed_decision")
        return decision
    calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
    if len(calls) != 1:
        raise AgentBoundaryError("chat_interface_single_call_required")
    call = calls[0]
    if call.protocol_error:
        raise AgentBoundaryError("chat_interface_format_invalid")
    specs = {row["function"]["name"]: row["function"]["parameters"] for row in
             task_tool_specs(delegated=bool(context.get('_agent_delegation_host')),
                             workflows=context.get('_host_task_workflows'))}
    capability = None
    if call.name in specs:
        schema = specs[call.name]
    elif call.name in capabilities:
        capability = capabilities[call.name]
        schema = capability.parameters
    else:
        raise AgentBoundaryError("chat_execution_call_rejected")
    schema_valid = validate_arguments(call.arguments, schema)
    checked = schema_valid and capability is not None and capability.argument_guard is not None
    accepted = bool(capability.argument_guard(dict(call.arguments), context)) if checked else False
    if not schema_valid or checked and not accepted:
        props = schema.get("properties", {})
        args = call.arguments if isinstance(call.arguments, Mapping) else {}
        error = AgentBoundaryError("chat_interface_arguments_rejected")
        # Counts and fixed labels only; neither unknown names nor caller values
        # belong in the trace. The guard is evaluated once, after the schema.
        error.argument_diagnostic = {"interface": call.name, "schema_valid": schema_valid,
            "source_guard_checked": checked, "source_guard_accepted": accepted,
            "unknown_argument_count": len(set(args) - set(props)),
            "missing_required_count": len(set(schema.get("required", ())) - set(args)),
            "invalid_known_field_count": sum(not validate_arguments(value, props[name])
                for name, value in args.items() if name in props)}
        raise error
    return decision


def _validate_call_shapes(response):
    from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD
    def check_call(call):
        if not isinstance(call, Mapping):
            raise AgentBoundaryError("chat_interface_format_invalid")
        if "function" in call:
            if set(call) - {"function", "id", "type", "source"} or not isinstance(call["function"], Mapping):
                raise AgentBoundaryError("chat_interface_fields_rejected")
            if set(call["function"]) - {"name", "arguments"}:
                raise AgentBoundaryError("chat_interface_fields_rejected")
        elif "arguments" in call and set(call) - {"name", "type", "arguments", "id", "source", "protocol_error"}:
            raise AgentBoundaryError("chat_interface_fields_rejected")
    for key in ("tool_call", "invocation"):
        if response.get(key) is not None:
            check_call(response[key])
    if response.get("tool_calls") is not None:
        if not isinstance(response["tool_calls"], list):
            raise AgentBoundaryError("chat_interface_format_invalid")
        for call in response["tool_calls"]:
            check_call(call)
    native = response.get(NATIVE_TOOL_CALL_FIELD)
    if isinstance(native, Mapping) and native.get("_malformed") is not True and "tool_calls" in native:
        if not isinstance(native["tool_calls"], list) or not native["tool_calls"]:
            raise AgentBoundaryError("chat_interface_format_invalid")
        for call in native["tool_calls"]:
            check_call(call)
        first = native["tool_calls"][0]
        args = first.get("arguments")
        if not isinstance(args, Mapping):
            raise AgentBoundaryError("chat_interface_format_invalid")
        if set(native) - {"type", "_tool_source", "_tool_invocation_id", "tool_calls", *args}:
            raise AgentBoundaryError("chat_interface_fields_rejected")
        if any(native[key] != args[key] for key in args if key in native):
            raise AgentBoundaryError("chat_interface_mixed_decision")


def validate_read_query_response(response, capabilities, context):
    """The bounded query loop cannot acquire a task route or hidden grant."""
    if not isinstance(response, Mapping):
        raise AgentBoundaryError("chat_interface_format_invalid")
    link = response.get("task_link")
    if link is not None and (not isinstance(link, Mapping) or set(link) - {"mode", "track"}
            or link.get("mode") != "none" or link.get("track", "public_query") != "public_query"):
        raise AgentBoundaryError("chat_read_route_rejected")
    original = dict(response)
    value = {key: item for key, item in original.items() if key not in {"task_link", "task_outcome"}}
    decision = validate_interaction_response(value, capabilities, context)
    calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
    if any(call.name not in capabilities or capabilities[call.name].route != "public_query" for call in calls):
        raise AgentBoundaryError("chat_read_route_rejected")
    from dataclasses import replace
    return replace(decision, task_link=link, outcome=str(original.get("task_outcome") or ""))
