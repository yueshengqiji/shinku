"""Call-local parser observations and an explicit plain-reply candidate policy.

No model text, prompt, credentials or reasoning is retained in observations.
Only the QQ host opts in, and a recovered candidate still needs its reply gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class ResponseObservation:
    allow_plain_reply: bool = False
    allow_final_text_json: bool = False
    discard_final_task_metadata: bool = False
    classify_invalid_output: bool = False
    events: list[dict] = field(default_factory=list)
    finish_reason: str = "unknown"
    call_failure: dict = field(default_factory=dict)
    # 仅持有本次响应的结构元数据；正文和工具参数始终留在原调用中。
    response_shape: dict = field(default_factory=dict)


_observation = ContextVar("response_protocol_observation", default=None)


@contextmanager
def observe_response(*, allow_plain_reply=False, allow_final_text_json=False, discard_final_task_metadata=False,
                     classify_invalid_output=False):
    observation = ResponseObservation(allow_plain_reply=allow_plain_reply is True,
        allow_final_text_json=allow_final_text_json is True,
        discard_final_task_metadata=discard_final_task_metadata is True,
        classify_invalid_output=classify_invalid_output is True)
    token = _observation.set(observation)
    try:
        yield observation
    finally:
        _observation.reset(token)


def plain_reply_enabled():
    observation = _observation.get()
    return observation is not None and observation.allow_plain_reply


def recover_final_text(content):
    """仅由当前宿主观察范围开启；无额外模型调用，不扩大工具解析范围。"""
    observation = _observation.get()
    if observation is None or not observation.allow_final_text_json:
        return None
    from .final_json import recover_final_text_json
    value = recover_final_text_json(content, discard_task_metadata=observation.discard_final_task_metadata)
    if value is not None:
        observation.events.append({'stage': 'wait_text_json_recovered' if value['kind'] == 'wait' else 'final_text_json_recovered'})
    return value


def response_diagnostics():
    """只读当前调用的安全诊断，不嵌套观察器，避免吞掉外层解析/截断信号。"""
    import copy
    observation = _observation.get()
    if observation is None:
        return {'failure': '', 'finish_reason': 'unknown', 'call_failure': {}}
    return {'failure': response_failure_kind(observation), 'finish_reason': observation.finish_reason,
            'call_failure': copy.deepcopy(observation.call_failure), 'events': copy.deepcopy(observation.events)}


def note_response(stage):
    if stage not in {"native_call", "json_object", "partial_object", "plain_reply_candidate",
                     "parse_invalid", "empty_response", "transport_error", "output_incomplete"}:
        raise ValueError("response_observation_stage_invalid")
    observation = _observation.get()
    if observation is not None:
        observation.events.append({"stage": stage})


def note_finish_reason(reason):
    """保存供应商实际结束标志；空白响应不能再只留下 unknown，亦不推断原因。"""
    observation = _observation.get()
    if observation is not None and reason in {"stop", "tool_calls", "length", "content_filter"}:
        observation.finish_reason = reason


def note_response_shape(content, *, transport, native_call=None):
    """记录有界长度与固定形态，空正文但有原生调用时不能误报模型空返回。"""
    observation = _observation.get()
    if observation is None:
        return
    body = content.strip() if isinstance(content, str) else ""
    shape = ("empty" if content is None or isinstance(content, str) and not body else
             "object" if body.startswith("{") else "array" if body.startswith("[") else
             "fenced" if body.startswith("```") else "text" if isinstance(content, str) else "other")
    observation.response_shape = {
        "transport": "stream" if transport == "stream" else "json",
        "body_shape": shape,
        "body_chars": min(len(content), 2**31 - 1) if isinstance(content, str) else 0,
        "native_call": "malformed" if isinstance(native_call, dict) and native_call.get("_malformed") is True
                       else "present" if native_call is not None else "none",
    }


def note_incomplete_response(finish_reason):
    """Keep a fixed termination signal without changing existing parser events."""
    note_response("output_incomplete")
    observation = _observation.get()
    if observation is not None and observation.finish_reason == "unknown":
        observation.finish_reason = finish_reason if isinstance(finish_reason, str) and finish_reason in {"length", "content_filter"} else "unknown"


def call_failure_metadata(phase, exc, *, response_received=False, finish_reason="unknown", usage=None):
    """只投影阶段、异常类型、HTTP安全类别与已报告用量，不把服务正文带入任务。"""
    from .provider_failure import provider_failure
    provider = provider_failure(exc)
    if not isinstance(phase, str) or phase not in {"configuration", "payload", "dispatch", "response_usage", "cache", "response", "parse",
                     "context_usage", "service_call", "service_result", "host_wait"}:
        phase = "service_call"
    allowed = {"TimeoutError", "ValueError", "TypeError", "KeyError", "RuntimeError", "OSError",
               "PermissionError", "ConnectionError", "APIConnectionError", "APITimeoutError",
               "APIError", "APIStatusError", "BadRequestError", "AuthenticationError",
               "PermissionDeniedError", "NotFoundError", "ConflictError", "RateLimitError",
               "InternalServerError", "UnprocessableEntityError", "APIResponseValidationError",
               "HTTPStatusError", "TransportError", "TimeoutException", "ConnectTimeout",
               "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError",
               "WriteError", "CloseError", "NetworkError", "ProtocolError", "LocalProtocolError",
               "RemoteProtocolError", "ProxyError", "UnsupportedProtocol", "DecodingError"}
    types, seen = [], set()
    for _ in range(8):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        name = type(exc).__name__
        types.append(name if name in allowed else "OtherError")
        exc = exc.__cause__ or exc.__context__
    counts = {k: v for k, v in (usage or {}).items()
              if k in {"input_tokens", "output_tokens", "total_tokens"} and type(v) is int and v >= 0}
    return {"schema_version": 1, "phase": phase, "exception_types": types, **provider,
            "response_received": response_received is True,
            "finish_reason": finish_reason if isinstance(finish_reason, str) and finish_reason in {"stop", "tool_calls", "length", "content_filter"} else "unknown",
            "reported_usage": counts, "secondary_failures": [], "diagnostic_only": True}


def note_call_failure(phase, exc, *, response_received=False, finish_reason="unknown", usage=None, secondary=False):
    observation = _observation.get()
    if observation is None:
        return
    metadata = call_failure_metadata(phase, exc, response_received=response_received,
                                     finish_reason=finish_reason, usage=usage)
    if not observation.call_failure:
        observation.call_failure = metadata
    elif secondary and len(observation.call_failure["secondary_failures"]) < 2:
        # An exception raised in cleanup implicitly chains the primary failure,
        # which is already retained above. Keep only the cleanup class here.
        observation.call_failure["secondary_failures"].append({"phase": metadata["phase"],
            "exception_types": metadata["exception_types"][:1]})


def response_failure_kind(observation):
    """Classify the first host-observed failure without inspecting model prose."""
    if observation is None:
        return ""
    for event in observation.events:
        if event["stage"] == "output_incomplete":
            return {"length": "output_length", "content_filter": "output_filtered"}.get(
                observation.finish_reason, "output_incomplete")
        if event["stage"] == "transport_error":
            timed_out = {"TimeoutError", "APITimeoutError", "TimeoutException", "ConnectTimeout",
                         "ReadTimeout", "WriteTimeout", "PoolTimeout"}
            return ("call_timeout" if timed_out.intersection(
                observation.call_failure.get("exception_types", [])) else "call_failed")
    # Agent 决策不得把运行时 fallback 当成模型主动失败。其他结构化消费者
    # 维持原兼容策略；不改变 QQ 自有格式修复或为此增加模型调用。
    if observation.classify_invalid_output:
        stages = {event['stage'] for event in observation.events}
        if 'empty_response' in stages:
            return 'output_empty'
        if 'parse_invalid' in stages:
            return 'output_invalid'
    return ""


def recover_plain_reply(content):
    if not plain_reply_enabled() or not isinstance(content, str):
        return None
    text = content.strip()
    # Recover complete ordinary prose only. Malformed object/array/code output
    # belongs to format repair; never strip fields or manufacture tool calls.
    if not text or len(text) > 4000 or any(c in text for c in ("{", "}", "[", "]", "```")):
        return None
    note_response("plain_reply_candidate")
    return {"kind": "final", "text": text}
