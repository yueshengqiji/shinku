"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_json_call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from shinku.agent.budget import TaskBudgetExceeded
from shinku.tools.invocation import (
    NATIVE_OPENAI,
    NATIVE_TOOL_CALL_FIELD,
    TOOL_SOURCE_FIELD,
)

from .native_messages import NativeProtocolError, assistant_message
from .response_protocol import (
    note_finish_reason,
    note_incomplete_response,
    note_response,
    note_response_shape,
    plain_reply_enabled,
    recover_final_text,
    recover_plain_reply,
)
from .runtime_types import ModelBundle


@dataclass(frozen=True)
class RuntimeJsonCallDependencies:
    """Bound callbacks and live getters for this responsibility."""
    add_metric: Callable[..., Any]
    build_payload: Callable[..., Any]
    coerce_tools: Callable[..., Any]
    dispatch_completion: Callable[..., Any]
    note_error: Callable[..., Any]
    note_parse_failure: Callable[..., Any]
    note_truncated: Callable[..., Any]
    parse_json: Callable[..., Any]
    read_content_text: Callable[..., Any]
    read_native_tool_call: Callable[..., Any]
    record_cache: Callable[..., Any]
    record_context_budget_usage: Callable[..., Any]
    recover_agent_plain_text: Callable[..., Any]
    recover_partial: Callable[..., Any]
    chat: Callable[..., Any]
    optional_is_malformed_native_tool_call: Callable[..., Any]


def run_json_call(
    *,
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
    temperature: float,
    prompt_cache_key: str,
    user_images: list[dict[str, Any]] | None=None,
    native_tools: list[dict[str, Any]] | None=None,
    native_tool_choice: Any='',
    system_extra_blocks: list[str] | None=None,
    history_turns: list[dict[str, str]] | None=None,
    prompt_audit_sections: list[dict[str, Any]] | None=None,
    attempt_budget: dict[str, int] | None=None,
    native_history: list[dict[str, Any]] | None=None,
    provider_message_sink: Callable[[dict[str, Any]], None] | None=None,
    deps: RuntimeJsonCallDependencies,
) -> dict[str, Any]:
    from .context_compaction import note_sdk_input_path
    note_sdk_input_path()
    native_requested = bool(deps.coerce_tools(native_tools))
    budget_context: dict[str, Any] = {}
    response: Any = None
    try:
        if provider_message_sink is not None and not callable(provider_message_sink):
            raise NativeProtocolError("native_message_sink_invalid")
        payload = deps.build_payload(
            bundle=bundle,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True,
            prompt_cache_key=prompt_cache_key,
            user_images=user_images,
            native_tools=native_tools,
            native_tool_choice=native_tool_choice,
            system_extra_blocks=system_extra_blocks,
            history_turns=history_turns,
            prompt_audit_sections=prompt_audit_sections,
            budget_context=budget_context,
            **({"native_history": native_history} if native_history is not None else {}),
        )
        from . import json_stream
        if json_stream.enabled(bundle):
            payload = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
            response = json_stream.consume(deps.dispatch_completion, bundle=bundle,
                                           payload=payload, attempt_budget=attempt_budget)
        else:
            response = deps.dispatch_completion(
                bundle=bundle, payload=payload, attempt_budget=attempt_budget)
        from .context_compaction import record_effective_input
        record_effective_input(payload)
        if provider_message_sink is not None:
            provider_message_sink(assistant_message(response))
        if not deps.record_cache(
            response,
            prompt_cache_key=prompt_cache_key,
            metric_suffix="chat" if bundle is deps.chat() else "aux",
        ):
            deps.add_metric("cache_usage_unavailable_calls")
        choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
        if choices:
            first = choices[0]
            note_finish_reason(first.get('finish_reason') if isinstance(first, dict) else getattr(first, 'finish_reason', None))
        content = deps.read_content_text(response)
        note_response_shape(content, transport="json")
        incomplete = next((
            (c.get("finish_reason") if isinstance(c, dict) else getattr(c, "finish_reason", None))
            for c in choices or [] if (c.get("finish_reason") if isinstance(c, dict) else getattr(c, "finish_reason", None))
            in {"length", "content_filter"}
        ), None)
        if incomplete:
            # Strict consumers need the finish signal even on compatibility
            # JSON paths. The task reply policy also stops parsing immediately.
            note_incomplete_response(incomplete)
            if plain_reply_enabled():
                return dict(fallback)
        native_tool_call = deps.read_native_tool_call(response)
        if native_tool_call is not None:
            note_response_shape(content, transport="json", native_call=native_tool_call)
            note_response("native_call")
            deps.add_metric("native_tool_malformed_call" if native_tool_call.get("_malformed") is True else "native_tool_call_extracted")
            return {NATIVE_TOOL_CALL_FIELD: native_tool_call, "tool_call": None}
        if deps.optional_is_malformed_native_tool_call()(response):
            note_response_shape(content, transport="json", native_call={"_malformed": True})
            note_response("native_call")
            deps.add_metric("native_tool_malformed_call")
            return {
                NATIVE_TOOL_CALL_FIELD: {
                    TOOL_SOURCE_FIELD: NATIVE_OPENAI,
                    "tool_calls": [],
                    "_malformed": True,
                    "protocol_error": "native_tool_arguments_malformed",
                },
                "tool_call": None,
            }
        if native_requested:
            deps.add_metric("native_tool_no_call")
        deps.note_truncated(response, phase="call_json")
        # 原生工具调用已在上方处理；只有没有工具且没有正文才属于空返回。
        if not content or isinstance(content, str) and not content.strip():
            note_response('empty_response')
        # 直接交付的自然语言正文可能引用快照/工具示例 JSON；不能从中挖出
        # 第一个对象当成决策。真正的原生调用已先处理，结构化根对象保持原路径。
        if (not incomplete and fallback.get('_agent_reply_delivery') is True
                and isinstance(content, str) and not content.lstrip().startswith(('{', '[', '```'))):
            prose = deps.recover_agent_plain_text(content, fallback=fallback)
            if prose is not None:
                note_response('plain_reply_candidate')
                deps.add_metric('chat_json_plain_text_recoveries')
                return prose
        # 完整停止的纯成稿可以恢复换行；工具/截断响应始终不走这个分支。
        parsed = recover_final_text(content) if not incomplete else None
        if parsed is None:
            parsed = deps.parse_json(content)
        if isinstance(parsed, dict):
            note_response("json_object")
            return parsed
        recovered = None if plain_reply_enabled() else deps.recover_partial(content, fallback=fallback)
        if isinstance(recovered, dict):
            note_response("partial_object")
            return recovered
        candidate = recover_plain_reply(content)
        if candidate is not None:
            deps.add_metric("chat_json_plain_text_recoveries")
            return candidate
        plain_text = deps.recover_agent_plain_text(content, fallback=fallback)
        if isinstance(plain_text, dict):
            note_response("plain_reply_candidate")
            deps.add_metric("chat_json_plain_text_recoveries")
            return plain_text
        deps.note_parse_failure(content, phase="call_json")
        note_response("parse_invalid")
    except (TaskBudgetExceeded, NativeProtocolError):
        raise
    except Exception as exc:
        from .response_protocol import note_call_failure
        note_call_failure("service_call", exc, response_received=response is not None)
        note_response("transport_error")
        deps.add_metric("errors")
        deps.note_error(exc, phase="call_json")
    finally:
        deps.record_context_budget_usage(budget_context, response)
    deps.add_metric("chat_json_fallbacks")
    return dict(fallback)


def recover_agent_plain_text(content: Any, *, fallback: dict[str, Any]) -> dict[str, Any] | None:
    """保留 Agent 的纯文本答复，兼容不尊重 JSON mode 的网关。

        Agent planner 的结构化外壳是 ``{"kind": "final", "text": ...}``，
        但部分 OpenAI 兼容网关即使收到 ``response_format=json_object``，仍会
        返回正常的一句文本。对记忆路由等其它结构化调用不能做这个推断，只有
        明确带 Agent ``kind`` 的 fallback 才允许降级为 final 文本。
        """

    if not isinstance(fallback, dict):
        return None
    kind = str(fallback.get("kind") or "").strip().lower()
    if kind not in {"final", "wait", "waiting_user"}:
        return None
    text = str(content or "").strip()
    if not text or text.startswith("{") or text.startswith("["):
        return None
    recovered = dict(fallback)
    direct_reply = recovered.pop('_agent_reply_delivery', False) is True
    # 直接交付宿主已接收完整停止的纯文本时，失败默认值并非模型的任务报告。
    # 记为部分交付，不伪造目标完成；保留正文，权限与执行结果仍由宿主回执决定。
    recovered["text"] = text if direct_reply else text[:4000]
    if direct_reply and recovered.get('task_outcome') == 'failed':
        recovered['task_outcome'] = 'partial'
    recovered.pop("_agent_transport_fallback", None)
    verify_plain_text = recovered.pop("_agent_plain_text_verify", False)
    if verify_plain_text is True and recovered.get("task_outcome") == "failed":
        # This failed value came from the transport-error default, not an
        # explicit model decision. Hosts opt in only when the recovered
        # answer still passes a source/coverage or action receipt gate.
        recovered.pop("task_outcome", None)
    return recovered
