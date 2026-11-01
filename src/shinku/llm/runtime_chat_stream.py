"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_chat_stream."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD

from .response_protocol import (
    note_finish_reason,
    note_incomplete_response,
    note_response,
    note_response_shape,
    plain_reply_enabled,
    recover_final_text,
    recover_plain_reply,
)
from .runtime_stream_parser import StreamTap
from .runtime_types import ChatJSONStreamResult, ModelBundle, logger


@dataclass(frozen=True)
class RuntimeChatStreamDependencies:
    """Bound callbacks and live getters for this responsibility."""
    add_metric: Callable[..., Any]
    assemble_stream_tool_call: Callable[..., Any]
    build_payload: Callable[..., Any]
    chat_reply_needs_rescue: Callable[..., Any]
    close_response: Callable[..., Any]
    coerce_tool_choice: Callable[..., Any]
    coerce_tools: Callable[..., Any]
    collect_stream_tool_parts: Callable[..., Any]
    dispatch_completion: Callable[..., Any]
    note_parse_failure: Callable[..., Any]
    parse_json: Callable[..., Any]
    read_stream_text: Callable[..., Any]
    record_cache: Callable[..., Any]
    record_context_budget_usage: Callable[..., Any]
    redact_secrets: Callable[..., Any]
    rescue_stream_via_json: Callable[..., Any]
    run_json_call: Callable[..., Any]
    scan_stream_tool_call: Callable[..., Any]
    runtime_policy: Callable[..., Any]
    snapshot_last_error: Callable[..., Any]


@dataclass(frozen=True)
class _ChatStreamRequest:
    """Original request values are shared by preflight, stream and recovery."""

    bundle: ModelBundle
    system_prompt: str
    user_prompt: str
    fallback: dict[str, Any]
    temperature: float
    early_tool_call_validator: Callable[[dict[str, Any]], bool] | None
    prompt_cache_key: str
    user_images: list[dict[str, Any]] | None
    native_tools: list[dict[str, Any]] | None
    native_tool_choice: Any
    system_extra_blocks: list[str] | None
    history_turns: list[dict[str, str]] | None
    prompt_audit_sections: list[dict[str, Any]] | None


@dataclass
class _ChatStreamProgress:
    """Per-call progress; emitted-output and cache facts survive reconnects."""

    error: str = ""
    raw_parts: list[str] = field(default_factory=list)
    native_tool_parts: dict[int, dict[str, Any]] = field(default_factory=dict)
    tap: StreamTap = field(default_factory=lambda: StreamTap())
    start_at: float = field(default_factory=lambda: time.perf_counter())
    request_start_at: float = field(default_factory=lambda: time.perf_counter())
    stopped_early: bool = False
    early_tool_call: dict[str, Any] | None = None
    tool_probe_disabled: bool = False
    yielded_any: bool = False
    cache_usage_reported: bool = False
    latest_usage_response: Any = None
    first_chunk_ms: float | None = None
    transport_chunks: int = 0
    finish_reason: str = ""


def stream_chat(
    *,
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
    temperature: float,
    early_tool_call_validator: Callable[[dict[str, Any]], bool] | None,
    prompt_cache_key: str,
    user_images: list[dict[str, Any]] | None=None,
    native_tools: list[dict[str, Any]] | None=None,
    native_tool_choice: Any='',
    system_extra_blocks: list[str] | None=None,
    history_turns: list[dict[str, str]] | None=None,
    prompt_audit_sections: list[dict[str, Any]] | None=None,
    deps: RuntimeChatStreamDependencies,
) -> Generator[dict[str, Any], None, ChatJSONStreamResult]:
    native_requested = bool(deps.coerce_tools(native_tools))
    # Preflight, reconnects, cache fallback and JSON recovery share one ceiling.
    attempt_budget = {"remaining": max(deps.runtime_policy().http_attempts, deps.runtime_policy().stream_attempts) + 1}
    request = _ChatStreamRequest(
        bundle=bundle,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback=fallback,
        temperature=temperature,
        early_tool_call_validator=early_tool_call_validator,
        prompt_cache_key=prompt_cache_key,
        user_images=user_images,
        native_tools=native_tools,
        native_tool_choice=native_tool_choice,
        system_extra_blocks=system_extra_blocks,
        history_turns=history_turns,
        prompt_audit_sections=prompt_audit_sections,
    )
    required_result = _try_required_tool_call(request, native_requested, attempt_budget, deps=deps)
    if required_result is not None:
        return required_result
    progress = _ChatStreamProgress()
    yield from _consume_chat_stream(request, progress, attempt_budget, deps=deps)
    return _finish_chat_stream(request, progress, native_requested, attempt_budget, deps=deps)


def _try_required_tool_call(request, native_requested, attempt_budget, *, deps):
    """Try the original required-tool non-stream request using the shared attempt budget."""
    normalized_tool_choice = deps.coerce_tool_choice(request.native_tool_choice)
    if native_requested and normalized_tool_choice == "required":
        try:
            non_stream = deps.run_json_call(
                bundle=request.bundle,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                fallback=request.fallback,
                temperature=request.temperature,
                prompt_cache_key=request.prompt_cache_key,
                user_images=request.user_images,
                native_tools=request.native_tools,
                native_tool_choice=request.native_tool_choice,
                system_extra_blocks=request.system_extra_blocks,
                history_turns=request.history_turns,
                prompt_audit_sections=request.prompt_audit_sections,
                attempt_budget=attempt_budget,
            )
            native_call = non_stream.get(NATIVE_TOOL_CALL_FIELD)
            if native_call is not None:
                return ChatJSONStreamResult(
                    parsed={NATIVE_TOOL_CALL_FIELD: native_call, "tool_call": None},
                    raw_text="",
                    elapsed_ms=0.0,
                    error="",
                    latest_emotion="",
                    latest_speech="",
                    latest_reply_medium="",
                    stopped_early=True,
                    early_tool_call=None,
                )
        except Exception as exc:
            error = str(exc or "").strip()
            deps.add_metric("errors")
            logger.error(
                "FORCE_TOOL non-stream fallback failed: %s", error[:200],
            )
    return None


def _consume_chat_stream(request, progress, attempt_budget, *, deps):
    """Reconnect only before output; usage accounting and response closure stay in finally."""
    max_stream_attempts = deps.runtime_policy().stream_attempts
    stream_attempt = 0
    while True:
        stream_attempt += 1
        response = None
        progress.error = ""
        progress.raw_parts = []
        progress.native_tool_parts = {}
        progress.tap = StreamTap()
        progress.start_at = time.perf_counter()
        progress.stopped_early = False
        progress.early_tool_call = None
        progress.tool_probe_disabled = False
        token_usage_seen: set[str] = set()
        budget_context: dict[str, Any] = {}
        progress.latest_usage_response: Any = None
        progress.finish_reason = ""
        try:
            response = deps.dispatch_completion(
                bundle=request.bundle,
                attempt_budget=attempt_budget,
                payload=deps.build_payload(
                    bundle=request.bundle,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    temperature=request.temperature,
                    stream=True,
                    json_mode=True,
                    prompt_cache_key=request.prompt_cache_key,
                    user_images=request.user_images,
                    native_tools=request.native_tools,
                    native_tool_choice=request.native_tool_choice,
                    system_extra_blocks=request.system_extra_blocks,
                    history_turns=request.history_turns,
                    prompt_audit_sections=request.prompt_audit_sections,
                    budget_context=budget_context,
                ),
            )
            yield from _read_stream_events(response, request, progress, token_usage_seen, deps=deps)
        except Exception as exc:
            progress.error = str(exc or "").strip()
            deps.add_metric("errors")
            # 只在还没产出任何内容（连接层失败）时重试一次
            if (stream_attempt < max_stream_attempts and attempt_budget["remaining"] > 0
                    and not progress.yielded_any and not progress.raw_parts and not progress.native_tool_parts):
                deps.add_metric("chat_stream_retries")
                if deps.runtime_policy().stream_retry_delay_seconds:
                    time.sleep(deps.runtime_policy().stream_retry_delay_seconds)
                continue
        finally:
            progress.cache_usage_reported = deps.record_cache(
                response,
                prompt_cache_key=request.prompt_cache_key,
                metric_suffix="chat",
                token_usage_seen=token_usage_seen,
            ) or progress.cache_usage_reported
            deps.record_context_budget_usage(budget_context, progress.latest_usage_response or response)
            deps.close_response(response)
        break


def _read_stream_events(response, request, progress, token_usage_seen, *, deps):
    """Consume chunks, account usage and stop only after a validated early tool call."""
    for chunk in response:
        progress.transport_chunks += 1
        choices = chunk.get("choices", []) if isinstance(chunk, dict) else getattr(chunk, "choices", [])
        for choice in choices or []:
            reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
            if reason:
                progress.finish_reason = str(reason)
        if progress.first_chunk_ms is None:
            progress.first_chunk_ms = round((time.perf_counter() - progress.request_start_at) * 1000, 1)
        if getattr(chunk, "usage", None) is not None or (isinstance(chunk, dict) and chunk.get("usage") is not None):
            progress.latest_usage_response = chunk
        progress.cache_usage_reported = deps.record_cache(
            chunk,
            prompt_cache_key=request.prompt_cache_key,
            metric_suffix="chat",
            token_usage_seen=token_usage_seen,
        ) or progress.cache_usage_reported
        deps.collect_stream_tool_parts(chunk, progress.native_tool_parts)
        text = deps.read_stream_text(chunk)
        if not text:
            continue
        progress.raw_parts.append(text)
        for event in progress.tap.feed(text):
            progress.yielded_any = True
            yield event
        if not progress.tool_probe_disabled:
            probe_state, probe_call = deps.scan_stream_tool_call("".join(progress.raw_parts))
            if probe_state == "object" and isinstance(probe_call, dict):
                if request.early_tool_call_validator is None or request.early_tool_call_validator(probe_call):
                    progress.early_tool_call = dict(probe_call)
                    progress.stopped_early = True
                    break
                progress.tool_probe_disabled = True
            elif probe_state in {"null", "none"}:
                progress.tool_probe_disabled = True
        if progress.stopped_early:
            break


def _parse_stream_reply(raw_text, progress, native_requested, *, deps):
    """Prefer native calls, then validated early calls, then the original JSON parser."""
    native_tool_call = deps.assemble_stream_tool_call(progress.native_tool_parts)
    note_response_shape(raw_text, transport="stream", native_call=native_tool_call)
    # 与非流式口径一致：工具调用没有正文是合法结果；只有两者皆无才记为空。
    if native_tool_call is None and not raw_text.strip():
        note_response("empty_response")
    if native_tool_call is not None:
        deps.add_metric("native_tool_malformed_call" if native_tool_call.get("_malformed") is True else "native_tool_call_extracted")
        parsed = {NATIVE_TOOL_CALL_FIELD: native_tool_call, "tool_call": None}
    elif native_requested:
        deps.add_metric("native_tool_no_call")
        parsed = recover_final_text(raw_text) if progress.finish_reason == "stop" else None
        if parsed is None:
            parsed = deps.parse_json(raw_text)
    elif progress.early_tool_call is not None:
        parsed = {"tool_call": progress.early_tool_call}
    else:
        parsed = recover_final_text(raw_text) if progress.finish_reason == "stop" else None
        if parsed is None:
            parsed = deps.parse_json(raw_text)
    return parsed


def _recover_stream_reply(request, progress, raw_text, parsed, attempt_budget, *, deps):
    """Recover within the same budget and discard partial tap state after authoritative rescue."""
    rescue_attempted = False
    rescue_succeeded = False
    if plain_reply_enabled():
        # The task host owns its single format repair. No hidden non-stream
        # rescue, partial-field reconstruction, or extra provider request here.
        if progress.error:
            note_response("transport_error")
            return dict(request.fallback), False, False
        if progress.finish_reason not in {"stop", "tool_calls"}:
            note_incomplete_response(progress.finish_reason)
            return dict(request.fallback), False, False
        if isinstance(parsed, dict):
            note_response("native_call" if NATIVE_TOOL_CALL_FIELD in parsed else "json_object")
            return dict(parsed), False, False
        candidate = recover_plain_reply(raw_text)
        if candidate is not None:
            deps.add_metric("chat_json_plain_text_recoveries")
            return candidate, False, False
        note_response("parse_invalid")
        return dict(request.fallback), False, False
    if deps.chat_reply_needs_rescue(parsed) and attempt_budget["remaining"] > 0:
        rescue_attempted = True
        rescued, rescue_error = deps.rescue_stream_via_json(
            bundle=request.bundle,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            prompt_cache_key=request.prompt_cache_key,
            user_images=request.user_images,
            native_tools=request.native_tools,
            native_tool_choice=request.native_tool_choice,
            system_extra_blocks=request.system_extra_blocks,
            history_turns=request.history_turns,
            prompt_audit_sections=request.prompt_audit_sections,
            attempt_budget=attempt_budget,
        )
        if rescued is not None:
            parsed = rescued
            rescue_succeeded = True
            # 非流式的完整响应是权威结果：失败流的半截 tap 状态不能渗回。
            progress.tap.latest_emotion = ""
            progress.tap.latest_speech = ""
            progress.tap.latest_reply_medium = ""
            progress.error = ""
        elif not progress.error and rescue_error:
            progress.error = rescue_error
    if not isinstance(parsed, dict):
        deps.add_metric("chat_json_fallbacks")
        deps.note_parse_failure(raw_text, phase="stream_chat_json")
        parsed = dict(request.fallback)
    else:
        parsed = dict(parsed)
    return parsed, rescue_attempted, rescue_succeeded


def _finish_chat_stream(request, progress, native_requested, attempt_budget, *, deps):
    """Normalize the final reply and return the original complete stream receipt."""
    if not progress.cache_usage_reported:
        deps.add_metric(
            "cache_usage_incomplete_streams" if progress.stopped_early else "cache_usage_unavailable_calls"
        )

    raw_text = "".join(progress.raw_parts)
    # 正常停止也须记录结束标记；仅在异常分支记它会使空 SSE 无法与损坏正文区分。
    note_finish_reason(progress.finish_reason)
    parsed = _parse_stream_reply(raw_text, progress, native_requested, deps=deps)
    parsed, rescue_attempted, rescue_succeeded = _recover_stream_reply(
        request, progress, raw_text, parsed, attempt_budget, deps=deps,
    )
    if not plain_reply_enabled() and progress.tap.latest_emotion and not parsed.get("emotion"):
        parsed["emotion"] = progress.tap.latest_emotion
    if not plain_reply_enabled() and progress.tap.latest_speech and not parsed.get("speech"):
        parsed["speech"] = progress.tap.latest_speech
    if not plain_reply_enabled() and progress.tap.latest_reply_medium and not parsed.get("reply_medium"):
        parsed["reply_medium"] = progress.tap.latest_reply_medium

    elapsed_ms = round((time.perf_counter() - progress.start_at) * 1000, 1)
    return ChatJSONStreamResult(
        parsed=parsed,
        raw_text=raw_text,
        elapsed_ms=elapsed_ms,
        error=progress.error,
        latest_emotion=progress.tap.latest_emotion,
        latest_speech=progress.tap.latest_speech,
        latest_reply_medium=progress.tap.latest_reply_medium,
        stopped_early=progress.stopped_early,
        early_tool_call=progress.early_tool_call,
        rescue_attempted=rescue_attempted,
        rescue_succeeded=rescue_succeeded,
        first_chunk_ms=progress.first_chunk_ms,
        transport_chunks=progress.transport_chunks,
    )


def rescue_stream_via_json(
    *,
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    prompt_cache_key: str,
    user_images: list[dict[str, Any]] | None,
    native_tools: list[dict[str, Any]] | None,
    native_tool_choice: Any,
    system_extra_blocks: list[str] | None,
    history_turns: list[dict[str, str]] | None,
    prompt_audit_sections: list[dict[str, Any]] | None,
    attempt_budget: dict[str, int] | None=None,
    deps: RuntimeChatStreamDependencies,
) -> tuple[dict[str, Any] | None, str]:
    """流式废了就用一次完整的非流式请求把它救回来。

        只在流式产出不可用时跑一次。用哨兵 fallback 区分「真的解析成功」
        和 ``_run_json_call`` 自己的兜底，不暴露实现字段。
        """

    deps.add_metric("chat_stream_rescue_attempts")
    sentinel_key = "__shinku_stream_rescue_fallback__"
    try:
        rescued = deps.run_json_call(
            bundle=bundle,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback={sentinel_key: True},
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            user_images=user_images,
            native_tools=native_tools,
            native_tool_choice=native_tool_choice,
            system_extra_blocks=system_extra_blocks,
            history_turns=history_turns,
            prompt_audit_sections=prompt_audit_sections,
            attempt_budget=attempt_budget,
        )
    except Exception as exc:  # pragma: no cover - 防御：_run_json_call 通常自己吞错
        deps.add_metric("chat_stream_rescue_failures")
        return None, deps.redact_secrets(str(exc or ""))

    if isinstance(rescued, dict) and sentinel_key not in rescued:
        deps.add_metric("chat_stream_rescue_successes")
        return dict(rescued), ""

    deps.add_metric("chat_stream_rescue_failures")
    detail = deps.snapshot_last_error()
    return None, str(detail.get("message") or "stream_json_rescue_failed")
