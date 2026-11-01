"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_ndjson_call."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .runtime_types import ModelBundle, NDJSONCallResult


@dataclass(frozen=True)
class RuntimeNdjsonCallDependencies:
    """Bound callbacks and live getters for this responsibility."""
    add_metric: Callable[..., Any]
    build_payload: Callable[..., Any]
    close_response: Callable[..., Any]
    dispatch_completion: Callable[..., Any]
    drain_buffer: Callable[..., Any]
    parse_line: Callable[..., Any]
    read_stream_text: Callable[..., Any]
    record_cache: Callable[..., Any]
    record_context_budget_usage: Callable[..., Any]


def run_ndjson_call(
    *,
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    on_event: Callable[[dict[str, Any]], bool] | None,
    temperature: float,
    prompt_cache_key: str,
    deps: RuntimeNdjsonCallDependencies,
) -> NDJSONCallResult:
    events: list[dict[str, Any]] = []
    event_timings: list[dict[str, Any]] = []
    response: Any = None
    buffer = ""
    start_at = time.perf_counter()
    stopped_early = False
    cache_usage_reported = False
    token_usage_seen: set[str] = set()
    stop_event_type = ""
    error = ""
    budget_context: dict[str, Any] = {}
    latest_usage_response: Any = None
    try:
        response = deps.dispatch_completion(
            bundle=bundle,
            payload=deps.build_payload(
                bundle=bundle,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                stream=True,
                prompt_cache_key=prompt_cache_key,
                budget_context=budget_context,
            ),
        )
        for chunk in response:
            if getattr(chunk, "usage", None) is not None or (isinstance(chunk, dict) and chunk.get("usage") is not None):
                latest_usage_response = chunk
            cache_usage_reported = deps.record_cache(
                chunk,
                prompt_cache_key=prompt_cache_key,
                metric_suffix="aux",
                token_usage_seen=token_usage_seen,
            ) or cache_usage_reported
            text = deps.read_stream_text(chunk)
            if not text:
                continue
            buffer += text
            buffer, parsed_events = deps.drain_buffer(buffer)
            for event in parsed_events:
                events.append(event)
                event_type = str(event.get("type") or "").strip().lower()
                event_timings.append(
                    {
                        "index": len(events) - 1,
                        "type": event_type or "unknown",
                        "elapsed_ms": round((time.perf_counter() - start_at) * 1000, 1),
                    }
                )
                if on_event and on_event(event):
                    stopped_early = True
                    stop_event_type = event_type or "unknown"
                    break
            if stopped_early:
                break

        tail_event = deps.parse_line(buffer) if not stopped_early else None
        if tail_event is not None:
            events.append(tail_event)
            event_type = str(tail_event.get("type") or "").strip().lower()
            event_timings.append(
                {
                    "index": len(events) - 1,
                    "type": event_type or "unknown",
                    "elapsed_ms": round((time.perf_counter() - start_at) * 1000, 1),
                }
            )
            if on_event:
                if on_event(tail_event):
                    stopped_early = True
                    stop_event_type = event_type or "unknown"
    except Exception as exc:
        error = str(exc or "").strip()
        deps.add_metric("errors")
    finally:
        deps.record_context_budget_usage(budget_context, latest_usage_response or response)
        deps.close_response(response)
    if not cache_usage_reported:
        deps.add_metric(
            "cache_usage_incomplete_streams" if stopped_early else "cache_usage_unavailable_calls"
        )
    elapsed_ms = round((time.perf_counter() - start_at) * 1000, 1)
    return NDJSONCallResult(
        events=events,
        event_timings=event_timings,
        elapsed_ms=elapsed_ms,
        stopped_early=stopped_early,
        completed_stream=(not stopped_early and not error),
        stop_event_type=stop_event_type,
        error=error,
    )
