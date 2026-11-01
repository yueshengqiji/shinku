"""Consume a model stream while keeping publication behind the host's gates."""
from __future__ import annotations

import time
from collections.abc import Mapping


def call_reply_model(model, *, trace_id: str = "", allow_stream: bool = True,
                     relevance_guard=None, **options):
    def stale_reason():
        return relevance_guard("reply") if callable(relevance_guard) else ""

    if reason := stale_reason():
        model.effects.trace(trace_id, "autonomous_relevance", status="suppressed", reason=reason)
        return dict(options["fallback"])
    stream_call = getattr(model.runtime, "stream_chat_json", None)
    if not allow_stream or not model.stream_enabled() or not callable(stream_call):
        response = model.runtime.call_chat_json(**options)
        return dict(options["fallback"]) if stale_reason() else response
    started = time.perf_counter()
    stream = stream_call(**options, early_tool_call_validator=lambda _call: False)
    count = 0
    result = None
    superseded = ""
    model.effects.trace(trace_id, "chat_stream_start", publication="after_host_validation")
    try:
        while True:
            superseded = stale_reason()
            if superseded:
                break
            try:
                next(stream)
            except StopIteration as done:
                result = done.value
                break
            count += 1
            # Model fragments can still become tool calls, be repaired, or fail
            # evidence review. Only the authoritative result may be published.
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    parsed = getattr(result, "parsed", None)
    failed = bool(getattr(result, "error", ""))
    model.effects.trace(trace_id, "chat_stream_complete", chunks=count,
                       first_chunk_ms=getattr(result, "first_chunk_ms", None),
                       transport_chunks=getattr(result, "transport_chunks", 0),
                       duration_ms=round((time.perf_counter() - started) * 1000, 1),
                       status="suppressed" if superseded else "failed" if failed else "completed",
                       reason=superseded,
                       rescued=bool(getattr(result, "rescue_succeeded", False)))
    if superseded or failed or not isinstance(parsed, Mapping):
        return dict(options["fallback"])
    return dict(parsed)
