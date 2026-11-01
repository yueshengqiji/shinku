"""One-shot, no-tool expression transport using the existing runtime adapter.

No provider is selected or instantiated here. Hosts inject an already approved
runtime; Gemini protocol/quality and spend approval remain separate gates.
"""
from __future__ import annotations

import json
import re

from shinku.agent.budget import bind_task_budget
from shinku.agent_runtime.contracts import canonical_hash
from shinku.result_expression.models import ModelReply
from .response_protocol import note_call_failure
from .runtime_transport import completion_deadline
from .runtime_types import ModelBundle


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def response_usage(response):
    source = _field(response, "usage")
    output = {}
    for name, aliases in {
        "input_tokens": ("prompt_tokens", "input_tokens", "promptTokenCount"),
        "output_tokens": ("completion_tokens", "output_tokens", "candidatesTokenCount"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
    }.items():
        for alias in aliases:
            count = _field(source, alias)
            if type(count) is int and count >= 0:
                output[name] = count
                break
    if "total_tokens" not in output and {"input_tokens", "output_tokens"}.issubset(output):
        output["total_tokens"] = output["input_tokens"] + output["output_tokens"]
    # Token reports are not a price table. Without an explicit invoice/report
    # the expression archive preserves unknown monetary cost.
    return output


class RuntimeExpressionAdapter:
    def __init__(self, runtime, *, channel="chat"):
        if channel not in {"chat", "aux"}:
            raise ValueError("invalid_expression_channel")
        self.runtime, self.channel = runtime, channel

    @property
    def identity(self):
        bundle = getattr(self.runtime, self.channel)
        return canonical_hash({"channel": self.channel, "model": bundle.model,
                               "endpoint": str(getattr(bundle.client, "base_url", ""))})

    def call_expression_json(self, *, system_prompt, user_prompt, native_tools=None,
                             timeout_seconds=10, max_output_tokens=2048):
        if native_tools is not None:
            raise ValueError("expression_tools_forbidden")
        runtime = self.runtime
        phase, response, usage, finish = "configuration", None, {}, "unknown"
        def note_failure(exc, *, at=None, secondary=False):
            note_call_failure(at or phase, exc, response_received=response is not None,
                              finish_reason=finish, usage=usage, secondary=secondary)
        try:
            configured = getattr(runtime, self.channel)
            client = configured.client
            if getattr(client, "max_retries", 0) != 0:
                clone = getattr(client, "with_options", None)
                if not callable(clone):
                    raise ValueError("expression_transport_retry_policy_unknown")
                client = clone(max_retries=0)
            bundle = ModelBundle(client, configured.model)
            budget_context = {}
            with bind_task_budget(None), completion_deadline(timeout_seconds):
                phase = "payload"
                payload = runtime._build_payload(bundle=bundle, system_prompt=system_prompt,
                    user_prompt=user_prompt, temperature=.3, json_mode=True, prompt_cache_key="",
                    native_tools=None, native_tool_choice="", history_turns=None,
                    user_images=None, system_extra_blocks=None, budget_context=budget_context)
                # Context trimming must not silently drop canonical facts.
                if (payload.get("tools") or payload.get("tool_choice") or payload.get("stream")
                        or payload["messages"][-1].get("content") != user_prompt
                        or not str(payload["messages"][0].get("content", "")).startswith(system_prompt)):
                    raise ValueError("expression_payload_binding_changed")
                payload["max_tokens"] = max_output_tokens
                runtime._add_metric("expression_json_calls")
                failed = False
                try:
                    phase = "dispatch"
                    response = runtime._dispatch_completion(bundle=bundle, payload=payload, attempt_budget={"remaining": 1})
                    phase = "response_usage"
                    usage = response_usage(response)
                    choices = _field(response, "choices", []) or []
                    if choices:
                        finish = _field(choices[0], "finish_reason", "unknown")
                    phase = "cache"
                    runtime._record_cache(response, prompt_cache_key="", metric_suffix="expression")
                    phase = "response"
                    if runtime._read_native_tool_call(response) is not None:
                        return ModelReply(None, usage, "expression_native_tool_call_forbidden")
                    for choice in choices:
                        if _field(choice, "finish_reason") == "length":
                            return ModelReply(None, usage, "expression_output_truncated")
                        if _field(choice, "finish_reason") == "content_filter":
                            return ModelReply(None, usage, "expression_output_rejected")
                    content = runtime._read_content_text(response)
                    phase = "parse"
                    # Only one complete, standalone JSON fence is wrapping.
                    wrapped = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content.strip(), re.S | re.I)
                    if wrapped is not None:
                        content = wrapped[1]
                    try:
                        parsed = json.loads(content)
                    except (TypeError, ValueError):
                        reason = "expression_wrapping_invalid" if isinstance(content, str) and "```" in content else "expression_json_invalid"
                        return ModelReply(None, usage, reason)
                    if not isinstance(parsed, dict):
                        return ModelReply(None, usage, "expression_protocol_invalid")
                    return ModelReply(parsed, usage)
                except Exception as exc:
                    failed = True
                    note_failure(exc)
                    raise
                finally:
                    try:
                        runtime._record_context_budget_usage(budget_context, response)
                    except Exception as exc:
                        note_failure(exc, at="context_usage", secondary=failed)
                        # A bookkeeping exception must not replace the actual
                        # transport failure. A sole bookkeeping failure still
                        # rejects the candidate, as on the existing path.
                        if not failed:
                            raise
        except Exception as exc:
            note_failure(exc)
            raise
