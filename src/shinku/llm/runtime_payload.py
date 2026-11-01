"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_payload."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from shinku.llm.context_budget import ContextBudgetPolicy, assess_and_trim_payload

from .native_messages import (
    NativeProtocolError,
    validate_native_messages,
    without_reasoning,
)
from .output_budget import apply_output_budget, current_output_budget
from .runtime_types import ModelBundle


@dataclass(frozen=True)
class RuntimePayloadDependencies:
    """Bound callbacks and live getters for this responsibility."""
    add_metric: Callable[..., Any]
    build_cache_kwargs: Callable[..., Any]
    build_thinking_kwargs: Callable[..., Any]
    coerce_image_items: Callable[..., Any]
    coerce_system_blocks: Callable[..., Any]
    coerce_tool_choice: Callable[..., Any]
    coerce_tools: Callable[..., Any]
    ensure_json_hint: Callable[..., Any]
    is_anthropic: Callable[..., Any]
    maybe_record_audit: Callable[..., Any]
    record_context_budget: Callable[..., Any]
    supports_multi_system: Callable[..., Any]
    supports_thinking_control: Callable[..., Any]
    tool_profile: Callable[..., Any]
    wants_response_json: Callable[..., Any]
    wants_stream_usage: Callable[..., Any]
    chat: Callable[..., Any]


def build_payload(
    *,
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool=False,
    json_mode: bool=False,
    prompt_cache_key: str='',
    user_images: list[dict[str, Any]] | None=None,
    native_tools: list[dict[str, Any]] | None=None,
    native_tool_choice: Any='',
    system_extra_blocks: list[str] | None=None,
    history_turns: list[dict[str, str]] | None=None,
    prompt_audit_sections: list[dict[str, Any]] | None=None,
    budget_context: dict[str, Any] | None=None,
    native_history: list[dict[str, Any]] | None=None,
    deps: RuntimePayloadDependencies,
) -> dict[str, Any]:
    user_content: str | list[dict[str, Any]]
    image_items = deps.coerce_image_items(user_images)
    if image_items:
        user_content = [{"type": "text", "text": user_prompt}, *image_items]
    else:
        user_content = user_prompt
    filtered_system_extra_blocks = deps.coerce_system_blocks(system_extra_blocks)
    effective_system_prompt = str(system_prompt or "")
    messages: list[dict[str, Any]] = [{"role": "system", "content": effective_system_prompt}]
    if filtered_system_extra_blocks and deps.supports_multi_system(bundle):
        messages.extend(
            {"role": "system", "content": block}
            for block in filtered_system_extra_blocks
        )
    elif filtered_system_extra_blocks and not deps.is_anthropic(bundle):
        # 部分兼容端点只认一条 system 消息：保留拼接回退。
        messages[0]["content"] = "\n\n".join(
            part for part in [effective_system_prompt.strip(), *filtered_system_extra_blocks] if part
        )
    for turn in history_turns or []:
        role = str(turn.get("role", "") or "").strip().lower()
        content = str(turn.get("content", "") or "").strip()
        if content and role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    current_message = messages[-1]
    if native_history is not None:
        if deps.is_anthropic(bundle):
            raise NativeProtocolError("native_history_protocol_unsupported")
        messages.extend(validate_native_messages(native_history))
    payload: dict[str, Any] = {
        "model": bundle.model,
        "temperature": temperature,
        "messages": messages,
    }
    if stream:
        payload["stream"] = True
        if deps.wants_stream_usage(bundle):
            payload["stream_options"] = {"include_usage": True}
    normalized_tools = deps.coerce_tools(native_tools)
    native_tool_profile = deps.tool_profile(bundle)
    should_send_native_tools = bool(normalized_tools and native_tool_profile.supports_native_tools)
    # 强制工具调用与 JSON 响应格式互斥：后者会把模型的输出引向
    # 纯 JSON 正文，挤掉工具调用通道。
    normalized_tool_choice = deps.coerce_tool_choice(native_tool_choice)
    force_tool_call = bool(
        normalized_tools and should_send_native_tools and (normalized_tool_choice == 'required'),
    )
    should_use_response_json = bool(
        json_mode
        and deps.wants_response_json(bundle)
        and not force_tool_call
    )
    if json_mode and should_send_native_tools and not force_tool_call:
        if native_tool_profile.native_tools_coexist_with_forced_json:
            should_use_response_json = True
        else:
            should_use_response_json = False
            deps.add_metric("native_tool_forced_json_suppressed")
    if should_use_response_json:
        payload["response_format"] = {"type": "json_object"}
        deps.ensure_json_hint(messages)
    if normalized_tools and should_send_native_tools:
        payload["tools"] = normalized_tools
        deps.add_metric("native_tool_decision_sent")
        if normalized_tool_choice:
            payload["tool_choice"] = normalized_tool_choice
    elif normalized_tools:
        deps.add_metric("native_tool_provider_unsupported")
    payload.update(deps.build_thinking_kwargs(bundle=bundle))
    payload.update(
        deps.build_cache_kwargs(
            bundle=bundle,
            prompt_cache_key=prompt_cache_key,
        )
    )
    if filtered_system_extra_blocks and deps.is_anthropic(bundle):
        payload["system_extra_blocks"] = filtered_system_extra_blocks
    # 只在明确绑定的独立Agent中启用，先压缩完整payload再计量/派发；旧原生链不改写。
    from .context_compaction import compact_payload
    from .cache_prefix import summary_payload
    # 只有宿主内部摘要调用可以覆盖封装；它复用原SDK配置，原件不变，仍经过
    # 同一输出约束、输入窗口检查、传输、预算和用量账本，不直接调用客户端。
    replay = summary_payload(bundle)
    if replay is not None:
        payload = replay
    current_index = next(i for i, message in enumerate(messages) if message is current_message)
    payload = compact_payload(payload, bundle, current_index)
    payload = apply_output_budget(payload)
    budget_policy = ContextBudgetPolicy.from_environment()
    from .context_compaction import compaction_enabled
    from .model_limits import model_limits
    if compaction_enabled() or replay is not None:
        budget_policy = replace(budget_policy, window_tokens=model_limits(bundle)['window_tokens'])
    output_budget = current_output_budget()
    if output_budget is not None and (payload.get('max_completion_tokens') or payload.get('max_tokens')):
        # 输入窗口须预留本次结构化输出；不能仍按通用短答的2048估量。
        budget_policy = replace(budget_policy, output_reserve_tokens=max(
            budget_policy.output_reserve_tokens, payload.get('max_completion_tokens') or payload['max_tokens']))
    if budget_policy.mode != "off":
        payload, budget_record = assess_and_trim_payload(payload, budget_policy)
        if ((native_history is not None or replay is not None) and budget_policy.mode == "enforce"
                and budget_policy.context_window(bundle.model)
                and budget_record["before"]["total"] > budget_policy.input_limit(bundle.model)):
            raise NativeProtocolError("native_context_budget_exceeded")
        budget_record["channel"] = "chat" if bundle is deps.chat() else "aux"
        recorded = deps.record_context_budget(budget_record)
        if budget_context is not None:
            budget_context.update(recorded)
        messages = payload["messages"]
        if budget_record["status"] == "trimmed_history":
            history_turns = [
                {"role": item["role"], "content": item["content"]}
                for item in messages[:-1]
                if item.get("role") in {"user", "assistant"}
            ]
    messages = payload['messages']
    deps.maybe_record_audit(
        bundle=bundle,
        prompt_cache_key=prompt_cache_key,
        messages=without_reasoning(messages) if native_history is not None or replay is not None else messages,
        system_extra_blocks=filtered_system_extra_blocks if deps.is_anthropic(bundle) else [],
        history_turns=history_turns,
        user_prompt=user_prompt,
        user_image_count=len(image_items),
        prompt_audit_sections=prompt_audit_sections,
        stream=stream,
        json_mode=json_mode,
        native_tool_count=len(normalized_tools),
    )
    return payload


def coerce_system_blocks(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def is_anthropic(bundle: ModelBundle) -> bool:
    protocol = str(getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", "")) or "").strip().lower(
    )
    return protocol == "anthropic"


def supports_multi_system(bundle: ModelBundle) -> bool:
    """所选协议能不能吃多条 system 消息。

        OpenAI 兼容端（含 DeepSeek、Ollama）可以保住第一条 system 稳定、
        后面再补运行期规则。未知协议沿用单条回退。
        """
    protocol = str(
        getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", ""))
        or ""
    ).strip().lower()
    return protocol in {"openai", "ollama"}


def wants_stream_usage(bundle: ModelBundle, *, deps: RuntimePayloadDependencies) -> bool:
    return deps.supports_thinking_control(bundle)
