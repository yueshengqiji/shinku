"""LLM 运行时的模型能力协商。

这一层只回答一个问题：当前供应商、协议和模型，能安全地接收哪些扩展
字段。传输、重试和回复解析留在 ``runtime_core``；审计与用量留在
``runtime_audit``。这样工具、图片和 thinking 的变化不会污染基础通道。
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from shinku.llm.runtime_core import (
    CONFIG_ALLOWLISTED_PROVIDER_TOOL_PROFILE,
    DEFAULT_PROVIDER_TOOL_PROFILE,
    ModelBundle,
    PROVIDER_TOOL_PROFILES,
    ProviderToolProfile,
    logger,
)
from shinku.tools.invocation import (
    NATIVE_OPENAI,
    TOOL_INVOCATION_ID_FIELD,
    TOOL_SOURCE_FIELD,
)


_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CALL_NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,80}")


class RuntimeCapabilitiesMixin:
    """原生工具、图片、响应格式和 thinking 的能力层。"""

    def _coerce_tools(self, value: Any) -> list[dict[str, Any]]:
        """把调用方工具描述收敛为 OpenAI function-tool 形状。"""
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        names: set[str] = set()
        for candidate in value[:64]:
            if not isinstance(candidate, dict) or str(candidate.get("type") or "").strip() != "function":
                continue
            function = candidate.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name or name in names or _TOOL_NAME_RE.fullmatch(name) is None:
                continue
            schema = function.get("parameters")
            if not isinstance(schema, dict):
                schema = {"type": "object", "additionalProperties": True}
            description = " ".join(str(function.get("description") or "").split())[:900]
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or f"Call Shinku tool {name}.",
                    "parameters": schema,
                },
            })
            names.add(name)
        return result

    def _coerce_tool_choice(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        text = str(value or "").strip().lower()
        return text if text in {"auto", "none", "required"} else ""

    def _wants_native_tools(self, bundle: ModelBundle) -> bool:
        return bool(self._tool_profile(bundle).supports_native_tools)

    def _tool_profile(self, bundle: ModelBundle) -> ProviderToolProfile:
        if self._protocol_of(bundle) != "openai":
            return DEFAULT_PROVIDER_TOOL_PROFILE
        host = self._bundle_host(bundle)
        model = str(getattr(bundle, "model", "") or "").strip().lower()
        known = PROVIDER_TOOL_PROFILES.get((host, model))
        return known if known is not None else self._configured_tool_profile(host=host, model=model)

    def _configured_tool_profile(self, *, host: str, model: str) -> ProviderToolProfile:
        allowed = str(self._env("NATIVE_TOOL_PROVIDER_ALLOWLIST", "") or "").strip()
        clean_host = str(host or "").strip().lower()
        clean_model = str(model or "").strip().lower()
        if not allowed or not clean_host or not clean_model:
            return DEFAULT_PROVIDER_TOOL_PROFILE
        for raw in allowed.split(","):
            parsed = self._parse_allowlist_item(raw)
            if parsed is None:
                continue
            item_host, item_model, coexist_json = parsed
            if item_host not in {"*", clean_host} or item_model not in {"*", clean_model}:
                continue
            if coexist_json:
                return ProviderToolProfile(
                    supports_native_tools=True,
                    native_tools_coexist_with_forced_json=True,
                    verified=False,
                    notes=(
                        "Enabled by NATIVE_TOOL_PROVIDER_ALLOWLIST with json coexistence. "
                        "Use only after probing the gateway/model."
                    ),
                )
            return CONFIG_ALLOWLISTED_PROVIDER_TOOL_PROFILE
        return DEFAULT_PROVIDER_TOOL_PROFILE

    def _parse_allowlist_item(self, value: Any) -> tuple[str, str, bool] | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if "://" in text:
            parsed = urlparse(text)
            host_part = parsed.netloc or parsed.path
            path_part = parsed.path.strip("/")
            text = f"{host_part}:{path_part}" if parsed.netloc and ":" in path_part else host_part
        fields = [field.strip() for field in text.split(":") if field.strip()]
        if not fields:
            return None
        host = self._normalize_allowlist_host(fields[0])
        model = fields[1] if len(fields) > 1 else "*"
        mode = fields[2] if len(fields) > 2 else ""
        return (host, model, mode in {"json", "response_json", "forced_json"}) if host and model else None

    def _normalize_allowlist_host(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text == "*":
            return text
        if "://" in text:
            text = urlparse(text).netloc
        text = text.split("/", 1)[0].rsplit("@", 1)[-1]
        return text.split(":", 1)[0].strip()

    def _bundle_host(self, bundle: ModelBundle) -> str:
        raw = str(getattr(bundle.client, "base_url", "") or "").strip()
        if raw and "://" not in raw:
            raw = f"https://{raw}"
        try:
            return str(urlparse(raw).hostname or "").strip().lower()
        except Exception:
            return ""

    def _protocol_of(self, bundle: ModelBundle) -> str:
        return str(
            getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", ""))
            or ""
        ).strip().lower()

    def _read_native_tool_call(self, response: Any) -> dict[str, Any] | None:
        try:
            message = response.choices[0].message
        except Exception:
            return None
        calls = self._value_of(message, "tool_calls")
        if not isinstance(calls, list) or not calls:
            return None
        if len(calls) > 1:
            self._add_metric("native_tool_calls_extra", len(calls) - 1)
        first = calls[0]
        function = self._value_of(first, "function")
        name = str(self._value_of(function, "name") or "").strip()
        if not name or _CALL_NAME_RE.fullmatch(name) is None:
            return None
        result = {
            **self._decode_tool_args(self._value_of(function, "arguments")),
            "type": name,
            TOOL_SOURCE_FIELD: NATIVE_OPENAI,
        }
        call_id = str(self._value_of(first, "id") or "").strip()
        if call_id:
            result[TOOL_INVOCATION_ID_FIELD] = call_id
        return result

    def _collect_stream_tool_parts(self, chunk: Any, parts: dict[int, dict[str, Any]]) -> None:
        try:
            choice = chunk.choices[0]
        except Exception:
            return
        calls = self._value_of(self._value_of(choice, "delta"), "tool_calls")
        if not isinstance(calls, list):
            return
        for offset, raw_call in enumerate(calls):
            try:
                index = int(self._value_of(raw_call, "index"))
            except (TypeError, ValueError):
                index = offset
            slot = parts.setdefault(index, {"arguments_parts": []})
            call_id = str(self._value_of(raw_call, "id") or "").strip()
            if call_id:
                slot["id"] = call_id
            function = self._value_of(raw_call, "function")
            name = str(self._value_of(function, "name") or "").strip()
            if name:
                slot["name"] = name
            fragment = self._value_of(function, "arguments")
            if fragment not in (None, ""):
                slot.setdefault("arguments_parts", []).append(str(fragment))

    def _assemble_stream_tool_call(self, parts: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
        if not parts:
            return None
        indexes = sorted(parts)
        if len(indexes) > 1:
            self._add_metric("native_tool_calls_extra", len(indexes) - 1)
        first = parts.get(indexes[0]) or {}
        name = str(first.get("name") or "").strip()
        if not name or _CALL_NAME_RE.fullmatch(name) is None:
            return None
        result = {
            **self._decode_tool_args("".join(str(piece) for piece in first.get("arguments_parts", []))),
            "type": name,
            TOOL_SOURCE_FIELD: NATIVE_OPENAI,
        }
        call_id = str(first.get("id") or "").strip()
        if call_id:
            result[TOOL_INVOCATION_ID_FIELD] = call_id
        return result

    def _decode_tool_args(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        raw = str(value or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _value_of(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _build_thinking_kwargs(self, *, bundle: ModelBundle) -> dict[str, Any]:
        mode = str(self._env("LLM_THINKING_MODE", "disabled") or "").strip().lower()
        if mode in {"", "default", "auto"} or mode not in {"enabled", "disabled"}:
            return {}
        if not self._supports_thinking_control(bundle):
            logger.warning(
                "thinking_control_skipped: mode=%s model=%s protocol=%s base_url=%s",
                mode,
                str(getattr(bundle, "model", "") or ""),
                self._protocol_of(bundle),
                str(getattr(bundle.client, "base_url", "") or ""),
            )
            return {}
        logger.debug("thinking_control_applied: mode=%s model=%s", mode, str(getattr(bundle, "model", "") or ""))
        return {"extra_body": {"thinking": {"type": mode}}}

    def _supports_thinking_control(self, bundle: ModelBundle) -> bool:
        if self._protocol_of(bundle) != "openai":
            return False
        model = str(getattr(bundle, "model", "") or "").strip().lower()
        if model.startswith("deepseek-"):
            return True
        host = self._bundle_host(bundle)
        return host == "api.deepseek.com" or host.endswith(".deepseek.com")

    def _coerce_image_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for candidate in value[:5]:
            if not isinstance(candidate, dict):
                continue
            url = str(candidate.get("data_url") or candidate.get("dataUrl") or candidate.get("url") or "").strip()
            if url.startswith("data:image/"):
                result.append({"type": "image_url", "image_url": {"url": url}})
        return result

    def _wants_response_json(self, bundle: ModelBundle) -> bool:
        if self._protocol_of(bundle) not in {"ollama", "openai"}:
            return False
        return bool(self._tool_profile(bundle).force_response_json)

    def _ensure_json_hint(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and "json" in content.lower():
                return
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "json" in str(part.get("text") or "").lower():
                        return
        if messages and str(messages[0].get("role") or "") == "system":
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "（本轮只输出一个合法的 JSON object，不要输出多余文字。）",
                },
            )

    def _is_official_openai(self, base_url: str) -> bool:
        raw = str(base_url or "").strip()
        if not raw:
            return True
        try:
            host = str(urlparse(raw).hostname or "").strip().lower()
        except Exception:
            return False
        return bool(host) and (host == "api.openai.com" or host.endswith(".openai.com"))
