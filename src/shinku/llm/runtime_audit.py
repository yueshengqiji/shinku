"""LLM 运行时的审计、用量和缓存观测层。

本模块不保存提示词明文：审计记录只保留分段大小、估算 token 和摘要。
供应商的 usage 字段则统一为 chat/aux 两套计数，并处理流式响应重复上报。
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shinku.llm.runtime_core import ModelBundle


_AUDIT_LOCK = threading.RLock()


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


class RuntimeAuditMixin:
    """prompt 审计与 token / 缓存用量度量。"""

    def _record_cache(
        self,
        response: Any,
        *,
        prompt_cache_key: str = "",
        token_usage_seen: set[str] | None = None,
    ) -> bool:
        try:
            usage = getattr(response, "usage", None)
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            if usage is None:
                return False

            suffix = "chat" if str(prompt_cache_key or "").strip().startswith("chat:") else "aux"
            self._record_tokens(usage, metric_suffix=suffix, seen=token_usage_seen)

            read_present, read = self._read_usage_field(usage, "cache_read_input_tokens")
            create_present, created = self._read_usage_field(usage, "cache_creation_input_tokens")
            if not read_present:
                read_present, read = self._read_usage_field(usage, "prompt_cache_hit_tokens")
            if not create_present:
                create_present, created = self._read_usage_field(usage, "prompt_cache_miss_tokens")

            self._add_metric("cache_usage_reports")
            self._add_metric(f"cache_usage_reports_{suffix}")
            if not read_present and not create_present:
                self._add_metric("cache_fields_unavailable_reports")
                return True
            if read_present:
                self._add_metric("cache_read_tokens", read)
                self._add_metric(f"cache_read_tokens_{suffix}", read)
                if read == 0:
                    self._add_metric("cache_read_zero_reports")
            if create_present:
                self._add_metric("cache_creation_tokens", created)
                self._add_metric(f"cache_creation_tokens_{suffix}", created)
                if created == 0:
                    self._add_metric("cache_creation_zero_reports")
            return True
        except Exception:
            return False

    def _record_tokens(
        self,
        usage: Any,
        *,
        metric_suffix: str,
        seen: set[str] | None = None,
    ) -> bool:
        aliases = {
            "input_tokens": ("prompt_tokens", "input_tokens", "promptTokenCount"),
            "output_tokens": ("completion_tokens", "output_tokens", "candidatesTokenCount"),
            "total_tokens": ("total_tokens", "totalTokenCount"),
        }
        values: dict[str, int] = {}
        present: list[str] = []
        for target, candidates in aliases.items():
            for candidate in candidates:
                found, value = self._read_usage_field(usage, candidate)
                if found:
                    values[target] = value
                    present.append(f"{target}:{value}")
                    break
        if not values:
            return False
        if "total_tokens" not in values and {"input_tokens", "output_tokens"}.issubset(values):
            values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
            present.append(f"total_tokens:{values['total_tokens']}")
        fingerprint = "|".join(sorted(present))
        if seen is not None:
            if fingerprint in seen:
                return True
            seen.add(fingerprint)
        self._add_metric("token_usage_reports")
        self._add_metric(f"token_usage_reports_{metric_suffix}")
        for key, value in values.items():
            self._add_metric(key, value)
            self._add_metric(f"{key}_{metric_suffix}", value)
        return True

    def _read_usage_field(self, usage: Any, key: str) -> tuple[bool, int]:
        if isinstance(usage, dict):
            if key not in usage:
                return False, 0
            return True, self._coerce_int(usage, key)
        if not hasattr(usage, key):
            return False, 0
        return True, self._coerce_int(usage, key)

    def _coerce_int(self, usage: Any, key: str) -> int:
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _maybe_record_audit(
        self,
        *,
        bundle: ModelBundle,
        prompt_cache_key: str,
        messages: list[dict[str, Any]],
        system_extra_blocks: list[str],
        history_turns: list[dict[str, str]] | None,
        user_prompt: str,
        user_image_count: int,
        prompt_audit_sections: list[dict[str, Any]] | None,
        stream: bool,
        json_mode: bool,
        native_tool_count: int,
    ) -> None:
        if not self._audit_enabled(prompt_cache_key):
            return
        try:
            sections = self._build_audit_sections(
                messages=messages,
                system_extra_blocks=system_extra_blocks,
                history_turns=history_turns,
                user_prompt=user_prompt,
            )
            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "prompt_cache_key": str(prompt_cache_key or ""),
                "model": str(getattr(bundle, "model", "") or ""),
                "protocol": str(
                    getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", ""))
                    or ""
                ),
                "stream": bool(stream),
                "json_mode": bool(json_mode),
                "message_count": len(messages),
                "history_turn_count": len(history_turns or []),
                "user_image_count": max(0, int(user_image_count or 0)),
                "native_tool_count": max(0, int(native_tool_count or 0)),
                "payload_totals": self._sum_sections(sections),
                "payload_sections": sections,
                "source_sections": self._coerce_audit_sections(prompt_audit_sections),
            }
            self._write_audit_record(record)
        except Exception:
            return

    def _audit_enabled(self, prompt_cache_key: str) -> bool:
        if not _flag(self._env("LLM_PROMPT_AUDIT_ENABLED", "0"), False):
            return False
        key = str(prompt_cache_key or "").strip()
        if key == "chat:final":
            return True
        return _flag(self._env("LLM_PROMPT_AUDIT_INCLUDE_AUX", "0"), False)

    def _build_audit_sections(
        self,
        *,
        messages: list[dict[str, Any]],
        system_extra_blocks: list[str],
        history_turns: list[dict[str, str]] | None,
        user_prompt: str,
    ) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        system_index = 0
        for message in messages:
            if str(message.get("role") or "").strip().lower() != "system":
                continue
            name = "payload.system_message" if system_index == 0 else f"payload.system_message.{system_index}"
            sections.append(self._audit_section(name, self._flatten_content(message.get("content"))))
            system_index += 1
        history_text = "\n".join(
            f"{str(turn.get('role') or '').strip().lower()}:{str(turn.get('content') or '').strip()}"
            for turn in history_turns or []
            if str(turn.get("content") or "").strip()
        )
        sections.append(self._audit_section("payload.history_turns", history_text))
        sections.append(self._audit_section("payload.user_prompt", user_prompt))
        if system_extra_blocks:
            sections.append(self._audit_section("payload.system_extra_blocks", "\n\n".join(system_extra_blocks)))
        return sections

    def _coerce_audit_sections(self, sections: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not isinstance(sections, list):
            return normalized
        for index, item in enumerate(sections):
            if isinstance(item, dict):
                name = str(item.get("name") or f"section_{index}").strip() or f"section_{index}"
                text = item.get("text", "")
            else:
                name = f"section_{index}"
                text = item
            normalized.append(self._audit_section(name, self._flatten_content(text)))
        return normalized

    def _audit_section(self, name: str, text: str) -> dict[str, Any]:
        raw = str(text or "")
        digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16] if raw else ""
        return {
            "name": str(name or "section"),
            "chars": len(raw),
            "estimated_tokens": self._estimate_tokens(raw),
            "sha256_16": digest,
            "empty": not bool(raw),
        }

    def _estimate_tokens(self, text: str) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        cjk = sum(1 for char in raw if "\u4e00" <= char <= "\u9fff")
        other = max(0, len(raw) - cjk)
        return int(cjk + ((other + 3) // 4))

    def _sum_sections(self, sections: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "chars": sum(int(section.get("chars") or 0) for section in sections),
            "estimated_tokens": sum(int(section.get("estimated_tokens") or 0) for section in sections),
        }

    def _write_audit_record(self, record: dict[str, Any]) -> None:
        root = Path(str(self._env("LOG_DIR", "logs") or "logs"))
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = root / "llm_prompt_audit" / f"{day}.jsonl"
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _AUDIT_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _build_cache_kwargs(
        self,
        *,
        bundle: ModelBundle,
        prompt_cache_key: str,
    ) -> dict[str, Any]:
        if not self._wants_cache_hints(bundle):
            return {}
        result: dict[str, Any] = {}
        key = self._coerce_cache_key(prompt_cache_key)
        retention = self._coerce_cache_retention(self._env("PROMPT_CACHE_RETENTION", ""))
        if key:
            result["prompt_cache_key"] = key
        if retention:
            result["prompt_cache_retention"] = retention
        return result

    def _wants_cache_hints(self, bundle: ModelBundle) -> bool:
        if not _flag(self._env("PROMPT_CACHE_HINTS_ENABLED", "1"), True):
            return False
        protocol = str(
            getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", ""))
            or ""
        ).strip().lower()
        if protocol != "openai":
            return False
        if _flag(self._env("PROMPT_CACHE_HINTS_FORCE", "0"), False):
            return True
        return self._is_official_openai(str(getattr(bundle.client, "base_url", "") or ""))

    def _coerce_cache_key(self, prompt_cache_key: Any) -> str:
        raw = str(prompt_cache_key or "").strip().strip(":")
        if not raw:
            return ""
        namespace = str(self._env("PROMPT_CACHE_NAMESPACE", "shinku") or "").strip().strip(":")
        return f"{namespace}:{raw}" if namespace else raw

    def _coerce_cache_retention(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        return raw if raw in {"in_memory", "24h"} else ""
