"""LLM 调用引擎 —— 审计与用量度量 mixin（C2-5a 阶段为最小实现）。

本模块承载 ``LLMRuntime`` 的「观测」面：

- prompt 审计记录（分段字数 / 估算 token / 摘要，不含明文 prompt）；
- token 用量记录（input / output / total，含 chat / aux 双口径）；
- 缓存用量记录（Anthropic 与 DeepSeek 两套字段名）；
- prompt cache 提示（key / retention）的组装。

**C2-5a 状态：** 这里只放「审计关闭、无 usage 上报、不带缓存提示」
路径所需的最小实现——即各方法在默认环境下的等价返回。usage 别名表、
去重指纹、缓存字段双口径与审计落盘的完整实现与测试面属 C2-5c 批次，
届时逐方法替换本文件的桩体。
"""

from __future__ import annotations

from typing import Any

from shinku.llm.runtime_core import ModelBundle


class RuntimeAuditMixin:
    """prompt 审计与 token / 缓存用量度量（C2-5a 桩；完整实现见 C2-5c）。"""

    def _record_cache(
        self,
        response: Any,
        *,
        prompt_cache_key: str = "",
        token_usage_seen: set[str] | None = None,
    ) -> bool:
        # 桩：响应不带 usage 时等价返回 False（完整版见 C2-5c）。
        return False

    def _record_tokens(
        self,
        usage: Any,
        *,
        metric_suffix: str,
        seen: set[str] | None = None,
    ) -> bool:
        # 桩：无可用字段时等价返回 False（完整版见 C2-5c）。
        return False

    def _read_usage_field(self, usage: Any, key: str) -> tuple[bool, int]:
        # 桩：字段缺失时等价返回（完整版见 C2-5c）。
        return False, 0

    def _coerce_int(self, usage: Any, key: str) -> int:
        # 桩：占位（完整版见 C2-5c）。
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
        # 桩：审计开关关闭时等价 no-op（完整版见 C2-5c）。
        if not self._audit_enabled(prompt_cache_key):
            return
        return None

    def _audit_enabled(self, prompt_cache_key: str) -> bool:
        # 桩：开关未设时等价返回 False（完整版含 chat:final 与 INCLUDE_AUX 例外，C2-5c）。
        return False

    def _build_audit_sections(
        self,
        *,
        messages: list[dict[str, Any]],
        system_extra_blocks: list[str],
        history_turns: list[dict[str, str]] | None,
        user_prompt: str,
    ) -> list[dict[str, Any]]:
        # 桩：审计关闭时不被调用（完整版见 C2-5c）。
        return []

    def _coerce_audit_sections(
        self, sections: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        # 桩：审计关闭时不被调用（完整版见 C2-5c）。
        return []

    def _audit_section(self, name: str, text: str) -> dict[str, Any]:
        raw = str(text or "")
        digest = ""
        if raw:
            import hashlib

            digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
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
        cjk_chars = sum(1 for char in raw if "\u4e00" <= char <= "\u9fff")
        non_cjk_chars = max(0, len(raw) - cjk_chars)
        return int(cjk_chars + ((non_cjk_chars + 3) // 4))

    def _sum_sections(self, sections: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "chars": sum(int(section.get("chars") or 0) for section in sections),
            "estimated_tokens": sum(int(section.get("estimated_tokens") or 0) for section in sections),
        }

    def _write_audit_record(self, record: dict[str, Any]) -> None:
        # 桩：审计关闭时不被调用（完整版含 LOG_DIR 落盘，C2-5c）。
        return None

    def _build_cache_kwargs(
        self,
        *,
        bundle: ModelBundle,
        prompt_cache_key: str,
    ) -> dict[str, Any]:
        # 桩：不带缓存提示时等价返回空（完整版见 C2-5c）。
        return {}

    def _wants_cache_hints(self, bundle: ModelBundle) -> bool:
        # 桩：默认环境（开关开、非官方 openai 端、不强制）下等价 False（C2-5c）。
        return False

    def _coerce_cache_key(self, prompt_cache_key: Any) -> str:
        raw = str(prompt_cache_key or "").strip().strip(":")
        if not raw:
            return ""
        namespace = str(self._env("PROMPT_CACHE_NAMESPACE", "shinku") or "").strip().strip(":")
        return f"{namespace}:{raw}" if namespace else raw

    def _coerce_cache_retention(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        return raw if raw in {"in_memory", "24h"} else ""
