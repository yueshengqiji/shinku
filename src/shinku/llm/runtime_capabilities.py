"""LLM 调用引擎 —— 能力协商 mixin（C2-5a 阶段为最小实现）。

本模块承载 ``LLMRuntime`` 的「供应商能力」面：

- 原生工具（tools / tool_choice / tool_calls）的规范化与抽取；
- (host, model) 画像驱动的发送决策（``ProviderToolProfile``）；
- 配置 allowlist 的画像扩展；
- 思考控制（DeepSeek thinking）请求参数；
- 用户图片项的规范化；
- response_format=json_object 的适用判定与 JSON 关键字提示。

**C2-5a 状态：** 这里只放「不启用原生工具、不启用思考控制」路径所需的
最小实现——即 ``_coerce_tools`` 等方法在输入为空时的等价返回。原生
工具画像表、allowlist 解析、流式 tool_calls 收集与 thinking 控制的完整
实现与测试面属 C2-5b 批次，届时逐方法替换本文件的桩体。
"""

from __future__ import annotations

from typing import Any

from shinku.llm.runtime_core import (
    DEFAULT_PROVIDER_TOOL_PROFILE,
    ModelBundle,
    ProviderToolProfile,
)


class RuntimeCapabilitiesMixin:
    """原生工具与协议能力协商（C2-5a 桩；完整实现见 C2-5b）。"""

    def _coerce_tools(self, value: Any) -> list[dict[str, Any]]:
        # 桩：空输入等价返回（完整版含 name/参数过滤，C2-5b）。
        return []

    def _coerce_tool_choice(self, value: Any) -> Any:
        # 桩：空输入等价返回（完整版含 auto/none/required 白名单，C2-5b）。
        return ""

    def _tool_profile(self, bundle: ModelBundle) -> ProviderToolProfile:
        # 桩：默认画像（完整版按 (host, model) 查 PROVIDER_TOOL_PROFILES
        # 与 SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST，C2-5b）。
        return DEFAULT_PROVIDER_TOOL_PROFILE

    def _wants_native_tools(self, bundle: ModelBundle) -> bool:
        return bool(self._tool_profile(bundle).supports_native_tools)

    def _read_native_tool_call(self, response: Any) -> dict[str, Any] | None:
        # 桩：无 tool_calls 时等价返回（完整版见 C2-5b）。
        return None

    def _collect_stream_tool_parts(
        self, chunk: Any, parts: dict[int, dict[str, Any]]
    ) -> None:
        # 桩：流式 chunk 不含 tool_calls 时等价 no-op（C2-5b）。
        return None

    def _assemble_stream_tool_call(
        self, parts: dict[int, dict[str, Any]]
    ) -> dict[str, Any] | None:
        # 桩：无 parts 时等价返回（C2-5b）。
        return None

    def _decode_tool_args(self, value: Any) -> dict[str, Any]:
        # 桩：空/无效输入等价返回（C2-5b）。
        return {}

    def _coerce_image_items(self, value: Any) -> list[dict[str, Any]]:
        # 桩：空输入等价返回（完整版含 data:image/ 前缀过滤与 5 张上限，C2-5b）。
        return []

    def _build_thinking_kwargs(self, *, bundle: ModelBundle) -> dict[str, Any]:
        mode = str(self._env("LLM_THINKING_MODE", "disabled") or "").strip().lower()
        if mode in {"", "default", "auto"}:
            return {}
        if mode not in {"enabled", "disabled"}:
            return {}
        if not self._supports_thinking_control(bundle):
            # 思考控制没生效时的诊断：协议 / 模型名 / host 不匹配会让 reasoning 拉满
            from shinku.llm.runtime_core import logger

            logger.warning(
                "thinking_control_skipped: mode=%s model=%s protocol=%s base_url=%s",
                mode,
                str(getattr(bundle, "model", "") or ""),
                str(getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", "")) or ""),
                str(getattr(bundle.client, "base_url", "") or ""),
            )
            return {}
        from shinku.llm.runtime_core import logger

        logger.debug(
            "thinking_control_applied: mode=%s model=%s", mode, str(getattr(bundle, "model", "") or "")
        )
        return {"extra_body": {"thinking": {"type": mode}}}

    def _supports_thinking_control(self, bundle: ModelBundle) -> bool:
        # 桩：非 DeepSeek 组合恒 False（完整版含模型名前缀与 host 判定，C2-5b）。
        return False

    def _wants_response_json(self, bundle: ModelBundle) -> bool:
        # OpenAI 兼容端强制 response_format=json_object，让 persona 结构化
        # JSON 真正被网关约束住（而不只是 prompt 里求一下）。Anthropic 的
        # API 形态不同，不能收 response_format。
        protocol = str(getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", "")) or "").strip().lower()
        if protocol not in {"ollama", "openai"}:
            return False
        return bool(self._tool_profile(bundle).force_response_json)

    def _ensure_json_hint(self, messages: list[dict[str, Any]]) -> None:
        # OpenAI/DeepSeek 要求 response_format=json_object 时消息里必须出现
        # 字面量 "json"。persona prompt 是中文、不一定带，缺了就补一条短提示。
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and "json" in content.lower():
                return
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "json" in str(part.get("text") or "").lower():
                        return
        if messages and str(messages[0].get("role") or "") == "system":
            # 不改写第一条稳定 system；JSON 约束放后续动态 system，
            # 避免两种模式形成两套缓存前缀。
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "（本轮只输出一个合法的 JSON object，不要输出多余文字。）",
                },
            )

    def _is_official_openai(self, base_url: str) -> bool:
        from urllib.parse import urlparse

        raw = str(base_url or "").strip()
        if not raw:
            return True
        try:
            parsed = urlparse(raw)
        except Exception:
            return False
        hostname = str(parsed.hostname or "").strip().lower()
        if not hostname:
            return False
        return hostname == "api.openai.com" or hostname.endswith(".openai.com")
