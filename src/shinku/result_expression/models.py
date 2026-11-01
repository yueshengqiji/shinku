"""Explicit, disabled-by-default expression configuration and transport output."""
from dataclasses import dataclass
import math
from typing import Any

from shinku.agent_runtime.contracts import canonical_hash


PARAGRAPH_REFERENCE_VERSION = 1
PARAGRAPH_PRESENTATION_VERSION = 1


EXPRESSION_RESPONSE_FAILURES = frozenset({
    "expression_output_truncated", "expression_output_rejected", "expression_json_invalid",
    "expression_wrapping_invalid", "expression_protocol_invalid", "expression_native_tool_call_forbidden",
})


@dataclass(frozen=True)
class ModelReply:
    value: Any
    usage: dict | None = None
    failure_reason: str = ""


@dataclass(frozen=True)
class ExpressionProfile:
    runtime: Any = None
    configuration_id: str = "template-v1"
    style_prompt: str = ""
    timeout_seconds: float = 10
    max_input_chars: int = 24000
    max_output_tokens: int = 2048
    checker: Any = None
    checker_timeout_seconds: float = 10
    format_version: int = 1
    detail_policy_version: int = 0
    paragraph_presentation_version: int = PARAGRAPH_PRESENTATION_VERSION
    semantic_review_version: int = 0
    # 可信宿主选择表达策略；selective只取消正文覆盖门，不赋予执行权限。
    content_policy: str = "complete"

    def __post_init__(self):
        if self.content_policy not in {"complete", "selective"}:
            raise ValueError("invalid_expression_content_policy")
        if type(self.format_version) is not int or self.format_version not in (1, 2):
            raise ValueError("invalid_expression_format")
        if (type(self.detail_policy_version) is not int or self.detail_policy_version not in (0, 2)
                or self.detail_policy_version and self.format_version != 2):
            raise ValueError("invalid_expression_detail_policy")
        if (type(self.paragraph_presentation_version) is not int or self.paragraph_presentation_version not in (1, 2, 3, 4, 5, 6, 7)
                or self.paragraph_presentation_version in (2, 3, 4, 5, 6, 7) and (
                    self.format_version != 2 or self.detail_policy_version != 2)
                or self.paragraph_presentation_version in (5, 6, 7) and self.semantic_review_version != 1):
            raise ValueError("invalid_expression_presentation_policy")
        if (type(self.semantic_review_version) is not int or self.semantic_review_version not in (0, 1)
                or self.semantic_review_version and self.format_version != 2):
            raise ValueError("invalid_expression_semantic_review")
        if (not isinstance(self.configuration_id, str) or not self.configuration_id.strip()
                or len(self.configuration_id) > 256 or not isinstance(self.style_prompt, str)
                or len(self.style_prompt) > 65536):
            raise ValueError("invalid_expression_profile")
        for value in (self.timeout_seconds, self.checker_timeout_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 60:
                raise ValueError("invalid_expression_timeout")
        if (type(self.max_input_chars) is not int or not 512 <= self.max_input_chars <= 65536
                or type(self.max_output_tokens) is not int or not 128 <= self.max_output_tokens <= 65536):
            raise ValueError("invalid_expression_limits")

    @property
    def config_hash(self):
        # Explicit configuration_id must change when a host changes model or
        # endpoint; adapters also expose an identity without credentials.
        def transport_identity(runtime):
            try:
                return str(getattr(runtime, "identity", type(runtime).__qualname__))
            except Exception:
                return type(runtime).__qualname__ + ":unavailable"
        connected_rules = {}
        if self.format_version == 2 and self.paragraph_presentation_version in (5, 6, 7):
            from .natural import CONNECTED_PROSE_RULES, CONNECTED_PROSE_CHECK_RULES
            connected_rules["connected_prose_rules_hash"] = canonical_hash({
                "expression": CONNECTED_PROSE_RULES, "checker": CONNECTED_PROSE_CHECK_RULES})
        if self.format_version == 2 and self.paragraph_presentation_version in (6, 7):
            from .persona_body import RULES, CHECK_RULES
            connected_rules["persona_body_rules_hash"] = canonical_hash({"expression": RULES, "checker": CHECK_RULES})
        return canonical_hash({"id": self.configuration_id, "style": self.style_prompt,
            "runtime": transport_identity(self.runtime),
            "checker": transport_identity(self.checker),
            "timeout": self.timeout_seconds, "check_timeout": self.checker_timeout_seconds,
            "input": self.max_input_chars, "output": self.max_output_tokens, "format": self.format_version,
            **({"paragraph_reference_version": PARAGRAPH_REFERENCE_VERSION} if self.format_version == 2 else {}),
            **({"paragraph_presentation_version": self.paragraph_presentation_version} if self.format_version == 2 else {}),
            **({"detail_policy_version": self.detail_policy_version} if self.detail_policy_version else {}),
            **({"semantic_review_version": self.semantic_review_version} if self.semantic_review_version else {}),
            **({"content_policy": self.content_policy} if self.content_policy != "complete" else {}),
            **connected_rules})
