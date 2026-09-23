"""可选的、确定性的 QQ 入群欢迎策略。

入群欢迎属于外部副作用，不经过模型、不写入记忆，也不在默认配置下发送。
它只把一个已归一化的 ``group_increase`` 事件转换成模板文本；实际投递仍由
NapCatHost 的受控 transport 完成。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .message import IncomingMessage

__all__ = ["GroupWelcomePolicy"]


def _flag(value: object, *, default: bool = False) -> bool:
    raw = str(value or "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "开启", "开"}


def _ids(value: object) -> frozenset[str]:
    raw = str(value or "")
    return frozenset(item.strip() for item in raw.replace("，", ",").split(",") if item.strip())


@dataclass(frozen=True)
class GroupWelcomePolicy:
    """入群欢迎的显式配置；默认关闭，且只允许固定模板文本。"""

    enabled: bool = False
    template: str = "欢迎 {user} 加入群聊。"
    allowed_group_ids: frozenset[str] = frozenset()
    cooldown_seconds: float = 60.0
    max_chars: int = 240

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "GroupWelcomePolicy":
        env = os.environ if environ is None else environ
        raw_cooldown = str(env.get("SHINKU_QQ_GROUP_WELCOME_COOLDOWN_SECONDS", "60") or "60").strip()
        try:
            cooldown = max(0.0, float(raw_cooldown))
        except (TypeError, ValueError):
            cooldown = 60.0
        raw_max_chars = str(env.get("SHINKU_QQ_GROUP_WELCOME_MAX_CHARS", "240") or "240").strip()
        try:
            max_chars = max(32, min(1000, int(raw_max_chars)))
        except (TypeError, ValueError):
            max_chars = 240
        return cls(
            enabled=_flag(env.get("SHINKU_QQ_GROUP_WELCOME_ENABLED"), default=False),
            template=str(
                env.get("SHINKU_QQ_GROUP_WELCOME_TEMPLATE", cls.template) or cls.template
            ).strip(),
            allowed_group_ids=_ids(env.get("SHINKU_QQ_GROUP_WELCOME_GROUP_IDS")),
            cooldown_seconds=cooldown,
            max_chars=max_chars,
        )

    def render(self, message: IncomingMessage) -> str:
        """渲染一条固定欢迎语；模板错误时安全地不发送。"""

        if not self.enabled or not message.is_group_increase:
            return ""
        group_id = str(message.conversation_id or "").strip()
        if self.allowed_group_ids and group_id not in self.allowed_group_ids:
            return ""
        user_id = str(message.sender_id or "").strip()
        user = str(message.sender_label or "").strip() or user_id or "新成员"
        try:
            text = self.template.format(user=user, user_id=user_id or "unknown", group=group_id or "unknown")
        except (KeyError, IndexError, ValueError):
            return ""
        return " ".join(str(text).split())[: self.max_chars].strip()

