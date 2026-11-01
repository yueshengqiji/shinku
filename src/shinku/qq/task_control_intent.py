"""Explicit cancellation command grammar; task scope is checked by the host."""
from __future__ import annotations

import re


def requests_cancellation(request: str, task_id: str) -> bool:
    """Require an affirmative command, not a verb anywhere in arbitrary prose.

    A task ID may be quoted as an operand. A quoted command, a negation, a
    question or conditional prose cannot match the entire command. Omitted IDs
    still require the caller's existing unique scoped task binding.
    """
    target = rf"(?-i:{re.escape(task_id)})"
    operand = rf'(?:{target}|`{target}`|"{target}"|\'{target}\'|“{target}”|「{target}」|『{target}』)'
    polite = r"(?:(?:请|麻烦|帮我|替我|现在|先|直接|立即)\s*)*"
    verb = r"(?:取消|停止|终止|cancel\b|stop\b|abort\b)"
    task = r"(?:(?:这(?:个)?|当前|该)\s*)?(?:任务|task\b)(?:号|编号)?\s*[:：]?\s*"
    return re.fullmatch(
        rf"{polite}{verb}\s*(?:一下\s*)?(?:{task})?(?:{operand})?\s*(?:一下|吧|了)?\s*[。.!！]*",
        str(request or "").strip(), re.I,
    ) is not None
