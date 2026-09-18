"""公开发言的准入与配额门。

一次公开发言同时受两个上限约束：正在进行的条数（并发）与当天累计的条数（每日）。
这个类把两条约束收在一起，跨日自动复位，并且释放永不把计数压到负数。

对外可观察的判定结果有四个：``"disabled"``（开关关着，直接放行）、``"ok"``（放行并占额度）、
``"busy"``（并发满了）、``"daily_limit"``（当天额度用完）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from zoneinfo import ZoneInfo

#: 调用方没给消息（或只给了空白）时用的兜底文案。属面向用户的输出，逐字保留。
_FALLBACK_BUSY = "当前体验人数较多，请稍后再试。"
_FALLBACK_DAILY_LIMIT = "今日体验名额已满，明天再来看看吧。"

#: 跨日判定用的时区与日期格式。
_DEFAULT_ZONE = "Asia/Shanghai"
_STAMP_FORMAT = "%Y-%m-%d"


def _message_or(given: object, fallback: str) -> str:
    """给定值转字符串、去空白；空了就用兜底。"""

    text = str(given or fallback).strip()
    return text or fallback


@dataclass(frozen=True)
class GuardDecision:
    """一次准入判定的结果。"""

    allowed: bool
    acquired: bool
    reason: str
    message: str = ""


class PublicThinkGuard:
    """并发与每日两道配额，决定一次公开发言能不能开始。"""

    def __init__(
        self,
        *,
        enabled: bool,
        max_concurrent_thinks: int,
        daily_think_limit: int,
        busy_message: str,
        daily_limit_message: str,
        timezone_name: str = _DEFAULT_ZONE,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_concurrent_thinks = max(0, int(max_concurrent_thinks))
        self.daily_think_limit = max(0, int(daily_think_limit))
        self.busy_message = _message_or(busy_message, _FALLBACK_BUSY)
        self.daily_limit_message = _message_or(daily_limit_message, _FALLBACK_DAILY_LIMIT)
        self._zone = ZoneInfo(timezone_name)
        self._state_lock = RLock()
        self._in_flight = 0
        self._served_today = 0
        self._stamp = self._today()

    def try_acquire(self) -> GuardDecision:
        """申请开始一次公开发言。"""

        if not self.enabled:
            return GuardDecision(True, False, "disabled")
        with self._state_lock:
            self._roll_over()
            if self._quota_spent():
                return GuardDecision(False, False, "daily_limit", self.daily_limit_message)
            if self._saturated():
                return GuardDecision(False, False, "busy", self.busy_message)
            self._in_flight += 1
            self._served_today += 1
            return GuardDecision(True, True, "ok")

    def release(self) -> None:
        """释放一次公开发言占用的并发名额。当日已用不回退。"""

        if not self.enabled:
            return
        with self._state_lock:
            self._in_flight = max(0, self._in_flight - 1)

    def snapshot(self) -> dict[str, int | str | bool]:
        """当前计数与日期键的只读快照。"""

        with self._state_lock:
            self._roll_over()
            return {
                "enabled": self.enabled,
                "max_concurrent_thinks": self.max_concurrent_thinks,
                "daily_think_limit": self.daily_think_limit,
                "active_thinks": self._in_flight,
                "used_today": self._served_today,
                "day_key": self._stamp,
            }

    def _quota_spent(self) -> bool:
        """当天额度是否已用完。额度为 0 表示不限。"""

        return bool(self.daily_think_limit) and self._served_today >= self.daily_think_limit

    def _saturated(self) -> bool:
        """并发是否已满。上限为 0 表示不限。"""

        return bool(self.max_concurrent_thinks) and self._in_flight >= self.max_concurrent_thinks

    def _today(self) -> str:
        return datetime.now(self._zone).strftime(_STAMP_FORMAT)

    def _roll_over(self) -> None:
        """日期键变了就把当日已用归零；进行中的条数不动。"""

        stamp = self._today()
        if stamp != self._stamp:
            self._stamp = stamp
            self._served_today = 0


__all__ = ["GuardDecision", "PublicThinkGuard"]
