"""上游连续报错时，进程级别的静默开关。

上游不可达时，每个进来的对话事件都会各自发起一轮完整回合，逐轮失败后各回一句兜底台词，
同一个群里几秒内就能刷出一串重复的道歉。这里放一个进程级开关挡住后续回合：
失败攒够阈值就进入静默，静默时长按已经开闸的次数翻倍（有上限），任何一次成功都把状态整只清零。

开关只决定"要不要去发起新的上游回合"，其余本地能力（命令处理、旁观入库等）不在此列。
"""

from __future__ import annotations

import threading
import time


class LlmCircuitBreaker:
    """带锁的上游静默开关。

    构造参数会被就地归一化，归一化后的值直接挂在实例上供外部读取。
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        base_cooldown_seconds: float = 60.0,
        max_cooldown_seconds: float = 600.0,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.base_cooldown_seconds = max(1.0, float(base_cooldown_seconds))
        self.max_cooldown_seconds = max(
            self.base_cooldown_seconds, float(max_cooldown_seconds)
        )
        self._guard = threading.RLock()
        self._streak = 0
        self._quiet_until = 0.0
        self._trip_count = 0
        self._failed_at = 0.0
        self._recovered_at = 0.0

    def record_success(self) -> None:
        """上游恢复：连续计数与静默状态一起归零。"""
        with self._guard:
            self._streak = 0
            self._quiet_until = 0.0
            self._trip_count = 0
            self._recovered_at = time.time()

    def record_failure(self) -> None:
        """记一次上游失败；没攒到阈值就只是计数，不动静默状态。"""
        with self._guard:
            now = time.time()
            self._failed_at = now
            self._streak += 1
            if self._streak < self.failure_threshold:
                return
            self._trip_count += 1
            cooldown = min(
                self.max_cooldown_seconds,
                self.base_cooldown_seconds * (2 ** (self._trip_count - 1)),
            )
            self._quiet_until = now + cooldown

    def is_open(self) -> bool:
        """静默期是否还没走完。"""
        with self._guard:
            return time.time() < self._quiet_until

    def remaining_seconds(self) -> float:
        """静默期还剩几秒；不在静默期就是 0。"""
        with self._guard:
            return max(0.0, self._quiet_until - time.time())

    def snapshot(self) -> dict:
        """健康检查与日志用的一次性读数。"""
        with self._guard:
            now = time.time()
            return {
                "open": now < self._quiet_until,
                "consecutiveFailures": self._streak,
                "openCount": self._trip_count,
                "remainingSeconds": round(max(0.0, self._quiet_until - now), 1),
                "lastFailureAt": round(self._failed_at, 3),
                "lastSuccessAt": round(self._recovered_at, 3),
            }


_SHARED_SWITCH = LlmCircuitBreaker()


def get_llm_circuit_breaker() -> LlmCircuitBreaker:
    """取本进程共用的那一个开关，让调用层与事件层看到同一份状态。"""
    return _SHARED_SWITCH


__all__ = ["LlmCircuitBreaker", "get_llm_circuit_breaker"]
