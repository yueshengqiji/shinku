"""青空自己的主循环。受信事件 -> 状态投影 -> 一次决策 -> 原子写入。

此处没有真红的 persona、会话路由、任务执行器或隐式后台活动。表达候选只进入 outbox；
宿主观察和实际送达另外由受信入口报告。模型失败保留事件，不伪造回复或替换模型。
"""
from __future__ import annotations

import json

from .contracts import Decider, Decision, Event, KernelError, encode
from .store import Store


class Kernel:
    def __init__(self, store: Store, decider: Decider):
        self.store = store
        self.decider = decider

    def handle(self, event: Event, *, retry: bool = False) -> dict:
        view, context = self.store.begin(event, retry=retry)
        if context is None:
            return view
        try:
            # 模型适配器拿到独立副本，不可修改校验所依据的持久上下文。
            decision = self.decider.decide(json.loads(encode(context)))
            if not isinstance(decision, Decision):
                raise KernelError("typed_decision_required")
            return self.store.finish(event, view["attempt"], decision)
        except Exception as exc:
            # 只保存稳定内核错误码；第三方异常原文可能含 token、URL 或私用输入。
            code = str(exc) if isinstance(exc, KernelError) else "decider_failed"
            self.store.fail(event, view["attempt"], code)
            return self.store.status(event.event_id, event.scope)
        except BaseException:
            self.store.cancel(event.event_id, event.scope, expected_attempt=view["attempt"])
            raise
