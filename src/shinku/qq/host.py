"""NapCat 消息宿主的组合边界。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .adapter import QQIngressAdapter, QQIngressResult
from .config import NapCatConnectionConfig
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, deliver
from .napcat import (
    NapCatActionTransport,
    NapCatEventDecoder,
    NapCatHttpActionCaller,
    NapCatVisualBatch,
    NapCatVisualInputBridge,
)
from .turns import TurnDispatcher

__all__ = ["NapCatHost", "NapCatHostResult"]


@dataclass(frozen=True)
class NapCatHostResult:
    """一次入站事件经过宿主后的可观察结果。"""

    ingress: QQIngressResult
    visual: NapCatVisualBatch

    @property
    def scheduled(self) -> bool:
        return self.ingress.scheduled

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingress": self.ingress.as_dict(),
            "visual": {
                "ready": self.visual.ready,
                "pending": self.visual.pending,
                "inputs": [
                    {
                        "source_kind": item.source_kind,
                        "reference": item.reference,
                        "ready": item.ready,
                    }
                    for item in self.visual.inputs
                ],
            },
        }


class NapCatHost:
    """把 NapCat 事件、QQ 入站策略、视觉附件和出站传输组合起来。

    该对象是宿主内核，不是监听服务器：HTTP/WebSocket/QQ SDK 只需把收到的 body
    交给 ``handle_event``，再消费返回的结构化结果。这样不会在导入或构造阶段
    偷开端口、读取 token 或触发网络。
    """

    def __init__(
        self,
        *,
        decoder: NapCatEventDecoder | None = None,
        ingress: QQIngressAdapter | None = None,
        visual_bridge: NapCatVisualInputBridge | None = None,
        transport: DeliveryTransport | None = None,
        turn_dispatcher: TurnDispatcher | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.decoder = decoder or NapCatEventDecoder()
        self.ingress = ingress or QQIngressAdapter()
        self.visual_bridge = visual_bridge or NapCatVisualInputBridge()
        self.transport = transport
        self.turn_dispatcher = turn_dispatcher
        self.clock = clock or time.time

    @classmethod
    def from_http(
        cls,
        base_url: str,
        *,
        access_token: str = "",
        opener: Callable[..., Any] | None = None,
        visual_bridge: NapCatVisualInputBridge | None = None,
        ingress: QQIngressAdapter | None = None,
        turn_dispatcher: TurnDispatcher | None = None,
        clock: Callable[[], float] | None = None,
    ) -> "NapCatHost":
        caller = NapCatHttpActionCaller(base_url, access_token=access_token, opener=opener)
        return cls(
            ingress=ingress,
            visual_bridge=visual_bridge,
            transport=NapCatActionTransport(caller),
            turn_dispatcher=turn_dispatcher,
            clock=clock,
        )

    @classmethod
    def from_config(
        cls,
        config: NapCatConnectionConfig,
        *,
        opener: Callable[..., Any] | None = None,
        visual_bridge: NapCatVisualInputBridge | None = None,
        ingress: QQIngressAdapter | None = None,
        turn_dispatcher: TurnDispatcher | None = None,
        clock: Callable[[], float] | None = None,
    ) -> "NapCatHost":
        errors = config.validate()
        if errors:
            raise ValueError("invalid NapCat configuration: " + ",".join(errors))
        caller = NapCatHttpActionCaller(
            config.base_url,
            access_token=config.token_value(),
            timeout=config.timeout,
            opener=opener,
        )
        return cls(
            ingress=ingress,
            visual_bridge=visual_bridge,
            transport=NapCatActionTransport(caller),
            turn_dispatcher=turn_dispatcher,
            clock=clock,
        )

    @classmethod
    def from_env(
        cls,
        environ: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "NapCatHost":
        return cls.from_config(NapCatConnectionConfig.from_env(environ), **kwargs)

    def handle_event(self, payload: Any, *, at: float | None = None) -> NapCatHostResult:
        message = self.decoder.decode(payload)
        ingress = self.ingress.ingest_message(message, at=float(self.clock() if at is None else at))
        visual = self.visual_bridge.build(message)
        result = NapCatHostResult(ingress=ingress, visual=visual)
        if self.turn_dispatcher is not None:
            self.turn_dispatcher.submit(result)
        return result

    def flush(self):
        return self.ingress.flush()

    def set_turn_dispatcher(self, dispatcher: TurnDispatcher | None) -> None:
        """在宿主创建后接入回合调度器，避免构造阶段的循环依赖。"""

        self.turn_dispatcher = dispatcher

    def send(self, message: OutgoingMessage) -> DeliveryResult:
        if self.transport is None:
            return DeliveryResult(False, "transport_unconfigured", error="napcat_transport_unconfigured")
        return deliver(self.transport, message)
