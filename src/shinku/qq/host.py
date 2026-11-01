"""NapCat 消息宿主的组合边界。"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any, Callable

from .adapter import QQIngressAdapter, QQIngressResult
from .config import NapCatConnectionConfig
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, build_reply, deliver
from .napcat import (
    NapCatActionTransport,
    NapCatEventDecoder,
    NapCatHttpActionCaller,
    NapCatVisualBatch,
    NapCatVisualInputBridge,
)
from .message import IncomingMessage
from .cards import QQCardReader
from .turns import TurnDispatcher
from .welcome import GroupWelcomePolicy

__all__ = ["NapCatHost", "NapCatHostResult"]


@dataclass(frozen=True)
class NapCatHostResult:
    """一次入站事件经过宿主后的可观察结果。"""

    ingress: QQIngressResult
    visual: NapCatVisualBatch
    quote_fetched: bool = False
    welcome: DeliveryResult | None = None

    @property
    def scheduled(self) -> bool:
        return self.ingress.scheduled

    def as_dict(self) -> dict[str, Any]:
        result = {
            "ingress": self.ingress.as_dict(),
            "visual": {
                "ready": self.visual.ready,
                "pending": self.visual.pending,
                "inputs": [
                    {
                        "source_kind": item.source_kind,
                        "reference": item.reference,
                        "ready": item.ready,
                        "source_message_id": item.source_message_id,
                        "source_sender_id": item.source_sender_id,
                        "source_sender_label": item.source_sender_label,
                        "reference_kind": item.reference_kind,
                    }
                    for item in self.visual.inputs
                ],
            },
            "quote_fetched": self.quote_fetched,
        }
        if self.welcome is not None:
            result["welcome"] = self.welcome.as_dict()
        return result


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
        passive_observer: Callable[[Any], Any] | None = None,
        quote_fetcher: Callable[[str], Mapping[str, Any] | None] | None = None,
        forward_fetcher: Callable[[str], Mapping[str, Any] | None] | None = None,
        welcome_policy: GroupWelcomePolicy | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.decoder = decoder or NapCatEventDecoder()
        self.ingress = ingress or QQIngressAdapter()
        self.visual_bridge = visual_bridge or NapCatVisualInputBridge()
        self.transport = transport
        self.turn_dispatcher = turn_dispatcher
        self.passive_observer = passive_observer
        self.quote_fetcher = quote_fetcher
        self.forward_fetcher = forward_fetcher
        self.welcome_policy = welcome_policy or GroupWelcomePolicy()
        self._welcome_last_sent: dict[tuple[str, str], float] = {}
        self.last_observer_error = ""
        self.last_quote_error = ""
        self.clock = clock or time.time
        self._bind_visual_loader()

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
        passive_observer: Callable[[Any], Any] | None = None,
        forward_fetcher: Callable[[str], Mapping[str, Any] | None] | None = None,
        welcome_policy: GroupWelcomePolicy | None = None,
        clock: Callable[[], float] | None = None,
    ) -> "NapCatHost":
        caller = NapCatHttpActionCaller(base_url, access_token=access_token, opener=opener)
        return cls(
            ingress=ingress,
            visual_bridge=visual_bridge,
            transport=NapCatActionTransport(caller),
            turn_dispatcher=turn_dispatcher,
            passive_observer=passive_observer,
            quote_fetcher=caller.get_message,
            forward_fetcher=forward_fetcher or caller.get_forward_message,
            welcome_policy=welcome_policy,
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
        passive_observer: Callable[[Any], Any] | None = None,
        forward_fetcher: Callable[[str], Mapping[str, Any] | None] | None = None,
        welcome_policy: GroupWelcomePolicy | None = None,
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
            passive_observer=passive_observer,
            quote_fetcher=caller.get_message,
            forward_fetcher=forward_fetcher or caller.get_forward_message,
            welcome_policy=welcome_policy,
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
        event_time = float(self.clock() if at is None else at)
        admission = self.ingress.admit_message(message, at=event_time)
        quote_fetched = False
        welcome_result: DeliveryResult | None = None
        if admission.accepted:
            if message.is_group_increase:
                welcome_result = self._maybe_send_welcome(message, event_time)
            else:
                # 每个事件独享预算，当前附件先于引用；准入/去重失败不产生平台请求。
                reader = QQCardReader(self.forward_fetcher)
                message = replace(message, card_context=reader.render(message.segments))
                message, quote_fetched = self._resolve_quote(message, card_reader=reader)
            # Observation intentionally precedes route/attention filtering.  A
            # message that is not addressed to Shinku must still be available as
            # historical evidence for a later reply-chain or recall request.
            if not message.is_group_increase and self.passive_observer is not None:
                try:
                    self.passive_observer(message)
                    self.last_observer_error = ""
                except Exception as exc:  # pragma: no cover - observer isolation guard
                    self.last_observer_error = f"observer_error:{type(exc).__name__}"
        try:
            ingress = self.ingress.ingest_message(message, at=event_time, admission=admission)
        except TypeError as exc:
            # Keep the host compatible with small test/custom adapters written
            # before the admission argument existed.  Only fall back for that
            # signature mismatch; do not hide arbitrary adapter TypeErrors.
            if "admission" not in str(exc):
                raise
            ingress = self.ingress.ingest_message(message, at=event_time)
        visual = self.visual_bridge.build(message)
        result = NapCatHostResult(
            ingress=ingress,
            visual=visual,
            quote_fetched=quote_fetched,
            welcome=welcome_result,
        )
        if self.turn_dispatcher is not None:
            self.turn_dispatcher.submit(result)
        return result

    def flush(self):
        return self.ingress.flush()

    def set_turn_dispatcher(self, dispatcher: TurnDispatcher | None) -> None:
        """在宿主创建后接入回合调度器，避免构造阶段的循环依赖。"""

        self.turn_dispatcher = dispatcher
        self._bind_visual_loader()

    def _bind_visual_loader(self) -> None:
        setter = getattr(self.turn_dispatcher, "set_visual_loader", None)
        if callable(setter):
            setter(self.visual_bridge.build)

    def set_passive_observer(self, observer: Callable[[Any], Any] | None) -> None:
        """Attach or remove the pre-route passive memory observer."""

        self.passive_observer = observer

    def _resolve_quote(self, message: IncomingMessage, *, card_reader: QQCardReader | None = None) -> tuple[IncomingMessage, bool]:
        reader = card_reader or QQCardReader(self.forward_fetcher)
        if message.quoted_message:
            snapshot = dict(message.quoted_message)
            # 快照的 segments 是标准化后的形状，恢复原 payload 再解析；不信任其自报读取状态。
            snapshot["segments"] = [item.get("payload", item) for item in snapshot.get("segments", ())
                                    if isinstance(item, Mapping)]
            quoted = IncomingMessage.from_event(snapshot)
            quoted = replace(quoted, card_context=reader.render(quoted.segments))
            return message.with_quoted_message(quoted), False
        if not message.reply_message_ids or self.quote_fetcher is None:
            return message, False
        for message_id in message.reply_message_ids:
            try:
                raw = self.quote_fetcher(message_id)
            except Exception as exc:  # pragma: no cover - action transport isolation
                self.last_quote_error = f"quote_fetch_error:{type(exc).__name__}"
                continue
            if not isinstance(raw, Mapping):
                continue
            try:
                quoted = IncomingMessage.from_event(raw)
                quoted = replace(quoted, card_context=reader.render(quoted.segments))
            except (TypeError, ValueError, KeyError):
                continue
            self.last_quote_error = ""
            return message.with_quoted_message(quoted), True
        return message, False

    def send(self, message: OutgoingMessage) -> DeliveryResult:
        if self.transport is None:
            return DeliveryResult(False, "transport_unconfigured", error="napcat_transport_unconfigured")
        if str(message.conversation_type or "").lower() not in {"private", "group"}:
            return DeliveryResult(True, "desktop_delivered")
        return deliver(self.transport, message)

    def _maybe_send_welcome(self, message: IncomingMessage, at: float) -> DeliveryResult | None:
        policy = self.welcome_policy
        if not policy.enabled:
            return None
        text = policy.render(message)
        if not text:
            return DeliveryResult(False, "welcome_filtered", error="welcome_policy_filtered")
        key = (str(message.conversation_id or ""), str(message.sender_id or ""))
        last = self._welcome_last_sent.get(key)
        if last is not None and float(at) - last < policy.cooldown_seconds:
            return DeliveryResult(False, "welcome_cooldown", error="welcome_cooldown")
        outgoing = build_reply(message.conversation_id, text, conversation_type="group")
        result = self.send(outgoing)
        if result.sent:
            self._welcome_last_sent[key] = float(at)
        return result
