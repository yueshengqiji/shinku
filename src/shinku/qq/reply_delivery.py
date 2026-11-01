"""QQ 回复格式化、分段投递与送达回执（Reply Delivery）。

职责与上下游：
- 职责：将大模型或 Agent 生成的最终回复文本进行排版（分段、标点、语音转换、@ 目标与引用组装），
  通过底层消息发送器（`QQMessageSender`）进行物理发送，并收集底层确认回执（`QQDeliveryReceipt`）。
- 上游：由 `agent_bridge.py`（经由 `_deliver_text`）、`turn_routing.py`（`deliver_direct_chat`）调用。
- 下游：调用底层 OneBot/NapCat 发送驱动；通过 `accept_receipt` 将收集到的回执交给宿主，
  由宿主记录已送达正文、更新对话连续性和群聊观察；`finalize_usage` 另行核实检索记忆在回复中的实际使用。

设计原因与状态归属：
1. 生成与送达分离：发送器返回 `DeliveryResult.sent=True` 时累计成功发送，消息 ID 可以为空；
   已知 `message_ids` 另外保留为回执，不据此把整份回复或未知正文判定为完整送达；
2. 状态所有权：`_TransportAttempt` 独占当前单次投递尝试的传输进度状态；
3. 部分发送与通知边界：存在分段失败且已确认部分送达时标记为 `partial`，其余分段仍可继续尝试；
   `uncertain_transport` 防止后续气泡再次附带首次通知的 @ 或引用，不提供跨投递调用的重试去重；
4. 正文入库与使用核实分开：宿主在 `record_memory=True` 且回执有送达记录时，记录已送达的非空正文；
   分段失败仍可保留已送达片段及 partial 元数据。`finalize_usage` 只在没有分段失败、存在成功发送，
   且 Agent 状态属于 completed/chat/partial 并允许记录记忆时执行，不负责写入助手回复正文。

阅读入口：
- 核心交付入口：`deliver_qq_reply`
- 投递计划组装：`_prepare_delivery`
- 传输过程驱动：`_TransportAttempt.run`
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from . import mentions
from .delivery import DeliveryResult, OutgoingMessage, OutgoingSegment
from .message import IncomingMessage
from .reply_host import ReplyHost
from .reply_models import QQAgentReply, QQMessageSender
from .turns import QQTurn


@dataclass(frozen=True)
class QQReplyPublication:
    turn: QQTurn
    final_text: str
    agent_status: str = "completed"
    reason: str = ""
    record_memory: bool = True
    track_chat_flow: bool = True
    trace_id: str = ""
    tool_envelopes: Sequence[Any] | None = None
    autonomous_authorized: bool = False
    expression_id: str = ""
    result_hash: str = ""
    publication_guard: Callable[[], str] | None = None


@dataclass(frozen=True)
class QQDeliveryReceipt:
    """Separate actual content, known message IDs and uncertain failures."""

    delivered_text: str
    message_ids: tuple[str, ...]
    delivered_segments: int
    content_complete: bool
    expanded: bool
    segment_failures: tuple[tuple[int, str, str], ...]

    @property
    def has_delivery(self) -> bool:
        return bool(self.delivered_segments or self.message_ids)

    @property
    def status(self) -> str:
        return "partial" if self.segment_failures else "sent"


@dataclass(frozen=True)
class QQReplyDeliveryDependencies:
    sender: QQMessageSender | None
    reply_host: ReplyHost
    voice_renderer: Callable[[str], str] | None
    bot_ids: Sequence[str]
    segment_delay_seconds: float
    responses_enabled: Callable[[], bool]
    reply_mode: Callable[[], str]
    can_silence_autonomous: Callable[[QQTurn], bool]
    response_message: Callable[[QQTurn], IncomingMessage | None]
    ground_text: Callable[[QQTurn, str, str, str], str]
    finish_chat_flow: Callable[..., None]
    accept_receipt: Callable[[QQReplyPublication, QQDeliveryReceipt], None]
    finalize_usage: Callable[[QQReplyPublication, str], None]
    trace: Callable[..., None]


@dataclass(frozen=True)
class _DeliveryPlan:
    turn: QQTurn
    final_text: str
    segments: tuple[str, ...]
    mention_targets: tuple[tuple[str, str], ...]
    allow_mentions: bool
    response_mentions: tuple[str, ...]
    quote_id: str
    expanded: bool


@dataclass
class _DeliveryProgress:
    message_ids: list[str] = field(default_factory=list)
    delivered_segments: int = 0
    delivered_text_segments: list[str] = field(default_factory=list)
    voice_delivered: bool = False
    uncertain_transport: bool = False
    segment_failures: list[tuple[int, str, str]] = field(default_factory=list)


def _publication_block_reason(publication):
    # 普通新消息在生成中交给已有群聊审查；实际每段发送前仍须覆盖最新游标，避免漏掉更正。
    if publication.turn.ambient_reply_reviewed:
        if reason := publication.turn.suppression_reason("publish"):
            return reason
    if publication.publication_guard is None:
        return ""
    try:
        return "result_binding_changed" if publication.publication_guard() != "" else ""
    except Exception:
        return "result_binding_changed"


def _publication_current(publication):
    return not _publication_block_reason(publication)


def deliver_qq_reply(
    publication: QQReplyPublication, *, dependencies: QQReplyDeliveryDependencies,
) -> QQAgentReply:
    """执行准入检查、排版分段并驱动底层传输。

    设计原因与边界保障：
    1. 准入拦截：
       - 非管理指令检查：若全局响应被指令关闭（responses_enabled==False），抑制投递；
       - 自主静默检查：若处于自主模式但缺乏大模型明确决策授权，抑制投递；
       - 结果快照有效性检查（_publication_current）：若任务绑定在排队期间已被覆盖，放弃陈旧结果；
    2. 离线/干运行隔离：
       - 若未配置底层发送器（sender is None），标记为 `dry_run`，更新流控但不假装已发送到 QQ；
    3. 送达回执与记忆边界：
       - `receipt.has_delivery`（成功发送数或已知消息 ID 非空）时调用 `accept_receipt`；
         群聊只要有非空已送达正文也会交给该回调，宿主按回执条件分别处理各项状态。
       - 回执回调在分段失败判断之前执行，因此允许记录已送达片段；存在失败时返回
         `_failed_reply`，其 delivery_status 为 partial 或首个失败状态，整份回复的 sent 为 False。
       - 没有分段失败、delivered_segments > 0、record_memory=True 且 Agent 状态属于
         completed/chat/partial 时，才调用 `finalize_usage` 核实检索记忆的实际使用。
    """
    deps = dependencies
    if publication.agent_status != "command" and not deps.responses_enabled():
        deps.trace(publication.trace_id, "delivery_suppressed", reason="responses_disabled")
        return QQAgentReply(publication.agent_status, final_text=str(publication.final_text or ""),
                            delivery_status="suppressed", reason="responses_disabled")
    if deps.can_silence_autonomous(publication.turn) and not publication.autonomous_authorized:
        reason = publication.reason or "autonomous_no_model_decision"
        deps.trace(publication.trace_id, "autonomous_delivery", status="suppressed", reason=reason)
        return QQAgentReply(publication.agent_status, delivery_status="suppressed", reason=reason)
    if reason := _publication_block_reason(publication):
        deps.trace(publication.trace_id, "delivery_suppressed", reason=reason)
        return QQAgentReply("silent" if reason.startswith("autonomous_") else publication.agent_status,
                            delivery_status="suppressed", reason=reason)

    grounded_text = publication.final_text if publication.expression_id else deps.ground_text(
        publication.turn, publication.final_text, publication.agent_status, publication.trace_id)
    plan = _prepare_delivery(publication.turn, grounded_text, publication.agent_status, deps,
                             preserve_facts=bool(publication.expression_id))
    publication = replace(publication, turn=plan.turn, final_text=plan.final_text)
    deps.trace(publication.trace_id, "delivery_prepare", segment_count=len(plan.segments),
               expanded=plan.expanded, agent_status=publication.agent_status)
    if plan.turn.conversation_type == "desktop":
        # HTTP 适配器在响应体成功写出后确认送达；此处只有候选正文，不调用 QQ。
        if plan.turn.local_delivery is None:
            return QQAgentReply(publication.agent_status, final_text=plan.final_text,
                                delivery_status="dry_run", reason="local_delivery_unavailable")
        return plan.turn.local_delivery(publication, deps)
    if deps.sender is None:
        # 离线或无发送器时记录干运行，绝不假装 QQ 已真实送达
        deps.finish_chat_flow(plan.turn, plan.final_text, expanded=plan.expanded,
                              track=publication.track_chat_flow)
        return QQAgentReply(publication.agent_status, final_text=plan.final_text,
                            delivery_status="dry_run", reason=publication.reason or "sender_not_configured")

    attempt = _TransportAttempt(publication, plan, deps, deps.sender)
    segment_total = attempt.run()
    receipt = attempt.receipt()
    deps.trace(publication.trace_id, "delivery_summary", segment_total=segment_total,
               delivered_segments=receipt.delivered_segments,
               failed_segments=[item[0] for item in receipt.segment_failures])
    if receipt.has_delivery or (plan.turn.conversation_type == "group" and receipt.delivered_text.strip()):
        deps.accept_receipt(publication, receipt)
    if receipt.segment_failures:
        return _failed_reply(publication, receipt)
    # 助手正文已由回执回调按条件记录；此处仅在无分段失败且状态允许时核实检索记忆的实际使用
    if publication.record_memory and receipt.delivered_segments and publication.agent_status in {"completed", "chat", "partial"}:
        deps.finalize_usage(publication, receipt.delivered_text)
    return QQAgentReply(publication.agent_status, final_text=plan.final_text, sent=True,
                        delivery_status="sent", reason=publication.reason,
                        delivery=DeliveryResult(True, "sent", receipt.message_ids))


def _prepare_delivery(
    turn: QQTurn, final_text: str, agent_status: str, deps: QQReplyDeliveryDependencies, *, preserve_facts=False,
) -> _DeliveryPlan:
    raw_text = str(final_text or "").strip()
    targets = mentions.group_mention_targets(turn, bot_ids=deps.bot_ids)
    allow_mentions = str(turn.conversation_type or "").strip().casefold() == "group"
    response_target = deps.response_message(turn)
    quote_id = str(response_target.message_id or "") if response_target is not None and allow_mentions else ""
    allowed_member_ids = {identifier for identifier, _ in targets}
    internal_text = mentions.sanitize_mention_markers(raw_text, targets, allow_group=allow_mentions)
    legacy_mentions = [part.target_id for part in mentions.outgoing_text_segments(
        internal_text, targets, allow_group=allow_mentions,
    ) if part.kind == "at"]
    if turn.response_mention_ids is not None:
        response_mentions = [identifier for identifier in turn.response_mention_ids if identifier in allowed_member_ids][:2]
    elif legacy_mentions:
        response_mentions = list(dict.fromkeys(legacy_mentions))[:2]
    else:
        response_mentions = [str(response_target.sender_id)] if not quote_id and response_target is not None and str(response_target.sender_id) in allowed_member_ids else []
    # Default to a source quote. Explicit validated @ targets replace it.
    if response_mentions:
        quote_id = ""
    turn = replace(turn, response_mention_ids=tuple(response_mentions))
    final_text = mentions.render_mention_markers(internal_text, targets, allow_group=allow_mentions)
    shaped = deps.reply_host.shape(internal_text, user_message=turn.combined_text) if not preserve_facts else None
    # Questions awaiting authorization and command acknowledgements are whole
    # interactive messages. Ordinary reports share the sentence boundary.
    keep_as_one_message = agent_status in {"waiting_user", "waiting_background", "command"}
    # Receipt-bound fact blocks must not pass through stylistic rewrites such
    # as voice phrase cleanup or punctuation stripping. Use the same transport
    # and routing, with each immutable block as one readable message.
    # A paragraph separator belongs inside a valid text bubble. Splitting it
    # into an empty send incorrectly marks an otherwise complete result partial.
    segments = tuple(re.split(r"(?<!\n)\n(?!\n)", internal_text)) if preserve_facts else (internal_text,) if keep_as_one_message else tuple(shaped.messages)
    expanded = bool(shaped and shaped.expanded) if not keep_as_one_message else False
    return _DeliveryPlan(turn, final_text, segments or (internal_text,), targets,
                         allow_mentions, tuple(response_mentions), quote_id, expanded)


class _TransportAttempt:
    """A fresh instance owns all transport progress for one publication."""

    def __init__(
        self, publication: QQReplyPublication, plan: _DeliveryPlan, deps: QQReplyDeliveryDependencies,
        sender: QQMessageSender,
    ) -> None:
        self.publication = publication
        self.plan = plan
        self.deps = deps
        self.sender = sender
        self.progress = _DeliveryProgress()

    def responses_stopped(self) -> bool:
        # /stop and /start acknowledgements remain explicit control messages.
        return (self.publication.agent_status != "command" and not self.deps.responses_enabled()) or not _publication_current(self.publication)

    def stop_reason(self):
        return "responses_disabled" if not self.deps.responses_enabled() else _publication_block_reason(self.publication) or "result_binding_changed"

    def trace(self, event: str, **fields: Any) -> None:
        self.deps.trace(self.publication.trace_id, event, **fields)

    def run(self) -> int:
        reply_mode = self.deps.reply_mode()
        voice_path = ""
        # Voice supplements the quoted text. Send readable content before TTS
        # so a slow/failed renderer cannot hide the reply or combine QQ quote
        # and record segments into an unsupported voice-only message.
        segments = self.plan.segments
        self.send_text_segments(segments)
        failures = self.progress.segment_failures
        if self.publication.agent_status == "completed" and reply_mode in {"voice", "both"} and self.deps.voice_renderer is not None and not failures:
            if self.responses_stopped():
                failures.append((len(segments) + 1, "suppressed", self.stop_reason()))
            else:
                voice_path = self.render_voice()
        if voice_path and not failures:
            self.send_voice(voice_path, segment_total=len(segments))
        return len(segments)

    def render_voice(self) -> str:
        renderer = self.deps.voice_renderer
        if renderer is None:
            return ""
        try:
            return str(renderer(self.plan.final_text) or "").strip()
        except Exception as exc:
            code = getattr(exc, "voice_error_code", "")
            self.trace("voice_render_failed", error=type(exc).__name__,
                       reason=code if code == "voice_translation_invalid" else "voice_render_error")
            return ""

    def send_text_segments(self, items: Sequence[str]) -> None:
        progress = self.progress
        plan = self.plan
        for index, text in enumerate(items):
            if not self.responses_stopped() and index and self.deps.segment_delay_seconds:
                time.sleep(self.deps.segment_delay_seconds)
            # Recheck after pacing before each text bubble.
            if self.responses_stopped():
                progress.segment_failures.append((index + 1, "suppressed", self.stop_reason()))
                self.trace("delivery_suppressed", segment_index=index + 1, reason=self.stop_reason())
                break
            outgoing_segments = mentions.outgoing_text_segments(text, plan.mention_targets, allow_group=plan.allow_mentions)
            # Unknown transport consumes notification routing, too. Later
            # bubbles never re-notify users through model mention markers.
            outgoing_segments = tuple(part for part in outgoing_segments if part.kind != "at")
            first_notification = not progress.delivered_segments and not progress.message_ids and not progress.uncertain_transport
            if plan.allow_mentions and first_notification:
                outgoing_segments = tuple(OutgoingSegment(kind="at", target_id=identifier)
                                          for identifier in plan.response_mentions) + outgoing_segments
            if not outgoing_segments:
                progress.segment_failures.append((index + 1, "empty_after_sanitization", "no_sendable_segments"))
                continue
            message = OutgoingMessage(
                conversation_id=plan.turn.conversation_id, segments=outgoing_segments,
                conversation_type=plan.turn.conversation_type,
                reply_to_message_id=plan.quote_id if first_notification else "",
            )
            try:
                delivery = self.sender(message)
            except Exception as exc:
                progress.uncertain_transport = True
                error = type(exc).__name__
                progress.segment_failures.append((index + 1, "transport_error", error))
                self.trace("delivery_segment", segment_index=index + 1,
                           segment_total=len(items), status="transport_error", error=error)
                continue
            if not isinstance(delivery, DeliveryResult):
                progress.uncertain_transport = True
                progress.segment_failures.append((index + 1, "sender_invalid", "sender_must_return_delivery_result"))
                self.trace("delivery_segment", segment_index=index + 1, segment_total=len(items), status="sender_invalid")
                continue
            if delivery.sent:
                progress.message_ids.extend(delivery.message_ids)
                progress.delivered_segments += 1
                progress.delivered_text_segments.append(mentions.render_mention_markers(
                    text, plan.mention_targets, allow_group=plan.allow_mentions,
                ))
                self.trace("delivery_segment", segment_index=index + 1,
                           segment_total=len(items), status="sent", message_id_count=len(delivery.message_ids))
                continue
            failure_status = str(delivery.status or "delivery_failed")
            failure_error = str(delivery.error or failure_status)
            if delivery.message_ids:
                progress.message_ids.extend(delivery.message_ids)
            elif not delivery.definitive_not_sent:
                progress.uncertain_transport = True
            progress.segment_failures.append((index + 1, failure_status, failure_error))
            self.trace("delivery_segment", segment_index=index + 1,
                       segment_total=len(items), status=failure_status, error=failure_error,
                       message_id_count=len(delivery.message_ids))

    def send_voice(self, voice_path: str, *, segment_total: int) -> None:
        progress = self.progress
        plan = self.plan
        if self.responses_stopped():
            progress.segment_failures.append((segment_total + 1, "suppressed", self.stop_reason()))
            return
        try:
            delivery = self.sender(OutgoingMessage(
                conversation_id=plan.turn.conversation_id,
                conversation_type=plan.turn.conversation_type,
                segments=(OutgoingSegment(kind="audio", local_path=voice_path),),
            ))
            if isinstance(delivery, DeliveryResult) and delivery.sent:
                progress.message_ids.extend(delivery.message_ids)
                progress.delivered_segments += 1
                progress.voice_delivered = True
                self.trace("voice_delivery", status="sent")
            else:
                known_ids = tuple(getattr(delivery, "message_ids", ()) or ())
                if known_ids:
                    progress.message_ids.extend(known_ids)
                    progress.delivered_segments += 1
                progress.segment_failures.append((segment_total + 1, "voice_delivery_failed", str(getattr(delivery, "error", "invalid_result"))))
        except Exception as exc:
            progress.segment_failures.append((segment_total + 1, "voice_transport_error", type(exc).__name__))
            self.trace("voice_delivery", status="failed", error=type(exc).__name__)

    def receipt(self) -> QQDeliveryReceipt:
        progress = self.progress
        delivered_text = "\n".join(progress.delivered_text_segments)
        content_complete = progress.voice_delivered or (
            len(progress.delivered_text_segments) == len(self.plan.segments)
            and re.sub(r"[\s。]+", "", delivered_text) == re.sub(r"[\s。]+", "", self.plan.final_text)
        )
        if content_complete:
            # Canonical punctuation belongs only to fully delivered content.
            delivered_text = self.plan.final_text
        return QQDeliveryReceipt(delivered_text, tuple(progress.message_ids), progress.delivered_segments,
                                 content_complete, self.plan.expanded, tuple(progress.segment_failures))


def _failed_reply(publication: QQReplyPublication, receipt: QQDeliveryReceipt) -> QQAgentReply:
    status = "partial" if receipt.has_delivery else receipt.segment_failures[0][1]
    summary = ";".join(f"segment_{index}:{failure_status}:{error}"
                       for index, failure_status, error in receipt.segment_failures)[:240]
    return QQAgentReply(publication.agent_status, final_text=publication.final_text, sent=False,
                        delivery_status=status, reason=summary,
                        delivery=DeliveryResult(False, status, receipt.message_ids, summary))
