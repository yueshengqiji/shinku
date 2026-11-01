"""低频群聊回顾执行器；复用宿主日账本、ConversationRuntime和统一发送回执。

读取和生成都在网络发送前完成。一次目标/日期只抢占一次模型机会，沉默、失败和
发送不确定均不自动重试；明确空闲与热度不足不消耗这次机会。账本不保存聊天正文。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from shinku.conversation import ConversationChannel, ConversationScope
from shinku.qq.delivery import build_reply, deliver
from shinku.qq.napcat import NapCatActionTransport
from .group_digest import DigestPolicy, read_digest_material, validated_digest


@dataclass(frozen=True)
class DigestSchedule:
    enabled: bool = False
    group_ids: tuple[str, ...] = ()
    excluded_sender_ids: tuple[str, ...] = ()
    start_hour: int = 20
    end_hour: int = 22

    @classmethod
    def from_environment(cls, env: Mapping[str, str]):
        def ids(key):
            return tuple(dict.fromkeys(x.strip() for x in str(env.get(key, "")).replace(";", ",").split(",") if x.strip()))
        try:
            start, end = (int(env.get("SHINKU_QQ_GROUP_DIGEST_START_HOUR", 20)),
                          int(env.get("SHINKU_QQ_GROUP_DIGEST_END_HOUR", 22)))
        except (TypeError, ValueError):
            start, end = 20, 22
        if not 0 <= start < end <= 24:
            start, end = 20, 22
        return cls(str(env.get("SHINKU_QQ_GROUP_DIGEST_ENABLED", "")).lower() in {"1", "true", "yes", "on"},
            ids("SHINKU_QQ_GROUP_DIGEST_GROUP_IDS"),
            tuple(dict.fromkeys(ids("SHINKU_QQ_BOT_IDS") + ids("SHINKU_QQ_GROUP_DIGEST_EXCLUDED_SENDERS"))), start, end)

    def in_window(self, now: datetime) -> bool:
        return self.enabled and self.start_hour <= now.hour < self.end_hour


def process_group_digest(
    *, now: datetime, schedule: DigestSchedule, recipient_ids: tuple[str, ...], memory_db: Path,
    ledger, conversation_runtime, generate: Callable[[dict], object], sender, memory_service=None,
    policy: DigestPolicy = DigestPolicy(), clock: Callable[[], datetime] | None = None,
    allowed: Callable[[], bool] = lambda: True, audit: Callable[..., None] | None = None,
) -> int:
    """返回实际成功发送的回顾数；收件人必须由宿主配置绑定，模型不能改目标。

    生成前/后核对同一会话许可，发送前重读资料和控制开关。已尝试投递不释放日槽，
    写记忆失败也不重发。普通定时问候通过同一日账本的recent_sent协调安静间隔。
    """
    if not schedule.in_window(now) or not schedule.group_ids or len(schedule.group_ids) > 16 or not allowed():
        return 0
    clock = clock or (lambda: now)
    day, sent_count = now.date().isoformat(), 0
    for recipient in tuple(dict.fromkeys(recipient_ids)):
        if not recipient or not allowed():
            continue
        target = f"private:{recipient}"
        key = dict(routine="group_digest", local_date=day, target=target)
        if ledger.already_claimed(**key) or ledger.recent_sent(target=target, since=int(now.timestamp()) - 3600):
            continue
        channel = ConversationChannel("private", recipient)
        permit = conversation_runtime.proactive_permit(channel)
        if not permit.allowed:
            continue
        material = read_digest_material(memory_db, recipient=recipient, allowed_groups=schedule.group_ids,
            now=now, policy=policy, excluded_senders=schedule.excluded_sender_ids)
        if material is None or not ledger.claim(**key, now=int(now.timestamp())):
            continue
        reason, sent, turn_id, message = "digest_silence", False, "", ""
        delivery_status = "suppressed"
        try:
            result = generate({"input_id": material.input_id, **material.payload})
            message = validated_digest(result, material)
            current = clock()
            fresh = read_digest_material(memory_db, recipient=recipient, allowed_groups=schedule.group_ids,
                now=current, cutoff=material.payload["cutoff"], policy=policy,
                excluded_senders=schedule.excluded_sender_ids) if message else None
            if not message:
                continue
            reason = "digest_stale"
            if (not allowed() or not schedule.in_window(current) or current.date().isoformat() != day
                    or fresh is None or fresh.input_id != material.input_id
                    or not conversation_runtime.validate_proactive(permit)):
                continue
            turn_id = f"group-digest:{day}:{recipient}"
            reserved = conversation_runtime.reserve_proactive(permit,
                ConversationScope("private", recipient, recipient), turn_id)
            if reserved is None or not allowed() or not conversation_runtime.validate_proactive(reserved, exclude_turn=turn_id):
                continue
            # 这是一条短回顾，可带自然换行；一次OneBot调用避免半段回顾触发补发。
            delivery_status = "uncertain"
            delivery = deliver(NapCatActionTransport(sender), build_reply(recipient, message, conversation_type="private"))
            delivery_status = delivery.status
            sent, reason = delivery.sent, "sent" if delivery.sent else "digest_delivery_uncertain"
            sent_count += int(sent)
            try:
                conversation_runtime.finish(turn_id, status="chat" if sent else "failed", reason=reason,
                    delivery_status=delivery.status, sent=sent, answer=message if sent else "")
                turn_id = ""
            except Exception:
                reason = "digest_runtime_failed_after_delivery"
                # 实际回执仍有效，继续记录已发送正文；不能把会话存储失败当作发送失败。
            if sent and memory_service is not None:
                source = f"group-digest:{day}:{recipient}"
                try:
                    memory_service.record_turn(conversation_id=recipient, conversation_type="private",
                        message_id=source, source_id=source, role="assistant", text=message,
                        metadata={"sender_id": recipient, "requester_id": recipient, "origin": "group_digest",
                            "delivery_status": "sent", "delivery_message_ids": list(delivery.message_ids),
                            "delivery_content_complete": True, "digest_input_id": material.input_id,
                            "digest_group_ids": [g["group_id"] for g in material.payload["groups"]]})
                except Exception:
                    reason = "digest_memory_failed_after_send"
        except Exception as exc:
            reason = f"digest_error:{type(exc).__name__}"
        finally:
            # 任何异常都终结本次机会，持久日槽留下记录；不能由失败路径再次调用sender。
            if turn_id:
                try:
                    conversation_runtime.finish(turn_id, status="chat" if sent else "ignored" if delivery_status == "suppressed" else "failed", reason=reason,
                        delivery_status=delivery_status, sent=sent, answer=message if sent else "")
                except Exception:
                    pass  # 日槽不释放；会话写入失败不能反推已发送消息消失。
            ledger.finish(**key, now=int(clock().timestamp()), sent=sent, error="" if reason == "sent" else reason,
                status="sent" if sent else "skipped" if reason in {"digest_silence", "digest_stale"} else "failed")
            if audit:
                try:
                    audit("group_digest", target_type="private", sent=sent, reason=reason, input_id=material.input_id)
                except Exception:
                    pass  # 审计失败不允许重发，也不终止其他问候宿主。
    return sent_count
