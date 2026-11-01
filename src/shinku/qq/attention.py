"""模型调用前的注意力门与短窗口消息合并。"""

from __future__ import annotations

import random
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shinku.runtime_limits import RuntimeLimits

from .message import IncomingMessage
from .routing import RouteDecision
from .attention_classifier import LinearAttentionClassifier, attention_features
from .visual_continuity import VISUAL_QUESTION_WINDOW_SECONDS, is_short_visual_question

__all__ = ["AttentionBatch", "AttentionBatcher", "AttentionDecision", "AttentionPolicy"]


@dataclass(frozen=True)
class AttentionDecision:
    should_schedule: bool
    immediate: bool
    score: int
    reason: str
    probability: float = 0.0
    signals: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    classifier_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_schedule": self.should_schedule,
            "immediate": self.immediate,
            "score": self.score,
            "reason": self.reason,
            "probability": self.probability,
            "signals": list(self.signals),
            "features": dict(self.features),
            "classifier_score": self.classifier_score,
        }


@dataclass(frozen=True)
class AttentionSignals:
    recent_directed_turn: bool = False
    autonomous_cooldown: bool = False
    reply_to_delivered_message: bool = False
    recent_bot_exchange: bool = False
    possible_bot_followup: bool = False
    participating_followup: bool = False


class AutonomousAttentionTracker:
    """Bounded per-group state. Cooldown starts only after a successful send."""

    def __init__(self, *, followup_seconds: float = 120.0, cooldown_seconds: float = 90.0) -> None:
        self.followup_seconds = max(0.0, float(followup_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._groups: dict[str, dict[str, Any]] = {}

    def signals(self, message: IncomingMessage, *, now: float) -> AttentionSignals:
        if message.conversation_type != "group":
            return AttentionSignals()
        state = self._groups.get(message.conversation_id)
        if state is None:
            return AttentionSignals()
        directed_at = float(state.get("directed_at", float("-inf")))
        autonomous_at = float(state.get("autonomous_at", float("-inf")))
        # 连续主动发言未得到明确承接时短时退让；只调整本群节奏，不推断喜恶。
        # 0 表示已获得承接，必须真正解除退让，不能再向上取整成一次未承接发言。
        unanswered = min(3, max(0, int(state.get("unanswered", 1))))
        cooldown = self.cooldown_seconds * (2 ** (unanswered - 1)) if unanswered else 0.0
        bot_sent_at = float(state.get("bot_sent_at", float("-inf")))
        followup_at = state.get("autonomous_followup_at")
        delivered_ids = {
            identifier for identifier, sent_at in state.get("delivered_ids", ())
            if 0 <= now - float(sent_at) <= 600.0
        }
        return AttentionSignals(
            # 参与状态仅来自桥接宿主的“承接审查通过 + 完整送达”，不是明确受话/执行授权。
            participating_followup=(
                0 <= now - float(state.get("participants", {}).get(message.sender_id, float("-inf"))) <= self.followup_seconds
            ),
            recent_directed_turn=(
                message.sender_id == state.get("directed_sender")
                and 0 <= now - directed_at <= self.followup_seconds
            ),
            autonomous_cooldown=0 <= now - autonomous_at < cooldown,
            reply_to_delivered_message=bool(delivered_ids.intersection(message.reply_message_ids)),
            recent_bot_exchange=(
                message.sender_id in state.get("bot_recipients", ())
                and 0 <= now - bot_sent_at <= self.followup_seconds
            ),
            # 主动发送只提供一次接话候选机会，不证明对方正在回应，也不重置退让。
            # 同批短消息允许一起进来；首批之后仍按原冷却处理，等待新的送达回执。
            possible_bot_followup=(
                message.sender_id in state.get("autonomous_recipients", ())
                and 0 <= now - autonomous_at <= self.followup_seconds
                and (followup_at is None or 0 <= now - float(followup_at) <= 2.0)
            ),
        )

    def observe(self, message: IncomingMessage, route: RouteDecision, decision: AttentionDecision, *, now: float) -> None:
        if message.conversation_type != "group" or not message.conversation_id:
            return
        state = self._groups.setdefault(message.conversation_id, {})
        if route.addressed or decision.reason == "reply_to_delivered_bot_message":
            state["directed_at"] = float(now)
            state["directed_sender"] = message.sender_id
            state["unanswered"] = 0
        if decision.should_schedule and decision.reason == "possible_bot_followup":
            # 只有真正获准进入生成的候选才消耗机会；预算拒绝不会提前吃掉窗口。
            if state.get("autonomous_followup_at") is None:
                state["autonomous_followup_at"] = float(now)
        state["last_at"] = float(now)
        if len(self._groups) > 256:
            cutoff = float(now) - max(self.followup_seconds, self.cooldown_seconds, 600.0)
            self._groups = {
                group_id: item for group_id, item in self._groups.items()
                if float(item.get("last_at", float("-inf"))) >= cutoff
            }

    def note_delivered(
        self, conversation_id: str, *, recipient_ids: tuple[str, ...],
        message_ids: tuple[str, ...], autonomous: bool, now: float,
    ) -> None:
        key = str(conversation_id or "").strip()
        if not key:
            return
        state = self._groups.setdefault(key, {})
        # An unsolicited interjection is not evidence that its recipient has
        # entered a conversation with us. Only inbound addressing/reply chains
        # (observe) or a response to a directed turn establish that relation.
        if not autonomous:
            state["bot_sent_at"] = float(now)
            state["bot_recipients"] = tuple(dict.fromkeys(str(item) for item in recipient_ids if str(item)))[:8]
        delivered = [
            (identifier, sent_at) for identifier, sent_at in state.get("delivered_ids", ())
            if 0 <= now - float(sent_at) <= 600.0
        ]
        delivered.extend((str(item), float(now)) for item in message_ids if str(item))
        state["delivered_ids"] = tuple({identifier: (identifier, sent_at) for identifier, sent_at in delivered}.values())[-24:]
        if autonomous:
            previous = float(state.get("autonomous_at", float("-inf")))
            # 十分钟没有连续自主发送就重新开始；失败发送不会进入此回调。
            count = int(state.get("unanswered", 0)) if 0 <= now - previous <= 600 else 0
            state["unanswered"] = min(3, count + 1)
            state["autonomous_at"] = float(now)
            state["autonomous_recipients"] = tuple(dict.fromkeys(
                str(item) for item in recipient_ids if str(item)))[:8]
            state["autonomous_followup_at"] = None
        state["last_at"] = float(now)

    def note_continuation(self, conversation_id: str, recipient_id: str, *, now: float) -> None:
        """由宿主在完整送达后续期；候选、模型草稿或单次插话不能创建参与关系。"""
        state = self._groups.get(conversation_id)
        if state is None or not recipient_id:
            return
        participants = {who: at for who, at in state.get("participants", {}).items()
                        if 0 <= now - at <= self.followup_seconds}
        participants[recipient_id] = float(now)
        state["participants"] = dict(sorted(participants.items(), key=lambda item: item[1])[-8:])
        state["unanswered"] = 0


@dataclass(frozen=True)
class AttentionPolicy:
    """无远程调用的注意力策略；本地分类器仅在提供已训练模型时启用。"""

    autonomous_enabled: bool = False
    autonomous_probability: float = 0.0
    random_source: Callable[[], float] = random.random
    autonomous_enabled_getter: Callable[[], bool] | None = None
    response_enabled_getter: Callable[[], bool] | None = None
    autonomous_enabled_for_message_getter: Callable[[IncomingMessage], bool] | None = None
    owner_user_ids: frozenset[str] = frozenset()
    autonomous_max_probability: float = 0.32
    mode: str = "legacy_probability"
    classifier: LinearAttentionClassifier | None = None
    classifier_mode: str = "shadow"
    classifier_error: str = ""
    _candidate_state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    address_aliases: tuple[str, ...] | None = None

    @classmethod
    def from_environment(cls, environ: Any = None) -> "AttentionPolicy":
        """默认轮次判断；保留旧采样参数作显式回退。"""

        env = os.environ if environ is None else environ
        raw = str(env.get("SHINKU_QQ_AUTONOMOUS_REPLY_PROBABILITY", "0") or "0").strip()
        try:
            probability = float(raw)
        except (TypeError, ValueError):
            probability = 0.0
        if not math.isfinite(probability):
            probability = 0.0
        raw_owners = str(env.get("SHINKU_MEMORY_MASTER_USER_IDS", "") or "")
        owners = frozenset(
            item.strip()
            for item in raw_owners.replace("，", ",").replace("；", ",").replace(";", ",").replace("\n", ",").split(",")[:32]
            if item.strip()
        )
        mode = str(env.get("SHINKU_QQ_ATTENTION_MODE", "turn_state") or "turn_state").strip().casefold()
        if mode not in {"turn_state", "legacy_probability"}:
            mode = "turn_state"
        classifier = None
        classifier_error = ""
        model_path = str(env.get("SHINKU_QQ_ATTENTION_MODEL_PATH", "") or "").strip()
        if model_path:
            try:
                classifier = LinearAttentionClassifier.load(model_path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                classifier_error = f"{type(exc).__name__}:{exc}"[:160]
        classifier_mode = str(env.get("SHINKU_QQ_ATTENTION_CLASSIFIER_MODE", "shadow") or "shadow").strip().casefold()
        if classifier_mode not in {"shadow", "active"}:
            classifier_mode = "shadow"
        from shinku.character import load_character
        character_file = str(env.get("SHINKU_CHARACTER_FILE", "") or "").strip()
        aliases = load_character(character_file).aliases if character_file else None
        return cls(
            address_aliases=aliases,
            autonomous_probability=max(0.0, min(1.0, probability)), owner_user_ids=owners,
            mode=mode, classifier=classifier, classifier_mode=classifier_mode,
            classifier_error=classifier_error,
        )

    def evaluate(
        self, message: IncomingMessage, route: RouteDecision, *, signals: AttentionSignals | None = None,
        recent: tuple[dict[str, Any], ...] = (), other_focus: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> AttentionDecision:
        # Commands are the control plane and must remain available while
        # normal replies are paused.  /start is therefore able to recover a
        # stopped instance without requiring a second hidden route.
        if route.reason == "slash_command":
            return AttentionDecision(True, True, route.priority, route.reason)
        if self.response_enabled_getter is not None:
            try:
                if not bool(self.response_enabled_getter()):
                    return AttentionDecision(False, False, 0, "responses_disabled_by_command")
            except Exception:
                return AttentionDecision(False, False, 0, "responses_disabled_by_command")
        if route.should_reply:
            return AttentionDecision(True, True, route.priority, route.reason)
        if message.is_poke:
            # Notices have an explicit recipient. A denied poke is not open
            # group dialogue and must never re-enter via autonomous sampling.
            return AttentionDecision(False, False, 0, "poke_not_for_bot")
        autonomous_enabled = self.autonomous_enabled
        if self.autonomous_enabled_for_message_getter is not None:
            try:
                autonomous_enabled = bool(self.autonomous_enabled_for_message_getter(message))
            except Exception:
                autonomous_enabled = False
        elif self.autonomous_enabled_getter is not None:
            try:
                autonomous_enabled = bool(self.autonomous_enabled_getter())
            except Exception:
                autonomous_enabled = False
        if not autonomous_enabled:
            return AttentionDecision(False, False, 0, "autonomous_reply_disabled")
        if message.conversation_type in {"group", "guild"} and message.has_at_mention:
            # An explicit @ that did not match the bot's direct-reply route is
            # addressed elsewhere. Do not let the random attention sampler
            # turn it into an unsolicited reply on the bot's behalf.
            return AttentionDecision(False, False, 0, "autonomous_at_other_member")
        if not message.text.strip() and not message.has_media:
            return AttentionDecision(False, False, 0, "empty_message")
        local = signals or AttentionSignals()
        if self.mode == "turn_state":
            candidate = self._turn_state_decision(message, local, recent, other_focus or {}, now=now)
            if candidate.should_schedule:
                return self._admit_candidate(message, candidate, now=float(now if now is not None else message.timestamp or 0.0))
            return candidate
        base = max(0.0, min(1.0, float(self.autonomous_probability)))
        if base <= 0.0:
            return AttentionDecision(False, False, 0, "autonomous_probability_miss")
        # An explicit 100% test/operator setting means "always consider";
        # the probabilistic cooldown is only for normal sampled traffic.
        if local.autonomous_cooldown and self.autonomous_probability < 1.0:
            return AttentionDecision(False, False, 0, "autonomous_cooldown")
        bonuses: list[str] = []
        probability = base
        if message.sender_id and message.sender_id in self.owner_user_ids:
            probability += 0.08
            bonuses.append("owner")
        if message.text.rstrip().endswith(("?", "？")):
            probability += 0.05
            bonuses.append("question")
        if local.recent_directed_turn:
            probability += 0.10
            bonuses.append("directed_followup")
        probability = min(1.0, max(base, min(self.autonomous_max_probability, probability)))
        try:
            draw = float(self.random_source())
        except Exception:
            draw = 1.0
        if draw < probability:
            return AttentionDecision(True, False, 20, "autonomous_probability_hit", probability, tuple(bonuses))
        return AttentionDecision(False, False, 0, "autonomous_probability_miss", probability, tuple(bonuses))

    def _turn_state_decision(
        self, message: IncomingMessage, signals: AttentionSignals,
        recent: tuple[dict[str, Any], ...], other_focus: dict[str, Any], *, now: float | None,
    ) -> AttentionDecision:
        """Admit a plausible speaking turn, not a random fraction of traffic."""

        if message.conversation_type not in {"group", "guild"}:
            return AttentionDecision(False, False, 0, "autonomous_not_group")
        if message.has_media and not message.text.strip():
            return AttentionDecision(False, False, 0, "autonomous_image_without_request")
        features = attention_features(
            message, recent=recent, owner=message.sender_id in self.owner_user_ids,
            bot_exchange=signals.recent_bot_exchange,
            directed_followup=signals.recent_directed_turn,
            now=float(now if now is not None else message.timestamp or 0.0),
        )
        score = self.classifier.score(features) if self.classifier is not None else None
        if self._own_image_question(message, recent, now=now, aliases=self.address_aliases):
            # A stale member-to-member focus must not erase a new own-image
            # candidate. This admits attention, never selects image bytes or
            # confirms a reference; the model may select, clarify or stay silent.
            return AttentionDecision(True, False, 35, "own_image_followup",
                signals=("own_image_question",), features=features, classifier_score=score)
        # 引用真红实际发送的消息是更强的新证据，不能被旧的他人对话焦点覆盖。
        if signals.reply_to_delivered_message:
            return AttentionDecision(True, False, 45, "reply_to_delivered_bot_message", signals=("reply_chain",), features=features, classifier_score=score)
        focused_ids = set(str(item) for item in other_focus.get("participant_ids", ()))
        focused_messages = set(str(item) for item in other_focus.get("message_ids", ()))
        if focused_ids and (
            message.sender_id in focused_ids
            or focused_messages.intersection(message.reply_message_ids)
        ):
            return AttentionDecision(False, False, 0, "other_thread_active", features=features, classifier_score=score)
        last = recent[-1] if recent else {}
        last_sender = str(last.get("sender_id") or "")
        last_targets = tuple(str(item) for item in last.get("at_target_ids", ()))
        last_timestamp = float(last.get("timestamp") or 0.0)
        current_timestamp = float(now if now is not None else message.timestamp or 0.0)
        recent_other_exchange = bool(
            last_sender and last_sender != message.sender_id
            and last_targets and message.sender_id in last_targets
            and (not current_timestamp or not last_timestamp or current_timestamp - last_timestamp <= 120)
        )
        if recent_other_exchange:
            return AttentionDecision(False, False, 0, "replying_to_other_member", features=features, classifier_score=score)
        if signals.recent_bot_exchange or signals.recent_directed_turn:
            return AttentionDecision(True, False, 35, "bot_thread_followup", signals=("bot_thread",), features=features, classifier_score=score)
        if signals.participating_followup and not message.reply_message_ids:
            return AttentionDecision(True, False, 35, "participating_followup",
                signals=("participating_followup",), features=features, classifier_score=score)
        if signals.possible_bot_followup and not message.reply_message_ids:
            # 不要求刚收到回复的人再 @/引用一次。只越过自主冷却，不越过上方的
            # 他人对话焦点或显式引用对象；后续仍走候选限额、语义审查与告别收尾。
            return AttentionDecision(True, False, 30, "possible_bot_followup",
                signals=("possible_bot_followup",), features=features, classifier_score=score)
        if signals.autonomous_cooldown:
            return AttentionDecision(False, False, 0, "autonomous_cooldown", features=features, classifier_score=score)
        if score is not None and self.classifier_mode == "active":
            if score >= 0.80:
                return AttentionDecision(True, False, 25, "classifier_candidate", features=features, classifier_score=score)
            if score <= 0.20:
                return AttentionDecision(False, False, 0, "classifier_abstain", features=features, classifier_score=score)
        # Statements and exclamations can be natural speaking turns too. Busy
        # traffic is context/priority, not a hard semantic veto. Model silence
        # still costs a call; admission below bounds candidate batches.
        text = str(message.text_without_at_mentions or "").strip()
        is_question = text.endswith(("?", "？"))
        recent_window = [
            item for item in recent[-6:]
            if not current_timestamp or not item.get("timestamp")
            or 0 <= current_timestamp - float(item["timestamp"]) <= 20.0
        ]
        owner = message.sender_id in self.owner_user_ids
        reason = ("owner_open_question" if owner else "group_open_question") if is_question else "group_open_turn"
        busy = len(recent_window) >= 4
        return AttentionDecision(True, False, 15 if busy else 20, reason,
            signals=(("open_question",) if is_question else ("open_turn",)) + (("busy_group",) if busy else ()),
            features=features, classifier_score=score)

    @staticmethod
    def _own_image_question(message: IncomingMessage, recent, *, now: float | None, aliases=None) -> bool:
        if not message.sender_id or not is_short_visual_question(message.text_without_at_mentions, aliases=aliases):
            return False
        previous = next((item for item in reversed(recent)
                         if str(item.get("sender_id") or "") == message.sender_id), None)
        if not previous or not previous.get("has_image") or previous.get("at_target_ids"):
            return False
        # An explicit reply to a different message keeps its chosen recipient.
        reply_ids = tuple(message.reply_message_ids)
        if reply_ids and set(reply_ids) != {str(previous.get("message_id") or "")}:
            return False
        if previous.get("reply_message_ids"):
            return False
        current = float(now if now is not None else message.timestamp or 0.0)
        source = float(previous.get("timestamp") or 0.0)
        return bool(current and source and 0 <= current - source <= VISUAL_QUESTION_WINDOW_SECONDS)

    def _admit_candidate(self, message: IncomingMessage, candidate: AttentionDecision, *, now: float) -> AttentionDecision:
        # 新插话最多三批/分钟；已参与/明确互动的承接共享八批上限，不因聊得快耗尽插话额度。
        # 两类仍共享全局24批和每批2秒/8条的界限；参与候选不升级为明确受话或工具权限。
        windows = self._candidate_state.setdefault("windows", {})
        windows = {key: row for key, row in windows.items() if 0 <= now - row["last"] < 60}
        self._candidate_state["windows"] = windows
        global_slots = [stamp for stamp in self._candidate_state.get("global", []) if 0 <= now - stamp < 60]
        self._candidate_state["global"] = global_slots
        key = (message.conversation_type, message.conversation_id)
        row = windows.get(key)
        if row and 0 <= now - row["last"] <= 2 and row["messages"] < 8:
            row["messages"] += 1
            return candidate
        slots = [stamp for stamp in row["slots"] if 0 <= now - stamp < 60] if row else []
        ambient = [stamp for stamp in row.get("ambient", row["slots"]) if 0 <= now - stamp < 60] if row else []
        continuing = candidate.reason in {"participating_followup", "bot_thread_followup", "reply_to_delivered_bot_message"}
        if (len(slots) >= 8 or not continuing and len(ambient) >= 3
                or len(global_slots) >= 24 or len(windows) >= 256 and key not in windows):
            return AttentionDecision(False, False, 0, "autonomous_admission_budget", features=candidate.features,
                                     classifier_score=candidate.classifier_score)
        windows[key] = {"last": now, "messages": 1, "slots": [*slots, now],
                        "ambient": ambient if continuing else [*ambient, now]}
        global_slots.append(now)
        return candidate


@dataclass(frozen=True)
class AttentionBatch:
    conversation_id: str
    messages: tuple[IncomingMessage, ...]
    decisions: tuple[AttentionDecision, ...]
    primary_index: int

    @property
    def primary_message(self) -> IncomingMessage:
        return self.messages[self.primary_index]

    @property
    def combined_text(self) -> str:
        messages = [message for message in self.messages if message.text.strip()]
        speakers = {str(message.sender_id or "").strip() for message in messages}
        if len(speakers) <= 1 or not any(message.conversation_type == "group" for message in messages):
            return "\n".join(message.text.strip() for message in messages)
        # A short-window batch may contain several group members. Never pass
        # their unlabelled words to memory/planning as one person's speech.
        return "\n".join(
            f"{message.sender_label or message.sender_id}（QQ {message.sender_id}）：{message.text.strip()}"
            for message in messages
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_ids": [message.message_id for message in self.messages],
            "primary_message_id": self.primary_message.message_id,
            "combined_text": self.combined_text,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


class AttentionBatcher:
    """把同一会话短时间内的已调度消息合并成一轮。"""

    def __init__(self, *, window_seconds: float | None = None) -> None:
        configured = (
            RuntimeLimits.from_environment().qq_batch_window_seconds
            if window_seconds is None
            else window_seconds
        )
        self.window_seconds = max(0.0, float(configured))
        self._conversation_id = ""
        self._messages: list[IncomingMessage] = []
        self._decisions: list[AttentionDecision] = []
        self._last_at: float | None = None

    def push(
        self,
        message: IncomingMessage,
        decision: AttentionDecision,
        *,
        at: float,
    ) -> AttentionBatch | None:
        timestamp = float(at)
        if not decision.should_schedule:
            return None
        closed: AttentionBatch | None = None
        # Control messages are never merged with conversational turns.  A
        # command may arrive immediately after an @ message (or vice versa);
        # keeping the slash boundary intact is more important than the short
        # conversational batching window.
        if self._messages and (
            decision.reason == "slash_command"
            or self._decisions[-1].reason == "slash_command"
        ):
            closed = self.flush()
        # Pokes are transient notices, not dialogue content.  Keep them out
        # of the same short-window turn as a nearby text message (and do not
        # collapse two pokes into one reply), otherwise a later message can
        # inherit the poke's lightweight/no-memory semantics.
        if self._messages and (
            bool(getattr(message, "is_poke", False))
            or any(bool(getattr(item, "is_poke", False)) for item in self._messages)
        ):
            closed = self.flush()
        if self._messages and (
            message.conversation_id != self._conversation_id
            or self._last_at is None
            or timestamp - self._last_at > self.window_seconds
        ):
            closed = self.flush()
        if not self._messages:
            self._conversation_id = message.conversation_id
        self._messages.append(message)
        self._decisions.append(decision)
        self._last_at = timestamp
        return closed

    def flush(self) -> AttentionBatch | None:
        if not self._messages:
            return None
        primary_index = max(
            range(len(self._decisions)),
            key=lambda index: (self._decisions[index].score, -index),
        )
        result = AttentionBatch(
            conversation_id=self._conversation_id,
            messages=tuple(self._messages),
            decisions=tuple(self._decisions),
            primary_index=primary_index,
        )
        self._conversation_id = ""
        self._messages.clear()
        self._decisions.clear()
        self._last_at = None
        return result
