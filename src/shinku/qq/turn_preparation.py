"""准备归属化 QQ 上下文并持久化原始用户消息。

职责与上下游：
- 职责：从合并的 `QQTurn` 中提炼多模态视觉输入、受众身份与交接上下文，调用 `MemoryService` 检索长期/短期记忆，
  并将当前回合的原始消息进行事实入库归档，最终组装出模型调用与执行循环所需的标准化上下文字典。
- 上游：由 `turn_pipeline.py` 的 `handle_qq_turn` 在通过控制拦截后作为阶段 2 调用。
- 下游：构建出的 `QQPreparedTurn` 传递给 `approval_resume.py` 和 `turn_routing.py`。

阅读入口：
- 核心准备入口：`prepare_qq_turn`
- 记忆检索与存盘：`_prepare_memory`、`_record_user_messages`
- 综合上下文构造：`_build_turn_context`
"""

from __future__ import annotations

from shinku.phase_timing import timed

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
import re
import time
from typing import Any

from shinku.conversation import ConversationRuntime
from shinku.memory import MemoryService
from shinku.memory.models import MemoryContext, MemoryDecision
from shinku.memory.history_requirement import host_requests_history_check

from .chat_flow import ChatFlowController, ChatFlowDecision
from .commands import QQCommandState
from .turn_models import QQPreparedTurn, QQTurnEffects
from .turns import QQTurn
from .attachment_labels import attachment_label


@dataclass(frozen=True)
class QQTurnPreparationDependencies:
    bot_ids: Collection[str]
    command_state: QQCommandState
    chat_flow: ChatFlowController
    memory_service: MemoryService | None
    conversation_runtime: ConversationRuntime
    is_poke_turn: Callable[[QQTurn], bool]
    images: Callable[[QQTurn], list[dict[str, Any]]]
    audience_context: Callable[[QQTurn], dict[str, Any]]
    owner_handoff_context: Callable[..., str]
    memory_message_meta: Callable[..., dict[str, Any]]
    primary_task_request: Callable[[QQTurn], str]
    recent_context: Callable[[QQTurn], str]
    message_mentions_bot: Callable[[Any], bool]
    trace_memory: Callable[..., None]
    task_scope: Callable[[QQTurn], tuple[str, str]]
    event_context: Callable[[QQTurn], dict[str, Any]]
    group_context: Callable[..., str]
    self_state_snapshot: Callable[[], Mapping[str, Any]]
    request_scope: Callable[..., Any]
    conversation_scope: Callable[..., Any]
    effects: QQTurnEffects
    file_reader: Any = None


def prepare_qq_turn(
    turn: QQTurn,
    *,
    trace_id: str,
    deps: QQTurnPreparationDependencies,
) -> QQPreparedTurn:
    """提取会话上下文、检索记忆并持久化原始消息。

    设计原因与时序：
    1. 戳一戳（Poke）特化：若为 Poke 事件，不开启闲聊流控（`chat_flow` 置空），避免对物理动作套用对话回复频率；
    2. 视觉溯源：追踪图片附件就绪状态与上下文继承来源（`visual_context_inherited`），确保多模态大模型能引用正确图片；
    3. 记忆与归档：在 `_prepare_memory` 中检索记忆，并确保不论检索成功与否均存盘真实输入；
    4. 作用域绑定：解析任务请求者与会话标识（`task_scope`），为并发隔离与任务归属提供唯一凭据。
    """
    is_poke = deps.is_poke_turn(turn)
    deps.effects.trace(trace_id, "route", kind="chat", is_poke=is_poke)
    chat_flow = None if is_poke else deps.chat_flow.begin_turn(turn)
    images = deps.images(turn)
    visual_inputs = [item for _, inputs in turn.visual_inputs for item in inputs]
    audience_context = deps.audience_context(turn)
    owner_handoff_context = deps.owner_handoff_context(
        turn,
        address_only=bool(chat_flow and chat_flow.address_only),
        audience_context=audience_context,
        trace_id=trace_id,
    )
    deps.effects.trace(
        trace_id,
        "visual",
        attachment_count=len(visual_inputs),
        ready_count=sum(1 for item in visual_inputs if bool(getattr(item, "ready", False))),
        model_image_count=len(images),
        inherited=bool(turn.visual_context_inherited),
        source_message_id=turn.visual_context_source_message_id,
    )
    memory_context = _prepare_memory(turn, is_poke=is_poke, trace_id=trace_id, deps=deps)
    requester, session = deps.task_scope(turn)
    context = _build_turn_context(
        turn, trace_id=trace_id, is_poke=is_poke, images=images, memory_context=memory_context,
        chat_flow=chat_flow, audience_context=audience_context,
        owner_handoff_context=owner_handoff_context, requester=requester, session=session, deps=deps,
    )
    # 读取归可信宿主所有，只消费主消息的已收文件/明确引用；群成员不会因此获得执行工具权限。
    # 正文只留在本回合；来源和版本随实际送达回执保存，不在被动观察群消息时下载。
    if (deps.file_reader is not None and not is_poke
            and deps.command_state.is_agent_enabled() and not deps.command_state.is_stopped()):
        from .file_continuity import prepare_file_inputs
        files, sources = prepare_file_inputs(turn, deps.file_reader, deps.memory_service)
        if files:
            turn = replace(turn, file_input_sources=sources)
            context["qq_turn"] = turn
            context["qq_file_context"] = files
            deps.effects.trace(trace_id, "qq_file_read", count=len(files),
                statuses=[row["status"] for row in files],
                reasons=[row.get("reason", "") for row in files])
    return QQPreparedTurn(turn, trace_id, is_poke, chat_flow, audience_context, context, requester, session)


@timed('memory')
def _prepare_memory(
    turn: QQTurn,
    *,
    is_poke: bool,
    trace_id: str,
    deps: QQTurnPreparationDependencies,
) -> Any:
    """检索相关记忆上下文并执行入站消息记录。

    设计原因与异常边界：
    1. 戳一戳（Poke）隔离：Poke 不检索也不记录记忆，防止将动作干扰为文字事实；
    2. 读写隔离与容错边界：若记忆检索（prepare_turn）发生异常，必须回退到保底策略（正则识别历史词），
       绝不能中断下方的消息存盘（`_record_user_messages`），保证用户真实发言事实不丢失；
    3. 审计留痕：检索耗时与存盘耗时分别记录到 audit 中，暴露给 trace 供性能诊断。
    """
    memory_context = None
    if deps.memory_service is not None and not is_poke:
        started = time.monotonic()
        message_meta = deps.memory_message_meta(turn)
        message_meta["shinku_involved"] = True
        try:
            # 到达此分支说明回复路由已判定真红参与交互。
            # 该标记供主人记忆过滤器使用，不对模型暴露。
            message_meta["shinku_involved"] = True
            memory_context = deps.memory_service.prepare_turn(
                conversation_id=turn.conversation_id,
                conversation_type=turn.conversation_type,
                user_message=deps.primary_task_request(turn),
                recent_context=deps.recent_context(turn),
                message_meta=message_meta,
            )
        except Exception as exc:
            query = deps.primary_task_request(turn)
            needs_history = bool(re.search(r"记得|之前|以前|说过|上次|昨天|哪天", query))
            memory_context = MemoryContext(
                decision=MemoryDecision(need_retrieval=needs_history,
                    history_mode="retrieve" if needs_history else "none", reason="memory_read_failed"),
                audit={"failure_stage":"prepare", "error_type":type(exc).__name__,
                       "absence_means":"unknown"})
            deps.effects.trace(trace_id, "memory_failure", stage="prepare", error_type=type(exc).__name__)
        query_ms = round((time.monotonic()-started)*1000, 3)
        write_start = time.monotonic()
        write_error = ""
        try:
            # 检索失败绝不能阻止对实际用户输入的持久化入库
            _record_user_messages(turn, message_meta, deps)
        except Exception as exc:
            write_error = type(exc).__name__
            deps.effects.trace(trace_id, "memory_failure", stage="record", error_type=write_error)
        if isinstance(memory_context, MemoryContext):
            memory_context = replace(memory_context, audit={**memory_context.audit,
                "prepare_ms":query_ms, "record_ms":round((time.monotonic()-write_start)*1000,3),
                "record_error_type":write_error})
    deps.trace_memory(trace_id, memory_context)
    return memory_context


def _record_user_messages(
    turn: QQTurn,
    message_meta: dict[str, Any],
    deps: QQTurnPreparationDependencies,
) -> None:
    if str(turn.conversation_type or "").casefold() == "group":
        # A merged turn is not a single speaker. Persist each
        # original QQ message under its own ID and author.
        for message in turn.messages:
            item_id = str(getattr(message, "message_id", "") or "").strip()
            if not item_id and len(turn.messages) == 1:
                # Minimal adapters may put identity only on QQTurn.
                # A single message has an unambiguous primary ID;
                # never lend that identity to a merged multi-author batch.
                item_id = str(turn.primary_message_id or "").strip()
            item_meta = deps.memory_message_meta(turn, message=message)
            item_meta["shinku_involved"] = bool(
                item_id == turn.primary_message_id
                or deps.message_mentions_bot(message)
            )
            deps.memory_service.record_turn(
                conversation_id=turn.conversation_id,
                conversation_type=turn.conversation_type,
                message_id=item_id,
                text=str(getattr(message, "text", "") or "").strip()
                or attachment_label(getattr(message, "attachments", ()))
                or ("[图片附件]" if turn.visuals_for(item_id) else ""),
                metadata=item_meta,
            )
    else:
        deps.memory_service.record_turn(
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
            message_id=turn.primary_message_id,
            text=turn.combined_text or "".join(dict.fromkeys(
                attachment_label(getattr(message, "attachments", ())) for message in turn.messages))
                or ("[图片附件]" if turn.visual_inputs else ""),
            metadata=message_meta,
        )


def _build_turn_context(
    turn: QQTurn,
    *,
    trace_id: str,
    is_poke: bool,
    images: list[dict[str, Any]],
    memory_context: Any,
    chat_flow: ChatFlowDecision | None,
    audience_context: dict[str, Any],
    owner_handoff_context: str,
    requester: str,
    session: str,
    deps: QQTurnPreparationDependencies,
) -> dict[str, Any]:
    context = {
        "conversation_turn_id": trace_id,
        "qq_turn": turn,
        "user_images": images,
        "memory_context": memory_context.as_dict() if memory_context is not None else {},
        "chat_flow": chat_flow.as_dict() if chat_flow is not None else {},
        "qq_event": deps.event_context(turn) if is_poke else {},
        "group_context": deps.group_context(turn, bot_ids=deps.bot_ids),
        "owner_handoff_context": owner_handoff_context,
        "self_state": deps.self_state_snapshot(),
        "audience_context": audience_context,
        "agent_requester": requester,
        "agent_session": session,
        "agent_authority_revision": deps.command_state.authority_revision(),
        "agent_permission_mode": deps.command_state.permission_mode(),
    }
    context["agent_request_scope"] = deps.request_scope(turn)
    # This is a host intent, not a model-selected contract effect. A native
    # memory call must not evade history checks by choosing generic observe.
    # Both `history_mode` predicates this module needs — "gate the reply" and
    # "prove it from originals" — are defined in memory.history_requirement;
    # do not re-spell the mode sets here.
    memory_decision = context["memory_context"].get("decision", {})
    context["agent_history_query_request"] = (
        isinstance(memory_decision, Mapping) and host_requests_history_check(memory_decision)
    )
    try:
        context["conversation_runtime"] = deps.conversation_runtime.prompt_context(
            deps.conversation_scope(turn), exclude_turn=trace_id)
    except ValueError:
        context["conversation_runtime"] = {"available": False}
    return context
