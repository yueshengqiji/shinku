"""QQ 消息回合处理流水线。

职责与上下游：
- 职责：定义 QQ 单回合（Turn）处理的主阶段顺序，并提供轻量级控制分支拦截。
- 上游：由 `src/shinku/qq/agent_bridge.py` 的 `QQAgentBridge.handle_turn` 构建依赖后调用。
- 下游：按照“控制拦截 -> 上下文准备 -> 审批恢复 -> 路由决策 -> 执行/交付”的固定时序推进，
  分别调用 `turn_preparation.py`、`approval_resume.py`、`turn_routing.py` 及执行/投递模块。

阅读入口：
- 核心流水线入口：`handle_qq_turn`
- 控制指令与轻量路由拦截：`route_control_turn`
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

from .commands import QQCommandInterpreter, QQCommandState
from .message import IncomingMessage
from .reply_models import QQAgentReply
from .routing import ReplyRoutePolicy
from .turn_models import QQPreparedTurn, QQTurnDecision, QQTurnEffects
from .turns import QQTurn


@dataclass(frozen=True)
class QQControlRouteDependencies:
    """控制消息与轻量拦截的依赖契约。

    状态归属：
    - `command_state` 由宿主持有，管理真红当前运行状态（运行/停机/权限模式等）；
    - `command_interpreter` 为无状态或单例解释器，负责解析并执行斜杠指令；
    - `effects` 提供副作用桥接（日志追踪、审计步进、文本交付等）。
    """
    bot_ids: Collection[str]
    command_state: QQCommandState
    command_interpreter: QQCommandInterpreter
    command_text: Callable[[QQTurn], str]
    sender_id: Callable[[QQTurn], str]
    effects: QQTurnEffects


@dataclass(frozen=True)
class QQTurnPipelineDependencies:
    """流水线主阶段依赖契约。

    设计原因：
    - 将 `QQAgentBridge` 庞大的组件解耦为高内聚的阶段函数，使回合生命周期具有明确的阶段契约；
    - 本容器描述阶段调用关系；准备、审批和投递等阶段仍会通过依赖读写状态。
    - 依赖注入不提供回合级互斥，并发约束由被调用组件的实际实现决定。
    """
    control: Callable[[QQTurn, str], QQAgentReply | None]
    prepare: Callable[[QQTurn, str], QQPreparedTurn]
    resume_approval: Callable[[QQPreparedTurn], QQAgentReply | None]
    select: Callable[[QQPreparedTurn], QQTurnDecision | QQAgentReply]
    dispatch_agent: Callable[[QQTurnDecision], QQAgentReply]
    deliver_direct: Callable[[QQTurnDecision], QQAgentReply]


def handle_qq_turn(
    turn: QQTurn,
    *,
    trace_id: str,
    dependencies: QQTurnPipelineDependencies,
) -> QQAgentReply:
    """驱动单次回合的主时序调度。

    阶段顺序与设计原因：
    1. `control`: 控制与轻量指令必须在最前执行。
       - 原因：若用户发送 `/stop`、`/permission` 等管理指令或心跳检测，必须立即响应，
         严禁先调用大模型、检索向量记忆或占用执行配额，防止系统故障时因模型不可用导致无法停机。
    2. `prepare`: 组装多模态上下文、历史消息、受众身份与记忆检索（进入 `turn_preparation.py`）。
    3. `resume_approval`: 挂起审批恢复判定（进入 `approval_resume.py`）。
       - 原因：若用户当前输入是针对此前高危工具调用的授权回复（如“同意”/“拒绝”），必须优先挂接
         原切片会话与原任务预算，不得当作全新问题再次经历普通路由。
    4. `select`: 对话意图分流（进入 `turn_routing.py`）。
       - 区分直接回复（闲聊/轻量感知/历史问答）与需要多步工具规划的复杂任务。
    5. 执行分流：
       - `requires_agent == True`: 调度到 Agent 执行引擎（`dispatch_agent`）；
       - `requires_agent == False`: 投递直接对话结果（`deliver_direct`）。

    状态归属与副作用：
    - 本函数本身不持有状态；
    - 返回 `QQAgentReply` 表示本回合处理完成（成功、失败、忽略或转入后台排队）。
    """
    deps = dependencies
    # 阶段 1：控制与指令拦截
    reply = deps.control(turn, trace_id)
    if reply is not None:
        return reply

    # 阶段 2：上下文与记忆准备
    prepared = deps.prepare(turn, trace_id)

    # 阶段 3：工具调用审批恢复（若有挂起审批且此轮为确认回复）
    reply = deps.resume_approval(prepared)
    if reply is not None:
        return reply

    # 阶段 4：路由与意图决策
    selection = deps.select(prepared)
    if isinstance(selection, QQAgentReply):
        return selection

    # 阶段 5：任务派发或直接回复投递
    if selection.requires_agent:
        return deps.dispatch_agent(selection)
    return deps.deliver_direct(selection)


def route_control_turn(
    turn: QQTurn,
    *,
    trace_id: str,
    deps: QQControlRouteDependencies,
) -> QQAgentReply | None:
    """控制指令、戳一戳（Poke）与启停状态拦截。

    设计原因与关键边界：
    1. 戳一戳（Poke）边界：
       - 群聊中的 Poke 往往只发给特定人。必须严格核对是否指向真红自己的 `bot_ids`。
       - 非指向真红的 Poke 必须在此被抑制（suppressed/ignored），防止误唤醒和打扰其他群员。
    2. 斜杠指令（Slash Command）边界：
       - 私聊允许直接以 `/` 开头；群聊则要求必须显式 `@真红` + 指令，防止群内其他机器人的指令互串。
       - 执行命令不计入记忆（`record_memory=False`），不影响闲聊节奏（`track_chat_flow=False`）。
    3. 响应开关（CommandState.responses_enabled）：
       - 若处于全局暂停或禁用状态，直接提前返回 `stopped` 或 `agent_disabled` 回执，拒绝进入后续管线。
    """
    poke_policy = ReplyRoutePolicy(bot_ids=deps.bot_ids)
    if any(bool(getattr(message, "is_poke", False)) and (
        not isinstance(message, IncomingMessage) or not poke_policy.decide(message).should_reply
    ) for message in turn.messages):
        deps.effects.trace(trace_id, "route", kind="ignored", reason="poke_not_for_bot")
        return QQAgentReply("ignored", reason="poke_not_for_bot", delivery_status="suppressed")

    # 斜杠指令分支：故意放在记忆检索、工具审批和模型循环之前。
    # 私聊可直接使用前导斜杠；群聊必须明确 @ 本机器人。
    command_text = deps.command_text(turn)
    if command_text:
        deps.effects.conversation_step(trace_id, route="command")
        deps.effects.trace(trace_id, "route", kind="slash_command")
        sender_id = deps.sender_id(turn)
        outcome = deps.command_interpreter.execute(
            command_text,
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
            sender_id=sender_id,
        )
        if outcome is None:
            return QQAgentReply("command", reason="slash_command_not_parsed")
        if outcome.quiet or not outcome.reply.strip():
            return QQAgentReply("command", reason=outcome.reason or outcome.status)
        return deps.effects.deliver(
            turn,
            outcome.reply,
            agent_status="command",
            reason=outcome.reason or outcome.status,
            record_memory=False,
            track_chat_flow=False,
            trace_id=trace_id,
        )

    # 全局响应能力检查：检查管理员是否通过指令关闭了响应或停机
    if not deps.command_state.responses_enabled():
        deps.effects.trace(
            trace_id,
            "route",
            kind="disabled",
            stopped=deps.command_state.is_stopped(),
            agent_enabled=deps.command_state.is_agent_enabled(),
        )
        return QQAgentReply(
            "stopped" if deps.command_state.is_stopped() else "agent_disabled",
            reason="responses_disabled_by_command",
        )
    return None
