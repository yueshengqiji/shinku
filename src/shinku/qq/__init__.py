"""QQ/消息适配边界。"""

from .message import IncomingMessage, MessageSegment, is_group_increase_event, is_poke_event
from .welcome import GroupWelcomePolicy
from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .routing import ReplyRoutePolicy, RouteDecision
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, OutgoingSegment, build_reply, deliver
from .reply_host import ReplyHost, ReplyShape
from .chat_flow import ChatFlowController, ChatFlowDecision, append_chat_flow_guidance
from .commands import (
    CommandOutcome,
    QQCommandInterpreter,
    QQCommandState,
    SlashCommand,
    command_text_for_conversation,
    is_slash_command,
    parse_slash_command,
)
from .adapter import QQIngressAdapter, QQIngressResult
from .ingress_guard import IngressAdmission, IngressPolicy
from .group_context import GroupContextTracker
from .host import NapCatHost, NapCatHostResult
from .agent_bridge import QQAgentBridge, QQAgentReply
from .assembly import QQAgentAssembly, QQAgentConfig, QQAgentConfigError, assemble_qq_agent, load_qq_agent_config
from .turns import NapCatTurnDispatcher, QQTurn, TurnDispatchReceipt
from .config import NapCatConnectionConfig
from .webhook import create_napcat_webhook_router
from .napcat import (
    NapCatActionTransport,
    NapCatEventAdapter,
    NapCatEventDecoder,
    NapCatHttpActionCaller,
    NapCatVisualBatch,
    NapCatVisualInput,
    NapCatVisualInputBridge,
    NapCatImageMaterializer,
)

__all__ = [
    "AttentionBatch",
    "AttentionBatcher",
    "AttentionDecision",
    "AttentionPolicy",
    "DeliveryResult",
    "DeliveryTransport",
    "IncomingMessage",
    "MessageSegment",
    "is_group_increase_event",
    "is_poke_event",
    "GroupWelcomePolicy",
    "IngressAdmission",
    "IngressPolicy",
    "GroupContextTracker",
    "NapCatActionTransport",
    "NapCatEventAdapter",
    "NapCatEventDecoder",
    "NapCatHttpActionCaller",
    "NapCatVisualBatch",
    "NapCatVisualInput",
    "NapCatVisualInputBridge",
    "NapCatImageMaterializer",
    "QQIngressAdapter",
    "QQIngressResult",
    "NapCatHost",
    "NapCatHostResult",
    "QQAgentBridge",
    "QQAgentReply",
    "QQAgentAssembly",
    "QQAgentConfig",
    "QQAgentConfigError",
    "assemble_qq_agent",
    "load_qq_agent_config",
    "NapCatTurnDispatcher",
    "QQTurn",
    "TurnDispatchReceipt",
    "NapCatConnectionConfig",
    "create_napcat_webhook_router",
    "OutgoingMessage",
    "OutgoingSegment",
    "ReplyRoutePolicy",
    "RouteDecision",
    "ReplyHost",
    "ReplyShape",
    "ChatFlowController",
    "ChatFlowDecision",
    "append_chat_flow_guidance",
    "CommandOutcome",
    "QQCommandInterpreter",
    "QQCommandState",
    "SlashCommand",
    "command_text_for_conversation",
    "is_slash_command",
    "parse_slash_command",
    "build_reply",
    "deliver",
]
