"""Transport-independent Agent boundary; no QQ or persona bootstrap."""

from .contracts import AgentBoundaryError, AgentRequest, AgentTurnRequest, AgentScope, AgentSupplement, TaskHandle
from .execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from .service import AgentClient, AgentService

__all__ = ["AgentBoundaryError", "AgentRequest", "AgentTurnRequest", "AgentScope", "AgentSupplement", "TaskHandle",
           "AgentExecutor", "ExecutionGrant", "ExecutionProfile", "AgentClient", "AgentService"]
