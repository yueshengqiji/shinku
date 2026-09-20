"""Contracts shared by Shinku's runtime, hosts and adapters.

This package holds the value objects and protocol constants that describe how
the parts of Shinku talk to one another.  Nothing in here reaches out to a host,
a database, a model SDK or a workflow engine, so every layer - including the
standalone hosts - can depend on it without pulling the rest of the runtime in.

The package is documented by the public runtime and contract documentation.
"""

from __future__ import annotations

from .capability import (
    CapabilityDescriptor,
    CapabilityIOSlot,
    CapabilityManifest,
    CapabilityManifestError,
    CapabilityProtocolError,
    CapabilityResult,
    ConfirmPolicy,
    EndpointConfig,
    HealthConfig,
    HealthStatus,
    InvalidManifest,
    InvocationContext,
    RiskLevel,
    SourceLayer,
    TierConfig,
    TriggerConfig,
)
from .http import json_response
from .retrieval import RetrievalPipelineResult
from .tasks import (
    AGENT_ALIASES,
    AGENT_ALLOWED_TOOLS,
    WorkerDelegation,
    WorkerRunSummary,
)

__all__ = [
    "AGENT_ALIASES",
    "AGENT_ALLOWED_TOOLS",
    "CapabilityDescriptor",
    "CapabilityIOSlot",
    "CapabilityManifest",
    "CapabilityManifestError",
    "CapabilityProtocolError",
    "CapabilityResult",
    "ConfirmPolicy",
    "EndpointConfig",
    "HealthConfig",
    "HealthStatus",
    "InvalidManifest",
    "InvocationContext",
    "RetrievalPipelineResult",
    "RiskLevel",
    "SourceLayer",
    "TierConfig",
    "TriggerConfig",
    "WorkerDelegation",
    "WorkerRunSummary",
    "json_response",
]
