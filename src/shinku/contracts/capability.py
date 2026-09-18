"""Value objects describing what a capability provider offers.

A provider is described by a manifest: who it is, where it lives, how it can be
reached, and what it can be asked to do.  Endpoint, health, tier, trigger and
I/O-slot records compose upwards into a descriptor, and descriptors compose into
the manifest itself.

Every record here is frozen and reaches for nothing outside the standard
library.  An adapter is free to talk to the network, to a subprocess or to a
workflow engine; the *description* of an adapter never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

__all__ = [
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
    "RiskLevel",
    "SourceLayer",
    "TierConfig",
    "TriggerConfig",
]

SourceLayer = Literal["builtin", "profile"]
RiskLevel = Literal["low", "medium", "high"]
ConfirmPolicy = Literal["never", "first_time", "always"]


@dataclass(frozen=True)
class EndpointConfig:
    """Where a provider is reached, and whether that must stay on loopback."""

    url: str
    loopback_only: bool = False
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HealthConfig:
    """How to probe a provider, and which statuses count as healthy."""

    method: str
    path: str
    timeout_seconds: float
    expect_status: tuple[int, ...]
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TierConfig:
    """One named quality tier a provider may be invoked at."""

    id: str
    label: str
    preset: Mapping[str, Any]
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TriggerConfig:
    """The condition under which a capability is allowed to fire."""

    kind: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CapabilityIOSlot:
    """One named input or output of a capability invocation.

    ``max_bytes`` is ``None`` when the slot carries no size ceiling; ``delivery``
    names the transport the slot is expected to arrive over, and is empty when
    the provider does not distinguish transports.
    """

    name: str
    kind: str
    required: bool = False
    max_bytes: int | None = None
    delivery: str = ""
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Everything a caller needs to know before invoking one capability.

    ``risk`` and ``confirm`` drive the approval policy; ``effects`` lists the
    externally visible consequences, and ``visible_in`` / ``prompt_exposed``
    decide which surfaces are told the capability exists.
    """

    id: str
    display_name: str
    short_hint: str
    visible_in: tuple[str, ...]
    prompt_exposed: bool
    risk: RiskLevel
    confirm: ConfirmPolicy
    effects: tuple[str, ...]
    trigger: TriggerConfig | None
    inputs: tuple[CapabilityIOSlot, ...]
    outputs: tuple[CapabilityIOSlot, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CapabilityManifest:
    """A parsed, validated description of one provider.

    ``source_path`` and ``source_layer`` together answer "where did this come
    from" - a file on disk plus whether it is a built-in or a user profile.
    ``endpoint`` and ``health`` are optional because a local provider may need
    neither.
    """

    schema: str
    provider_id: str
    provider_type: str
    display_name: str
    endpoint: EndpointConfig | None
    health: HealthConfig | None
    tiers: tuple[TierConfig, ...]
    capabilities: tuple[CapabilityDescriptor, ...]
    secrets: tuple[str, ...]
    source_path: Path
    source_layer: SourceLayer
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class InvalidManifest:
    """A manifest that failed validation, kept with enough context to report it.

    Validation returns this instead of raising, so one bad provider file does not
    stop the rest of the registry from loading.
    """

    source_path: Path
    source_layer: SourceLayer
    reason: str
    detail: str = ""
    provider_id: str = ""


@dataclass(frozen=True)
class HealthStatus:
    """The outcome of one health probe against a provider."""

    ok: bool
    status: str
    reason: str = ""


@dataclass(frozen=True)
class CapabilityResult:
    """The outcome of one capability invocation.

    ``is_error`` is the flag callers branch on; ``content`` carries whatever the
    adapter produced, and ``status`` / ``reason`` add machine-readable and
    human-readable detail.
    """

    is_error: bool
    content: Any = None
    status: str = ""
    reason: str = ""


@dataclass(frozen=True)
class InvocationContext:
    """Who and where an invocation is happening for.

    Every field defaults to empty, so a call with no caller information is
    representable without inventing placeholder values.
    """

    profile_user_id: str = ""
    session_id: str = ""
    client_mode: str = ""
    qq_user_id: str = ""
    master_qq: str = ""


class CapabilityManifestError(ValueError):
    """Raised by callers that prefer fail-fast manifest validation."""


class CapabilityProtocolError(RuntimeError):
    """Raised when an adapter breaks protocol after its manifest was accepted."""
