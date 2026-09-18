"""Capability providers: the manifests that describe them and the safety bounds
their configuration has to respect.

The package is the boundary between Shinku and anything outside it that offers
a capability - a local tool server, a workflow engine, a plugin.  Providers
describe themselves in files; this package turns those files into records the
runtime can trust, and keeps the constants that decide what "trustworthy"
means in one place.
"""

from __future__ import annotations

from .manifest import (
    ALLOWED_ADAPTER_TYPES,
    CONFIRM_VALUES,
    EFFECT_VALUES,
    PROVIDER_ID_RE,
    RISK_VALUES,
    SCHEMA,
    SECRET_VALUE_RE,
    VISIBLE_IN_VALUES,
    load_manifest,
)
from .safety import LOOPBACK_HOSTS, MCP_SAFE_TYPE_RE, MCP_SECRET_MARKERS

__all__ = [
    "ALLOWED_ADAPTER_TYPES",
    "CONFIRM_VALUES",
    "EFFECT_VALUES",
    "LOOPBACK_HOSTS",
    "MCP_SAFE_TYPE_RE",
    "MCP_SECRET_MARKERS",
    "PROVIDER_ID_RE",
    "RISK_VALUES",
    "SCHEMA",
    "SECRET_VALUE_RE",
    "VISIBLE_IN_VALUES",
    "load_manifest",
]
