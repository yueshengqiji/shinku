"""Turn the YAML documents a provider ships into validated records.

A provider describes itself in a file on disk: who it is, where it lives, how it
can be reached, and what it can be asked to do.  This module decides whether such
a file can be trusted.  Because those files arrive from outside the process, the
answer is a value rather than an exception - callers get either a finished record
or an explanation of the refusal, so one bad file cannot take the rest of the
registry down with it.

Validation is organised as a short pipeline.  Each stage reads one part of the
document and either produces a value or signals a refusal through a private
exception.  :func:`load_manifest` is the only place that turns a refusal into
something the caller can inspect, which is what makes the "first problem wins"
ordering easy to see.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, NoReturn, cast
from urllib.parse import urlparse

import yaml

from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityIOSlot,
    CapabilityManifest,
    ConfirmPolicy,
    EndpointConfig,
    HealthConfig,
    InvalidManifest,
    RiskLevel,
    SourceLayer,
    TierConfig,
    TriggerConfig,
)
from .safety import LOOPBACK_HOSTS, MCP_SAFE_TYPE_RE, MCP_SECRET_MARKERS

__all__ = [
    "SCHEMA",
    "ALLOWED_ADAPTER_TYPES",
    "VISIBLE_IN_VALUES",
    "RISK_VALUES",
    "CONFIRM_VALUES",
    "EFFECT_VALUES",
    "PROVIDER_ID_RE",
    "SECRET_VALUE_RE",
    "load_manifest",
    "CapabilityDescriptor",
    "CapabilityIOSlot",
    "CapabilityManifest",
    "ConfirmPolicy",
    "EndpointConfig",
    "HealthConfig",
    "InvalidManifest",
    "RiskLevel",
    "SourceLayer",
    "TierConfig",
    "TriggerConfig",
]


SCHEMA = "capability_adapter/v1"

ALLOWED_ADAPTER_TYPES = frozenset(
    {"mcp_stdio", "comfyui", "openai_compat_tts", "openai_compat_asr", "python_plugin"}
)

VISIBLE_IN_VALUES = frozenset({"base", "web", "desktop", "qq"})
RISK_VALUES = frozenset({"low", "medium", "high"})
CONFIRM_VALUES = frozenset({"never", "first_time", "always"})
EFFECT_VALUES = frozenset(
    {
        "file_read",
        "file_write",
        "command_exec",
        "network_outbound",
        "browser_action",
        "media_generation",
        "state_mutation",
    }
)

PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SECRET_VALUE_RE = re.compile(
    r"(?i)(\bbearer\s+\S+|\bsk-[A-Za-z0-9]|\bghp_[A-Za-z0-9]|\bxox[baprs]-[A-Za-z0-9]|=|:)"
)

# Effects that leave no room for negotiation once declared.
_FORCING_EFFECTS = frozenset({"command_exec", "browser_action"})
# Effects that lift a "low" risk to the middle tier.
_PROMOTING_EFFECTS = frozenset({"file_write", "network_outbound"})


class _ManifestRejected(Exception):
    """Internal short-circuit used while reading a manifest.

    The public surface never lets this escape; :func:`load_manifest` catches it
    and converts it into an :class:`InvalidManifest`.
    """

    def __init__(self, reason: str, detail: str = "", provider_id: str = "") -> None:
        super().__init__(reason, detail, provider_id)
        self.reason = reason
        self.detail = detail
        self.provider_id = provider_id


def _refuse(reason: str, detail: str = "", provider_id: str = "") -> NoReturn:
    raise _ManifestRejected(reason, detail, provider_id)


def load_manifest(path: Path, *, source_layer: SourceLayer) -> CapabilityManifest | InvalidManifest:
    """Read a provider's YAML file and report what it declares.

    Returns a validated record, or an :class:`InvalidManifest` carrying the reason
    the file was rejected.  Reading and YAML syntax problems are handled here;
    everything after parsing is delegated to the stages below.
    """

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        return InvalidManifest(
            source_path=path, source_layer=source_layer, reason="yaml_parse_error", detail=str(exc)
        )
    except OSError as exc:
        return InvalidManifest(
            source_path=path, source_layer=source_layer, reason="read_error", detail=str(exc)
        )

    try:
        return _validated(document, path, source_layer)
    except _ManifestRejected as rejection:
        return InvalidManifest(
            source_path=path,
            source_layer=source_layer,
            reason=rejection.reason,
            detail=rejection.detail,
            provider_id=rejection.provider_id,
        )


def _validated(document: Any, path: Path, source_layer: SourceLayer) -> CapabilityManifest:
    """Run the document through every stage, in the order the contract fixes."""

    if not isinstance(document, Mapping):
        _refuse("manifest_must_be_mapping", "top-level yaml must be a mapping")
    raw: Mapping[str, Any] = document

    if _trim(raw.get("schema")) != SCHEMA:
        _refuse("schema_mismatch", f"expected {SCHEMA}")

    provider = raw.get("provider")
    if not isinstance(provider, Mapping):
        _refuse("missing_provider", "provider must be a mapping")

    provider_id = _trim(provider.get("id"))
    if not provider_id:
        _refuse("missing_provider_id", "provider.id is required")
    if not PROVIDER_ID_RE.fullmatch(provider_id):
        _refuse("invalid_provider_id", "provider.id contains unsupported characters", provider_id)

    provider_type = _trim(provider.get("type"))
    if not provider_type:
        _refuse("missing_provider_type", "provider.type is required", provider_id)
    if not MCP_SAFE_TYPE_RE.fullmatch(provider_type):
        _refuse("provider_type_invalid", "provider.type contains unsupported characters", provider_id)
    if provider_type not in ALLOWED_ADAPTER_TYPES:
        _refuse("provider_type_not_allowed", provider_type, provider_id)

    endpoint = _endpoint_of(provider.get("endpoint"), provider_id)
    secrets = _secrets_of(provider.get("secrets"), provider_id)
    tiers = _tiers_of(provider.get("tiers"), provider_id)
    capabilities = _capabilities_of(raw.get("capabilities"), provider_id)

    return CapabilityManifest(
        schema=SCHEMA,
        provider_id=provider_id,
        provider_type=provider_type,
        display_name=_trim(provider.get("display_name")) or provider_id,
        endpoint=endpoint,
        health=_health_of(provider.get("health")),
        tiers=tiers,
        capabilities=capabilities,
        secrets=secrets,
        source_path=path,
        source_layer=source_layer,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# per-section readers
# --------------------------------------------------------------------------- #


def _endpoint_of(value: Any, provider_id: str) -> EndpointConfig | None:
    if _absent(value):
        return None
    if not isinstance(value, Mapping):
        _refuse("endpoint_must_be_mapping", "provider.endpoint must be a mapping", provider_id)

    url = _trim(value.get("url"))
    loopback_only = bool(value.get("loopback_only"))
    if loopback_only:
        host = (urlparse(url).hostname or "").strip().lower()
        if host not in LOOPBACK_HOSTS:
            _refuse("endpoint_not_loopback", f"host={host or '<empty>'}", provider_id)
    return EndpointConfig(url=url, loopback_only=loopback_only, raw=value)


def _health_of(value: Any) -> HealthConfig | None:
    """Health settings are advisory, so a malformed block is simply dropped."""

    if not isinstance(value, Mapping):
        return None

    statuses: list[int] = []
    for candidate in value.get("expect_status") or []:
        try:
            statuses.append(int(candidate))
        except (TypeError, ValueError):
            continue

    try:
        timeout_seconds = float(value.get("timeout_seconds") or 3)
    except (TypeError, ValueError):
        timeout_seconds = 3.0

    return HealthConfig(
        method=_trim(value.get("method")).upper() or "GET",
        path=_trim(value.get("path")),
        timeout_seconds=timeout_seconds,
        expect_status=tuple(statuses or [200]),
        raw=value,
    )


def _tiers_of(value: Any, provider_id: str) -> tuple[TierConfig, ...]:
    if _absent(value):
        return ()
    if not isinstance(value, list):
        _refuse("tiers_must_be_list", "provider.tiers must be a list", provider_id)

    tiers: list[TierConfig] = []
    claimed: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            _refuse("tier_must_be_mapping", "tier entry must be a mapping", provider_id)
        tier_id = _trim(item.get("id"))
        if not tier_id or tier_id in claimed:
            _refuse("tier_id_not_unique", tier_id or "<empty>", provider_id)
        claimed.add(tier_id)
        preset = item.get("preset") if isinstance(item.get("preset"), Mapping) else {}
        tiers.append(TierConfig(tier_id, _trim(item.get("label")) or tier_id, preset, item))
    return tuple(tiers)


def _secrets_of(value: Any, provider_id: str) -> tuple[str, ...]:
    """Collect secret *key names*, refusing anything that looks like material."""

    if _absent(value):
        return ()
    if not isinstance(value, list):
        _refuse(
            "secrets_must_be_key_names",
            "provider.secrets must be a list of key names",
            provider_id,
        )

    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            _refuse("secrets_must_be_key_names", "secret entries must be strings", provider_id)
        name = item.strip()
        if _carries_secret_material(name):
            _refuse("secrets_must_be_key_names", "secret entry looks like a value", provider_id)
        if name.lower() in MCP_SECRET_MARKERS:
            _refuse("secrets_must_be_key_names", "secret entry is too generic", provider_id)
        names.append(name)
    return tuple(names)


def _carries_secret_material(name: str) -> bool:
    """A key name is a bare identifier - no spaces, no length, no punctuation."""

    if not name or len(name) > 120:
        return True
    if re.search(r"\s", name):
        return True
    return SECRET_VALUE_RE.search(name) is not None


def _capabilities_of(value: Any, provider_id: str) -> tuple[CapabilityDescriptor, ...]:
    if _absent(value):
        return ()
    if not isinstance(value, list):
        _refuse("capabilities_must_be_list", "capabilities must be a list", provider_id)

    descriptors: list[CapabilityDescriptor] = []
    for item in value:
        if not isinstance(item, Mapping):
            _refuse("capability_must_be_mapping", "capability entry must be a mapping", provider_id)

        capability_id = _trim(item.get("id"))
        if not capability_id:
            _refuse("missing_capability_id", "capability.id is required", provider_id)

        visible_in = _as_name_tuple(item.get("visible_in"))
        if any(surface not in VISIBLE_IN_VALUES for surface in visible_in):
            _refuse("visible_in_invalid", ",".join(visible_in), provider_id)

        risk, confirm = _declared_policy(item)

        effects = _as_name_tuple(item.get("effects"))
        unrecognised = [effect for effect in effects if effect not in EFFECT_VALUES]
        if unrecognised:
            _refuse("effects_invalid", ",".join(unrecognised), provider_id)

        risk, confirm = _policy_after_effects(risk, confirm, effects)

        descriptors.append(
            CapabilityDescriptor(
                id=capability_id,
                display_name=_trim(item.get("display_name")) or capability_id,
                short_hint=_trim(item.get("short_hint")),
                visible_in=visible_in,
                prompt_exposed=bool(item.get("prompt_exposed", False)),
                risk=risk,
                confirm=confirm,
                effects=effects,
                trigger=_trigger_of(item.get("trigger")),
                inputs=_io_slots_of(item.get("inputs")),
                outputs=_io_slots_of(item.get("outputs")),
                raw=item,
            )
        )
    return tuple(descriptors)


def _trigger_of(value: Any) -> TriggerConfig | None:
    if not isinstance(value, Mapping):
        return None
    kind = _trim(value.get("kind"))
    return TriggerConfig(kind=kind, raw=value) if kind else None


def _io_slots_of(value: Any) -> tuple[CapabilityIOSlot, ...]:
    """Slots without both a name and a kind are dropped, not reported."""

    if not isinstance(value, list):
        return ()

    slots: list[CapabilityIOSlot] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _trim(item.get("name"))
        kind = _trim(item.get("kind"))
        if not name or not kind:
            continue
        slots.append(
            CapabilityIOSlot(
                name=name,
                kind=kind,
                required=bool(item.get("required")),
                max_bytes=_byte_ceiling(item.get("max_bytes")),
                delivery=_trim(item.get("delivery")),
                raw=item,
            )
        )
    return tuple(slots)


def _byte_ceiling(value: Any) -> int | None:
    if _absent(value):
        return None
    try:
        return int(str(value).replace("_", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #


def _declared_policy(raw: Mapping[str, Any]) -> tuple[RiskLevel, ConfirmPolicy]:
    return _coerced_risk(raw.get("risk")), _coerced_confirm(raw.get("confirm"))


def _coerced_risk(value: Any) -> RiskLevel:
    candidate = _trim(value).lower()
    return cast(RiskLevel, candidate) if candidate in RISK_VALUES else "medium"


def _coerced_confirm(value: Any) -> ConfirmPolicy:
    candidate = _trim(value).lower()
    return cast(ConfirmPolicy, candidate) if candidate in CONFIRM_VALUES else "first_time"


def _policy_after_effects(
    risk: RiskLevel, confirm: ConfirmPolicy, effects: tuple[str, ...]
) -> tuple[RiskLevel, ConfirmPolicy]:
    """Declared effects can only ever tighten the approval policy."""

    declared = set(effects)
    if declared & _FORCING_EFFECTS:
        return "high", "always"
    if declared & _PROMOTING_EFFECTS and risk == "low":
        promoted = "first_time" if confirm == "never" else confirm
        return "medium", promoted
    if risk == "high":
        return "high", "always"
    return risk, confirm


# --------------------------------------------------------------------------- #
# small readers
# --------------------------------------------------------------------------- #


def _absent(value: Any) -> bool:
    """True for the two spellings YAML uses for "section not written"."""

    return value is None or value == ""


def _trim(value: Any) -> str:
    return str(value or "").strip()


def _as_name_tuple(value: Any) -> tuple[str, ...]:
    """Read a YAML list of names, dropping blanks; non-lists yield nothing."""

    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        text = _trim(item)
        if text:
            names.append(text)
    return tuple(names)
