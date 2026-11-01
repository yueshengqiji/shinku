"""Completion verifier facade; contract checks, effect receipts and target parsing are separate responsibilities: verification_contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from shinku.paths import absolute_file_path

from .verification_rules import _UNCHANGED_INPUTS


@dataclass(frozen=True)
class VerificationContractsDependencies:
    """Bound callbacks and live getters for this responsibility."""
    explicit_target_roles: Callable[..., Any]
    extract_user_messages: Callable[..., Any]
    has_effect: Callable[..., Any]
    normalize_target_values: Callable[..., Any]
    receipt: Callable[..., Any]
    selector_matches: Callable[..., Any]
    targets_match: Callable[..., Any]
    verify_inplace_read: Callable[..., Any]
    verify_read_unchanged: Callable[..., Any]
    verify_requirement: Callable[..., Any]
    result_handlers: Callable[..., Any]
    tool_capabilities: Callable[..., Any]


def verify_read_unchanged(state, target, *, deps: VerificationContractsDependencies):
    observations = []
    for call in state.tool_calls:
        capability = deps.tool_capabilities().get(call.get("name"), {})
        if "read" not in capability.get("effects", ()) or not deps.has_effect([call], "read", [target]):
            continue
        receipt = deps.receipt(call)
        refresh = getattr(deps.result_handlers().get(call.get("name")), "agent_verify_result", None)
        if not receipt.get("sha256") or not callable(refresh):
            return False
        try:
            if refresh(receipt=receipt, context=state.context).get("result_valid") is not True:
                return False
        except Exception:
            return False
        observations.append(receipt["sha256"])
    if not observations or len(set(observations)) != 1:
        return False
    # Known task-local source mutations cannot be erased by restoring bytes
    # or taking a new read receipt. This does not attest arbitrary external
    # filesystem history between two observations.
    for call in state.tool_calls:
        receipt = deps.receipt(call)
        if call.get("status") != "ok" or not {"write", "delete"}.intersection(receipt.get("verified_effects", ())):
            continue
        if any(deps.targets_match(target, n) for value in receipt.get("completion_targets", ())
               for n in deps.normalize_target_values(value)):
            return False
    return True


def verify_inplace_read(
    state,
    candidate,
    target,
    calls,
    *,
    deps: VerificationContractsDependencies,
):
    """A requested in-place write may consume an observed pre-write input.

        Require the actual writer's pre-image hash AND its declared, currently
        valid output condition. Merely re-reading a changed input is not proof.
        """
    messages = deps.extract_user_messages(state)
    if _UNCHANGED_INPUTS.search("\n".join(messages)):
        return False
    roles = deps.explicit_target_roles(messages)
    if "write" not in roles.get(target, ()):
        return False
    requirements = (state.context or {}).get("agent_completion_contract", {}).get("requirements", ())
    output_conditions = [row for row in requirements if row.get("effect") == "write" and any(
        deps.targets_match(target, n) for value in row.get("targets", ()) for n in deps.normalize_target_values(value))]
    if not output_conditions or not any(deps.verify_requirement(state, row, calls) for row in output_conditions):
        return False
    seen = False
    for call in state.tool_calls:
        if call is candidate:
            seen = True
            continue
        if not seen or not deps.has_effect([call], "write", [target]):
            continue
        if call.get("status") == "ok" and deps.receipt(call).get("previous_sha256") == deps.receipt(candidate).get("sha256"):
            return True
        return False
    return False


def verify_requirement(state, requirement, calls, *, deps: VerificationContractsDependencies):
    """Coordinate selector, answer and resource requirements without changing check order."""
    effect = str(requirement["effect"])
    selector = requirement.get("target_selector")
    if selector is not None:
        return _verify_selected_target(state, requirement, calls, effect, selector, deps=deps)
    if effect == "research_answer":
        checks = requirement.get("checks", {})
        return (isinstance(checks, Mapping) and not set(checks) - {"required_fields", "min_sources", "require_dates", "date_kinds"}
                and state.research.satisfies(checks))
    if effect == "local_answer":
        return not requirement.get("targets") and state.local_answer.satisfies(requirement.get("checks", {}))
    if effect == "history_answer":
        return not requirement.get("targets") and state.history_answer.satisfies(requirement.get("checks", {}))
    if effect == "command_run":
        checks = requirement.get("checks", {})
        if (requirement.get("targets") or set(checks) != {"command", "exit_code"}
                or not isinstance(checks.get("command"), str) or checks.get("exit_code") != 0):
            return False
        digest = hashlib.sha256(checks["command"].strip().encode("utf-8")).hexdigest()
        for call in reversed(state.tool_calls):
            if "command_run" not in deps.tool_capabilities().get(call.get("name"), {}).get("effects", ()):
                continue
            receipt = deps.receipt(call)
            session = receipt.get("execution_session", {})
            if session.get("command_sha256") == digest:
                return bool(call.get("status") == "ok" and session.get("status") == "completed"
                    and session.get("returncode") == 0 and "command_run" in receipt.get("verified_effects", ()))
        return False
    targets = [deps.normalize_target_values(t)[0] for t in requirement.get("targets", ())]
    if effect == "read" and not targets:
        return False
    if effect.startswith("tool:"):
        targets = _observation_targets(state, effect, targets, deps=deps)
        if targets is None:
            return False
    checks = _requirement_checks(requirement, effect)
    if checks is None:
        return False
    for target in targets or [None]:
        if not _verify_target_receipt(state, effect, target, calls, checks, deps=deps):
            return False
    return True


def _verify_selected_target(state, requirement, calls, effect, selector, *, deps):
    """Bind a singular selector only to effect-specific executor receipts."""
    if effect not in {"delete", "write", "send", "document_translation"}:
        return False
    # An unresolved name is a predicate, not a placeholder path. Bind it
    # ONLY to executor-issued, effect-specific absolute file receipts.
    # Arguments/model notes never establish this binding. More than one
    # matching path is ambiguous and cannot complete a singular target.
    from .task_state import TaskContractError, validate_contract
    try:
        validate_contract({"goal": "verify", "deliverables": ["verify"], "requirements": [
            {"id": "verify", "description": "verify", **requirement}]})
    except TaskContractError:
        return False
    paths = set()
    for call in calls:
        receipt = deps.receipt(call)
        if effect not in deps.tool_capabilities().get(call.get("name"), {}).get("effects", ()):
            continue
        if effect not in receipt.get("verified_effects", ()):
            continue
        actual = receipt.get("completion_targets", ())
        if not isinstance(actual, (list, tuple)):
            continue
        for value in actual:
            if (isinstance(value, str) and "://" not in value and absolute_file_path(value)
                    and deps.selector_matches(selector, value)):
                paths.add(deps.normalize_target_values(value)[0])
    if len(paths) != 1:
        return False
    resolved = {k: v for k, v in requirement.items() if k != "target_selector"}
    resolved["targets"] = list(paths)
    return deps.verify_requirement(state, resolved, calls)


def _observation_targets(state, effect, targets, *, deps):
    """Admit the latest observation and keep only resource identifiers as targets."""
    name = effect[5:]
    capability = deps.tool_capabilities().get(name, {})
    if effect not in capability.get("effects", ()):
        return None
    # Free-form observation descriptions are not resource identifiers.
    if not capability.get("target_arguments"):
        targets = []
    matching = [c for c in state.tool_calls if c.get("name") == name and c.get("executed") is not False]
    if not matching or (not targets and matching[-1].get("status") != "ok"):
        return None
    receipt = deps.receipt(matching[-1])
    required_fields = getattr(deps.result_handlers().get(name), "agent_observation_required_fields", ())
    if any(not receipt.get(field) for field in required_fields):
        return None
    return targets


def _requirement_checks(requirement, effect):
    """Validate supported condition keys and mandatory content constraints."""
    checks = requirement.get("checks", {})
    if not isinstance(checks, Mapping):
        return None
    # A narrowly recognizable mandatory content condition must not be
    # treated as proof merely because the model omitted result checks.
    mandatory = re.search(r"(?:文件)?内容必须为\s*([^。\n]+)", str(requirement.get("description", "")))
    if mandatory and checks.get("contains") != mandatory.group(1).strip():
        return None
    allowed = {"contains", "sha256", "min_bytes", "http_status", "page_contains", "receipt_required", "source_pages", "unchanged"}
    if set(checks) - allowed:
        return None
    if "unchanged" in checks and (effect != "read" or type(checks["unchanged"]) is not bool):
        return None
    return checks


def _verify_target_receipt(state, effect, target, calls, checks, *, deps):
    """Select the latest applicable result and refresh its current validity."""
    if effect == "read" and checks.get("unchanged") is True and not deps.verify_read_unchanged(state, target):
        return False
    candidates = [c for c in calls if deps.has_effect([c], effect, [target] if target else [])]
    if not candidates:
        return False
    candidate = candidates[-1]
    # A newer executor receipt for the same target supersedes historical
    # success, including deletion, overwrite and failed site access.
    if target:
        for c in reversed(state.tool_calls):
            # Host-rejected dispatches did not touch the resource. They cannot
            # replace a real receipt; executed and unknown failures remain
            # conservative, and the selected receipt is still refreshed.
            if c.get("executed") is False:
                continue
            related = deps.tool_capabilities().get(c.get("name"), {}).get("effects", ())
            relevant = {"write", "delete"} if effect in {"write", "delete"} else {effect}
            if not relevant.intersection(related):
                continue
            actual = list(deps.receipt(c).get("completion_targets", []))
            arguments = c.get("arguments", {})
            if isinstance(arguments, Mapping):
                actual.extend(arguments[key] for key in deps.tool_capabilities().get(c.get("name"), {}).get("target_arguments", ()) if key in arguments)
            if any(deps.targets_match(target, n) for a in actual for n in deps.normalize_target_values(a)):
                if not deps.has_effect([c], effect, [target]) or c.get("status") != "ok":
                    return False
                candidate = c
                break
    receipt = dict(deps.receipt(candidate))
    if effect == "browser_access" and (receipt.get("site_access_ok") is False or
            isinstance(receipt.get("http_status"), int) and receipt["http_status"] >= 400):
        return False
    handler = deps.result_handlers().get(candidate.get("name"))
    refresh = getattr(handler, "agent_verify_result", None)
    resource_read = "read" in deps.tool_capabilities().get(candidate.get("name"), {}).get("effects", ())
    if (effect == "read" or effect.startswith("tool:") and resource_read) and (
            receipt.get("read_complete") is not True or not callable(refresh) or not receipt.get("sha256")):
        return False
    if callable(refresh):
        try:
            refreshed = refresh(receipt=receipt, context=state.context)
        except Exception:
            return False
        if (effect == "read" and checks.get("unchanged") is not True
                and refreshed.get("result_valid") is False
                and deps.verify_inplace_read(state, candidate, target, calls)):
            # Keep the historical input's bytes/hash, not the output's
            # refreshed contents. Unsupported input text checks fail.
            receipt["result_valid"] = True
        else:
            receipt.update(refreshed)
    if "artifact_receipts" in receipt:
        # A batch executor publishes one independent host observation per file.
        artifacts = [a for a in receipt["artifact_receipts"] if isinstance(a, Mapping)
            and effect in a.get("verified_effects", ())
            and any(deps.targets_match(target, n) for t in a.get("completion_targets", ())
                    for n in deps.normalize_target_values(t))]
        if len(artifacts) != 1 or not callable(refresh):
            return False
        receipt = dict(artifacts[0])
    if receipt.get("result_valid") is False:
        return False
    if (effect == "read" or effect.startswith("tool:") and resource_read) and receipt.get("result_valid") is not True:
        return False
    return _receipt_meets_checks(effect, receipt, checks, refresh)


def _receipt_meets_checks(effect, receipt, checks, refresh):
    """Apply translation coverage, content, digest, size and delivery conditions."""
    if effect == "document_translation":
        if (receipt.get("translation_verified") is not True or receipt.get("result_valid") is not True
                or type(receipt.get("done_chunks")) is not int or receipt["done_chunks"] < 1
                or receipt["done_chunks"] != receipt.get("chunk_count")
                or not str(receipt.get("path", "")).lower().endswith(".pdf")
                or not callable(refresh)):
            return False
    if "source_pages" in checks and (effect != "document_translation" or type(checks["source_pages"]) is not int
            or receipt.get("source_pages") != checks["source_pages"]):
        return False
    if "contains" in checks and checks["contains"] not in str(receipt.get("file_text", "")):
        return False
    if "sha256" in checks and str(checks["sha256"]).lower() != receipt.get("sha256"):
        return False
    if "min_bytes" in checks and (type(checks["min_bytes"]) is not int or receipt.get("bytes", -1) < checks["min_bytes"]):
        return False
    if "http_status" in checks and receipt.get("http_status") != checks["http_status"]:
        return False
    if "page_contains" in checks and checks["page_contains"] not in str(receipt.get("page_text", "")):
        return False
    if checks.get("receipt_required") and not receipt.get("delivery_receipt"):
        return False
    return True
