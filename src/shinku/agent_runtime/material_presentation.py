"""Host-only recovery of bound receipt facts before result expression.

The canonical result and its hash never change. Reads use the existing task
material limits; this adapter has no execution tools, model or sender.
"""
from __future__ import annotations

import copy
import json

from .contracts import AgentBoundaryError, canonical_hash
from .information import safe_source


_FIELDS = ("label", "value", "unit", "condition", "reference")


def _agrees(visible, original):
    if isinstance(visible, dict) and set(visible) == {"excerpt", "omitted"}:
        return isinstance(original, str) and original.startswith(visible["excerpt"]) and visible["omitted"] == "text_limit"
    if isinstance(visible, dict) and visible == {"omitted": "depth_limit"}:
        return True
    if visible == "[omitted]":
        return True
    if isinstance(visible, dict):
        return isinstance(original, dict) and all(k in original and _agrees(v, original[k]) for k, v in visible.items())
    if isinstance(visible, list):
        return isinstance(original, list) and len(visible) <= len(original) and all(_agrees(v, original[i]) for i, v in enumerate(visible))
    return type(visible) is type(original) and visible == original


def recover_presentation(store, result, *, scope, still_authorized, delivery=False):
    task = store.store.get(result["task_id"], requester=scope.requester, session=scope.session)
    if (not task or task.get("execution_attempt", 0) != result["attempt"] or task["revision"] != result["version"]
            or task.get("execution_result", {}).get("result_hash") != result["result_hash"]):
        raise AgentBoundaryError("material_result_binding_conflict")
    store.current(task, still_authorized)
    information = result.get("information", {})
    losses = information.get("projection", {}).get("gap_count", 0) or information.get("snapshot", {}).get("gap_count", 0)
    projection = {"schema_version": 1, "source_result_hash": result["result_hash"], "status": "complete",
                  "sources": [], "gaps": []}
    for row in information.get("sources", []):
        if row.get("required") is True:
            try:
                record = store._record(row["reference"], task)
                if record["content_sha256"] != row["content_sha256"] or record["version"] != row["version"]:
                    raise AgentBoundaryError("material_version_conflict")
                store.resolve(row["reference"], task=task, still_authorized=still_authorized)
            except AgentBoundaryError as exc:
                projection["gaps"].append({"reference": row["reference"], "reason": exc.code})
    if not losses:
        projection["status"] = "incomplete" if projection["gaps"] else "complete"
        return projection
    bound = {r["reference"]: r for r in task.get("material_package", {}).get("items", [])}
    recovered = {}
    wanted = None
    diagnostic = set()
    if delivery:
        # 默认交付只补正文事实，不为文件列表、代码原文、进程快照重读全部执行回执。
        # 原资料、绑定与读取账本仍在库内；必需输入仍由上方逐项重新校验。
        from .handoff import diagnostic_invocations
        diagnostic = diagnostic_invocations(result)
        wanted = {r.get('invocation_id') for r in result.get('facts', [])} - diagnostic
        for call in task.get('tool_calls', []):
            if call.get('data', {}).get('state_updates', {}).get('result_facts'):
                wanted.add(call.get('id'))
        wanted -= diagnostic | {None, ''}
    for source in information.get("sources", []):
        if source.get("kind") != "executor_result":
            continue
        reference = source.get("reference", "")
        if wanted is not None and source.get('provenance', {}).get('invocation_id') not in wanted:
            continue
        try:
            item = bound.get(reference)
            if (not item or any(item.get(k) != source.get(k) for k in ("content_sha256", "version", "kind", "provenance"))):
                raise AgentBoundaryError("material_version_conflict")
            invocation = source.get("provenance", {}).get("invocation_id", "")
            if not invocation:
                raise AgentBoundaryError("material_receipt_binding_missing")
            # Resolve each page under current authorization. A retained full
            # JSON value is not a receipt that the bounded reader supplied it.
            offset, parts, reads = 0, [], []
            while True:
                receipt = store.read(reference, task=task, still_authorized=still_authorized,
                    offset=offset, length=min(store.policy.page_chars, store.limits(task)["page_chars"]))
                parts.append(receipt["content"])
                reads.append({k: receipt[k] for k in ("read_id", "offset", "returned_chars", "page_sha256")})
                if receipt["next_offset"] is None:
                    break
                offset = receipt["next_offset"]
            value = json.loads("".join(parts))
            projection["sources"].append({"reference": reference, "content_sha256": source["content_sha256"],
                                          "version": source["version"], "reads": reads})
            if value.get("execution_status") == "ok":
                for index, row in enumerate(value.get("state_updates", {}).get("result_facts", [])):
                    if isinstance(row, dict):
                        recovered[(invocation, index)] = {**{k: copy.deepcopy(row[k]) for k in _FIELDS if k in row},
                            "source": "executor_receipt", "invocation_id": invocation, "source_index": index}
            if source.get("retained_complete") is not True:
                raise AgentBoundaryError("material_upstream_incomplete")
        except (AgentBoundaryError, ValueError, TypeError, KeyError) as exc:
            projection["gaps"].append({"reference": reference,
                "reason": exc.code if isinstance(exc, AgentBoundaryError) else "material_receipt_invalid"})
    facts = []
    for row in result.get("facts", []):
        if row.get('invocation_id') in diagnostic:
            continue
        key = (row.get("invocation_id", ""), row.get("source_index", -1))
        restored = recovered.pop(key, None)
        if restored is not None and not _agrees(row, restored):
            projection["gaps"].append({"reason": "material_fact_projection_conflict", "invocation_id": key[0]})
            restored = None
        facts.append(restored or copy.deepcopy(row))
    facts.extend(recovered.values())
    projection["facts"] = safe_source(facts)
    if losses and not projection["sources"] and (wanted is None or wanted):
        projection["gaps"].append({"reason": "material_full_receipts_unavailable"})
    # Non-receipt losses remain explicitly present in the canonical blocks;
    # restored receipt facts never upgrade input/goal or task acceptance.
    projection["status"] = "incomplete" if projection["gaps"] else "complete"
    current = store.current(task, still_authorized)
    if current["revision"] != result["version"] or current.get("execution_result", {}).get("result_hash") != result["result_hash"]:
        raise AgentBoundaryError("material_result_binding_conflict")
    projection["projection_hash"] = canonical_hash(projection)
    return projection


def validate_presentation(store, result, projection, *, scope, still_authorized):
    """No new pages/call budget: revalidate already archived source identity."""
    task = store.store.get(result["task_id"], requester=scope.requester, session=scope.session)
    if (not task or task["revision"] != result["version"] or task.get("execution_attempt", 0) != result["attempt"]
            or task.get("execution_result", {}).get("result_hash") != result["result_hash"]):
        return False
    rows = [*projection.get("sources", [])]
    if projection.get("status") == "complete":
        rows.extend(r for r in result.get("information", {}).get("sources", []) if r.get("required") is True)
    for row in rows:
        record = store._record(row["reference"], task)
        if record["content_sha256"] != row["content_sha256"] or record["version"] != row["version"]:
            return False
        store.resolve(row["reference"], task=task, still_authorized=still_authorized)
    store.current(task, still_authorized)
    return True
