"""Host allocations for bounded tasks. Model output can never grant more credit."""
from __future__ import annotations

from collections.abc import Mapping
import math

LIMIT_KEYS = ("max_requests", "max_tokens", "max_tools", "max_active_seconds")
ABSOLUTE_LIMITS = dict(zip(LIMIT_KEYS, (1000, 10000000, 1024, 86400)))
PROFILES = {
    "light": dict(zip(LIMIT_KEYS, (40, 200000, 32, 600))),
    "standard": dict(zip(LIMIT_KEYS, (400, 4000000, 256, 7200))),
    "heavy": dict(zip(LIMIT_KEYS, (800, 8000000, 512, 14400))),
}
BUDGET_STOPS = frozenset({"task_budget_low", "task_model_budget_exhausted", "task_token_budget_exhausted",
                        "task_tool_budget_exhausted", "task_time_budget_exhausted"})


def limits_for(profile: str, ceilings: Mapping) -> dict:
    if profile not in PROFILES:
        raise ValueError("invalid_budget_profile")
    result = {}
    for key in LIMIT_KEYS:
        cap = ceilings.get(key, PROFILES["heavy"][key])
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("invalid_budget_ceiling")
        result[key] = min(PROFILES[profile][key], cap, ABSOLUTE_LIMITS[key])
    return result


def choose_profile(contract: Mapping | None) -> str:
    """Coarse allocation from validated work structure, never a completion claim."""
    if not isinstance(contract, Mapping):
        return "standard"
    rows = contract.get("requirements", [])
    targets = sum(len(r.get("targets", [])) for r in rows)
    fields = sum(len(r.get("checks", {}).get("required_fields", [])) for r in rows)
    pages = max((r.get("checks", {}).get("source_pages", 0) for r in rows), default=0)
    if len(rows) >= 4 or targets >= 8 or fields >= 6 or pages >= 10:
        return "heavy"
    # One read target can be an entire repository or document collection.
    # Lightweight public queries have their own fast path; admitted background
    # tasks must not receive a tiny allocation just because their contract is short.
    return "standard"


def configure_task_budget(budget, task: dict, ceilings: Mapping) -> None:
    """Called after worker claim, before persisting its first checkpoint."""
    prior = task.get("budget")
    allocation = task.get("budget_allocation")
    if allocation is None:
        # An older task keeps its recorded allocation and used counters.
        profile = "legacy" if prior else choose_profile(task.get("contract"))
        limits = ({k: prior.get(k, ceilings.get(k, PROFILES["standard"][k])) for k in LIMIT_KEYS}
                  if prior else limits_for(profile, ceilings))
        allocation = {"profile": profile, "limits": limits, "grants": []}
        task["budget_allocation"] = allocation
    profile = allocation.get("profile")
    limits = allocation.get("limits", {})
    if profile not in {*PROFILES, "legacy"} or set(limits) != set(LIMIT_KEYS):
        raise ValueError("invalid_budget_allocation")
    for key, value in limits.items():
        if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                or value > ABSOLUTE_LIMITS[key] or (key != "max_active_seconds" and type(value) is not int)):
            raise ValueError("invalid_budget_allocation")
        setattr(budget, key, value)
    budget.managed = True
    budget.profile = profile
    # Limits are host-owned and have already been checked. The existing merge
    # still validates counters and carries latched errors, never resets usage.
    if prior:
        prior = {**prior, **limits}
    budget.absorb_prior(prior)
    task["budget"] = budget.snapshot()


def grant_budget(task: dict, *, profile: str, ceilings: Mapping, request_id: str, requester: str) -> None:
    """Prepare a detached grant. Only explicit host control calls this function.

    The caller commits with the task revision after claiming the execution slot.
    Duplicate requests and queued work cannot spend or silently reset a ledger.
    """
    old = task.get("budget")
    if not isinstance(old, dict) or task.get("status") not in {"blocked", "partial", "failed"}:
        raise ValueError("task_not_budget_paused")
    reason = old.get("reason") or task.get("notes", {}).get("blocker")
    if reason not in BUDGET_STOPS:
        raise ValueError("task_not_budget_paused")
    if task.get("inflight") or task.get("uncertain_attempts"):
        raise ValueError("uncertain_execution_requires_review")
    allocation = task.get("budget_allocation") or {
        "profile": "legacy", "limits": {k: old[k] for k in LIMIT_KEYS}, "grants": []}
    grants = allocation.get("grants", [])
    if not request_id or any(g.get("request_id") == request_id for g in grants):
        raise ValueError("budget_grant_already_applied")
    if len(grants) >= 16:
        raise ValueError("budget_grant_limit")
    added = limits_for(profile, ceilings)
    limits = {k: allocation["limits"][k] + added[k] for k in LIMIT_KEYS}
    if any(v > ABSOLUTE_LIMITS[k] for k, v in limits.items()):
        raise ValueError("budget_lifetime_limit")
    task["budget_allocation"] = {"profile": allocation["profile"], "limits": limits,
        "grants": [*grants, {"request_id": request_id, "requester": requester, "profile": profile,
                              "added": added}]}
    # Clear only a budget stop after an explicit grant. Keep all counters.
    task["budget"] = {**old, **limits, "reason": "", "rejected_reservation": 0}
    task["notes"] = {**task.get("notes", {}), "blocker": ""}
