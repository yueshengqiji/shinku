"""This turn's history requirement, decided in exactly one place.

``MemoryDecision.history_mode`` is produced by the router.  Two carriers used to
re-derive its meaning downstream — a host boolean on the bridge and a
completion contract the bridge synthesized itself — and every consumer then
formatted its own refusal text.  A fix applied to one carrier therefore left the
other unchanged, and nothing failed.

This module is the single definition.  Carriers and consumers read
``history_requirement`` (or ``history_requirement_for_context``) instead of
re-inspecting the mode, so a mode added in the router cannot silently miss a
consumer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import MemoryDecision

#: Modes in which the user is asking about the past and the answer must be
#: proven from original messages.
#:
#: ``recent`` is deliberately absent.  It is the cheap continuation path: the
#: router only folds the latest records into the prompt so the model keeps the
#: thread.  It is not a request to locate and prove anything, so it must never
#: enter receipt-owned verification.
VERIFICATION_MODES: tuple[str, ...] = ("retrieve", "summary")

#: Modes in which the host wants this turn receipt-gated at all.  This is a
#: weaker question than :data:`VERIFICATION_MODES`: gating means "do not let
#: unchecked prose stand in for what memory returned", which is worth doing on a
#: continuation turn too, because the model may still choose to read memory.
#:
#: The two predicates are intentionally separate.  Collapsing them makes the
#: receipt gate depend on routing, which drops the gate on every turn the router
#: did not label — and those are exactly the turns where a memory read most
#: easily goes unchecked.
GATING_MODES: tuple[str, ...] = ("recent", "retrieve", "summary")

#: The one field every history requirement asks the reviewer to cover.
REQUIREMENT_FIELD = "原问题"

__all__ = [
    "GATING_MODES",
    "HistoryRequirement",
    "REQUIREMENT_FIELD",
    "VERIFICATION_MODES",
    "history_requirement",
    "history_requirement_for_context",
    "host_requests_history_check",
]


@dataclass(frozen=True)
class HistoryRequirement:
    """A turn that has to be answered from original messages."""

    mode: str
    goal: str

    def as_completion_requirement(self) -> dict[str, Any]:
        """The completion contract entry this requirement amounts to."""
        return {
            "id": "history_original",
            "effect": "history_answer",
            "targets": [],
            "description": self.goal,
            "checks": {"required_fields": [REQUIREMENT_FIELD], "min_sources": 1},
        }

    def contract_context(self, context: Any = None) -> dict[str, Any]:
        """``context`` with this requirement attached as the completion contract."""
        base = dict(context) if isinstance(context, Mapping) else {}
        base["agent_completion_contract"] = {
            "requirements": [self.as_completion_requirement()]
        }
        return base


def _mode_of(decision: Any) -> str:
    if isinstance(decision, MemoryDecision):
        return decision.history_mode
    if isinstance(decision, Mapping):
        return str(decision.get("history_mode") or "").strip().lower()
    return ""


def _decision_in(context: Any) -> Any:
    if not isinstance(context, Mapping):
        return None
    memory_context = context.get("memory_context")
    return memory_context.get("decision") if isinstance(memory_context, Mapping) else None


def history_requirement(decision: Any, goal: Any) -> HistoryRequirement | None:
    """The requirement for this turn, or ``None`` when there is not one.

    ``decision`` is a :class:`MemoryDecision` or its ``as_dict()`` form; ``goal``
    is the turn's original request, supplied by the caller because each entry
    point owns its own notion of "the request".
    """
    mode = _mode_of(decision)
    if mode not in VERIFICATION_MODES:
        return None
    text = str(goal or "").strip()
    if not text:
        return None
    return HistoryRequirement(mode=mode, goal=text)


def history_requirement_for_context(
    context: Any, goal: Any, *, obligation: bool = False
) -> HistoryRequirement | None:
    """Same as :func:`history_requirement`, reading the decision out of a context.

    ``obligation`` is set by the host when it has already decided, through its
    own reply-route review, that this turn re-opens an unresolved history
    question — even though the router did not label it as one.  The goal is then
    the user's own request, because that is what has to be proven.
    """
    decision = _decision_in(context)
    requirement = history_requirement(decision, goal)
    if requirement is not None or not obligation:
        return requirement
    text = str(goal or "").strip()
    if not text:
        return None
    return HistoryRequirement(mode=_mode_of(decision), goal=text)


def host_requests_history_check(decision: Any) -> bool:
    """Whether the host wants this turn's reply receipt-gated.

    Weaker than :func:`history_requirement`: a continuation turn is still gated,
    because the model may read memory on it even though nobody asked to have
    anything proven.
    """
    return _mode_of(decision) in GATING_MODES
