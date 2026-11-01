"""Two separate questions about one turn, and one definition for each.

Regression for 2026-10-06.  A private follow-up that only continued the
conversation was routed ``recent``; the bridge then synthesized a history
contract whose "original question" was the user's own reassurance, the model
searched memory because the contract told it to, and the receipt-owned refusal
replaced her reply.

The mistaken fix is to let routing switch the receipt gate off — that drops the
gate on exactly the turns where a memory read most easily goes unchecked.  The
gate and the verification requirement are therefore asserted separately here.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shinku.agent.history_answer import (
    NEGATIVE_RECEIPTS,
    UNCONFIRMED_FALLBACK_TEXT,
    HistoryAnswerLedger,
)
from shinku.memory.history_requirement import (
    GATING_MODES,
    VERIFICATION_MODES,
    history_requirement,
    history_requirement_for_context,
    host_requests_history_check,
)
from shinku.memory.models import MemoryContext, MemoryDecision
from shinku.tools.invocation import ToolResultEnvelope

REPO = Path(__file__).resolve().parents[1]
REASSURANCE = "没事，我带着你一起去，不落下你就够了"


def context(mode: str) -> dict:
    """The host context shape the bridge passes around."""
    return {
        "memory_context": MemoryContext(
            decision=MemoryDecision(
                need_retrieval=mode in VERIFICATION_MODES, history_mode=mode
            )
        ).as_dict(),
    }


def receipt(status: str = "candidates") -> ToolResultEnvelope:
    return ToolResultEnvelope(
        "read-1",
        "error" if status == "read_failed" else "ok",
        "fixture",
        data={
            "state_updates": {
                "memory_lookup": {
                    "status": status,
                    "query_completed": status != "read_failed",
                    "requested_scope_supported": status != "scope_limited",
                    "retrieval_scopes": ["private:1"],
                    "requested_scopes": ["private:1"],
                },
                "memory_inspection": {
                    "records": [
                        {
                            "source_id": "turn:1:a",
                            "scope": "private:1",
                            "text": "我带着你一起去。",
                            "created_at": 100,
                        }
                    ]
                },
            }
        },
    )


def observed(*, verification_requested: bool, status: str = "candidates") -> HistoryAnswerLedger:
    value = HistoryAnswerLedger(enabled=True, verification_requested=verification_requested)
    value.observe(receipt(status), SimpleNamespace(agent_history_answer_source=True))
    return value


def test_recent_mode_is_not_a_verification_requirement():
    assert "recent" not in VERIFICATION_MODES
    assert history_requirement({"history_mode": "recent"}, REASSURANCE) is None
    assert history_requirement(MemoryDecision(False, history_mode="recent"), REASSURANCE) is None


def test_recent_mode_still_requests_the_receipt_gate():
    """The two predicates must not collapse into one another."""
    assert set(VERIFICATION_MODES) < set(GATING_MODES)
    assert host_requests_history_check({"history_mode": "recent"}) is True
    assert host_requests_history_check({"history_mode": "retrieve"}) is True
    assert host_requests_history_check({"history_mode": "summary"}) is True
    assert host_requests_history_check({"history_mode": "none"}) is False
    assert host_requests_history_check(None) is False


def test_the_1006_turn_shape_gets_no_verification_requirement():
    """Private 1:1 follow-up, same sender, non-question: router says ``recent``."""
    assert history_requirement_for_context(context("recent"), REASSURANCE) is None


@pytest.mark.parametrize("mode", ["retrieve", "summary"])
def test_real_history_modes_still_require_verification(mode: str):
    """Guard against over-narrowing: these must keep working."""
    requirement = history_requirement_for_context(context(mode), REASSURANCE)
    assert requirement is not None
    assert requirement.mode == mode
    assert requirement.goal == REASSURANCE
    entry = requirement.as_completion_requirement()
    assert entry["effect"] == "history_answer"
    assert entry["description"] == REASSURANCE
    assert requirement.contract_context({"keep": 1})["agent_completion_contract"]["requirements"] == [entry]


def test_the_verification_refusal_needs_a_request():
    """The 10-06 replacement: nobody asked, so it must not be spoken."""
    text = "我读到一些相关记录，但还没找到能回答你这个问题的原消息，现在不能确认。你说的是哪一次？"
    quiet = observed(verification_requested=False)
    quiet.apply_review(text, {"disposition": "unconfirmed", "claims": [], "limitation_fragment": text}, {})
    assert quiet.answer_check["disposition"] == "unconfirmed"
    assert quiet.unconfirmed_feedback() is None

    asked = observed(verification_requested=True)
    asked.apply_review(text, {"disposition": "unconfirmed", "claims": [], "limitation_fragment": text}, {})
    assert asked.unconfirmed_feedback() is not None


@pytest.mark.parametrize(
    "status,expected",
    [("not_found", "partial"), ("scope_limited", "blocked"), ("read_failed", "failed")],
)
def test_the_receipt_gate_fires_even_without_a_request(status: str, expected: str):
    """A negative read finding is about the read, so it is always reportable."""
    value = observed(verification_requested=False, status=status)
    feedback = value.negative_feedback()
    assert feedback is not None
    assert feedback[1] == expected


def test_a_requested_ledger_still_engages_end_to_end():
    value = observed(verification_requested=True)
    assert len(value.lookups) == 1
    value.publish_receipt_reply("我读到一些相关记录，但还没找到能回答你这个问题的原消息，现在不能确认。",
                                "unconfirmed", {})
    assert value.answer_check["valid"] is True
    assert value.answer_check["disposition"] == "unconfirmed"


def test_the_agent_path_configures_both_flags_together():
    """On the agent path the planner only declares the effect for real questions."""
    assert HistoryAnswerLedger().enabled is False
    assert HistoryAnswerLedger().verification_requested is False

    blocked = HistoryAnswerLedger()
    blocked.configure({"agent_history_query_request": False})
    assert blocked.enabled is False and blocked.verification_requested is False

    asked = HistoryAnswerLedger()
    asked.configure({"agent_history_query_request": True})
    assert asked.enabled is True and asked.verification_requested is True

    declared = HistoryAnswerLedger()
    declared.configure({"agent_completion_contract": {"requirements": [
        {"effect": "history_answer", "description": "核对以前那句诗"}]}})
    assert declared.enabled is True and declared.verification_requested is True


def test_receipt_reply_refuses_when_the_query_did_not_complete():
    value = observed(verification_requested=True, status="read_failed")
    value.publish_receipt_reply("这次历史读取没有成功，还没查完，不能当作没有记录。", "read_failed", {})
    # read_failed is the one negative receipt allowed through with an incomplete query
    assert value.answer_check["valid"] is True
    assert value.answer_check["disposition"] == "read_failed"

    blocked = observed(verification_requested=True)
    blocked.lookups[-1]["query_completed"] = False
    blocked.publish_receipt_reply("我读到一些相关记录，但还没找到能回答你这个问题的原消息，现在不能确认。",
                                  "unconfirmed", {})
    assert blocked.answer_check["valid"] is False
    assert blocked.answer_check["disposition"] == "unverified"


def test_refusal_wording_lives_in_exactly_one_file():
    """The shape of the bug: the same refusal written twice, and one copy drifted."""
    strings = sorted({row[0] for row in NEGATIVE_RECEIPTS.values()} | {UNCONFIRMED_FALLBACK_TEXT})
    owners = {}
    for path in (REPO / "src" / "shinku").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found = [item for item in strings if item in text]
        if found:
            owners[path.name] = found
    assert set(owners) == {"history_answer.py"}, owners


def test_every_negative_receipt_status_has_wording():
    """A status the tool can write must not fall through to the reviewer."""
    assert set(NEGATIVE_RECEIPTS) == {"not_found", "scope_limited", "read_failed", "scope_missing"}
    for status in ("not_found", "scope_limited", "read_failed", "scope_missing"):
        value = observed(verification_requested=False, status=status)
        assert value.negative_feedback() is not None, status
