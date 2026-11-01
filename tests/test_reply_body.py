"""One definition for "which text actually gets sent".

Regression for 2026-10-06 (P0-2).  The rule "blank output or the private
silence token means there is nothing to send" was hand-written on four
separate paths -- the direct-chat branch, the autonomous silence branch, the
public-query fast path and lightweight perception.  Four copies of one rule is
how the refusal wording drifted apart before, so the rule now has a single
home and the token may not appear anywhere else in the source tree.

The token itself matters as much as the rule: the prompt tells the model to
emit it, and the runtime screens it out.  If those two ever disagree the
control value either stops working or leaks to the user.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shinku.agent.reply_body import (
    AUTONOMOUS_SILENCE_MARKER,
    finalize_reply,
    is_empty_reply,
    is_silence_marker,
)

REPO = Path(__file__).resolve().parents[1]
FALLBACK = "……刚才那句我没接好。你再说一遍？"


def test_a_normal_reply_passes_through_stripped():
    assert finalize_reply("  嗯，我在这儿。  ", FALLBACK) == "嗯，我在这儿。"


@pytest.mark.parametrize("blank", [None, "", "   ", "\n\t "])
def test_a_blank_reply_falls_back(blank):
    assert finalize_reply(blank, FALLBACK) == FALLBACK
    assert is_empty_reply(blank)


@pytest.mark.parametrize(
    "written",
    ["<SILENCE>", "  <SILENCE>  ", "<silence>", "\n<sIlEnCe>\t", "<SiLeNcE>"],
)
def test_the_silence_token_never_reaches_the_user(written):
    """However the model spells or pads it, it collapses to the fallback."""
    assert is_silence_marker(written)
    assert finalize_reply(written, FALLBACK) == FALLBACK


def test_a_reply_that_only_mentions_the_token_is_still_a_reply():
    """Equality, not containment: talking *about* the token is normal prose."""
    prose = "<SILENCE> 这个标记是什么意思？"
    assert not is_silence_marker(prose)
    assert finalize_reply(prose, FALLBACK) == prose


def test_an_explicit_fallback_is_returned_verbatim():
    assert finalize_reply("", "  spaced  ") == "  spaced  "


def test_silence_token_lives_in_exactly_one_file():
    """The shape of the bug: the prompt and the screen each spell their own token."""
    owners = {}
    for path in (REPO / "src" / "shinku").rglob("*.py"):
        if AUTONOMOUS_SILENCE_MARKER in path.read_text(encoding="utf-8"):
            owners[path.name] = path
    assert set(owners) == {"reply_body.py"}, owners
