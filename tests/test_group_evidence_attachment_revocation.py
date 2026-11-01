"""A review cannot bless media whose content or provenance changed while waiting."""
import pytest

from test_group_evidence_grounding import final, group, verdict
from test_memory_read_acceptance import bridge_factory


@pytest.mark.parametrize("changed", [
    {"data_url": "data:image/png;base64,d29ybGQ="},
    {"source_message_id": "replacement-message"},
    {"source_sender_id": "another-speaker"},
])
def test_attachment_change_during_review_invalidates_the_answer(group, changed):
    p = group
    original = {"data_url": "data:image/png;base64,aGVsbG8=",
        "source_message_id": "current", "source_sender_id": "owner"}
    images = [original]
    # Preparation retains this list in the actual turn context. Replace that
    # in-flight attachment, rather than only changing a future builder result.
    p.bridge._images = lambda _turn: images
    text = "画面有一条横线。"

    def reviewed_then_replaced(options):
        result = final(text)
        images[:] = [{**original, **changed}]
        return result

    p.queue.append(reviewed_then_replaced)
    reply = p.send()

    assert reply.agent_status == "failed" and reply.reason == "dialogue_sources_changed"
    assert len(p.calls) == 1 and not p.queue
    assert p.calls[0]["user_images"] == [original]
    assert text not in reply.final_text
    assert all(text not in segment.text for output in p.sent for segment in output.segments)
    assert p.memory.store.event_by_source_id(source_id="turn:g:current:assistant", scope="group:g") is None
