"""Records and vocabulary for work Shinku delegates to a specialist.

A delegation is described twice: once as the request handed to a specialist,
and once as the report that comes back when the specialist stops.  The request
is frozen - it is a fact about what was asked, so nothing downstream may edit
it.  The report is mutable, because the runtime fills it in as rounds complete.

The two vocabularies underneath are protocol constants.  ``AGENT_ALLOWED_TOOLS``
is the allow-list every specialist is held to, and ``AGENT_ALIASES`` lets a
caller use a short, legacy-friendly name instead of the canonical one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AGENT_ALIASES",
    "AGENT_ALLOWED_TOOLS",
    "WorkerDelegation",
    "WorkerRunSummary",
]


@dataclass(frozen=True)
class WorkerDelegation:
    """A request to hand one task to a named specialist.

    ``handle_id`` and ``started`` are filled in by the dispatcher once the
    specialist has actually been reached, so a freshly built delegation carries
    them empty.
    """

    task_id: str
    assigned_agent: str
    brief: str
    handle_id: str = ""
    started: bool = False


@dataclass
class WorkerRunSummary:
    """What a specialist reports once it has stopped working a delegation.

    ``messages`` and ``tool_results`` start empty and accumulate as the run
    progresses; each summary owns its own lists.
    """

    task_id: str
    assigned_agent: str
    status: str
    rounds: int = 0
    messages: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


AGENT_ALLOWED_TOOLS: dict[str, set[str]] = {
    "document_agent": {
        "sync_attachment_workspace",
        "inspect_attachment",
        "read_attachment_section",
        "compose_file",
        "revise_generated_file",
        "apply_style_to_existing_file",
        "inspect_generated_file",
    },
    "media_agent": {
        "fetch_media_from_url",
        "sync_attachment_workspace",
        "inspect_attachment",
        "inspect_media_info",
        "convert_media_file",
        "separate_audio_stems",
        "clean_voice_track",
        "transcribe_media",
        "prepare_voice_dataset",
        "inspect_generated_file",
        "compose_file",
    },
    "speech_agent": {
        "sync_attachment_workspace",
        "inspect_attachment",
        "inspect_media_info",
        "separate_audio_stems",
        "clean_voice_track",
        "transcribe_media",
        "prepare_voice_dataset",
        "compose_file",
        "revise_generated_file",
        "inspect_generated_file",
    },
    "resource_agent": {
        "sync_attachment_workspace",
        "inspect_attachment",
        "read_attachment_section",
        "inspect_generated_file",
    },
}

AGENT_ALIASES: dict[str, str] = {
    "auto": "media_agent",
    "worker": "media_agent",
    "video_agent": "media_agent",
    "audio_agent": "media_agent",
    "file_agent": "document_agent",
    "doc_agent": "document_agent",
}
