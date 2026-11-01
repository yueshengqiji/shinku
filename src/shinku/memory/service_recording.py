"""Service responsibilities: recording."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ForgetReport

from .service_models import MemoryServiceRecordingDependencies


from . import service_observation, service_turn_writes
from .service_recording_models import MemoryObservedMessage, MemoryRecordedTurn


def observe_message(message: Any, *, deps: MemoryServiceRecordingDependencies) -> bool:
    """Passively persist a normalized QQ message before reply routing.

        This method never calls an LLM and never decides whether Shinku should
        answer.  It only records bounded evidence for later local routing.
        """

    q = MemoryObservedMessage(message=message)
    if not service_observation.prepare_observation(q, deps):
        return False
    service_observation.read_observation_links(q, deps)
    service_observation.choose_observation_topic(q, deps)
    return service_observation.persist_observation(q, deps)


def record_turn(
    *,
    conversation_id: str,
    conversation_type: str,
    message_id: str,
    text: str,
    role: str='user',
    source_id: str='',
    metadata: Mapping[str, Any] | None=None,
    deps: MemoryServiceRecordingDependencies,
) -> ForgetReport | None:
    q = MemoryRecordedTurn(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        message_id=message_id,
        text=text,
        role=role,
        source_id=source_id,
        metadata=metadata,
    )
    if not service_turn_writes.prepare_recorded_turn(q, deps):
        return None
    service_turn_writes.note_turn_interaction(q, deps)
    service_turn_writes.write_turn_history(q, deps)
    from .participant_memory import participant_id, promote_participant_event
    if participant_id(deps.policy(), scope=q.scope, metadata=q.turn_metadata):
        # 参与者不是主人：其话语与回复仅按经历保存，不执行共享库记住/忘记命令。
        service_turn_writes.record_assistant_events(q, deps)
        promote_participant_event(deps.store(), deps.policy(), scope=q.scope,
            source_id=q.effective_source_id, owner_scope=deps.owner_scope(), now=q.now)
        return None
    if q.normalized_role != 'user':
        return service_turn_writes.record_assistant_events(q, deps)
    forgotten = service_turn_writes.forget_requested_memory(q, deps)
    if forgotten is not None:
        return forgotten
    service_turn_writes.write_user_memory(q, deps)
    return None
