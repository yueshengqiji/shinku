"""Host admission of bounded observations, not model-generated task authority."""

from collections.abc import Mapping


class ObservationRoutePolicy:
    """One capability-based boundary shared by initial chat and native recovery.

    A declared history reader has a bounded query/receipt contract already. It
    need not invent a persistent business task when a native provider omits
    textual routing metadata. Unknown tools, side-effect goals and explicit
    task routes still require the existing strict admission path.
    """

    _FORBIDDEN_EFFECTS = frozenset({"write", "delete", "send", "process_stop", "reminder", "browser_access"})
    _LEGACY_EXTERNAL_TOOLS = frozenset({
        "private_browser_tweet_account_resolve", "private_browser_tweet_feed",
        "private_netease_music_search", "private_web_search", "private_browser_open",
    })

    @classmethod
    def admits(cls, decision, *, handlers, policy, strict: bool,
               asynchronous: bool, action_goal: bool = False) -> bool:
        if decision.kind != "tool" or decision.task_contract is not None or action_goal:
            return False
        calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
        if not calls or len(calls) > 4:
            return False
        bounded_history = all(
            getattr(handlers.get(call.name), "agent_history_answer_source", False) is True
            and getattr(handlers.get(call.name), "agent_fast_perception", False) is True
            for call in calls
        )
        link = decision.task_link
        if isinstance(link, Mapping):
            if link.get("mode") != "none":
                return False
            # `track` is an advisory model label, never the authority for an
            # evidence domain. An admitted no-task history capability reads
            # history even if the model tags it public_query/agent. It cannot
            # grow a later tool chain; explicit new/continue stay task-bound.
            if link.get("track") in {"public_query", "agent"} and not bounded_history:
                return False
            if link.get("track") == "history" and not bounded_history:
                return False
        elif link is not None:
            return False
        # Omitted native metadata is safe only for a registered, explicitly
        # bounded history capability, not an arbitrary read that may start a task.
        if link is None and (strict or asynchronous) and not bounded_history:
            return False
        for call in calls:
            handler = handlers.get(call.name)
            if handler is None or call.name in cls._LEGACY_EXTERNAL_TOOLS:
                return False
            if getattr(handler, "agent_fast_perception", None) is False:
                return False
            if (str(getattr(handler, "agent_permission_class", "")).casefold() != "read"
                    or str(getattr(handler, "agent_progress_class", "")).casefold() != "observe"):
                return False
            effects = {str(value).casefold() for value in getattr(handler, "agent_completion_effects", ())}
            if effects & cls._FORBIDDEN_EFFECTS:
                return False
            if any(getattr(handler, field, "") for field in (
                "agent_research_candidate_field", "agent_research_content_field", "agent_evidence_collection_field",
            )):
                return False
            if not policy.accepts(call.name) or policy.requires_approval(call.name):
                return False
        return True
