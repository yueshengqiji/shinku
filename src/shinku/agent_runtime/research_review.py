"""Host-selected JSON review transport; the execution transport stays separate."""
from __future__ import annotations

from .contracts import AgentBoundaryError, canonical_hash


class ResearchAnswerReviewRuntime:
    def __init__(self, execution_runtime, review_runtime, *, execution_profile_id, review_profile_id):
        if not callable(getattr(review_runtime, "call_chat_json", None)):
            raise AgentBoundaryError("research_review_runtime_unavailable")
        self.execution_runtime = execution_runtime
        self.review_runtime = review_runtime
        # The host changes this opaque identity whenever the review selection
        # changes. It does not expose endpoints, credentials or model prose.
        self.research_review_identity = canonical_hash({
            "execution": execution_profile_id, "research_review": review_profile_id})

    def __getattr__(self, name):
        return getattr(self.execution_runtime, name)

    def call_chat_json(self, **kwargs):
        return self.execution_runtime.call_chat_json(**kwargs)

    def call_research_answer_review_json(self, **kwargs):
        if any(kwargs.get(name) for name in (
                "native_tools", "native_tool_choice", "user_images", "native_history", "provider_message_sink")):
            raise AgentBoundaryError("research_review_tools_forbidden")
        options = dict(kwargs)
        options["prompt_cache_key"] = "research-review:" + self.research_review_identity
        # Failure remains a review failure. Never retry through the executor
        # or return an execution decision as a substitute for review evidence.
        return self.review_runtime.call_chat_json(**options)
