from __future__ import annotations

import unittest

from shinku.tools.execution import (
    ExecutionPolicy,
    execute_invocation,
    result_to_envelope,
    validate_invocation,
    validation_to_envelope,
)
from shinku.tools.invocation import ToolInvocation


class FakeResult:
    tool_type = "search"
    followup_context = "找到了 2 条结果。"
    state_updates = {"last_url": "https://example.test"}
    stream_events = [{"type": "progress", "value": 1}]


class FakeHandler:
    tool_type = "search"

    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = FakeResult() if result is None else result
        self.error = error
        self.calls: list[tuple[dict, object]] = []

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        self.calls.append((call, context))
        if self.error:
            raise self.error
        return self.result


class ToolExecutionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = FakeHandler()
        self.handlers = {"search": self.handler}

    def test_policy_allows_empty_allowlist_and_denies_blocked_name(self) -> None:
        self.assertTrue(ExecutionPolicy().accepts("search"))
        self.assertFalse(ExecutionPolicy(blocked=frozenset({"search"})).accepts("search"))
        self.assertFalse(ExecutionPolicy(allowed=frozenset({"write"})).accepts("search"))

    def test_validation_accepts_good_call_and_rejects_missing_or_bad_args(self) -> None:
        self.assertTrue(validate_invocation(ToolInvocation("search", {"query": "x"}), self.handlers).ok)
        self.assertEqual(validate_invocation(ToolInvocation("search"), self.handlers).code, "bad_args")
        self.assertEqual(validate_invocation(ToolInvocation("missing"), self.handlers).code, "unknown_tool")
        self.assertEqual(validate_invocation(ToolInvocation("search", {"query": "x"}), self.handlers, policy=ExecutionPolicy(blocked=frozenset({"search"}))).code, "tool_blocked")

    def test_validation_without_handlers_is_safe(self) -> None:
        result = validate_invocation(ToolInvocation("search", {"query": "x"}), None)
        self.assertEqual((result.ok, result.code), (False, "unknown_tool"))

    def test_validation_envelope_is_model_readable_and_keeps_code(self) -> None:
        invocation = ToolInvocation("search", id="call-1")
        envelope = validation_to_envelope(invocation=invocation, validation=validate_invocation(invocation, {}))
        self.assertEqual(envelope.status, "error")
        self.assertIn("tool_use_error", envelope.model_feedback)
        self.assertEqual(envelope.data["code"], "unknown_tool")
        self.assertEqual(envelope.invocation_id, "call-1")

    def test_result_envelope_supports_object_and_mapping_results(self) -> None:
        object_envelope = result_to_envelope(invocation=ToolInvocation("search", id="o"), result=FakeResult())
        mapping_envelope = result_to_envelope(
            invocation=ToolInvocation("search", id="m"),
            result={"tool_type": "search", "followup_context": "ok", "state_updates": {"x": 1}, "stream_events": [{"type": "done"}]},
        )
        self.assertEqual(object_envelope.status, "ok")
        self.assertEqual(object_envelope.data["state_updates"]["last_url"], "https://example.test")
        self.assertEqual(mapping_envelope.model_feedback, "ok")
        self.assertEqual(mapping_envelope.events, [{"type": "done"}])

    def test_none_result_is_a_recoverable_error(self) -> None:
        envelope = result_to_envelope(invocation=ToolInvocation("search", id="empty"), result=None)
        self.assertEqual(envelope.data["code"], "empty_result")
        self.assertEqual(envelope.status, "error")

    def test_execute_passes_context_and_preserves_result(self) -> None:
        context = {"session_id": "s-1"}
        result, envelope = execute_invocation(ToolInvocation("search", {"query": "shinku"}), handlers=self.handlers, context=context)
        self.assertIsInstance(result, FakeResult)
        self.assertEqual(envelope.status, "ok")
        self.assertEqual(self.handler.calls, [({"type": "search", "query": "shinku"}, context)])

    def test_execute_validation_failure_does_not_call_handler(self) -> None:
        result, envelope = execute_invocation(ToolInvocation("search"), handlers=self.handlers)
        self.assertIsNone(result)
        self.assertEqual(envelope.data["code"], "bad_args")
        self.assertEqual(self.handler.calls, [])

    def test_execute_handler_exception_is_safe_and_retryable_by_caller(self) -> None:
        broken = FakeHandler(error=TimeoutError("secret endpoint"))
        result, envelope = execute_invocation(ToolInvocation("search", {"query": "x"}), handlers={"search": broken})
        self.assertIsNone(result)
        self.assertEqual(envelope.data["code"], "handler_error")
        self.assertEqual(envelope.data["error_type"], "TimeoutError")
        self.assertNotIn("secret endpoint", envelope.model_feedback)


if __name__ == "__main__":
    unittest.main()
