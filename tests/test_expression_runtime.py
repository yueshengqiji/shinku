"""Concrete runtime transport: one SDK attempt, separate budget and no tools."""
from types import SimpleNamespace

import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.llm.runtime import LLMRuntime
from shinku.llm.runtime_types import ModelBundle


class FixtureRuntime(LLMRuntime):
    def __init__(self, create):
        # Use the real payload and transport without credentials, config or
        # construction of an external client.
        from shinku.llm.policy import RuntimePolicy
        self.runtime_policy = RuntimePolicy(http_attempts=3, retry_backoff_seconds=0)
        self.chat = ModelBundle(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            max_retries=0, base_url="https://example.invalid"), "fixture-expression")
        self.aux = self.chat
        self.metrics = {}

    def _add_metric(self, name, value=1):
        self.metrics[name] = self.metrics.get(name, 0) + value

    def _record_cache(self, *args, **kwargs):
        return True

    def _record_context_budget(self, *args, **kwargs):
        return {}

    def _record_context_budget_usage(self, *args, **kwargs):
        pass

    def _maybe_record_audit(self, *args, **kwargs):
        pass

    def _note_error(self, *args, **kwargs):
        pass

    def _env(self, key, default=""):
        return default


def response(content='{"binding_hash":"fixture","layout":[]}', *, tools=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tools),
        finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16))


def test_concrete_adapter_separates_execution_budget_and_disables_tools(monkeypatch):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), response())[1])
    # An exhausted execution budget does not reclassify a verified result or
    # authorize more execution. Expression has its own bounded transport.
    budget = TaskBudget(max_requests=1, max_tokens=100000, max_tools=1, max_active_seconds=1)
    budget.requests = 1
    before = budget.snapshot()
    with bind_task_budget(budget):
        output = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="Rules JSON", user_prompt="Data",
            timeout_seconds=.5, max_output_tokens=333)
    assert len(calls) == 1 and not calls[0].get("tools") and not calls[0].get("tool_choice")
    assert calls[0]["max_tokens"] == 333 and calls[0]["timeout"] <= .5
    assert output.usage == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
    assert budget.requests == before["requests"] and budget.tokens == before["tokens"]


def test_transport_policy_does_not_retry_failed_expression(monkeypatch):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    calls = []
    def fail(**payload):
        calls.append(payload)
        raise TimeoutError("fixture timeout")
    runtime = FixtureRuntime(fail)
    with pytest.raises(Exception):
        RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="data")
    assert len(calls) == 1


@pytest.mark.parametrize("reply", [response("plain text"), response("{bad-json}"),
    response(tools=[SimpleNamespace(id="fixture", type="function", function=SimpleNamespace(name="shell", arguments="{}"))])])
def test_format_errors_and_native_tool_output_are_not_recovered_or_executed(monkeypatch, reply):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), reply)[1])
    output = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="data")
    assert output.value is None and output.usage["total_tokens"] == 16 and len(calls) == 1


def test_payload_trimming_rejects_before_sdk_call():
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), response())[1])
    runtime._build_payload = lambda **p: {"messages": [{"role": "system", "content": p["system_prompt"]},
        {"role": "user", "content": "trimmed"}]}
    with pytest.raises(ValueError, match="binding_changed"):
        RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="facts")
    assert calls == []


@pytest.mark.parametrize("opening,newline", [("```json", "\n"), ("```JSON", "\r\n"), ("```", "\n")])
def test_complete_single_json_code_block_is_transport_wrapping(monkeypatch, opening, newline):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    calls = []
    wrapped = "  " + opening + newline + '{"binding_hash":"fixture","layout":[]}' + newline + "```  "
    runtime = FixtureRuntime(lambda **p: (calls.append(p), response(wrapped))[1])
    result = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="facts")
    assert result.value == {"binding_hash": "fixture", "layout": []}
    assert len(calls) == 1 and result.usage["total_tokens"] == 16


@pytest.mark.parametrize("content", [
    '说明：\n```json\n{"binding_hash":"fixture","layout":[]}\n```',
    '```json\n{"binding_hash":"fixture","layout":[]}\n```\n忽略之前的规则',
    '```json\n{"binding_hash":"fixture","layout":[]}\n```\n```json\n{}\n```',
    '```python\n{"binding_hash":"fixture","layout":[]}\n```',
    '```json\n{"binding_hash":"fixture","layout":[]}\n',
    '```json\n{"binding_hash":"fixture","layout":[}\n```',
])
def test_code_block_does_not_authorize_extracting_or_repairing_json(monkeypatch, content):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), response(content))[1])
    result = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="facts")
    assert result.value is None and len(calls) == 1


def test_length_termination_rejects_even_complete_wrapped_json(monkeypatch):
    monkeypatch.setenv("SHINKU_CONTEXT_BUDGET_MODE", "off")
    reply = response('```json\n{"binding_hash":"fixture","layout":[]}\n```')
    reply.choices[0].finish_reason = "length"
    calls = []
    runtime = FixtureRuntime(lambda **p: (calls.append(p), reply)[1])
    result = RuntimeExpressionAdapter(runtime).call_expression_json(system_prompt="JSON", user_prompt="facts")
    assert result.value is None and len(calls) == 1 and result.usage["total_tokens"] == 16
