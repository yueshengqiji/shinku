"""真实OpenAI SDK合成SSE：展示先于完成，执行仍等待完整原生消息及结束信号。"""
from contextlib import contextmanager
import itertools
import json
from types import SimpleNamespace
import time
import threading

import httpx
from openai import OpenAI
import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.angelos.runtime_view import AngelosRuntime, ReasoningView
from shinku.angelos.usage_view import UsageView
from shinku.llm.context_compaction import continuation_guard
from shinku.llm.native_messages import NativeProtocolError
from shinku.llm.runtime_types import ModelBundle
from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD
from test_expression_runtime import FixtureRuntime


def delta(value=None, finish=None):
    return {'id': 'synthetic-stream', 'object': 'chat.completion.chunk', 'created': 0,
            'model': 'deepseek-flash', 'choices': [{'index': 0, 'delta': value or {}, 'finish_reason': finish}]}


USAGE = {'id': 'synthetic-stream', 'object': 'chat.completion.chunk', 'created': 0,
         'model': 'deepseek-flash', 'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
             'total_tokens': 120, 'prompt_cache_hit_tokens': 80, 'prompt_cache_miss_tokens': 20}}


@contextmanager
def stream_sdk(script, monkeypatch, *, allow_summary=False):
    from shinku.llm import json_stream
    ticks = itertools.count()
    monkeypatch.setattr(json_stream, 'time', SimpleNamespace(monotonic=lambda: next(ticks), time=time.time))
    calls, closed = [], []
    class Events(httpx.SyncByteStream):
        def __iter__(self):
            for item in script():
                yield b'data: ' + json.dumps(item, ensure_ascii=False).encode() + b'\n\n'
            yield b'data: [DONE]\n\n'
        def close(self): closed.append(True)
    def handle(request):
        p = json.loads(request.content); calls.append(p)
        if allow_summary and not p.get('stream'):
            return httpx.Response(200, json={'id': 'summary', 'object': 'chat.completion', 'created': 0,
                'model': 'deepseek-flash', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': '{"summary":"保留任务条件和读取定位"}'}}], 'usage': USAGE['usage']})
        assert p['stream'] is True and p['stream_options'] == {'include_usage': True}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Events())
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        with OpenAI(api_key='offline-synthetic', base_url='https://offline.invalid/v1',
                    max_retries=0, http_client=http) as client:
            client._shinku_protocol = 'openai'
            base = FixtureRuntime(client.chat.completions.create)
            base.chat = base.aux = ModelBundle(client, 'deepseek-flash'); base.incremental_json_supported = True
            base._swap_lock = threading.RLock()
            base._env = lambda key, default='': {'NATIVE_TOOL_PROVIDER_ALLOWLIST': 'offline.invalid:deepseek-flash',
                'LLM_THINKING_MODE': 'enabled', 'LLM_REASONING_EFFORT': 'high'}.get(key, default)
            yield base, calls, closed


def call(host, **extra):
    return host.call_chat_json(system_prompt='保持权限并返回JSON', user_prompt='合成请求', fallback={}, **extra)


def test_reasoning_visible_before_completion_and_final_protocol_is_exact(monkeypatch, tmp_path):
    view = ReasoningView(tmp_path/'reasoning.sqlite3'); ledger = UsageView(); seen = []; intermediate = []
    host = None
    raw = '  先核对合成条件。\n保留实际回执。  '
    arguments = ' {"path": "file\\\\name.py", "query": "a\\\"b"} '
    def script():
        yield delta({'reasoning_content': raw[:8]})
        intermediate.append((view.read('task'), list(seen)))
        yield delta({'reasoning_content': raw[8:]})
        yield delta({'tool_calls': [{'index': 0, 'id': 'read-', 'type': 'function',
                                    'function': {'name': 'read_', 'arguments': arguments[:12]}}]})
        intermediate.append((view.read('task'), list(seen)))
        yield delta({'tool_calls': [{'index': 0, 'id': 'one', 'function': {'name': 'file', 'arguments': arguments[12:]}}]})
        yield delta(finish='tool_calls'); yield USAGE
    with stream_sdk(script, monkeypatch) as (base, calls, closed):
        host = AngelosRuntime(base, view, usage_sink=ledger)
        budget = TaskBudget(max_requests=1, max_tokens=1000000)
        tools = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {'type': 'object'}}}]
        with host.bind_task('task'), bind_task_budget(budget):
            answer = call(host, native_tools=tools, provider_message_sink=seen.append)
        assert len(calls) == budget.requests == 1 and closed
        assert all('max_tokens' not in p and 'max_completion_tokens' not in p for p in calls)
        assert calls[0]['reasoning_effort'] == 'high'
        assert all(rows[0]['streaming'] and not messages for rows, messages in intermediate)
        assert intermediate[0][0][0]['text'] == raw[:8]
        assert len(seen) == 1 and seen[0]['reasoning_content'] == raw
        assert seen[0]['tool_calls'][0]['function']['arguments'] == arguments
        assert seen[0]['tool_calls'][0]['id'] == 'read-one'
        assert NATIVE_TOOL_CALL_FIELD in answer
        rows = view.read('task')
        assert len(rows) == 1 and rows[0]['text'] == raw and not rows[0]['streaming']
        assert not view.read('other')
        recorded = ledger.read(['task'])
        assert len(recorded) == 1 and recorded[0]['usage']['input_tokens'] == 100
        assert budget.reported_tokens == 120
        assert 'reasoning' not in json.dumps(answer)


@pytest.mark.parametrize('mode', ['missing_finish', 'length', 'content_filter', 'cancel', 'transport', 'bad_index'])
def test_partial_stream_never_delivers_a_decision_or_retries(monkeypatch, tmp_path, mode):
    allowed = [True]; seen = []; view = ReasoningView(tmp_path/'reasoning.sqlite3')
    def script():
        yield delta({'reasoning_content': '合成推理第一段。', 'content': '{"kind":"final","text":"原稿'})
        if mode == 'cancel': allowed[0] = False
        if mode == 'transport': raise httpx.ReadError('synthetic broken stream')
        if mode == 'bad_index':
            yield delta({'tool_calls': [{'index': 40, 'function': {'name': 'read', 'arguments': '{}'}}]})
        if mode in ('length', 'content_filter'): yield delta(finish=mode)
        if mode == 'cancel': yield delta({'content': '迟到片段'})
        yield USAGE
    with stream_sdk(script, monkeypatch) as (base, calls, closed):
        host = AngelosRuntime(base, view)
        with host.bind_task('task'), continuation_guard(lambda: allowed[0]):
            if mode == 'transport':
                assert call(host, provider_message_sink=seen.append) == {}
            else:
                with pytest.raises(NativeProtocolError): call(host, provider_message_sink=seen.append)
        assert len(calls) == 1 and closed and seen == []
        assert view.read('task')[0]['status'] in ('interrupted', 'incomplete')
        assert view.read('task')[0]['text'] == '合成推理第一段。'
        assert '原稿' in view.outputs('task')[0]['text']
        if mode != 'transport':
            assert view.outputs('task')[0]['diagnostics']['protocol_error'].startswith('model_stream_')


def test_split_secret_is_redacted_from_every_persisted_snapshot(monkeypatch):
    view = ReasoningView(); snapshots = []
    def script():
        for part in ('api_key=fixture', '-secret ', '正常推理。'):
            yield delta({'reasoning_content': part})
            snapshots.append(view.read('task'))
        yield delta({'content': '{"reply":"合成回答"}'}, finish='stop'); yield USAGE
    with stream_sdk(script, monkeypatch) as (base, calls, _):
        host = AngelosRuntime(base, view)
        with host.bind_task('task'): assert call(host) == {'reply': '合成回答'}
    assert all('fixture' not in json.dumps(rows) and 'secret ' not in json.dumps(rows) for rows in snapshots)
    assert '[redacted]' in view.read('task')[0]['text']


def test_view_failure_does_not_retry_or_change_json_result(monkeypatch):
    def script():
        yield delta({'reasoning_content': '合成推理。'})
        yield delta({'content': '{"reply":"继续"}'}, finish='stop'); yield USAGE
    with stream_sdk(script, monkeypatch) as (base, calls, _):
        view = ReasoningView()
        monkeypatch.setattr(view, 'update_stream', lambda *a: (_ for _ in ()).throw(OSError('synthetic full disk')))
        host = AngelosRuntime(base, view)
        with host.bind_task('task'): assert call(host) == {'reply': '继续'}
        assert len(calls) == 1


def test_streaming_continuation_keeps_compaction_and_usage_separate(monkeypatch, tmp_path):
    from test_cache_prefix import kwargs
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '100000')
    def script():
        yield delta({'reasoning_content': '合成推理。'})
        yield delta({'content': '{"reply":"继续"}'}, finish='stop'); yield USAGE
    with stream_sdk(script, monkeypatch, allow_summary=True) as (base, calls, _):
        ledger = UsageView(); view = ReasoningView()
        host = AngelosRuntime(base, view, usage_sink=ledger, context_awareness=True,
                              compaction_path=tmp_path/'context.sqlite3')
        with host.bind_task('task'):
            assert host.call_chat_json(**kwargs()) == {'reply': '继续'}
            host.request_context_compaction('task', '阶段完成')
            assert host.call_chat_json(**kwargs()) == {'reply': '继续'}
        assert len(calls) == 3 and calls[0]['stream'] and not calls[1].get('stream') and calls[2]['stream']
        assert calls[1]['tool_choice'] == 'none'
        assert len(view.read('task')) == 2
        rows = sorted(ledger.read(['task']), key=lambda r: r['started_at'])
        assert [r['prefix']['category'] for r in rows] == ['ordinary', 'summary', 'post_compaction']
        assert rows[1]['prefix']['matchedMessages'] == len(calls[0]['messages'])


def test_time_budget_stops_an_inflight_stream_before_decision(monkeypatch):
    from shinku.agent.budget import TaskBudgetExceeded
    budget = TaskBudget(max_requests=2, max_tokens=1000000, max_active_seconds=30)
    def script():
        yield delta({'reasoning_content': '合成第一段。'})
        budget.active_seconds = 30
        yield delta({'content': '{"reply":"迟到正文"}'}, finish='stop'); yield USAGE
    with stream_sdk(script, monkeypatch) as (base, calls, closed):
        host = AngelosRuntime(base, ReasoningView())
        with host.bind_task('task'), bind_task_budget(budget):
            with pytest.raises(TaskBudgetExceeded, match='task_time_budget_exhausted'): call(host)
        assert len(calls) == 1 and closed
