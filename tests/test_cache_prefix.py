"""真实SDK隔离检查摘要前缀及分组：结构不变不等于真实缓存命中。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json

import httpx
from openai import OpenAI
import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.angelos.runtime_view import AngelosRuntime, ReasoningView
from shinku.angelos.usage_view import UsageView, summarize
from shinku.llm.cache_prefix import audit_scope, clean_prefix
from shinku.llm.context_compaction import ContextCompactor, continuation_guard
from shinku.llm.native_messages import NativeProtocolError, validate_native_messages
from shinku.llm.request_ledger import bind_provider_calls, observed_completion
from shinku.llm.runtime_types import ModelBundle
from test_angelos_usage import event, response
from test_context_compaction import Summarizer, payload, bundle
from test_expression_runtime import FixtureRuntime


@contextmanager
def wire(responder=None):
    """只替换HTTP响应；真实OpenAI SDK、payload、预算和账本沿用正式调用链。"""
    calls = []
    def handle(request):
        p = json.loads(request.content); calls.append(p)
        message = responder(p) if responder else {'content': json.dumps(
            {'summary': '保留约束与原文定位，实际结果仍需核验'} if is_summary(p) else {'reply': '继续'})}
        return httpx.Response(200, json={'id': 'offline-prefix', 'object': 'chat.completion', 'created': 0,
            'model': 'deepseek-flash', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', **message}}], 'usage': response()['usage']})
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        with OpenAI(api_key='offline-synthetic', base_url='https://offline.invalid/v1',
                    max_retries=0, http_client=http) as client:
            client._shinku_protocol = 'openai'
            runtime = FixtureRuntime(client.chat.completions.create)
            runtime.chat = runtime.aux = ModelBundle(client, 'deepseek-flash')
            runtime._env = lambda key, default='': {'LLM_THINKING_MODE': 'enabled',
                'LLM_REASONING_EFFORT': 'high',
                'NATIVE_TOOL_PROVIDER_ALLOWLIST': 'offline.invalid:deepseek-flash'}.get(key, default)
            yield runtime, calls


def is_summary(p):
    return '宿主历史整理请求' in p['messages'][-1].get('content', '') or '你只压缩历史' in p['messages'][0]['content']


def sdk_body(payload):
    """OpenAI SDK把extra_body展开至HTTP JSON；档案保存的仍是SDK原参数。"""
    result = copy.deepcopy(payload)
    result.update(result.pop('extra_body', {}))
    return result


def kwargs():
    return dict(system_prompt='保持权限和真实执行检查。返回JSON。', user_prompt='当前目标：只读分析',
        history_turns=[{'role': 'user', 'content': '合成背景' + 'a' * 6000}],
        native_tools=[{'type': 'function', 'function': {'name': 'read_file', 'parameters': {'type': 'object'}}}],
        native_tool_choice='auto', native_history=[{'role': 'assistant', 'content': '',
            'reasoning_content': 'SYNTHETIC_REASONING_FOR_TOOL_PAIR', 'tool_calls': [
                {'id': 'read-one', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'read-one', 'content': '合成原文回执'}],
        fallback={}, temperature=.25, prompt_cache_key='same-execution-cache')


def test_sdk_summary_keeps_exact_prefix_tools_thinking_and_only_disables_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '100000')
    view = UsageView(tmp_path/'usage.sqlite3')
    with wire() as (base, calls):
        host = AngelosRuntime(base, ReasoningView(), usage_sink=view,
                              compaction_path=tmp_path/'context.sqlite3', context_awareness=True)
        k = kwargs(); original = copy.deepcopy(k)
        with host.bind_task('task'):
            assert host.call_chat_json(**k) == {'reply': '继续'}
            host.request_context_compaction('task', '已读完资料，整理后继续分析')
            assert host.call_chat_json(**k) == {'reply': '继续'}
            assert host.call_chat_json(**k) == {'reply': '继续'}
        assert len(calls) == 4 and k == original
        before, summary, after, repeat = calls
        assert summary['messages'][:-1] == before['messages']
        assert summary['tools'] == before['tools'] == after['tools']
        assert summary['tool_choice'] == 'none' and after['tool_choice'] == 'auto'
        assert summary['temperature'] == before['temperature'] == .25
        assert all(p['reasoning_effort'] == 'high' and p['thinking'] == {'type': 'enabled'} for p in calls)
        assert all('max_tokens' not in p and 'max_completion_tokens' not in p for p in calls)
        assert 'SYNTHETIC_REASONING_FOR_TOOL_PAIR' in json.dumps(summary)
        validate_native_messages(summary['messages'][3:-1])
        assert after['messages'] == repeat['messages']
        rows = sorted(view.read(['task']), key=lambda r: r['started_at'])
        assert [r['prefix']['category'] for r in rows] == ['ordinary', 'summary', 'post_compaction', 'ordinary']
        assert rows[1]['prefix']['matchedMessages'] == len(before['messages'])
        assert rows[1]['prefix']['changedEnvelope'] == ['tool_choice']
        stats = summarize(rows)
        assert stats['requests'] == 4 and stats['cacheHitRatio'] == .8
        assert stats['cacheGroups']['ordinary']['requests'] == 2
        assert stats['cacheGroups']['summary']['requests'] == stats['cacheGroups']['post_compaction']['requests'] == 1
        saved = json.loads(host.compactor.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
        assert saved['summary_strategy'] == 'prefix_replay' and sdk_body(saved['original_payload']) == before
        # 签名库与展示库都不持久化正文；原件仍在独立压缩档案中。
        signatures = host.compactor.db.execute('SELECT payload FROM context_request_prefix').fetchone()[0]
        assert 'SYNTHETIC_REASONING' not in signatures and '当前目标' not in signatures
        assert 'SYNTHETIC_REASONING' not in json.dumps(rows)
        fresh = ContextCompactor(base, tmp_path/'context.sqlite3', aware=True)
        assert fresh.prefix_audit.observe_request('task', repeat)['category'] == 'ordinary'


@pytest.mark.parametrize('failure', ['tool', 'malformed', 'cancel', 'budget'])
def test_replay_failure_never_runs_tools_or_continuation_and_preserves_original(monkeypatch, tmp_path, failure):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '100000')
    allowed = [True]
    def answer(p):
        if is_summary(p):
            if failure == 'tool':
                return {'content': '', 'tool_calls': [{'id': 'forbidden', 'type': 'function',
                    'function': {'name': 'read_file', 'arguments': '{}'}}]}
            if failure == 'malformed':
                return {'content': '{bad json'}
            if failure == 'cancel':
                allowed[0] = False
            return {'content': '{"summary":"保留用户目标"}'}
        return {'content': '{"reply":"继续"}'}
    with wire(answer) as (base, calls):
        host = AngelosRuntime(base, ReasoningView(), context_awareness=True,
                              compaction_path=tmp_path/'context.sqlite3')
        budget = TaskBudget(max_requests=2, max_tokens=1000000)
        with host.bind_task('task'), bind_task_budget(budget), continuation_guard(lambda: allowed[0]):
            host.call_chat_json(**kwargs())
            host.request_context_compaction('task', '阶段完成')
            with pytest.raises(TaskBudgetExceeded if failure == 'budget' else NativeProtocolError):
                host.call_chat_json(**kwargs())
        assert len(calls) == budget.requests == 2
        assert calls[1]['tool_choice'] == 'none'
        saved = json.loads(host.compactor.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
        assert sdk_body(saved['original_payload']) == calls[0] and saved['raw_summary'] is not None
        assert saved['status'] == ('failed' if failure in ('tool', 'malformed') else 'completed')


def test_oversized_and_unsupported_replay_use_original_fragment_fallback(monkeypatch):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '3000')
    runtime = Summarizer(); host = ContextCompactor(runtime, aware=True)
    p = payload(14000); before = copy.deepcopy(p)
    host.project(p, bundle(), 2, 'task')
    saved = json.loads(host.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert saved['summary_strategy'] == 'serialized_fragments' and len(runtime.calls) >= 2 and p == before
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '100000')
    other = bundle(); other.client._shinku_protocol = 'anthropic'
    host.request('other', '阶段完成'); host.project(payload(6000), other, 2, 'other')
    saved = json.loads(host.db.execute('SELECT payload FROM context_compactions WHERE binding="other"').fetchone()[0])
    assert saved['summary_strategy'] == 'serialized_fragments'


def test_scope_and_change_diagnostics_are_persistent_and_do_not_invent_cache(tmp_path):
    host = ContextCompactor(Summarizer(), tmp_path/'context.sqlite3'); p = payload(30)
    view = UsageView()
    def send(binding, value):
        with audit_scope(host.prefix_audit, binding), bind_provider_calls(binding, view):
            observed_completion(lambda **kw: response(), value)
    send('a', p)
    appended = copy.deepcopy(p); appended['messages'].append({'role': 'user', 'content': '追加'})
    send('a', appended)
    changed = copy.deepcopy(appended); changed['messages'][0]['content'] += '修改'
    changed['tools'][0]['function']['name'] = 'changed'
    send('a', changed)
    send('b', p)
    rows = sorted(view.read(['a']), key=lambda r: r['started_at'])
    assert rows[1]['prefix']['matchedMessages'] == 3 and rows[1]['prefix']['firstChangedMessage'] == 3
    assert rows[2]['prefix']['matchedMessages'] == 0 and rows[2]['prefix']['changedEnvelope'] == ['tools']
    assert not view.read(['b'])[0]['prefix']['baseline']
    assert all(not r['prefix']['providerCacheEvidence'] for r in rows)
    assert clean_prefix({'available': True, 'category': '<script>'}) == {'available': False}
    fresh = ContextCompactor(Summarizer(), tmp_path/'context.sqlite3')
    assert fresh.prefix_audit.observe_request('a', changed)['matchedMessages'] == 4


def test_groups_preserve_unknown_invalid_and_weighted_usage_and_actual_order():
    rows = []
    for index, (category, inp, hit) in enumerate([('ordinary', 100, 90), ('summary', 1000, 100), ('post_compaction', 10, 0)]):
        row = event(request_id=str(9-index), reply=response(inp=inp, hit=hit))
        row.update(started_at=index+1, prefix={'available': True, 'category': category, 'baseline': False,
            'matchedMessages': 0, 'matchedMessageBytes': 0, 'messageCount': 1, 'firstChangedMessage': None, 'changedEnvelope': []})
        rows.append(row)
    stats = summarize(list(reversed(rows)))
    assert stats['cacheHitRatio'] == 190/1110
    assert [r['requestId'] for r in stats['recentRequests']] == ['9', '8', '7']
    assert stats['cacheGroups']['ordinary']['cacheHitRatio'] == .9
    rows.append(event(request_id='legacy', reply={}))
    stats = summarize(rows)
    assert stats['cacheGroups']['unclassified']['requests'] == 1 and stats['cacheHitRatio'] is None
    rows[1]['usage']['cache_hit_tokens'] = 2000
    assert summarize(rows)['cacheGroups']['summary']['cacheHitRatio'] is None
    assert summarize(rows)['invalidCache']


def test_observation_scope_does_not_leak_between_threads():
    host = ContextCompactor(Summarizer()); p = payload(10)
    def run(binding):
        events = []
        with audit_scope(host.prefix_audit, binding), bind_provider_calls(binding, events.append):
            observed_completion(lambda **kw: response(), p)
        return events[-1]['prefix']
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(run, ['a', 'b']))
    assert all(not r['baseline'] for r in result)


def test_deadline_uses_replay_and_second_summary_reuses_projected_history(monkeypatch, tmp_path):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '30000')
    with wire() as (base, calls):
        host = AngelosRuntime(base, ReasoningView(), context_awareness=True,
                              compaction_path=tmp_path/'context.sqlite3')
        k = kwargs(); k['history_turns'][0]['content'] = 'a'*98000
        with host.bind_task('task'):
            host.call_chat_json(**k)
            assert len(calls) == 2 and is_summary(calls[0])
            saved = json.loads(host.compactor.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
            assert saved['summary_strategy'] == 'prefix_replay'
            assert host.compactor.status('task')['projected_tokens'] < 30000*.9
            # 新增完整工具组及控制约束；第二次整理只压已整理投影+新增历史，
            # 保留最近完整组，不能重新塞回已归档的98k背景。
            k['native_history'] += [{'role': 'user', 'content': '新的用户约束必须逐字保留'},
                {'role': 'assistant', 'content': 'b'*4000}, {'role': 'assistant', 'content': '最新完整回答'}]
            host.call_chat_json(**k)
            warm = calls[-1]
            host.request_context_compaction('task', '阶段结束')
            host.call_chat_json(**k)
        summary = calls[-2]
        assert is_summary(summary)
        assert summary['messages'][:-1] == warm['messages'][:-1]
        assert 'a'*98000 not in json.dumps(summary)
        assert any(m.get('content') == '新的用户约束必须逐字保留' for m in calls[-1]['messages'])
        assert calls[-1]['messages'][-1]['content'] == '最新完整回答'


def test_failed_diagnostics_and_budget_rejection_do_not_dispatch_or_update_baseline(monkeypatch):
    host = ContextCompactor(Summarizer()); events = []; calls = []
    budget = TaskBudget(max_requests=1, max_tokens=1)
    from shinku.agent.budget import metered_completion
    with audit_scope(host.prefix_audit, 'task'), bind_provider_calls('task', events.append), bind_task_budget(budget):
        with pytest.raises(TaskBudgetExceeded):
            metered_completion(lambda **kw: observed_completion(lambda **p: calls.append(p), kw), payload())
    assert calls == events == []
    assert host.db.execute('SELECT count(*) FROM context_request_prefix').fetchone()[0] == 0
    def fail(*args):
        raise OSError('synthetic audit unavailable')
    monkeypatch.setattr(host.prefix_audit, 'observe_request', fail)
    with audit_scope(host.prefix_audit, 'task'), bind_provider_calls('task', events.append):
        assert observed_completion(lambda **kw: response(), payload()) == response()
    assert events[-1]['prefix'] == {'available': False}


def test_old_usage_database_keeps_rows_and_new_details_are_bounded(tmp_path):
    import sqlite3
    path = tmp_path/'usage.sqlite3'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE usage_calls (binding TEXT, request_id TEXT, status TEXT, payload TEXT, PRIMARY KEY(binding,request_id))')
    old = event(); db.execute('INSERT INTO usage_calls VALUES (?,?,?,?)', ('t', 'one', 'completed', json.dumps(old)))
    db.commit(); db.close()
    view = UsageView(path)
    for i in range(25):
        view(dict(event(request_id='new-'+str(i)), started_at=i+1))
    rows = view.read(['t']); stats = summarize(rows)
    assert stats['requests'] == 26 and len(stats['recentRequests']) == 20
    assert stats['omittedRequestDetails'] == 6 and stats['cacheGroups']['unclassified']['requests'] == 26
    assert next(r for r in rows if r['request_id'] == 'one') == old


@pytest.mark.parametrize('phase', ['preparation', 'answer_review', 'verification', 'candidate_review'])
def test_strict_stage_system_contract_does_not_enter_replay(monkeypatch, phase):
    from shinku.phase_timing import phase_scope
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '100000')
    with wire() as (base, calls):
        host = AngelosRuntime(base, ReasoningView(), context_awareness=True)
        with host.bind_task('prepare:task' if phase == 'preparation' else 'task'), phase_scope(phase):
            binding = host.binding.get()
            host.request_context_compaction(binding, '阶段整理')
            host.call_chat_json(**kwargs())
        assert len(calls) == 2
        assert '你只压缩历史' in calls[0]['messages'][0]['content']
        assert 'tools' not in calls[0]
        assert not is_summary(calls[1])
        saved = json.loads(host.compactor.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
        assert saved['summary_strategy'] == 'serialized_fragments'
