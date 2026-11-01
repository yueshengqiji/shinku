"""历史问答跨读取、未认证草稿与最终表达的合成回归；不使用真实身份或对话。"""
import json
from types import SimpleNamespace

import pytest

from shinku.agent_runtime.results import build_query_result
from shinku.conversation import AnswerEvidence
from shinku.result_expression.facts import archive_result
from shinku.result_expression.selective import materials
from shinku.tools.invocation import ToolResultEnvelope
from test_memory_read_acceptance import bridge_factory
from test_context_projection import NOW, put
from test_private_dialogue_continuity import turn


SOURCE = '那就叫“溪”吧，南庭溪；这只是我们约定的称呼。'
DRAFT = '当时我给你选的是南庭溪，作为我们约定的称呼。'


def query_material(*, status='ok', invalid=False):
    message = SimpleNamespace(message_id='current', sender_id='owner', text='你给我选的称呼是什么？')
    turn = SimpleNamespace(primary_message_id='current', conversation_type='private',
                           conversation_id='owner', messages=(message,))
    record = dict(record_id='event:source', scope='private:owner', source_id='turn:owner:old:assistant',
        text=SOURCE, created_at=1800000000, metadata=dict(role='assistant', sender_id='bot',
            requester_id='owner', source_version='version', reading_representation='source'))
    envelope = ToolResultEnvelope('read-1', status, SOURCE, {'tool_type': 'memory_search',
        'state_updates': {'memory_inspection': {'records': [record]}, 'memory_lookup': {
            'status': 'candidates', 'query_completed': True, 'mode': 'retrieve', 'record_count': 1}}})
    evidence = AnswerEvidence('history', 'unverified', False, 1, 1)
    source = build_query_result({'bot_history_sources_invalid': invalid}, turn, [envelope],
        route='perception', status='completed', text=DRAFT, answer_evidence=evidence)
    return source, materials(archive_result(source), 'a' * 64, 20000)


def test_history_draft_and_attributed_source_survive_export_without_becoming_verified():
    source, payload = query_material()
    assert payload['draft'] == DRAFT
    assert SOURCE in json.dumps(payload['read_observations'], ensure_ascii=False)
    row = payload['read_observations'][0]['records'][0]
    assert row['metadata']['role'] == 'assistant' and row['metadata']['requester_id'] == 'owner'
    assert row['created_at'] == 1800000000
    assert not source['verification']['accepted'] and 'verified_answer' not in source
    assert source['facts'] == [] and not source['execution_report']['is_fact_evidence']
    assert source['execution_report']['source'] == 'unverified_query_draft'


@pytest.mark.parametrize('status,invalid', [('error', False), ('ok', True)])
def test_failed_or_invalid_history_cannot_export_old_draft_or_source(status, invalid):
    source, payload = query_material(status=status, invalid=invalid)
    assert DRAFT not in json.dumps(source, ensure_ascii=False)
    assert SOURCE not in json.dumps(payload, ensure_ascii=False)


def test_expressed_history_keeps_budget_omission_explicit_without_mutating_archive():
    source, _ = query_material()
    source['read_observations'][0]['records'][0]['text'] = SOURCE * 500
    original = json.dumps(source, ensure_ascii=False, sort_keys=True)
    payload = materials(source, 'a' * 64, 1800)
    assert payload['material_excerpted'] and len(json.dumps(payload, ensure_ascii=False)) <= 1800
    assert json.dumps(source, ensure_ascii=False, sort_keys=True) == original


def read_call(query, *, limit=3):
    return {'kind': 'tool', 'tool_call': {'name': 'memory_search', 'arguments': {
        'mode': 'retrieve', 'query': query, 'limit': limit,
        'recorded_time': {'start': '', 'end': ''}}}}


def configure_history(bridge_factory, entry):
    bridge, chat, memory, coordinator = bridge_factory(entry, 'unused', read=False)
    put(memory.store, 'wrong', '错向命名：你想不想换个名字，你现在的名字是我给你起的。', age=6*86400)
    correct = put(memory.store, 'chosen:assistant', SOURCE, age=5*86400, role='assistant', sender='bot',
        requester_id='owner', delivery_status='sent', delivery_message_ids=['synthetic-receipt'],
        delivery_content_complete=True)
    return bridge, chat, memory, coordinator, correct


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_wrong_relationship_can_be_reread_and_original_reaches_final_expression(bridge_factory, entry):
    from shinku.result_expression import ExpressionProfile, ExpressionService, ExpressionArchive, ModelReply
    from shinku.qq.delivery import DeliveryResult
    bridge, chat, memory, coordinator, correct = configure_history(bridge_factory, entry)
    calls, reads, final_inputs, sent = [], [], [], []
    original = memory.inspect
    def inspect(**kw):
        reads.append(kw['query'])
        return original(**kw)
    memory.inspect = inspect
    def respond(**options):
        calls.append(options)
        if len(calls) == 1:
            return read_call('错向命名')
        if len(calls) == 2:
            assert '错向命名' in options['user_prompt']
            assert SOURCE not in options['user_prompt']
            assert {s['function']['name'] for s in options['native_tools']} == {'memory_search'}
            return read_call('南庭溪 约定称呼')
        assert len(calls) == 3 and SOURCE in options['user_prompt']
        return {'kind': 'final', 'text': DRAFT}
    chat.call_chat_json = respond
    class Writer:
        def call_expression_json(self, **options):
            payload = json.loads(options['user_prompt']);final_inputs.append(payload)
            assert SOURCE in json.dumps(payload['read_observations'], ensure_ascii=False)
            assert payload['draft'] == DRAFT and payload['host_state']['accepted'] is False
            return ModelReply({'binding_hash': payload['binding_hash'], 'text': DRAFT})
    bridge.result_expression = ExpressionService(ExpressionArchive(bridge.task_store),
        ExpressionProfile(Writer(), configuration_id='synthetic-history', content_policy='selective'))
    bridge.sender = lambda output: sent.append(output) or DeliveryResult(True, 'sent', ('synthetic-out',))
    reply = bridge(turn('你给我选的称呼是什么？', mid='reread'))
    assert reply.sent and reply.final_text == DRAFT and reply.agent_status == 'partial'
    assert reads == ['错向命名', '南庭溪 约定称呼']
    assert len(calls) == 3 and len(final_inputs) == 1 and len(sent) == 1
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())


@pytest.mark.parametrize('entry', ['legacy', 'task'])
@pytest.mark.parametrize('stop', ['duplicate', 'limit', 'other_tool'])
def test_reread_is_bounded_readonly_and_never_becomes_a_task(bridge_factory, entry, stop):
    bridge, chat, memory, coordinator, _ = configure_history(bridge_factory, entry)
    calls, reads = [], []
    original = memory.inspect
    def inspect(**kw):
        reads.append(kw['query']);return original(**kw)
    memory.inspect = inspect
    def respond(**options):
        calls.append(options)
        if stop == 'other_tool' and len(calls) > 1:
            return {'kind': 'tool', 'tool_call': {'name': 'agent_task_submit', 'arguments': {'goal': '写文件'}}}
        return read_call('错向命名' if stop != 'limit' else '错向命名 ' + str(len(calls)))
    chat.call_chat_json = respond
    reply = bridge(turn('你给我选的称呼是什么？', mid='stop-'+stop))
    assert len(reads) <= 3 and len(calls) <= 4
    if stop == 'duplicate':
        assert len(reads) == 1
    if stop == 'limit':
        assert len(reads) == 3 and len(calls) == 4
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_history_revoked_during_final_expression_is_not_sent(bridge_factory, entry):
    from shinku.result_expression import ExpressionProfile, ExpressionService, ExpressionArchive, ModelReply
    from shinku.qq.delivery import DeliveryResult
    bridge, chat, memory, coordinator, correct = configure_history(bridge_factory, entry)
    calls, sent, expression = [], [], []
    def respond(**options):
        calls.append(options)
        return read_call('南庭溪 约定称呼') if len(calls) == 1 else {'kind': 'final', 'text': DRAFT}
    chat.call_chat_json = respond
    class Writer:
        def call_expression_json(self, **options):
            expression.append(options)
            memory.store.forget(scope=correct.scope, record_id=correct.record_id, query='', now=NOW)
            return ModelReply({'binding_hash': json.loads(options['user_prompt'])['binding_hash'], 'text': DRAFT})
    bridge.result_expression = ExpressionService(ExpressionArchive(bridge.task_store),
        ExpressionProfile(Writer(), configuration_id='synthetic-revoke', content_policy='selective'))
    bridge.sender = lambda output: sent.append(output) or DeliveryResult(True, 'sent', ('not-allowed',))
    reply = bridge(turn('你给我选的称呼是什么？', mid='revoke-expression'))
    assert not reply.sent and sent == [] and len(expression) == 1
    assert len(calls) == 2 and reply.delivery_status == 'suppressed'


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_found_event_survives_expression_without_an_extra_read(bridge_factory, entry):
    from shinku.result_expression import ExpressionProfile, ExpressionService, ExpressionArchive, ModelReply
    from shinku.qq.delivery import DeliveryResult
    bridge, chat, memory, coordinator = bridge_factory(entry, 'unused', read=False)
    event = '星环缺口是小说里我们炸出来的；这里只是在讨论故事设定。'
    answer = '你说过星环缺口是故事里我们炸出来的，是小说设定。'
    put(memory.store, 'story', event, age=7*86400)
    calls, expressions, sent = [], [], []
    def respond(**options):
        calls.append(options)
        return read_call('星环缺口 小说') if len(calls) == 1 else {'kind': 'final', 'text': answer}
    class Writer:
        def call_expression_json(self, **options):
            payload = json.loads(options['user_prompt']);expressions.append(payload)
            assert event in json.dumps(payload['read_observations'], ensure_ascii=False)
            assert payload['draft'] == answer and not payload['host_state']['accepted']
            return ModelReply({'binding_hash': payload['binding_hash'], 'text': answer})
    chat.call_chat_json = respond
    bridge.result_expression = ExpressionService(ExpressionArchive(bridge.task_store),
        ExpressionProfile(Writer(), configuration_id='synthetic-event', content_policy='selective'))
    bridge.sender = lambda output: sent.append(output) or DeliveryResult(True, 'sent', ('story-out',))
    reply = bridge(turn('还记得我们说的星环缺口吗？', mid='story-recall'))
    assert reply.sent and reply.final_text == answer
    assert len(calls) == 2 and len(expressions) == 1 and len(sent) == 1


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_native_sdk_can_request_a_second_scoped_memory_read(bridge_factory, entry):
    from test_agent_conversation_loop import native_sdk
    bridge, chat, memory, coordinator, _ = configure_history(bridge_factory, entry)
    def respond(payload, index):
        messages = '\n'.join(str(m.get('content') or '') for m in payload['messages'])
        if index <= 2:
            if index == 2:
                assert '错向命名' in messages and SOURCE not in messages
                assert {t['function']['name'] for t in payload['tools']} == {'memory_search'}
            args = read_call('错向命名' if index == 1 else '南庭溪 约定称呼')['tool_call']['arguments']
            offered = {t['function']['name'] for t in payload['tools']}
            name = 'memory_search' if 'memory_search' in offered else '__shinku_tool_dispatch__'
            if name != 'memory_search':
                args = dict(request_kind='query', tool_name='memory_search', arguments=json.dumps(args))
            return dict(role='assistant', content='', tool_calls=[dict(id=f'read-{index}', type='function',
                function=dict(name=name, arguments=json.dumps(args)))]), 'tool_calls'
        assert index == 3 and SOURCE in messages
        return dict(role='assistant', content=json.dumps(dict(kind='final', text=DRAFT))), 'stop'
    with native_sdk(respond) as (runtime, payloads):
        env = runtime._env
        runtime._env = lambda key, default='': ('offline.example.invalid:deepseek-flash'
            if key == 'NATIVE_TOOL_PROVIDER_ALLOWLIST' else env(key, default))
        chat.call_chat_json = runtime.call_chat_json
        reply = bridge(turn('你给我选的称呼是什么？', mid='native-reread'))
    assert reply.final_text == DRAFT and len(payloads) == 3
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_permission_revoked_by_second_read_stops_before_another_generation(bridge_factory, entry):
    bridge, chat, memory, coordinator, _ = configure_history(bridge_factory, entry)
    calls, reads = [], []
    original = memory.inspect
    def inspect(**kw):
        reads.append(kw['query']);result = original(**kw)
        if len(reads) == 2:
            bridge.command_state.set_agent_enabled(False)
        return result
    memory.inspect = inspect
    def respond(**options):
        calls.append(options)
        return read_call('错向命名' if len(calls) == 1 else '南庭溪 约定称呼')
    chat.call_chat_json = respond
    reply = bridge(turn('你给我选的称呼是什么？', mid='revoked-after-read'))
    assert len(reads) == 2 and len(calls) == 2 and reply.agent_status == 'blocked'
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())
