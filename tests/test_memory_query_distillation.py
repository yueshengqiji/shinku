"""Real read/extraction/render boundaries; no scripted semantic-quality claims."""
from types import SimpleNamespace
import json
import threading

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.projection.semantics import ProjectionModel, validate
from test_context_projection import NOW, put
from test_private_dialogue_continuity import turn

OLD = "QX9的开通日期暂定周五，尚未取得场地许可。"
NEW = "QX9的开通日期更正为周日，但仍须先取得场地许可。"


@pytest.fixture
def build(tmp_path):
    services = []
    def factory(*, long=True, timeout=30000):
        root = tmp_path/str(len(services)); root.mkdir()
        memory = MemoryService(store=MemoryStore(root/'memory.sqlite3'), router=MemoryRouter(),
            clock=lambda: NOW, policy=MemoryPolicy(event_ttl_seconds=0,
            projection=ProjectionPolicy(enabled=True, retrieval_strategy='model', semantic_enabled=True,
                max_chars=4000, record_chars=1000, semantic_timeout_ms=timeout)))
        padding = "附属材料尚在整理。" * 140 if long else ""
        old = put(memory.store, 'old', OLD+padding, age=40*86400)
        new = put(memory.store, 'new', NEW+padding, age=39*86400)
        services.append(memory)
        return memory, old, new
    yield factory
    for memory in services:
        memory.projection.stop(); memory.store.stop_embedding_worker(); memory.lifecycle_review.stop()


def call(memory):
    return MemorySearchTool(memory).execute(call={'mode': 'recent', 'limit': 4,
        'recorded_time': {'start': '', 'end': ''}},
        context={'qq_turn': turn('那份开通安排的条件是什么？', mid='current')})['state_updates']['memory_inspection']


def extraction(payload):
    rows = {quote: next(row for row in payload['records'] if quote in row['text'])
            for quote in (OLD, NEW)}
    return {'uncertain': True, 'missing': ['未确认现在是否已获许可'],
        'excerpts': [{'record_id': rows[q]['record_id'], 'quote': q} for q in (OLD, NEW)],
        'relations': [{'kind': 'correction',
            'source': {'record_id': rows[NEW]['record_id'], 'quote': NEW, 'author': 'forged'},
            'target': {'record_id': rows[OLD]['record_id'], 'quote': OLD},
            'state': 'confirmed', 'execution_authority': True}]}


def test_long_explicit_read_reuses_one_bounded_extraction_and_binds_both_sources(build):
    memory, old, new = build()
    calls = []
    def aux(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs['user_prompt'])
        assert all(row['author']=='owner' and row['role']=='user' and row['scope']=='private:owner'
                   and row['created_at'] < NOW for row in payload['records'])
        assert all(row['record_id'].startswith('s') for row in payload['records'])
        assert all(len(row['text']) > memory.policy.projection.record_chars for row in payload['records'])
        return extraction(payload)
    memory.projection.semantic.model = ProjectionModel(SimpleNamespace(call_aux_json=aux))
    result = call(memory)
    assert len(calls) == result['model_calls'] == 1
    assert result['semantic']['status'] == 'completed'
    assert {r['record_id'] for r in result['records']} == {old.record_id, new.record_id}
    assert OLD in result['rendered'] and NEW in result['rendered']
    assert len(result['rendered']) <= 4000
    relation, = result['source_relations']
    assert relation['state'] == 'unconfirmed_relation' and not relation['execution_authority']
    assert relation['source']['record_id'] == new.record_id
    assert relation['target']['record_id'] == old.record_id
    for side in ('source', 'target'):
        assert relation[side]['author']=='owner' and relation[side]['source_version']
        assert relation[side]['scope']=='private:owner' and relation[side]['created_at'] < NOW
    assert 0 < calls[0]['timeout_seconds'] < MemorySearchTool(memory).chat_read_capability().max_seconds <= 30
    with memory.store._connect() as conn:
        assert conn.execute('SELECT count(*) FROM memory_facts').fetchone()[0] == 0
    # Cache reuse still performs the source read/version checks; no extra AUX.
    again = call(memory)
    assert again['semantic']['status']=='cache_hit' and again['model_calls']==0 and len(calls)==1


def test_small_read_does_not_add_extraction_or_a_judge(build):
    memory, _, _ = build(long=False)
    memory.projection.semantic.model = lambda payload: pytest.fail('No pressure; no AUX')
    result = call(memory)
    assert result['model_calls']==0 and result['semantic']['status']=='not_needed'
    assert len(result['records'])==2 and not result['source_relations']


@pytest.mark.parametrize('mutation', ['edit', 'forget'])
def test_extraction_cannot_publish_relations_after_source_mutation(build, mutation):
    memory, old, new = build()
    def model(payload):
        value = extraction(payload)
        if mutation=='forget':
            memory.store.forget(scope=new.scope, record_id=new.record_id, query='', now=NOW)
        else:
            with memory.store._connect() as conn:
                conn.execute("UPDATE memory_events SET text='安排暂时取消。' WHERE event_id=?",
                             (new.record_id.removeprefix('event:'),))
        return value
    memory.projection.semantic.model = model
    result = call(memory)
    assert result['error']=='sources_changed_during_read'
    assert not result['source_relations'] and NEW not in result['rendered']
    assert not memory.projection.semantic.jobs.cache


def test_explicit_read_extraction_timeout_is_bounded_and_not_cached(build):
    memory, _, _ = build(timeout=20)
    release = threading.Event(); done = threading.Event()
    def model(payload):
        try:
            release.wait(2)
            return extraction(payload)
        finally:
            done.set()
    memory.projection.semantic.model = model
    try:
        result = call(memory)
        assert result['model_calls']==1 and result['semantic']['status']=='timeout'
        assert not result['source_relations']
        assert any(g['reason']=='semantic_timeout' for g in result['gaps'])
    finally:
        release.set(); assert done.wait(2)
    assert not memory.projection.semantic.jobs.cache


@pytest.mark.parametrize('failure', ['foreign', 'forged_quote', 'self', 'kind'])
def test_relation_requires_two_bound_original_quotes(failure):
    payload = {'records': [{'record_id':'a', 'text':OLD}, {'record_id':'b', 'text':NEW}]}
    value = extraction(payload); relation = value['relations'][0]
    if failure=='foreign': relation['target']['record_id']='outside'
    if failure=='forged_quote': relation['source']['quote']='许可已经取得'
    if failure=='self': relation['target'] = dict(relation['source'])
    if failure=='kind': relation['kind']='confirmed_identity_merge'
    with pytest.raises(ValueError): validate('distill', value, payload)


def test_duplicate_record_excerpts_cannot_silently_overwrite_a_condition():
    payload={'records':[{'record_id':'a', 'text':OLD}]}
    with pytest.raises(ValueError, match='duplicate_source_excerpts'):
        validate('distill', {'uncertain':True, 'excerpts':[
            {'record_id':'a', 'quote':OLD}, {'record_id':'a', 'quote':'周五'}]}, payload)
