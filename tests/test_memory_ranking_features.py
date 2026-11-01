"""评分分量与精确来源检索的合成反例；不依赖私用数据或真实模型。"""
from __future__ import annotations

import math
import sqlite3
import pytest

from shinku.memory import MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy
from shinku.memory.store_relevance import query_relevance


class Provider:
    dimension = 2
    def collection_key(self):
        return 'ranking-features-v1'
    def embed_text(self, text):
        score = .9 if 'relevant' in text else .8 if 'noise' in text else 1.0
        return [score, math.sqrt(1-score*score)]
    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


def store_at(tmp_path):
    return MemoryStore(tmp_path/'memory.sqlite3', embedding_provider=Provider(), semantic_enabled=True,
                       event_scan_limit=10, lifecycle_policy=LifecyclePolicy(mode='off'))


def add(store, source, text, *, scope='private:one', at=100, **kw):
    store.record_event(conversation_id=scope, scope=scope, source_id=source, text=text, created_at=at, **kw)


def test_auxiliary_signals_cannot_overturn_clear_relevance(tmp_path):
    store = store_at(tmp_path)
    add(store, 'answer', 'relevant', importance=.1, confidence=.5)
    add(store, 'other', 'noise', at=900, importance=1, confidence=1,
        metadata={'topic_batch_id':'same','sender_id':'same'})
    rows=store.search(query='question',scope='private:one',now=1000,topic_batch_id='same',sender_id='same')
    assert rows[0].source_id=='answer'
    for row in rows:
        components=row.metadata['ranking_components']
        assert components['auxiliary'] <= .05
        assert sum(components.values()) == pytest.approx(row.score)


@pytest.mark.parametrize('query,positive,negative', [
    ('R7 的结果','R7 已通过','R70 已通过'),
    ('“R7” 的结果','R7 已通过','R70 已通过'),
    ('file_v2.1.html','file_v2.1.html 已完成','file_v2.10.html 已完成'),
    ('file_v2.1.html','文件是 file_v2.1.html.','文件是 file_v2.1.html.bak'),
    ('A_B_LIMIT','A_B_LIMIT=20','OTHER_A_B_LIMIT=20'),
    ('“render_packet_w”','render_packet_window','packet_render_window'),
    ('“请保留原始记录”','操作中请保留原始记录','请保留处理后的记录'),
])
def test_exact_matching_obeys_identifier_boundaries_and_explicit_quotes(query,positive,negative):
    features=query_relevance(query,[])
    assert features.scores(positive)[1]==1
    assert features.scores(negative)[1]==0


@pytest.mark.parametrize('query', ['“什么”','之前是什么情况','normal words','123456'])
def test_generic_phrases_do_not_open_exact_channel(query):
    assert not query_relevance(query,[]).anchors


@pytest.mark.parametrize('kind',['event','fact'])
def test_old_unindexed_exact_source_is_retrieved_beyond_recent_window(tmp_path,monkeypatch,kind):
    store=store_at(tmp_path)
    if kind=='event':
        add(store,'old','ArcTile_17.svg 的原始图纸',at=1)
        for i in range(20):add(store,f'noise-{i}','noise ordinary',at=100+i)
    else:
        store.configure(fact_scan_limit=10)
        store.upsert_fact(conversation_id='one',scope='private:one',fact_key='old',source_id='old',text='ArcTile_17.svg 的原始图纸',now=1)
        for i in range(20):store.upsert_fact(conversation_id='one',scope='private:one',fact_key=f'n{i}',source_id=f'n{i}',text='noise ordinary',now=100+i)
    monkeypatch.setattr(store,'backfill_embeddings',lambda **kw:0)
    rows=store.search(query='ArcTile_17.svg 的说明',scope='private:one',now=1000)
    assert rows[0].source_id=='old'
    assert rows[0].metadata['exact_match_score']==1
    assert 'semantic_similarity' not in rows[0].metadata


def test_auxiliary_features_are_recomputed_from_source_not_stored_metadata(tmp_path):
    store=store_at(tmp_path)
    add(store,'match','relevant',importance=.1,confidence=.1,metadata={'ranking_auxiliary':{'importance':1,'recency':1,'topic':1,'sender':1}})
    row=store.search(query='question',scope='private:one',now=1000)[0]
    assert row.metadata['ranking_components']['auxiliary'] < .02


@pytest.mark.parametrize('mutation',['forgotten','edited'])
def test_exact_source_is_rechecked_after_other_retrieval_work(tmp_path,monkeypatch,mutation):
    store=store_at(tmp_path)
    add(store,'old','ArcTile_17.svg 的原始图纸',at=1)
    for i in range(20):add(store,f'n{i}','noise ordinary',at=100+i)
    monkeypatch.setattr(store,'backfill_embeddings',lambda **kw:0)
    def change(**kw):
        with sqlite3.connect(store.path) as conn:
            if mutation=='forgotten':conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='old'")
            else:conn.execute("UPDATE memory_events SET text='已更正为另一份材料' WHERE source_id='old'")
        return []
    monkeypatch.setattr(store,'_semantic_candidates',change)
    assert 'old' not in {r.source_id for r in store.search(query='ArcTile_17.svg',scope='private:one',now=1000)}


def test_exact_channel_preserves_scope_layer_time_and_forgetting(tmp_path,monkeypatch):
    store=store_at(tmp_path)
    for source,kw in [('valid',{'at':101}),('foreign',{'scope':'private:two','at':101}),
                      ('wrong-layer',{'layer':'summary','at':101}),('end',{'at':102}),('gone',{'at':101})]:
        add(store,source,'ArcTile_17.svg '+source,**kw)
    store.forget(query='gone',scope='private:one',now=1000)
    monkeypatch.setattr(store,'backfill_embeddings',lambda **kw:0)
    rows=store.search(query='ArcTile_17.svg',scope='private:one',now=1000,layers={'episodic'},start_at=101,end_at=102)
    assert [r.source_id for r in rows]==['valid']


def test_exact_features_survive_source_reread_with_search_index(tmp_path):
    store=store_at(tmp_path);store.configure_search_keys(enabled=True)
    add(store,'match','ArcTile_17.svg 已确认',metadata={'sender_id':'one'})
    row=store.search(query='ArcTile_17.svg',scope='private:one',now=1000)[0]
    assert row.metadata['exact_match_score']==1
    assert row.metadata['ranking_components']
    assert row.metadata['source_version'] and row.metadata['sender_id']=='one'


def test_exact_match_keeps_semantic_discrimination_between_quoting_sources(tmp_path):
    """精确片段相同不代表来源同样相关，不能把向量差异覆盖成同一个满分。"""
    store=store_at(tmp_path)
    add(store,'answer','ArcTile_17.svg relevant',importance=.1,confidence=.5)
    add(store,'quotation','ArcTile_17.svg noise',importance=.8,confidence=.9)
    rows=store.search(query='ArcTile_17.svg',scope='private:one',now=1000)
    assert rows[0].source_id=='answer'


def test_partial_identifier_noise_cannot_exhaust_exact_source_budget(tmp_path,monkeypatch):
    store=store_at(tmp_path)
    add(store,'old','R7 已验证',at=1)
    for i in range(25):add(store,f'n{i}',f'R70 已验证 {i}',at=100+i)
    monkeypatch.setattr(store,'backfill_embeddings',lambda **kw:0)
    rows=store.search(query='R7',scope='private:one',now=1000)
    assert rows[0].source_id=='old'
    assert all(r.metadata['exact_match_score']==0 for r in rows[1:])


def test_exact_anchor_survives_term_truncation_and_zero_importance(tmp_path,monkeypatch):
    store=store_at(tmp_path)
    add(store,'old','ArcTile_17.svg',at=1,importance=0)
    for i in range(25):add(store,f'n{i}','noise ordinary',at=100+i)
    monkeypatch.setattr(store,'backfill_embeddings',lambda **kw:0)
    rows=store.search(query='请帮我查找之前提到的一个很久以前保存的文件 ArcTile_17.svg',scope='private:one',now=1000)
    assert rows[0].source_id=='old'


def test_fact_auxiliary_signal_does_not_gain_extra_weight_from_record_type(tmp_path):
    store=store_at(tmp_path)
    add(store,'event','relevant',importance=.5,confidence=.6)
    store.upsert_fact(conversation_id='one',scope='private:one',fact_key='fact',source_id='fact',
                      text='relevant',importance=.5,confidence=.6,now=100)
    rows=store.search(query='question',scope='private:one',now=1000)
    assert {r.source_id for r in rows}=={'event','fact'}
    assert rows[0].score==pytest.approx(rows[1].score)
