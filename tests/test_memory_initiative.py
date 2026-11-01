"""主动发现只交付旧来源位置；合成主题、真实 SQLite/读取器和受控模型。"""
import json
from dataclasses import replace

import pytest

from test_memory_cross_context import memory, group_pair, projected, inspect
from test_context_projection import NOW, put
from test_memory_read_acceptance import bridge_factory


TOPIC = '你提到 Starforge 我就想到那个配乐团。'
OLD = '我一直在玩 Starforge，也常听他们的配乐；休息时才会开游戏。'


def test_current_topic_discovers_old_source_without_recall_keywords(memory):
    service, clock = memory
    group_pair(service, clock, text=OLD, age=6*86400, mid='old-topic')
    view = projected(service, query=TOPIC)
    related = [r for r in view['source_catalog'] if r.get('discovery')=='query_related']
    assert related and any(r['origin_scope']=='group:g-one' for r in related)
    assert OLD not in view['rendered'] and '休息时' not in json.dumps(related,ensure_ascii=False)
    assert all(r['content_not_read_in_prompt'] and not r['execution_authority'] for r in related)
    assert all((r['record_id'],r['scope'],r['source_version']) in view['source_dependencies'] for r in related)
    assert view['diagnostics']['query_catalog_count']>=1
    result = inspect(service, query='Starforge', mode='retrieve',
        source_ids=[r['record_id'] for r in related])
    assert OLD in result['rendered'] and not result.get('error')


@pytest.mark.parametrize('mutation',['forget','edit','expire'])
def test_related_catalog_never_bypasses_reread_validation(memory,mutation):
    service,clock=memory
    group_pair(service,clock,text=OLD,age=5*86400,mid='old-topic')
    first=projected(service,query=TOPIC)
    related=[r for r in first['source_catalog'] if r.get('discovery')=='query_related']
    assert related
    rid=related[0]['record_id'].split(':',1)[1]
    with service.store._lock,service.store._connect() as db:
        if mutation=='forget':db.execute("update memory_events set status='forgotten' where event_id=?",(rid,))
        if mutation=='edit':db.execute("update memory_events set text='Changed source; old wording unavailable.' where event_id=?",(rid,))
        if mutation=='expire':db.execute('update memory_events set expires_at=? where event_id=?',(NOW-1,rid))
    current=projected(service,query=TOPIC,mid='after-change')
    same=[r for r in current['source_catalog'] if r['record_id']==related[0]['record_id']]
    assert not same or same[0]['source_version']!=related[0]['source_version']


@pytest.mark.parametrize('sender,kind,conversation',[('other','private','other'),('owner','group','g-two')])
def test_unavailable_shared_scope_does_not_run_topic_discovery(memory,sender,kind,conversation):
    service,clock=memory
    group_pair(service,clock,text=OLD,age=5*86400)
    service.projection.archive.discover=lambda *a,**k:pytest.fail('共享目录未获准时不能做相关来源发现')
    view=projected(service,query=TOPIC,sender=sender,kind=kind,conversation=conversation)
    assert not view['source_catalog']


def test_topic_discovery_timeout_retains_recent_directory_and_total_bounds(memory):
    service,clock=memory
    group_pair(service,clock,text='近期的安排尚未确认。',age=600,mid='recent')
    def timeout(request,deadline,*,limit):
        assert deadline.remaining<=.251 and limit<=8
        raise TimeoutError('synthetic bounded discovery')
    service.projection.archive.discover=timeout
    view=projected(service,query=TOPIC)
    assert view['source_catalog'] and len(view['source_catalog'])<=8
    assert all(r.get('discovery')=='recent' for r in view['source_catalog'])
    assert len(view['rendered'])<=service.policy.projection.max_chars
    assert view['diagnostics']['candidate_reads']<=service.policy.projection.candidate_limit


def test_old_personal_statement_is_visible_as_location_not_automatic_fact(memory):
    service,clock=memory
    group_pair(service,clock,text=OLD,age=5*86400)
    group_pair(service,clock,group='busy',mid='newer',text='当前群聊的另一件事情。',age=60)
    view=projected(service,query=TOPIC)
    related=[r for r in view['source_catalog'] if r.get('discovery')=='query_related']
    assert related and related[0]['source_kind']=='user_statement'
    assert not any('text' in r or 'summary' in r for r in related)
    assert OLD not in view['rendered'] and view['topic']['unresolved']


def test_read_prompt_distinguishes_authority_coverage_and_remaining_reads():
    from shinku.qq.perception_query import _history_read_basis
    observation={'invocation_id':'one','records':[
        {'record_id':'event:guess','text':'助手以前推测的说法','metadata':{'role':'assistant'}},
        {'record_id':'event:user','text':'用户的条件原话','metadata':{'role':'user'}}],
        'lookup':{'coverage':{'complete':False,'truncated':True}}}
    data=_history_read_basis([observation],remaining_reads=2)
    assert data['user_statement_ids']==['event:user']
    assert data['assistant_utterance_ids']==['event:guess']
    assert data['remaining_reads']==2 and data['coverage_incomplete']
    assert '助手以前推测的说法' not in json.dumps(data,ensure_ascii=False)


@pytest.mark.parametrize('entry',['legacy','task'])
def test_guidance_reaches_first_sdk_request_without_forcing_a_read(bridge_factory,entry):
    from shinku.qq.memory_guidance import MEMORY_READ_GUIDANCE
    from test_agent_conversation_loop import native_sdk
    from test_private_dialogue_continuity import turn
    bridge,chat,service,coordinator=bridge_factory(entry,'unused',read=False)
    reads=[]
    original=service.inspect
    service.inspect=lambda **kw: reads.append(kw) or original(**kw)
    def respond(payload,index):
        system='\n'.join(m['content'] for m in payload['messages'] if m['role']=='system')
        assert MEMORY_READ_GUIDANCE in system
        assert '个人经历判断' in system and 'query_related' in system and '旧助手推测' in system
        return {'role':'assistant','content':'{"kind":"final","text":"早呀，今天想聊什么？"}'},'stop'
    with native_sdk(respond) as (runtime,calls):
        chat.call_chat_json=runtime.call_chat_json
        reply=bridge(turn('早呀。',mid='greeting'))
        assert reply.final_text=='早呀，今天想聊什么？' and len(calls)==1
        assert reads==[] and not reply.background_task_id
        assert coordinator is None or not coordinator.list()


def test_small_candidate_budget_keeps_old_topic_and_recent_origins(memory):
    service, clock = memory
    service.projection.policy = replace(service.projection.policy, candidate_limit=8)
    group_pair(service, clock, text=OLD, age=6*86400, mid='old-topic')
    for index in range(4):
        group_pair(service, clock, group=f'recent-{index}', mid=f'recent-{index}',
                   text='最近另一处的安排尚未确认。', age=600+index)
    view = projected(service, query=TOPIC)
    assert any(row['discovery']=='query_related' for row in view['source_catalog'])
    assert any(row['discovery']=='recent' for row in view['source_catalog'])
    assert view['diagnostics']['candidate_reads']<=8


def test_related_catalog_uses_retained_vectors_without_lexical_match_or_index_rebuild(tmp_path, monkeypatch):
    from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
    from shinku.memory.projection.contracts import ProjectionPolicy
    query = '休憩打开星铸战役'
    encoded = []
    class Encoder:
        name, version, dimension = 'controlled', 'initiative', 2
        def collection_key(self):
            return 'controlled_initiative_2'
        def embed_text(self, text):
            encoded.append(text)
            return [1., 0.] if text in {OLD, query} else [0., 1.]
        def embed_texts(self, texts):
            return [self.embed_text(text) for text in texts]
    store = MemoryStore(tmp_path/'vectors.db', embedding_provider=Encoder(), semantic_enabled=True)
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda:NOW,
        policy=MemoryPolicy(master_user_ids=('owner',), event_ttl_seconds=0,
            projection=ProjectionPolicy(enabled=True, retrieval_strategy='model')))
    try:
        target = put(store, 'old-vector', OLD, age=6*86400)
        store.backfill_embeddings(limit=32)
        with store._connect() as conn:
            conn.execute('DROP TABLE memory_projection_fts')
        monkeypatch.setattr(store, 'backfill_embeddings', lambda **kw:pytest.fail('请求内不能补建向量'))
        monkeypatch.setattr(service.projection.archive, 'maintain', lambda:pytest.fail('请求内不能维护索引'))
        encoded.clear()
        view = projected(service, query=query)
        assert target.record_id in {r['record_id'] for r in view['source_catalog'] if r['discovery']=='query_related'}
        assert OLD not in view['rendered'] and encoded==[query]
        assert view['diagnostics']['maintenance_in_request']==0
    finally:
        service.projection.stop()
        store.stop_embedding_worker()
        service.lifecycle_review.stop()


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_old_related_catalog_change_during_generation_rejects_candidate(bridge_factory, entry):
    from test_private_dialogue_continuity import turn
    bridge, chat, service, _ = bridge_factory(entry, 'unused', read=False)
    service.policy = replace(service.policy, master_user_ids=('owner',))
    clock = [NOW]
    service.clock = lambda:clock[0]
    group_pair(service, clock, text=OLD, age=6*86400, mid='old-topic')
    def respond(**options):
        assert 'query_related' in options['user_prompt'] and OLD not in options['user_prompt']
        service.store.forget(scope='group:g-one', record_id='', query='Starforge', now=NOW)
        return dict(kind='final', text='失效位置不应支持这句答复。')
    chat.call_chat_json = respond
    reply = bridge(turn(TOPIC, mid='old-catalog-change'))
    assert reply.agent_status=='failed' and '失效位置不应' not in reply.final_text


def sdk_read(payload, index, *, minimal=False, **arguments):
    """按真实入口提供的 schema 调用，不绕过紧凑派发器或参数解析。"""
    arguments = dict(mode='retrieve', limit=4, recorded_time={'start':'', 'end':''}, **arguments)
    if minimal:
        # 复现真实模型仅填写 schema 必填日期与精确来源、略去可选 mode/query 的返回。
        arguments = {key:arguments[key] for key in ('recorded_time','source_ids')}
    offered = {row['function']['name'] for row in payload['tools']}
    name = 'memory_search' if 'memory_search' in offered else '__shinku_tool_dispatch__'
    assert name in offered
    if name=='__shinku_tool_dispatch__':
        arguments = dict(request_kind='query', tool_name='memory_search', arguments=json.dumps(arguments))
    return dict(role='assistant', content='', tool_calls=[dict(id=f'read-{index}', type='function',
        function=dict(name=name, arguments=json.dumps(arguments)))]), 'tool_calls'


def sdk_basis(payload):
    text = '\n'.join(str(row.get('content') or '') for row in payload['messages'])
    marker = '【本轮历史依据边界，仅为署名和预算】\n'
    return json.loads(text.split(marker, 1)[1].splitlines()[0])


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_sdk_current_topic_reads_old_catalog_without_recall_words(bridge_factory, entry):
    from test_agent_conversation_loop import native_sdk
    from test_private_dialogue_continuity import turn
    bridge, chat, service, coordinator = bridge_factory(entry, 'unused', read=False)
    service.policy = replace(service.policy, master_user_ids=('owner',))
    clock = [NOW]
    service.clock = lambda:clock[0]
    group_pair(service, clock, text=OLD, age=6*86400, mid='old-topic')
    reads = []
    original = service.inspect
    def inspect_read(**options):
        result = original(**options)
        reads.append((options, result))
        return result
    service.inspect = inspect_read
    def respond(payload, index):
        text = '\n'.join(str(row.get('content') or '') for row in payload['messages'])
        if index==1:
            assert OLD not in text
            header = next(json.loads(line) for line in text.splitlines()
                          if line.startswith('{') and '"source_catalog"' in line)
            ids = [row['record_id'] for row in header['source_catalog'] if row['discovery']=='query_related']
            assert ids
            return sdk_read(payload, index, query='', source_ids=ids)
        assert index==2 and OLD in text
        assert sdk_basis(payload)['user_statement_ids']
        return dict(role='assistant', content=json.dumps(dict(kind='final', text='你早就在玩，休息时才开游戏。'))), 'stop'
    with native_sdk(respond) as (runtime, calls):
        env = runtime._env
        runtime._env = lambda key, default='': ('offline.example.invalid:deepseek-flash'
            if key=='NATIVE_TOOL_PROVIDER_ALLOWLIST' else env(key, default))
        chat.call_chat_json = runtime.call_chat_json
        reply = bridge(turn(TOPIC, mid='topic-no-recall'))
    assert reply.agent_status=='completed' and reply.final_text=='你早就在玩，休息时才开游戏。'
    assert len(reads)==1 and len(calls)==2 and reads[0][0]['source_ids']
    assert any(row['text']==OLD for row in reads[0][1]['records'])
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())


@pytest.mark.parametrize('entry', ['legacy', 'task'])
@pytest.mark.parametrize('source_only', [False, True])
def test_sdk_weak_assistant_read_can_be_followed_by_user_source_and_expression(bridge_factory, entry, source_only):
    from shinku.result_expression import ExpressionProfile, ExpressionService, ExpressionArchive, ModelReply
    from shinku.qq.delivery import DeliveryResult
    from test_agent_conversation_loop import native_sdk
    from test_private_dialogue_continuity import turn
    bridge, chat, service, coordinator = bridge_factory(entry, 'unused', read=False)
    guess = '猜测标签：Starforge 你应该刚入门，我准备帮你预习。'
    put(service.store, 'guess:assistant', guess, age=600, role='assistant', sender='bot',
        requester_id='owner', delivery_status='sent', delivery_message_ids=['guess-out'],
        delivery_content_complete=True)
    correct = put(service.store, 'old-user', OLD, age=6*86400)
    answer = '你早就在玩 Starforge，只在休息时开游戏；之前是我自己猜错了。'
    reads, expressed, sent = [], [], []
    original = service.inspect
    def inspect_read(**options):
        result = original(**options)
        reads.append(result)
        return result
    service.inspect = inspect_read
    def respond(payload, index):
        text = '\n'.join(str(row.get('content') or '') for row in payload['messages'])
        if index==1:
            return sdk_read(payload, index, query='猜测标签')
        basis = sdk_basis(payload)
        if index==2:
            assert guess in text and OLD not in text
            assert basis['assistant_utterance_ids'] and not basis['user_statement_ids']
            assert basis['remaining_reads']==2 and basis['coverage_incomplete']
            assert '主动换查询或定位符补查' in text
            if source_only:
                return sdk_read(payload, index, minimal=True, source_ids=[correct.record_id])
            return sdk_read(payload, index, query='Starforge 配乐 休息')
        assert index==3 and OLD in text
        assert basis['user_statement_ids'] and basis['assistant_utterance_ids']
        assert basis['remaining_reads']==1 and basis['source_validity_is_not_claim_support']
        return dict(role='assistant', content=json.dumps(dict(kind='final', text=answer))), 'stop'
    class Writer:
        def call_expression_json(self, **options):
            payload = json.loads(options['user_prompt'])
            expressed.append(payload)
            sources = json.dumps(payload['read_observations'], ensure_ascii=False)
            assert OLD in sources and guess in sources
            assert payload['draft']==answer and not payload['host_state']['accepted']
            assert '旧助手推测不能循环佐证用户经历' in options['system_prompt']
            return ModelReply({'binding_hash':payload['binding_hash'], 'text':answer})
    bridge.result_expression = ExpressionService(ExpressionArchive(bridge.task_store),
        ExpressionProfile(Writer(), configuration_id='synthetic-initiative', content_policy='selective'))
    bridge.sender = lambda output: sent.append(output) or DeliveryResult(True, 'sent', ('synthetic-out',))
    with native_sdk(respond) as (runtime, calls):
        env = runtime._env
        runtime._env = lambda key, default='': ('offline.example.invalid:deepseek-flash'
            if key=='NATIVE_TOOL_PROVIDER_ALLOWLIST' else env(key, default))
        chat.call_chat_json = runtime.call_chat_json
        reply = bridge(turn('Starforge 的配乐说到哪了？', mid='weak-then-reread'))
    # 表达可以送出有来源的回答，但宿主仍保留未做语义认证的 partial 状态。
    assert reply.agent_status=='partial' and reply.final_text==answer and reply.sent
    assert len(reads)==2 and len(calls)==3 and len(expressed)==len(sent)==1
    assert not reply.background_task_id and (coordinator is None or not coordinator.list())


@pytest.mark.parametrize('changes,expected', [
    ({}, 'retrieve'),
    ({'mode':'retrieve'}, 'retrieve'),
    ({'source_ids':[]}, 'recent'),
    ({'mode':'recent'}, None),
    ({'mode':'summary'}, None),
    ({'mode':'activity'}, None),
    ({'task_id':'synthetic-task'}, None),
    ({'recorded_time':{}}, None),
    ({'source_ids':['event:x']*9}, None),
])
def test_optional_mode_for_source_ids_preserves_explicit_restrictions(memory, changes, expected):
    from shinku.memory import MemorySearchTool
    tool = MemorySearchTool(memory[0])
    call = dict(source_ids=['event:synthetic'], recorded_time={'start':'','end':''})
    call.update(changes)
    normalized = tool.normalize_call(call)
    if expected is None:
        assert normalized is None
    else:
        assert normalized['mode']==expected and normalized['query']==''
