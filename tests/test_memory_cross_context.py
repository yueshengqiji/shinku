"""跨会话资料发现与多锚点回归；只使用合成身份、真实 SQLite/读取器和 Bridge。"""
from dataclasses import replace
import json

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.projection.sources import ProjectionSources
from shinku.qq.message import IncomingMessage
from test_context_projection import NOW, put
from test_memory_read_acceptance import bridge_factory
from test_private_dialogue_continuity import turn
from test_agent_conversation_loop import native_sdk


GAME = "我本来想玩 Oriole，后来改玩了 Riverstep；还没决定明晚玩哪款。"
REPLY = "原来你最后开的是 Riverstep，明晚的安排还没有定。"


@pytest.fixture
def memory(tmp_path):
    clock = [NOW]
    service = MemoryService(store=MemoryStore(tmp_path / 'memory.sqlite3'),
        router=MemoryRouter(judge=lambda **_: pytest.fail('不能增加记忆分类模型调用')),
        policy=MemoryPolicy(master_user_ids=('owner',), event_ttl_seconds=0,
            projection=ProjectionPolicy(enabled=True, retrieval_strategy='model')),
        clock=lambda: clock[0])
    yield service, clock
    service.projection.stop()
    service.lifecycle_review.stop()


def group_pair(memory, clock, *, group='g-one', mid='game', text=GAME, age=600, passive=False):
    clock[0] = NOW - age
    if passive:
        event = IncomingMessage.from_event(dict(post_type='message', message_type='group',
            group_id=group, user_id='owner', self_id='bot', message_id=mid, time=clock[0]-3,
            message=[dict(type='text', data={'text': text})]))
        memory.observe_message(event)
    memory.record_turn(conversation_id=group, conversation_type='group', message_id=mid, text=text,
        metadata=dict(sender_id='owner', self_id='bot', message_id=mid, shinku_involved=True))
    clock[0] += 1
    memory.record_turn(conversation_id=group, conversation_type='group', message_id=mid+':assistant',
        source_id=f'turn:{group}:{mid}:assistant', role='assistant', text=REPLY,
        metadata=dict(sender_id='bot', self_id='bot', requester_id='owner', message_id=mid,
                      shinku_involved=True, delivery_status='sent', delivery_message_ids=['receipt-'+mid],
                      delivery_content_complete=True))
    clock[0] = NOW


def projected(memory, *, query='就按那边说的来吧。', mid='current', sender='owner', conversation='owner', kind='private', bot='bot'):
    meta = dict(sender_id=sender, self_id=bot, message_id=mid, topic_continuation=True)
    context = memory.prepare_turn(conversation_id=conversation, conversation_type=kind,
        user_message=query, recent_context='当前双方的对话', message_meta=meta)
    assert context.decision.reason == 'main_model_retrieval'
    return memory.project_context(conversation_id=conversation, conversation_type=kind,
        user_message=query, message_meta=meta, memory_context=context.as_dict())


def inspect(memory, *, query='', mid='read', mode='recent', sender='owner', conversation='owner', kind='private', bot='bot', source_ids=()):
    return memory.inspect(conversation_id=conversation, conversation_type=kind, requester_id=sender,
        query=query, mode=mode, message_id=mid, self_id=bot, user_text='承接前面的事情', limit=8,
        recorded_time={'start':'', 'end':''}, source_ids=source_ids)


def test_catalog_discovers_shared_sources_outside_five_minutes_without_reading_text(memory):
    service, clock = memory
    group_pair(service, clock, passive=True)
    for i in range(24):
        put(service.store, f'noise-{i}', f'当前私聊的其他内容 {i}', age=10+i)
    view = projected(service)
    assert view['source_catalog']
    assert {row['origin_scope'] for row in view['source_catalog']} == {'group:g-one'}
    assert all(row['content_not_read_in_prompt'] for row in view['source_catalog'])
    assert GAME not in view['rendered'] and REPLY not in view['rendered']
    assert 'Oriole' not in json.dumps(view['source_catalog'])
    assert all(tuple((row['record_id'],row['scope'],row['source_version'])) in view['source_dependencies']
               for row in view['source_catalog'])
    result = inspect(service, mode='retrieve', source_ids=[r['record_id'] for r in view['source_catalog']])
    assert GAME in result['rendered'] and REPLY in result['rendered']
    assert result['retrieval_scopes'] == ['private:owner','private:master']
    assert not result.get('error') and not result['execution_authority']


@pytest.mark.parametrize('sender,kind,conversation', [('other','private','other'), ('owner','group','g-two'), ('other','group','g-two')])
def test_catalog_and_recent_do_not_grant_other_callers_shared_memory(memory, sender, kind, conversation):
    service, clock = memory
    group_pair(service, clock)
    assert not projected(service, sender=sender, kind=kind, conversation=conversation)['source_catalog']
    result = inspect(service, sender=sender, kind=kind, conversation=conversation)
    assert 'private:master' not in result['retrieval_scopes']
    assert GAME not in result['rendered']


def test_catalog_keeps_multiple_origins_and_never_picks_a_referent(memory):
    service, clock = memory
    group_pair(service, clock)
    group_pair(service, clock, group='g-two', mid='other', text='另一处讨论 Stonebridge，仍未选定。', age=500)
    view = projected(service, query='那个呢？')
    assert {r['origin_scope'] for r in view['source_catalog']} == {'group:g-one','group:g-two'}
    assert view['topic']['last_subject'] == '' and view['topic']['unresolved']
    assert view['diagnostics']['semantic']['calls'] == 0


def test_busy_dialogue_retains_a_readable_catalog_entry_within_prompt_budget(memory):
    service,clock=memory
    for index in range(4):
        group_pair(service,clock,group=f'origin-{index}',mid=f'game-{index}',age=600+index)
    for index in range(24):
        put(service.store,f'busy-{index}','当前私聊的其他事情。',age=index+1)
    view=projected(service)
    header=next(json.loads(line) for line in view['rendered'].splitlines()
                if line.startswith('{') and '"read_at"' in line)
    assert header['source_catalog'] and header['source_catalog']==view['source_catalog']
    assert len(view['rendered'])<=service.policy.projection.max_chars


def test_actual_read_anchors_survive_window_and_restart_but_do_not_cache_text(memory):
    service, clock = memory
    group_pair(service, clock, age=3*86400)
    first = inspect(service, query='Oriole Riverstep', mode='retrieve', mid='first-read')
    assert GAME in first['rendered']
    group_pair(service, clock, group='g-two', mid='arrangement', text='Juniper arrangement: only after approval; not approved.', age=3*86400-50)
    second = inspect(service, query='Juniper arrangement', mode='retrieve', mid='second-read')
    assert 'not approved' in second['rendered']
    clock[0] += 700
    service.projection.stop()
    from shinku.memory.projection.engine import ProjectionEngine
    service.projection = ProjectionEngine(service.store, service.policy.projection)
    view = projected(service, mid='later')
    assert GAME in view['rendered'] and 'not approved' in view['rendered']
    with service.store._connect() as db:
        entries=db.execute('SELECT handles_json FROM memory_projection_read_anchors').fetchall()
    assert len(entries)==2 and all('Oriole' not in row[0] and 'Juniper' not in row[0] for row in entries)
    assert GAME not in projected(service, mid='different-bot', bot='other-bot')['rendered']
    assert GAME not in projected(service, mid='different-person', sender='other', conversation='owner')['rendered']
    assert GAME not in projected(service, mid='different-conversation', conversation='desktop:other')['rendered']


@pytest.mark.parametrize('mutation', ['edit','forget','expire'])
def test_old_anchor_and_catalog_revalidate_sources(memory, mutation):
    service, clock = memory
    group_pair(service, clock, age=3*86400)
    inspect(service, query='Oriole', mode='retrieve')
    with service.store._connect() as db:
        row=db.execute("SELECT event_id FROM memory_events WHERE scope='group:g-one' AND source_id='turn:g-one:game'").fetchone()
    if mutation=='forget':
        service.store.forget(scope='group:g-one',record_id='event:'+row[0],query='',now=NOW)
    else:
        with service.store._connect() as db:
            if mutation=='edit':db.execute('UPDATE memory_events SET text=? WHERE event_id=?',('已更正原话',row[0]))
            else:db.execute('UPDATE memory_events SET expires_at=? WHERE event_id=?',(NOW-1,row[0]))
    view=projected(service, mid='after-change')
    assert GAME not in view['rendered']


@pytest.mark.parametrize('damage', ['copy_time','turn_author','turn_text','missing_turn'])
def test_passive_clock_compatibility_requires_exact_formal_turn(memory, damage):
    service, clock=memory
    group_pair(service, clock, passive=True)
    with service.store._connect() as db:
        if damage=='copy_time':db.execute("UPDATE memory_events SET created_at=created_at+10 WHERE source_id='owner:turn:g-one:game'")
        elif damage=='turn_author':db.execute("UPDATE memory_turns SET metadata_json=json_set(metadata_json,'$.sender_id','other') WHERE scope='group:g-one' AND source_id='turn:g-one:game'")
        elif damage=='turn_text':db.execute("UPDATE memory_turns SET content='伪造正式回合' WHERE scope='group:g-one' AND source_id='turn:g-one:game'")
        else:db.execute("DELETE FROM memory_turns WHERE scope='group:g-one' AND source_id='turn:g-one:game'")
        row=db.execute("SELECT event_id FROM memory_events WHERE source_id='owner:turn:g-one:game'").fetchone()
    request=ProjectionRequest('private:owner',('private:owner','private:master'),'owner','bot','','current',NOW)
    checked=ProjectionSources(service.store).read_checked((('event:'+row[0],'private:master',''),),request,Deadline(300))
    assert checked.records==(None,) and checked.reasons==('owner_origin_changed',)


@pytest.mark.parametrize('entry', ['legacy','task'])
@pytest.mark.parametrize('message', ['就玩那边说的那个？','还是照刚才那件事来吧。','That one we mentioned elsewhere.'])
def test_real_bridge_lets_main_model_read_shared_sources_without_dispatch_or_judge(bridge_factory, entry, message):
    bridge, chat, service, coordinator=bridge_factory(entry,'unused',read=False,answer='前面的原话还没有确定明晚的选择。')
    service.policy=replace(service.policy, master_user_ids=('owner',))
    clock=[NOW];service.clock=lambda:clock[0]
    group_pair(service,clock,passive=True)
    calls=[]
    def model(**options):
        calls.append(options)
        if len(calls)==1:
            assert 'source_catalog' in options['user_prompt'] and GAME not in options['user_prompt']
            header=next(json.loads(line) for line in options['user_prompt'].splitlines()
                        if line.startswith('{') and '"source_catalog"' in line)
            return {'kind':'tool','tool_call':{'name':'memory_search','arguments':{
                'mode':'retrieve','query':'','limit':8,'recorded_time':{'start':'','end':''},
                'source_ids':[row['record_id'] for row in header['source_catalog']]}}}
        assert GAME in options['user_prompt'] and REPLY in options['user_prompt']
        return {'kind':'final','text':'前面的原话还没有确定明晚的选择。'}
    chat.call_chat_json=model
    reply=bridge(turn(message,mid='current'))
    assert reply.agent_status=='completed',reply
    assert len(calls)==2 and not reply.sent and not reply.background_task_id
    assert coordinator is None or not coordinator.list()


@pytest.mark.parametrize('entry', ['legacy', 'task'])
def test_native_sdk_preserves_catalog_handle_array_and_real_read_receipt(bridge_factory, entry):
    bridge, chat, service, _ = bridge_factory(entry, 'unused', read=False)
    service.policy = replace(service.policy, master_user_ids=('owner',))
    clock=[NOW]; service.clock=lambda:clock[0]
    group_pair(service, clock, passive=True)
    def respond(payload, index):
        messages='\n'.join(str(m.get('content') or '') for m in payload['messages'])
        if index==1:
            header=next(json.loads(line) for line in messages.splitlines()
                        if line.startswith('{') and '"source_catalog"' in line)
            args=dict(mode='retrieve', query='', limit=8, recorded_time={'start':'','end':''},
                      source_ids=[r['record_id'] for r in header['source_catalog']])
            assert GAME not in messages
            offered={row['function']['name'] for row in payload['tools']}
            name='memory_search' if 'memory_search' in offered else '__shinku_tool_dispatch__'
            assert name in offered
            if name=='__shinku_tool_dispatch__':
                args=dict(request_kind='query', tool_name='memory_search', arguments=json.dumps(args))
            else:
                assert name=='memory_search'
            return dict(role='assistant', content='', tool_calls=[dict(id='catalog-read', type='function',
                function=dict(name=name, arguments=json.dumps(args)))]), 'tool_calls'
        assert index==2 and GAME in messages and REPLY in messages
        assert 'memory_lookup' in messages
        if entry=='legacy':
            assert any(m['role']=='tool' and 'memory_lookup' in m['content'] for m in payload['messages'])
        return dict(role='assistant',content=json.dumps(dict(kind='final',text='明晚还没有选定。'))), 'stop'
    with native_sdk(respond) as (runtime, payloads):
        env=runtime._env
        # 隔离域名没有内建供应商能力；通过真实配置入口声明此合成传输支持工具。
        runtime._env=lambda key,default='': ('offline.example.invalid:deepseek-flash'
            if key=='NATIVE_TOOL_PROVIDER_ALLOWLIST' else env(key,default))
        chat.call_chat_json=runtime.call_chat_json
        reply=bridge(turn('那边聊的安排呢？',mid='sdk-current'))
    assert reply.agent_status=='completed',reply
    assert len(payloads)==2 and not reply.sent and not reply.background_task_id


@pytest.mark.parametrize('sender,kind,conversation', [('other','private','other'),('other','group','g-two')])
def test_forged_shared_id_does_not_grant_scope(memory, sender, kind, conversation):
    service, clock=memory
    group_pair(service,clock)
    ids=[r['record_id'] for r in projected(service)['source_catalog']]
    result=inspect(service, mode='retrieve', source_ids=ids, sender=sender, kind=kind, conversation=conversation)
    assert result['error']=='source_handles_unavailable'
    assert not result['records'] and GAME not in result['rendered']


def test_unknown_and_revoked_handle_are_failed_reads_not_absence(memory):
    service,clock=memory
    group_pair(service,clock)
    ids=[r['record_id'] for r in projected(service)['source_catalog']]
    service.store.forget(scope='group:g-one',record_id='',query='Oriole',now=NOW)
    for source_id in ['event:unknown',*ids]:
        result=inspect(service,mode='retrieve',source_ids=[source_id])
        if GAME in result['rendered']:
            pytest.fail('撤销原文不能经关联记录复活')
        # 已送达的答复可以独立存在，但它不证明已撤销原文仍然可读。
        if source_id=='event:unknown' or not result['records']:
            assert result.get('error')=='source_handles_unavailable'


@pytest.mark.parametrize('bad', [123, ['event:x']*9, ['private:master'], [123], ['event:'+'a'*130]])
def test_source_handle_arguments_remain_bounded(memory, bad):
    tool=MemorySearchTool(memory[0])
    assert tool.normalize_call(dict(mode='retrieve',source_ids=bad,recorded_time={'start':'','end':''})) is None


def test_anchor_cache_is_bounded_and_expires_without_deleting_memory(memory):
    service,clock=memory
    group_pair(service,clock,age=3*86400)
    for index in range(6):
        inspect(service,query='Oriole',mode='retrieve',mid=f'read-{index}')
        clock[0]+=1
    with service.store._connect() as db:
        assert db.execute('SELECT count(*) FROM memory_projection_read_anchors').fetchone()[0]==4
    clock[0]+=service.policy.projection.cache_seconds+1
    assert GAME not in projected(service,mid='expired-anchor')['rendered']
    assert GAME in inspect(service,query='Oriole',mode='retrieve',mid='read-again')['rendered']


@pytest.mark.parametrize('entry', ['legacy','task'])
def test_catalog_change_during_generation_rejects_candidate(bridge_factory, entry):
    bridge,chat,service,_=bridge_factory(entry,'unused',read=False)
    service.policy=replace(service.policy,master_user_ids=('owner',))
    clock=[NOW];service.clock=lambda:clock[0]
    group_pair(service,clock)
    def respond(**options):
        assert 'source_catalog' in options['user_prompt']
        service.store.forget(scope='group:g-one',record_id='',query='Oriole',now=NOW)
        return dict(kind='final',text='未经核实的目录回答。')
    chat.call_chat_json=respond
    reply=bridge(turn('那边的事情呢？',mid='catalog-changed'))
    assert reply.agent_status=='failed' and '未经核实' not in reply.final_text


@pytest.mark.parametrize('kind', ['private','group'])
def test_model_strategy_keeps_registered_participant_attribution_and_revocation(memory, kind):
    service,clock=memory
    service.policy=replace(service.policy,owner_participant_ids=('peer',),passive_observe_private=True)
    event=IncomingMessage.from_event(dict(post_type='message',message_type=kind,
        **({'group_id':'peer-group'} if kind=='group' else {}), user_id='peer',self_id='bot',
        message_id='peer-msg',time=NOW-10,message=[dict(type='text',data={'text':'仅在天气晴朗时看望远镜，还没确定。'})]))
    service.observe_message(event)
    view=projected(service)
    assert view['source_catalog'] and all(r['author']=='peer' for r in view['source_catalog'])
    ids=[r['record_id'] for r in view['source_catalog']]
    result=inspect(service,mode='retrieve',source_ids=ids)
    assert not result.get('error') and result['records'][0]['metadata']['sender_id']=='peer'
    assert '天气晴朗' in result['rendered'] and not service.is_master_user('peer')
    service.policy=replace(service.policy,owner_participant_ids=())
    assert inspect(service,mode='retrieve',source_ids=ids)['error']=='source_handles_unavailable'
    from shinku.qq.reply_sources import checked_prompt_snapshot
    assert not checked_prompt_snapshot(service,turn('那件事',mid='check-peer'),
        tuple(tuple(h) for h in view['source_dependencies'])).valid
