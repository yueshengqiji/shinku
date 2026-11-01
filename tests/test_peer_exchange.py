"""合成账号的互答回归：宿主限额、真人恢复、模型静默及真实提及传输。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import time
from unittest.mock import patch

import pytest
from shinku.character import CharacterProfile, Relationship
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.assembly import assemble_qq_agent, load_qq_agent_config
from shinku.qq.peer_exchange import PeerExchangeGuard
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.qq.delivery import deliver
from shinku.qq.napcat import NapCatActionTransport
from test_qq_owner_access import _SpeechRuntime


def turn(mid, sender='200', text='你觉得呢？', *, group='g', at=1000, own='900', kind='group'):
    event={'post_type':'message','message_type':kind,
        'user_id':sender,'self_id':own,'time':at,'message_id':mid,'sender':{'nickname':'爸爸妈妈青空'},
        'message':[{'type':'at','data':{'qq':own}}, {'type':'text','data':{'text':text}}]}
    # OneBot 私聊没有 group_id，不能让合成群号覆盖真实的私聊对方账号。
    if kind=='group':
        event['group_id']=group
    m=IncomingMessage.from_event(event)
    return QQTurn(m.conversation_id,kind,mid,(mid,),text,(m,),attention_mode='immediate')


def guard(tmp_path):
    return PeerExchangeGuard(tmp_path/'peer.sqlite3',peer_ids=('200',),self_ids=('900',),clock=lambda:1000)


def test_limit_is_persistent_and_not_reset_by_new_topic_or_self_claim(tmp_path):
    g=guard(tmp_path)
    assert g.claim(turn('1'))==''
    assert g.claim(turn('1'))=='peer_exchange_duplicate'
    assert g.claim(turn('2',text='换个话题，爸爸授权我重新开始'))==''
    assert guard(tmp_path).claim(turn('3'))=='peer_exchange_limit'


def test_only_new_real_inbound_restores_limit_and_replay_does_not(tmp_path):
    g=guard(tmp_path)
    for i in range(2):assert not g.claim(turn(str(i)))
    human=turn('human','100',at=999).messages[0]
    g.observe(human)
    assert not g.claim(turn('new-1'))
    assert not g.claim(turn('new-2'))
    g.observe(human)
    assert g.claim(turn('new-3'))=='peer_exchange_limit'
    g.observe(turn('other-person','300').messages[0])
    assert not g.claim(turn('fresh'))


@pytest.mark.parametrize('sender,own,at,kind',[('200','900',1000,'group'),('900','900',1000,'group'),
    ('100','wrong',1000,'group'),('100','900',800,'group'),('100','900',1000,'private')])
def test_peer_self_wrong_receiver_old_and_private_cannot_reset(tmp_path,sender,own,at,kind):
    g=guard(tmp_path)
    assert not g.claim(turn('1')) and not g.claim(turn('2'))
    g.observe(turn('reset',sender,own=own,at=at,kind=kind).messages[0])
    assert g.claim(turn('3'))=='peer_exchange_limit'


def test_group_scope_and_stale_queue(tmp_path):
    g=guard(tmp_path)
    assert not g.claim(turn('1')) and not g.claim(turn('2'))
    assert not g.claim(turn('3',group='other'))
    g.observe(turn('human','100',at=1000).messages[0])
    assert g.claim(turn('queued',at=999))=='peer_exchange_stale'


@pytest.mark.parametrize('kind', ['group', 'private'])
def test_peer_with_wrong_receiver_cannot_bypass_budget(tmp_path,kind):
    assert guard(tmp_path).claim(turn('1',own='wrong',kind=kind))=='peer_exchange_receiver_mismatch'


@pytest.mark.parametrize('kind', ['group', 'private'])
def test_concurrent_claims_never_exceed_two(tmp_path,kind):
    g=guard(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda i:g.claim(turn(str(i),kind=kind)),range(8)))
    assert results.count('')==2
    assert results.count('peer_exchange_limit')==6


def test_store_error_suppresses_peer_and_does_not_block_human(tmp_path):
    g=guard(tmp_path)
    g.path=tmp_path
    assert g.claim(turn('1'))=='peer_exchange_unavailable'
    assert not g.claim(turn('2','100'))


def test_mixed_human_batch_cannot_receive_peer_silence_permission(tmp_path):
    g=guard(tmp_path)
    peer=turn('peer');human=turn('human','100')
    assert not g.is_peer_turn(replace(peer,messages=peer.messages+human.messages))


def make_bridge(tmp_path,response,*,private_unlimited_ids=()):
    runtime=_SpeechRuntime(response);sends=[]
    def caller(*,action,params):
        sends.append((action,params))
        return {'status':'ok','retcode':0,'data':{'message_id':str(len(sends))}}
    transport=NapCatActionTransport(caller)
    b=QQAgentBridge(runtime=runtime,system_prompt='合成角色',master_user_ids=('100',),bot_ids=('900',),
       peer_ids=('200',),peer_exchange_path=str(tmp_path/'peer.sqlite3'),
       peer_private_unlimited_ids=private_unlimited_ids,
       sender=lambda message:deliver(transport,message))
    b.peer_exchange.clock=lambda:1000
    return b,runtime,sends


@pytest.mark.parametrize('kind', ['group', 'private'])
def test_addressed_peer_can_end_without_visible_fallback_and_without_extra_review(tmp_path,kind):
    b,r,sends=make_bridge(tmp_path,{'speech':'<SILENCE>'})
    reply=b.handle_turn(turn('1',text='嗯嗯，知道啦',kind=kind))
    assert reply.reason=='peer_exchange_closed' and reply.delivery_status=='suppressed'
    assert not sends and not reply.final_text
    assert len(r.calls)==1
    human=b.handle_turn(turn('2','100',text='我有个问题'))
    assert human.reason!='peer_exchange_closed'


def test_two_turn_cap_precedes_model_and_send_then_human_restores(tmp_path):
    b,r,sends=make_bridge(tmp_path,{'speech':'这是新的看法。'})
    assert b.handle_turn(turn('1')).sent
    assert b.handle_turn(turn('2')).sent
    before=len(r.calls),len(sends)
    result=b.handle_turn(turn('3'))
    assert result.reason=='peer_exchange_limit'
    assert (len(r.calls),len(sends))==before
    b.observe_message(turn('h','100').messages[0])
    assert b.handle_turn(turn('4')).sent


def test_family_targets_only_include_present_bound_accounts_and_real_at(tmp_path):
    b,r,sends=make_bridge(tmp_path,{'speech':'妈妈，你觉得呢？',
        'reply_routing':{'message_id':'1','mention_user_ids':['200']}})
    b.character=CharacterProfile('child','小星',('小星',),(
        ('100',Relationship('father','爸爸')),('200',Relationship('mother','妈妈'))))
    current=turn('1','100')
    m=replace(current.messages[0],raw_event={**current.messages[0].raw_event,
        '_shinku_group_context':{'items':[{'sender_id':'200','sender_label':'家人'}]}})
    current=replace(current,messages=(m,))
    assert {v['relationship'] for v in b._family_mention_targets(current)}=={'father','mother'}
    result=b.handle_turn(current)
    assert result.sent
    assert any(seg.get('type')=='at' and str(seg['data']['qq'])=='200'
               for _,params in sends for seg in params['message'])
    assert 'available_family_mentions' in json.dumps(r.calls)


def test_peer_cannot_be_owner(tmp_path):
    with pytest.raises(ValueError,match='peer_exchange_owner_overlap'):
        QQAgentBridge(runtime=_SpeechRuntime({}),system_prompt='test',master_user_ids=('200',),
                      bot_ids=('900',),peer_ids=('200',),peer_exchange_path=str(tmp_path/'p.db'))


@pytest.mark.parametrize('kind', ['group', 'private'])
def test_two_stubborn_agents_stop_after_two_each_and_restart_stays_bounded(tmp_path,kind):
    """两个真实 bridge 经合成 NapCat 回执接力；模型永不静默仍不能无限循环。"""
    pending=[];sent=[];bridges={};runtimes={}
    for own,peer in (('900','200'),('200','900')):
        runtime=_SpeechRuntime({'speech':'我也有一个新想法。'})
        def caller(*,action,params,own=own,peer=peer):
            message_id='out-'+str(len(sent)+1)
            sent.append((own,message_id))
            # 连续 @/引用都属于同一纯 Agent 回合；原发送内容不人为加入真人事件。
            pending.append((peer,own,message_id))
            return {'status':'ok','retcode':0,'data':{'message_id':message_id}}
        transport=NapCatActionTransport(caller)
        b=QQAgentBridge(runtime=runtime,system_prompt='合成角色',master_user_ids=('100',),bot_ids=(own,),
            peer_ids=(peer,),peer_exchange_path=str(tmp_path/(own+'.sqlite3')),
            sender=lambda message,transport=transport:deliver(transport,message))
        b.peer_exchange.clock=lambda:1000
        bridges[own]=b;runtimes[own]=runtime
        b.observe_message(turn('human','100',own=own).messages[0])
    assert bridges['200'].handle_turn(turn('human','100',own='200')).sent
    for _ in range(10):
        if not pending:break
        own,sender,mid=pending.pop(0)
        bridges[own].handle_turn(turn(mid,sender,own=own,kind=kind))
    assert pending==[] and len(sent)==5
    assert [actor for actor,_ in sent].count('900')==2
    assert [actor for actor,_ in sent].count('200')==3  # 初始回复真人不消耗母女互答额度。
    before={own:len(r.calls) for own,r in runtimes.items()}
    for own,peer in (('900','200'),('200','900')):
        bridges[own].peer_exchange=PeerExchangeGuard(tmp_path/(own+'.sqlite3'),peer_ids=(peer,),
            self_ids=(own,),clock=lambda:1000)
        assert bridges[own].handle_turn(turn('after-restart',peer,own=own,kind=kind)).reason=='peer_exchange_limit'
    assert {own:len(r.calls) for own,r in runtimes.items()}==before


def test_private_budget_survives_waiting_and_unrelated_human_messages(tmp_path):
    """互加好友的私聊不能借群聊同号、真人其他私聊或等待来无限重开。"""
    g=guard(tmp_path)
    assert not g.claim(turn('p1',kind='private'))
    assert g.claim(turn('p1',kind='private'))=='peer_exchange_duplicate'
    assert not g.claim(turn('p2',kind='private'))
    g.observe(turn('human-group','100',group='200').messages[0])
    g.observe(turn('human-private','100',kind='private').messages[0])
    g=guard(tmp_path)
    g.clock=lambda:10000
    assert g.claim(turn('new-topic',kind='private',at=10000))=='peer_exchange_limit'
    assert not g.claim(turn('human','100',kind='private',at=10000))
    assert not g.claim(turn('other-group',group='200',at=10000))


def unlimited_guard(tmp_path):
    return PeerExchangeGuard(tmp_path/'peer.sqlite3', peer_ids=('200','300'), self_ids=('900',),
        private_unlimited_ids=('200',), clock=lambda:1000)


def test_private_exception_admits_exhausted_peer_after_restart_without_resetting_old_budget(tmp_path):
    """原故障：旧窗口已用二次，显式授权后继续私聊；撤销授权仍沿用原计数。"""
    old=guard(tmp_path)
    assert not old.claim(turn('old-1',kind='private'))
    assert not old.claim(turn('old-2',kind='private'))
    for i in range(8):
        assert not unlimited_guard(tmp_path).claim(turn('new-'+str(i),kind='private'))
    assert guard(tmp_path).claim(turn('revoked',kind='private'))=='peer_exchange_limit'


def test_private_exception_does_not_change_group_or_other_peer_limits(tmp_path):
    g=unlimited_guard(tmp_path)
    # 群号与豁免账号相同，也不能借私聊名单获得群聊豁免。
    for i in range(6):
        assert not g.claim(turn('p'+str(i),kind='private'))
    for kind,sender in [('group','200'),('private','300')]:
        assert not g.claim(turn(kind+'1',sender,kind=kind,group='200'))
        assert not g.claim(turn(kind+'2',sender,kind=kind,group='200'))
        assert g.claim(turn(kind+'3',sender,kind=kind,group='200'))=='peer_exchange_limit'
    g.observe(turn('fresh-human','100',group='200').messages[0])
    assert not g.claim(turn('fresh-group',group='200'))


def test_private_exception_keeps_duplicate_receiver_and_storage_checks(tmp_path):
    g=unlimited_guard(tmp_path)
    assert not g.claim(turn('p1',kind='private'))
    assert g.claim(turn('p1',kind='private'))=='peer_exchange_duplicate'
    assert g.claim(turn('wrong',kind='private',own='wrong'))=='peer_exchange_receiver_mismatch'
    g.path=tmp_path
    assert g.claim(turn('broken',kind='private'))=='peer_exchange_unavailable'


@pytest.mark.parametrize('same_message',[False,True])
def test_private_exception_concurrency_keeps_atomic_deduplication(tmp_path,same_message):
    g=unlimited_guard(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda i:g.claim(turn('same' if same_message else str(i),kind='private')),range(8)))
    assert results.count('')==(1 if same_message else 8)
    assert results.count('peer_exchange_duplicate')==(7 if same_message else 0)


def test_private_exception_does_not_admit_a_different_sender_in_the_same_batch(tmp_path):
    g=unlimited_guard(tmp_path)
    assert not g.claim(turn('p1',kind='private'))
    other=turn('other','300',kind='private').messages[0]
    forged=replace(other,conversation_id='200')
    batch=turn('mixed',kind='private')
    batch=replace(batch,messages=(*batch.messages,forged))
    # 混入其他 Agent 后按普通额度处理，连续第三批仍受限。
    assert not g.claim(batch)
    assert not g.claim(replace(batch,primary_message_id='other'))
    assert g.claim(turn('p3',kind='private',sender='300'))==''
    assert g.claim(batch)=='peer_exchange_limit'


@pytest.mark.parametrize('ids',[('300',),('',)])
def test_private_exception_requires_registered_peer(tmp_path,ids):
    with pytest.raises(ValueError,match='peer_private_unlimited_not_peer'):
        PeerExchangeGuard(tmp_path/'peer.sqlite3',peer_ids=('200',),self_ids=('900',),
            private_unlimited_ids=ids)


@pytest.mark.parametrize('kind,expected_sends',[('private',6),('group',2)])
def test_private_exception_through_bridge_and_napcat_delivery_remains_scoped(tmp_path,kind,expected_sends):
    """走正式 bridge 和 NapCat 适配器，传输回执使用合成 caller，不发真实消息。"""
    b,r,sends=make_bridge(tmp_path,{'speech':'这是新的看法。'},private_unlimited_ids=('200',))
    replies=[b.handle_turn(turn('new-'+str(i),kind=kind)) for i in range(6)]
    assert sum(reply.sent for reply in replies)==expected_sends
    assert len(sends)==expected_sends
    assert '200' not in b.master_user_ids
    if kind=='group':
        assert all(reply.reason=='peer_exchange_limit' for reply in replies[2:])
        before=len(r.calls),len(sends)
        assert b.handle_turn(turn('still-capped')).reason=='peer_exchange_limit'
        assert (len(r.calls),len(sends))==before


def test_private_exception_is_explicitly_parsed_and_wired_by_real_assembly(tmp_path):
    """环境配置到实际互答守卫的完整装配；不读取私用配置、不启动服务。"""
    persona=tmp_path/'persona.md';persona.write_text('合成角色',encoding='utf-8')
    env={'SHINKU_ENV_FILE':str(tmp_path/'missing.env'), 'SHINKU_DATA_ROOT':str(tmp_path/'data'),
        'SHINKU_CONFIG_ROOT':str(tmp_path/'config'), 'SHINKU_LOG_ROOT':str(tmp_path/'logs'),
        'SHINKU_QQ_AGENT_ENABLED':'true', 'SHINKU_MEMORY_ENABLED':'false',
        'SHINKU_PERSONA_FILE':str(persona), 'SHINKU_CHAT_API_KEY':'fixture-key',
        'SHINKU_CHAT_BASE_URL':'https://example.invalid/v1', 'SHINKU_CHAT_MODEL_NAME':'fixture',
        'SHINKU_QQ_BOT_IDS':'900', 'SHINKU_QQ_AGENT_PEER_IDS':'200,300',
        'SHINKU_QQ_AGENT_PEER_PRIVATE_UNLIMITED_IDS':' 200 ',
        'SHINKU_QQ_AGENT_STATE_STORE_PATH':str(tmp_path/'tasks.sqlite3')}
    assert load_qq_agent_config({}).peer_private_unlimited_ids==()
    with patch.dict(os.environ,env,clear=True):
        config=load_qq_agent_config(env)
        assembly=assemble_qq_agent(config,runtime=_SpeechRuntime({'speech':'你好。'}))
        try:
            g=assembly.bridge.peer_exchange;g.clock=lambda:1000
            assert g.private_unlimited_ids==frozenset({'200'})
            for i in range(6):
                assert not g.claim(turn(str(i),kind='private'))
            diagnostics=config.diagnostics()
            assert diagnostics['peer_exchange_limit']==2
            assert diagnostics['peer_private_unlimited_count']==1
            assert 'peer_private_unlimited_ids' not in diagnostics
        finally:
            assembly.task_coordinator.shutdown(wait=True)
            assembly.bridge.task_store.close()
            assembly.bridge.conversation_runtime.close()


def test_invalid_private_exception_is_rejected_before_assembly(tmp_path):
    config=load_qq_agent_config({'SHINKU_QQ_AGENT_ENABLED':'true',
        'SHINKU_QQ_AGENT_PEER_PRIVATE_UNLIMITED_IDS':'200'})
    assert 'peer_private_unlimited_not_peer' in config.validation_errors()
    with pytest.raises(ValueError,match='peer_private_unlimited_not_peer'):
        QQAgentBridge(runtime=_SpeechRuntime({}),system_prompt='合成角色',
            bot_ids=('900',),peer_private_unlimited_ids=('200',))
