"""排队消息只读取其到达前的来源；模拟时钟、真实记忆库和正式聊天入口。"""
from dataclasses import replace
import json
import pytest
from shinku.agent.loop import AgentState
from shinku.memory.owner_continuity import owner_continuity
from shinku.qq.reply_sources import check_context_sources
from test_context_source_protocol import mode_setup
from test_desktop_qq_continuity import NOW, qq_turn, record


def queued_context(rig):
    record(rig.memory, '旧约定只有确认才执行。', mid='first')
    rig.clock[0] = NOW + 4
    record(rig.memory, '补充七天。', mid='queued')
    rig.clock[0] = NOW + 16
    rig.memory.record_turn(conversation_id='owner', conversation_type='private',
        message_id='first:assistant', source_id='turn:owner:first:assistant',
        text='晚到回复标记，上一条已经回复。', role='assistant',
        metadata={'sender_id':'bot','requester_id':'owner','self_id':'bot',
                  'delivery_status':'sent','delivery_message_ids':['receipt'],
                  'delivery_content_complete':True,'delivery_channel':'qq'})
    turn=qq_turn('补充七天。',mid='queued')
    turn=replace(turn,messages=(replace(turn.messages[0],timestamp=NOW+4),))
    return turn


@pytest.mark.parametrize('mode',['legacy','task'])
def test_queued_reply_uses_consistent_cutoff_and_keeps_original(tmp_path,mode,request):
    rig=mode_setup(tmp_path,mode,request)
    turn=queued_context(rig)
    with rig.memory.store._connect() as conn:before=tuple(conn.iterdump())
    state=AgentState(context={'qq_turn':turn},transcript=[])
    parts=rig.bridge._build_prompt(state)
    assert '晚到回复标记' not in parts.user_prompt
    assert '旧约定只有确认才执行' in parts.user_prompt
    check=check_context_sources(rig.memory,state.context,owner_reader=rig.bridge._owner_continuity_context)
    assert check.valid,check.reason
    with rig.memory.store._connect() as conn:assert tuple(conn.iterdump())==before
    rig.runtime.call_chat_json=lambda **kw: rig.runtime.calls.append(kw) or {'kind':'final','text':'收到，补充为七天。'}
    result=rig.bridge.handle_turn(turn)
    assert result.agent_status=='completed'
    assert result.final_text=='收到，补充为七天。'
    assert len(rig.runtime.calls)==1


def test_later_arrival_stays_out_of_prompt_but_keeps_dependency_check(tmp_path,request):
    rig=mode_setup(tmp_path,'task',request);queued_context(rig)
    args=dict(conversation_id='owner',conversation_type='private',requester='owner',self_id='bot',message_id='queued')
    before=owner_continuity(rig.memory,**args)
    rig.clock[0]+=1
    record(rig.memory,'后到消息不属于该轮。',mid='later')
    after=owner_continuity(rig.memory,**args)
    assert after['handles']==before['handles']
    assert '后到消息不属于该轮' not in after['rendered']
    assert after['signature']!=before['signature']


@pytest.mark.parametrize('phase',['before','after'])
def test_source_rejection_is_not_format_retry(tmp_path,request,monkeypatch,phase):
    rig=mode_setup(tmp_path,'task',request)
    record(rig.memory,'即将撤销的合成来源。',mid='before')
    original=rig.bridge._build_prompt
    def build(*a,**kw):
        parts=original(*a,**kw)
        if phase=='before':
            rig.memory.store.forget(query='即将撤销',scope='private:master',now=rig.clock[0])
        return parts
    monkeypatch.setattr(rig.bridge,'_build_prompt',build)
    if phase=='after':
        rig.runtime.before_return=lambda: rig.memory.store.forget(query='即将撤销',scope='private:master',now=rig.clock[0])
    result=rig.bridge.handle_turn(qq_turn('接着说'))
    assert result.agent_status=='failed' and result.reason=='dialogue_sources_changed'
    assert len(rig.runtime.calls)==(0 if phase=='before' else 1)
    stages=[s for t in rig.bridge.trace_store.snapshot(limit=1) for s in t['stages']]
    assert len([s for s in stages if s['name']=='context_sources'])==(1 if phase=='before' else 2)
    assert not any(s['name'].startswith('chat_interface_') for s in stages)
