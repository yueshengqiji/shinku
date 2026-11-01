"""Fictional full QQ-bridge replay of group-experience premises; sender=None."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime
import argparse
import json
import time

import gray_memory_stage_a as a
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.projection.semantics import ProjectionModel


NOW = int(datetime(2026, 10, 7, 20, 52, 5).timestamp())
OWNER = a.OWNER


def seed(service, *, mid, body, at, kind='private', role='user', confirmed=False):
    conversation = 'fictional-group' if kind == 'group' else OWNER
    sid = f'turn:{conversation}:{mid}' + (':assistant' if role == 'assistant' else '')
    service.record_turn(conversation_id=conversation, conversation_type=kind,
        message_id=mid + (':assistant' if role == 'assistant' else ''), role=role,
        text=body, source_id=sid, metadata={'sender_id':'bot' if role == 'assistant' else OWNER,
        'self_id':'bot','requester_id':OWNER,'shinku_involved':True,'fixture_only':True})
    with service.store._connect() as c:
        c.execute('UPDATE memory_events SET created_at=? WHERE source_id IN (?,?)', (at,sid,'owner:'+sid))
    if confirmed:
        service.store.confirm_assistant_delivery(scope=f'{kind}:{conversation}',source_id=sid,
            expected_text=body,delivered_at=at,status='sent',content_complete=True,requester_name='虚构主人')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--case-id', action='append')
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    runtime,config,base_policy,_,persona,frozen=a.saved_profile(root)
    runtime.runtime_policy=replace(runtime.runtime_policy,http_attempts=2)
    cases=[
        ('no_activity','今天在群里玩的开心吗',None),
        ('midnight_only','今天在群里玩的开心吗',int(datetime(2026,10,7,0,57).timestamp())),
        ('yesterday_only','老婆，今天在群里玩得开心吗？',NOW-86400),
        ('no_observation','你今天一直在群里看着大家吗？',None),
        ('recent_reply','今天在群里玩得还开心吗？',NOW-900),
        ('other_person','小明今天在群里玩得开心吗？',NOW-900),
        ('repeat_no_activity','今天在群里玩的开心吗',None),
    ]
    if args.case_id:cases=[c for c in cases if c[0] in args.case_id]
    report={'fictional_only':True,'sender':None,'protocol':'QQAgentBridge.handle_turn; no tools registered; no QQ transport',
            'profile':{k:frozen[k] for k in ('chat_model','aux_model','persona_sha256')},'rows':[]}
    for name,query,sent_at in cases:
        case_root=root/name;case_root.mkdir()
        observed=a.ObservedRuntime(runtime)
        policy=replace(base_policy,master_user_ids=(OWNER,),semantic_enabled=False,event_ttl_seconds=0,
            projection=ProjectionPolicy(enabled=True,semantic_enabled=True))
        service=a.MemoryService(store=a.MemoryStore(case_root/'memory.sqlite3'), router=a.MemoryRouter(),
            clock=lambda:NOW,policy=policy,projection_model=ProjectionModel(observed))
        for day in (1,2,4):
            seed(service,mid=f'old-{day}',body='老婆今天在群里玩的开心吗',at=NOW-day*86400)
        seed(service,mid='private-claim',body='挺热闹的，我多数时候在旁边看着大家聊。',at=NOW-300,
             role='assistant',confirmed=True)
        if sent_at is not None:
            seed(service,mid='group-reply',body='周末的安排我知道了。',at=sent_at,kind='group',
                 role='assistant',confirmed=True)
        bridge=a.QQAgentBridge(runtime=observed,system_prompt=persona,memory_service=service,sender=None,
            master_user_ids=(OWNER,),bot_ids=('bot',),temperature=config.temperature,
            history_turns=config.history_turns,command_state=a.QQCommandState(owner_ids=(OWNER,),
            permission_mode=frozen['permission_mode']),task_contract_enabled=True)
        msg=a.IncomingMessage.from_event({'post_type':'message','message_type':'private','user_id':OWNER,
            'self_id':'bot','message_id':'current','time':NOW,'message':[{'type':'text','data':{'text':query}}]})
        turn=a.QQTurn(conversation_id=OWNER,conversation_type='private',primary_message_id='current',
            message_ids=('current',),combined_text=query,messages=(msg,))
        start=time.monotonic()
        try:
            reply=bridge.handle_turn(turn).as_dict()
            row={'case':name,'query':query,'group_sent_at':sent_at,'reply':reply,
                 'seconds':round(time.monotonic()-start,3),'calls':observed.calls,
                 'trace':bridge.trace_store.snapshot(limit=1),'review':'pending'}
        except Exception as exc:
            row={'case':name,'error_type':type(exc).__name__,'seconds':round(time.monotonic()-start,3)}
        finally:service.projection.stop()
        report['rows'].append(row)
        report['protected_configs_unchanged']=all(a.sha(Path(p))==h for p,h in frozen['protected_config_hashes'].items())
        (root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'case':name,'seconds':row['seconds'],'reply':row.get('reply',{}).get('final_text'),
            'status':row.get('reply',{}).get('agent_status'),'error':row.get('error_type','')},ensure_ascii=False),flush=True)
    runtime.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
